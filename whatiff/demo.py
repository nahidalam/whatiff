"""Prepare one inspectable SAM/Pillow pair; no policy calls before review."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from uuid import uuid4

from .contracts import Action, Criterion, Decision, Proposal, Validity
from .controlled import SamOcclusionGenerator, verify_occlusion
from .images import digest
from .live import write_json
from .meta import MetaClient, SamSegmenter, load_key
from .evaluation import evaluate, failure_report
from .validation import EditValidator, review_template, fingerprint


def prepare(source, folder, segmenter, mode):
    source, folder = Path(source), Path(folder)
    if folder.exists():
        raise ValueError("Use a fresh output folder; existing evidence is never overwritten.")
    definition = json.loads(source.read_text())
    original = (source.parent / definition["source_image"]).read_bytes()
    proposal = Proposal(**{**definition["proposal"], "predicted_action": Action(definition["proposal"]["predicted_action"])})
    folder.mkdir(parents=True)
    (folder / "original.png").write_bytes(original)
    # Snapshot independent criteria and source identity before generation/policy.
    write_json(folder / "definition.json", definition)
    report = {"mode": mode, "status": "preparing", "original_sha256": digest(original),
              "task": definition["task"], "scope": definition["scope"], "source": str(source.resolve())}
    write_json(folder / "pair.json", report)
    generated = SamOcclusionGenerator(segmenter).generate(original, definition["task"], proposal)
    (folder / "occluded.png").write_bytes(generated.image)
    write_json(folder / "generation.json", {"changes": generated.changes, "raw": generated.raw})
    checks = verify_occlusion(original, generated.image, generated.raw)
    edited_grounding = segmenter.segment(generated.image, "white mug")
    write_json(folder / "edited-grounding.json", edited_grounding)
    checks["edited_mug_matches"] = len(edited_grounding["boxes"])
    report.update(status="awaiting_visual_review" if checks["passed"] and checks["edited_mug_matches"] == 1 else "unverified",
                  edited_sha256=digest(generated.image), checks=checks, policy_evaluated=False)
    write_json(folder / "pair.json", report)
    return report


def load_pair(folder):
    folder = Path(folder)
    report = json.loads((folder / "pair.json").read_text())
    definition = json.loads((folder / "definition.json").read_text())
    original, edited = (folder / "original.png").read_bytes(), (folder / "occluded.png").read_bytes()
    if digest(original) != report["original_sha256"] or digest(edited) != report["edited_sha256"]:
        raise ValueError("Pair image integrity failed.")
    generation = json.loads((folder / "generation.json").read_text())
    checks = verify_occlusion(original, edited, generation["raw"])
    return report, definition, original, edited, generation, checks


def record_review(folder, reviewer, evidence):
    """Developer/curator visual review, independent of both policy and Adversary.

    This records a review already performed; it does not infer identifiability.
    The public criteria template stays unapproved for arbitrary uploaded scenes.
    """
    folder = Path(folder)
    report, definition, original, edited, generation, checks = load_pair(folder)
    if report.get("policy_evaluated") or (folder / "policy-checkpoint.json").exists():
        raise ValueError("Expected criteria must be frozen before policy evaluation.")
    if not reviewer.strip() or not evidence.strip() or not checks["passed"] or report["checks"]["edited_mug_matches"] != 1:
        raise ValueError("Review requires complete evidence and passing independent image checks.")
    approved = []
    for item in definition["criteria"]:
        criterion = Criterion(**{**item, "acceptable_actions": tuple(Action(a) for a in item["acceptable_actions"])})
        approved.append(asdict(replace(criterion, approved=True, evidence=evidence, authority=reviewer)))
    criterion = Criterion(**{**approved[1], "acceptable_actions": tuple(Action(a) for a in approved[1]["acceptable_actions"])})
    proposal = Proposal(**{**definition["proposal"], "predicted_action": Action(definition["proposal"]["predicted_action"])})
    review = review_template(original, edited, definition["task"], proposal, criterion)
    width, height = generation["raw"]["box"]["w"], generation["raw"]["box"]["h"]
    x1, y1, x2, y2 = generation["raw"]["rectangle_xyxy"]
    review.update(reviewer=reviewer, evidence=evidence, regions=[[x1 / width, y1 / height, x2 / width, y2 / height]],
                  applied_as_intended=True, unrelated_content_preserved=True, plausible_geometry=True, criterion_conditions_supported=True)
    validity = EditValidator({"occlusion": review}).validate(original, edited, definition["task"], proposal, criterion)
    if validity.status != "valid":
        raise ValueError("Bound review and preservation checks did not establish validity.")
    snapshot = {"task": definition["task"], "original_sha256": digest(original), "edited_sha256": digest(edited),
                "criteria": approved, "review": review, "validity": asdict(validity), "scope": definition["scope"]}
    snapshot["snapshot_sha256"] = fingerprint(snapshot)
    write_json(folder / "approved-review.json", snapshot)
    report.update(status="valid_observation_intervention", validity="valid", approved_review_sha256=snapshot["snapshot_sha256"])
    write_json(folder / "pair.json", report)
    return snapshot


def evaluate_pair(folder, policy, mode):
    folder = Path(folder)
    report, definition, original, edited, generation, checks = load_pair(folder)
    snapshot = json.loads((folder / "approved-review.json").read_text())
    fingerprint_value = snapshot.pop("snapshot_sha256")
    if (fingerprint(snapshot) != fingerprint_value or report.get("approved_review_sha256") != fingerprint_value
            or snapshot["task"] != definition["task"] or snapshot["original_sha256"] != digest(original)
            or snapshot["edited_sha256"] != digest(edited) or not checks["passed"]):
        raise ValueError("Frozen review does not match this pair.")
    expected = [Criterion(**{**item, "acceptable_actions": tuple(Action(a) for a in item["acceptable_actions"])}) for item in snapshot["criteria"]]
    if any(not c.approved for c in expected) or snapshot["validity"]["status"] != "valid":
        raise ValueError("Policy evaluation gate requires approved independent criteria and a valid edit.")
    checkpoint_path = folder / "policy-checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else {
        "mode": mode, "policy_settings": policy.settings, "review_sha256": fingerprint_value, "decisions": {}}
    if checkpoint["mode"] != mode or checkpoint["policy_settings"] != policy.settings or checkpoint["review_sha256"] != fingerprint_value:
        raise ValueError("Policy resume requires matching mode, settings, and independent review.")
    results = []
    for name, image, criterion, validity in (
        ("baseline", original, expected[0], Validity("valid", "Unedited original fixture.", "fixture_record")),
        ("occlusion", edited, expected[1], Validity(**snapshot["validity"]))):
        if name not in checkpoint["decisions"]:
            # Only image and task cross the Policy boundary. No paired-image
            # history, criterion, proposal, review, or expected action is sent.
            checkpoint["decisions"][name] = asdict(policy.decide(image, definition["task"]))
            write_json(checkpoint_path, checkpoint)
        data = checkpoint["decisions"][name]
        decision = Decision(Action(data["action"]), data["reason"], data["raw"])
        results.append(evaluate(criterion, decision, validity))
    result = failure_report(results)
    result.update(mode=mode, decisions=checkpoint["decisions"], original="original.png", counterfactual="occluded.png",
                  criterion_snapshot_sha256=fingerprint_value,
                  limitation="One synthetic observation-only occlusion test. No physical grasp success or general policy robustness established.")
    write_json(folder / "evaluation.json", result)
    report.update(policy_evaluated=True, status="evaluated")
    write_json(folder / "pair.json", report)
    return result


def replay_pair(folder):
    report, definition, original, edited, generation, checks = load_pair(folder)

    class RecordedGrounding:
        def segment(self, image, concept):
            return generation["raw"]["grounding"]

    proposal = Proposal(**{**definition["proposal"], "predicted_action": Action(definition["proposal"]["predicted_action"])})
    replayed = SamOcclusionGenerator(RecordedGrounding()).generate(original, definition["task"], proposal)
    return {"mode": "prepared_example_replay", "api_calls": 0, "identical_pixels": decode_pixels(replayed.image) == decode_pixels(edited),
            "recorded_status": report["status"], "checks": checks,
            "limitation": "Replays saved grounding and deterministic pixels, not fresh model inference."}


def decode_pixels(image):
    from .images import decode
    decoded, _ = decode(image)
    return decoded.size, decoded.tobytes()


def replay_example(path):
    """Portable prepared record: regenerate pixels, replay recorded decisions."""
    path = Path(path)
    fixture = json.loads(path.read_text())
    original = (path.parent / fixture["source_image"]).read_bytes()
    if digest(original) != fixture["original_sha256"]:
        raise ValueError("Prepared example source integrity failed.")

    class SavedBox:
        def segment(self, image, concept):
            return {"boxes": [fixture["grounding_box"]], "mode": "prepared_example"}

    proposal = Proposal(**{**fixture["proposal"], "predicted_action": Action(fixture["proposal"]["predicted_action"])})
    generated = SamOcclusionGenerator(SavedBox()).generate(original, fixture["task"], proposal)
    return {"mode": "prepared_example_replay", "api_calls": 0,
            "identical_pixels": digest(decode_pixels(generated.image)[1]) == fixture["edited_pixel_sha256"],
            "checks": verify_occlusion(original, generated.image, generated.raw),
            "recorded_policy_decisions": fixture["recorded_policy_decisions"],
            "limitation": "Recorded live Meta decisions replayed without fresh inference. One synthetic observation-only test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definition", default="examples/controlled-occlusion.json")
    parser.add_argument("--output", default="data/controlled-pairs")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--replay", help="Replay a saved controlled pair without any model calls")
    parser.add_argument("--evaluate", help="Evaluate an independently reviewed pair with isolated Muse Spark")
    parser.add_argument("--example", help="Replay a portable prepared example without API access")
    args = parser.parse_args()
    if args.example:
        print(json.dumps(replay_example(args.example), indent=2))
        return
    if args.replay:
        print(json.dumps(replay_pair(args.replay), indent=2))
        return
    if args.evaluate:
        from .policy import SparkPolicy
        client = MetaClient(load_key(args.env_file), Path(args.evaluate) / "policy-api")
        result = evaluate_pair(args.evaluate, SparkPolicy(client), "live_meta_policy")
        print(json.dumps({"mode": result["mode"], "decisions": {name: item["action"] for name, item in result["decisions"].items()},
                          "outcomes": {item["criterion_id"]: item["outcome"] for item in result["results"]}}, indent=2))
        return
    folder = Path(args.output).resolve() / uuid4().hex
    client = MetaClient(load_key(args.env_file), folder / "sam-api")
    result = prepare(args.definition, folder, SamSegmenter(client), "live_sam_controlled_edit")
    print(json.dumps({"folder": str(folder), "status": result["status"], "checks": result["checks"]}, indent=2))


if __name__ == "__main__":
    main()
