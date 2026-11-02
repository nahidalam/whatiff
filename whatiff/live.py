"""Milestone 2 live adapter probe, explicitly separate from policy evaluation."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .contracts import Action, Criterion, Proposal
from .images import decode
from .meta import MetaClient, MetaError, MuseImageGenerator, SamSegmenter, SparkAdversary, load_key
from .store import RunStore
from .validation import EditValidator, review_template


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_criteria(path):
    data = json.loads(Path(path).read_text())
    criteria = tuple(Criterion(**{**item, "acceptable_actions": tuple(Action(a) for a in item["acceptable_actions"])}) for item in data["criteria"])
    if len(criteria) != 4 or len({c.id for c in criteria}) != 4 or len({c.intervention for c in criteria}) != 4 or sum(c.test_type == "baseline" for c in criteria) != 1:
        raise ValueError("Require one baseline and three distinct intervention criteria.")
    return data["task"], criteria


def review_run(folder, reviews):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "probe.json").read_text())
    store = RunStore(folder.parent)
    original = store.read_image(manifest["id"], manifest["original"])
    validator = EditValidator(reviews)
    for item in manifest["observations"]:
        criterion = Criterion(**{**item["criterion"], "acceptable_actions": tuple(Action(a) for a in item["criterion"]["acceptable_actions"])})
        proposal = Proposal(**{**item["proposal"], "predicted_action": Action(item["proposal"]["predicted_action"])})
        edited = store.read_image(manifest["id"], item["artifact"])
        item["validity"] = asdict(validator.validate(original, edited, manifest["task"], proposal, criterion))
        item["evaluation_outcome"] = "inconclusive"
    write_json(folder / "probe.json", manifest)
    return manifest


def probe(client, store, run_id, image, task, criteria, with_sam=False, sam_concept=None):
    """Saved partial progress is retryable without repeating successful image calls."""
    folder = store.folder(run_id)
    path = folder / "probe.json"
    adversary, generator = SparkAdversary(client, tuple(c.intervention for c in criteria if c.test_type != "baseline")), MuseImageGenerator(client)
    if path.exists():
        manifest = json.loads(path.read_text())
        if (manifest["mode"] != client.mode + "_adapter_probe"
                or manifest["settings"] != {"adversary": adversary.settings, "generator": generator.settings}
                or manifest["task"] != task or store.read_image(run_id, manifest["original"]) != image
                or manifest["criteria"] != json.loads(json.dumps([asdict(c) for c in criteria]))):
            raise ValueError("Resume requires the original execution mode, settings, image, task, and criteria.")
    else:
        manifest = {"id": run_id, "mode": client.mode + "_adapter_probe", "status": "running", "task": task,
                    "criteria": [asdict(c) for c in criteria], "original": store.image(run_id, image, decode(image)[1]),
                    "settings": {"adversary": adversary.settings, "generator": generator.settings}, "observations": [],
                    "limitation": "Adapter validation only; no robot policy decisions or failure claims."}
    manifest.update(status="running", error=None)
    write_json(path, manifest)
    try:
        manifest["capabilities"] = client.capabilities(with_sam)
        write_json(path, manifest)
        if not all(manifest["capabilities"]["required"].values()):
            raise MetaError("required_model_unavailable")
        if "proposals" not in manifest:
            manifest["proposals"] = [asdict(p) for p in adversary.propose(image, task)]
            write_json(path, manifest)
        expected = {c.intervention: c for c in criteria if c.test_type != "baseline"}
        concepts = {name: sam_concept for name in expected} if with_sam and sam_concept else {}
        validator = EditValidator(segmenter=SamSegmenter(client) if with_sam else None, concepts=concepts)
        templates = json.loads((folder / "review-template.json").read_text()) if (folder / "review-template.json").exists() else {}
        for data in manifest["proposals"][len(manifest["observations"]):]:
            proposal = Proposal(**{**data, "predicted_action": Action(data["predicted_action"])})
            criterion = expected[proposal.intervention]
            generated = generator.generate(image, task, proposal)
            artifact = store.image(run_id, generated.image, generated.media_type)
            validity = validator.validate(image, generated.image, task, proposal, criterion)
            manifest["observations"].append({"proposal": data, "criterion": asdict(criterion), "artifact": artifact,
                                             "changes": generated.changes, "raw": generated.raw, "validity": asdict(validity),
                                             "evaluation_outcome": "inconclusive"})
            templates[proposal.intervention] = review_template(image, generated.image, task, proposal, criterion)
            write_json(folder / "review-template.json", templates)
            write_json(path, manifest)
        manifest["status"] = "completed"
    except Exception as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "code": getattr(error, "code", "adapter_error"),
                                               "http_status": getattr(error, "status", None), "retryable": getattr(error, "retryable", True)})
    write_json(path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", default="data/meta-probes")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--criteria", default="examples/tabletop-criteria.json")
    parser.add_argument("--sam", action="store_true", help="Optional grounding evidence only")
    parser.add_argument("--sam-concept", help="Object noun phrase for optional grounding, e.g. white mug")
    parser.add_argument("--resume", help="Existing adapter probe UUID")
    parser.add_argument("--review-run", help="Saved probe directory; no API calls")
    parser.add_argument("--reviews", help="Curator-completed review-template.json")
    args = parser.parse_args()
    if args.review_run:
        if not args.reviews:
            parser.error("--review-run requires --reviews")
        result = review_run(args.review_run, json.loads(Path(args.reviews).read_text()))
        print(json.dumps({"status": result["status"], "validity": [item["validity"]["status"] for item in result["observations"]]}))
        return
    try:
        key = load_key(args.env_file)
        store = RunStore(Path(args.output).resolve())
        run_id = args.resume or uuid4().hex
        folder = store.folder(run_id)
        client = MetaClient(key, folder / "api")
        if args.preflight:
            print(json.dumps(client.capabilities(args.sam), indent=2))
            return
        if not args.image:
            parser.error("--image is required for live editing")
        image = Path(args.image).read_bytes()
        decode(image)
        task, criteria = load_criteria(args.criteria)
        if args.sam and not args.sam_concept:
            parser.error("--sam requires --sam-concept for image probes")
        result = probe(client, store, run_id, image, task, criteria, args.sam, args.sam_concept)
        print(json.dumps({"mode": result["mode"], "status": result["status"], "generated_images": len(result["observations"]),
                          "folder": str(folder), "error": result["error"]}, indent=2))
        if result["status"] != "completed":
            raise SystemExit(1)
    except MetaError as error:
        print(json.dumps({"status": "blocked", "code": error.code, "http_status": error.status}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
