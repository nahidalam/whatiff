"""Local UI adapter over existing generators, workflow, policy and scoring."""

import json
from copy import deepcopy
from datetime import datetime, timezone
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from .baseten import BasetenClient, BasetenError, BasetenSparkPolicy, load_baseten_key
from .contracts import Action, Criterion, Decision, Proposal, Validity
from .controlled import SamOcclusionGenerator, verify_occlusion
from .demo import decode_pixels, replay_example
from .evaluation import evaluate, failure_report
from .execution import projection, read_json
from .images import decode, digest
from .live import write_json
from .meta import MetaClient, MuseImageGenerator, SamSegmenter, SparkAdversary, load_key
from .policy import SparkPolicy
from .store import RunStore
from .validation import EditValidator, fingerprint
from .workflow import Workflow

LIMITATION = "Image-level action selection only. No robot execution, simulation, or physical grasp success."


def criterion(data):
    return Criterion(**{**data, "acceptable_actions": tuple(Action(a) for a in data["acceptable_actions"])})


def decision(data):
    return Decision(Action(data["action"]), data["reason"], data["raw"])


def task_criteria(task):
    """Task-bound placeholders, never ground truth or agent predictions."""
    return tuple(Criterion(name, name, "baseline" if name == "baseline" else "invariance",
                           tuple(Action), f'For task "{task}", independently determine acceptable actions for the {name} observation.',
                           "", "independent_human_review_required", False)
                 for name in ("baseline", "occlusion", "distractor", "relocation"))


class InterfaceService:
    def __init__(self, project, data=None):
        self.project = Path(project)
        self.root = Path(data or self.project / "data/web")
        self.root.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.lock, self.busy = threading.Lock(), False
        for path in self.root.glob("*/state.json"):
            saved = json.loads(path.read_text())
            if saved["status"] == "running":
                saved.update(status="failed", phase="Run interrupted. Retry to resume saved steps.", error="interrupted")
                write_json(path, saved)

    def folder(self, run_id):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise ValueError("Invalid run ID.")
        return self.root / run_id

    def get(self, run_id):
        state = json.loads((self.folder(run_id) / "state.json").read_text())
        state["artifact_directory"] = str(self.folder(run_id).resolve())
        if state["status"] == "completed" and state.get("criteria_task") != state["task"]:
            self.repair_legacy_criteria(state)
        recorded = read_json(self.project / "examples/washing-dishes-run/execution.json") if state["mode"] == "prepared_example" else None
        state["execution"] = projection(state, self.folder(run_id), recorded)
        return state

    def repair_legacy_criteria(self, state):
        """Preserve the old audit report before repairing task-mismatched templates."""
        folder = self.folder(state["id"])
        report = state["report"]
        archive = folder / "report-before-task-criteria.json"
        if not archive.exists():
            write_json(archive, report)
        known = False
        if state["backend"] == "controlled":
            fixture = self.reviewed_fixture()
            known = state["task"] == fixture["task"] and digest(self.asset(state["id"], "original.png").read_bytes()) == fixture["original_sha256"]
        rules = {rule.id: rule for rule in task_criteria(state["task"])}
        def repair(result):
            result = deepcopy(result)
            if not known and not (result["criterion_approved"] and result["criterion_authority"].startswith("human:")):
                rule = rules[result["criterion_id"]]
                result.update(requirement=rule.requirement, acceptable_actions=[a.value for a in rule.acceptable_actions],
                              criterion_evidence="", criterion_authority=rule.authority, criterion_approved=False,
                              test_type=rule.test_type, outcome="inconclusive", reason="Expected-action criteria lack independent approval.")
            return result
        dimensions = {}
        before, _ = decode(self.asset(state["id"], "original.png").read_bytes())
        for index, case in enumerate(state["cases"]):
            after, _ = decode(self.asset(state["id"], f"counterfactual-{index}.png").read_bytes())
            dimensions[case["name"].lower()] = {"original": list(before.size), "generated": list(after.size), "normalization": "none"}
            case["image_dimensions"] = dimensions[case["name"].lower()]
        repair_rule = repair
        def repair(result):
            result = repair_rule(result)
            size = dimensions.get(result["criterion_id"])
            if size:
                result["validity"]["raw"]["dimensions"] = size
                if size["original"] != size["generated"]:
                    result["validity"].update(status="invalid", evidence="Image dimensions changed.", source="image_checks")
                    result.update(outcome="inconclusive", reason="Intervention validity is not established.")
            return result
        results = [repair(result) for result in report["results"]]
        if "review" in report:
            report["review"]["original_results"] = [repair(result) for result in report["review"]["original_results"]]
        indexed = {result["criterion_id"]: result for result in results}
        for index, observation in enumerate([state["original"], *state["cases"]]):
            result = indexed["baseline" if index == 0 else observation["name"].lower()]
            observation.update(expected_actions=result["acceptable_actions"], criterion_approved=result["criterion_approved"],
                               outcome=result["outcome"], explanation=result["reason"])
            if index:
                observation["validity"] = result["validity"]
        report.update(failure_report(results), original=state["original"], counterfactuals=state["cases"])
        state["criteria_task"] = state["task"]
        write_json(folder / "report.json", report)
        write_json(folder / "state.json", state)

    def compact_export(self, run_id):
        state = self.get(run_id)
        if state["status"] != "completed":
            raise ValueError("The evaluation report is not ready.")
        scenarios = []
        for index, case in enumerate(state["cases"]):
            validity = case["validity"]
            edit_approval = case.get("review", {}).get("edit_status", "approved" if validity["status"] == "valid" else "pending")
            sizes = case["image_dimensions"]
            same_size = sizes["original"] == sizes["generated"]
            verified = same_size and validity["status"] == "valid" and case["criterion_approved"] and state["original"]["criterion_approved"] and (not case.get("checks") or case["checks"]["passed"])
            verification = "verified" if verified else "invalid" if validity["status"] == "invalid" or not same_size else "unverified"
            if not same_size:
                reason = "Image dimensions changed."
            elif validity["status"] == "invalid":
                reason = "Edit rejected by reviewer." if edit_approval == "rejected" else validity["evidence"][:160]
            elif not case["criterion_approved"] or not state["original"]["criterion_approved"]:
                reason = "Expected actions await independent approval."
            elif not verified:
                reason = "Image edit awaits verification."
            else:
                reason = "Observed action matches the approved criterion." if case["outcome"] == "pass" else "Observed action violates the approved criterion."
            scenarios.append({"intervention": case["name"].lower(), "image": f"counterfactual-{index}.png",
                              "expected_actions": case["expected_actions"] if case["criterion_approved"] else None,
                              "observed_action": case["observed_action"], "edit_approval": edit_approval,
                              "criterion_approval": "approved" if case["criterion_approved"] else "pending",
                              "verification_status": verification, "evaluation_outcome": case["outcome"] if verified else "inconclusive",
                              "reason": reason, "evidence_id": f"whatiff:{run_id}:{case['id']}"})
        compact = {"schema_version": 1, "evidence_id": f"whatiff:{run_id}", "mode": state["mode"], "scope": "image_action_selection",
                   "task": state["task"], "policy_model": state["report"]["policy_model"], "policy_provider": state["report"].get("policy_provider", "meta"),
                   "original": {"image": "original.png", "observed_action": state["original"]["observed_action"]}, "scenarios": scenarios}
        write_json(self.folder(run_id) / "evaluation.json", compact)
        return compact

    def update(self, run_id, **values):
        state = self.get(run_id)
        state.update(values)
        write_json(self.folder(run_id) / "state.json", state)
        return self.get(run_id)

    def base(self, mode, task, backend):
        run_id = uuid4().hex
        self.folder(run_id).mkdir()
        state = {"id": run_id, "mode": mode, "status": "running", "task": task, "backend": backend,
                 "phase": "Starting", "original": None, "cases": [], "contributions": [], "report": None,
                 "error": None, "limitation": LIMITATION}
        write_json(self.folder(run_id) / "state.json", state)
        return state

    def reviewed_fixture(self):
        review = json.loads((self.project / "examples/controlled-review.json").read_text())
        frozen = {key: value for key, value in review.items() if key != "snapshot_sha256"}
        if fingerprint(frozen) != review.get("snapshot_sha256"):
            raise ValueError("Independent fixture review integrity failed.")
        return review

    def asset(self, run_id, name):
        if not re.fullmatch(r"original\.png|counterfactual-[0-2]\.png", name):
            raise ValueError("Unknown image artifact.")
        return self.folder(run_id) / name

    def observation(self, run_id, name, result, policy):
        return {"image_url": f"/api/runs/{run_id}/images/{name}", "observed_action": policy["action"],
                "policy_reason": policy["reason"], "expected_actions": result["acceptable_actions"],
                "outcome": result["outcome"], "explanation": result["reason"],
                "criterion_approved": result.get("criterion_approved", False),
                "policy_model": policy.get("raw", {}).get("settings", {}).get("model", "muse-spark-1.3"),
                "policy_provider": policy.get("raw", {}).get("settings", {}).get("provider", "meta")}

    def complete(self, state, original_policy, generated, results, contributions, report):
        # Input results stay in observation order; failure_report ranks separately.
        run_id = state["id"]
        original = self.observation(run_id, "original.png", results[0], original_policy)
        cases = []
        for index, item in enumerate(generated):
            result = results[index + 1]
            cases.append({"id": str(index), "name": item["proposal"]["intervention"].capitalize(),
                          **self.observation(run_id, f"counterfactual-{index}.png", result, item["policy"]),
                          "intervention": item["proposal"]["description"], "hypothesis": item["proposal"]["rationale"],
                          "validity": result["validity"], "checks": item.get("checks")})
        report.update(mode=state["mode"], task=state["task"], limitation=LIMITATION,
                      policy_model=original_policy.get("raw", {}).get("settings", {}).get("model", "muse-spark-1.3"),
                      policy_provider=original_policy.get("raw", {}).get("settings", {}).get("provider", "meta"), contributions=contributions,
                      original=original, counterfactuals=cases)
        write_json(self.folder(run_id) / "report.json", report)
        self.update(run_id, status="completed", phase="Evaluation complete", original=original,
                    cases=cases, contributions=contributions, report=report, error=None)
        self.compact_export(run_id)
        return self.get(run_id)

    def replay(self):
        """Replay the saved washing-dishes Meta run without credentials or calls."""
        folder = self.project / "examples/washing-dishes-run"
        saved = json.loads((folder / "state.json").read_text())
        manifest = json.loads((folder / "images.json").read_text())
        if saved["status"] != "completed":
            raise ValueError("Saved washing-dishes run is incomplete.")
        images = {}
        for name, expected_digest in manifest.items():
            content = (folder / name).read_bytes()
            if digest(content) != expected_digest:
                raise ValueError("Saved washing-dishes image integrity failed.")
            images[name] = content
        state = self.base("prepared_example", saved["task"], saved["backend"])
        run_id = state["id"]
        for name, content in images.items():
            self.asset(run_id, name).write_bytes(content)
        original, cases = saved["original"], saved["cases"]
        original["image_url"] = f"/api/runs/{run_id}/images/original.png"
        for index, item in enumerate(cases):
            item["image_url"] = f"/api/runs/{run_id}/images/counterfactual-{index}.png"
        contributions = saved["contributions"]
        for item in contributions:
            item["source"] = "Recorded · " + item["source"]
        report = saved["report"]
        report.update(mode="prepared_example", source_run_id=saved["id"], original=original,
                      counterfactuals=cases, contributions=contributions)
        write_json(self.folder(run_id) / "report.json", report)
        self.update(run_id, status="completed", phase="Saved washing-dishes evaluation",
                    original=original, cases=cases, contributions=contributions, report=report)
        self.compact_export(run_id)
        return self.get(run_id)

    def review(self, run_id, case_id, data):
        """Human edit review and independent criteria; never call a model."""
        with self.lock:
            state = self.get(run_id)
            if state["status"] != "completed":
                raise ValueError("Only completed results can be reviewed.")
            case = next((item for item in state["cases"] if item["id"] == case_id), None)
            if case is None:
                raise ValueError("Unknown counterfactual.")
            reviewer = str(data.get("reviewer", "")).strip()
            if not reviewer or len(reviewer) > 200:
                raise ValueError("Enter a reviewer name.")
            if not set(data) <= {"reviewer", "edit_status", "edit_note", "acceptable_actions", "baseline_actions", "criterion_evidence"}:
                raise ValueError("Unknown review field.")
            report = state["report"]
            review = report.setdefault("review", {"original_results": deepcopy(report["results"]), "cases": {}, "history": []})
            entry = review["cases"].setdefault(case_id, {"original_validation": deepcopy(case["validity"])})
            event = {"case_id": case_id, "reviewer": reviewer, "timestamp": datetime.now(timezone.utc).isoformat()}
            if "edit_status" in data:
                status = data["edit_status"]
                if status not in ("approved", "rejected"):
                    raise ValueError("Choose Approve edit or Reject edit.")
                note = str(data.get("edit_note", "")).strip()
                if len(note) > 2000:
                    raise ValueError("Review note is too long.")
                entry.update(edit_status=status, edit_note=note, edit_reviewer=reviewer)
                event.update(edit_status=status, edit_note=note)
            if "acceptable_actions" in data:
                evidence = str(data.get("criterion_evidence", "")).strip()
                if not evidence or len(evidence) > 2000:
                    raise ValueError("Explain the expected actions independently of the model decision.")
                def actions(key):
                    values = data.get(key)
                    if not isinstance(values, list) or not values:
                        raise ValueError("Select acceptable actions for both images.")
                    return list(dict.fromkeys(Action(value).value for value in values))
                entry.update(acceptable_actions=actions("acceptable_actions"), baseline_actions=actions("baseline_actions"),
                             criterion_evidence=evidence, criterion_reviewer=reviewer)
                review["baseline"] = {key: entry[key] for key in ("baseline_actions", "criterion_evidence", "criterion_reviewer")}
                event.update(acceptable_actions=entry["acceptable_actions"], baseline_actions=entry["baseline_actions"], criterion_evidence=evidence)
            if len(event) == 3:
                raise ValueError("No review action supplied.")
            review["history"].append(event)
            originals = {result["criterion_id"]: result for result in review["original_results"]}
            baseline = review.get("baseline")
            results = []
            observations = [state["original"], *state["cases"]]
            for index, observation in enumerate(observations):
                original = originals["baseline" if index == 0 else observation["name"].lower()]
                item_review = review["cases"].get(observation.get("id"), {}) if index else {}
                expected = baseline if index == 0 else item_review
                approved = bool(expected and ("baseline_actions" if index == 0 else "acceptable_actions") in expected)
                rule = Criterion(original["criterion_id"], "baseline" if index == 0 else observation["name"].lower(),
                                 original["test_type"], tuple(Action(value) for value in (expected["baseline_actions" if index == 0 else "acceptable_actions"] if approved else original["acceptable_actions"])),
                                 ("Human-specified acceptable action for task: " + state["task"]) if approved else original["requirement"], expected["criterion_evidence"] if approved else original["criterion_evidence"],
                                 "human:" + expected["criterion_reviewer"] if approved else original["criterion_authority"], approved or original["criterion_approved"])
                validity = Validity(**original["validity"])
                if index and item_review.get("edit_status") and not (item_review["edit_status"] == "approved" and original["validity"]["status"] == "invalid"):
                    validity = Validity("valid" if item_review["edit_status"] == "approved" else "invalid",
                                        item_review.get("edit_note") or "Human edit " + item_review["edit_status"],
                                        "human:" + item_review["edit_reviewer"], {"original_validation": original["validity"]})
                result = evaluate(rule, Decision(Action(observation["observed_action"]), observation["policy_reason"], {}), validity)
                observation.update(outcome=result["outcome"], explanation=result["reason"], expected_actions=result["acceptable_actions"], criterion_approved=result["criterion_approved"])
                if index:
                    observation.update(validity=result["validity"], review=item_review)
                results.append(result)
            report.update(failure_report(results), original=state["original"], counterfactuals=state["cases"], review=review)
            write_json(self.folder(run_id) / "report.json", report)
            self.update(run_id, original=state["original"], cases=state["cases"], report=report)
            self.compact_export(run_id)
            return self.get(run_id)

    @staticmethod
    def contributions(proposal, changes, adversary_source, prepared=False):
        return [{"role": "Adversary", "source": adversary_source, "summary": proposal.rationale,
                 "details": proposal.description},
                {"role": "Generator", "source": "Saved SAM + Pillow replay" if prepared else "Live SAM 3.1 + Pillow",
                 "summary": changes, "details": "One bounded sensor occlusion. No changes outside the edit region are permitted."},
                {"role": "Evaluator", "source": "Independent criteria + deterministic scoring",
                 "summary": "Compared isolated policy actions with the separately defined acceptable actions.",
                 "details": "Policy explanations and Adversary predictions do not establish correctness. Unverified edits remain Inconclusive."}]

    def start(self, task, backend, image=None, policy_provider="baseten"):
        if backend not in ("controlled", "muse") or not task.strip() or len(task) > 2000:
            raise ValueError("Choose a backend and enter a task under 2,000 characters.")
        if image is None:
            image = (self.project / "examples/tabletop-original.png").read_bytes()
        decode(image)
        # Check credentials before creating a job; never send them to the UI.
        if policy_provider not in ("baseten", "meta"):
            raise ValueError("Choose Baseten or the explicit hosted Meta fallback.")
        load_key(self.project / ".env")
        if policy_provider == "baseten":
            load_baseten_key(self.project / ".env")
        with self.lock:
            if self.busy:
                raise RuntimeError("A live run is already in progress.")
            self.busy = True
        state = self.base("live", task.strip(), backend)
        state = self.update(state["id"], policy_provider=policy_provider)
        self.asset(state["id"], "original.png").write_bytes(image)
        self.executor.submit(self._work, state["id"])
        return state

    def retry(self, run_id):
        state = self.get(run_id)
        if state["status"] != "failed" or state["mode"] != "live":
            raise ValueError("Only failed live runs can be retried.")
        with self.lock:
            if self.busy:
                raise RuntimeError("A live run is already in progress.")
            self.busy = True
        self.update(run_id, status="running", phase="Resuming saved steps", error=None)
        self.executor.submit(self._work, run_id)
        return self.get(run_id)

    def _work(self, run_id):
        try:
            state = self.get(run_id)
            if state["backend"] == "controlled":
                self._controlled(state)
            else:
                self._muse(state)
        except Exception as error:
            code = getattr(error, "code", None) or type(error).__name__
            self.update(run_id, status="failed", phase="Run paused. Completed steps are saved.", error=str(code),
                        error_provider="baseten" if isinstance(error, BasetenError) else "meta_or_workflow")
        finally:
            with self.lock:
                self.busy = False

    def step(self, run_id, name, operation):
        path = self.folder(run_id) / (name + ".json")
        if path.exists():
            return json.loads(path.read_text())
        value = operation()
        write_json(path, value)
        return value

    def policy_for(self, state):
        folder = self.folder(state["id"])
        if state.get("policy_provider", "meta") == "baseten":
            return BasetenSparkPolicy(BasetenClient(load_baseten_key(self.project / ".env"), folder / "policy-api"))
        return SparkPolicy(MetaClient(load_key(self.project / ".env"), folder / "policy-api"))

    def _controlled(self, state):
        run_id, task = state["id"], state["task"]
        folder = self.folder(run_id)
        original = self.asset(run_id, "original.png").read_bytes()
        key = load_key(self.project / ".env")
        client = MetaClient(key, folder / "agent-api")
        policy = self.policy_for(state)
        fixture = json.loads((self.project / "examples/controlled-occlusion.json").read_text())
        reviewed = self.step(run_id, "independent-review-snapshot", self.reviewed_fixture)
        # Freeze the separate fixture rule before any new policy calls. Approval
        # transfers ONLY when image/task and later edited bytes match exactly.
        known_source = digest(original) == reviewed["original_sha256"] and task == reviewed["task"]
        expected = [criterion(c) for c in reviewed["criteria"]] if known_source else list(task_criteria(task))[:2]
        snapshot = self.step(run_id, "criteria-snapshot", lambda: {"criteria": [asdict(c) for c in expected], "task": task, "original_sha256": digest(original)})
        expected = [criterion(c) for c in snapshot["criteria"]]
        self.update(run_id, phase="Policy · reading the original observation")
        baseline = self.step(run_id, "baseline-policy", lambda: asdict(policy.decide(original, task)))
        self.update(run_id, phase="Adversary · proposing counterfactual changes")
        proposals = self.step(run_id, "adversary", lambda: [asdict(p) for p in SparkAdversary(client).propose(original, task)])
        proposed = next(p for p in proposals if p["intervention"] == "occlusion")
        selected = Proposal(**{**proposed, "predicted_action": Action(proposed["predicted_action"])})
        actual = Proposal(**{**fixture["proposal"], "predicted_action": Action(fixture["proposal"]["predicted_action"])})
        self.update(run_id, phase="Generator · grounding the mug and applying one controlled edit")

        def generate():
            generated = SamOcclusionGenerator(SamSegmenter(client)).generate(original, task, actual)
            self.asset(run_id, "counterfactual-0.png").write_bytes(generated.image)
            return {"changes": generated.changes, "raw": generated.raw}

        generation = self.step(run_id, "generation", generate)
        edited = self.asset(run_id, "counterfactual-0.png").read_bytes()
        checks = verify_occlusion(original, edited, generation["raw"])
        self.step(run_id, "edited-grounding", lambda: SamSegmenter(client).segment(edited, "white mug"))
        if known_source and digest(edited) == reviewed["edited_sha256"] and checks["passed"]:
            validity = EditValidator({"occlusion": reviewed["review"]}).validate(original, edited, task, actual, expected[1])
        else:
            validity = Validity("unverified", "Exact independent fixture review does not cover this image/task/edit. Review is required.", "independent_review_gate", {"checks": checks})
        if not checks["passed"]:
            validity = Validity("invalid", "Exact controlled edit checks failed.", "pixel_checks", {"checks": checks})
        self.step(run_id, "validity", lambda: asdict(validity))
        self.update(run_id, phase="Policy · reading the counterfactual independently")
        observed = self.step(run_id, "counterfactual-policy", lambda: asdict(policy.decide(edited, task)))
        results = [evaluate(expected[0], decision(baseline), Validity("valid", "Original input.", "input_record")),
                   evaluate(expected[1], decision(observed), validity)]
        contributions = self.contributions(selected, generation["changes"], "Live Muse Spark · hypothesis, not ground truth")
        contributions[1]["details"] += " Generator used the fixed occlusion recipe rather than free-form Adversary edits. Only occlusion was executed; other proposals are saved."
        self.complete(state, baseline, [{"proposal": asdict(actual), "policy": observed, "checks": checks}], results, contributions, failure_report(results))

    def _muse(self, state):
        run_id = state["id"]
        folder = self.folder(run_id)
        client = MetaClient(load_key(self.project / ".env"), folder / "agent-api")
        policy = self.policy_for(state)
        adversary, generator = SparkAdversary(client), MuseImageGenerator(client)
        store = RunStore(folder / "workflow")
        workflow = Workflow(store, policy, adversary, generator, EditValidator(),
                            {"policy": policy.settings, "adversary": adversary.settings, "generator": generator.settings})
        self.update(run_id, phase="Live workflow · original policy, three proposals, three edits, and evaluation")

        def create():
            original = self.asset(run_id, "original.png").read_bytes()
            return {"workflow_id": workflow.create(original, decode(original)[1], state["task"], task_criteria(state["task"]))["id"]}

        ref = self.step(run_id, "workflow-reference", create)
        run = workflow.resume(ref["workflow_id"])
        if run["status"] != "completed":
            if (run.get("error") or {}).get("type") == "BasetenError":
                raise BasetenError("Baseten: policy testing failed; see saved policy audit")
            raise RuntimeError("Workflow failed at " + str(run.get("current_step")))
        results = {r["criterion_id"]: r for r in run["report"]["results"]}
        ordered = [results["baseline"]]
        generated = []
        for index, proposal in enumerate(run["steps"]["adversary"]):
            image = store.read_image(run["id"], run["steps"][f"generate:{index}"]["artifact"])
            self.asset(run_id, f"counterfactual-{index}.png").write_bytes(image)
            generated.append({"proposal": proposal, "policy": run["steps"][f"policy:{index}"]})
            ordered.append(results[proposal["intervention"]])
        contributions = [{"role": "Adversary", "source": "Live Muse Spark", "summary": "Proposed three controlled intervention hypotheses.", "details": "Predictions were not used as expected-action criteria."},
                         {"role": "Generator", "source": "Live Muse Image · optional backend", "summary": "Edited the original reference image independently three times.", "details": "No mask constraint. Edit fidelity requires independent review."},
                         {"role": "Evaluator", "source": "Deterministic scoring", "summary": "Unreviewed expected actions and edits remain Inconclusive.", "details": "No correctness claims are inferred from the policy's own explanation."}]
        self.complete(state, run["steps"]["policy:baseline"], generated, ordered, contributions, run["report"])
