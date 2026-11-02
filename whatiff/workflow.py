"""A bounded controller: three interventions, checkpointed after each operation.

One process executes a run at a time. A provider call interrupted before its
checkpoint may execute again on retry; completed operations are reused.
"""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

from .contracts import Action, ACTION_MEANINGS, Criterion, Decision, Proposal, Validity
from .evaluation import evaluate, failure_report


def now():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


class Workflow:
    def __init__(self, store, policy, adversary, generator, validator, configuration):
        self.store = store
        self.policy = policy
        self.adversary = adversary
        self.generator = generator
        self.validator = validator
        # Non-secret model IDs, prompt versions and inference settings only.
        self.configuration = plain(configuration)

    def create(self, image, media_type, task, criteria, mode="live"):
        if mode not in ("live", "prepared_example", "test"):
            raise ValueError("Unknown run mode.")
        if not task.strip():
            raise ValueError("Task must not be empty.")
        if len(criteria) != 4 or len({c.id for c in criteria}) != 4:
            raise ValueError("Provide one baseline and three distinct intervention criteria.")
        baselines = [c for c in criteria if c.test_type == "baseline"]
        if len(baselines) != 1 or baselines[0].intervention != "baseline":
            raise ValueError("Exactly one baseline criterion is required.")
        if len({c.intervention for c in criteria}) != 4:
            raise ValueError("Interventions must be distinct.")
        baseline = baselines[0]
        for criterion in criteria:
            overlap = set(baseline.acceptable_actions) & set(criterion.acceptable_actions)
            if criterion.test_type == "invariance" and set(criterion.acceptable_actions) != set(baseline.acceptable_actions):
                raise ValueError("Invariance criteria must preserve the baseline acceptable actions.")
            if criterion.test_type == "adaptation" and overlap:
                raise ValueError("Adaptation criteria must require a different action.")
        run_id = uuid4().hex
        original = self.store.image(run_id, image, media_type)
        run = {
            "schema_version": 1, "id": run_id, "mode": mode, "task": task,
            "created_at": now(), "status": "pending", "configuration": self.configuration,
            "action_meanings": ACTION_MEANINGS,
            "criteria": plain([asdict(c) for c in criteria]), "original": original,
            "steps": {}, "activity": [], "error": None,
        }
        self.store.save(run)
        return run

    def _event(self, run, step, status):
        run["activity"].append({"at": now(), "step": step, "status": status})
        run["updated_at"] = now()
        self.store.save(run)

    def _step(self, run, key, operation):
        if key in run["steps"]:
            return run["steps"][key]
        run["current_step"] = key
        self._event(run, key, "started")
        result = plain(operation())
        run["steps"][key] = result
        self._event(run, key, "completed")
        return result

    def resume(self, run_id):
        run = self.store.load(run_id)
        if run["configuration"] != self.configuration:
            raise ValueError("Resume requires the original provider configuration.")
        if run["status"] == "completed":
            return run
        run.update(status="running", error=None)
        try:
            original = self.store.read_image(run_id, run["original"])
            criteria = [Criterion(**{**c, "acceptable_actions": tuple(Action(a) for a in c["acceptable_actions"])}) for c in run["criteria"]]
            baseline = next(c for c in criteria if c.test_type == "baseline")
            interventions = {c.intervention: c for c in criteria if c.test_type != "baseline"}
            baseline_decision = self._step(run, "policy:baseline", lambda: asdict(self.policy.decide(original, run["task"])))

            def propose():
                proposals = self.adversary.propose(original, run["task"])
                if len(proposals) != 3 or {p.intervention for p in proposals} != set(interventions):
                    raise ValueError("Adversary must propose exactly the three specified intervention types.")
                if any(not p.description.strip() or not p.rationale.strip() or not isinstance(p.predicted_action, Action) for p in proposals):
                    raise ValueError("Intervention proposals need descriptions, rationale, and valid predictions.")
                return [asdict(p) for p in proposals]

            proposals = self._step(run, "adversary", propose)
            baseline_result = evaluate(baseline, self._decision(baseline_decision), Validity("valid", "Unedited original observation.", "input_record"))
            baseline_result.update(original=run["original"], counterfactual=None, task=run["task"], intervention=None)
            results = [baseline_result]
            for index, data in enumerate(proposals):
                proposal = Proposal(**{**data, "predicted_action": Action(data["predicted_action"])})
                criterion = interventions[proposal.intervention]

                def generate():
                    observation = self.generator.generate(original, run["task"], proposal)
                    if not observation.changes.strip():
                        raise ValueError("Generator must record changes.")
                    artifact = self.store.image(run_id, observation.image, observation.media_type)
                    return {"artifact": artifact, "changes": observation.changes, "raw": observation.raw}

                generated = self._step(run, f"generate:{index}", generate)
                edited = self.store.read_image(run_id, generated["artifact"])
                validity = self._step(run, f"validate:{index}", lambda: asdict(self.validator.validate(original, edited, run["task"], proposal, criterion)))
                if edited == original:
                    validity = asdict(Validity("invalid", "Generator returned the unchanged original.", "image_integrity"))
                    run["steps"][f"validate:{index}"] = validity
                    self.store.save(run)
                decision = self._step(run, f"policy:{index}", lambda: asdict(self.policy.decide(edited, run["task"])))
                result = evaluate(criterion, self._decision(decision), Validity(**validity))
                result.update(original=run["original"], counterfactual=generated["artifact"], intervention=data, task=run["task"])
                results.append(result)
            run["report"] = failure_report(results)
            run.update(status="completed", current_step=None)
            self._event(run, "report", "completed")
        except Exception as error:
            # Keep only the error type here: provider exceptions may contain credentials.
            run.update(status="failed", error={"step": run.get("current_step"), "type": type(error).__name__, "retryable": True})
            self._event(run, run.get("current_step"), "failed")
        return run

    @staticmethod
    def _decision(data):
        return Decision(Action(data["action"]), data["reason"], data["raw"])
