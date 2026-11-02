import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from whatiff.contracts import Action, Criterion, Decision, GeneratedObservation, Proposal, Validity
from whatiff.evaluation import evaluate
from whatiff.store import RunStore
from whatiff.workflow import Workflow


def criteria(approved=True):
    return tuple(Criterion(name, name, kind, (action,), "Curator-defined requirement", "Test evidence" if approved else "", "test_curator", approved)
                 for name, kind, action in (
                     ("baseline", "baseline", Action.PICK_TARGET),
                     ("occlusion", "invariance", Action.PICK_TARGET),
                     ("distractor", "invariance", Action.PICK_TARGET),
                     ("relocation", "adaptation", Action.REPOSITION)))


class FakePolicy:
    """Unit-test double using opaque bytes; not a demonstrated robot policy."""
    def __init__(self):
        self.calls = []

    def decide(self, image, task):
        self.calls.append((image, task))
        action = Action.ABSTAIN if image.endswith(b"distractor") else Action.REPOSITION if image.endswith(b"relocation") else Action.PICK_TARGET
        return Decision(action, "Test response", {"action": action.value})


class FakeAdversary:
    def __init__(self):
        self.calls = 0

    def propose(self, image, task):
        self.calls += 1
        # Deliberately wrong predictions prove they do not become ground truth.
        return tuple(Proposal(name, "Edit " + name, "Test hypothesis", Action.ABSTAIN)
                     for name in ("occlusion", "distractor", "relocation"))


class FakeGenerator:
    def __init__(self):
        self.calls = []
        self.fail_once = None

    def generate(self, image, task, proposal):
        self.calls.append((image, proposal.intervention))
        if self.fail_once == proposal.intervention:
            self.fail_once = None
            raise RuntimeError("Simulated transient failure")
        return GeneratedObservation(b"edited:" + proposal.intervention.encode(), "image/png", proposal.description, {"test": True})


class FakeValidator:
    def __init__(self):
        self.status = "valid"

    def validate(self, original, edited, task, proposal, criterion):
        return Validity(self.status, "Test-only edit evidence", "test_curator")


class ScoringTests(unittest.TestCase):
    def test_invariance_pass_does_not_use_adversary_prediction(self):
        result = evaluate(criteria()[1], Decision(Action.PICK_TARGET, "", {}), Validity("valid", "Checked", "curator"))
        self.assertEqual(result["outcome"], "pass")

    def test_changed_action_is_valid_adaptation(self):
        result = evaluate(criteria()[3], Decision(Action.REPOSITION, "", {}), Validity("valid", "Checked", "curator"))
        self.assertEqual(result["outcome"], "pass")

    def test_unchanged_action_can_violate_adaptation(self):
        result = evaluate(criteria()[3], Decision(Action.PICK_TARGET, "", {}), Validity("valid", "Checked", "curator"))
        self.assertEqual(result["outcome"], "potential_failure")
        self.assertEqual(result["scope"], "action_selection")

    def test_unapproved_criteria_are_inconclusive(self):
        result = evaluate(criteria(False)[1], Decision(Action.ABSTAIN, "", {}), Validity("valid", "Checked", "curator"))
        self.assertEqual(result["outcome"], "inconclusive")

    def test_invalid_and_unverified_edits_are_inconclusive(self):
        for status in ("invalid", "unverified"):
            with self.subTest(status=status):
                result = evaluate(criteria()[1], Decision(Action.ABSTAIN, "", {}), Validity(status, "", ""))
                self.assertEqual(result["outcome"], "inconclusive")

    def test_multiple_acceptable_actions(self):
        criterion = replace(criteria()[1], acceptable_actions=(Action.PICK_TARGET, Action.REPOSITION))
        result = evaluate(criterion, Decision(Action.REPOSITION, "", {}), Validity("valid", "Checked", "curator"))
        self.assertEqual(result["outcome"], "pass")

    def test_malformed_evidence_and_actions_rejected(self):
        with self.assertRaises(ValueError):
            replace(criteria()[1], evidence="")
        with self.assertRaises(ValueError):
            Validity("valid", "", "curator")
        with self.assertRaises(ValueError):
            Decision("FLY", "", {})


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = RunStore(self.directory.name)
        self.policy = FakePolicy()
        self.adversary = FakeAdversary()
        self.generator = FakeGenerator()
        self.validator = FakeValidator()
        self.workflow = self.controller()

    def controller(self):
        return Workflow(self.store, self.policy, self.adversary, self.generator, self.validator,
                        {"policy": "test-double-v1", "agents": "test-double-v1"})

    def create(self, expected=None):
        return self.workflow.create(b"test-only-original", "image/png", "Pick up the white mug.", expected or criteria(), mode="test")

    def test_bounded_workflow_artifacts_and_ranked_report(self):
        run = self.workflow.resume(self.create()["id"])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(len(self.policy.calls), 4)
        self.assertEqual(len(self.generator.calls), 3)
        self.assertTrue(all(image == b"test-only-original" for image, _ in self.generator.calls))
        report = run["report"]
        self.assertEqual(report["results"][0]["criterion_id"], "distractor")
        self.assertEqual(report["results"][0]["outcome"], "potential_failure")
        self.assertEqual(sum(r["outcome"] == "pass" for r in report["results"]), 3)
        self.assertEqual(report["scope"], "action_selection")
        self.assertEqual(len(list(self.store.folder(run["id"]).glob("images/*"))), 4)
        self.assertEqual(run["steps"]["policy:baseline"]["raw"]["action"], "PICK_TARGET")
        self.assertEqual(self.store.load(run["id"]), run)

    def test_resume_after_failure_reuses_completed_steps(self):
        self.generator.fail_once = "distractor"
        failed = self.workflow.resume(self.create()["id"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error"]["step"], "generate:1")
        completed = self.controller().resume(failed["id"])
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(self.adversary.calls, 1)
        self.assertEqual(len(self.policy.calls), 4)
        self.assertEqual([name for _, name in self.generator.calls].count("occlusion"), 1)
        self.controller().resume(completed["id"])
        self.assertEqual(len(self.policy.calls), 4)

    def test_interrupted_run_resumes_from_disk(self):
        run = self.create()
        run["status"] = "running"
        run["steps"]["policy:baseline"] = {"action": "PICK_TARGET", "reason": "Saved before interruption", "raw": {}}
        self.store.save(run)
        self.assertEqual(self.controller().resume(run["id"])["status"], "completed")
        self.assertEqual(len(self.policy.calls), 3)

    def test_configuration_changes_block_resume(self):
        run = self.create()
        self.workflow.configuration = {"policy": "different"}
        with self.assertRaises(ValueError):
            self.workflow.resume(run["id"])

    def test_validation_retry_reuses_generated_artifact(self):
        original_validate = self.validator.validate

        def fail(*args):
            raise RuntimeError("Unavailable validation service")

        self.validator.validate = fail
        run = self.workflow.resume(self.create()["id"])
        self.assertEqual(run["error"]["step"], "validate:0")
        self.validator.validate = original_validate
        self.assertEqual(self.controller().resume(run["id"])["status"], "completed")
        self.assertEqual(len(self.generator.calls), 3)

    def test_unapproved_expectations_never_produce_supported_outcomes(self):
        run = self.workflow.resume(self.create(criteria(False))["id"])
        self.assertTrue(all(result["outcome"] == "inconclusive" for result in run["report"]["results"]))

    def test_unchanged_generated_observation_is_inconclusive(self):
        self.generator.generate = lambda image, task, proposal: GeneratedObservation(image, "image/png", "No edit", {})
        run = self.workflow.resume(self.create()["id"])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(sum(r["outcome"] == "inconclusive" for r in run["report"]["results"]), 3)

    def test_invalid_proposal_count_is_checkpointed_as_failure(self):
        self.adversary.propose = lambda image, task: ()
        run = self.workflow.resume(self.create()["id"])
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["error"]["step"], "adversary")
        self.assertEqual(self.generator.calls, [])

    def test_no_sam_needed_and_unverified_edits_are_inconclusive(self):
        self.validator.status = "unverified"
        run = self.workflow.resume(self.create()["id"])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(sum(r["outcome"] == "inconclusive" for r in run["report"]["results"]), 3)

    def test_inconsistent_test_types_rejected(self):
        expected = list(criteria())
        expected[3] = replace(expected[3], acceptable_actions=(Action.PICK_TARGET,))
        with self.assertRaises(ValueError):
            self.create(expected)

    def test_tampered_original_is_detected(self):
        run = self.create()
        path = self.store.folder(run["id"]) / run["original"]["path"]
        path.write_bytes(b"tampered")
        self.assertEqual(self.workflow.resume(run["id"])["status"], "failed")
        self.assertEqual(self.policy.calls, [])

    def test_store_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            self.store.load("../elsewhere")

    def test_example_template_is_unapproved(self):
        template = json.loads((Path(__file__).parents[1] / "examples/tabletop-criteria.json").read_text())
        self.assertEqual(len(template["criteria"]), 4)
        self.assertTrue(all(not c["approved"] for c in template["criteria"]))


if __name__ == "__main__":
    unittest.main()
