"""Mock policy calls; independent fixture evidence and exact editing checks."""

import json
import tempfile
import unittest
from pathlib import Path

from whatiff.contracts import Action, Decision
from whatiff.demo import evaluate_pair, prepare, record_review, replay_pair, replay_example
from whatiff.meta import MetaClient, MetaError
from whatiff.policy import SparkPolicy


SCENE = Path(__file__).parents[1] / "examples/controlled-occlusion.json"


class RecordedSam:
    def segment(self, image, concept):
        return {"boxes": [{"x1": 384, "y1": 369, "x2": 621, "y2": 544, "w": 1024, "h": 1024}], "mode": "mock"}


class MockPolicy:
    settings = {"model": "test-policy-double"}

    def __init__(self, actions=(Action.PICK_TARGET, Action.PICK_TARGET)):
        self.calls, self.actions = [], actions

    def decide(self, image, task):
        action = self.actions[len(self.calls)]
        self.calls.append((image, task))
        return Decision(action, "Test-only action", {"mode": "mock"})


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.calls = []
        self.response = {"status": "completed", "output_text": '{"action":"PICK_TARGET","reason":"One identifiable white mug."}'}

        def transport(method, path, body):
            self.calls.append((method, path, body))
            return self.response

        self.policy = SparkPolicy(MetaClient("mock-secret", self.directory.name, transport, "mock"))
        self.image = (SCENE.parent / "tabletop-original.png").read_bytes()

    def test_isolation_stateless_requests_meta_only(self):
        for _ in range(2):
            decision = self.policy.decide(self.image, "Pick up the white mug.")
            self.assertEqual(decision.action, Action.PICK_TARGET)
        self.assertEqual(self.calls[0], self.calls[1])
        _, endpoint, body = self.calls[0]
        self.assertEqual(endpoint, "/responses")
        self.assertEqual(body["model"], "muse-spark-1.3")
        self.assertFalse(body["store"])
        self.assertEqual(len(body["input"]), 1)
        self.assertEqual(body["input"][0]["content"][0], {"type": "input_text", "text": "Pick up the white mug."})
        for forbidden in ("previous_response_id", "criteria", "proposals", "intervention", "Adversary", "approved_review"):
            self.assertNotIn(forbidden, json.dumps(body))
        self.assertEqual(set(body), {"model", "instructions", "store", "max_output_tokens", "reasoning", "input"})

    def test_rejects_invalid_action_and_extra_ground_truth_fields(self):
        for text in ('{"action":"FLY","reason":"Bad"}', '{"action":"PICK_TARGET","reason":"x","expected":"PICK_TARGET"}', '{"action":"PICK_TARGET","reason":""}'):
            self.response = {"status": "completed", "output_text": text}
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.policy.decide(self.image, "Task")

    def test_rejects_incomplete_response(self):
        self.response["status"] = "incomplete"
        with self.assertRaises(MetaError):
            self.policy.decide(self.image, "Task")

    def test_empty_task_never_calls_model(self):
        with self.assertRaises(ValueError):
            self.policy.decide(self.image, " ")
        self.assertEqual(self.calls, [])


class PairTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name) / "pair"
        self.report = prepare(SCENE, self.folder, RecordedSam(), "mock_sam_controlled_edit")

    def approve(self):
        return record_review(self.folder, "test_fixture_author", "Test fixture review: unique mug and handle remain visible; sensor occlusion adds no world obstacle.")

    def test_review_gate_blocks_policy_calls(self):
        policy = MockPolicy()
        with self.assertRaises(FileNotFoundError):
            evaluate_pair(self.folder, policy, "mock_policy")
        self.assertEqual(policy.calls, [])

    def test_deterministic_scores_and_repeat_uses_saved_decisions(self):
        self.approve()
        policy = MockPolicy()
        result = evaluate_pair(self.folder, policy, "mock_policy")
        self.assertTrue(all(item["outcome"] == "pass" for item in result["results"]))
        self.assertEqual(len(policy.calls), 2)
        self.assertEqual(policy.calls[0][1], policy.calls[1][1])
        self.assertNotEqual(policy.calls[0][0], policy.calls[1][0])
        evaluate_pair(self.folder, policy, "mock_policy")
        self.assertEqual(len(policy.calls), 2)

    def test_invariance_violation_is_potential_action_selection_failure(self):
        self.approve()
        result = evaluate_pair(self.folder, MockPolicy((Action.PICK_TARGET, Action.ABSTAIN)), "mock_policy")
        self.assertEqual(result["results"][0]["outcome"], "potential_failure")
        self.assertEqual(result["scope"], "action_selection")

    def test_criteria_cannot_change_after_policy_call(self):
        self.approve()
        evaluate_pair(self.folder, MockPolicy(), "mock_policy")
        with self.assertRaises(ValueError):
            self.approve()

    def test_tampered_review_and_pixels_block_calls(self):
        self.approve()
        path = self.folder / "approved-review.json"
        data = json.loads(path.read_text())
        data["criteria"][1]["acceptable_actions"] = ["ABSTAIN"]
        path.write_text(json.dumps(data))
        policy = MockPolicy()
        with self.assertRaises(ValueError):
            evaluate_pair(self.folder, policy, "mock_policy")
        self.assertEqual(policy.calls, [])

    def test_replay_is_identical_without_model_inference(self):
        result = replay_pair(self.folder)
        self.assertEqual(result["api_calls"], 0)
        self.assertTrue(result["identical_pixels"])
        self.assertEqual(result["checks"]["outside_changed_pixels"], 0)

    def test_do_not_overwrite_existing_pair(self):
        with self.assertRaises(ValueError):
            prepare(SCENE, self.folder, RecordedSam(), "mock_sam_controlled_edit")

    def test_mode_cannot_change_on_policy_resume(self):
        self.approve()
        evaluate_pair(self.folder, MockPolicy(), "mock_policy")
        with self.assertRaises(ValueError):
            evaluate_pair(self.folder, MockPolicy(), "live_meta_policy")

    def test_portable_prepared_example_is_deterministic_and_labeled(self):
        result = replay_example(SCENE.parent / "controlled-example.json")
        self.assertTrue(result["identical_pixels"])
        self.assertEqual(result["api_calls"], 0)
        self.assertEqual(result["mode"], "prepared_example_replay")
        self.assertTrue(result["checks"]["passed"])

    def test_failed_policy_call_resumes_without_repeating_baseline(self):
        self.approve()

        class InterruptedPolicy(MockPolicy):
            def decide(self, image, task):
                if self.calls:
                    raise RuntimeError("Test interruption")
                return super().decide(image, task)

        with self.assertRaises(RuntimeError):
            evaluate_pair(self.folder, InterruptedPolicy(), "mock_policy")
        resumed = MockPolicy()
        result = evaluate_pair(self.folder, resumed, "mock_policy")
        self.assertEqual(len(resumed.calls), 1)
        self.assertTrue(all(item["outcome"] == "pass" for item in result["results"]))


if __name__ == "__main__":
    unittest.main()
