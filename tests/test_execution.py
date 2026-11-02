"""Execution projection tests use synthetic audit records, never live model calls."""

import json
import tempfile
import unittest
from pathlib import Path

from whatiff.execution import audited_calls, projection


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.state = {"mode": "live", "status": "completed", "report": {"results": [{}, {}, {}, {}]}}

    def call(self, area, name, model, status="completed"):
        folder = self.folder / area / name
        folder.mkdir(parents=True)
        (folder / "request.json").write_text(json.dumps({"mode": "mock", "body": {"model": model, "instructions": "PRIVATE PROMPT SECRET"}}))
        if status != "running":
            (folder / "response.json").write_text(json.dumps({"status": status}))
        if status == "failed":
            (folder / "error.json").write_text(json.dumps({"code": "PRIVATE ERROR"}))

    def stages(self, **options):
        result = projection(self.state, self.folder, **options)
        return result, {stage["role"]: stage for stage in result["stages"]}

    def test_controlled_records_show_sam_pillow_and_skip_muse_image(self):
        self.call("agent-api", "plan", "muse-spark-1.3")
        for index in range(2):
            self.call("agent-api", f"sam-{index}", "sam-3.1")
            self.call("policy-api", f"policy-{index}", "muse-spark-1.3")
        (self.folder / "generation.json").write_text(json.dumps({"raw": {"settings": {"backend": "pillow_controlled_occlusion"}}}))
        result, stages = self.stages()
        self.assertEqual(result["model_calls_this_execution"], 5)
        generator = {tool["name"]: tool for tool in stages["Generator"]["tools"]}
        self.assertEqual(generator["SAM 3.1"]["calls"], 2)
        self.assertEqual(generator["SAM 3.1"]["status"], "completed")
        self.assertEqual(generator["Pillow"]["status"], "completed")
        self.assertEqual(generator["Muse Image"]["status"], "skipped")
        self.assertEqual(stages["Evaluator"]["tools"][0]["name"], "Deterministic verification")
        self.assertNotIn("PRIVATE", json.dumps(result))

    def test_muse_records_match_counts_and_unused_sam_is_skipped(self):
        self.call("agent-api", "plan", "muse-spark-1.3")
        for index in range(3):
            self.call("agent-api", f"image-{index}", "muse-image-1.0")
        for index in range(4):
            self.call("policy-api", f"policy-{index}", "muse-spark-1.3")
        result, stages = self.stages()
        calls = audited_calls(self.folder)
        self.assertEqual(result["model_calls_this_execution"], len(calls))
        self.assertEqual(len(calls), 8)
        self.assertEqual(stages["Adversary"]["tools"][0]["calls"], 1)
        generator = {tool["name"]: tool for tool in stages["Generator"]["tools"]}
        self.assertEqual(generator["Muse Image"]["calls"], 3)
        self.assertEqual(generator["SAM 3.1"]["status"], "skipped")
        self.assertEqual(stages["Policy"]["tools"][0]["calls"], 4)

    def test_request_without_response_is_running(self):
        self.state.update(status="running", report=None)
        self.call("policy-api", "pending", "muse-spark-1.3", "running")
        result, stages = self.stages()
        self.assertEqual(stages["Policy"]["status"], "running")
        self.assertEqual(stages["Adversary"]["status"], "waiting")
        self.assertEqual(stages["Evaluator"]["status"], "waiting")
        self.assertEqual(result["model_calls_this_execution"], 1)

    def test_failed_and_unrecorded_models_are_not_completed(self):
        self.state.update(status="failed", report=None)
        self.call("agent-api", "error", "muse-image-1.0", "failed")
        _, stages = self.stages()
        self.assertEqual(stages["Generator"]["status"], "failed")
        self.assertEqual(stages["Policy"]["status"], "skipped")
        self.assertEqual(stages["Evaluator"]["status"], "skipped")

    def test_workflow_checkpoint_progress_is_not_assumed_from_selected_backend(self):
        self.state.update(status="running", report=None)
        self.call("policy-api", "baseline", "muse-spark-1.3")
        (self.folder / "workflow-reference.json").write_text(json.dumps({"workflow_id": "a" * 32}))
        checkpoint = self.folder / "workflow" / ("a" * 32)
        checkpoint.mkdir(parents=True)
        (checkpoint / "run.json").write_text(json.dumps({"current_step": "generate:1", "steps": {"adversary": [{}, {}, {}], "generate:0": {}, "validate:0": {}}}))
        _, stages = self.stages()
        self.assertEqual(stages["Generator"]["status"], "running")
        self.assertEqual(stages["Policy"]["status"], "waiting")
        self.assertEqual(stages["Evaluator"]["status"], "running")

    def test_replay_counts_only_recorded_calls_and_exposes_no_audit_payload(self):
        fixture = Path(__file__).resolve().parents[1] / "examples/washing-dishes-run/execution.json"
        recorded = json.loads(fixture.read_text())
        self.state["mode"] = "prepared_example"
        result, stages = self.stages(recorded=recorded)
        self.assertEqual(result["mode"], "saved_replay")
        self.assertEqual(result["model_calls_this_execution"], 0)
        self.assertEqual(result["recorded_model_calls"], len(recorded["calls"]))
        self.assertEqual(result["recorded_model_calls"], 8)
        for stage in result["stages"]:
            self.assertEqual(stage["status"], "completed")
        self.assertNotIn("request", json.dumps(result))
        self.assertNotIn("instructions", json.dumps(result))
        self.assertNotIn("/Users/", json.dumps(result))
        self.assertEqual(stages["Generator"]["tools"][1]["status"], "skipped")

    def test_failed_later_sam_call_does_not_appear_completed(self):
        self.state.update(status="failed", report=None)
        self.call("agent-api", "sam-first", "sam-3.1")
        self.call("agent-api", "sam-second", "sam-3.1", "failed")
        _, stages = self.stages()
        self.assertEqual(stages["Generator"]["status"], "failed")
        self.assertEqual(stages["Generator"]["tools"][1]["status"], "failed")

    def test_backend_selection_without_audit_does_not_invent_calls(self):
        self.state["backend"] = "muse"
        result, stages = self.stages()
        self.assertEqual(result["model_calls_this_execution"], 0)
        self.assertEqual(stages["Generator"]["status"], "skipped")
        self.assertEqual(stages["Adversary"]["status"], "skipped")

    def test_baseten_policy_hides_unused_hosted_model(self):
        self.state['policy_provider'] = 'baseten'
        model = 'meta-models/muse-spark-open-1026'
        self.call('policy-api', 'open', model, 'running')
        _, stages = self.stages()
        self.assertEqual(stages['Policy']['tools'], [{'name': 'Muse Spark Open · Baseten', 'model': model, 'status': 'running', 'calls': 1}])

    def test_selected_policy_without_calls_never_invents_completion(self):
        self.state.update(status='running', policy_provider='baseten')
        _, stages = self.stages()
        self.assertEqual(len(stages['Policy']['tools']), 1)
        self.assertEqual(stages['Policy']['tools'][0]['status'], 'waiting')
        self.assertEqual(stages['Policy']['tools'][0]['calls'], 0)

    def test_replay_model_comes_from_saved_calls_not_new_default(self):
        self.state.update(mode='prepared_example', policy_provider='baseten')
        fixture = Path(__file__).resolve().parents[1] / 'examples/washing-dishes-run/execution.json'
        _, stages = self.stages(recorded=json.loads(fixture.read_text()))
        self.assertEqual(len(stages['Policy']['tools']), 1)
        self.assertEqual(stages['Policy']['tools'][0]['name'], 'Muse Spark 1.3')
        self.assertEqual(stages['Policy']['tools'][0]['calls'], 4)
