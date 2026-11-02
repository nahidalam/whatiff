"""Local API/service tests. All model interactions are mocked here."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from whatiff.contracts import Action, Decision, Proposal
from whatiff.controlled import SamOcclusionGenerator
from whatiff.interface import InterfaceService, task_criteria
from whatiff.web import PROJECT, create_app


class FakePolicy:
    settings = {"model": "muse-spark-1.3", "test": True}
    calls = []

    def __init__(self, client):
        pass

    def decide(self, image, task):
        self.calls.append((image, task))
        return Decision(Action.PICK_TARGET, "Mock decision for an identifiable white mug.", {"mode": "mock"})


class FakeAdversary:
    settings = {"model": "muse-spark-1.3", "test": True}
    calls = 0

    def __init__(self, client):
        pass

    def propose(self, image, task):
        type(self).calls += 1
        return tuple(Proposal(name, "Mock " + name, "Mock hypothesis", Action.ABSTAIN) for name in ("occlusion", "distractor", "relocation"))


class FakeSam:
    def __init__(self, client):
        pass

    def segment(self, image, concept):
        return {"mode": "mock", "boxes": [{"object_id": 0, "x1": 384, "y1": 369, "x2": 621, "y2": 544, "w": 1024, "h": 1024}]}


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.app = create_app(PROJECT, Path(self.directory.name) / "web")
        self.service = self.app.state.service
        self.client = TestClient(self.app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        FakePolicy.calls = []
        FakeAdversary.calls = 0

    def wait(self):
        self.service.executor.submit(lambda: None).result(timeout=5)

    def test_saved_replay_has_no_api_calls_and_downloadable_report(self):
        with patch("whatiff.meta.HTTPTransport.__call__", side_effect=AssertionError("Replay called a model")), patch("whatiff.interface.load_key", side_effect=AssertionError("Replay loaded a credential")):
            response = self.client.post("/api/replay")
        self.assertEqual(response.status_code, 200)
        run = response.json()
        self.assertEqual(run["mode"], "prepared_example")
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["cases"][0]["outcome"], "inconclusive")
        self.assertEqual(run["task"], "Pick up the sponge to clean the dish.")
        self.assertIn("Recorded", run["contributions"][0]["source"])
        for item in (run["original"], run["cases"][0]):
            image = self.client.get(item["image_url"])
            self.assertEqual(image.status_code, 200)
            self.assertIn(image.headers["content-type"], ("image/png", "image/jpeg"))
        report = self.client.get(f"/api/runs/{run['id']}/report")
        self.assertIn("attachment", report.headers["content-disposition"])
        self.assertEqual(report.json()["mode"], "prepared_example")
        self.assertEqual(self.client.get(f"/api/runs/{run['id']}").json(), run)

    def test_review_preserves_failed_checks_and_records_rejection(self):
        run = self.client.post("/api/replay").json()
        case = run["cases"][0]
        self.assertEqual(case["validity"]["status"], "invalid")
        url = f"/api/runs/{run['id']}/cases/0/review"
        response = self.client.post(url, json={"reviewer": "Reviewer", "edit_status": "approved"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["cases"][0]["review"]["edit_status"], "approved")
        self.assertEqual(response.json()["cases"][0]["validity"]["status"], "invalid")
        response = self.client.post(url, json={"reviewer": "Reviewer", "edit_status": "rejected", "edit_note": "Image dimensions changed."})
        self.assertEqual(response.status_code, 200)
        saved = response.json()
        self.assertEqual(saved["cases"][0]["review"]["edit_status"], "rejected")
        self.assertEqual(saved["cases"][0]["outcome"], "inconclusive")
        report = self.service.get(run['id'])["report"]
        self.assertEqual(report["review"]["cases"]["0"]["original_validation"], case["validity"])
        self.assertEqual(report["review"]["history"][1]["edit_note"], "Image dimensions changed.")
        self.assertEqual(self.service.get(run['id'])["cases"][0]["review"]["edit_reviewer"], "Reviewer")

    def test_expected_action_review_cannot_validate_an_invalid_edit(self):
        run = self.client.post("/api/replay").json()
        response = self.client.post(f"/api/runs/{run['id']}/cases/0/review", json={"reviewer": "Reviewer", "acceptable_actions": ["PICK_TARGET"], "baseline_actions": ["PICK_TARGET"], "criterion_evidence": "Target is uniquely identifiable in both observations."})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["cases"][0]["criterion_approved"])
        self.assertEqual(result["cases"][0]["outcome"], "inconclusive")
        self.assertEqual(result["cases"][0]["validity"]["status"], "invalid")
        self.assertEqual(result["original"]["outcome"], "pass")

    def test_unverified_edit_and_criteria_require_separate_approval(self):
        run = self.client.post("/api/replay").json()
        # Test fixture only: exercise an edit with no failed automatic check.
        validity = {"status": "unverified", "evidence": "Mock edit awaiting review.", "source": "mock", "raw": {}}
        run["cases"][0]["validity"] = validity
        for result in run["report"]["results"]:
            if result["criterion_id"] == "occlusion":
                result["validity"] = validity
        self.service.update(run['id'], cases=run["cases"], report=run["report"])
        url = f"/api/runs/{run['id']}/cases/0/review"
        with patch("whatiff.meta.HTTPTransport.__call__", side_effect=AssertionError("Review called a model")):
            result = self.client.post(url, json={"reviewer": "Reviewer", "edit_status": "approved"}).json()
            self.assertEqual(result["cases"][0]["outcome"], "inconclusive")
            self.assertFalse(result["cases"][0]["criterion_approved"])
            self.assertNotEqual(self.service.compact_export(run['id'])["scenarios"][0]["verification_status"], "verified")
            result = self.client.post(url, json={"reviewer": "Reviewer", "acceptable_actions": ["PICK_TARGET"], "baseline_actions": ["PICK_TARGET"], "criterion_evidence": "Mock independent visual review supports pickup."}).json()
        self.assertEqual(result["cases"][0]["outcome"], "pass")
        result = self.client.post(url, json={"reviewer": "Reviewer", "acceptable_actions": ["ABSTAIN"], "baseline_actions": ["PICK_TARGET"], "criterion_evidence": "Mock independent review requires abstention."}).json()
        self.assertEqual(result["cases"][0]["outcome"], "potential_failure")
        self.assertEqual(len(result["report"]["review"]["history"]), 3)

    def test_review_rejects_missing_reviewer_evidence_and_unknown_actions(self):
        run = self.client.post("/api/replay").json()
        url = f"/api/runs/{run['id']}/cases/0/review"
        for data in ({"edit_status": "rejected"}, {"reviewer": "Reviewer", "acceptable_actions": ["PICK_TARGET"]}, {"reviewer": "Reviewer", "acceptable_actions": ["UNKNOWN"], "baseline_actions": ["PICK_TARGET"], "criterion_evidence": "Independent evidence"}):
            self.assertEqual(self.client.post(url, json=data).status_code, 400)
        self.assertNotIn("review", self.service.get(run['id'])["report"])

    def test_replay_audit_references_are_bundled_and_portable(self):
        folder = PROJECT / "examples/washing-dishes-run"
        saved = json.loads((folder / "state.json").read_text())
        self.assertEqual(saved["recording_provenance"]["replay_model_calls"], 0)
        self.assertIn("Actual live Meta", saved["recording_provenance"]["source"])
        references = []
        for result in saved["report"]["results"]:
            audit = (result.get("intervention") or {}).get("raw", {}).get("audit", {})
            for key in ("request", "response"):
                if key in audit:
                    path = Path(audit[key])
                    self.assertFalse(path.is_absolute())
                    self.assertTrue((folder / path).is_file())
                    references.append(path)
        self.assertTrue(references)

    def test_compact_export_is_small_relative_and_has_no_audit_payload(self):
        run = self.client.post("/api/replay").json()
        compact = self.client.get(f"/api/runs/{run['id']}/report").json()
        self.assertEqual(compact["task"], run["task"])
        self.assertEqual(compact["policy_model"], "muse-spark-1.3")
        self.assertEqual(compact["original"]["image"], "original.png")
        self.assertEqual(len(compact["scenarios"]), 3)
        self.assertLess(len(json.dumps(compact)), 3500)
        self.assertNotIn("review", compact)
        self.assertNotIn("results", compact)
        for index, scenario in enumerate(compact["scenarios"]):
            self.assertEqual(scenario["image"], f"counterfactual-{index}.png")
            self.assertTrue(self.service.asset(run['id'], scenario["image"]).is_file())
            self.assertIsNone(scenario["expected_actions"])
            self.assertEqual(scenario["verification_status"], "invalid")
            self.assertEqual(scenario["evaluation_outcome"], "inconclusive")
            self.assertTrue(scenario["evidence_id"].startswith(compact["evidence_id"]))
        folder = self.service.folder(run['id'])
        self.assertEqual(json.loads((folder / "evaluation.json").read_text()), compact)
        detailed = json.loads((folder / "report.json").read_text())
        self.assertIn("raw", detailed["results"][0]["validity"])
        self.assertTrue((folder / "report-before-task-criteria.json").is_file())

    def test_compact_verified_requires_edit_criteria_and_checks(self):
        from PIL import Image
        from io import BytesIO
        run = self.client.post("/api/replay").json()
        # Independent mock fixture, not a normalization strategy for real edits.
        source = self.service.asset(run['id'], "original.png").read_bytes()
        image = Image.open(BytesIO(source)).convert("RGB")
        image.putpixel((0, 0), (0, 0, 0))
        output = BytesIO(); image.save(output, format="PNG")
        self.service.asset(run['id'], "counterfactual-0.png").write_bytes(output.getvalue())
        validity = {"status": "unverified", "evidence": "Mock semantic review pending.", "source": "mock", "raw": {"preservation": {"status": "consistent"}}}
        run["cases"][0].update(validity=validity, image_dimensions={"original": [960, 540], "generated": [960, 540], "normalization": "none"}, checks={"passed": True})
        for result in run["report"]["results"]:
            if result["criterion_id"] == "occlusion":
                result["validity"] = validity
        self.service.update(run['id'], cases=run["cases"], report=run["report"])
        url = f"/api/runs/{run['id']}/cases/0/review"
        self.client.post(url, json={"reviewer": "Reviewer", "edit_status": "approved"})
        self.assertEqual(self.service.compact_export(run['id'])["scenarios"][0]["verification_status"], "unverified")
        self.client.post(url, json={"reviewer": "Reviewer", "acceptable_actions": ["PICK_TARGET"], "baseline_actions": ["PICK_TARGET"], "criterion_evidence": "Mock independent target eligibility review."})
        result = self.service.compact_export(run['id'])["scenarios"][0]
        self.assertEqual(result["verification_status"], "verified")
        self.assertEqual(result["evaluation_outcome"], "pass")
        current = self.service.get(run['id'])
        current["cases"][0]["checks"]["passed"] = False
        self.service.update(run['id'], cases=current["cases"])
        self.assertNotEqual(self.service.compact_export(run['id'])["scenarios"][0]["verification_status"], "verified")

    def test_task_specific_criteria_never_reuse_mug_requirements(self):
        for task in ("Pick up the sponge to clean the dish.", "Pick up the knife to cut carrots."):
            rules = task_criteria(task)
            self.assertEqual(len(rules), 4)
            for rule in rules:
                self.assertIn(task, rule.requirement)
                self.assertNotIn("mug", rule.requirement)
                self.assertFalse(rule.approved)
                self.assertEqual(set(rule.acceptable_actions), set(Action))
        run = self.client.post("/api/replay").json()
        for result in run["report"]["results"]:
            self.assertIn(run["task"], result["requirement"])
            self.assertNotIn("mug", result["requirement"])
        archived = json.loads((self.service.folder(run['id']) / "report-before-task-criteria.json").read_text())
        self.assertTrue(any("mug" in result["requirement"] for result in archived["results"]))

    def test_dimension_failure_survives_both_approvals_and_compact_export(self):
        run = self.client.post("/api/replay").json()
        url = f"/api/runs/{run['id']}/cases/0/review"
        first = self.service.compact_export(run['id'])["scenarios"][0]["evidence_id"]
        self.client.post(url, json={"reviewer": "Reviewer", "edit_status": "approved"})
        self.client.post(url, json={"reviewer": "Reviewer", "acceptable_actions": ["PICK_TARGET"], "baseline_actions": ["PICK_TARGET"], "criterion_evidence": "Independent visual review supports target pickup."})
        compact = self.client.get(f"/api/runs/{run['id']}/report").json()
        scenario = compact["scenarios"][0]
        self.assertEqual(scenario["edit_approval"], "approved")
        self.assertEqual(scenario["criterion_approval"], "approved")
        self.assertEqual(scenario["expected_actions"], ["PICK_TARGET"])
        self.assertEqual(scenario["verification_status"], "invalid")
        self.assertEqual(scenario["evaluation_outcome"], "inconclusive")
        self.assertEqual(scenario["evidence_id"], first)
        detailed = self.service.get(run['id'])["report"]
        result = next(result for result in detailed["results"] if result["criterion_id"] == "occlusion")
        self.assertEqual(result["validity"]["raw"]["dimensions"], {"original": [960, 540], "generated": [2048, 1152], "normalization": "none"})
        self.assertEqual(len(detailed["review"]["history"]), 2)

    def test_reveal_opens_only_the_saved_run_folder(self):
        run = self.client.post("/api/replay").json()
        self.assertEqual(run["artifact_directory"], str(self.service.folder(run['id']).resolve()))
        with patch("whatiff.web.subprocess.run") as opened:
            response = self.client.post(f"/api/runs/{run['id']}/reveal")
        self.assertEqual(response.status_code, 200)
        opened.assert_called_once_with(["open", run["artifact_directory"]], check=True, timeout=10)
        with patch("whatiff.web.subprocess.run") as opened:
            self.assertEqual(self.client.post("/api/runs/invalid/reveal").status_code, 400)
            opened.assert_not_called()

    def test_live_controlled_reuses_backend_and_preserves_exact_fixture_review(self):
        with patch("whatiff.interface.load_key", return_value="test-key"), patch("whatiff.interface.SparkPolicy", FakePolicy), patch("whatiff.interface.SparkAdversary", FakeAdversary), patch("whatiff.interface.SamSegmenter", FakeSam):
            response = self.client.post("/api/runs", data={"task": "Pick up the white mug.", "policy_provider": "meta", "backend": "controlled"})
            self.assertEqual(response.status_code, 202)
            self.wait()
        run = self.service.get(response.json()["id"])
        self.assertEqual(run["status"], "completed", run.get("error"))
        self.assertEqual(run["mode"], "live")
        self.assertEqual(run["cases"][0]["outcome"], "pass")
        self.assertEqual(len(FakePolicy.calls), 2)
        self.assertEqual(FakeAdversary.calls, 1)
        self.assertTrue(all(task == run["task"] for _, task in FakePolicy.calls))

    def test_changed_task_cannot_inherit_fixture_approval(self):
        with patch("whatiff.interface.load_key", return_value="test-key"), patch("whatiff.interface.SparkPolicy", FakePolicy), patch("whatiff.interface.SparkAdversary", FakeAdversary), patch("whatiff.interface.SamSegmenter", FakeSam):
            response = self.client.post("/api/runs", data={"task": "Pick up the green circle.", "policy_provider": "meta", "backend": "controlled"})
            self.wait()
        run = self.service.get(response.json()["id"])
        self.assertEqual(run["cases"][0]["outcome"], "inconclusive")
        self.assertFalse(run["original"]["criterion_approved"])
        self.assertEqual(run["cases"][0]["validity"]["status"], "unverified")

    def test_optional_muse_backend_uses_existing_three_case_workflow(self):
        class FakeMuse:
            settings = {"model": "muse-image-1.0", "test": True}

            def __init__(self, client):
                pass

            def generate(self, image, task, proposal):
                return SamOcclusionGenerator(FakeSam(None)).generate(image, task, Proposal("occlusion", "Mock edit", "Test only", Action.ABSTAIN))

        with patch("whatiff.interface.load_key", return_value="test-key"), patch("whatiff.interface.SparkPolicy", FakePolicy), patch("whatiff.interface.SparkAdversary", FakeAdversary), patch("whatiff.interface.MuseImageGenerator", FakeMuse):
            response = self.client.post("/api/runs", data={"task": "Pick up the sponge to clean the dish.", "policy_provider": "meta", "backend": "muse"})
            self.wait()
        run = self.service.get(response.json()["id"])
        self.assertEqual(run["status"], "completed", run.get("error"))
        self.assertEqual(len(run["cases"]), 3)
        self.assertTrue(all(c["outcome"] == "inconclusive" for c in run["cases"]))
        self.assertEqual(len(FakePolicy.calls), 4)

        ref = json.loads((self.service.folder(run['id']) / "workflow-reference.json").read_text())
        checkpoint = json.loads((self.service.folder(run['id']) / "workflow" / ref["workflow_id"] / "run.json").read_text())
        for rule in checkpoint["criteria"]:
            self.assertIn(run["task"], rule["requirement"])
            self.assertNotIn("mug", rule["requirement"])
            self.assertFalse(rule["approved"])

    def test_bad_image_and_unknown_backend_rejected(self):
        response = self.client.post("/api/runs", data={"task": "Task", "policy_provider": "meta", "backend": "controlled"}, files={"image": ("invalid.png", b"not an image", "image/png")})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/runs", data={"task": "Task", "policy_provider": "meta", "backend": "other"})
        self.assertEqual(response.status_code, 400)

    def test_missing_key_does_not_fabricate_live_result(self):
        from whatiff.meta import MetaError
        with patch("whatiff.interface.load_key", side_effect=MetaError("missing_model_api_key")):
            response = self.client.post("/api/runs", data={"task": "Task", "policy_provider": "meta"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "missing_model_api_key")

    def test_only_one_live_run_and_no_export_before_completion(self):
        with patch("whatiff.interface.load_key", return_value="test-key"), patch.object(self.service.executor, "submit"):
            response = self.client.post("/api/runs", data={"task": "Task", "policy_provider": "meta"})
            run_id = response.json()["id"]
            self.assertEqual(self.client.post("/api/runs", data={"task": "Task", "policy_provider": "meta"}).status_code, 409)
        self.assertEqual(self.client.get(f"/api/runs/{run_id}/report").status_code, 409)

    def test_retry_keeps_completed_policy_and_adversary_steps(self):
        class OfflineSam(FakeSam):
            def segment(self, image, concept):
                raise RuntimeError("Mock SAM outage")

        with patch("whatiff.interface.load_key", return_value="test-key"), patch("whatiff.interface.SparkPolicy", FakePolicy), patch("whatiff.interface.SparkAdversary", FakeAdversary), patch("whatiff.interface.SamSegmenter", OfflineSam):
            response = self.client.post("/api/runs", data={"task": "Pick up the white mug.", "policy_provider": "meta"})
            self.wait()
        run_id = response.json()["id"]
        self.assertEqual(self.service.get(run_id)["status"], "failed")
        with patch("whatiff.interface.load_key", return_value="test-key"), patch("whatiff.interface.SparkPolicy", FakePolicy), patch("whatiff.interface.SparkAdversary", FakeAdversary), patch("whatiff.interface.SamSegmenter", FakeSam):
            self.assertEqual(self.client.post(f"/api/runs/{run_id}/retry").status_code, 202)
            self.wait()
        self.assertEqual(self.service.get(run_id)["status"], "completed")
        self.assertEqual(len(FakePolicy.calls), 2)
        self.assertEqual(FakeAdversary.calls, 1)

    def test_untrusted_origin_rejected(self):
        response = self.client.post("/api/replay", headers={"Origin": "https://unrelated.example"})
        self.assertEqual(response.status_code, 403)

    def test_invalid_ids_and_arbitrary_file_names_are_not_served(self):
        self.assertEqual(self.client.get("/api/runs/not-a-run").status_code, 400)
        run = self.client.post("/api/replay").json()
        self.assertEqual(self.client.get(f"/api/runs/{run['id']}/images/.env").status_code, 400)
        self.assertEqual(self.client.get("/.env").status_code, 404)

    def test_restart_marks_incomplete_runs_retryable(self):
        state = self.service.base("live", "Task", "controlled")
        restarted = InterfaceService(PROJECT, self.service.root)
        self.addCleanup(restarted.executor.shutdown)
        self.assertEqual(restarted.get(state["id"])["status"], "failed")

    def test_corrupted_independent_review_is_rejected(self):
        original = Path.read_text

        def altered(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path.name == "controlled-review.json":
                data = json.loads(text)
                data["criteria"][1]["acceptable_actions"] = ["ABSTAIN"]
                return json.dumps(data)
            return text

        with patch.object(Path, "read_text", altered), self.assertRaises(ValueError):
            self.service.reviewed_fixture()
