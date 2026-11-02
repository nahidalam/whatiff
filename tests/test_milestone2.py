"""Offline tests: synthetic pixels + mocked HTTP, never live model validation."""

import base64
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from whatiff.contracts import Action, Criterion, Decision, Proposal
from whatiff.evaluation import evaluate
from whatiff.images import data_url, decode, digest, preservation
from whatiff.live import probe, review_run
from whatiff.meta import (HTTPTransport, MetaClient, MetaError, MuseImageGenerator,
                          SamSegmenter, SparkAdversary, load_key, parse_sam_boxes, parse_stream)
from whatiff.store import RunStore
from whatiff.validation import EditValidator, review_template
from whatiff.workflow import Workflow


def png(edited=False, background="gray", size=(128, 96)):
    image = Image.new("RGB", size, background)
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 40, 40), fill="red" if edited else "white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def expected():
    return tuple(Criterion(name, name, kind, (action,), "Independent synthetic-test condition", "Test-only fixture evidence", "test_curator", True)
                 for name, kind, action in (("baseline", "baseline", Action.PICK_TARGET), ("occlusion", "invariance", Action.PICK_TARGET),
                                          ("distractor", "invariance", Action.PICK_TARGET), ("relocation", "adaptation", Action.REPOSITION)))


def proposal():
    return Proposal("occlusion", "Partially cover the mug.", "May reduce visibility.", Action.ABSTAIN)


def spark_response():
    data = {"proposals": [{"intervention": name, "description": "Edit " + name, "rationale": "Visibility hypothesis", "predicted_action": "ABSTAIN"}
                          for name in ("occlusion", "distractor", "relocation")]}
    return {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(data)}]}]}


def image_response(image=None):
    return {"data": [{"b64_json": base64.b64encode(image or png(True)).decode()}], "output_format": "png", "usage": {"test": True}}


class MockTransport:
    def __init__(self):
        self.calls = []
        self.responses = []

    def __call__(self, method, path, body):
        self.calls.append((method, path, body))
        if self.responses:
            result = self.responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result
        if path == "/models":
            return {"data": [{"id": "muse-spark-1.3"}, {"id": "muse-image-1.0"}]}
        return spark_response() if path == "/responses" else image_response()


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.transport = MockTransport()
        self.client = MetaClient("test-secret", Path(self.directory.name) / "api", self.transport, mode="mock")

    def test_injected_transport_requires_mock_label(self):
        with self.assertRaises(ValueError):
            MetaClient("key", self.directory.name, self.transport)

    def test_spark_image_input_exactly_three_hypotheses_and_audit(self):
        proposals = SparkAdversary(self.client).propose(png(), "Pick the mug.")
        self.assertEqual(len(proposals), 3)
        self.assertTrue(all(p.predicted_action == Action.ABSTAIN for p in proposals))
        request = self.transport.calls[0][2]
        self.assertFalse(request["store"])
        self.assertEqual(request["reasoning"], {"effort": "low"})
        self.assertEqual(request["max_output_tokens"], 6000)
        self.assertEqual(request["input"][0]["content"][1]["image_url"], data_url(png()))
        self.assertNotIn("acceptable_actions", json.dumps(request))
        self.assertNotIn("criterion_authority", json.dumps(request))
        record = proposals[0].raw["audit"]
        self.assertEqual(record["mode"], "mock")
        self.assertTrue(Path(record["request"]).is_file())
        self.assertEqual(json.loads(Path(record["response"]).read_text()), spark_response())

    def test_spark_rejects_unknown_action_duplicate_types_and_extra_fields(self):
        for mutation in ("action", "duplicate", "extra", "count"):
            response = spark_response()
            data = json.loads(response["output"][0]["content"][0]["text"])
            if mutation == "action":
                data["proposals"][0]["predicted_action"] = "FLY"
            elif mutation == "duplicate":
                data["proposals"][0]["intervention"] = "distractor"
            elif mutation == "extra":
                data["proposals"][0]["acceptable_actions"] = ["ABSTAIN"]
            else:
                data["proposals"].pop()
            response["output"][0]["content"][0]["text"] = json.dumps(data)
            self.transport.responses.append(response)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                SparkAdversary(self.client).propose(png(), "Task")

    def test_incomplete_spark_output_is_rejected_and_saved(self):
        self.transport.responses.append({"status": "incomplete", "output_text": "{}"})
        with self.assertRaises(MetaError):
            SparkAdversary(self.client).propose(png(), "Task")
        self.assertEqual(len(list(Path(self.directory.name).glob("api/*/response.json"))), 1)

    def test_image_reference_request_and_decoded_output(self):
        generated = MuseImageGenerator(self.client).generate(png(), "Task", proposal())
        self.assertEqual(generated.image, png(True))
        self.assertEqual(generated.media_type, "image/png")
        method, path, body = self.transport.calls[0]
        self.assertEqual((method, path), ("POST", "/images/edits"))
        self.assertEqual(body["images"], [{"image_url": data_url(png())}])
        self.assertFalse(body["tool_enablement"]["enable_web_search"])
        self.assertNotIn("mask", body)
        self.assertIn("all unrelated objects", body["prompt"])
        self.assertEqual(generated.raw["original_sha256"], digest(png()))
        self.assertTrue(Path(generated.raw["audit"]["response"]).exists())

    def test_image_rejects_bad_base64_nonimage_and_missing_output(self):
        for response in ({"data": [{"b64_json": "!!!"}]}, image_response(b"not an image"), {"data": []}):
            self.transport.responses.append(response)
            with self.subTest(response=str(response)[:50]), self.assertRaises(Exception):
                MuseImageGenerator(self.client).generate(png(), "Task", proposal())

    def test_invalid_input_makes_no_api_request(self):
        with self.assertRaises(Exception):
            MuseImageGenerator(self.client).generate(b"invalid", "Task", proposal())
        self.assertEqual(self.transport.calls, [])

    def test_resized_output_is_saved_for_invalidity_review(self):
        self.transport.responses.append(image_response(png(True, size=(256, 192))))
        observation = MuseImageGenerator(self.client).generate(png(), "Task", proposal())
        validity = EditValidator().validate(png(), observation.image, "Task", proposal(), expected()[1])
        self.assertEqual(validity.status, "invalid")
        self.assertEqual(evaluate(expected()[1], Decision(Action.ABSTAIN, "test", {}), validity)["outcome"], "inconclusive")

    def test_model_preflight_keeps_sam_optional(self):
        result = self.client.capabilities(sam=True)
        self.assertTrue(all(result["required"].values()))
        self.assertFalse(result["sam"])

    def test_audit_redacts_credential_echoes(self):
        self.transport.responses.append({"echo": "test-secret"})
        _, audit = self.client.request("/models", method="GET")
        self.assertNotIn("test-secret", Path(audit["response"]).read_text())

    def test_http_failure_body_saved_and_classified(self):
        self.transport.responses.append({"_http_status": 403, "_http_content_type": "application/json", "_http_body": '{"error":"test-secret"}'})
        with self.assertRaises(MetaError) as caught:
            self.client.request("/models", method="GET")
        self.assertEqual(caught.exception.status, 403)
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn("test-secret", "".join(p.read_text() for p in Path(self.directory.name).glob("api/*/*.json")))

    def test_malformed_json_wire_response_retained(self):
        self.transport.responses.append({"_http_status": 200, "_http_content_type": "application/json", "_http_body": "not JSON"})
        with self.assertRaises(ValueError):
            self.client.request("/models", method="GET")
        response = next(Path(self.directory.name).glob("api/*/response.json"))
        self.assertIn("not JSON", response.read_text())

    def test_missing_key_and_ignored_file_loading(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(MetaError):
                load_key()
            path = Path(self.directory.name) / ".env"
            path.write_text('MODEL_API_KEY="test-secret"\n')
            self.assertEqual(load_key(path), "test-secret")

    def test_network_errors_do_not_expose_exception_text(self):
        from urllib.error import URLError
        with patch("whatiff.meta.build_opener") as opener:
            opener.return_value.open.side_effect = URLError("test-secret")
            with self.assertRaises(MetaError) as caught:
                HTTPTransport("test-secret")("GET", "/models", None)
            self.assertEqual(str(caught.exception), "network_error")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.before, self.after = png(), png(True)
        self.criterion, self.proposal = expected()[1], proposal()
        self.review = review_template(self.before, self.after, "Task", self.proposal, self.criterion)
        self.review.update(reviewer="test_curator", evidence="Synthetic test rectangle changed; outside unchanged.", regions=[[0.1, 0.1, 0.4, 0.5]],
                           applied_as_intended=True, unrelated_content_preserved=True, plausible_geometry=True, criterion_conditions_supported=True)

    def validate(self, review=None, after=None, segmenter=None):
        return EditValidator({"occlusion": self.review if review is None else review}, segmenter, {"occlusion": "mug"}).validate(
            self.before, self.after if after is None else after, "Task", self.proposal, self.criterion)

    def test_outside_region_preserved_with_bound_review(self):
        validity = self.validate()
        self.assertEqual(validity.status, "valid")
        self.assertEqual(validity.raw["preservation"]["outside_changed_fraction"], 0)

    def test_visual_plausibility_without_review_is_unverified(self):
        self.assertEqual(self.validate({}).status, "unverified")
        self.assertEqual(preservation(self.before, self.after, [])["status"], "unverified")

    def test_stale_review_never_establishes_validity(self):
        for field in ("original_sha256", "edited_sha256", "task_sha256", "proposal_sha256", "criterion_sha256"):
            review = {**self.review, field: "stale"}
            with self.subTest(field=field):
                self.assertEqual(self.validate(review).status, "unverified")

    def test_unrelated_background_change_invalidates_edit(self):
        after = png(True, background="blue")
        review = {**self.review, "edited_sha256": digest(after)}
        validity = self.validate(review, after)
        self.assertEqual(validity.status, "invalid")
        self.assertGreater(validity.raw["preservation"]["outside_changed_fraction"], 0.02)

    def test_small_global_drift_is_not_silently_accepted(self):
        after = png(True, background=(140, 140, 140))
        self.assertEqual(self.validate({**self.review, "edited_sha256": digest(after)}, after).status, "invalid")

    def test_broad_edit_region_cannot_hide_background_changes(self):
        self.assertEqual(self.validate({**self.review, "regions": [[0, 0, 1, 1]]}).status, "unverified")

    def test_curator_rejection_and_incomplete_review(self):
        self.assertEqual(self.validate({**self.review, "applied_as_intended": False}).status, "invalid")
        self.assertEqual(self.validate({**self.review, "plausible_geometry": None}).status, "unverified")

    def test_bad_bounds_rejected(self):
        self.assertEqual(self.validate({**self.review, "regions": [[-1, 0, 1, 1]]}).status, "invalid")

    def test_sam_outage_does_not_block_valid_review(self):
        class OfflineSam:
            def segment(self, image, concept):
                raise MetaError("unavailable")
        validity = self.validate(segmenter=OfflineSam())
        self.assertEqual(validity.status, "valid")
        self.assertEqual(validity.raw["grounding"]["status"], "unavailable")

    def test_unchanged_pixels_invalid_even_with_different_encoding(self):
        image, _ = decode(self.before)
        output = BytesIO()
        image.save(output, format="PNG", compress_level=0)
        self.assertEqual(self.validate(after=output.getvalue()).status, "invalid")


class SegmentationTests(unittest.TestCase):
    def test_special_token_boxes_and_empty_matches(self):
        text = "<0f>2<|box;x1=1;y1=2;x2=4;y2=6;w=128;h=96|><|mask;x=0;y=0;data=5,4,~abc|>"
        self.assertEqual(parse_sam_boxes(text)[0]["object_id"], 2)
        self.assertEqual(parse_sam_boxes(""), [])
        with self.assertRaises(ValueError):
            parse_sam_boxes("made up coordinates")

    def test_sse_requires_completed_event(self):
        self.assertEqual(parse_stream('data: {"type":"response.completed"}\n\n')["output_text"], "")
        with self.assertRaises(MetaError):
            parse_stream('data: {"type":"response.output_text.delta","delta":"partial"}\n\n')

    def test_sam_adapter_stream_without_image_edit_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            transport = MockTransport()
            transport.responses.append({"status": "completed", "output_text": ""})
            result = SamSegmenter(MetaClient("test-key", directory, transport, "mock")).segment(png(), "white mug")
            self.assertEqual(result["boxes"], [])
            self.assertTrue(transport.calls[0][2]["stream"])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.transport = MockTransport()
        self.client = MetaClient("test-key", Path(self.directory.name) / "api", self.transport, "mock")

    def test_real_adapters_in_workflow_with_mock_http_and_policy(self):
        class TestPolicy:
            def decide(self, image, task):
                return Decision(Action.PICK_TARGET, "Test double", {"test": True})
        store = RunStore(Path(self.directory.name) / "runs")
        workflow = Workflow(store, TestPolicy(), SparkAdversary(self.client), MuseImageGenerator(self.client), EditValidator(), {"mode": "mock"})
        created = workflow.create(png(), "image/png", "Task", expected(), "test")
        run = workflow.resume(created["id"])
        self.assertEqual(run["status"], "completed")
        self.assertEqual(sum(item["outcome"] == "inconclusive" for item in run["report"]["results"]), 3)
        self.assertEqual(sum(call[1] == "/images/edits" for call in self.transport.calls), 3)
        self.assertTrue(all(call[2]["images"][0]["image_url"] == data_url(png()) for call in self.transport.calls if call[1] == "/images/edits"))
        self.assertTrue(Path(run["steps"]["adversary"][0]["raw"]["audit"]["response"]).exists())

    def test_probe_resume_saves_artifacts_and_does_not_repeat_successful_edits(self):
        store = RunStore(Path(self.directory.name) / "probes")
        run_id = "a" * 32
        self.transport.responses = [{"data": [{"id": "muse-spark-1.3"}, {"id": "muse-image-1.0"}]}, spark_response(), image_response(), MetaError("network_error", retryable=True)]
        run = probe(self.client, store, run_id, png(), "Task", expected())
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["mode"], "mock_adapter_probe")
        self.assertEqual(len(run["observations"]), 1)
        run = probe(self.client, store, run_id, png(), "Task", expected())
        self.assertEqual(run["status"], "completed")
        self.assertEqual(len(run["observations"]), 3)
        self.assertEqual(sum(call[1] == "/responses" for call in self.transport.calls), 1)
        reviewed = review_run(store.folder(run_id), {})
        self.assertTrue(all(item["validity"]["status"] == "unverified" for item in reviewed["observations"]))
        self.assertTrue(all(item["evaluation_outcome"] == "inconclusive" for item in reviewed["observations"]))

    def test_missing_required_model_blocks_generation_but_sam_does_not(self):
        store = RunStore(Path(self.directory.name) / "probes")
        self.transport.responses.append({"data": [{"id": "muse-spark-1.3"}]})
        result = probe(self.client, store, "b" * 32, png(), "Task", expected())
        self.assertEqual(result["error"]["code"], "required_model_unavailable")
        self.assertEqual(len(self.transport.calls), 1)

    def test_probe_cannot_resume_mock_evidence_as_live(self):
        store = RunStore(Path(self.directory.name) / "probes")
        run_id = "c" * 32
        probe(self.client, store, run_id, png(), "Task", expected())
        self.client.mode = "live"
        before = len(self.transport.calls)
        with self.assertRaises(ValueError):
            probe(self.client, store, run_id, png(), "Task", expected())
        self.assertEqual(len(self.transport.calls), before)


if __name__ == "__main__":
    unittest.main()
