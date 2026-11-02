"""Small Meta HTTP adapters; transport injection keeps tests offline."""

import base64
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPRedirectHandler
from uuid import uuid4

from .contracts import Action, GeneratedObservation, Proposal
from .images import data_url, decode, digest
from . import prompts


class MetaError(RuntimeError):
    def __init__(self, code, status=None, retryable=False):
        super().__init__(code)
        self.code, self.status, self.retryable = code, status, retryable


def load_key(env_file=None):
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key and env_file and Path(env_file).is_file():
        for line in Path(env_file).read_text().splitlines():
            if line.strip().startswith("MODEL_API_KEY="):
                key = line.strip().split("=", 1)[1].strip().strip("\"'")
    if not key or key.startswith("your_"):
        raise MetaError("missing_model_api_key")
    return key


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class HTTPTransport:
    def __init__(self, key, timeout=120):
        self.key, self.timeout = key, timeout

    def __call__(self, method, path, body):
        request = Request("https://api.meta.ai/v1" + path,
                          data=None if body is None else json.dumps(body, allow_nan=False).encode(),
                          headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"}, method=method)
        try:
            with build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                content = response.read(64 * 1024 * 1024 + 1)
                if len(content) > 64 * 1024 * 1024:
                    raise MetaError("response_too_large")
                return {"_http_body": content.decode("utf-8"), "_http_status": response.status,
                        "_http_content_type": response.headers.get("Content-Type", "")}
        except HTTPError as error:
            return {"_http_body": error.read(64 * 1024 * 1024).decode("utf-8", errors="replace"),
                    "_http_status": error.code, "_http_content_type": error.headers.get("Content-Type", "")}
        except (URLError, TimeoutError, OSError):
            raise MetaError("network_error", retryable=True) from None


def parse_stream(text):
    events, parts, complete = [], [], False
    for frame in re.split(r"\r?\n\r?\n", text):
        data = "\n".join(line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:"))
        if not data or data == "[DONE]":
            continue
        event = json.loads(data)
        events.append(event)
        if event.get("type") in ("error", "response.failed", "response.incomplete"):
            raise MetaError("segmentation_stream_failed")
        if event.get("type") == "response.output_text.delta":
            parts.append(event["delta"])
        if event.get("type") == "response.completed":
            complete = True
    if not complete:
        raise MetaError("segmentation_stream_incomplete")
    return {"events": events, "output_text": "".join(parts), "status": "completed"}


class MetaClient:
    def __init__(self, key, audit_dir, transport=None, mode="live"):
        if mode not in ("live", "mock"):
            raise ValueError("Unknown adapter mode.")
        if transport is not None and mode != "mock":
            raise ValueError("Injected transport must be explicitly labeled mock.")
        self.key, self.audit_dir, self.mode = key, Path(audit_dir), mode
        self.transport = transport or HTTPTransport(key)

    def _save(self, path, value):
        # Requests omit auth headers; scrub a provider echo of the credential as well.
        text = json.dumps(value, indent=2, allow_nan=False)
        if self.key:
            text = text.replace(self.key, "[REDACTED]")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)

    def request(self, path, payload=None, method="POST"):
        call_id = uuid4().hex
        folder = self.audit_dir / call_id
        folder.mkdir(parents=True)
        self._save(folder / "request.json", {"mode": self.mode, "at": datetime.now(timezone.utc).isoformat(),
                   "method": method, "endpoint": "https://api.meta.ai/v1" + path, "body": payload})
        try:
            response = self.transport(method, path, payload)
            self._save(folder / "response.json", response)
            if "_http_body" in response:
                status = response["_http_status"]
                if not 200 <= status < 300:
                    raise MetaError("http_error", status, status in (408, 429, 500, 502, 503, 504))
                response = (parse_stream(response["_http_body"]) if "text/event-stream" in response["_http_content_type"]
                            else json.loads(response["_http_body"]))
            return response, {"call_id": call_id, "mode": self.mode, "request": str(folder / "request.json"), "response": str(folder / "response.json")}
        except Exception as error:
            self._save(folder / "error.json", {"type": type(error).__name__, "code": getattr(error, "code", "provider_error"),
                       "status": getattr(error, "status", None), "retryable": getattr(error, "retryable", False)})
            raise

    def capabilities(self, sam=False):
        response, audit = self.request("/models", method="GET")
        available = {model["id"] for model in response["data"]}
        required = ("muse-spark-1.3", "muse-image-1.0")
        return {"required": {model: model in available for model in required}, "sam": "sam-3.1" in available,
                "sam_requested": sam, "audit": audit,
                "limitation": "Catalog presence is not proof that inference succeeds."}


def output_text(response):
    if response.get("status") != "completed":
        raise MetaError("incomplete_model_response")
    if "output_text" in response:
        return response["output_text"]
    return "".join(part["text"] for item in response.get("output", []) if item.get("type") == "message"
                   for part in item.get("content", []) if part.get("type") == "output_text")


class SparkAdversary:
    def __init__(self, client, interventions=("occlusion", "distractor", "relocation")):
        if len(interventions) != 3 or len(set(interventions)) != 3 or not all(isinstance(i, str) and i.strip() for i in interventions):
            raise ValueError("Configure three distinct intervention types.")
        self.client, self.interventions = client, tuple(interventions)
        self.settings = {"model": "muse-spark-1.3", "prompt_version": prompts.VERSION,
                         "max_output_tokens": 6000, "reasoning_effort": "low"}

    def propose(self, image, task):
        payload = {"model": self.settings["model"], "store": False, "max_output_tokens": self.settings["max_output_tokens"],
                   "reasoning": {"effort": self.settings["reasoning_effort"]},
                   "instructions": prompts.ADVERSARY,
                   "input": [{"role": "user", "content": [
                       {"type": "input_text", "text": json.dumps({"task": task, "intervention_types": self.interventions})},
                       {"type": "input_image", "image_url": data_url(image)}]}]}
        response, audit = self.client.request("/responses", payload)
        data = json.loads(output_text(response))
        items = data.get("proposals")
        if not isinstance(items, list) or len(items) != 3:
            raise ValueError("Spark must return three proposals.")
        proposals = []
        for item in items:
            if set(item) != {"intervention", "description", "rationale", "predicted_action"}:
                raise ValueError("Unexpected proposal fields.")
            if any(not isinstance(item[name], str) or not item[name].strip() for name in item):
                raise ValueError("Proposal fields must be nonempty strings.")
            proposals.append(Proposal(item["intervention"], item["description"], item["rationale"], Action(item["predicted_action"]),
                                      {"audit": audit, "settings": self.settings}))
        if {p.intervention for p in proposals} != set(self.interventions):
            raise ValueError("Spark returned duplicate or unsupported interventions.")
        return tuple(proposals)


class MuseImageGenerator:
    def __init__(self, client):
        self.client = client
        self.settings = {"model": "muse-image-1.0", "prompt_version": prompts.VERSION, "output_format": "png",
                         "n": 1, "reasoning_strength": "low"}

    def generate(self, image, task, proposal):
        before, _ = decode(image)
        payload = {"model": self.settings["model"], "prompt": prompts.EDIT.format(task=task, description=proposal.description),
                   "images": [{"image_url": data_url(image)}], "n": 1, "output_format": "png", "response_format": "b64_json",
                   "size": f"{before.width}x{before.height}", "reasoning_strength": "low",
                   "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}}
        response, audit = self.client.request("/images/edits", payload)
        items = response.get("data")
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0].get("b64_json"), str):
            raise ValueError("Image edit must return one base64 image.")
        encoded = items[0]["b64_json"]
        if len(encoded) > 28 * 1024 * 1024:
            raise ValueError("Encoded output exceeds image limit.")
        content = base64.b64decode(encoded, validate=True)
        after, media_type = decode(content)
        raw = {"audit": audit, "settings": self.settings, "original_sha256": digest(image),
               "original_size": list(before.size), "edited_size": list(after.size),
               "limitation": "Reference editing has no mask constraint; dimensions and unrelated content require validation."}
        # Preserve decodable rejected edits as artifacts; the validator marks them invalid.
        return GeneratedObservation(content, media_type, "Requested edit (not verified): " + proposal.description, raw)


class SamSegmenter:
    """Optional grounding adapter; no automatic authority over edit validity."""
    def __init__(self, client):
        self.client = client

    def segment(self, image, concept):
        if not concept.strip():
            raise ValueError("A concrete object concept is required.")
        payload = {"model": "sam-3.1", "stream": True,
                   "input": [{"role": "user", "content": [{"type": "input_text", "text": concept},
                                                              {"type": "input_image", "image_url": data_url(image)}]}]}
        response, audit = self.client.request("/responses", payload)
        text = output_text(response)
        return {"model": "sam-3.1", "concept": concept, "boxes": parse_sam_boxes(text),
                "mask_tokens": text, "audit": audit,
                "limitation": "Masks retained as raw SAM tokens; boxes are grounding evidence, not verified object identity."}


def parse_sam_boxes(text):
    if not text.strip():
        return []
    if not text.startswith("<0f>") or "\n" in text.strip():
        raise ValueError("Expected one still-image SAM frame.")
    records = re.findall(r"(\d+)<\|box;([^|]+)\|><\|mask;(.*?)\|>", text)
    if not records or "<0f>" + ",".join(f"{i}<|box;{b}|><|mask;{m}|>" for i, b, m in records) != text.strip():
        raise ValueError("Malformed SAM geometry.")
    boxes = []
    for object_id, box, mask in records:
        fields = dict(field.split("=", 1) for field in box.split(";"))
        values = {name: int(fields[name]) for name in ("x1", "y1", "x2", "y2", "w", "h")}
        if not (0 <= values["x1"] < values["x2"] <= values["w"] and 0 <= values["y1"] < values["y2"] <= values["h"]):
            raise ValueError("Invalid SAM box bounds.")
        boxes.append({"object_id": int(object_id), **values})
    return boxes
