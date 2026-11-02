"""Baseten transport exclusively for the Evaluator's isolated policy test."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener
from uuid import uuid4

from .contracts import Action, Decision
from .images import data_url
from .meta import MetaClient, MetaError, NoRedirect
from .policy import POLICY_PROMPT

MODEL = "meta-models/muse-spark-open-1026"
BASE_URL = "https://inference.baseten.co/v1"


class BasetenError(MetaError):
    pass


def load_baseten_key(env_file):
    for line in Path(env_file).read_text().splitlines() if Path(env_file).is_file() else []:
        name, separator, value = line.strip().partition("=")
        if separator and name == "BASETEN_API_KEY":
            key = value.strip().strip("\"'")
            if key and not key.startswith("your_"):
                return key
    raise BasetenError("Baseten: missing BASETEN_API_KEY in .env")


class BasetenClient(MetaClient):
    def __init__(self, key, audit_dir):
        super().__init__(key, audit_dir)

    def request(self, path, payload=None, method="POST"):
        call_id = uuid4().hex
        folder = self.audit_dir / call_id
        folder.mkdir(parents=True)
        started = time.perf_counter()
        self._save(folder / "request.json", {"mode": "live", "provider": "baseten", "at": datetime.now(timezone.utc).isoformat(),
                    "method": method, "endpoint": BASE_URL + path, "body": payload})
        request = Request(BASE_URL + path, data=json.dumps(payload).encode(), method=method,
                          headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
        try:
            with build_opener(NoRedirect()).open(request, timeout=120) as response:
                content = response.read(4 * 1024 * 1024 + 1)
                if len(content) > 4 * 1024 * 1024:
                    raise BasetenError("Baseten: response too large")
                data = json.loads(content)
            self._save(folder / "response.json", data)
            self._save(folder / "metrics.json", {"latency_seconds": round(time.perf_counter() - started, 3), "usage": data.get("usage")})
            return data, {"call_id": call_id, "provider": "baseten", "mode": "live", "request": str(folder / "request.json"), "response": str(folder / "response.json")}
        except Exception as error:
            code = f"Baseten: HTTP {error.code}" if isinstance(error, HTTPError) else "Baseten: network error" if isinstance(error, (URLError, OSError, TimeoutError)) else getattr(error, "code", "Baseten: invalid response")
            self._save(folder / "error.json", {"provider": "baseten", "code": code, "latency_seconds": round(time.perf_counter() - started, 3)})
            raise BasetenError(code, getattr(error, "code", None)) from None


class BasetenSparkPolicy:
    def __init__(self, client):
        self.client = client
        self.settings = {"provider": "baseten", "model": MODEL, "policy_prompt_version": "action-selection-v1", "max_tokens": 6000}

    def decide(self, image, task):
        if not isinstance(task, str) or not task.strip():
            raise ValueError("A task instruction is required.")
        payload = {"model": MODEL, "max_tokens": self.settings["max_tokens"], "response_format": {"type": "json_object"},
                   "messages": [{"role": "system", "content": POLICY_PROMPT}, {"role": "user", "content": [
                       {"type": "text", "text": task}, {"type": "image_url", "image_url": {"url": data_url(image)}}]}]}
        response, audit = self.client.request("/chat/completions", payload)
        try:
            choice = response["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("incomplete")
            answer = json.loads(choice["message"]["content"])
            if not isinstance(answer, dict) or set(answer) != {"action", "reason"} or not isinstance(answer["reason"], str) or not answer["reason"].strip():
                raise ValueError("invalid contract")
            return Decision(Action(answer["action"]), answer["reason"], {"answer": answer, "audit": audit, "settings": self.settings, "usage": response.get("usage")})
        except (KeyError, IndexError, TypeError, ValueError):
            # Keep provider identity visible without exposing its raw payload.
            self.client._save(Path(audit["response"]).parent / "error.json", {"provider": "baseten", "code": "Baseten: invalid policy response"})
            raise BasetenError("Baseten: invalid policy response") from None
