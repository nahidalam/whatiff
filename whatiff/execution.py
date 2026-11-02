"""Safe execution projection from existing API audits and workflow checkpoints."""

import json
from pathlib import Path

MODEL_NAMES = {"muse-spark-1.3": "Muse Spark", "muse-image-1.0": "Muse Image", "sam-3.1": "SAM 3.1", "meta-models/muse-spark-open-1026": "Muse Spark Open · Baseten"}
ROLES = {"Adversary": "Scenario planning", "Generator": "Scene generation and grounding", "Policy": "Action selection", "Evaluator": "Deterministic verification"}


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def audited_calls(folder):
    calls = []
    for area in ("agent-api", "policy-api"):
        for path in sorted((Path(folder) / area).glob("*/request.json")):
            request = read_json(path)
            model = (request.get("body") or {}).get("model")
            if model not in MODEL_NAMES:
                continue
            role = "Policy" if area == "policy-api" else "Adversary" if model == "muse-spark-1.3" else "Generator"
            response = read_json(path.parent / "response.json")
            status = "running"
            if (path.parent / "error.json").is_file():
                status = "failed"
            elif response:
                code = response.get("_http_status", 200)
                status = "completed" if 200 <= code < 300 else "failed"
                if model == "meta-models/muse-spark-open-1026":
                    if response.get("error") or not response.get("choices") or response["choices"][0].get("finish_reason") != "stop":
                        status = "failed"
                if model == "muse-spark-1.3":
                    body = response
                    if "_http_body" in response:
                        try:
                            body = json.loads(response["_http_body"])
                        except (ValueError, TypeError):
                            body = {}
                    if body.get("error") or body.get("status") != "completed":
                        status = "failed"
            calls.append({"id": path.parent.name, "role": role, "model": model,
                          "status": status, "mode": request.get("mode", "unknown")})
    return calls


def projection(state, folder, recorded=None):
    replay = state["mode"] == "prepared_example"
    calls = recorded.get("calls", []) if replay and recorded else audited_calls(folder) if not replay else []
    # Only known metadata is surfaced. Never include request bodies or filesystem paths.
    calls = [{key: call[key] for key in ("id", "role", "model", "status", "mode")}
             for call in calls if call.get("model") in MODEL_NAMES and call.get("role") in ROLES]
    terminal = state["status"] in ("completed", "failed")
    workflow = {}
    ref = read_json(Path(folder) / "workflow-reference.json")
    if ref:
        workflow = read_json(Path(folder) / "workflow" / ref["workflow_id"] / "run.json")
    steps = workflow.get("steps", {})
    current = workflow.get("current_step")
    current_role = ("Policy" if str(current).startswith("policy:") else "Generator" if str(current).startswith("generate:")
                    else "Evaluator" if str(current).startswith("validate:") else "Adversary" if current == "adversary" else None)
    pillow = (recorded or {}).get("pillow", False) if replay else read_json(Path(folder) / "generation.json").get("raw", {}).get("settings", {}).get("backend") == "pillow_controlled_occlusion"
    # Policy tools come from this run's audits, including historical replay.
    policy_models = tuple(dict.fromkeys(call["model"] for call in calls if call["role"] == "Policy"))
    if not policy_models and state.get("policy_provider") in ("baseten", "meta"):
        policy_models = ("meta-models/muse-spark-open-1026" if state["policy_provider"] == "baseten" else "muse-spark-1.3",)
    stages = []
    for role, summary in ROLES.items():
        tools = []
        models = ("muse-image-1.0", "sam-3.1") if role == "Generator" else policy_models if role == "Policy" else ("muse-spark-1.3",) if role == "Adversary" else ()
        for model in models:
            matched = [call for call in calls if call["role"] == role and call["model"] == model]
            status = ("running" if any(call["status"] == "running" for call in matched)
                      else "failed" if any(call["status"] == "failed" for call in matched) and (state["status"] == "failed" or not any(call["status"] == "completed" for call in matched))
                      else "completed" if matched else "skipped" if terminal else "waiting")
            tools.append({"name": "Muse Spark 1.3" if role == "Policy" and model == "muse-spark-1.3" else MODEL_NAMES[model], "model": model, "status": status, "calls": len(matched)})
        if role == "Generator" and pillow:
            tools.append({"name": "Pillow", "status": "completed", "calls": 0})
        if role == "Evaluator":
            evaluated = len((state.get("report") or {}).get("results", []))
            status = "completed" if evaluated else "running" if any(key.startswith("validate:") for key in steps) or (Path(folder) / "validity.json").is_file() else "skipped" if terminal else "waiting"
            tools = [{"name": "Deterministic verification", "status": status, "calls": 0}]
        else:
            statuses = [tool["status"] for tool in tools]
            status = "running" if "running" in statuses else "completed" if "completed" in statuses else "failed" if "failed" in statuses else "skipped" if terminal else "waiting"
        if role == "Policy" and not terminal and status == "completed":
            status = "waiting"  # Baseline may be done while edited observations are pending.
        if role == "Generator" and not terminal and workflow and status == "completed":
            planned = len(steps.get("adversary", []))
            generated = sum(key.startswith("generate:") for key in steps)
            if not planned or generated < planned:
                status = "waiting"
        if current_role == role and not terminal:
            status = "running"
        if current_role == role and state["status"] == "failed":
            status = "failed"
        stages.append({"role": role, "summary": summary, "status": status, "tools": tools})
    return {"mode": "saved_replay" if replay else "live", "model_calls_this_execution": 0 if replay else len(calls),
            "recorded_model_calls": len(calls) if replay else 0, "stages": stages}
