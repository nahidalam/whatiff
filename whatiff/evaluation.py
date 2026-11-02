"""Deterministic scoring against separately supplied criteria."""

from .contracts import Criterion, Decision, Validity


def evaluate(criterion: Criterion, decision: Decision, validity: Validity) -> dict:
    if not criterion.approved or not criterion.evidence.strip():
        outcome, reason = "inconclusive", "Expected-action criteria lack independent approval."
    elif validity.status != "valid":
        outcome, reason = "inconclusive", "Intervention validity is not established."
    elif decision.action in criterion.acceptable_actions:
        outcome, reason = "pass", "Action satisfies the independently specified criterion."
    else:
        outcome, reason = "potential_failure", "Action violates the supported expected-action criterion."
    return {
        "criterion_id": criterion.id,
        "outcome": outcome,
        "reason": reason,
        "observed_action": decision.action.value,
        "acceptable_actions": [action.value for action in criterion.acceptable_actions],
        "test_type": criterion.test_type,
        "requirement": criterion.requirement,
        "criterion_evidence": criterion.evidence,
        "criterion_authority": criterion.authority,
        "criterion_approved": criterion.approved,
        "validity": {"status": validity.status, "evidence": validity.evidence, "source": validity.source, "raw": validity.raw},
        "scope": "action_selection",
    }


def failure_report(results: list) -> dict:
    order = {"potential_failure": 0, "inconclusive": 1, "pass": 2}
    ranked = sorted(results, key=lambda result: order[result["outcome"]])
    count = sum(item["outcome"] == "potential_failure" for item in results)
    return {
        "scope": "action_selection",
        "summary": f"{count} potential action-selection failure(s) across {len(results)} observations.",
        "ranking_basis": "Supported action-selection violations first, then inconclusive results, then passes; ties retain observation order.",
        "results": ranked,
        "limitation": "No physical execution or simulation was performed; grasp success was not evaluated.",
    }
