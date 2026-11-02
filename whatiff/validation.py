"""Conservative edit verification using pixel evidence and bound curator review."""

import json
from dataclasses import asdict

from .contracts import Validity
from .images import digest, preservation


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def review_template(original, edited, task, proposal, criterion):
    return {
        "original_sha256": digest(original), "edited_sha256": digest(edited),
        "task_sha256": digest(task.encode()), "proposal_sha256": fingerprint({"intervention": proposal.intervention, "description": proposal.description}),
        "criterion_sha256": fingerprint(asdict(criterion)),
        "reviewer": "", "evidence": "", "regions": [],
        "applied_as_intended": None, "unrelated_content_preserved": None,
        "plausible_geometry": None, "criterion_conditions_supported": None,
        "note": "Curator must inspect both images. Set normalized xyxy edit regions and justify all four checks. This is action-selection evidence, not physical validation.",
    }


class EditValidator:
    def __init__(self, reviews=None, segmenter=None, concepts=None):
        self.reviews = reviews or {}
        self.segmenter, self.concepts = segmenter, concepts or {}

    def validate(self, original, edited, task, proposal, criterion):
        review = self.reviews.get(proposal.intervention, {})
        binding = review_template(original, edited, task, proposal, criterion)
        matches = all(review.get(name) == value for name, value in binding.items() if name.endswith("sha256"))
        raw = {"review": review, "review_matches": matches}
        try:
            metrics = preservation(original, edited, review.get("regions", []) if matches else [])
        except Exception as error:
            raw["decode_error"] = type(error).__name__
            return Validity("invalid", "Image decoding or region validation failed.", "image_checks", raw)
        raw["preservation"] = metrics
        # Optional SAM may add evidence, but never gates the core editing path.
        concept = self.concepts.get(proposal.intervention)
        if self.segmenter and concept:
            try:
                raw["grounding"] = {"original": self.segmenter.segment(original, concept),
                                    "edited": self.segmenter.segment(edited, concept)}
            except Exception as error:
                raw["grounding"] = {"status": "unavailable", "error_type": type(error).__name__}
        if metrics["status"] == "invalid":
            return Validity("invalid", metrics["reason"], "image_checks", raw)
        if not matches or not review.get("reviewer", "").strip() or not review.get("evidence", "").strip():
            return Validity("unverified", "Exact-image independent curator review is missing or stale.", "image_checks", raw)
        checks = ("applied_as_intended", "unrelated_content_preserved", "plausible_geometry", "criterion_conditions_supported")
        if any(review.get(check) is False for check in checks):
            return Validity("invalid", review["evidence"], "curator:" + review["reviewer"], raw)
        if metrics["status"] != "consistent" or not all(review.get(check) is True for check in checks):
            return Validity("unverified", "Preservation or semantic review remains incomplete.", "image_checks", raw)
        return Validity("valid", review["evidence"], "curator:" + review["reviewer"], raw)
