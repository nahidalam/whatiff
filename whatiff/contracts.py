"""Provider-independent contracts. Ground truth is never an agent output."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Protocol, Tuple


class Action(str, Enum):
    PICK_TARGET = "PICK_TARGET"
    REPOSITION = "REPOSITION"
    ABSTAIN = "ABSTAIN"


ACTION_MEANINGS = {
    Action.PICK_TARGET.value: "Select the uniquely identifiable target for pickup.",
    Action.REPOSITION.value: "Change viewpoint or approach before attempting pickup.",
    Action.ABSTAIN.value: "Decline because the target cannot be identified adequately.",
}


@dataclass(frozen=True)
class Criterion:
    id: str
    intervention: str
    test_type: str
    acceptable_actions: Tuple[Action, ...]
    requirement: str
    evidence: str
    authority: str
    approved: bool = False

    def __post_init__(self):
        if not all((self.id, self.intervention, self.requirement, self.authority)):
            raise ValueError("Criteria need identity, requirement, and independent authority.")
        if self.test_type not in ("baseline", "invariance", "adaptation"):
            raise ValueError("Unknown test type.")
        if not self.acceptable_actions or any(not isinstance(a, Action) for a in self.acceptable_actions):
            raise ValueError("Criteria must contain valid acceptable actions.")
        if self.approved and not self.evidence.strip():
            raise ValueError("Approved criteria need supporting evidence.")


@dataclass(frozen=True)
class Proposal:
    intervention: str
    description: str
    rationale: str
    predicted_action: Action
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    raw: Dict[str, Any]

    def __post_init__(self):
        if not isinstance(self.action, Action):
            raise ValueError("Policy returned an unknown action.")


@dataclass(frozen=True)
class GeneratedObservation:
    image: bytes
    media_type: str
    changes: str
    raw: Dict[str, Any]


@dataclass(frozen=True)
class Validity:
    status: str
    evidence: str
    source: str
    raw: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.status not in ("valid", "invalid", "unverified"):
            raise ValueError("Unknown intervention validity.")
        if self.status == "valid" and not (self.evidence.strip() and self.source.strip()):
            raise ValueError("Valid edits need evidence and a source.")


class Policy(Protocol):
    def decide(self, image: bytes, task: str) -> Decision: ...


class Adversary(Protocol):
    def propose(self, image: bytes, task: str) -> Tuple[Proposal, ...]: ...


class Generator(Protocol):
    def generate(self, image: bytes, task: str, proposal: Proposal) -> GeneratedObservation: ...


class Validator(Protocol):
    def validate(self, original: bytes, edited: bytes, task: str,
                 proposal: Proposal, criterion: Criterion) -> Validity: ...


class Segmenter(Protocol):
    """Optional validation enhancement; not a workflow dependency."""
    def segment(self, image: bytes, concept: str) -> Dict[str, Any]: ...
