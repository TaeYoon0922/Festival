"""Immutable contracts for bounded clarification decisions.

Candidates are supplied by deterministic system components.  The optional
classifier may choose only among their IDs; labels and provenance never come
from model output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


MAX_CANDIDATES = 12
MAX_LISTED_CANDIDATES = 3


class ClarificationState(str, Enum):
    RESOLVED = "resolved"
    CLARIFY = "clarify"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ClarificationCandidate:
    id: str
    label: str
    semantic_type: str
    provenance: str
    value: str | None = None

    def __post_init__(self) -> None:
        for name in ("id", "label", "semantic_type", "provenance"):
            value = str(getattr(self, name) or "").strip()
            if not value:
                raise ValueError(f"clarification candidate {name} must not be empty")
            object.__setattr__(self, name, value)
        if len(self.id) > 80 or len(self.label) > 160:
            raise ValueError("clarification candidate id or label is too long")
        if self.value is not None:
            object.__setattr__(self, "value", str(self.value))

    def to_public_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "semantic_type": self.semantic_type,
            "provenance": self.provenance,
        }

    def to_classifier_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label}


@dataclass(frozen=True)
class ClarificationRequest:
    question: str
    candidates: tuple[ClarificationCandidate, ...] = ()
    reason: str = "bounded_candidates"
    target_slot: str | None = None
    single_candidate_safe: bool = False
    classifier_resolution_safe: bool = False
    fallback_state: ClarificationState = ClarificationState.INSUFFICIENT_EVIDENCE
    truncated: bool = False

    def __post_init__(self) -> None:
        question = str(self.question or "").strip()
        if not question:
            raise ValueError("clarification request question must not be empty")
        object.__setattr__(self, "question", question)
        candidates = tuple(self.candidates)
        if len(candidates) > MAX_CANDIDATES:
            raise ValueError(f"clarification requests support at most {MAX_CANDIDATES} candidates")
        identifiers = [candidate.id for candidate in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("clarification candidate ids must be unique")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "fallback_state", ClarificationState(self.fallback_state))
        if self.fallback_state not in {
            ClarificationState.INSUFFICIENT_EVIDENCE,
            ClarificationState.UNSUPPORTED,
        }:
            raise ValueError(
                "zero or one candidate fallback must be insufficient or unsupported"
            )


@dataclass(frozen=True)
class ClarificationDecision:
    state: ClarificationState
    reason: str
    candidates: tuple[ClarificationCandidate, ...] = ()
    selected_candidate_id: str | None = None
    classifier_status: str = "not_called"
    truncated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", ClarificationState(self.state))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.state is ClarificationState.RESOLVED:
            identifiers = {candidate.id for candidate in self.candidates}
            if self.selected_candidate_id not in identifiers:
                raise ValueError("resolved clarification must select a supplied candidate")
        elif self.selected_candidate_id is not None:
            raise ValueError("only resolved clarification may select a candidate")

    @property
    def selected_candidate(self) -> ClarificationCandidate | None:
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.id == self.selected_candidate_id
            ),
            None,
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "candidate_count": len(self.candidates),
            "candidates": [candidate.to_public_dict() for candidate in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "classifier_status": self.classifier_status,
            "truncated": self.truncated,
        }


def clarification_text(decision: ClarificationDecision) -> str:
    """Render only deterministic candidate labels, with a bounded list."""

    candidates = decision.candidates
    if decision.state is not ClarificationState.CLARIFY:
        raise ValueError("clarification text requires a clarify decision")
    if decision.truncated or len(candidates) > MAX_LISTED_CANDIDATES:
        return (
            "여러 가능한 해석이 확인되었습니다. 원하시는 지표, 공시일 또는 "
            "보고서 기준을 구체적으로 알려주세요."
        )
    labels = [candidate.label for candidate in candidates]
    if len(labels) == 2:
        return (
            f"{_with_object_particle(labels[0])} 말씀하시는 건가요, 아니면 "
            f"{_with_object_particle(labels[1])} 말씀하시는 건가요?"
        )
    if len(labels) == 3:
        return f"말씀하신 항목이 {', '.join(labels)} 중 어느 것을 의미하는지 알려주세요."
    return "확인하려는 항목을 조금 더 구체적으로 알려주세요."


def _with_object_particle(label: str) -> str:
    last = label.rstrip()[-1]
    codepoint = ord(last) - 0xAC00
    particle = "을" if 0 <= codepoint < 11172 and codepoint % 28 else "를"
    return f"{label}{particle}"


__all__ = [
    "MAX_CANDIDATES",
    "MAX_LISTED_CANDIDATES",
    "ClarificationCandidate",
    "ClarificationDecision",
    "ClarificationRequest",
    "ClarificationState",
    "clarification_text",
]
