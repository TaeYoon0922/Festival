"""Immutable contracts for bounded clarification decisions.

Candidates are supplied by deterministic system components.  The optional
classifier may choose only among their IDs; labels and provenance never come
from model output.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


MAX_CANDIDATES = 12
MAX_LISTED_CANDIDATES = 3

#: A candidate that is one filed event rather than one reading of a word.  The
#: two are asked about differently: which of several metrics was meant is a
#: question about the question, while which of several filings was meant is a
#: question about the corpus, and only the second can name the 공시일 that tells
#: them apart.
EVENT_INSTANCE = "event_instance"
#: The asker named a holder but not one of that holder's filings.  Kept apart
#: from EVENT_INSTANCE because the sentence differs: those candidates are
#: contracts, these are reports of one continuing position.
HOLDING_REPORT_INSTANCE = "holding_report_instance"


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
    #: The one filing that proves this candidate exists, when the provider
    #: could bind it.  Carried privately: it is what lets the public answer
    #: cite the ambiguity it is reporting, and it never reaches ``label`` or
    #: any public dict, because a served ``doc_id`` is evidence identity while
    #: a label is something a person reads aloud.  Both halves or neither --
    #: half an identity cannot be aligned against served evidence.
    source_doc_id: str | None = None
    source_chunk_id: str | None = None

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
        for name in ("source_doc_id", "source_chunk_id"):
            raw = getattr(self, name)
            object.__setattr__(self, name, str(raw).strip() if raw else None)
        if bool(self.source_doc_id) != bool(self.source_chunk_id):
            raise ValueError(
                "clarification candidate source needs both doc_id and chunk_id"
            )

    @property
    def source(self) -> tuple[str, str] | None:
        """The ``(chunk_id, doc_id)`` identity this candidate was proven by."""

        if self.source_doc_id and self.source_chunk_id:
            return (self.source_chunk_id, self.source_doc_id)
        return None

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
    #: Whether the candidate list is itself the answer and must be shown whole.
    #: The classifier narrows candidates when they are readings of one question
    #: -- 주식수 against 비율 -- and narrowing there is its job.  When the
    #: candidates are the filings a corpus holds, narrowing hides filings the
    #: asker is being invited to choose between, on no signal from a question
    #: that named none of them.
    preserve_candidates: bool = False

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


def clarification_text(
    decision: ClarificationDecision,
    *,
    citation_markers: Mapping[str, str] | None = None,
) -> str:
    """Render only deterministic candidate labels, with a bounded list.

    ``citation_markers`` maps candidate id to the marker the served evidence
    row for that candidate's filing carries.  Supplying it changes nothing
    about what is claimed: the markers point at the filings that prove two
    distinguishable contracts were disclosed, which is the only assertion this
    text makes.  A partial map is ignored, so a candidate the caller could not
    bind never leaves a marker standing next to one that is real.
    """

    candidates = decision.candidates
    if decision.state is not ClarificationState.CLARIFY:
        raise ValueError("clarification text requires a clarify decision")
    if candidates and all(
        candidate.semantic_type == HOLDING_REPORT_INSTANCE for candidate in candidates
    ):
        # Listed in full, above the guard below, because these labels are dates
        # rather than prose: the limit exists so an answer does not become a
        # wall of sentences, and eight dates is not that. The holder is already
        # right and the position is one timeline; what is missing is which
        # filing of it, so naming them is what lets the asker say back
        # something this system can act on.
        listed = "\n".join(f"- {candidate.label}" for candidate in candidates)
        return (
            "해당 보유자의 보고서가 여러 건 있습니다. 질문에 어느 보고서인지가 "
            "없어 하나를 고르지 않았습니다.\n"
            f"{listed}\n"
            "어느 보고서 기준으로 확인할지 알려주세요."
        )
    if decision.truncated or len(candidates) > MAX_LISTED_CANDIDATES:
        return (
            "여러 가능한 해석이 확인되었습니다. 원하시는 지표, 공시일 또는 "
            "보고서 기준을 구체적으로 알려주세요."
        )
    labels = [candidate.label for candidate in candidates]
    if all(candidate.semantic_type == EVENT_INSTANCE for candidate in candidates):
        # Each label is one filing, so naming the dimension is what makes the
        # question answerable: the asker is told what to say back, not just
        # shown two strings that happen to differ.
        markers = dict(citation_markers or {})
        if candidates and all(markers.get(candidate.id) for candidate in candidates):
            listed = "\n".join(
                f"- {candidate.label} {markers[candidate.id]}"
                for candidate in candidates
            )
            return (
                "같은 계약 설명에 해당하는 공시가 여러 건 있습니다.\n"
                f"{listed}\n"
                "어느 공시일의 계약을 말씀하시는지 알려주세요."
            )
        return (
            "같은 계약 설명에 해당하는 공시가 여러 건 있습니다. "
            f"어느 공시일의 계약을 말씀하시는지 알려주세요: {', '.join(labels)}"
        )
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
    "EVENT_INSTANCE",
    "MAX_CANDIDATES",
    "HOLDING_REPORT_INSTANCE",
    "MAX_LISTED_CANDIDATES",
    "ClarificationCandidate",
    "ClarificationDecision",
    "ClarificationRequest",
    "ClarificationState",
    "clarification_text",
]
