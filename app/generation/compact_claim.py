"""Build a short, fully-cited claim from an already-verified holding answer.

A language model asked to restate a 2,500-character evidence report does not
restate it — it writes its own summary and discards whatever it was given.  The
fix is not a longer prompt but a smaller task: hand the model one or two
sentences' worth of facts that the deterministic pipeline has already proved,
and let it fix the word order and particles.

Everything here is read structurally out of :class:`AnswerDraft` and the
resolver's output.  The rendered answer string is never parsed, no value is
recomputed, and no event is chosen that the composer did not already accept.
When a safe claim cannot be built the answer is ``None`` and the caller serves
the deterministic answer unchanged — a skip, not an error.

Only ``holding_event`` is supported.  Periodic facts keep their value inside a
free-text ``fact_text`` field, so extracting one would mean parsing prose.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Imported rather than reimplemented so a compact claim renders a number and its
# unit exactly the way the frozen citation-aware generator does.  These helpers
# live in a frozen module, so the coupling is stable.
from app.generation.answer_generator import _numeric_text, _text


#: The task the router assigns to holding questions.
SUPPORTED_TASK_TYPE = "holding_event"

#: Labels and units mirror the frozen renderer so a compact answer and a full
#: answer describe the same field with the same words.
TEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("reference_date", "변동일"),
    ("report_date", "보고일"),
    ("receipt_date", "접수일"),
)

NUMERIC_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("before_shares", "변동 전 주식수", "주"),
    ("change_shares", "증감 주식수", "주"),
    ("after_shares", "변동 후 주식수", "주"),
    ("before_ratio", "변동 전 비율", "%"),
    ("after_ratio", "변동 후 비율", "%"),
    ("change_ratio", "증감 비율", "%"),
)

_TEXT_LABELS = dict(TEXT_FIELDS)
_NUMERIC_LABELS = {field: (label, unit) for field, label, unit in NUMERIC_FIELDS}

#: How a correction version role is named in the answer.
CORRECTION_ROLE_LABELS: dict[str, str] = {
    "correction_before": "정정 전",
    "correction_after": "정정 후",
}

#: What a field is called once a correction role owns the before/after axis.
#: The frozen labels describe a position inside one filing's own change ("변동
#: 후 주식수"); a correction role describes which *version of the filing* states
#: the value.  Naming both on one line would read as one axis, so a field that
#: states the holding as of the filing is named by what it measures instead.
#: Any other field keeps its frozen label and only gains the role prefix.
ROLE_FIELD_LABELS: dict[str, str] = {
    "after_shares": "보유주식수",
    "after_ratio": "보유비율",
}

#: Guardrails.  A claim that grows past these is no longer compact, and the
#: failure this module exists to prevent starts coming back.
MAX_CLAIM_EVENTS = 3
MAX_CLAIM_LITERALS = 12


@dataclass(frozen=True)
class ClaimCitation:
    """One cited chunk, numbered in the order the claim first refers to it."""

    marker: str
    chunk_id: str
    doc_id: str
    source_refs: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_refs": copy.deepcopy(list(self.source_refs)),
        }


@dataclass(frozen=True)
class ClaimField:
    """One requested field, its verified value, and where that value came from."""

    name: str
    label: str
    value: str
    marker: str
    chunk_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "value": self.value,
            "marker": self.marker,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class CompactClaim:
    """A verbalizable statement small enough that a model will actually restate it."""

    question: str
    company: str | None
    reporter: str | None
    fields: tuple[ClaimField, ...]
    citations: tuple[ClaimCitation, ...]
    deterministic_text: str

    @property
    def required_terms(self) -> tuple[str, ...]:
        return tuple(value for value in (self.company, self.reporter) if value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "company": self.company,
            "reporter": self.reporter,
            "fields": [field.to_dict() for field in self.fields],
            "citations": [citation.to_dict() for citation in self.citations],
            "deterministic_text": self.deterministic_text,
        }


def build_compact_claim(
    draft: Any,
    resolution: Any,
    *,
    task_type: str | None,
) -> CompactClaim | None:
    """Return a compact claim, or ``None`` when one cannot be built safely."""

    if task_type != SUPPORTED_TASK_TYPE:
        return None
    if draft is None or resolution is None or not getattr(draft, "answerable", False):
        return None

    resolved = _resolution_mapping(resolution)
    requested = tuple(resolved.get("requested_fields") or ())
    if not requested:
        # Nothing specific was asked, so there is no claim to make.
        return None
    if resolved.get("unresolved_fields"):
        # The composer could not answer part of the question; do not paper over it.
        return None

    events = _matching_events(draft, requested)
    if not events or len(events) > MAX_CLAIM_EVENTS:
        return None

    citation_index = _citation_index(draft)
    builder = _ClaimBuilder(citation_index)
    for event_index, event in events:
        if not builder.add_event(event_index, event, requested):
            return None

    if not builder.fields or len(builder.literal_count()) > MAX_CLAIM_LITERALS:
        return None

    company = _text(events[0][1].get("corp_name"))
    reporter = _text(events[0][1].get("reporter"))
    text = _render(company, reporter, builder.fields)
    if not text:
        return None

    return CompactClaim(
        question=_text(getattr(draft, "question", None)) or "",
        company=company,
        reporter=reporter,
        fields=tuple(builder.fields),
        citations=tuple(builder.citations),
        deterministic_text=text,
    )


class _ClaimBuilder:
    """Collect claim fields while numbering citations in order of first use."""

    def __init__(self, citation_index: Mapping[tuple[int, str], Mapping[str, Any]]):
        self._citation_index = citation_index
        self._markers: dict[str, str] = {}
        self.fields: list[ClaimField] = []
        self.citations: list[ClaimCitation] = []

    def add_event(
        self, event_index: int, event: Mapping[str, Any], requested: Sequence[str]
    ) -> bool:
        # Set only when the correction graph bound this filing to a version
        # role, so an ordinary event is labelled exactly as it always was.
        role = event.get("correction_role")
        role = str(role) if isinstance(role, str) and role else None
        for field in requested:
            value = _field_value(event, field)
            if value is None:
                return False
            citation = self._citation_for(event_index, field)
            if citation is None:
                # A value we cannot attribute is a value we do not state.
                return False
            marker = self._marker_for(citation)
            self.fields.append(
                ClaimField(
                    name=field,
                    label=_label(field, role),
                    value=value,
                    marker=marker,
                    chunk_id=str(citation["chunk_id"]),
                )
            )
        return True

    def literal_count(self) -> list[str]:
        return [field.value for field in self.fields]

    def _citation_for(
        self, event_index: int, field: str
    ) -> Mapping[str, Any] | None:
        direct = self._citation_index.get((event_index, field))
        if direct is not None:
            return direct
        return self._citation_index.get((event_index, "event_evidence"))

    def _marker_for(self, citation: Mapping[str, Any]) -> str:
        chunk_id = str(citation["chunk_id"])
        existing = self._markers.get(chunk_id)
        if existing is not None:
            return existing
        marker = f"[{len(self._markers) + 1}]"
        self._markers[chunk_id] = marker
        self.citations.append(
            ClaimCitation(
                marker=marker,
                chunk_id=chunk_id,
                doc_id=str(citation.get("doc_id") or ""),
                source_refs=tuple(
                    copy.deepcopy(list(citation.get("source_refs") or ()))
                ),
            )
        )
        return marker


def _matching_events(
    draft: Any, requested: Sequence[str]
) -> list[tuple[int, Mapping[str, Any]]]:
    """Keep only events the resolver matched that carry every requested field.

    Events are taken in the order the composer produced them.  None is dropped
    for recency and none is preferred over another, so this adds no selection
    semantics of its own.
    """

    events: list[tuple[int, Mapping[str, Any]]] = []
    for section in getattr(draft, "answer_sections", ()) or ():
        content = dict(getattr(section, "content", {}) or {})
        rows = content.get("events")
        if not isinstance(rows, list):
            continue
        for offset, row in enumerate(rows, start=1):
            if not isinstance(row, Mapping):
                continue
            if row.get("matches_query") is not True:
                continue
            if row.get("field_conflict"):
                continue
            if any(_field_value(row, field) is None for field in requested):
                continue
            events.append((offset, row))
    return events


def _citation_index(draft: Any) -> dict[tuple[int, str], Mapping[str, Any]]:
    """Map (event index, field) to the citation that supports it."""

    index: dict[tuple[int, str], Mapping[str, Any]] = {}
    for citation in getattr(draft, "citations", ()) or ():
        entry = {
            "chunk_id": citation.chunk_id,
            "doc_id": citation.doc_id,
            "source_refs": citation.source_refs,
        }
        for path in citation.provenance_path or ():
            if not isinstance(path, Mapping):
                continue
            if path.get("resolver") != "holding_event":
                continue
            event_index = path.get("event_index")
            field = path.get("field")
            if not isinstance(event_index, int) or not isinstance(field, str):
                continue
            index.setdefault((event_index, field), entry)
    return index


def _field_value(event: Mapping[str, Any], field: str) -> str | None:
    """Read one field verbatim, never recomputing or reformatting the number."""

    if field in _NUMERIC_LABELS:
        _, unit = _NUMERIC_LABELS[field]
        return _numeric_text(event.get(field), unit)
    if field in _TEXT_LABELS:
        return _text(event.get(field))
    value = event.get(field)
    return _text(value) if isinstance(value, str) else None


def _label(field: str, role: str | None = None) -> str:
    prefix = CORRECTION_ROLE_LABELS.get(role or "")
    if prefix:
        base = ROLE_FIELD_LABELS.get(field) or _frozen_label(field)
        return f"{prefix} {base}"
    return _frozen_label(field)


def _frozen_label(field: str) -> str:
    if field in _NUMERIC_LABELS:
        return _NUMERIC_LABELS[field][0]
    return _TEXT_LABELS.get(field, field)


def _render(
    company: str | None, reporter: str | None, fields: Sequence[ClaimField]
) -> str:
    """Render the claim as plain facts for the model to smooth into a sentence."""

    subject = " ".join(part for part in (reporter, company) if part)
    parts = [f"{field.label} {field.value}{field.marker}" for field in fields]
    if not parts:
        return ""
    body = ", ".join(parts)
    return f"{subject} {body}".strip() if subject else body


def _resolution_mapping(resolution: Any) -> Mapping[str, Any]:
    if isinstance(resolution, Mapping):
        return resolution
    to_dict = getattr(resolution, "to_dict", None)
    return to_dict() if callable(to_dict) else {}
