"""Deterministic citation-aware rendering of structured answer drafts."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.reasoning.answer_composer import AnswerDraft
from app.reasoning.periodic_metric_view import project_periodic_metric_table


@dataclass(frozen=True)
class GeneratedCitation:
    citation_id: str
    chunk_id: str
    doc_id: str
    source_refs: tuple[Mapping[str, Any], ...]
    section: str
    evidence_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_refs": copy.deepcopy(list(self.source_refs)),
            "section": self.section,
            "evidence_type": self.evidence_type,
        }


@dataclass(frozen=True)
class GeneratedSection:
    title: str
    content: str
    citations: tuple[str, ...]
    metadata: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "citations": list(self.citations),
            "metadata": list(self.metadata),
        }


@dataclass(frozen=True)
class GeneratedAnswer:
    question: str
    answer_text: str
    citations: tuple[GeneratedCitation, ...]
    sections: tuple[GeneratedSection, ...]
    warnings: tuple[str, ...]
    confidence: Mapping[str, Any]
    answerable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer_text": self.answer_text,
            "citations": [citation.to_dict() for citation in self.citations],
            "sections": [section.to_dict() for section in self.sections],
            "warnings": list(self.warnings),
            "confidence": copy.deepcopy(dict(self.confidence)),
            "answerable": self.answerable,
        }


class CitationAwareAnswerGenerator:
    """Render only facts and provenance already present in an AnswerDraft."""

    def generate(self, draft: AnswerDraft) -> GeneratedAnswer:
        return generate_answer(draft)


AnswerGenerator = CitationAwareAnswerGenerator


def generate_answer(draft: AnswerDraft) -> GeneratedAnswer:
    registry = _CitationRegistry(draft)
    answer_kind = _answer_kind(draft)
    if answer_kind == "holding":
        sections, render_warnings, factual_supported = _holding_sections(
            draft, registry
        )
    elif answer_kind == "periodic":
        sections, render_warnings, factual_supported = _periodic_sections(
            draft, registry
        )
        sections, scope_warnings, scope_valid = validate_periodic_citation_scope(
            draft,
            sections,
            registry.citations,
        )
        render_warnings.extend(scope_warnings)
        factual_supported = factual_supported and scope_valid
    else:
        sections, render_warnings, factual_supported = _general_sections(
            draft, registry
        )

    warnings = [*draft.warnings, *registry.warnings, *render_warnings]
    answerable = bool(draft.answerable and factual_supported)
    if not answerable:
        warnings.append("answer_not_supported")
        sections.append(
            GeneratedSection(
                title="확인 필요",
                content="확인되지 않은 정보가 있습니다.",
                citations=(),
            )
        )

    confidence = _confidence(draft.confidence, answerable=answerable)
    sections.append(
        GeneratedSection(
            title="신뢰도",
            content=confidence["display_text"],
            citations=(),
        )
    )
    citations = registry.citations
    return GeneratedAnswer(
        question=draft.question,
        answer_text=_render_answer_text(sections, citations),
        citations=citations,
        sections=tuple(sections),
        warnings=tuple(dict.fromkeys(warnings)),
        confidence=confidence,
        answerable=answerable,
    )


class _CitationRegistry:
    def __init__(self, draft: AnswerDraft) -> None:
        self._ids_by_chunk: dict[str, list[str]] = {}
        citations: list[GeneratedCitation] = []
        warnings: list[str] = []
        for citation in draft.citations:
            if not _has_provenance(citation):
                warnings.append(f"citation_missing_provenance:{citation.chunk_id}")
                continue
            citation_id = f"[{len(citations) + 1}]"
            generated = GeneratedCitation(
                citation_id=citation_id,
                chunk_id=citation.chunk_id,
                doc_id=citation.doc_id,
                source_refs=tuple(copy.deepcopy(list(citation.source_refs))),
                section=_citation_section(draft, citation.chunk_id),
                evidence_type=_evidence_type(citation.provenance_path),
            )
            citations.append(generated)
            self._ids_by_chunk.setdefault(citation.chunk_id, []).append(citation_id)
        self.citations = tuple(citations)
        self.warnings = tuple(dict.fromkeys(warnings))

    def ids_for(self, chunk_ids: Sequence[str]) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                citation_id
                for chunk_id in chunk_ids
                for citation_id in self._ids_by_chunk.get(str(chunk_id), ())
            )
        )


def _holding_sections(
    draft: AnswerDraft, registry: _CitationRegistry
) -> tuple[list[GeneratedSection], list[str], bool]:
    events = [
        event
        for section in draft.answer_sections
        for event in _mapping_list(section.content.get("events"))
    ]
    requested_fields = _unique(
        [
            value
            for section in draft.answer_sections
            for value in _string_list(section.content.get("requested_fields"))
        ]
    )
    single_event = len(events) == 1 and not draft.ambiguity.get("temporal_ambiguity")
    markers = [
        registry.ids_for(_string_list(event.get("evidence_chunk_ids")))
        for event in events
    ]

    compact = None
    if not single_event and len(events) >= 2 and all(markers):
        # One line per event instead of eleven, but only when every verified
        # value survives the compression.  All events or none: a half-compact
        # answer would make two events look like different kinds of fact.
        compact = _holding_compact_lines(events, markers)
    if compact is not None:
        return (
            _holding_section_list(draft, compact, [id for ids in markers for id in ids]),
            [],
            True,
        )

    lines = [] if single_event else ["확인된 보유 변동 내역은 다음과 같습니다."]
    section_citations: list[str] = []
    warnings: list[str] = []
    supported = bool(events)
    for index, event in enumerate(events, start=1):
        citation_ids = markers[index - 1]
        if not single_event:
            lines.append(f"{index}.")
        if not citation_ids:
            lines.append("이 이벤트는 provenance가 없어 표시할 수 없습니다.")
            warnings.append(f"missing_provenance:holding_event:{index}")
            supported = False
            continue
        marker = " ".join(citation_ids)
        factual_lines = None
        if single_event:
            # Prose only when it provably states every verified value; the
            # record form is the fallback, never a silent partial answer.
            factual_lines = _holding_prose_lines(event, marker, requested_fields)
        if factual_lines is None:
            factual_lines = _holding_fact_lines(event, marker, requested_fields)
        if not factual_lines:
            lines.append("확인되지 않은 정보가 있습니다.")
            warnings.append(f"missing_fact_content:holding_event:{index}")
            supported = False
            continue
        lines.extend(factual_lines)
        section_citations.extend(citation_ids)

    return _holding_section_list(draft, lines, section_citations), warnings, supported


def _holding_section_list(
    draft: AnswerDraft, lines: Sequence[str], citations: Sequence[str]
) -> list[GeneratedSection]:
    """Assemble the holding sections, keeping the ambiguity notice intact."""

    sections = [
        GeneratedSection(
            title="보유 변동 내역",
            content="\n".join(lines),
            citations=_unique(list(citations)),
        )
    ]
    if bool(draft.ambiguity.get("temporal_ambiguity")):
        sections.append(
            GeneratedSection(
                title="주의",
                content=(
                    "여러 변동 이벤트가 확인되어 특정 시점을 자동 선택하지 "
                    "않았습니다."
                ),
                citations=(),
            )
        )
    return sections


def _holding_compact_lines(
    events: Sequence[Mapping[str, Any]],
    markers: Sequence[Sequence[str]],
) -> list[str] | None:
    """State each verified event on one line, or return ``None``.

    The record form spends eleven labelled lines per event, so ten events read
    as a hundred-line dump of the same six labels.  Every value still appears
    here; only the repetition of the labels is removed.

    ``None`` means the compression dropped something, and the caller falls back
    to the record form for the whole set.  Mixing the two would make otherwise
    identical events look like different kinds of fact.
    """

    if not events or len(events) != len(markers):
        return None

    shared_company = _holding_shared_company(events)
    rows = [
        _holding_compact_row(event, marker, omit_company=shared_company is not None)
        for event, marker in zip(events, markers)
    ]
    if any(row is None for row in rows):
        return None

    header = (
        f"{shared_company}에 대해 확인된 보유 변동은 다음과 같습니다."
        if shared_company
        else "확인된 보유 변동은 다음과 같습니다."
    )
    for event, row, marker in zip(events, rows, markers):
        if not _holding_compact_row_states_every_value(
            event, row, marker, header=header, company_hoisted=bool(shared_company)
        ):
            return None
    return [header, *rows]


def _holding_shared_company(events: Sequence[Mapping[str, Any]]) -> str | None:
    """The company name, when every event names the same one.

    Only then may it move to the header; otherwise each row keeps its own, so
    two companies are never collapsed into one.
    """

    names = {_text(event.get("corp_name")) for event in events}
    if len(names) == 1:
        return next(iter(names))
    return None


def _holding_compact_row(
    event: Mapping[str, Any], marker: Sequence[str], *, omit_company: bool
) -> str | None:
    """Render one event as ``date | reporter | shares | ratio | filing [n]``."""

    date = _text(event.get("reference_date"))
    if not date or not marker:
        return None

    parts = [date]
    if not omit_company:
        company = _text(event.get("corp_name"))
        if company:
            parts.append(company)
    reporter = _text(event.get("reporter"))
    if reporter:
        parts.append(reporter)

    shares = _holding_compact_movement(event, _SHARES)
    if shares:
        parts.append(shares)
    ratio = _holding_compact_movement(event, _RATIO)
    if ratio:
        parts.append(ratio)

    filing = _holding_compact_filing(event)
    if filing:
        parts.append(filing)

    if len(parts) == 1:
        return None
    return " | ".join(parts) + " " + " ".join(marker)


def _holding_compact_movement(event: Mapping[str, Any], kind: int) -> str | None:
    """``before → after (change, direction)``, dropping whichever part is absent."""

    if kind == _SHARES:
        unit, change_unit = _SHARE_UNIT, _SHARE_UNIT
        fields = ("before_shares", "after_shares", "change_shares")
    else:
        unit, change_unit = _RATIO_UNIT, _RATIO_CHANGE_UNIT
        fields = ("before_ratio", "after_ratio", "change_ratio")

    before = _numeric_text(event.get(fields[0]), unit)
    after = _numeric_text(event.get(fields[1]), unit)
    change = _numeric_text(event.get(fields[2]), change_unit)

    if before and after:
        movement = f"{before} → {after}"
    else:
        movement = before or after
    if not movement:
        return change

    # The direction word rides with the share movement, so the record form's
    # "변동 방향" value still appears verbatim exactly once.
    annotations = [value for value in (change,) if value]
    if kind == _SHARES:
        direction = _direction_text(event.get("change_direction"))
        if direction:
            annotations.append(direction)
    if annotations:
        movement += " (" + ", ".join(annotations) + ")"
    return movement


def _holding_compact_filing(event: Mapping[str, Any]) -> str | None:
    parts = []
    for field, label in (("report_date", "보고"), ("receipt_date", "접수")):
        value = _text(event.get(field))
        if value:
            parts.append(f"{label} {value}")
    return " ".join(parts) if parts else None


def _holding_compact_row_states_every_value(
    event: Mapping[str, Any],
    row: str,
    marker: Sequence[str],
    *,
    header: str,
    company_hoisted: bool,
) -> bool:
    """Require this event's values to appear in *this event's* row.

    Checking the whole body instead would let a value repeat its way to a false
    pass: holding data reuses figures constantly — one event's after_shares is
    the next event's before_shares, the same reporter files every time, and two
    filings can share a change ratio. A value missing from row A would then be
    "found" in row B, and the reader would be shown a row that quietly lost a
    fact.

    ``corp_name`` is the one exception, and only when every event named the
    same company and it was hoisted into the header.
    """

    for entry in (
        *_HOLDING_TEXT_FIELDS,
        *_HOLDING_NUMERIC_FIELDS,
        _HOLDING_DIRECTION,
    ):
        value = _holding_fact_value(event, entry)
        if not value:
            continue
        if entry[0] == "corp_name" and company_hoisted:
            if value not in header:
                return False
            continue
        if value not in row:
            return False

    # The row has to carry its own provenance, not a neighbour's.
    return bool(marker) and all(citation_id in row for citation_id in marker)


#: Rendered first, always: they say whose holding this is and as of when.
HOLDING_IDENTITY_FIELDS = ("corp_name", "reporter", "reference_date")

#: Share counts and ratios read differently in a sentence, so each carries its
#: own unit and its own verb.
_SHARE_UNIT = "주"
_RATIO_UNIT = "%"

#: A change in ratio is the difference between two percentages, so it is stated
#: in percentage points.  Gold evidence terms are compared after every
#: non-alphanumeric character is stripped, so the trailing "p" cannot hide the
#: value: "1.05%" and "1.05%p" both reduce to "105".
_RATIO_CHANGE_UNIT = "%p"

#: Predicates for a movement, per direction.  "unchanged" is absent on purpose:
#: it has no natural transition phrasing that still contains the word the
#: record form prints, so those events keep the record form.
_DIRECTION_PREDICATES = {
    "increase": ("증가했습니다", "상승했습니다"),
    "decrease": ("감소했습니다", "하락했습니다"),
}

_SHARES = 0
_RATIO = 1


def _holding_prose_lines(
    event: Mapping[str, Any],
    marker: str,
    requested_fields: Sequence[str] = (),
) -> list[str] | None:
    """State one verified event as sentences, or return ``None``.

    The record form is accurate but reads like a table dump: eleven labelled
    lines, each repeating the same citation.  One verified event is small
    enough to say in prose, led by whatever the question actually asked.

    ``None`` means prose could not carry every verified value, and the caller
    falls back to the record form.  That check is the whole safety story here:
    an earlier tuning round dropped unasked fields and cost the evaluation set
    its evidence coverage, so completeness is verified rather than assumed.
    """

    subject = _holding_subject(event)
    if subject is None:
        return None
    direction = _text(event.get("change_direction"))
    if direction and direction not in _DIRECTION_PREDICATES:
        return None

    stated: set[str] = set()
    sentences = [
        sentence
        for sentence in (
            _holding_lead_sentence(event, subject, requested_fields, stated),
            _holding_movement_sentence(event, _SHARES, stated),
            _holding_movement_sentence(event, _RATIO, stated),
            _holding_filing_sentence(event, stated),
        )
        if sentence
    ]
    if not sentences:
        return None

    prose = " ".join(sentences)
    if not _holding_prose_states_every_value(event, prose):
        return None
    return [f"{sentence} {marker}" for sentence in sentences]


def _holding_prose_states_every_value(
    event: Mapping[str, Any], prose: str
) -> bool:
    """Require every value the record form would print to appear in the prose."""

    for entry in (*_HOLDING_TEXT_FIELDS, *_HOLDING_NUMERIC_FIELDS, _HOLDING_DIRECTION):
        value = _holding_fact_value(event, entry)
        if value and value not in prose:
            return False
    return True


def _holding_subject(event: Mapping[str, Any]) -> str | None:
    corp = _text(event.get("corp_name"))
    date = _text(event.get("reference_date"))
    if not corp or not date:
        return None
    reporter = _text(event.get("reporter"))
    holder = f"{reporter}의 " if reporter else ""
    return f"{date} 기준 {holder}{corp}"


def _holding_lead_sentence(
    event: Mapping[str, Any],
    subject: str,
    requested_fields: Sequence[str],
    stated: set[str],
) -> str | None:
    """Answer what was asked, before anything else is said."""

    clauses = []
    for field in _holding_lead_fields(event, requested_fields):
        clause = _holding_lead_clause(event, field, stated)
        if clause:
            clauses.append(clause)
    if not clauses:
        return None
    return f"{subject} {'. '.join(clauses)}."


def _holding_lead_fields(
    event: Mapping[str, Any], requested_fields: Sequence[str]
) -> list[str]:
    """Requested fields lead; otherwise the holding as of the reference date."""

    preference = (
        "after_ratio",
        "after_shares",
        "change_shares",
        "change_ratio",
        "before_shares",
        "before_ratio",
    )
    requested = [
        field
        for field in preference
        if field in requested_fields and event.get(field) is not None
    ]
    if requested:
        return requested
    return [
        field
        for field in ("after_shares", "after_ratio")
        if event.get(field) is not None
    ]


def _holding_lead_clause(
    event: Mapping[str, Any], field: str, stated: set[str]
) -> str | None:
    templates = {
        "after_ratio": ("보유 비율은 {value}입니다", _RATIO_UNIT, None),
        "after_shares": ("보유 주식수는 {value}입니다", _SHARE_UNIT, None),
        "before_shares": ("직전 보유 주식수는 {value}였습니다", _SHARE_UNIT, None),
        "before_ratio": ("직전 보유 비율은 {value}였습니다", _RATIO_UNIT, None),
        "change_shares": ("직전 보고 대비 {value} {verb}", _SHARE_UNIT, _SHARES),
        "change_ratio": ("보유 비율은 {value} {verb}", _RATIO_CHANGE_UNIT, _RATIO),
    }
    entry = templates.get(field)
    if entry is None:
        return None
    template, unit, kind = entry
    value = _numeric_text(event.get(field), unit)
    if not value:
        return None
    stated.add(value)
    verb = "변동했습니다" if kind is None else _holding_predicate(event, kind)
    return template.format(value=value, verb=verb)


def _holding_movement_sentence(
    event: Mapping[str, Any], kind: int, stated: set[str]
) -> str | None:
    """Say a before/after movement once, keeping all three of its values."""

    if kind == _SHARES:
        # The topic particle is carried with the label: Korean picks 은 or 는
        # from the preceding syllable, and both labels here are fixed.
        label, particle = "보유 주식수", "는"
        unit, change_unit = _SHARE_UNIT, _SHARE_UNIT
        fields = ("before_shares", "after_shares", "change_shares")
    else:
        label, particle = "보유 비율", "은"
        unit, change_unit = _RATIO_UNIT, _RATIO_CHANGE_UNIT
        fields = ("before_ratio", "after_ratio", "change_ratio")

    before = _numeric_text(event.get(fields[0]), unit)
    after = _numeric_text(event.get(fields[1]), unit)
    change = _numeric_text(event.get(fields[2]), change_unit)
    values = [value for value in (before, after, change) if value]
    if not values or all(value in stated for value in values):
        return None

    predicate = _holding_predicate(event, kind)
    if before and after:
        # The change figure is left out when the lead already gave it, so the
        # same number is not repeated in consecutive sentences.
        middle = f"{change} " if change and change not in stated else ""
        stated.update(values)
        return f"{label}{particle} {before}에서 {after}로 {middle}{predicate}."

    remaining = [value for value in values if value not in stated]
    stated.update(values)
    return f"{label}{particle} " + ", ".join(remaining) + "로 확인됩니다."


def _holding_filing_sentence(
    event: Mapping[str, Any], stated: set[str]
) -> str | None:
    """Carry the filing dates without giving each one a line of its own."""

    parts = []
    for field, label in (("report_date", "보고일"), ("receipt_date", "접수일")):
        value = _text(event.get(field))
        if value and value not in stated:
            stated.add(value)
            parts.append(f"{label}은 {value}")
    if not parts:
        return None
    return ", ".join(parts) + "입니다."


def _holding_predicate(event: Mapping[str, Any], kind: int) -> str:
    predicates = _DIRECTION_PREDICATES.get(_text(event.get("change_direction")) or "")
    return predicates[kind] if predicates else "변동했습니다"


def _holding_fact_value(
    event: Mapping[str, Any], entry: tuple[str, ...]
) -> str | None:
    field = entry[0]
    if len(entry) == 3:
        return _numeric_text(event.get(field), entry[2])
    if field == "change_direction":
        return _direction_text(event.get(field))
    return _text(event.get(field))


def _holding_fact_lines(
    event: Mapping[str, Any],
    marker: str,
    requested_fields: Sequence[str] = (),
) -> list[str]:
    """Render every verified fact on an event, asked-for figures first.

    ``requested_fields`` decides order, never membership.  An earlier version
    dropped the fields the question did not name, which read well but deleted
    verified facts a reader needs to interpret the one they asked for: a change
    ratio without the share count it applies to, or an after value with nothing
    to compare it against.  Ordering gives the same directness without losing
    anything.
    """

    requested = tuple(dict.fromkeys(requested_fields))
    identity = [
        (field, label)
        for field, label in _HOLDING_TEXT_FIELDS
        if field in HOLDING_IDENTITY_FIELDS
    ]
    remaining = [
        entry
        for entry in (*_HOLDING_TEXT_FIELDS, *_HOLDING_NUMERIC_FIELDS, _HOLDING_DIRECTION)
        if entry[0] not in HOLDING_IDENTITY_FIELDS
    ]
    # Hoist what the question named, keeping the canonical order inside each
    # group so two questions never render the same event differently.
    ordered = [
        *identity,
        *[entry for entry in remaining if entry[0] in requested],
        *[entry for entry in remaining if entry[0] not in requested],
    ]

    lines = []
    for entry in ordered:
        line = _holding_fact_line(event, entry, marker)
        if line:
            lines.append(line)
    return lines


_HOLDING_TEXT_FIELDS = (
    ("corp_name", "회사"),
    ("reporter", "보고자"),
    ("reference_date", "변동일"),
    ("report_date", "보고일"),
    ("receipt_date", "접수일"),
)

_HOLDING_NUMERIC_FIELDS = (
    ("before_shares", "변동 전 주식수", "주"),
    ("change_shares", "증감 주식수", "주"),
    ("after_shares", "변동 후 주식수", "주"),
    ("before_ratio", "변동 전 비율", "%"),
    ("after_ratio", "변동 후 비율", "%"),
    ("change_ratio", "증감 비율", "%"),
)

_HOLDING_DIRECTION = ("change_direction", "변동 방향")


def _holding_fact_line(
    event: Mapping[str, Any], entry: tuple[str, ...], marker: str
) -> str | None:
    field, label = entry[0], entry[1]
    if len(entry) == 3:
        value = _numeric_text(event.get(field), entry[2])
    elif field == "change_direction":
        value = _direction_text(event.get(field))
    else:
        value = _text(event.get(field))
    return f"{label}: {value} {marker}" if value else None


def _periodic_sections(
    draft: AnswerDraft, registry: _CitationRegistry
) -> tuple[list[GeneratedSection], list[str], bool]:
    output: list[GeneratedSection] = []
    warnings: list[str] = []
    facts_seen = 0
    supported = True
    for section_index, answer_section in enumerate(draft.answer_sections, start=1):
        fact = answer_section.content.get("fact")
        if not isinstance(fact, Mapping):
            continue
        request = answer_section.content.get("request")
        request = dict(request) if isinstance(request, Mapping) else {}
        facts_seen += 1
        lines = ["확인된 사업 또는 공시 내용:"]
        metadata_lines: list[str] = []
        citation_ids: list[str] = []
        sources = _mapping_list(fact.get("sources"))
        if sources:
            for source_index, source in enumerate(sources, start=1):
                source_ids = registry.ids_for(
                    _string_list(source.get("chunk_id"))
                )
                lines.append(f"{source_index}.")
                if not source_ids:
                    lines.append("이 source는 provenance가 없어 표시할 수 없습니다.")
                    warnings.append(
                        f"missing_provenance:periodic_source:{section_index}:{source_index}"
                    )
                    supported = False
                    continue
                marker = " ".join(source_ids)
                source_lines = _periodic_source_lines(source, marker, request=request)
                if not source_lines:
                    lines.append("확인되지 않은 정보가 있습니다.")
                    warnings.append(
                        f"missing_fact_content:periodic_source:{section_index}:{source_index}"
                    )
                    supported = False
                    continue
                metadata_lines.extend(
                    _periodic_source_metadata(source, source_index, request=request)
                )
                lines.extend(source_lines)
                citation_ids.extend(source_ids)
        else:
            fallback_ids = registry.ids_for(
                _string_list(fact.get("evidence_chunk_ids"))
                or answer_section.supporting_evidence_ids
            )
            fallback_lines = _periodic_fact_fallback_lines(
                fact, fallback_ids, request=request
            )
            if fallback_lines:
                metadata_lines.extend(_periodic_fact_fallback_metadata(fact))
                lines.extend(fallback_lines)
                citation_ids.extend(fallback_ids)
            else:
                lines.append("이 fact는 provenance가 없어 표시할 수 없습니다.")
                warnings.append(f"missing_provenance:periodic_fact:{section_index}")
                supported = False

        (
            alternative_lines,
            alternative_metadata,
            alternative_ids,
            alternatives_supported,
        ) = (
            _periodic_alternative_lines(fact, registry)
        )
        lines.extend(alternative_lines)
        metadata_lines.extend(alternative_metadata)
        citation_ids.extend(alternative_ids)
        supported = supported and alternatives_supported
        fact_markers = " ".join(_unique(citation_ids))
        if fact.get("repeated_across_periods") and fact_markers:
            metadata_lines.append(
                "여러 기간의 공시에서 동일 사실이 확인됩니다."
            )
        if fact.get("period_evolution") and fact_markers:
            metadata_lines.append("기간별 내용 변화가 함께 보존되어 있습니다.")
        output.append(
            GeneratedSection(
                title=answer_section.title,
                content="\n".join(lines),
                citations=_unique(citation_ids),
                metadata=_unique(metadata_lines),
            )
        )

    if bool(draft.ambiguity.get("temporal_ambiguity")):
        output.append(
            GeneratedSection(
                title="주의",
                content="여러 기간 또는 사실 후보를 자동으로 선택하지 않았습니다.",
                citations=(),
            )
        )
    return output, warnings, bool(facts_seen) and supported


def _periodic_source_lines(
    source: Mapping[str, Any],
    marker: str,
    *,
    request: Mapping[str, Any] | None = None,
) -> list[str]:
    fact_text = _text(source.get("fact_text"))
    if not fact_text:
        return []
    metric = _text((request or {}).get("metric"))
    display = (
        project_periodic_metric_table(
            fact_text,
            metric=metric,
            period=(request or {}).get("period"),
            comparison=(request or {}).get("comparison"),
            raw_query=_text((request or {}).get("raw_query")),
        )
        or fact_text
    )
    return [f"내용: {display} {marker}"]


def _periodic_source_metadata(
    source: Mapping[str, Any],
    source_index: int,
    *,
    request: Mapping[str, Any] | None = None,
) -> list[str]:
    lines = []
    period = source.get("reporting_period")
    period_label = _period_label(period if isinstance(period, Mapping) else {})
    if period_label:
        lines.append(f"근거 {source_index} 보고 기간: {period_label}")
    report_name = _text(source.get("report_name"))
    if report_name:
        lines.append(f"근거 {source_index} 보고서: {report_name}")
    basis_label = _basis_label(request, source)
    if basis_label:
        lines.append(f"근거 {source_index} 재무제표 기준: {basis_label}")
    return lines


def _basis_label(
    request: Mapping[str, Any] | None, source: Mapping[str, Any]
) -> str | None:
    requested = _text((request or {}).get("basis"))
    if requested == "consolidated":
        return "연결"
    if requested == "standalone":
        return "별도"
    section = " ".join(str(part) for part in (source.get("section_path") or []) if part)
    if "연결" in section:
        return "연결"
    if "별도" in section or "개별" in section:
        return "별도"
    return None


def _periodic_fact_fallback_lines(
    fact: Mapping[str, Any],
    citation_ids: Sequence[str],
    *,
    request: Mapping[str, Any] | None = None,
) -> list[str]:
    if not citation_ids:
        return []
    marker = " ".join(citation_ids)
    fact_text = _text(fact.get("fact_text"))
    if not fact_text:
        return []
    metric = _text((request or {}).get("metric"))
    display = (
        project_periodic_metric_table(
            fact_text,
            metric=metric,
            period=(request or {}).get("period"),
            comparison=(request or {}).get("comparison"),
            raw_query=_text((request or {}).get("raw_query")),
        )
        or fact_text
    )
    return [f"내용: {display} {marker}"]


def _periodic_fact_fallback_metadata(fact: Mapping[str, Any]) -> list[str]:
    lines = []
    for period in _mapping_list(fact.get("reporting_periods")):
        label = _period_label(period)
        if label:
            lines.append(f"보고 기간: {label}")
    for report_name in _string_list(fact.get("report_names")):
        lines.append(f"보고서: {report_name}")
    return lines


def _periodic_alternative_lines(
    fact: Mapping[str, Any], registry: _CitationRegistry
) -> tuple[list[str], list[str], list[str], bool]:
    alternatives = _mapping_list(fact.get("alternatives"))
    fact_conflict = bool(fact.get("fact_conflict"))
    period_evolution = bool(fact.get("period_evolution"))
    if not alternatives or (
        len(alternatives) == 1 and not fact_conflict and not period_evolution
    ):
        return [], [], [], True
    lines = [
        "상충하는 대안:" if fact_conflict else "기간별로 구분되는 내용:"
    ]
    used_ids: list[str] = []
    metadata: list[str] = []
    supported = True
    for index, alternative in enumerate(alternatives, start=1):
        ids = registry.ids_for(_string_list(alternative.get("evidence_chunk_ids")))
        if not ids:
            lines.append(f"대안 {index}: provenance가 없어 표시할 수 없습니다.")
            supported = False
            continue
        marker = " ".join(ids)
        fact_texts = _string_list(alternative.get("fact_texts"))
        if fact_texts:
            for text in fact_texts:
                lines.append(f"대안 {index}: {text} {marker}")
        else:
            lines.append(f"대안 {index}: 확인된 원문이 없습니다. {marker}")
            supported = False
        for period in _mapping_list(alternative.get("reporting_periods")):
            label = _period_label(period)
            if label:
                metadata.append(f"대안 {index} 보고 기간: {label}")
        used_ids.extend(ids)
    return lines, metadata, used_ids, supported


def validate_periodic_citation_scope(
    draft: AnswerDraft,
    sections: Sequence[GeneratedSection],
    citations: Sequence[GeneratedCitation],
) -> tuple[list[GeneratedSection], list[str], bool]:
    """Remove periodic claims not copied from their selected cited sources."""

    source_text_by_chunk = _selected_periodic_source_text(draft)
    chunk_by_citation = {
        citation.citation_id: citation.chunk_id for citation in citations
    }
    output: list[GeneratedSection] = []
    warnings: list[str] = []
    valid = True
    for section_index, section in enumerate(sections, start=1):
        if not section.citations:
            output.append(section)
            continue
        kept_lines = []
        for line_index, line in enumerate(section.content.splitlines(), start=1):
            payload = _periodic_claim_payload(line)
            if payload is None:
                kept_lines.append(line)
                continue
            inline_ids = tuple(
                f"[{value}]" for value in re.findall(r"\[(\d+)\]", line)
            )
            citation_ids = inline_ids or section.citations
            allowed_text = "\n".join(
                source_text_by_chunk.get(chunk_by_citation.get(citation_id, ""), "")
                for citation_id in citation_ids
            )
            if _claim_in_selected_source(payload, allowed_text):
                kept_lines.append(line)
                continue
            valid = False
            warnings.append(
                "unsupported_periodic_claim_removed:"
                f"section={section_index}:line={line_index}"
            )
        output.append(
            GeneratedSection(
                title=section.title,
                content="\n".join(kept_lines),
                citations=section.citations,
                metadata=section.metadata,
            )
        )
    return output, list(dict.fromkeys(warnings)), valid


def _selected_periodic_source_text(draft: AnswerDraft) -> dict[str, str]:
    selected = set(draft.evidence_references)
    output: dict[str, list[str]] = {}
    for section in draft.answer_sections:
        fact = section.content.get("fact")
        if not isinstance(fact, Mapping):
            continue
        sources = _mapping_list(fact.get("sources"))
        for source in sources:
            chunk_ids = _string_list(source.get("chunk_id"))
            fact_text = _text(source.get("fact_text"))
            for chunk_id in chunk_ids:
                if chunk_id in selected and fact_text:
                    output.setdefault(chunk_id, []).append(fact_text)
        if not sources:
            fact_text = _text(fact.get("fact_text"))
            if fact_text:
                for chunk_id in section.supporting_evidence_ids:
                    if chunk_id in selected:
                        output.setdefault(chunk_id, []).append(fact_text)
    return {
        chunk_id: "\n".join(dict.fromkeys(values))
        for chunk_id, values in output.items()
    }


def _periodic_claim_payload(line: str) -> str | None:
    text = re.sub(r"\[(\d+)\]", "", str(line)).strip()
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
    if not text or text.endswith(":"):
        return None
    if re.fullmatch(r"\|(?:\s*:?-{3,}:?\s*\|)+", text):
        return None
    payload = re.sub(r"^(?:내용|대안\s+\d+):\s*", "", text).strip()
    if _is_periodic_table_header(payload):
        return None
    if not re.search(r"[0-9A-Za-z가-힣]", text):
        return None
    if any(
        marker in text
        for marker in (
            "확인된 사업 또는 공시 내용",
            "provenance가 없어",
            "확인되지 않은 정보가 있습니다",
        )
    ):
        return None
    return payload or None


def _is_periodic_table_header(text: str) -> bool:
    if not str(text or "").strip().startswith("|"):
        return False
    cells = [cell.strip() for cell in str(text).strip().strip("|").split("|")]
    if len(cells) < 2:
        return False
    first = re.sub(r"[^0-9a-z가-힣]+", "", cells[0].casefold())
    if first not in {"열1", "구분", "계정과목", "과목"}:
        return False
    return any(
        re.search(r"제\s*\d+\s*기|20\d{2}\s*년|분기|반기|누적|3\s*개월", cell)
        for cell in cells[1:]
    )


def _claim_in_selected_source(claim: str, source_text: str) -> bool:
    normalized_claim = re.sub(
        r"[^0-9a-z가-힣]+", "", str(claim).casefold()
    )
    normalized_source = re.sub(
        r"[^0-9a-z가-힣]+", "", str(source_text).casefold()
    )
    return bool(normalized_claim and normalized_claim in normalized_source)


def _general_sections(
    draft: AnswerDraft, registry: _CitationRegistry
) -> tuple[list[GeneratedSection], list[str], bool]:
    output = []
    warnings = []
    evidence_seen = 0
    supported = True
    for section_index, answer_section in enumerate(draft.answer_sections, start=1):
        rows = _mapping_list(answer_section.content.get("evidence"))
        if not rows:
            continue
        lines = []
        used_ids: list[str] = []
        for row_index, row in enumerate(rows, start=1):
            evidence_seen += 1
            ids = registry.ids_for(_string_list(row.get("chunk_id")))
            if not ids:
                lines.append(
                    f"{row_index}. provenance가 없어 evidence를 표시할 수 없습니다."
                )
                warnings.append(
                    f"missing_provenance:general_evidence:{section_index}:{row_index}"
                )
                supported = False
                continue
            marker = " ".join(ids)
            text = _text(row.get("evidence_text"))
            if text:
                lines.append(f"{row_index}. {text} {marker}")
                used_ids.extend(ids)
            else:
                lines.append(f"{row_index}. 확인되지 않은 정보가 있습니다.")
                supported = False
        output.append(
            GeneratedSection(
                title=answer_section.title,
                content="\n".join(lines),
                citations=_unique(used_ids),
            )
        )
    return output, warnings, bool(evidence_seen) and supported


def _answer_kind(draft: AnswerDraft) -> str:
    if any("events" in section.content for section in draft.answer_sections):
        return "holding"
    if any("fact" in section.content for section in draft.answer_sections):
        return "periodic"
    return "general"


def _has_provenance(citation: Any) -> bool:
    if not _text(getattr(citation, "chunk_id", None)) or not _text(
        getattr(citation, "doc_id", None)
    ):
        return False
    paths = getattr(citation, "provenance_path", ()) or ()
    return any(isinstance(path, Mapping) and bool(path) for path in paths)


def _citation_section(draft: AnswerDraft, chunk_id: str) -> str:
    return next(
        (
            section.title
            for section in draft.answer_sections
            if chunk_id in section.supporting_evidence_ids
        ),
        "Evidence",
    )


def _evidence_type(paths: Sequence[Mapping[str, Any]]) -> str:
    values = [
        _text(path.get("resolver"))
        for path in paths
        if isinstance(path, Mapping)
    ]
    values = [value for value in values if value]
    return values[0] if values else "general_evidence"


def _period_label(period: Mapping[str, Any]) -> str | None:
    year = _text(
        period.get("fiscal_year") or period.get("base_year") or period.get("year")
    )
    quarter = _text(period.get("quarter"))
    period_type = _text(period.get("period_type") or period.get("basis_period"))
    end_date = _text(
        period.get("to_date")
        or period.get("to")
        or period.get("period_end")
        or period.get("report_period")
    )
    values = []
    if year:
        values.append(f"{year}년")
    if quarter:
        values.append(f"{quarter}분기")
    if period_type and period_type not in {"fiscal_year", "fiscal_quarter"}:
        values.append(period_type)
    if end_date:
        values.append(end_date)
    return " ".join(values) or None


def _numeric_text(value: Any, unit: str) -> str | None:
    if not isinstance(value, Mapping):
        return None
    raw = _text(value.get("raw"))
    if raw is None and value.get("normalized") is not None:
        raw = str(value["normalized"])
    if raw is None:
        return None
    return raw if raw.endswith(unit) else f"{raw}{unit}"


def _direction_text(value: Any) -> str | None:
    return {
        "increase": "증가",
        "decrease": "감소",
        "unchanged": "변동 없음",
    }.get(_text(value) or "", _text(value))


def _confidence(value: Mapping[str, Any], *, answerable: bool) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    level = _text(output.get("level")) or "low"
    if not answerable:
        level = "low"
    display = {
        "high": "답변 신뢰도: 높음",
        "medium": "답변 신뢰도: 중간",
        "low": "추가 확인이 필요합니다.",
    }.get(level, "추가 확인이 필요합니다.")
    output["level"] = level
    output["display_text"] = display
    output["answerable"] = answerable
    return output


def _render_answer_text(
    sections: Sequence[GeneratedSection],
    citations: Sequence[GeneratedCitation],
) -> str:
    blocks = []
    for section in sections:
        body = "\n".join([*section.metadata, section.content]).strip()
        if body:
            blocks.append(f"{section.title}\n{body}")
    if citations:
        citation_lines = ["인용"]
        for citation in citations:
            citation_lines.extend(
                (
                    citation.citation_id,
                    f"doc_id: {citation.doc_id}",
                    f"chunk_id: {citation.chunk_id}",
                )
            )
        blocks.append("\n".join(citation_lines))
    return "\n\n".join(blocks)


def _mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item is not None and str(item)]
    return []


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None
