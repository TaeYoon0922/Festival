"""Deterministic citation-aware rendering of structured answer drafts."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.reasoning.answer_composer import AnswerDraft


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
    lines = ["확인된 보유 변동 내역은 다음과 같습니다."]
    section_citations: list[str] = []
    warnings: list[str] = []
    supported = bool(events)
    for index, event in enumerate(events, start=1):
        chunk_ids = _string_list(event.get("evidence_chunk_ids"))
        citation_ids = registry.ids_for(chunk_ids)
        lines.append(f"{index}.")
        if not citation_ids:
            lines.append("이 이벤트는 provenance가 없어 표시할 수 없습니다.")
            warnings.append(f"missing_provenance:holding_event:{index}")
            supported = False
            continue
        marker = " ".join(citation_ids)
        factual_lines = _holding_fact_lines(event, marker, requested_fields)
        if not factual_lines:
            lines.append("확인되지 않은 정보가 있습니다.")
            warnings.append(f"missing_fact_content:holding_event:{index}")
            supported = False
            continue
        lines.extend(factual_lines)
        section_citations.extend(citation_ids)

    sections = [
        GeneratedSection(
            title="보유 변동 내역",
            content="\n".join(lines),
            citations=_unique(section_citations),
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
    return sections, warnings, supported


#: Rendered first, always: they say whose holding this is and as of when.
HOLDING_IDENTITY_FIELDS = ("corp_name", "reporter", "reference_date")


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
                source_lines = _periodic_source_lines(source, marker)
                if not source_lines:
                    lines.append("확인되지 않은 정보가 있습니다.")
                    warnings.append(
                        f"missing_fact_content:periodic_source:{section_index}:{source_index}"
                    )
                    supported = False
                    continue
                metadata_lines.extend(
                    _periodic_source_metadata(source, source_index)
                )
                lines.extend(source_lines)
                citation_ids.extend(source_ids)
        else:
            fallback_ids = registry.ids_for(
                _string_list(fact.get("evidence_chunk_ids"))
                or answer_section.supporting_evidence_ids
            )
            fallback_lines = _periodic_fact_fallback_lines(fact, fallback_ids)
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


def _periodic_source_lines(source: Mapping[str, Any], marker: str) -> list[str]:
    fact_text = _text(source.get("fact_text"))
    return [f"내용: {fact_text} {marker}"] if fact_text else []


def _periodic_source_metadata(
    source: Mapping[str, Any], source_index: int
) -> list[str]:
    lines = []
    period = source.get("reporting_period")
    period_label = _period_label(period if isinstance(period, Mapping) else {})
    if period_label:
        lines.append(f"근거 {source_index} 보고 기간: {period_label}")
    report_name = _text(source.get("report_name"))
    if report_name:
        lines.append(f"근거 {source_index} 보고서: {report_name}")
    return lines


def _periodic_fact_fallback_lines(
    fact: Mapping[str, Any], citation_ids: Sequence[str]
) -> list[str]:
    if not citation_ids:
        return []
    marker = " ".join(citation_ids)
    fact_text = _text(fact.get("fact_text"))
    return [f"내용: {fact_text} {marker}"] if fact_text else []


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
    return re.sub(r"^(?:내용|대안\s+\d+):\s*", "", text).strip() or None


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
