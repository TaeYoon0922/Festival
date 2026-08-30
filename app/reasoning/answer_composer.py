"""Read-only structured answer composition over resolved evidence."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.reasoning.evidence_builder import EvidenceItem, EvidenceSet
from app.reasoning.holding_event_resolver import HoldingEvent, HoldingResolution
from app.reasoning.holding_event_selection import (
    EXACT,
    NOT_APPLICABLE,
    is_semantically_unique,
)
from app.reasoning.holding_reporter import reporter_matches
from app.reasoning.periodic_fact_resolver import (
    PeriodicFact,
    PeriodicFactResolution,
)


@dataclass(frozen=True)
class EvidenceCitation:
    chunk_id: str
    doc_id: str
    source_refs: tuple[Mapping[str, Any], ...]
    provenance_path: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_refs": copy.deepcopy(list(self.source_refs)),
            "provenance_path": copy.deepcopy(list(self.provenance_path)),
        }


@dataclass(frozen=True)
class AnswerSection:
    title: str
    content: Mapping[str, Any]
    supporting_evidence_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": copy.deepcopy(dict(self.content)),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
        }


@dataclass(frozen=True)
class AnswerDraft:
    question: str
    task_type: str | None
    answer_sections: tuple[AnswerSection, ...]
    evidence_references: tuple[str, ...]
    citations: tuple[EvidenceCitation, ...]
    ambiguity: Mapping[str, Any]
    warnings: tuple[str, ...]
    confidence: Mapping[str, Any]
    answerable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "task_type": self.task_type,
            "answer_sections": [section.to_dict() for section in self.answer_sections],
            "evidence_references": list(self.evidence_references),
            "citations": [citation.to_dict() for citation in self.citations],
            "ambiguity": copy.deepcopy(dict(self.ambiguity)),
            "warnings": list(self.warnings),
            "confidence": copy.deepcopy(dict(self.confidence)),
            "answerable": self.answerable,
        }


class AnswerComposer:
    """Compose an LLM-ready draft without selecting or reranking evidence."""

    def compose(
        self,
        evidence_set: EvidenceSet,
        *,
        holding_resolution: HoldingResolution | None = None,
        periodic_resolution: PeriodicFactResolution | None = None,
        selection_mode: str = NOT_APPLICABLE,
    ) -> AnswerDraft:
        supplied = sum(
            value is not None for value in (holding_resolution, periodic_resolution)
        )
        if supplied != 1:
            raise ValueError("exactly one resolver output must be supplied")
        if holding_resolution is not None:
            return compose_holding_answer(
                holding_resolution, evidence_set, selection_mode=selection_mode
            )
        assert periodic_resolution is not None
        return compose_periodic_answer(periodic_resolution, evidence_set)

    def compose_holding(
        self,
        resolution: HoldingResolution,
        evidence_set: EvidenceSet,
        *,
        selection_mode: str = NOT_APPLICABLE,
    ) -> AnswerDraft:
        return compose_holding_answer(
            resolution, evidence_set, selection_mode=selection_mode
        )

    def compose_periodic(
        self, resolution: PeriodicFactResolution, evidence_set: EvidenceSet
    ) -> AnswerDraft:
        return compose_periodic_answer(resolution, evidence_set)


def compose_holding_answer(
    resolution: HoldingResolution,
    evidence_set: EvidenceSet,
    *,
    selection_mode: str = NOT_APPLICABLE,
) -> AnswerDraft:
    evidence_by_id = _evidence_by_id(evidence_set)
    citation_builder = _CitationBuilder(evidence_by_id)
    reported_events = _reported_holding_events(resolution)
    event_rows = []
    for index, event in enumerate(reported_events, start=1):
        citation_builder.add_holding_event(index, event)
        event_rows.append(_holding_event_content(event))
    evidence_ids = _unique_text(
        chunk_id for event in reported_events for chunk_id in event.evidence_chunk_ids
    )
    sections = (
        (
            AnswerSection(
                title="Holding events",
                content={
                    "events": event_rows,
                    "requested_fields": list(resolution.requested_fields),
                },
                supporting_evidence_ids=evidence_ids,
            ),
        )
        if event_rows
        else ()
    )

    citations = citation_builder.build()
    temporal_ambiguity = (
        resolution.temporal_ambiguity or resolution.matching_event_count > 1
    )
    # Whether one event may be called *the* answer is a question about the
    # question, not about how many events retrieval exposed.  A count of one is
    # routinely an artifact of which projections were served, so it is recorded
    # separately from whether the query ever identified a single event.
    observed = len(reported_events)
    semantic_unique = is_semantically_unique(
        selection_mode, resolution.matching_event_count
    )
    scoped = bool(resolution.requested_fields) and bool(observed)
    under_specified = scoped and selection_mode not in (NOT_APPLICABLE, EXACT)
    exact_multi = (
        scoped and selection_mode == EXACT and resolution.matching_event_count > 1
    )
    warnings = [*evidence_set.warnings, *resolution.warnings]
    if temporal_ambiguity:
        warnings.append("multiple_matching_holding_events")
    answerable, requirements = _holding_answerability(resolution, citations)
    if not answerable:
        warnings.append("answer_not_supported")
    confidence = _holding_confidence(
        resolution,
        answerable=answerable,
        citation_count=len(citations),
        unresolved=requirements,
    )
    return AnswerDraft(
        question=resolution.question,
        task_type=evidence_set.task_type or "holding_change",
        answer_sections=sections,
        evidence_references=evidence_ids,
        citations=citations,
        ambiguity={
            "temporal_ambiguity": temporal_ambiguity,
            "matching_event_count": resolution.matching_event_count,
            "alternative_event_count": len(resolution.events),
            "latest_event_selected": False,
            "selection_mode": selection_mode,
            "semantic_unique": semantic_unique,
            "under_specified": under_specified,
            "exact_multi_match": exact_multi,
            "observable_matching_event_count": observed,
        },
        warnings=tuple(dict.fromkeys(warnings)),
        confidence=confidence,
        answerable=answerable,
    )


def compose_periodic_answer(
    resolution: PeriodicFactResolution, evidence_set: EvidenceSet
) -> AnswerDraft:
    evidence_by_id = _evidence_by_id(evidence_set)
    citation_builder = _CitationBuilder(evidence_by_id)
    request = _periodic_request(evidence_set.query_plan)
    sections: list[AnswerSection] = []
    for index, fact in enumerate(resolution.facts, start=1):
        citation_builder.add_periodic_fact(index, fact)
        sections.append(
            AnswerSection(
                title=f"Periodic fact {index}",
                content=_periodic_fact_content(fact, request=request),
                supporting_evidence_ids=fact.evidence_chunk_ids,
            )
        )

    citations = citation_builder.build()
    answerable, requirements = _periodic_answerability(resolution, citations)
    warnings = [*evidence_set.warnings, *resolution.warnings]
    if not answerable:
        warnings.append("answer_not_supported")
    evidence_ids = _unique_text(
        chunk_id for fact in resolution.facts for chunk_id in fact.evidence_chunk_ids
    )
    confidence = _periodic_confidence(
        resolution,
        answerable=answerable,
        citation_count=len(citations),
        unresolved=requirements,
    )
    return AnswerDraft(
        question=resolution.question,
        task_type=resolution.task_type or evidence_set.task_type,
        answer_sections=tuple(sections),
        evidence_references=evidence_ids,
        citations=citations,
        ambiguity={
            "temporal_ambiguity": resolution.temporal_ambiguity,
            "matching_fact_count": resolution.matching_fact_count,
            "alternative_fact_count": len(resolution.facts),
            "latest_period_selected": False,
        },
        warnings=tuple(dict.fromkeys(warnings)),
        confidence=confidence,
        answerable=answerable,
    )


def _reported_holding_events(
    resolution: HoldingResolution,
) -> tuple[HoldingEvent, ...]:
    """Report the events the question actually asked about.

    The resolver already decides which events satisfy the question's company,
    reporter, direction, and date constraints.  Reporting every retrieved event
    regardless answers a question nobody asked: given "2022-12-05 기준 보유
    비율", the reader also gets 2023 and 2024 holdings, and given "감소 후
    주식수" the reader is shown increases.

    So every event that satisfies the question is reported, and nothing else.
    That is one event when the question identifies one and several when it does
    not; which of several the reader wanted is a question this function does not
    answer, and no event is ever picked for ranking first, or for being the
    newest or the closest.  A history question names no field and keeps its
    whole timeline, and when nothing satisfies the question the existing
    fallback still shows what was retrieved rather than nothing at all.
    """

    events = tuple(resolution.events)
    if not resolution.requested_fields:
        return events
    matching = tuple(event for event in events if event.matches_query is True)
    reporter_constraint = getattr(resolution, "reporter_constraint", None)
    if (
        reporter_constraint
        and not matching
        and not any(
            _holding_reporter_matches(event.reporter, reporter_constraint)
            for event in events
        )
    ):
        return ()
    if not matching:
        return events
    return matching


def _holding_reporter_matches(value: str | None, constraint: str | None) -> bool:
    """The same contract the resolver applies, so the two cannot disagree."""

    return reporter_matches(value, constraint)


def _holding_event_content(event: HoldingEvent) -> dict[str, Any]:
    serialized = event.to_dict()
    return {
        "corp_name": event.corp_name,
        "reporter": event.reporter,
        "reference_date": event.reference_date,
        "report_date": event.report_date,
        "receipt_date": event.receipt_date,
        "before_shares": serialized["before_shares"],
        "change_shares": serialized["change_shares"],
        "after_shares": serialized["after_shares"],
        "before_ratio": serialized["before_ratio"],
        "after_ratio": serialized["after_ratio"],
        "change_ratio": serialized["change_ratio"],
        "change_direction": event.change_direction,
        "temporal_match": event.temporal_match,
        "direction_match": event.direction_match,
        "matches_query": event.matches_query,
        "field_conflict": event.field_conflict,
        "conflicting_fields": list(event.conflicting_fields),
        "field_alternatives": {
            field: [value.to_dict() for value in provenance.alternatives]
            for field, provenance in event.field_provenance.items()
            if provenance.alternatives
        },
        "completeness": copy.deepcopy(dict(event.completeness)),
        "confidence": copy.deepcopy(dict(event.confidence)),
        "evidence_chunk_ids": list(event.evidence_chunk_ids),
        # Only for an event whose own row proved an acquisition, so every other
        # event's content keeps exactly the keys it had.
        **(
            {
                "acquisition_date": getattr(event, "acquisition_date", None),
                "acquired_shares": serialized.get("acquired_shares"),
                "transaction_method": getattr(event, "transaction_method", None),
            }
            if getattr(event, "transaction_method", None) is not None
            else {}
        ),
    }


def _periodic_fact_content(
    fact: PeriodicFact, *, request: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    content = {
        "fact": {
            "corp_name": fact.corp_name,
            "fact_type": fact.fact_type,
            "subject": fact.subject,
            "fact_text": fact.fact_text,
            "normalized_fact_text": fact.normalized_fact_text,
            "section_path": list(fact.section_path),
            "repeated_across_periods": fact.repeated_across_periods,
            "reporting_periods": [
                copy.deepcopy(dict(period)) for period in fact.reporting_periods
            ],
            "report_names": list(fact.report_names),
            "doc_ids": list(fact.doc_ids),
            "period_evolution": fact.period_evolution,
            "fact_conflict": fact.fact_conflict,
            "conflict_type": fact.conflict_type,
            "alternatives": [value.to_dict() for value in fact.alternatives],
            "sources": [source.to_dict() for source in fact.sources],
            "completeness": copy.deepcopy(dict(fact.completeness)),
            "confidence": copy.deepcopy(dict(fact.confidence)),
            "evidence_chunk_ids": list(fact.evidence_chunk_ids),
        }
    }
    if request:
        content["request"] = copy.deepcopy(dict(request))
    return content


def _periodic_request(query_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    plan = dict(query_plan or {})
    period = plan.get("period")
    comparison = plan.get("comparison")
    return {
        "metric": plan.get("metric"),
        "basis": plan.get("basis"),
        "period": copy.deepcopy(dict(period)) if isinstance(period, Mapping) else {},
        "comparison": (
            copy.deepcopy(dict(comparison)) if isinstance(comparison, Mapping) else {}
        ),
        "raw_query": plan.get("raw_query"),
    }


class _CitationBuilder:
    def __init__(self, evidence_by_id: Mapping[str, EvidenceItem]) -> None:
        self._evidence_by_id = evidence_by_id
        self._rows: dict[tuple[str, str], dict[str, Any]] = {}

    def add_holding_event(self, event_index: int, event: HoldingEvent) -> None:
        cited_chunks: set[str] = set()
        for field, provenance in event.field_provenance.items():
            for source in provenance.sources:
                cited_chunks.add(source.chunk_id)
                self._add(
                    chunk_id=source.chunk_id,
                    doc_id=source.doc_id,
                    source_refs=source.source_refs,
                    path={
                        "resolver": "holding_event",
                        "event_index": event_index,
                        "field": field,
                        "direct_field_ref": source.direct_field_ref,
                        "derived": source.derived,
                    },
                )
        for chunk_id in event.evidence_chunk_ids:
            if chunk_id in cited_chunks:
                continue
            item = self._evidence_by_id.get(chunk_id)
            if item is None:
                continue
            self._add(
                chunk_id=item.chunk_id,
                doc_id=item.doc_id,
                source_refs=item.source_refs,
                path={
                    "resolver": "holding_event",
                    "event_index": event_index,
                    "field": "event_evidence",
                    "source_chunk_id": item.provenance.get("source_chunk_id"),
                    "source_doc_id": item.provenance.get("source_doc_id"),
                },
            )

    def add_periodic_fact(self, fact_index: int, fact: PeriodicFact) -> None:
        for source in fact.sources:
            source_chunk = source.provenance.get("source_chunk") or {}
            self._add(
                chunk_id=source.chunk_id,
                doc_id=source.doc_id,
                source_refs=source.source_refs,
                path={
                    "resolver": "periodic_fact",
                    "fact_index": fact_index,
                    "evidence_group_id": fact.evidence_group_id,
                    "reporting_period": copy.deepcopy(dict(source.reporting_period)),
                    "source_chunk_id": source.provenance.get("source_chunk_id"),
                    "source_doc_id": source.provenance.get("source_doc_id"),
                    "original_source_chunk_id": source_chunk.get("chunk_id"),
                    "original_source_doc_id": source_chunk.get("doc_id"),
                },
            )

    def _add(
        self,
        *,
        chunk_id: str,
        doc_id: str,
        source_refs: Sequence[Mapping[str, Any]],
        path: Mapping[str, Any],
    ) -> None:
        key = (chunk_id, doc_id)
        row = self._rows.setdefault(
            key,
            {"source_refs": [], "source_ref_keys": set(), "paths": []},
        )
        for ref in source_refs:
            copied = copy.deepcopy(dict(ref))
            ref_key = json.dumps(copied, ensure_ascii=False, sort_keys=True, default=str)
            if ref_key not in row["source_ref_keys"]:
                row["source_ref_keys"].add(ref_key)
                row["source_refs"].append(copied)
        row["paths"].append(copy.deepcopy(dict(path)))

    def build(self) -> tuple[EvidenceCitation, ...]:
        return tuple(
            EvidenceCitation(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source_refs=tuple(row["source_refs"]),
                provenance_path=tuple(row["paths"]),
            )
            for (chunk_id, doc_id), row in self._rows.items()
        )


def _holding_answerability(
    resolution: HoldingResolution,
    citations: Sequence[EvidenceCitation],
) -> tuple[bool, tuple[str, ...]]:
    unresolved = list(resolution.unresolved_fields)
    if not resolution.events:
        unresolved.append("holding_event")
    elif resolution.matching_event_count == 0:
        unresolved.append("matching_holding_event")
    if not citations:
        unresolved.append("evidence_provenance")
    required = resolution.requested_fields
    candidates = [event for event in resolution.events if event.matches_query is True]
    if not candidates:
        candidates = list(resolution.events)
    complete_events = [
        all(_holding_field_supported(event, field) for field in required)
        for event in candidates
    ]
    if required and (not complete_events or not any(complete_events)):
        unresolved.append("complete_requested_holding_fields")
    return not unresolved, tuple(dict.fromkeys(unresolved))


def _holding_field_supported(event: HoldingEvent, field: str) -> bool:
    value = getattr(event, field, None)
    provenance = event.field_provenance.get(field)
    return bool(
        value is not None
        and provenance is not None
        and not provenance.field_conflict
        and provenance.sources
    )


def _periodic_answerability(
    resolution: PeriodicFactResolution,
    citations: Sequence[EvidenceCitation],
) -> tuple[bool, tuple[str, ...]]:
    unresolved = list(resolution.unresolved_requirements)
    if not resolution.facts:
        unresolved.append(resolution.requested_fact or "periodic_fact")
    if not citations:
        unresolved.append("evidence_provenance")
    candidates = list(resolution.facts)
    supported = any(
        bool(fact.sources)
        and bool(fact.fact_text or fact.alternatives)
        and all(
            source.chunk_id
            and source.doc_id
            and source.provenance.get("source_chunk_id") == source.chunk_id
            and source.provenance.get("source_doc_id") == source.doc_id
            for source in fact.sources
        )
        for fact in candidates
    )
    if candidates and not supported:
        unresolved.append("fact_content_with_provenance")
    return not unresolved, tuple(dict.fromkeys(unresolved))


def _holding_confidence(
    resolution: HoldingResolution,
    *,
    answerable: bool,
    citation_count: int,
    unresolved: Sequence[str],
) -> dict[str, Any]:
    ratios = [
        float(event.completeness.get("requested_ratio") or 0.0)
        for event in resolution.events
    ]
    ratio = min(ratios, default=0.0)
    conflict = any(event.field_conflict for event in resolution.events)
    level = "low" if not answerable or conflict else "high" if ratio >= 1.0 else "medium"
    return {
        "level": level,
        "score": round(ratio if answerable else 0.0, 6),
        "answerable": answerable,
        "citation_count": citation_count,
        "unresolved_requirements": list(unresolved),
        "basis": "resolver_completeness_and_provenance",
    }


def _periodic_confidence(
    resolution: PeriodicFactResolution,
    *,
    answerable: bool,
    citation_count: int,
    unresolved: Sequence[str],
) -> dict[str, Any]:
    scores = [
        float(fact.confidence.get("score") or 0.0) for fact in resolution.facts
    ]
    score = min(scores, default=0.0) if answerable else 0.0
    conflict = any(fact.fact_conflict for fact in resolution.facts)
    level = "low" if not answerable or conflict else "high" if score >= 0.9 else "medium"
    return {
        "level": level,
        "score": round(score, 6),
        "answerable": answerable,
        "citation_count": citation_count,
        "unresolved_requirements": list(unresolved),
        "basis": "resolver_completeness_and_provenance",
    }


def _evidence_by_id(evidence_set: EvidenceSet) -> dict[str, EvidenceItem]:
    return {
        item.chunk_id: item
        for group in evidence_set.evidence_groups
        for item in group.items
    }


def _unique_text(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value) for value in values if value is not None and str(value)
        )
    )
