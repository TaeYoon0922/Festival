"""Conservative, read-only reconstruction of periodic disclosure facts."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.reasoning.evidence_builder import EvidenceGroup, EvidenceItem, EvidenceSet


_PERIODIC_GROUP_TYPES = {
    "periodic_repeated_fact",
    "document_evidence",
    "standalone_evidence",
}


@dataclass(frozen=True)
class PeriodicFactSource:
    """One unchanged EvidenceItem supporting a reconstructed periodic fact."""

    chunk_id: str
    doc_id: str
    fact_text: str
    normalized_fact_text: str
    section_path: tuple[str, ...]
    reporting_period: Mapping[str, Any]
    report_name: str | None
    receipt_date: str | None
    temporal_match: bool | None
    retrieval_rank: int
    retrieval_score: float
    source_refs: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "fact_text": self.fact_text,
            "normalized_fact_text": self.normalized_fact_text,
            "section_path": list(self.section_path),
            "reporting_period": copy.deepcopy(dict(self.reporting_period)),
            "report_name": self.report_name,
            "receipt_date": self.receipt_date,
            "temporal_match": self.temporal_match,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": self.retrieval_score,
            "source_refs": copy.deepcopy(list(self.source_refs)),
            "provenance": copy.deepcopy(dict(self.provenance)),
        }


@dataclass(frozen=True)
class PeriodicFactAlternative:
    """A normalized fact representation and every source that supports it."""

    fact_texts: tuple[str, ...]
    normalized_fact_text: str
    reporting_periods: tuple[Mapping[str, Any], ...]
    evidence_chunk_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_texts": list(self.fact_texts),
            "normalized_fact_text": self.normalized_fact_text,
            "reporting_periods": [
                copy.deepcopy(dict(period)) for period in self.reporting_periods
            ],
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
        }


@dataclass(frozen=True)
class PeriodicFact:
    company_id: str | None
    corp_code: str | None
    corp_name: str | None
    fact_type: str
    subject: str | None
    fact_text: str | None
    normalized_fact_text: str | None
    section_path: tuple[str, ...]
    reporting_periods: tuple[Mapping[str, Any], ...]
    report_names: tuple[str, ...]
    doc_ids: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]
    source_refs: tuple[Mapping[str, Any], ...]
    sources: tuple[PeriodicFactSource, ...]
    temporal_matches: tuple[tuple[str, bool | None], ...]
    repeated_across_periods: bool
    supporting_evidence_count: int
    fact_conflict: bool
    conflict_type: str | None
    conflicting_periods: tuple[str, ...]
    period_evolution: bool
    alternatives: tuple[PeriodicFactAlternative, ...]
    completeness: Mapping[str, Any]
    confidence: Mapping[str, Any]
    warnings: tuple[str, ...]
    evidence_group_id: str
    evidence_group_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "corp_code": self.corp_code,
            "corp_name": self.corp_name,
            "fact_type": self.fact_type,
            "subject": self.subject,
            "fact_text": self.fact_text,
            "normalized_fact_text": self.normalized_fact_text,
            "section_path": list(self.section_path),
            "reporting_periods": [
                copy.deepcopy(dict(period)) for period in self.reporting_periods
            ],
            "report_names": list(self.report_names),
            "doc_ids": list(self.doc_ids),
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "source_refs": copy.deepcopy(list(self.source_refs)),
            "fact_provenance": [source.to_dict() for source in self.sources],
            "temporal_matches": dict(self.temporal_matches),
            "repeated_across_periods": self.repeated_across_periods,
            "supporting_evidence_count": self.supporting_evidence_count,
            "conflict": {
                "fact_conflict": self.fact_conflict,
                "conflict_type": self.conflict_type,
                "conflicting_periods": list(self.conflicting_periods),
                "period_evolution": self.period_evolution,
                "alternatives": [value.to_dict() for value in self.alternatives],
            },
            "completeness": copy.deepcopy(dict(self.completeness)),
            "confidence": copy.deepcopy(dict(self.confidence)),
            "warnings": list(self.warnings),
            "evidence_group_id": self.evidence_group_id,
            "evidence_group_type": self.evidence_group_type,
        }


@dataclass(frozen=True)
class PeriodicFactResolution:
    question: str
    task_type: str | None
    requested_fact: str | None
    facts: tuple[PeriodicFact, ...]
    matching_fact_count: int
    temporal_ambiguity: bool
    unresolved_requirements: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "task_type": self.task_type,
            "requested_fact": self.requested_fact,
            "facts": [fact.to_dict() for fact in self.facts],
            "matching_fact_count": self.matching_fact_count,
            "temporal_ambiguity": self.temporal_ambiguity,
            "unresolved_requirements": list(self.unresolved_requirements),
            "warnings": list(self.warnings),
        }


class PeriodicFactResolver:
    """Resolve periodic facts without filtering or reordering retrieval evidence."""

    def resolve(
        self, evidence_set: EvidenceSet, *, query_plan: Any | None = None
    ) -> PeriodicFactResolution:
        return resolve_periodic_facts(evidence_set, query_plan=query_plan)


def resolve_periodic_facts(
    evidence_set: EvidenceSet, *, query_plan: Any | None = None
) -> PeriodicFactResolution:
    plan = _plan_mapping(query_plan or evidence_set.query_plan)
    constraint = _temporal_constraint(evidence_set, plan)
    fact_type = _fact_type(plan, evidence_set.task_type)
    subject = _subject(plan, fact_type)
    requested = _requested_fact(plan, fact_type)
    facts: list[PeriodicFact] = []

    for group in evidence_set.evidence_groups:
        if group.group_type not in _PERIODIC_GROUP_TYPES:
            continue
        periodic_items = tuple(
            item
            for item in group.items
            if item.doc_group == "periodic" and not _is_holding_item(item)
        )
        for items in _partition_by_company(periodic_items):
            facts.append(
                _resolve_group(
                    group,
                    items,
                    fact_type=fact_type,
                    subject=subject,
                    explicit_temporal=bool(constraint.get("explicit")),
                    temporal_constraint=constraint,
                )
            )

    explicit = bool(constraint.get("explicit"))
    matching: list[PeriodicFact]
    if _peer_rate_company_comparison(plan):
        matching = list(facts)
        explicit = False
    elif explicit:
        matching = [fact for fact in facts if _fact_temporal_match(fact, explicit)]
    else:
        matching = list(facts)
    matching_count = len(matching) if explicit else len(facts)
    temporal_ambiguity = _temporal_ambiguity(facts, matching, explicit)
    unresolved: list[str] = []
    if not facts:
        unresolved.append(requested or "periodic_fact")
    if explicit and not matching:
        unresolved.append("explicit_period")
    active = matching if explicit else facts
    if active and all(fact.fact_conflict for fact in active):
        unresolved.append("unconflicted_fact")

    warnings = list(evidence_set.warnings)
    if not facts:
        warnings.append("no_periodic_fact_evidence")
    if explicit and not matching:
        warnings.append("explicit_period_unmatched")
    if temporal_ambiguity:
        warnings.append("multiple_periodic_fact_alternatives")
    if any(fact.fact_conflict for fact in facts):
        warnings.append("same_period_fact_conflict")
    if any(fact.period_evolution for fact in facts):
        warnings.append("period_evolution_preserved")

    return PeriodicFactResolution(
        question=evidence_set.question,
        task_type=evidence_set.task_type or _text(plan.get("task_type")),
        requested_fact=requested,
        facts=tuple(facts),
        matching_fact_count=matching_count,
        temporal_ambiguity=temporal_ambiguity,
        unresolved_requirements=tuple(dict.fromkeys(unresolved)),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _resolve_group(
    group: EvidenceGroup,
    items: Sequence[EvidenceItem],
    *,
    fact_type: str,
    subject: str | None,
    explicit_temporal: bool,
    temporal_constraint: Mapping[str, Any],
) -> PeriodicFact:
    sources = tuple(
        _source(
            item,
            explicit_temporal=explicit_temporal,
            temporal_constraint=temporal_constraint,
        )
        for item in items
    )
    variants = _alternatives(sources)
    periods = _unique_mappings(source.reporting_period for source in sources)
    period_signatures = {
        signature
        for signature in (_period_signature(source.reporting_period) for source in sources)
        if signature
    }
    conflicts = _conflicts(sources)
    normalized_values = {
        source.normalized_fact_text for source in sources if source.normalized_fact_text
    }
    stable_text = sources[0].fact_text if len(normalized_values) == 1 else None
    stable_normalized = next(iter(normalized_values)) if len(normalized_values) == 1 else None
    fact_warnings: list[str] = []
    if conflicts["fact_conflict"]:
        fact_warnings.append("same_period_fact_conflict")
    if conflicts["period_evolution"]:
        fact_warnings.append("period_evolution_preserved")
    source_refs = _unique_mappings(ref for source in sources for ref in source.source_refs)
    primary = items[0]
    completeness = _completeness(
        sources,
        company_identity=primary.company_id or primary.corp_code or primary.corp_name,
        stable_text=stable_text,
        alternatives=variants,
    )
    confidence = _confidence(
        sources,
        fact_conflict=bool(conflicts["fact_conflict"]),
        period_evolution=bool(conflicts["period_evolution"]),
        completeness=completeness,
    )
    return PeriodicFact(
        company_id=primary.company_id,
        corp_code=primary.corp_code,
        corp_name=primary.corp_name,
        fact_type=fact_type,
        subject=subject,
        fact_text=stable_text,
        normalized_fact_text=stable_normalized,
        section_path=primary.section_path,
        reporting_periods=periods,
        report_names=_unique_text(source.report_name for source in sources),
        doc_ids=_unique_text(source.doc_id for source in sources),
        evidence_chunk_ids=tuple(source.chunk_id for source in sources),
        source_refs=source_refs,
        sources=sources,
        temporal_matches=tuple(
            (source.chunk_id, source.temporal_match) for source in sources
        ),
        repeated_across_periods=len(period_signatures) > 1,
        supporting_evidence_count=len(sources),
        fact_conflict=bool(conflicts["fact_conflict"]),
        conflict_type=_text(conflicts.get("conflict_type")),
        conflicting_periods=tuple(conflicts["conflicting_periods"]),
        period_evolution=bool(conflicts["period_evolution"]),
        alternatives=variants,
        completeness=completeness,
        confidence=confidence,
        warnings=tuple(fact_warnings),
        evidence_group_id=group.group_id,
        evidence_group_type=group.group_type,
    )


def _source(
    item: EvidenceItem,
    *,
    explicit_temporal: bool,
    temporal_constraint: Mapping[str, Any],
) -> PeriodicFactSource:
    source_chunk = item.provenance.get("source_chunk") or {}
    text = str(source_chunk.get("content") or item.evidence_text or "").strip()
    temporal_match = (
        item.temporal_match
        if item.temporal_match is not None
        else _period_matches(item, temporal_constraint)
        if explicit_temporal
        else None
    )
    return PeriodicFactSource(
        chunk_id=item.chunk_id,
        doc_id=item.doc_id,
        fact_text=text,
        normalized_fact_text=_normalize_fact(text),
        section_path=item.section_path,
        reporting_period=copy.deepcopy(dict(item.period)),
        report_name=item.report_nm,
        receipt_date=item.rcept_dt,
        temporal_match=temporal_match if explicit_temporal else None,
        retrieval_rank=item.retrieval_rank,
        retrieval_score=item.retrieval_score,
        source_refs=tuple(copy.deepcopy(list(item.source_refs))),
        provenance=copy.deepcopy(dict(item.provenance)),
    )


def _alternatives(
    sources: Sequence[PeriodicFactSource],
) -> tuple[PeriodicFactAlternative, ...]:
    buckets: dict[str, list[PeriodicFactSource]] = {}
    for source in sources:
        buckets.setdefault(source.normalized_fact_text, []).append(source)
    return tuple(
        PeriodicFactAlternative(
            fact_texts=_unique_text(source.fact_text for source in bucket),
            normalized_fact_text=normalized,
            reporting_periods=_unique_mappings(
                source.reporting_period for source in bucket
            ),
            evidence_chunk_ids=tuple(source.chunk_id for source in bucket),
        )
        for normalized, bucket in buckets.items()
    )


def _conflicts(sources: Sequence[PeriodicFactSource]) -> dict[str, Any]:
    by_period: dict[str, set[str]] = {}
    for source in sources:
        signature = _period_signature(source.reporting_period) or "unknown"
        by_period.setdefault(signature, set()).add(source.normalized_fact_text)
    conflicting = tuple(
        signature for signature, values in by_period.items() if len(values) > 1
    )
    all_values = {
        source.normalized_fact_text for source in sources if source.normalized_fact_text
    }
    known_periods = {key for key in by_period if key != "unknown"}
    period_evolution = len(all_values) > 1 and len(known_periods) > 1
    conflict_type = None
    if conflicting:
        conflict_type = (
            "undated_source_conflict"
            if conflicting == ("unknown",)
            else "same_period_source_conflict"
        )
    return {
        "fact_conflict": bool(conflicting),
        "conflict_type": conflict_type,
        "conflicting_periods": conflicting,
        "period_evolution": period_evolution,
    }


def _completeness(
    sources: Sequence[PeriodicFactSource],
    *,
    company_identity: str | None,
    stable_text: str | None,
    alternatives: Sequence[PeriodicFactAlternative],
) -> dict[str, Any]:
    checks = {
        "company_identity": bool(company_identity),
        "fact_content": bool(stable_text or alternatives),
        "evidence_chunk_ids": all(source.chunk_id for source in sources),
        "source_provenance": all(
            source.provenance.get("source_chunk_id") == source.chunk_id
            for source in sources
        ),
    }
    resolved = sum(bool(value) for value in checks.values())
    return {
        "checks": checks,
        "resolved_count": resolved,
        "required_count": len(checks),
        "ratio": resolved / len(checks),
    }


def _confidence(
    sources: Sequence[PeriodicFactSource],
    *,
    fact_conflict: bool,
    period_evolution: bool,
    completeness: Mapping[str, Any],
) -> dict[str, Any]:
    if fact_conflict:
        level, score = "low", 0.4
    elif period_evolution:
        level, score = "medium", 0.75
    else:
        level, score = "high", 1.0
    score *= float(completeness.get("ratio") or 0.0)
    return {
        "level": level,
        "score": round(score, 6),
        "source_count": len(sources),
        "basis": "deterministic_evidence_consistency",
    }


def _partition_by_company(
    items: Sequence[EvidenceItem],
) -> tuple[tuple[EvidenceItem, ...], ...]:
    buckets: dict[str, list[EvidenceItem]] = {}
    for item in items:
        company = item.corp_code or item.company_id or item.corp_name or item.doc_id
        buckets.setdefault(company, []).append(item)
    return tuple(tuple(bucket) for bucket in buckets.values())


def _is_holding_item(item: EvidenceItem) -> bool:
    return bool(
        item.holding.get("projection_type")
        in {"holding_detail_row", "holding_report"}
        or item.holding.get("reporter")
        or item.holding.get("reference_date")
    )


def _fact_type(plan: Mapping[str, Any], task_type: str | None) -> str:
    evidence = plan.get("evidence")
    evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
    return str(
        evidence.get("periodic_intent")
        or task_type
        or plan.get("task_type")
        or plan.get("metric")
        or "periodic_fact"
    )


def _subject(plan: Mapping[str, Any], fact_type: str) -> str | None:
    evidence = plan.get("evidence")
    evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
    return _text(
        evidence.get("periodic_intent_evidence")
        or plan.get("metric")
        or plan.get("event_type")
        or fact_type
    )


def _requested_fact(plan: Mapping[str, Any], fact_type: str) -> str | None:
    return _text(plan.get("metric") or fact_type)


def _peer_rate_company_comparison(plan: Mapping[str, Any]) -> bool:
    evidence = plan.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("derived_metric") != "peer_rate":
        return False
    comparison = plan.get("comparison")
    return isinstance(comparison, Mapping) and comparison.get("type") == "company_comparison"


def _temporal_constraint(
    evidence_set: EvidenceSet, plan: Mapping[str, Any]
) -> dict[str, Any]:
    ambiguity = evidence_set.ambiguity
    existing = ambiguity.get("temporal_constraint")
    if isinstance(existing, Mapping):
        return copy.deepcopy(dict(existing))
    period = plan.get("period")
    if hasattr(period, "to_dict"):
        period = period.to_dict()
    period = dict(period) if isinstance(period, Mapping) else {}
    year = _integer(period.get("year"))
    quarter = _integer(period.get("quarter"))
    from_date = _normalize_date(period.get("from") or period.get("from_date"))
    to_date = _normalize_date(period.get("to") or period.get("to_date"))
    return {
        "explicit": any(value is not None for value in (year, quarter, from_date, to_date)),
        "year": year,
        "quarter": quarter,
        "from_date": from_date,
        "to_date": to_date,
        "period_type": _text(period.get("period_type")),
    }


def _period_matches(
    item: EvidenceItem, constraint: Mapping[str, Any]
) -> bool | None:
    if not constraint.get("explicit"):
        return None
    period_type = str(constraint.get("period_type") or "")
    item_date = (
        _normalize_date(item.rcept_dt)
        if period_type == "receipt_date"
        else _period_date(item.period)
    )
    item_year = _period_year(item.period, item_date)
    if constraint.get("year") is not None:
        if item_year is None:
            return None
        if item_year != constraint["year"]:
            return False
    item_quarter = _period_quarter(item.period, item_date)
    if constraint.get("quarter") is not None:
        if item_quarter is None:
            return None
        if item_quarter != constraint["quarter"]:
            return False
    if constraint.get("from_date") or constraint.get("to_date"):
        if item_date is None:
            return None
        if constraint.get("from_date") and item_date < constraint["from_date"]:
            return False
        if constraint.get("to_date") and item_date > constraint["to_date"]:
            return False
    return True


def _fact_temporal_match(fact: PeriodicFact, explicit: bool) -> bool:
    if not explicit:
        return True
    return any(value is True for _, value in fact.temporal_matches)


def _temporal_ambiguity(
    facts: Sequence[PeriodicFact],
    matching: Sequence[PeriodicFact],
    explicit: bool,
) -> bool:
    if explicit:
        return len(matching) > 1
    if len(facts) < 2:
        return False
    periods = {
        signature
        for fact in facts
        for signature in (_period_signature(period) for period in fact.reporting_periods)
        if signature
    }
    return len(periods) > 1


def _period_signature(period: Mapping[str, Any]) -> str | None:
    if not period:
        return None
    date = _period_date(period)
    year = _period_year(period, date)
    quarter = _period_quarter(period, date)
    period_type = _text(period.get("period_type") or period.get("basis_period"))
    if year is None and quarter is None and not period_type and not date:
        return None
    return f"{year or '?'}:{quarter or '?'}:{_normalize_fact(period_type or '')}:{date or '?'}"


def _period_date(period: Mapping[str, Any]) -> str | None:
    for key in ("to_date", "to", "period_end", "report_period"):
        date = _normalize_date(period.get(key))
        if date:
            return date
    return None


def _period_year(period: Mapping[str, Any], date: str | None) -> int | None:
    for key in ("fiscal_year", "base_year", "year"):
        value = _integer(period.get(key))
        if value is not None:
            return value
    return _integer(date[:4]) if date else None


def _period_quarter(period: Mapping[str, Any], date: str | None) -> int | None:
    quarter = _integer(period.get("quarter"))
    if quarter is not None and 1 <= quarter <= 4:
        return quarter
    if date:
        month = _integer(date[5:7])
        if month:
            return (month - 1) // 3 + 1
    return None


def _normalize_fact(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").casefold())


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(
        r"((?:19|20)\d{2})\s*(?:년|[.\-/])\s*(\d{1,2})\s*"
        r"(?:월|[.\-/])\s*(\d{1,2})(?:\s*일)?",
        str(value),
    )
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _unique_mappings(
    values: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        copied = copy.deepcopy(dict(value))
        key = json.dumps(copied, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            output.append(copied)
    return tuple(output)


def _unique_text(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(str(value) for value in values if value is not None and str(value))
    )


def _plan_mapping(plan: Any) -> dict[str, Any]:
    if hasattr(plan, "to_dict"):
        return copy.deepcopy(dict(plan.to_dict()))
    if isinstance(plan, Mapping):
        return copy.deepcopy(dict(plan))
    raise TypeError("query_plan must be a QueryPlan or mapping")


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None
