"""Deterministic answer-time selection over resolved periodic evidence.

The selector never mutates or reranks retrieval results.  It creates a bounded
view of a ``PeriodicFactResolution`` for answer composition only.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.reasoning.periodic_fact_resolver import (
    PeriodicFact,
    PeriodicFactAlternative,
    PeriodicFactResolution,
    PeriodicFactSource,
)


DEFAULT_MAX_PERIODIC_EVIDENCE = 3
_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
_PERIOD_TOKEN = re.compile(r"^(?:(?:19|20)\d{2}년?|[1-4]분기|\d{1,2}월)$")
_QUERY_STOPWORDS = {
    "무엇인가",
    "무엇인지",
    "얼마인가",
    "알려줘",
    "확인",
    "기준",
    "관련",
    "공시",
    "내용은",
    "보고서",
    "분기보고서",
    "반기보고서",
    "사업보고서",
}
_TABLE_METRIC_TERMS = {
    "가동률",
    "금액",
    "당기순이익",
    "매출",
    "매출액",
    "생산능력",
    "생산량",
    "생산실적",
    "수량",
    "수주액",
    "수주잔고",
    "순이익",
    "영업이익",
    "주식수",
    "판매량",
    "평균가동률",
    "비율",
}

# A product/facility phrase identifies which row the question is asking for,
# while table metrics identify the requested columns.  Both should outweigh a
# small retrieval-rank difference at answer-selection time.
_FOCUS_TERM_BOOST = 10.0
_FOCUS_PHRASE_BOOST = 24.0
_FOCUS_COMPLETE_BOOST = 12.0
_TABLE_METRIC_BOOST = 8.0
_TABLE_METRIC_COMPLETE_BOOST = 16.0


@dataclass(frozen=True)
class PeriodicEvidenceSelection:
    resolution: PeriodicFactResolution
    selected_chunk_ids: tuple[str, ...]
    excluded_chunk_ids: tuple[str, ...]
    max_evidence: int
    warnings: tuple[str, ...]


class PeriodicEvidenceSelector:
    """Select question-aligned periodic sources with a small deterministic cap."""

    def __init__(self, *, max_evidence: int = DEFAULT_MAX_PERIODIC_EVIDENCE) -> None:
        if (
            not isinstance(max_evidence, int)
            or isinstance(max_evidence, bool)
            or max_evidence <= 0
        ):
            raise ValueError("max_evidence must be a positive integer")
        self.max_evidence = max_evidence

    def select(
        self,
        resolution: PeriodicFactResolution,
        *,
        query_plan: Any | None = None,
    ) -> PeriodicEvidenceSelection:
        return select_periodic_evidence(
            resolution,
            query_plan=query_plan,
            max_evidence=self.max_evidence,
        )


def select_periodic_evidence(
    resolution: PeriodicFactResolution,
    *,
    query_plan: Any | None = None,
    max_evidence: int = DEFAULT_MAX_PERIODIC_EVIDENCE,
) -> PeriodicEvidenceSelection:
    """Return a bounded answer view while keeping ``resolution`` unchanged."""

    if (
        not isinstance(max_evidence, int)
        or isinstance(max_evidence, bool)
        or max_evidence <= 0
    ):
        raise ValueError("max_evidence must be a positive integer")
    plan = _plan_mapping(query_plan)
    explicit_period = _explicit_period(plan)
    signals = _query_signals(resolution.question, plan)
    candidates: list[tuple[float, int, str, PeriodicFact, PeriodicFactSource]] = []
    all_sources = [source for fact in resolution.facts for source in fact.sources]
    has_temporal_match = any(source.temporal_match is True for source in all_sources)

    for fact in resolution.facts:
        for source in fact.sources:
            if explicit_period:
                if has_temporal_match and source.temporal_match is not True:
                    continue
                if not has_temporal_match:
                    continue
            score, eligible = _source_relevance(source, fact, signals, plan)
            if eligible:
                candidates.append(
                    (score, source.retrieval_rank, source.chunk_id, fact, source)
                )

    annual_report_preferred = False
    if _is_fiscal_year_query(plan):
        annual_candidates = [row for row in candidates if _is_annual_source(row[4])]
        if annual_candidates:
            candidates = annual_candidates
            annual_report_preferred = True

    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    if candidates:
        seed = candidates[0][4]
        candidates = [
            (
                score
                + (2.0 if source.doc_id == seed.doc_id else 0.0)
                + (
                    2.0
                    if _period_signature(source.reporting_period)
                    == _period_signature(seed.reporting_period)
                    else 0.0
                ),
                rank,
                chunk_id,
                fact,
                source,
            )
            for score, rank, chunk_id, fact, source in candidates
        ]
        candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    selected_rows = candidates[:max_evidence]
    selected_ids = tuple(row[4].chunk_id for row in selected_rows)
    selected_set = set(selected_ids)
    excluded_ids = tuple(
        source.chunk_id for source in all_sources if source.chunk_id not in selected_set
    )

    selected_facts = []
    for fact in resolution.facts:
        sources = tuple(
            source for source in fact.sources if source.chunk_id in selected_set
        )
        if sources:
            selected_facts.append(_rebuild_fact(fact, sources))
    selected_facts.sort(
        key=lambda fact: min(source.retrieval_rank for source in fact.sources)
    )

    warnings = []
    if excluded_ids:
        warnings.append("irrelevant_periodic_evidence_excluded")
    if len(candidates) > max_evidence:
        warnings.append(f"periodic_evidence_limited:max={max_evidence}")
    if annual_report_preferred:
        warnings.append("annual_report_source_preferred")
    if explicit_period and not selected_facts:
        warnings.append("explicit_period_evidence_unmatched")

    selected_resolution = _selected_resolution(
        resolution,
        tuple(selected_facts),
        explicit_period=explicit_period,
        selection_warnings=warnings,
    )
    return PeriodicEvidenceSelection(
        resolution=selected_resolution,
        selected_chunk_ids=selected_ids,
        excluded_chunk_ids=excluded_ids,
        max_evidence=max_evidence,
        warnings=tuple(warnings),
    )


def _source_relevance(
    source: PeriodicFactSource,
    fact: PeriodicFact,
    signals: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> tuple[float, bool]:
    text = " ".join([source.fact_text, *source.section_path])
    normalized = _normalize(text)
    text_tokens = set(_tokens(text))
    core_terms = tuple(signals["core_terms"])
    matched = tuple(term for term in core_terms if _normalize(term) in normalized)
    phrases = tuple(signals["phrases"])
    phrase_matches = tuple(
        phrase for phrase in phrases if _normalize(phrase) in normalized
    )
    strong_phrase_matches = tuple(
        phrase for phrase in phrase_matches if len(_tokens(phrase)) >= 2
    )
    metric = str(plan.get("metric") or "").strip()
    metric_match = bool(metric and _normalize(metric) in normalized)
    subject = str(fact.subject or "").strip()
    subject_match = bool(subject and _normalize(subject) in normalized)
    distinct_subject_match = bool(
        subject_match and (not metric or _normalize(subject) != _normalize(metric))
    )
    focus_terms = tuple(signals["focus_terms"])
    matched_focus_terms = tuple(
        term for term in focus_terms if _normalize(term) in normalized
    )
    focus_phrase = str(signals["focus_phrase"] or "")
    focus_phrase_match = bool(
        focus_phrase and _normalize(focus_phrase) in normalized
    )
    table_metrics = tuple(signals["table_metrics"])
    matched_table_metrics = tuple(
        term for term in table_metrics if _normalize(term) in normalized
    )
    focus_complete = bool(
        focus_terms and len(matched_focus_terms) == len(focus_terms)
    )
    table_metrics_complete = bool(
        table_metrics and len(matched_table_metrics) == len(table_metrics)
    )

    if len(core_terms) <= 1:
        eligible = bool(
            matched
            or phrase_matches
            or metric_match
            or subject_match
            or matched_focus_terms
            or matched_table_metrics
        )
    else:
        non_metric_matches = [
            term for term in matched if not metric or _normalize(term) != _normalize(metric)
        ]
        eligible = bool(
            strong_phrase_matches
            or len(matched) >= 2
            or (metric_match and non_metric_matches)
            or (distinct_subject_match and matched)
            or focus_phrase_match
            or focus_complete
            or (
                table_metrics_complete
                and (not focus_terms or bool(matched_focus_terms))
            )
        )
    score = (
        len(phrase_matches) * 8.0
        + len(matched) * 3.0
        + (5.0 if metric_match else 0.0)
        + (3.0 if subject_match else 0.0)
        + len(matched_focus_terms) * _FOCUS_TERM_BOOST
        + (_FOCUS_PHRASE_BOOST if focus_phrase_match else 0.0)
        + (_FOCUS_COMPLETE_BOOST if focus_complete else 0.0)
        + len(matched_table_metrics) * _TABLE_METRIC_BOOST
        + (
            _TABLE_METRIC_COMPLETE_BOOST
            if table_metrics_complete
            else 0.0
        )
        + (4.0 if source.temporal_match is True else 0.0)
        + min(float(source.retrieval_score), 1.0)
        + (1.0 / max(source.retrieval_rank, 1))
    )
    if any(term in text_tokens for term in core_terms):
        score += 1.0
    return score, eligible


def _query_signals(question: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    evidence = plan.get("evidence")
    evidence = dict(evidence) if isinstance(evidence, Mapping) else {}
    companies = {
        _normalize(value)
        for value in [
            plan.get("company"),
            *(plan.get("companies") or []),
        ]
        if value
    }
    phrases = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in (
                plan.get("lexical_query"),
                plan.get("metric"),
                evidence.get("periodic_intent_evidence"),
            )
            if value and str(value).strip()
        )
    )
    term_source = " ".join([question, *phrases])
    metric = str(plan.get("metric") or "").strip()
    lexical_query = str(plan.get("lexical_query") or question).strip()
    lexical_tokens = _tokens(lexical_query)
    table_metrics = tuple(
        dict.fromkeys(
            token
            for token in lexical_tokens
            if _normalize(token) in _TABLE_METRIC_TERMS
            or (metric and _normalize(token) == _normalize(metric))
        )
    )
    metric_norms = {_normalize(value) for value in table_metrics}
    focus_terms = tuple(
        dict.fromkeys(
            token
            for token in lexical_tokens
            if len(_normalize(token)) >= 2
            and _normalize(token) not in companies
            and _normalize(token) not in metric_norms
            and token not in _QUERY_STOPWORDS
            and not _PERIOD_TOKEN.fullmatch(token)
        )
    )
    core_terms = []
    for token in _tokens(term_source):
        normalized = _normalize(token)
        if (
            len(normalized) < 2
            or normalized in companies
            or token in _QUERY_STOPWORDS
            or _PERIOD_TOKEN.fullmatch(token)
        ):
            continue
        core_terms.append(token)
    return {
        "phrases": phrases,
        "core_terms": tuple(dict.fromkeys(core_terms)),
        "focus_terms": focus_terms,
        "focus_phrase": " ".join(focus_terms),
        "table_metrics": table_metrics,
    }


def _is_fiscal_year_query(plan: Mapping[str, Any]) -> bool:
    period = plan.get("period")
    period = dict(period) if isinstance(period, Mapping) else {}
    year = period.get("year")
    quarter = period.get("quarter")
    period_type = _normalize(period.get("period_type"))
    return bool(
        year is not None
        and quarter is None
        and period_type in {"", "fiscalyear", "year", "annual"}
    )


def _is_annual_source(source: PeriodicFactSource) -> bool:
    report_name = _normalize(source.report_name)
    if report_name:
        if "분기보고서" in report_name or "반기보고서" in report_name:
            return False
        if "사업보고서" in report_name:
            return True
    period_type = _normalize(
        source.reporting_period.get("period_type")
        or source.reporting_period.get("basis_period")
    )
    quarter = source.reporting_period.get("quarter")
    return period_type in {"annual", "year", "fiscalyear"} and quarter in {
        None,
        4,
        "4",
    }


def _rebuild_fact(
    original: PeriodicFact,
    sources: Sequence[PeriodicFactSource],
) -> PeriodicFact:
    ordered = tuple(sorted(sources, key=lambda row: (row.retrieval_rank, row.chunk_id)))
    normalized_values = {
        source.normalized_fact_text for source in ordered if source.normalized_fact_text
    }
    stable_text = ordered[0].fact_text if len(normalized_values) == 1 else None
    stable_normalized = next(iter(normalized_values)) if len(normalized_values) == 1 else None
    alternatives = _alternatives(ordered)
    periods = _unique_mappings(source.reporting_period for source in ordered)
    period_signatures = {
        signature for signature in (_period_signature(period) for period in periods) if signature
    }
    by_period: dict[str, set[str]] = {}
    for source in ordered:
        signature = _period_signature(source.reporting_period) or "unknown"
        by_period.setdefault(signature, set()).add(source.normalized_fact_text)
    conflicting = tuple(key for key, values in by_period.items() if len(values) > 1)
    all_values = {
        source.normalized_fact_text for source in ordered if source.normalized_fact_text
    }
    known_periods = {key for key in by_period if key != "unknown"}
    period_evolution = len(all_values) > 1 and len(known_periods) > 1
    fact_conflict = bool(conflicting)
    warnings = []
    if fact_conflict:
        warnings.append("same_period_fact_conflict")
    if period_evolution:
        warnings.append("period_evolution_preserved")
    confidence = copy.deepcopy(dict(original.confidence))
    if fact_conflict:
        confidence.update({"level": "low", "score": min(float(confidence.get("score") or 0.4), 0.4)})
    source_refs = _unique_mappings(ref for source in ordered for ref in source.source_refs)
    return replace(
        original,
        fact_text=stable_text,
        normalized_fact_text=stable_normalized,
        section_path=ordered[0].section_path,
        reporting_periods=periods,
        report_names=_unique_text(source.report_name for source in ordered),
        doc_ids=_unique_text(source.doc_id for source in ordered),
        evidence_chunk_ids=tuple(source.chunk_id for source in ordered),
        source_refs=source_refs,
        sources=ordered,
        temporal_matches=tuple((source.chunk_id, source.temporal_match) for source in ordered),
        repeated_across_periods=len(period_signatures) > 1,
        supporting_evidence_count=len(ordered),
        fact_conflict=fact_conflict,
        conflict_type=("same_period_source_conflict" if fact_conflict else None),
        conflicting_periods=conflicting,
        period_evolution=period_evolution,
        alternatives=alternatives,
        confidence=confidence,
        warnings=tuple(warnings),
    )


def _selected_resolution(
    original: PeriodicFactResolution,
    facts: tuple[PeriodicFact, ...],
    *,
    explicit_period: bool,
    selection_warnings: Sequence[str],
) -> PeriodicFactResolution:
    recalculated = {
        "no_periodic_fact_evidence",
        "explicit_period_unmatched",
        "multiple_periodic_fact_alternatives",
        "same_period_fact_conflict",
        "period_evolution_preserved",
    }
    warnings = [value for value in original.warnings if value not in recalculated]
    warnings.extend(selection_warnings)
    unresolved = [
        value
        for value in original.unresolved_requirements
        if value not in {original.requested_fact, "periodic_fact", "explicit_period", "unconflicted_fact"}
    ]
    if not facts:
        unresolved.append(original.requested_fact or "periodic_fact")
        warnings.append("no_periodic_fact_evidence")
        if explicit_period:
            unresolved.append("explicit_period")
            warnings.append("explicit_period_unmatched")
    if facts and all(fact.fact_conflict for fact in facts):
        unresolved.append("unconflicted_fact")
        warnings.append("same_period_fact_conflict")
    if any(fact.period_evolution for fact in facts):
        warnings.append("period_evolution_preserved")
    return replace(
        original,
        facts=facts,
        matching_fact_count=len(facts),
        temporal_ambiguity=(
            len({_period_signature(period) for fact in facts for period in fact.reporting_periods}) > 1
            if explicit_period
            else original.temporal_ambiguity
        ),
        unresolved_requirements=tuple(dict.fromkeys(unresolved)),
        warnings=tuple(dict.fromkeys(warnings)),
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
            reporting_periods=_unique_mappings(source.reporting_period for source in bucket),
            evidence_chunk_ids=tuple(source.chunk_id for source in bucket),
        )
        for normalized, bucket in buckets.items()
    )


def _explicit_period(plan: Mapping[str, Any]) -> bool:
    period = plan.get("period")
    period = dict(period) if isinstance(period, Mapping) else {}
    return any(
        period.get(key) is not None
        for key in ("year", "quarter", "from", "to", "from_date", "to_date")
    )


def _period_signature(period: Mapping[str, Any]) -> str | None:
    if not period:
        return None
    values = (
        period.get("fiscal_year") or period.get("base_year") or period.get("year"),
        period.get("quarter"),
        period.get("period_type") or period.get("basis_period"),
        period.get("to_date") or period.get("period_end") or period.get("report_period"),
    )
    return ":".join(str(value or "?") for value in values)


def _unique_mappings(values: Sequence[Mapping[str, Any]] | Any) -> tuple[Mapping[str, Any], ...]:
    output: dict[str, Mapping[str, Any]] = {}
    for value in values:
        copied = copy.deepcopy(dict(value))
        key = json.dumps(copied, ensure_ascii=False, sort_keys=True, default=str)
        output.setdefault(key, copied)
    return tuple(output.values())


def _unique_text(values: Sequence[Any] | Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if value is not None and str(value)))


def _tokens(value: Any) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(str(value or "").casefold()))


def _normalize(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").casefold())


def _plan_mapping(plan: Any | None) -> dict[str, Any]:
    if plan is None:
        return {}
    if hasattr(plan, "to_dict"):
        return copy.deepcopy(dict(plan.to_dict()))
    if isinstance(plan, Mapping):
        return copy.deepcopy(dict(plan))
    raise TypeError("query_plan must be a QueryPlan, mapping, or None")
