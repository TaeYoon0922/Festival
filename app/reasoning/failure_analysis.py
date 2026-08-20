"""Read-only failure analysis for saved PostgreSQL hybrid evaluation reports.

This module deliberately consumes serialized evaluation output.  It does not
construct a retriever, query PostgreSQL, or recalculate relevance metrics.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


FAILURE_ANALYSIS_VERSION = "1"


def analyze_gold60_failures(report: Mapping[str, Any]) -> dict[str, Any]:
    """Analyze Recall@10 misses without mutating or rerunning retrieval."""

    questions = report.get("questions")
    if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)):
        raise ValueError("evaluation report must contain a questions list")

    failures = [
        _analyze_failure(dict(row))
        for row in questions
        if isinstance(row, Mapping) and _is_recall_at_10_miss(row)
    ]
    category_counts = Counter(row["failure_category"] for row in failures)
    contributing_counts = Counter(
        category
        for row in failures
        for category in row["contributing_categories"]
    )
    return {
        "analysis_version": FAILURE_ANALYSIS_VERSION,
        "source": {
            "question_count": int(report.get("question_count") or len(questions)),
            "rerank_mode": _nested(report, "method", "rerank_mode"),
            "top_k": _nested(report, "method", "top_k"),
        },
        "summary": {
            "total_questions": len(questions),
            "recall_at_10_misses": len(failures),
            "category_counts": dict(sorted(category_counts.items())),
            "contributing_category_counts": dict(
                sorted(contributing_counts.items())
            ),
        },
        "failures": failures,
    }


def write_failure_analysis(analysis: Mapping[str, Any], output_dir: Path) -> None:
    """Write stable JSON and Markdown views of an analysis result."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "failure_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "failure_analysis.md").write_text(
        render_failure_analysis_markdown(analysis), encoding="utf-8"
    )


def render_failure_analysis_markdown(analysis: Mapping[str, Any]) -> str:
    summary = _mapping(analysis.get("summary"))
    lines = [
        "# Gold60 Failure Analysis",
        "",
        "## Summary",
        "",
        f"- Total questions: {summary.get('total_questions', 0)}",
        f"- Recall@10 misses: {summary.get('recall_at_10_misses', 0)}",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    counts = _mapping(summary.get("category_counts"))
    if counts:
        lines.extend(f"| {key} | {value} |" for key, value in counts.items())
    else:
        lines.append("| none | 0 |")

    contributing = _mapping(summary.get("contributing_category_counts"))
    if contributing:
        lines.extend(
            [
                "",
                "### Contributing diagnostics",
                "",
                "| Category | Count |",
                "|---|---:|",
                *(
                    f"| {key} | {value} |"
                    for key, value in contributing.items()
                ),
            ]
        )

    for failure_value in analysis.get("failures") or []:
        failure = _mapping(failure_value)
        gold = _mapping(failure.get("gold"))
        retrieval = _mapping(failure.get("retrieval"))
        candidate = _mapping(failure.get("candidate_analysis"))
        lines.extend(
            [
                "",
                f"## {failure.get('question_id') or 'unknown'}",
                "",
                "Question:",
                str(failure.get("question") or ""),
                "",
                "Gold:",
                f"- doc_id: `{gold.get('doc_id')}`",
                f"- chunk_id: `{gold.get('chunk_id')}`",
                f"- section: `{gold.get('section')}`",
                f"- doc_group: `{gold.get('doc_group')}`",
                f"- company_id: `{gold.get('company_id')}`",
                "",
                "Retrieval ranks:",
                f"- lexical: `{retrieval.get('lexical_rank')}`",
                f"- vector: `{retrieval.get('vector_rank')}`",
                f"- fusion: `{retrieval.get('fusion_rank')}`",
                f"- hybrid: `{retrieval.get('hybrid_rank')}`",
                "",
                "Top Retrieval:",
                "",
            ]
        )
        competitors = candidate.get("top_competitors") or []
        if competitors:
            for competitor_value in competitors:
                competitor = _mapping(competitor_value)
                lines.extend(
                    [
                        f"{competitor.get('rank')}. `{competitor.get('chunk_id')}`",
                        f"   - doc_id: `{competitor.get('doc_id')}`",
                        f"   - section: `{competitor.get('section')}`",
                        f"   - relation: `{', '.join(competitor.get('relations') or []) or 'unknown'}`",
                        f"   - reason: {competitor.get('reason')}",
                    ]
                )
        else:
            lines.append("No hybrid Top10 rows were serialized.")
        lines.extend(
            [
                "",
                f"Failure: `{failure.get('failure_category')}`",
                "",
                f"Reason: {failure.get('reason')}",
            ]
        )
        extra = failure.get("contributing_categories") or []
        if extra:
            lines.extend(
                ["", f"Contributing diagnostics: `{', '.join(extra)}`"]
            )
    return "\n".join(lines) + "\n"


def _analyze_failure(row: Mapping[str, Any]) -> dict[str, Any]:
    gold_payload = _mapping(row.get("gold"))
    relevant_chunks = [
        _mapping(value) for value in gold_payload.get("relevant_chunks") or []
    ]
    fusion = _mapping(row.get("gold_fusion_diagnostic"))
    gold_chunk_ids = _gold_chunk_ids(gold_payload, fusion)
    representative = relevant_chunks[0] if relevant_chunks else fusion
    gold_doc_id = _first_text(
        gold_payload.get("doc_id"),
        representative.get("doc_id"),
    )
    gold_section = _section_label(representative) or _first_text(
        gold_payload.get("target_id")
    )
    company_id = _company_id(row, representative)

    lexical_rank = _rank(row.get("lexical_raw_gold_rank"))
    vector_rank = _rank(row.get("vector_gold_rank"))
    fusion_rank = _rank(fusion.get("fusion_rank"))
    reported_hybrid_rank = _rank(row.get("hybrid_gold_rank"))
    diagnostic_final_rank = _rank(fusion.get("final_rank"))
    effective_hybrid_rank = reported_hybrid_rank or diagnostic_final_rank

    source_presence = {
        "lexical": lexical_rank is not None
        or _has_production_gold_match(row.get("lexical_gold_matches")),
        "vector": vector_rank is not None
        or _has_production_gold_match(row.get("vector_gold_matches")),
        "fusion": fusion_rank is not None,
    }
    gold_in_source_pool = any(source_presence.values())
    gold_in_routed_candidates = bool(
        gold_payload.get("candidate_relevant_chunk_ids")
        or row.get("gold_relevant_candidate_count")
    )

    competitors = [
        _competitor_summary(
            item,
            gold_doc_id=gold_doc_id,
            gold_section=gold_section,
            company_id=company_id,
            single_company_scope=_single_company_scope(row),
        )
        for item in (row.get("hybrid_top10") or [])[:10]
        if isinstance(item, Mapping)
    ]
    contributing = _contributing_categories(row, competitors)

    if not gold_in_source_pool:
        failure_category = "candidate_missing"
        reason = (
            "Gold chunk is absent from the serialized production lexical, "
            "vector, and fusion candidate pools."
        )
    else:
        failure_category = "reranking_failure"
        rank_text = (
            str(effective_hybrid_rank)
            if effective_hybrid_rank is not None
            else "outside the serialized Top10"
        )
        reason = (
            "Gold chunk exists in a production retrieval/fusion candidate pool "
            f"but its final hybrid rank is {rank_text}."
        )

    return {
        "question_id": str(row.get("question_id") or ""),
        "question": str(row.get("question") or ""),
        "gold": {
            "chunk_id": gold_chunk_ids[0] if gold_chunk_ids else None,
            "chunk_ids": gold_chunk_ids,
            "doc_id": gold_doc_id,
            "section": gold_section,
            "doc_group": row.get("doc_group"),
            "company_id": company_id,
        },
        "retrieval": {
            "lexical_rank": lexical_rank,
            "lexical_reranked_rank": _rank(row.get("lexical_gold_rank")),
            "vector_rank": vector_rank,
            "fusion_rank": fusion_rank,
            "hybrid_rank": effective_hybrid_rank,
            "hybrid_top10_rank": reported_hybrid_rank,
            "lexical_diagnostic_rank": _rank(
                row.get("lexical_diagnostic_gold_rank")
            ),
            "vector_diagnostic_rank": _rank(row.get("vector_diagnostic_gold_rank")),
        },
        "candidate_analysis": {
            "gold_in_candidate_pool": gold_in_source_pool,
            "gold_in_routed_candidates": gold_in_routed_candidates,
            "source_presence": source_presence,
            "top_competitors": competitors,
        },
        "failure_category": failure_category,
        "contributing_categories": contributing,
        "reason": reason,
    }


def _is_recall_at_10_miss(row: Mapping[str, Any]) -> bool:
    hit = row.get("hit_at_10")
    if hit is not None:
        return not bool(hit)
    rank = _rank(row.get("hybrid_gold_rank"))
    return rank is None or rank > 10


def _gold_chunk_ids(
    gold: Mapping[str, Any], fusion: Mapping[str, Any]
) -> list[str]:
    values: list[Any] = []
    values.extend(gold.get("candidate_relevant_chunk_ids") or [])
    values.extend(
        chunk.get("chunk_id")
        for chunk in gold.get("relevant_chunks") or []
        if isinstance(chunk, Mapping)
    )
    values.append(fusion.get("chunk_id"))
    return list(dict.fromkeys(str(value) for value in values if value))


def _competitor_summary(
    item: Mapping[str, Any],
    *,
    gold_doc_id: str | None,
    gold_section: str | None,
    company_id: str | None,
    single_company_scope: bool,
) -> dict[str, Any]:
    doc_id = _first_text(item.get("doc_id"))
    section = _section_label(item)
    item_company_id = _first_text(
        item.get("company_id"), item.get("corp_code"), item.get("corp_id")
    )
    same_document = bool(gold_doc_id and doc_id and gold_doc_id == doc_id)
    same_company: bool | None
    if same_document or single_company_scope:
        same_company = True
    elif company_id and item_company_id:
        same_company = company_id == item_company_id
    else:
        same_company = None
    relations: list[str] = []
    if same_document:
        relations.append("same_document")
    if same_company:
        relations.append("same_company")
    if gold_section and section:
        relations.append("same_section" if gold_section == section else "different_section")
    if "same_document" in relations and "different_section" in relations:
        relation_reason = "Same document as gold, but a different section ranked higher."
    elif "same_company" in relations:
        relation_reason = "A higher-ranked candidate from the same company scope."
    else:
        relation_reason = "A higher-ranked serialized hybrid candidate."
    return {
        "rank": _rank(item.get("rank")),
        "chunk_id": _first_text(item.get("chunk_id")),
        "doc_id": doc_id,
        "section": section,
        "doc_group": item.get("doc_group"),
        "company_id": item_company_id,
        "report_nm": item.get("report_nm"),
        "retrieval_score": _first_number(
            _nested(item, "hybrid", "retrieval_score"),
            _nested(item, "score_diagnostic", "retrieval_score"),
        ),
        "deterministic_score": _first_number(
            _nested(item, "hybrid", "deterministic_rerank_score"),
            _nested(item, "score_diagnostic", "deterministic_score"),
        ),
        "final_score": _first_number(
            _nested(item, "hybrid", "final_score"), item.get("final_score")
        ),
        "relations": relations,
        "reason": relation_reason,
    }


def _contributing_categories(
    row: Mapping[str, Any], competitors: Sequence[Mapping[str, Any]]
) -> list[str]:
    categories: list[str] = []
    if any(
        "same_document" in (item.get("relations") or [])
        and "different_section" in (item.get("relations") or [])
        for item in competitors
    ):
        categories.append("section_confusion")

    expected_period = _period_signature(_mapping(row.get("query_plan")))
    metric = _first_text(
        _nested(row, "query_plan", "metric"),
        _nested(row, "query_plan", "financial_metric"),
    )
    for item in competitors:
        if "same_company" not in (item.get("relations") or []):
            continue
        competitor_period = _period_signature(item)
        report_nm = str(item.get("report_nm") or "")
        if (
            expected_period
            and competitor_period
            and expected_period != competitor_period
            and (not metric or _normalize(metric) in _normalize(report_nm))
        ):
            categories.append("same_fact_different_period")
            break

    expected_event = _first_text(
        _nested(row, "query_plan", "event_type"), row.get("doc_group")
    )
    if expected_event:
        for item in competitors:
            actual_event = _first_text(item.get("event_type"), item.get("doc_group"))
            if (
                "same_company" in (item.get("relations") or [])
                and actual_event
                and _normalize(actual_event) != _normalize(expected_event)
            ):
                categories.append("same_company_wrong_event")
                break
    return list(dict.fromkeys(categories))


def _single_company_scope(row: Mapping[str, Any]) -> bool:
    hard_filters = _mapping(row.get("hard_filters"))
    for key in ("corp_code", "company_id", "company"):
        value = hard_filters.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value) == 1
        if value:
            return True
    return False


def _company_id(row: Mapping[str, Any], gold: Mapping[str, Any]) -> str | None:
    hard_filters = _mapping(row.get("hard_filters"))
    return _first_text(
        gold.get("company_id"),
        gold.get("corp_code"),
        hard_filters.get("corp_code"),
        hard_filters.get("company_id"),
        _nested(row, "query_plan", "corp_code"),
    )


def _period_signature(value: Mapping[str, Any]) -> str | None:
    period = _mapping(value.get("period"))
    year = _first_text(
        value.get("fiscal_year"),
        value.get("base_year"),
        value.get("year"),
        period.get("year"),
    )
    period_type = _first_text(
        value.get("period_type"),
        value.get("quarter"),
        period.get("period_type"),
        period.get("quarter"),
    )
    report_nm = str(value.get("report_nm") or "")
    if not year:
        match = re.search(r"(?:19|20)\d{2}", report_nm)
        year = match.group(0) if match else None
    if not period_type:
        for marker in ("사업보고서", "반기보고서", "분기보고서", "연간"):
            if marker in report_nm:
                period_type = marker
                break
    if not year and not period_type:
        return None
    return f"{year or '?'}:{_normalize(period_type or '?')}"


def _has_production_gold_match(value: Any) -> bool:
    payload = _mapping(value)
    return bool(payload.get("production"))


def _section_label(value: Mapping[str, Any]) -> str | None:
    path = value.get("section_path")
    if isinstance(path, Sequence) and not isinstance(path, (str, bytes)):
        text = " > ".join(str(item) for item in path if item)
        return text or None
    return _first_text(path, value.get("section"), value.get("section_id"))


def _rank(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _first_number(*values: Any) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 1:
                value = value[0]
            else:
                continue
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _normalize(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value).casefold())
