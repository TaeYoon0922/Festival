"""Gold60 evaluation for metadata-scoped lexical/vector hybrid retrieval."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.reasoning.lexical_evaluation import (
    _candidate_is_relevant,
    _result_summary,
)
from app.reasoning.vector_coverage_policy import (
    assert_complete_coverage,
    claims_real_vectors,
)


ProgressCallback = Callable[[int, int, str], None]
_CUTOFFS = (1, 5, 10)
_EVIDENCE_STOPWORDS = {
    "무엇인가",
    "무엇인가요",
    "알려줘",
    "알려주세요",
    "설명",
    "조회",
}
_KOREAN_PARTICLE_SUFFIXES = (
    "으로부터",
    "에서부터",
    "에게서",
    "에서는",
    "으로",
    "에서",
    "에게",
    "까지",
    "부터",
    "에는",
    "의",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "와",
    "과",
)


class QueryPlanHybridEvaluator:
    """Compare hybrid and same-run lexical-only rankings on frozen gold sets."""

    def __init__(self, understanding: Any, executor: Any, *, top_k: int = 10) -> None:
        if top_k < 10:
            raise ValueError("top_k must be at least 10 for Recall@10 evaluation")
        self.understanding = understanding
        self.executor = executor
        self.top_k = top_k

    def evaluate(
        self,
        question_sets: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        scheduled = [
            (set_name, dict(question))
            for set_name, questions in question_sets.items()
            for question in questions
        ]
        rows: list[dict[str, Any]] = []
        for index, (set_name, question) in enumerate(scheduled, start=1):
            question_id = str(question["question_id"])
            if progress:
                progress(index, len(scheduled), question_id)
            rows.append(self._evaluate_question(set_name, question))

        # Before any metric exists: a run configured for real semantic vectors
        # may not publish numbers it did not earn.  Strictness is derived from
        # the configured provider rather than a flag, so it cannot be omitted
        # by a caller and an intentional hash diagnostic is unaffected.
        self._assert_vector_evaluation_is_honest(rows)

        hybrid = _summary(rows, "hybrid_gold_rank", question_sets)
        lexical = _summary(rows, "lexical_gold_rank", question_sets)
        candidate_union: set[str] = set()
        embedded_union: set[str] = set()
        coverage_available = True
        for row in rows:
            candidate_union.update(row.pop("_candidate_ids"))
            embedded_ids = row.pop("_embedded_candidate_ids")
            if embedded_ids is None:
                coverage_available = False
            else:
                embedded_union.update(embedded_ids)
        vector_coverage = {
            "available": coverage_available,
            "unique_candidate_count": len(candidate_union),
            "embedded_unique_candidate_count": (
                len(embedded_union) if coverage_available else None
            ),
            "ratio": (
                round(len(embedded_union) / len(candidate_union), 6)
                if coverage_available and candidate_union
                else (0.0 if coverage_available else None)
            ),
            "questions_without_embedded_candidates": (
                sum(row["embedded_vector_candidate_count"] == 0 for row in rows)
                if coverage_available
                else None
            ),
        }
        rerank_mode = getattr(
            getattr(self.executor, "config", None), "rerank_mode", "legacy"
        )
        return {
            "method": {
                "flow": (
                    "Question -> QueryUnderstanding -> QueryPlan/Router -> shared "
                    "metadata candidate scope -> lexical Top-N + vector Top-N -> RRF "
                    f"-> {rerank_mode} deterministic rerank -> Top-10; diagnostics "
                    "compare legacy and bounded rankings"
                ),
                "rerank_mode": rerank_mode,
                "rank_semantics": {
                    "lexical_raw_gold_rank": "raw production lexical Top-N",
                    "lexical_gold_rank": "deterministically reranked lexical Top-10",
                    "vector_gold_rank": "raw production vector Top-N",
                    "vector_top10_gold_rank": "raw vector Top-10 cutoff",
                    "hybrid_gold_rank": "final hybrid Top-10",
                    "vector_diagnostic_gold_rank": "opt-in diagnostic vector Top-N",
                },
                "gold_judgment": "app.parsing.final_validation._is_relevant",
                "top_k": self.top_k,
            },
            "question_count": len(rows),
            "hybrid": hybrid,
            "lexical_only": lexical,
            "improvement": _summary_delta(hybrid, lexical),
            "vector_coverage": vector_coverage,
            "failure_counts": dict(
                sorted(Counter(row["failure_class"] for row in rows).items())
            ),
            "vector_status_counts": dict(
                sorted(Counter(row["vector_status"] for row in rows).items())
            ),
            "failures": [row for row in rows if row["failure_class"] != "success"],
            "questions": rows,
        }

    def _assert_vector_evaluation_is_honest(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> None:
        """Refuse to compute metrics for a real-vector run without the vectors.

        Lexical fallback and partially covered hybrid runs both produce numbers
        that look like retrieval quality but are not: under partial coverage
        only embedded chunks can reach the vector lane, so the comparison is
        asymmetric while ``vector_status`` still reads ``ok``.
        """

        config = getattr(self.executor, "embedding_config", None)
        if config is None or not claims_real_vectors(getattr(config, "provider", None)):
            return
        assert_complete_coverage(
            {
                str(row["question_id"]): {
                    "available": row["embedded_vector_candidate_count"] is not None,
                    "candidate_count": row["candidate_chunk_count"],
                    "embedded_count": row["embedded_vector_candidate_count"] or 0,
                }
                for row in rows
            },
            identity={
                "provider": getattr(config, "provider", None),
                "model": getattr(config, "model", None),
                "version": getattr(config, "version", None),
                "dimensions": getattr(config, "dimensions", None),
            },
        )

    def _evaluate_question(
        self, set_name: str, question: Mapping[str, Any]
    ) -> dict[str, Any]:
        raw_query = str(question["query"])
        plan = self.understanding.understand(raw_query, top_k=self.top_k)
        evidence_profile = _query_evidence_profile(plan)
        execution = self.executor.execute(plan)
        chunks_by_id = {candidate.chunk_id: candidate.chunk for candidate in execution.chunks}
        candidate_doc_ids = {document.doc_id for document in execution.documents}
        gold_doc_id = str(question["doc_id"])
        relevant_candidate_chunks = [
            candidate
            for candidate in execution.chunks
            if _candidate_is_relevant(candidate.chunk, candidate.doc_id, question)
        ]
        relevant_candidates = [
            candidate.chunk_id for candidate in relevant_candidate_chunks
        ]

        lexical_raw_rank = _gold_rank(
            execution.lexical_results, chunks_by_id, question, limit=None
        )
        lexical_rank = _gold_rank(
            execution.lexical_final_results, chunks_by_id, question
        )
        vector_rank = _gold_rank(
            execution.vector_results, chunks_by_id, question, limit=None
        )
        vector_top10_rank = _gold_rank(
            execution.vector_results, chunks_by_id, question
        )
        hybrid_rank = _gold_rank(execution.results, chunks_by_id, question)
        lexical_diagnostic_rank = _gold_rank(
            execution.diagnostic_lexical_results,
            chunks_by_id,
            question,
            limit=None,
        )
        vector_diagnostic_rank = _gold_rank(
            execution.diagnostic_vector_results,
            chunks_by_id,
            question,
            limit=None,
        )
        gold_fusion_diagnostic = _gold_fusion_diagnostic(
            execution.rerank_diagnostics,
            chunks_by_id,
            question,
        )
        if gold_fusion_diagnostic is not None:
            gold_chunk = chunks_by_id.get(gold_fusion_diagnostic["chunk_id"], {})
            gold_fusion_diagnostic = _enrich_fusion_diagnostic(
                gold_fusion_diagnostic,
                gold_chunk,
                evidence_profile,
                task_type=plan.task_type,
            )
        hybrid_top10 = _hybrid_summaries(
            execution.results,
            chunks_by_id,
            question,
            evidence_profile=evidence_profile,
            task_type=plan.task_type,
        )

        if gold_doc_id not in candidate_doc_ids:
            failure_class = "metadata_filter_failure"
        elif not relevant_candidates:
            failure_class = "gold_mapping_or_route_failure"
        elif hybrid_rank is not None:
            failure_class = "success"
        elif lexical_rank is None and vector_rank is None:
            failure_class = "vector_missing_failure"
        else:
            failure_class = "fusion_ranking_failure"

        routing = dict(execution.routing)
        coverage = dict(execution.vector_coverage or {})
        embedded_count = coverage.get("embedded_count")
        return {
            "question_id": str(question["question_id"]),
            "evaluation_set": set_name,
            "doc_group": str(question.get("doc_group") or "unknown"),
            "task_type": plan.task_type or "unknown",
            "question": raw_query,
            "query_plan": plan.to_dict(),
            "lexical_query": plan.lexical_query,
            "hard_filters": dict(routing.get("hard_filters") or {}),
            "hard_routes": dict(routing.get("hard_routes") or {}),
            "soft_boosts": dict(routing.get("soft_boosts") or {}),
            "hybrid_config": dict(routing.get("hybrid") or {}),
            "gold": {
                "doc_id": gold_doc_id,
                "target_type": question["target_type"],
                "target_id": question["target_id"],
                "evidence_terms": list(question.get("evidence_terms") or []),
                "candidate_relevant_chunk_ids": relevant_candidates,
                "relevant_chunks": [
                    _gold_chunk_summary(
                        chunk_id=candidate.chunk_id,
                        doc_id=candidate.doc_id,
                        chunk=candidate.chunk,
                        evidence_profile=evidence_profile,
                        task_type=plan.task_type,
                    )
                    for candidate in relevant_candidate_chunks
                ],
            },
            "candidate_document_count": len(execution.documents),
            "candidate_chunk_count": len(execution.chunks),
            "embedded_vector_candidate_count": embedded_count,
            "vector_candidate_coverage": coverage.get("ratio"),
            "has_any_vector_candidate": (
                bool(embedded_count) if coverage.get("available") else None
            ),
            "gold_document_in_candidates": gold_doc_id in candidate_doc_ids,
            "gold_relevant_candidate_count": len(relevant_candidates),
            "lexical_raw_gold_rank": lexical_raw_rank,
            "lexical_gold_rank": lexical_rank,
            "vector_gold_rank": vector_rank,
            "vector_top10_gold_rank": vector_top10_rank,
            "hybrid_gold_rank": hybrid_rank,
            "lexical_diagnostic_gold_rank": lexical_diagnostic_rank,
            "vector_diagnostic_gold_rank": vector_diagnostic_rank,
            "diagnostic_source_top_n": {
                "lexical": len(execution.diagnostic_lexical_results),
                "vector": len(execution.diagnostic_vector_results),
            },
            "gold_fusion_diagnostic": gold_fusion_diagnostic,
            "lexical_gold_matches": {
                "production": _lexical_gold_matches(
                    execution.lexical_results, chunks_by_id, question
                ),
                "diagnostic": _lexical_gold_matches(
                    execution.diagnostic_lexical_results, chunks_by_id, question
                ),
            },
            "query_evidence_diagnostic": evidence_profile,
            "score_component_comparison": _score_component_comparison(
                gold_fusion_diagnostic, hybrid_top10
            ),
            "lexical_top10": _retrieval_summaries(
                execution.lexical_final_results, chunks_by_id, question
            ),
            "vector_top10": _vector_summaries(
                execution.vector_results, chunks_by_id, question
            ),
            "vector_production_top_n": _vector_summaries(
                execution.vector_results,
                chunks_by_id,
                question,
                limit=None,
            ),
            "vector_gold_matches": {
                "production": _vector_gold_matches(
                    execution.vector_results, chunks_by_id, question
                ),
                "diagnostic": _vector_gold_matches(
                    execution.diagnostic_vector_results, chunks_by_id, question
                ),
            },
            "vector_rank_diagnostic": _vector_rank_diagnostic(
                execution.vector_results,
                execution.diagnostic_vector_results,
                chunks_by_id,
                question,
            ),
            "hybrid_top10": hybrid_top10,
            "section_path_diagnostic": _section_path_diagnostic(
                relevant_candidate_chunks,
                lexical_results=execution.lexical_final_results,
                vector_results=execution.vector_results,
                hybrid_results=execution.results,
                chunks_by_id=chunks_by_id,
            ),
            "vector_status": execution.vector_status,
            "vector_error": execution.vector_error,
            "hit_at_1": bool(hybrid_rank and hybrid_rank <= 1),
            "hit_at_5": bool(hybrid_rank and hybrid_rank <= 5),
            "hit_at_10": bool(hybrid_rank and hybrid_rank <= 10),
            "failure_class": failure_class,
            "_candidate_ids": [candidate.chunk_id for candidate in execution.chunks],
            "_embedded_candidate_ids": (
                list(execution.embedded_candidate_ids)
                if coverage.get("available")
                else None
            ),
        }


def write_hybrid_evaluation_report(
    report: Mapping[str, Any], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "postgres_hybrid_gold60.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "postgres_hybrid_gold60.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    _write_question_csv(
        report["questions"], output_dir / "postgres_hybrid_questions.csv"
    )


def compare_hybrid_evaluation_reports(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two complete reports without encoding question-specific rules."""

    baseline_hybrid = baseline.get("hybrid") or {}
    current_hybrid = current.get("hybrid") or {}
    metric_comparison = {
        "overall": _metric_comparison(
            baseline_hybrid.get("overall") or {},
            current_hybrid.get("overall") or {},
        ),
        "by_evaluation_set": _group_metric_comparison(
            baseline_hybrid.get("by_evaluation_set") or {},
            current_hybrid.get("by_evaluation_set") or {},
        ),
        "by_doc_group": _group_metric_comparison(
            baseline_hybrid.get("by_doc_group") or {},
            current_hybrid.get("by_doc_group") or {},
        ),
    }
    baseline_rows = {
        str(row.get("question_id")): row for row in baseline.get("questions") or []
    }
    current_rows = {
        str(row.get("question_id")): row for row in current.get("questions") or []
    }
    rank_changes = []
    for question_id in sorted(set(baseline_rows).intersection(current_rows)):
        before = baseline_rows[question_id]
        after = current_rows[question_id]
        rank_changes.append(
            {
                "question_id": question_id,
                "doc_group": after.get("doc_group"),
                "periodic_intent": (
                    (after.get("query_plan") or {}).get("evidence") or {}
                ).get("periodic_intent"),
                "lexical_rank": {
                    "before": before.get("lexical_gold_rank"),
                    "after": after.get("lexical_gold_rank"),
                },
                "vector_rank": {
                    "before": before.get("vector_gold_rank"),
                    "after": after.get("vector_gold_rank"),
                },
                "hybrid_rank": {
                    "before": before.get("hybrid_gold_rank"),
                    "after": after.get("hybrid_gold_rank"),
                },
            }
        )
    new_failures = [
        row["question_id"]
        for row in rank_changes
        if row["hybrid_rank"]["before"] is not None
        and row["hybrid_rank"]["after"] is None
    ]
    recovered = [
        row["question_id"]
        for row in rank_changes
        if row["hybrid_rank"]["before"] is None
        and row["hybrid_rank"]["after"] is not None
    ]
    return {
        "metrics": metric_comparison,
        "question_rank_changes": rank_changes,
        "new_failures": new_failures,
        "recovered": recovered,
    }


def _group_metric_comparison(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        name: _metric_comparison(baseline.get(name) or {}, current.get(name) or {})
        for name in sorted(set(baseline).union(current))
    }


def _metric_comparison(
    baseline: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, Any]:
    fields = ("recall_at_1", "recall_at_5", "recall_at_10")
    return {
        "baseline": {field: baseline.get(field) for field in fields},
        "current": {field: current.get(field) for field in fields},
        "delta": {
            field: (
                round(float(current[field]) - float(baseline[field]), 6)
                if current.get(field) is not None and baseline.get(field) is not None
                else None
            )
            for field in fields
        },
    }


def _gold_rank(
    results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
    *,
    limit: int | None = 10,
) -> int | None:
    selected = results if limit is None else results[:limit]
    for result in selected:
        if _candidate_is_relevant(
            chunks_by_id.get(result.chunk_id, {}), result.doc_id, question
        ):
            return int(result.rank)
    return None


def _retrieval_summaries(
    results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for result in results[:10]:
        chunk = chunks_by_id.get(result.chunk_id, {})
        relevant = _candidate_is_relevant(chunk, result.doc_id, question)
        summary = _result_summary(result, chunk, relevant)
        summary.update(_diagnostic_identity(result.chunk_id, result.doc_id))
        output.append(summary)
    return output


def _lexical_gold_matches(
    results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for result in results:
        chunk = chunks_by_id.get(result.chunk_id, {})
        if not _candidate_is_relevant(chunk, result.doc_id, question):
            continue
        matches.append(
            {
                **_diagnostic_identity(result.chunk_id, result.doc_id),
                "rank": int(result.rank),
                "bm25_score": float(result.bm25_score),
                "section_path": _section_path_values(chunk),
            }
        )
    return matches


def _vector_summaries(
    results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
    *,
    limit: int | None = 10,
) -> list[dict[str, Any]]:
    output = []
    selected = results if limit is None else results[:limit]
    for result in selected:
        chunk = chunks_by_id.get(result.chunk_id, {})
        section_path = chunk.get("section_path") or []
        if isinstance(section_path, str):
            section_path = [section_path]
        output.append(
            {
                **_diagnostic_identity(result.chunk_id, result.doc_id),
                "rank": result.rank,
                "vector_score": float(result.vector_score),
                "section_path": list(section_path),
                "chunk_type": chunk.get("chunk_type"),
                "is_gold_relevant": _candidate_is_relevant(
                    chunk, result.doc_id, question
                ),
            }
        )
    return output


def _vector_gold_matches(
    results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for result in results:
        chunk = chunks_by_id.get(result.chunk_id, {})
        if not _candidate_is_relevant(chunk, result.doc_id, question):
            continue
        section_path = chunk.get("section_path") or []
        if isinstance(section_path, str):
            section_path = [section_path]
        matches.append(
            {
                **_diagnostic_identity(result.chunk_id, result.doc_id),
                "rank": int(result.rank),
                "vector_score": float(result.vector_score),
                "section_path": list(section_path),
            }
        )
    return matches


def _vector_rank_diagnostic(
    production_results: Sequence[Any],
    diagnostic_results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
) -> dict[str, Any]:
    production = _vector_gold_matches(
        production_results, chunks_by_id, question
    )
    diagnostic = _vector_gold_matches(
        diagnostic_results, chunks_by_id, question
    )
    diagnostic_rank = diagnostic[0]["rank"] if diagnostic else None
    if production:
        reason = "consistent_in_production_top_n"
    elif diagnostic_rank is not None and diagnostic_rank > len(production_results):
        reason = "outside_production_top_n"
    elif diagnostic:
        reason = "inconsistent_production_and_diagnostic_results"
    else:
        reason = "gold_not_returned_by_vector_search"
    return {
        "production_top_n": len(production_results),
        "diagnostic_top_n": len(diagnostic_results),
        "production_contains_gold": bool(production),
        "production_raw_rank": production[0]["rank"] if production else None,
        "diagnostic_rank": diagnostic_rank,
        "reason": reason,
        "same_query_filter_execution": True,
        "section_boost_applied_to_raw_vector": False,
        "rank_level": "chunk",
    }


def _query_evidence_profile(plan: Any) -> dict[str, Any]:
    lexical_query = str(plan.lexical_query or "").strip()
    raw_tokens = re.findall(r"[0-9A-Za-z]+(?:[-_.][0-9A-Za-z]+)*|[가-힣]+", lexical_query)
    core_terms: list[str] = []
    for raw_token in raw_tokens:
        token = _strip_korean_particle(raw_token)
        if not token or token.casefold() in _EVIDENCE_STOPWORDS:
            continue
        if token not in core_terms:
            core_terms.append(token)
    requested_holding_fields = (
        _requested_holding_fields(lexical_query)
        if plan.task_type == "holding_change"
        else []
    )
    return {
        "lexical_query": lexical_query,
        "whitespace_normalized_query": _whitespace_normalize(lexical_query),
        "compact_normalized_query": _alignment_normalize(lexical_query),
        "core_terms": core_terms,
        "requested_holding_fields": requested_holding_fields,
        "reporter": plan.reporter,
        "metric": plan.metric,
        "event_type": plan.event_type,
        "periodic_intent": plan.evidence.get("periodic_intent"),
    }


def _query_alignment(
    profile: Mapping[str, Any], chunk: Mapping[str, Any]
) -> dict[str, Any]:
    text = " ".join(
        str(value or "")
        for value in (chunk.get("retrieval_text"), chunk.get("content"))
    )
    whitespace_text = _whitespace_normalize(text)
    compact_text = _alignment_normalize(text)
    query = str(profile.get("lexical_query") or "")
    core_terms = [str(term) for term in profile.get("core_terms") or []]
    matched_terms = [
        term for term in core_terms if _alignment_normalize(term) in compact_text
    ]
    return {
        "exact_phrase_match": bool(
            query and _whitespace_normalize(query) in whitespace_text
        ),
        "normalized_phrase_match": bool(
            query and _alignment_normalize(query) in compact_text
        ),
        "matched_core_terms": matched_terms,
        "missing_core_terms": [term for term in core_terms if term not in matched_terms],
        "core_term_coverage_ratio": (
            round(len(matched_terms) / len(core_terms), 6) if core_terms else 0.0
        ),
    }


def _holding_structure(
    chunk: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    projection_fields = dict(chunk.get("projection_fields") or {})
    available_fields = set(projection_fields)
    if not available_fields:
        content = str(chunk.get("content") or chunk.get("retrieval_text") or "")
        for canonical, patterns in _HOLDING_FIELD_PATTERNS.items():
            if any(re.search(pattern, content) for pattern in patterns):
                available_fields.add(canonical)
    requested_fields = [
        str(field) for field in profile.get("requested_holding_fields") or []
    ]
    matched_fields = [field for field in requested_fields if field in available_fields]
    change_value = str(projection_fields.get("증감주식수") or "").strip()
    change_direction = None
    if change_value:
        change_direction = "decrease" if change_value.startswith("-") else "increase"
    return {
        "chunk_type": chunk.get("chunk_type"),
        "is_projection": chunk.get("chunk_type") == "table_projection",
        "projection_type": chunk.get("projection_type"),
        "projection_state": chunk.get("projection_state"),
        "projection_fields": projection_fields,
        "projection_field_refs": dict(chunk.get("projection_field_refs") or {}),
        "source_table_ids": list(chunk.get("source_table_ids") or []),
        "source_refs": list(chunk.get("source_refs") or []),
        "reporter": (
            projection_fields.get("보고자/보유자")
            or chunk.get("reporter")
        ),
        "reference_date": projection_fields.get("기준일/보고일"),
        "before_shares": projection_fields.get("직전 보유주식수"),
        "change_shares": projection_fields.get("증감주식수"),
        "after_shares": projection_fields.get("보유주식수"),
        "holding_ratio": projection_fields.get("보유비율"),
        "change_ratio": projection_fields.get("증감비율"),
        "change_direction": change_direction,
        "requested_fields": requested_fields,
        "available_fields": sorted(available_fields),
        "matched_fields": matched_fields,
        "field_alignment_ratio": (
            round(len(matched_fields) / len(requested_fields), 6)
            if requested_fields
            else 0.0
        ),
    }


_HOLDING_FIELD_PATTERNS: dict[str, tuple[str, ...]] = {
    "보고자/보유자": (r"보고자", r"보유자", r"성명\s*\(명칭\)"),
    "기준일/보고일": (r"변동\s*일", r"기준\s*일", r"보고\s*일"),
    "직전 보유주식수": (r"변동\s*전", r"직전.{0,8}주식\s*수"),
    "증감주식수": (r"증감.{0,6}주식\s*수", r"변동\s*내역\s*/\s*증감"),
    "보유주식수": (r"변동\s*후", r"보유\s*주식\s*수"),
    "보유비율": (r"보유\s*비율", r"지분\s*율"),
    "증감비율": (r"증감\s*비율",),
}


def _requested_holding_fields(query: str) -> list[str]:
    fields: list[str] = []
    patterns = {
        "보고자/보유자": (r"국민연금", r"보고자", r"보유자"),
        "기준일/보고일": (r"변동\s*일", r"기준\s*일", r"보고\s*일"),
        "직전 보유주식수": (r"변동\s*전", r"직전.{0,8}주식\s*수"),
        "증감주식수": (r"증감", r"증가", r"감소"),
        "보유주식수": (
            r"변동\s*후",
            r"(?:증가|감소)\s*후\s*주식\s*수",
            r"보유\s*주식\s*수",
        ),
        "보유비율": (r"비율", r"지분\s*율"),
    }
    for field, field_patterns in patterns.items():
        if any(re.search(pattern, query) for pattern in field_patterns):
            fields.append(field)
    return fields


def _strip_korean_particle(token: str) -> str:
    if not re.fullmatch(r"[가-힣]+", token):
        return token
    for suffix in _KOREAN_PARTICLE_SUFFIXES:
        if token.endswith(suffix) and len(token) > len(suffix) + 1:
            return token[: -len(suffix)]
    return token


def _whitespace_normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _alignment_normalize(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(value or "")).casefold()


def _gold_chunk_summary(
    *,
    chunk_id: str,
    doc_id: str,
    chunk: Mapping[str, Any],
    evidence_profile: Mapping[str, Any] | None = None,
    task_type: str | None = None,
) -> dict[str, Any]:
    section_path = chunk.get("section_path") or []
    if isinstance(section_path, str):
        section_path = [section_path]
    preview = " ".join(str(chunk.get("retrieval_text") or "").split())[:500]
    summary = {
        **_diagnostic_identity(chunk_id, doc_id),
        "chunk_type": chunk.get("chunk_type"),
        "section_path": list(section_path),
        "retrieval_text_preview": preview,
    }
    if evidence_profile is not None:
        summary["query_alignment"] = _query_alignment(evidence_profile, chunk)
    if task_type == "holding_change":
        summary["holding_structure"] = _holding_structure(
            chunk, evidence_profile or {}
        )
    return summary


def _section_path_diagnostic(
    gold_candidates: Sequence[Any],
    *,
    lexical_results: Sequence[Any],
    vector_results: Sequence[Any],
    hybrid_results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gold_sections = sorted(
        {
            _section_path_label(candidate.chunk)
            for candidate in gold_candidates
            if _section_path_label(candidate.chunk)
        }
    )
    distributions = {
        "lexical_top10": _section_path_distribution(
            lexical_results, chunks_by_id
        ),
        "vector_top10": _section_path_distribution(vector_results, chunks_by_id),
        "hybrid_top10": _section_path_distribution(hybrid_results, chunks_by_id),
    }
    dominant_gap: dict[str, Any] = {}
    for source, distribution in distributions.items():
        dominant = distribution[0] if distribution else None
        dominant_path = dominant["section_path"] if dominant else None
        dominant_gap[source] = {
            "dominant_section_path": dominant_path,
            "dominant_count": dominant["count"] if dominant else 0,
            "matches_gold_section": dominant_path in gold_sections,
            "gold_section_paths": gold_sections,
        }
    return {
        "gold_section_paths": gold_sections,
        **distributions,
        "dominant_gap": dominant_gap,
    }


def _section_path_distribution(
    results: Sequence[Any], chunks_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter(
        _section_path_label(chunks_by_id.get(result.chunk_id, {}))
        or "<missing-section>"
        for result in results[:10]
    )
    return [
        {"section_path": section_path, "count": count}
        for section_path, count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def _section_path_label(chunk: Mapping[str, Any]) -> str:
    section_path = chunk.get("section_path") or []
    if isinstance(section_path, str):
        return section_path.strip()
    return " > ".join(str(value).strip() for value in section_path if str(value).strip())


def _hybrid_summaries(
    results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
    *,
    evidence_profile: Mapping[str, Any] | None = None,
    task_type: str | None = None,
) -> list[dict[str, Any]]:
    output = _retrieval_summaries(results, chunks_by_id, question)
    for item, result in zip(output, results[:10]):
        chunk = chunks_by_id.get(result.chunk_id, {})
        hybrid = dict(result.metadata_match.get("hybrid") or {})
        item["hybrid"] = hybrid
        item["score_diagnostic"] = _score_diagnostic(
            hybrid,
            dict(result.metadata_match.get("score_components") or {}),
        )
        if evidence_profile is not None:
            item["query_alignment"] = _query_alignment(evidence_profile, chunk)
        if task_type == "holding_change":
            item["holding_structure"] = _holding_structure(
                chunk, evidence_profile or {}
            )
    return output


def _enrich_fusion_diagnostic(
    diagnostic: Mapping[str, Any],
    chunk: Mapping[str, Any],
    evidence_profile: Mapping[str, Any],
    *,
    task_type: str | None,
) -> dict[str, Any]:
    enriched = dict(diagnostic)
    section_path = chunk.get("section_path") or []
    if isinstance(section_path, str):
        section_path = [section_path]
    enriched.update(
        {
            **_diagnostic_identity(
                str(diagnostic.get("chunk_id") or ""),
                str(diagnostic.get("doc_id") or ""),
            ),
            "chunk_type": chunk.get("chunk_type"),
            "section_path": list(section_path),
            "retrieval_text_preview": " ".join(
                str(chunk.get("retrieval_text") or "").split()
            )[:500],
            "query_alignment": _query_alignment(evidence_profile, chunk),
        }
    )
    if task_type == "holding_change":
        enriched["holding_structure"] = _holding_structure(
            chunk, evidence_profile
        )
    return enriched


def _diagnostic_identity(chunk_id: Any, doc_id: Any) -> dict[str, str]:
    """Serialize identity from typed retrieval objects, never chunk payload aliases."""

    return {"chunk_id": str(chunk_id), "doc_id": str(doc_id)}


def _section_path_values(chunk: Mapping[str, Any]) -> list[str]:
    section_path = chunk.get("section_path") or []
    if isinstance(section_path, str):
        return [section_path]
    return [str(value) for value in section_path]


def _score_diagnostic(
    hybrid: Mapping[str, Any], components: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "lexical_rank": hybrid.get("lexical_rank"),
        "vector_rank": hybrid.get("vector_rank"),
        "vector_score": hybrid.get("vector_score"),
        "rrf_score": hybrid.get("rrf_score"),
        "normalized_rrf_score": hybrid.get("normalized_rrf_score"),
        "deterministic_rerank_score": hybrid.get("deterministic_rerank_score"),
        "final_score": hybrid.get("final_score"),
        "final_rank": hybrid.get("final_rank"),
        "components": {
            key: float(value)
            for key, value in components.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        },
    }


def _score_component_comparison(
    gold: Mapping[str, Any] | None,
    hybrid_top10: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if gold is None:
        return None
    top_score_rows = [
        dict(row.get("score_diagnostic") or {}) for row in hybrid_top10
    ]
    gold_components = {
        key: float(value)
        for key, value in dict(gold.get("score_components") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    top_components = [dict(row.get("components") or {}) for row in top_score_rows]
    component_names = sorted(
        set(gold_components).union(
            *(set(components) for components in top_components)
        )
    )
    components = {
        name: _gold_top10_numeric_comparison(
            gold_components.get(name),
            [row.get(name) for row in top_components],
        )
        for name in component_names
    }
    score_fields = (
        "normalized_rrf_score",
        "deterministic_rerank_score",
        "final_score",
    )
    scores = {
        field: _gold_top10_numeric_comparison(
            gold.get(field),
            [row.get(field) for row in top_score_rows],
        )
        for field in score_fields
    }
    return {"components": components, "scores": scores}


def _gold_top10_numeric_comparison(
    gold: Any, values: Sequence[Any]
) -> dict[str, float | None]:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "gold": (
            float(gold)
            if isinstance(gold, (int, float)) and not isinstance(gold, bool)
            else None
        ),
        "top10_mean": round(sum(numeric) / len(numeric), 8) if numeric else None,
        "top10_max": max(numeric) if numeric else None,
    }


def _gold_fusion_diagnostic(
    diagnostics: Sequence[Mapping[str, Any]],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
) -> dict[str, Any] | None:
    relevant = [
        dict(item)
        for item in diagnostics
        if _candidate_is_relevant(
            chunks_by_id.get(str(item.get("chunk_id") or ""), {}),
            str(item.get("doc_id") or ""),
            question,
        )
    ]
    if not relevant:
        return None
    return min(
        relevant,
        key=lambda item: (
            int(item.get("final_rank") or 10**9),
            str(item.get("chunk_id") or ""),
        ),
    )


def _summary(
    rows: Sequence[Mapping[str, Any]],
    rank_field: str,
    question_sets: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        "overall": _rank_metrics(rows, rank_field),
        "by_evaluation_set": {
            set_name: _rank_metrics(
                [row for row in rows if row["evaluation_set"] == set_name], rank_field
            )
            for set_name in question_sets
        },
        "by_task_type": _group_rank_metrics(rows, "task_type", rank_field),
        "by_doc_group": _group_rank_metrics(rows, "doc_group", rank_field),
    }


def _rank_metrics(
    rows: Sequence[Mapping[str, Any]], rank_field: str
) -> dict[str, Any]:
    count = len(rows)
    metrics: dict[str, Any] = {"question_count": count}
    for cutoff in _CUTOFFS:
        hits = sum(
            row.get(rank_field) is not None and int(row[rank_field]) <= cutoff
            for row in rows
        )
        metrics[f"recall_at_{cutoff}"] = round(hits / count, 6) if count else 0.0
    metrics["miss_count"] = sum(row.get(rank_field) is None for row in rows)
    return metrics


def _group_rank_metrics(
    rows: Sequence[Mapping[str, Any]], group_field: str, rank_field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_field) or "unknown")].append(row)
    return {
        name: _rank_metrics(grouped[name], rank_field) for name in sorted(grouped)
    }


def _summary_delta(
    hybrid: Mapping[str, Any], lexical: Mapping[str, Any]
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "overall": _metrics_delta(hybrid["overall"], lexical["overall"])
    }
    for key in ("by_evaluation_set", "by_task_type", "by_doc_group"):
        output[key] = {
            name: _metrics_delta(metrics, lexical[key][name])
            for name, metrics in hybrid[key].items()
            if name in lexical[key]
        }
    return output


def _metrics_delta(
    hybrid: Mapping[str, Any], lexical: Mapping[str, Any]
) -> dict[str, float]:
    return {
        f"recall_at_{cutoff}": round(
            float(hybrid[f"recall_at_{cutoff}"])
            - float(lexical[f"recall_at_{cutoff}"]),
            6,
        )
        for cutoff in _CUTOFFS
    }


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# PostgreSQL hybrid Gold60 evaluation",
        "",
        "Hybrid results use the same routed candidate universe as lexical retrieval.",
        "",
        "## Vector coverage",
        "",
        f"- Available: {report['vector_coverage']['available']}",
        f"- Embedded unique candidates: "
        f"{report['vector_coverage']['embedded_unique_candidate_count']} / "
        f"{report['vector_coverage']['unique_candidate_count']}",
        f"- Coverage ratio: {report['vector_coverage']['ratio']}",
        f"- Questions without embedded candidates: "
        f"{report['vector_coverage']['questions_without_embedded_candidates']}",
        "",
        "## Overall comparison",
        "",
        "| method | questions | R@1 | R@5 | R@10 |",
        "| --- | ---: | ---: | ---: | ---: |",
        _metrics_row("lexical-only", report["lexical_only"]["overall"]),
        _metrics_row("hybrid", report["hybrid"]["overall"]),
    ]
    for heading, key in (
        ("Evaluation set", "by_evaluation_set"),
        ("Task type", "by_task_type"),
        ("Document group", "by_doc_group"),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| scope | method | questions | R@1 | R@5 | R@10 |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, metrics in report["hybrid"][key].items():
            lexical = report["lexical_only"][key][name]
            lines.append(_scoped_metrics_row(name, "lexical-only", lexical))
            lines.append(_scoped_metrics_row(name, "hybrid", metrics))

    comparison = report.get("baseline_comparison")
    if isinstance(comparison, Mapping):
        overall = comparison["metrics"]["overall"]
        lines.extend(
            [
                "",
                "## Production baseline comparison",
                "",
                "| metric | baseline | current | delta |",
                "| --- | ---: | ---: | ---: |",
                *[
                    f"| {field.replace('recall_at_', 'R@')} | "
                    f"{overall['baseline'][field]} | {overall['current'][field]} | "
                    f"{overall['delta'][field]} |"
                    for field in ("recall_at_1", "recall_at_5", "recall_at_10")
                ],
                "",
                "- New failures: "
                + ", ".join(comparison["new_failures"] or ["none"]),
                "- Recovered: " + ", ".join(comparison["recovered"] or ["none"]),
            ]
        )

    lines.extend(["", "## Failures", ""])
    for row in report["failures"]:
        lines.extend(
            [
                f"### {row['question_id']} — {row['failure_class']}",
                "",
                f"- Question: {row['question']}",
                f"- Gold: `{row['gold']['doc_id']}` / `{row['gold']['target_id']}`",
                f"- Ranks (lexical-final/vector-raw/hybrid): "
                f"{row['lexical_gold_rank']} / {row['vector_gold_rank']} / "
                f"{row['hybrid_gold_rank']}",
                f"- Vector Top10/diagnostic ranks: "
                f"{row['vector_top10_gold_rank']} / "
                f"{row['vector_diagnostic_gold_rank']}",
                "- Gold fusion diagnostic: "
                + json.dumps(
                    row["gold_fusion_diagnostic"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                f"- Vector status: {row['vector_status']}",
                "- Hybrid Top10: "
                + ", ".join(
                    f"{item['rank']}:{item['chunk_id']}({item['doc_id']})"
                    for item in row["hybrid_top10"]
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _metrics_row(name: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {metrics['question_count']} | {metrics['recall_at_1']:.3f} | "
        f"{metrics['recall_at_5']:.3f} | {metrics['recall_at_10']:.3f} |"
    )


def _scoped_metrics_row(
    scope: str, method: str, metrics: Mapping[str, Any]
) -> str:
    return (
        f"| {scope} | {method} | {metrics['question_count']} | "
        f"{metrics['recall_at_1']:.3f} | {metrics['recall_at_5']:.3f} | "
        f"{metrics['recall_at_10']:.3f} |"
    )


def _write_question_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = [
        "question_id",
        "evaluation_set",
        "doc_group",
        "task_type",
        "question",
        "lexical_query",
        "candidate_document_count",
        "candidate_chunk_count",
        "embedded_vector_candidate_count",
        "vector_candidate_coverage",
        "has_any_vector_candidate",
        "lexical_raw_gold_rank",
        "lexical_gold_rank",
        "vector_gold_rank",
        "vector_top10_gold_rank",
        "hybrid_gold_rank",
        "lexical_diagnostic_gold_rank",
        "vector_diagnostic_gold_rank",
        "diagnostic_source_top_n",
        "gold_fusion_diagnostic",
        "lexical_gold_matches",
        "query_evidence_diagnostic",
        "score_component_comparison",
        "vector_status",
        "vector_error",
        "failure_class",
        "query_plan",
        "hard_filters",
        "soft_boosts",
        "gold",
        "lexical_top10",
        "vector_top10",
        "vector_production_top_n",
        "vector_gold_matches",
        "vector_rank_diagnostic",
        "hybrid_top10",
        "section_path_diagnostic",
    ]
    structured = {
        "query_plan",
        "hard_filters",
        "soft_boosts",
        "gold",
        "lexical_top10",
        "vector_top10",
        "vector_production_top_n",
        "vector_gold_matches",
        "vector_rank_diagnostic",
        "hybrid_top10",
        "diagnostic_source_top_n",
        "section_path_diagnostic",
        "gold_fusion_diagnostic",
        "lexical_gold_matches",
        "query_evidence_diagnostic",
        "score_component_comparison",
    }
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    **{
                        key: json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
                        for key in structured
                    },
                }
            )
