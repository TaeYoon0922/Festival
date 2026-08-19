"""Gold60 evaluation for metadata-scoped lexical/vector hybrid retrieval."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.reasoning.lexical_evaluation import (
    _candidate_is_relevant,
    _result_summary,
)


ProgressCallback = Callable[[int, int, str], None]
_CUTOFFS = (1, 5, 10)


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

        hybrid = _summary(rows, "hybrid_gold_rank", question_sets)
        lexical = _summary(rows, "lexical_gold_rank", question_sets)
        return {
            "method": {
                "flow": (
                    "Question -> QueryUnderstanding -> QueryPlan/Router -> shared "
                    "metadata candidate scope -> lexical Top-N + vector Top-N -> RRF "
                    "-> deterministic rerank -> Top-10"
                ),
                "gold_judgment": "app.parsing.final_validation._is_relevant",
                "top_k": self.top_k,
            },
            "question_count": len(rows),
            "hybrid": hybrid,
            "lexical_only": lexical,
            "improvement": _summary_delta(hybrid, lexical),
            "failure_counts": dict(
                sorted(Counter(row["failure_class"] for row in rows).items())
            ),
            "vector_status_counts": dict(
                sorted(Counter(row["vector_status"] for row in rows).items())
            ),
            "failures": [row for row in rows if row["failure_class"] != "success"],
            "questions": rows,
        }

    def _evaluate_question(
        self, set_name: str, question: Mapping[str, Any]
    ) -> dict[str, Any]:
        raw_query = str(question["query"])
        plan = self.understanding.understand(raw_query, top_k=self.top_k)
        execution = self.executor.execute(plan)
        chunks_by_id = {candidate.chunk_id: candidate.chunk for candidate in execution.chunks}
        candidate_doc_ids = {document.doc_id for document in execution.documents}
        gold_doc_id = str(question["doc_id"])
        relevant_candidates = [
            candidate.chunk_id
            for candidate in execution.chunks
            if _candidate_is_relevant(candidate.chunk, candidate.doc_id, question)
        ]

        lexical_raw_rank = _gold_rank(
            execution.lexical_results, chunks_by_id, question
        )
        lexical_rank = _gold_rank(
            execution.lexical_final_results, chunks_by_id, question
        )
        vector_rank = _gold_rank(execution.vector_results, chunks_by_id, question)
        hybrid_rank = _gold_rank(execution.results, chunks_by_id, question)

        if gold_doc_id not in candidate_doc_ids:
            failure_class = "metadata_filter_failure"
        elif not relevant_candidates:
            failure_class = "gold_mapping_or_route_failure"
        elif hybrid_rank is not None:
            failure_class = "success"
        elif vector_rank is None:
            failure_class = "vector_missing_failure"
        else:
            failure_class = "fusion_ranking_failure"

        routing = dict(execution.routing)
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
            },
            "candidate_document_count": len(execution.documents),
            "candidate_chunk_count": len(execution.chunks),
            "gold_document_in_candidates": gold_doc_id in candidate_doc_ids,
            "gold_relevant_candidate_count": len(relevant_candidates),
            "lexical_raw_gold_rank": lexical_raw_rank,
            "lexical_gold_rank": lexical_rank,
            "vector_gold_rank": vector_rank,
            "hybrid_gold_rank": hybrid_rank,
            "lexical_top10": _retrieval_summaries(
                execution.lexical_final_results, chunks_by_id, question
            ),
            "vector_top10": _vector_summaries(
                execution.vector_results, chunks_by_id, question
            ),
            "hybrid_top10": _hybrid_summaries(
                execution.results, chunks_by_id, question
            ),
            "vector_status": execution.vector_status,
            "vector_error": execution.vector_error,
            "hit_at_1": bool(hybrid_rank and hybrid_rank <= 1),
            "hit_at_5": bool(hybrid_rank and hybrid_rank <= 5),
            "hit_at_10": bool(hybrid_rank and hybrid_rank <= 10),
            "failure_class": failure_class,
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


def _gold_rank(
    results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
) -> int | None:
    for result in results[:10]:
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
        output.append(_result_summary(result, chunk, relevant))
    return output


def _vector_summaries(
    results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for result in results[:10]:
        chunk = chunks_by_id.get(result.chunk_id, {})
        output.append(
            {
                "rank": result.rank,
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "vector_score": float(result.vector_score),
                "is_gold_relevant": _candidate_is_relevant(
                    chunk, result.doc_id, question
                ),
            }
        )
    return output


def _hybrid_summaries(
    results: Sequence[Any],
    chunks_by_id: Mapping[str, Mapping[str, Any]],
    question: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = _retrieval_summaries(results, chunks_by_id, question)
    for item, result in zip(output, results[:10]):
        item["hybrid"] = dict(result.metadata_match.get("hybrid") or {})
    return output


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

    lines.extend(["", "## Failures", ""])
    for row in report["failures"]:
        lines.extend(
            [
                f"### {row['question_id']} — {row['failure_class']}",
                "",
                f"- Question: {row['question']}",
                f"- Gold: `{row['gold']['doc_id']}` / `{row['gold']['target_id']}`",
                f"- Ranks (lexical/vector/hybrid): "
                f"{row['lexical_gold_rank']} / {row['vector_gold_rank']} / "
                f"{row['hybrid_gold_rank']}",
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
        "lexical_raw_gold_rank",
        "lexical_gold_rank",
        "vector_gold_rank",
        "hybrid_gold_rank",
        "vector_status",
        "vector_error",
        "failure_class",
        "query_plan",
        "hard_filters",
        "soft_boosts",
        "gold",
        "lexical_top10",
        "vector_top10",
        "hybrid_top10",
    ]
    structured = {
        "query_plan",
        "hard_filters",
        "soft_boosts",
        "gold",
        "lexical_top10",
        "vector_top10",
        "hybrid_top10",
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
