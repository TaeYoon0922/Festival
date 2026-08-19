"""Gold-set evaluation for QueryPlan/Router lexical retrieval."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.parsing.final_validation import _is_relevant


ProgressCallback = Callable[[int, int, str], None]


class QueryPlanLexicalEvaluator:
    """Evaluate an understanding service and executor against frozen gold questions."""

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
        baselines: Mapping[str, Mapping[str, Any]] | None = None,
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

        by_set = {
            set_name: _metrics(
                [row for row in rows if row["evaluation_set"] == set_name]
            )
            for set_name in question_sets
        }
        by_task = _group_metrics(rows, "task_type")
        by_doc_group = _group_metrics(rows, "doc_group")
        normalized_baselines = {
            name: _normalize_metrics(metrics)
            for name, metrics in (baselines or {}).items()
        }
        if normalized_baselines and "overall" not in normalized_baselines:
            combined = _weighted_metrics(normalized_baselines)
            if combined:
                normalized_baselines["overall"] = combined
        current = {"overall": _metrics(rows), **by_set}
        return {
            "method": {
                "flow": (
                    "Question -> QueryUnderstanding -> QueryPlan/Router -> "
                    "PostgreSQL lexical retrieval -> Router rerank -> Top-10"
                ),
                "retrieval": "lexical only; no vector or RRF",
                "gold_judgment": "app.parsing.final_validation._is_relevant",
                "top_k": self.top_k,
            },
            "question_count": len(rows),
            "overall": current["overall"],
            "by_evaluation_set": by_set,
            "by_task_type": by_task,
            "by_doc_group": by_doc_group,
            "baseline": normalized_baselines,
            "baseline_comparison": _baseline_comparison(current, normalized_baselines),
            "failure_counts": dict(
                sorted(Counter(row["failure_class"] for row in rows).items())
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

        retrieved: list[dict[str, Any]] = []
        gold_rank: int | None = None
        for result in execution.results[:10]:
            chunk = chunks_by_id.get(result.chunk_id, {})
            relevant = _candidate_is_relevant(chunk, result.doc_id, question)
            if relevant and gold_rank is None:
                gold_rank = result.rank
            retrieved.append(_result_summary(result, chunk, relevant))

        if gold_doc_id not in candidate_doc_ids:
            failure_class = "metadata_filter_failure"
        elif not relevant_candidates:
            failure_class = "gold_mapping_or_route_failure"
        elif gold_rank is None:
            failure_class = "lexical_ranking_failure"
        else:
            failure_class = "success"

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
            "retrieved_top10": retrieved,
            "gold_rank": gold_rank,
            "hit_at_1": bool(gold_rank and gold_rank <= 1),
            "hit_at_5": bool(gold_rank and gold_rank <= 5),
            "hit_at_10": bool(gold_rank and gold_rank <= 10),
            "failure_class": failure_class,
        }


def load_baseline_reports(directory: Path) -> dict[str, dict[str, Any]]:
    """Load the existing fixed-40 and holding-20 BM25 summaries when present."""

    filenames = {
        "gold_40": "bm25_fixed_40.json",
        "holding_20": "bm25_holding_20.json",
    }
    baselines: dict[str, dict[str, Any]] = {}
    for name, filename in filenames.items():
        path = directory / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        baselines[name] = _normalize_metrics(payload.get("overall", payload))
    return baselines


def write_evaluation_report(report: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "postgres_lexical_gold60.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "postgres_lexical_gold60.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    _write_question_csv(report["questions"], output_dir / "postgres_lexical_questions.csv")


def _candidate_is_relevant(
    chunk: Mapping[str, Any], doc_id: str, question: Mapping[str, Any]
) -> bool:
    payload = dict(chunk)
    payload.setdefault("doc_id", doc_id)
    return _is_relevant(payload, dict(question))


def _result_summary(result: Any, chunk: Mapping[str, Any], relevant: bool) -> dict[str, Any]:
    section_path = chunk.get("section_path") or []
    if isinstance(section_path, str):
        section_path = [section_path]
    components = dict(result.metadata_match.get("score_components") or {})
    return {
        "rank": result.rank,
        "chunk_id": result.chunk_id,
        "doc_id": result.doc_id,
        "bm25_score": float(result.bm25_score),
        "final_score": components.get("final_score"),
        "score_components": components,
        "report_nm": chunk.get("report_nm"),
        "section_path": list(section_path),
        "chunk_type": chunk.get("chunk_type"),
        "content_preview": " ".join(str(chunk.get("content") or "").split())[:240],
        "is_gold_relevant": relevant,
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "question_count": count,
        "recall_at_1": round(sum(bool(row["hit_at_1"]) for row in rows) / count, 6)
        if count
        else 0.0,
        "recall_at_5": round(sum(bool(row["hit_at_5"]) for row in rows) / count, 6)
        if count
        else 0.0,
        "recall_at_10": round(sum(bool(row["hit_at_10"]) for row in rows) / count, 6)
        if count
        else 0.0,
        "failure_count": sum(row["failure_class"] != "success" for row in rows),
    }


def _group_metrics(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field) or "unknown")].append(row)
    return {name: _metrics(grouped[name]) for name in sorted(grouped)}


def _normalize_metrics(value: Mapping[str, Any]) -> dict[str, Any]:
    metrics: Mapping[str, Any] = value
    if isinstance(value.get("structural"), Mapping):
        metrics = value["structural"]
    return {
        "question_count": int(metrics.get("question_count") or 0),
        "recall_at_1": float(metrics.get("recall_at_1") or 0.0),
        "recall_at_5": float(metrics.get("recall_at_5") or 0.0),
        "recall_at_10": float(metrics.get("recall_at_10") or 0.0),
    }


def _weighted_metrics(values: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    count = sum(int(metrics.get("question_count") or 0) for metrics in values.values())
    if not count:
        return None
    return {
        "question_count": count,
        **{
            key: round(
                sum(
                    float(metrics.get(key) or 0.0)
                    * int(metrics.get("question_count") or 0)
                    for metrics in values.values()
                )
                / count,
                6,
            )
            for key in ("recall_at_1", "recall_at_5", "recall_at_10")
        },
    }


def _baseline_comparison(
    current: Mapping[str, Mapping[str, Any]],
    baselines: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    for name in current.keys() & baselines.keys():
        comparison[name] = {
            key: round(float(current[name][key]) - float(baselines[name][key]), 6)
            for key in ("recall_at_1", "recall_at_5", "recall_at_10")
        }
    return comparison


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# QueryPlan/Router PostgreSQL lexical Gold60 evaluation",
        "",
        "Vector retrieval and RRF are not included.",
        "",
        "## Summary",
        "",
        "| scope | questions | R@1 | R@5 | R@10 | failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    scopes = {"overall": report["overall"], **report["by_evaluation_set"]}
    for name, metrics in scopes.items():
        lines.append(_metrics_row(name, metrics))
    for heading, key in (
        ("Task type", "by_task_type"),
        ("Document group", "by_doc_group"),
    ):
        lines.extend(
            [
                "",
                f"## {heading}",
                "",
                "| scope | questions | R@1 | R@5 | R@10 | failures |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name, metrics in report[key].items():
            lines.append(_metrics_row(name, metrics))

    if report.get("baseline_comparison"):
        lines.extend(
            [
                "",
                "## Delta from baseline",
                "",
                "| scope | ΔR@1 | ΔR@5 | ΔR@10 |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, metrics in report["baseline_comparison"].items():
            lines.append(
                f"| {name} | {metrics['recall_at_1']:+.3f} | "
                f"{metrics['recall_at_5']:+.3f} | {metrics['recall_at_10']:+.3f} |"
            )

    lines.extend(["", "## Failures", ""])
    for row in report["failures"]:
        lines.extend(
            [
                f"### {row['question_id']} · {row['failure_class']}",
                "",
                f"- Question: {row['question']}",
                f"- Lexical query: {row['lexical_query']}",
                f"- Gold: `{row['gold']['doc_id']}` / `{row['gold']['target_id']}`",
                f"- Gold rank: {row['gold_rank']}",
                f"- Hard filters: `{json.dumps(row['hard_filters'], ensure_ascii=False)}`",
                f"- Soft boosts: `{json.dumps(row['soft_boosts'], ensure_ascii=False)}`",
                "- Top10: "
                + ", ".join(
                    f"{item['rank']}:{item['chunk_id']}({item['doc_id']})"
                    for item in row["retrieved_top10"]
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _metrics_row(name: str, metrics: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {metrics['question_count']} | {metrics['recall_at_1']:.3f} | "
        f"{metrics['recall_at_5']:.3f} | {metrics['recall_at_10']:.3f} | "
        f"{metrics.get('failure_count', '-')} |"
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
        "gold_rank",
        "hit_at_1",
        "hit_at_5",
        "hit_at_10",
        "failure_class",
        "query_plan",
        "hard_filters",
        "soft_boosts",
        "gold",
        "retrieved_top10",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    **{
                        key: json.dumps(row[key], ensure_ascii=False, separators=(",", ":"))
                        for key in (
                            "query_plan",
                            "hard_filters",
                            "soft_boosts",
                            "gold",
                            "retrieved_top10",
                        )
                    },
                }
            )
