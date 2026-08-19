"""Run the frozen Gold40 + Holding20 set through PostgreSQL lexical retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.final_validation import GOLD_QUESTIONS, HOLDING_ADDITIONAL_QUESTIONS
from app.reasoning import QueryExecutor, QueryUnderstanding
from app.reasoning.lexical_evaluation import (
    QueryPlanLexicalEvaluator,
    load_baseline_reports,
    write_evaluation_report,
)
from app.retrieval.postgres_backend import PostgresBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate QueryPlan/Router PostgreSQL lexical retrieval on Gold60."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "postgres_lexical_gold60",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "structural_v2_1_full_4204"
            / "final_release_gate"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    backend = PostgresBackend()
    understanding = QueryUnderstanding(company_resolver=backend.resolve_company)
    evaluator = QueryPlanLexicalEvaluator(
        understanding,
        QueryExecutor(backend),
        top_k=args.top_k,
    )
    baselines = load_baseline_reports(args.baseline_dir)
    question_sets = {
        "gold_40": GOLD_QUESTIONS,
        "holding_20": HOLDING_ADDITIONAL_QUESTIONS,
    }
    if len(GOLD_QUESTIONS) != 40 or len(HOLDING_ADDITIONAL_QUESTIONS) != 20:
        raise RuntimeError("frozen evaluation sets must contain Gold40 + Holding20")
    if not baselines:
        print(
            f"warning: no baseline reports found in {args.baseline_dir}",
            file=sys.stderr,
        )

    def progress(index: int, total: int, question_id: str) -> None:
        print(f"[{index:02d}/{total:02d}] {question_id}", flush=True)

    report = evaluator.evaluate(
        question_sets,
        baselines=baselines,
        progress=progress,
    )
    write_evaluation_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "by_evaluation_set": report["by_evaluation_set"],
                "by_task_type": report["by_task_type"],
                "by_doc_group": report["by_doc_group"],
                "baseline_comparison": report["baseline_comparison"],
                "failure_counts": report["failure_counts"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
