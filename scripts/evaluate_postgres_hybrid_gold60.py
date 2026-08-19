"""Run frozen Gold40 + Holding20 through PostgreSQL lexical/vector hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.final_validation import GOLD_QUESTIONS, HOLDING_ADDITIONAL_QUESTIONS
from app.reasoning.hybrid_evaluation import (
    QueryPlanHybridEvaluator,
    write_hybrid_evaluation_report,
)
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from app.retrieval.hybrid import (
    HybridQueryExecutor,
    HybridRetrievalConfig,
    RRFConfig,
)
from app.retrieval.postgres_backend import PostgresBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate metadata-scoped PostgreSQL hybrid retrieval on Gold60."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "postgres_hybrid_gold60",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lexical-top-n", type=int, default=50)
    parser.add_argument("--vector-top-n", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--fusion-weight", type=float, default=0.60)
    parser.add_argument("--deterministic-weight", type=float, default=0.40)
    args = parser.parse_args()

    embedding_config = EmbeddingConfig.from_env()
    embedder = create_embedding_provider(embedding_config)

    backend = PostgresBackend()
    understanding = QueryUnderstanding(company_resolver=backend.resolve_company)
    hybrid_config = HybridRetrievalConfig(
        lexical_top_n=args.lexical_top_n,
        vector_top_n=args.vector_top_n,
        final_top_k=args.top_k,
        fusion_weight=args.fusion_weight,
        deterministic_weight=args.deterministic_weight,
        rrf=RRFConfig(
            k=args.rrf_k,
            lexical_weight=args.lexical_weight,
            vector_weight=args.vector_weight,
        ),
    )
    executor = HybridQueryExecutor(
        backend,
        embedder,
        embedding_config,
        config=hybrid_config,
    )
    evaluator = QueryPlanHybridEvaluator(
        understanding,
        executor,
        top_k=args.top_k,
    )
    question_sets = {
        "gold_40": GOLD_QUESTIONS,
        "holding_20": HOLDING_ADDITIONAL_QUESTIONS,
    }
    if len(GOLD_QUESTIONS) != 40 or len(HOLDING_ADDITIONAL_QUESTIONS) != 20:
        raise RuntimeError("frozen evaluation sets must contain Gold40 + Holding20")

    def progress(index: int, total: int, question_id: str) -> None:
        print(f"[{index:02d}/{total:02d}] {question_id}", flush=True)

    report = evaluator.evaluate(question_sets, progress=progress)
    write_hybrid_evaluation_report(report, args.output_dir)
    print(
        json.dumps(
            {
                "hybrid": report["hybrid"],
                "lexical_only": report["lexical_only"],
                "improvement": report["improvement"],
                "failure_counts": report["failure_counts"],
                "vector_status_counts": report["vector_status_counts"],
                "vector_coverage": report["vector_coverage"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
