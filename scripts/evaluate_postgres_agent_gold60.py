"""Evaluate frozen Gold60 through PostgreSQL hybrid retrieval and the Agent stack."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.gold60_evaluation import (
    AgentGold60Evaluator,
    write_agent_gold60_evaluation,
)
from app.parsing.final_validation import GOLD_QUESTIONS, HOLDING_ADDITIONAL_QUESTIONS
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from app.retrieval.hybrid import HybridQueryExecutor, HybridRetrievalConfig, RRFConfig
from app.retrieval.postgres_backend import PostgresBackend


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run Gold60 through the existing PostgreSQL hybrid retrieval and "
            "deterministic Agent answer pipeline."
        )
    )
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON question list or mapping of evaluation-set names to "
            "question lists. Defaults to frozen Gold40 + Holding20."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "processed" / "postgres_agent_gold60"
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lexical-top-n", type=int, default=50)
    parser.add_argument("--vector-top-n", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--vector-weight", type=float, default=1.0)
    parser.add_argument("--fusion-weight", type=float, default=0.60)
    parser.add_argument("--deterministic-weight", type=float, default=0.40)
    parser.add_argument(
        "--rerank-mode", choices=("legacy", "bounded"), default="legacy"
    )
    parser.add_argument("--rerank-window-size", type=int, default=2)
    parser.add_argument("--diagnostic-top-n", type=int, default=None)
    args = parser.parse_args()

    question_sets = load_question_sets(args.questions_json)
    question_count = sum(len(values) for values in question_sets.values())
    if args.questions_json is None and question_count != 60:
        raise RuntimeError("frozen evaluation input must contain Gold40 + Holding20")

    embedding_config = EmbeddingConfig.from_env()
    embedder = create_embedding_provider(embedding_config)
    backend = PostgresBackend()
    understanding = QueryUnderstanding(company_resolver=backend.resolve_company)
    executor = HybridQueryExecutor(
        backend,
        embedder,
        embedding_config,
        config=HybridRetrievalConfig(
            lexical_top_n=args.lexical_top_n,
            vector_top_n=args.vector_top_n,
            final_top_k=args.top_k,
            fusion_weight=args.fusion_weight,
            deterministic_weight=args.deterministic_weight,
            rerank_mode=args.rerank_mode,
            rerank_window_size=args.rerank_window_size,
            diagnostic_top_n=args.diagnostic_top_n,
            rrf=RRFConfig(
                k=args.rrf_k,
                lexical_weight=args.lexical_weight,
                vector_weight=args.vector_weight,
            ),
        ),
    )
    evaluator = AgentGold60Evaluator(
        understanding,
        executor,
        top_k=args.top_k,
    )

    def progress(index: int, total: int, question_id: str) -> None:
        print(f"[{index:02d}/{total:02d}] {question_id}", flush=True)

    report = evaluator.evaluate(question_sets, progress=progress)
    analysis = write_agent_gold60_evaluation(report, args.output_dir)
    print(
        json.dumps(
            {
                "question_count": report["question_count"],
                "retrieval": report["hybrid"]["overall"],
                "agent": report["agent"]["overall"],
                "failure_counts": report["agent"]["failure_counts"],
                "failure_analysis": analysis["summary"],
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def load_question_sets(path: Path | None) -> dict[str, tuple[dict[str, Any], ...]]:
    if path is None:
        return {
            "gold_40": tuple(dict(row) for row in GOLD_QUESTIONS),
            "holding_20": tuple(
                dict(row) for row in HOLDING_ADDITIONAL_QUESTIONS
            ),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        question_sets = {
            str(name): _question_rows(rows)
            for name, rows in payload.items()
        }
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        question_sets = {"gold_60": _question_rows(payload)}
    else:
        raise ValueError("questions JSON must be a list or an object of lists")
    if not question_sets or not any(question_sets.values()):
        raise ValueError("questions JSON must contain at least one question")
    return question_sets


def _question_rows(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("each evaluation set must be a question list")
    rows = []
    required = {"question_id", "query", "doc_id", "target_type", "target_id"}
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("each question must be an object")
        row = dict(item)
        missing = sorted(required.difference(row))
        if missing:
            raise ValueError(
                "question is missing required fields: " + ", ".join(missing)
            )
        rows.append(row)
    return tuple(rows)


if __name__ == "__main__":
    raise SystemExit(main())
