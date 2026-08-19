"""Collect the Gold60 metadata-scoped chunk union without using gold answers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.final_validation import GOLD_QUESTIONS, HOLDING_ADDITIONAL_QUESTIONS
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.embedding_candidates import (
    Gold60EmbeddingCandidateCollector,
    write_candidate_collection,
)
from app.retrieval.postgres_backend import PostgresBackend


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect metadata-routed Gold60 chunks for pilot embedding."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "data" / "processed" / "gold60_embedding_candidates"
        ),
    )
    args = parser.parse_args()

    if len(GOLD_QUESTIONS) != 40 or len(HOLDING_ADDITIONAL_QUESTIONS) != 20:
        raise RuntimeError("frozen evaluation sets must contain Gold40 + Holding20")
    backend = PostgresBackend()
    collector = Gold60EmbeddingCandidateCollector(
        QueryUnderstanding(company_resolver=backend.resolve_company), backend
    )

    def progress(index: int, total: int, question_id: str, count: int) -> None:
        print(f"[{index:02d}/{total:02d}] {question_id}: {count} chunks", flush=True)

    collection = collector.collect(
        {
            "gold_40": GOLD_QUESTIONS,
            "holding_20": HOLDING_ADDITIONAL_QUESTIONS,
        },
        progress=progress,
    )
    write_candidate_collection(collection, args.output_dir)
    print(
        json.dumps(
            {
                key: collection.summary[key]
                for key in (
                    "total_questions",
                    "total_candidate_occurrences",
                    "unique_candidate_chunks",
                    "candidate_count_stats",
                    "doc_group_distribution",
                    "chunk_type_distribution",
                )
            }
            | {"output_dir": str(args.output_dir)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
