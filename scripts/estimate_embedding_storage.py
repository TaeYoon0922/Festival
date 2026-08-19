"""Estimate subset and full-corpus pgvector storage without touching PostgreSQL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embedding_storage import (
    FULL_CORPUS_CHUNKS,
    estimate_embedding_storage,
)
from app.retrieval.embeddings import EmbeddingConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate vector table and HNSW storage capacity."
    )
    parser.add_argument("--candidate-ids", type=Path)
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--full-corpus-count", type=int, default=FULL_CORPUS_CHUNKS)
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--storage-capacity-gb", type=float, default=100.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.candidate_count is None and args.candidate_ids is None:
        parser.error("provide --candidate-count or --candidate-ids")
    candidate_count = (
        args.candidate_count
        if args.candidate_count is not None
        else _count_ids(args.candidate_ids)
    )
    dimensions = args.dimensions or EmbeddingConfig.from_env().dimensions
    estimate = estimate_embedding_storage(
        candidate_count,
        dimensions,
        full_corpus_count=args.full_corpus_count,
        storage_capacity_gb=args.storage_capacity_gb,
    )
    output = json.dumps(estimate, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


def _count_ids(path: Path) -> int:
    with path.open("r", encoding="utf-8") as source:
        # Collector output is already unique and sorted; count it as a stream.
        return sum(1 for line in source if line.strip())


if __name__ == "__main__":
    main()
