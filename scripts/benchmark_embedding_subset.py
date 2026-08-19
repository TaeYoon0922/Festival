"""Benchmark real embedding inference on a small candidate subset without DB writes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embedding_benchmark import (
    EmbeddingBenchmarkConfig,
    EmbeddingSubsetBenchmark,
)
from app.retrieval.embedding_pipeline import RetryConfig, iter_candidate_file
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark embedding inference only; never writes PostgreSQL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--version")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    args = parser.parse_args()

    environment_config = EmbeddingConfig.from_env()
    config = replace(
        environment_config,
        provider=args.provider or environment_config.provider,
        model=args.model or environment_config.model,
        version=args.version or environment_config.version,
        dimensions=args.dimensions or environment_config.dimensions,
        batch_size=args.batch_size or environment_config.batch_size,
        max_length=args.max_length or environment_config.max_length,
        device=args.device or environment_config.device,
    )
    provider = create_embedding_provider(config)
    benchmark = EmbeddingSubsetBenchmark(
        provider,
        config=EmbeddingBenchmarkConfig(
            limit=args.limit,
            batch_size=config.batch_size,
            retry=RetryConfig(
                max_attempts=args.max_attempts,
                initial_delay_seconds=args.retry_delay_seconds,
            ),
        ),
    )
    report = benchmark.run(iter_candidate_file(args.input))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
