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
    CPU_BGE_M3_BASELINE_DOCUMENTS_PER_SECOND,
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
    parser.add_argument("--device", help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--min-batch-size", type=int)
    parser.add_argument(
        "--cuda-oom-retry", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--version")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--max-retry-delay-seconds", type=float, default=30.0)
    parser.add_argument(
        "--cpu-baseline-documents-per-second",
        type=float,
        default=CPU_BGE_M3_BASELINE_DOCUMENTS_PER_SECOND,
    )
    args = parser.parse_args()

    environment_config = EmbeddingConfig.from_env()
    config = replace(
        environment_config,
        provider=args.provider or environment_config.provider,
        model=args.model or environment_config.model,
        version=args.version or environment_config.version,
        dimensions=(
            environment_config.dimensions
            if args.dimensions is None
            else args.dimensions
        ),
        batch_size=(
            environment_config.batch_size
            if args.batch_size is None
            else args.batch_size
        ),
        max_length=(
            environment_config.max_length
            if args.max_length is None
            else args.max_length
        ),
        device=args.device or environment_config.device,
        cuda_oom_retry=(
            environment_config.cuda_oom_retry
            if args.cuda_oom_retry is None
            else args.cuda_oom_retry
        ),
        min_batch_size=(
            environment_config.min_batch_size
            if args.min_batch_size is None
            else args.min_batch_size
        ),
    )
    provider = create_embedding_provider(config)
    benchmark = EmbeddingSubsetBenchmark(
        provider,
        config=EmbeddingBenchmarkConfig(
            limit=args.limit,
            batch_size=config.batch_size,
            cpu_baseline_documents_per_second=(
                args.cpu_baseline_documents_per_second
            ),
            retry=RetryConfig(
                max_attempts=args.max_attempts,
                initial_delay_seconds=args.retry_delay_seconds,
                max_delay_seconds=args.max_retry_delay_seconds,
            ),
        ),
    )
    report = benchmark.run(iter_candidate_file(args.input))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
