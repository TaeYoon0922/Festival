"""Resumably embed and upsert a streamed candidate subset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.embedding_pipeline import (
    EmbeddingPipelineConfig,
    RetryConfig,
    SubsetEmbeddingPipeline,
    iter_candidate_file,
)
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from app.retrieval.postgres_backend import PostgresBackend


class _DryRunProvider:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config

    def embed_query(self, _text: str) -> list[float]:
        raise AssertionError("dry-run must not invoke the embedding provider")

    def embed_documents(self, _texts: list[str]) -> list[list[float]]:
        raise AssertionError("dry-run must not invoke the embedding provider")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed a candidate JSONL/ID subset with resumable DB upserts."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--provider")
    parser.add_argument("--model")
    parser.add_argument("--version")
    parser.add_argument("--dimensions", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--fetch-batch-size", type=int, default=1000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--max-retry-delay-seconds", type=float, default=30.0)
    parser.add_argument("--inter-batch-delay-seconds", type=float, default=0.0)
    parser.add_argument("--error-log", type=Path)
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
    backend = PostgresBackend()
    provider = (
        _DryRunProvider(config)
        if args.dry_run
        else create_embedding_provider(config)
    )
    error_log = args.error_log or args.input.with_name("embedding_errors.jsonl")
    pipeline = SubsetEmbeddingPipeline(
        provider,
        backend,
        config,
        pipeline_config=EmbeddingPipelineConfig(
            batch_size=config.batch_size,
            limit=args.limit,
            force=args.force,
            dry_run=args.dry_run,
            resume=args.resume,
            inter_batch_delay_seconds=args.inter_batch_delay_seconds,
            retry=RetryConfig(
                max_attempts=args.max_attempts,
                initial_delay_seconds=args.retry_delay_seconds,
                max_delay_seconds=args.max_retry_delay_seconds,
            ),
        ),
        error_path=error_log,
    )

    def progress(state: dict) -> None:
        print(
            "processed={unique_chunk_ids} embedded={embedded} skipped={already_embedded} "
            "failed={failed}".format(**state),
            flush=True,
        )

    summary = pipeline.run(
        iter_candidate_file(
            args.input,
            backend=backend,
            fetch_batch_size=args.fetch_batch_size,
        ),
        progress=progress,
    )
    summary["input"] = str(args.input)
    summary["error_log"] = str(error_log)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
