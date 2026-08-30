"""Embed the Gold60 candidate union with the pinned BGE-M3, resumably.

Reuses the production ``SubsetEmbeddingPipeline`` for batching, resume,
idempotent upsert and retry, and reuses ``chunk_embedding_text`` so the text
embedded here is byte-identical to what production ingestion would embed.

The only thing this adds is identity: the encoder is built from the pinned
snapshot rather than from whatever ``main`` points at today, and the run refuses
to start unless the configured provider/model/revision/dimension/device are the
ones the baseline is defined against.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.embedding_pipeline import (
    EmbeddingPipelineConfig,
    SubsetEmbeddingPipeline,
    iter_candidate_file,
)
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from app.retrieval.postgres_backend import PostgresBackend
from scripts.bge_eval_preflight import (
    assert_embedding_identity,
    describe,
    pinned_encoder_factory,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--error-log", default="/out/bge/embedding_errors.jsonl")
    parser.add_argument("--report", default=None)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="only for an explicitly CPU-scoped experiment")
    args = parser.parse_args()

    config = EmbeddingConfig.from_env()
    if args.batch_size:
        config = EmbeddingConfig(
            provider=config.provider, model=config.model, version=config.version,
            dimensions=config.dimensions, batch_size=args.batch_size,
            max_length=config.max_length, device=config.device,
            cuda_oom_retry=config.cuda_oom_retry,
            min_batch_size=config.min_batch_size)
    assert_embedding_identity(config, require_device=None if args.allow_cpu else "cuda")

    peak_vram = 0
    torch = None
    if config.device.casefold().startswith("cuda"):
        import torch  # noqa: F401 - used for the memory probes below

        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable; refusing to silently use CPU")
        torch.cuda.reset_peak_memory_stats()

    provider = create_embedding_provider(
        config, bge_encoder_factory=pinned_encoder_factory)
    backend = PostgresBackend()
    error_path = Path(args.error_log)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    pipeline = SubsetEmbeddingPipeline(
        provider, backend, config,
        pipeline_config=EmbeddingPipelineConfig(
            batch_size=config.batch_size, limit=args.limit, resume=True),
        error_path=error_path)

    seen = {"batches": 0}

    def progress(state):
        seen["batches"] += 1
        if seen["batches"] % 25 == 0:
            print(f"  … embedded={state.get('embedded')} "
                  f"skipped={state.get('already_embedded')} "
                  f"failed={state.get('failed')}", flush=True)

    started = time.perf_counter()
    state = pipeline.run(iter_candidate_file(Path(args.input)), progress=progress)
    elapsed = time.perf_counter() - started

    if torch is not None:
        peak_vram = round(torch.cuda.max_memory_allocated() / 2**20)
    embedded = int(state.get("embedded") or 0)
    state.update({
        "elapsed_seconds": round(elapsed, 1),
        "chunks_per_second": round(embedded / elapsed, 2) if elapsed else 0.0,
        "peak_vram_MiB": peak_vram,
        "identity": describe(config),
    })
    print(json.dumps(state, ensure_ascii=False, indent=1))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    return 1 if int(state.get("failed") or 0) else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
