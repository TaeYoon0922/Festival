"""Prove the pinned BGE-M3 really loaded on the GPU before anything is embedded.

A smoke test that only checks "did I get 1024 numbers" would pass on the hash
embedder too, so this checks the things that differ: the vectors are L2
normalized as the provider promises, repeated encoding of the same text agrees,
different texts disagree, and the work actually happened on CUDA.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from scripts.bge_eval_preflight import (
    PINNED_REVISION,
    assert_embedding_identity,
    describe,
    pinned_encoder_factory,
)


def main() -> int:
    config = EmbeddingConfig.from_env()
    assert_embedding_identity(config, require_device="cuda")
    report: dict[str, object] = {"identity": describe(config),
                             "pinned_revision": PINNED_REVISION}

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to fall back to CPU")
    torch.cuda.reset_peak_memory_stats()

    provider = create_embedding_provider(
        config, bge_encoder_factory=pinned_encoder_factory)
    started = time.perf_counter()
    provider.load()
    report["load_seconds"] = round(time.perf_counter() - started, 2)
    report["vram_after_load_MiB"] = round(torch.cuda.memory_allocated() / 2**20)

    texts = ["보유주식수 2,967,759", "보유주식수 2,967,759", "전혀 다른 문장입니다"]
    started = time.perf_counter()
    vectors = provider.embed_documents(texts)
    report["encode_seconds"] = round(time.perf_counter() - started, 3)

    dims = {len(v) for v in vectors}
    norms = [sum(x * x for x in v) ** 0.5 for v in vectors]
    finite = all(all(x == x and abs(x) != float("inf") for x in v) for v in vectors)
    same = sum(a * b for a, b in zip(vectors[0], vectors[1]))
    other = sum(a * b for a, b in zip(vectors[0], vectors[2]))

    report.update({
        "dimensions": sorted(dims),
        "all_finite": finite,
        "norms": [round(n, 6) for n in norms],
        "repeat_cosine": round(same, 9),
        "different_text_cosine": round(other, 6),
        "peak_vram_MiB": round(torch.cuda.max_memory_allocated() / 2**20),
        "cuda_device": torch.cuda.get_device_name(0),
    })

    failures = []
    if dims != {1024}:
        failures.append(f"expected 1024 dimensions, got {sorted(dims)}")
    if not finite:
        failures.append("non-finite values in embedding")
    if not all(abs(n - 1.0) < 1e-3 for n in norms):
        failures.append(f"vectors are not L2 normalized: {norms}")
    if abs(same - 1.0) > 1e-6:
        failures.append(f"repeated encoding not deterministic: cosine={same}")
    if other > 0.99:
        failures.append("different texts produced near-identical vectors")
    if report["peak_vram_MiB"] == 0:
        failures.append("no CUDA allocation observed; the model did not run on GPU")

    report["failures"] = failures
    report["status"] = "ok" if not failures else "failed"
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
