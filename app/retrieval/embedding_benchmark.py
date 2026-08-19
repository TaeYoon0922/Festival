"""DB-free embedding throughput benchmark for a streamed candidate subset."""

from __future__ import annotations

import math
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.embedding_pipeline import RetryConfig
from app.retrieval.embeddings import (
    EmbeddingDimensionMismatch,
    EmbeddingProvider,
    chunk_embedding_text,
    embedding_retry_delay,
    should_retry_embedding_error,
)


GOLD60_CANDIDATE_CHUNKS = 76_438
FULL_CORPUS_CHUNKS = 1_363_336
CPU_BGE_M3_BASELINE_DOCUMENTS_PER_SECOND = 1.489


@dataclass(frozen=True)
class EmbeddingBenchmarkConfig:
    limit: int = 10
    batch_size: int = 4
    gold60_candidate_count: int = GOLD60_CANDIDATE_CHUNKS
    full_corpus_count: int = FULL_CORPUS_CHUNKS
    cpu_baseline_documents_per_second: float | None = (
        CPU_BGE_M3_BASELINE_DOCUMENTS_PER_SECOND
    )
    retry: RetryConfig = field(default_factory=RetryConfig)

    def __post_init__(self) -> None:
        if self.limit <= 0 or self.batch_size <= 0:
            raise ValueError("benchmark limit and batch_size must be positive")
        if self.gold60_candidate_count < 0 or self.full_corpus_count < 0:
            raise ValueError("benchmark estimate counts must be non-negative")
        if (
            self.cpu_baseline_documents_per_second is not None
            and self.cpu_baseline_documents_per_second <= 0.0
        ):
            raise ValueError("CPU baseline throughput must be positive")


class EmbeddingSubsetBenchmark:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        config: EmbeddingBenchmarkConfig | None = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        peak_memory: Callable[[], int | None] | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or EmbeddingBenchmarkConfig(
            batch_size=provider.config.batch_size
        )
        self.clock = clock
        self.sleep = sleep
        self.peak_memory = peak_memory or process_peak_memory_bytes

    def run(self, candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        load_started = self.clock()
        load = getattr(self.provider, "load", None)
        if callable(load):
            load()
        model_load_seconds = max(self.clock() - load_started, 0.0)

        processed = 0
        failed = 0
        embedded = 0
        seen: set[str] = set()
        batch: list[str] = []
        latencies: list[float] = []
        for candidate in candidates:
            if processed >= self.config.limit:
                break
            chunk_id = str(candidate.get("chunk_id") or "").strip()
            if chunk_id and chunk_id in seen:
                continue
            if chunk_id:
                seen.add(chunk_id)
            processed += 1
            try:
                batch.append(chunk_embedding_text(candidate))
            except Exception:
                failed += 1
                continue
            if len(batch) >= self.config.batch_size:
                success, latency = self._embed_batch(batch)
                latencies.append(latency)
                if success:
                    embedded += len(batch)
                else:
                    failed += len(batch)
                batch = []
        if batch:
            success, latency = self._embed_batch(batch)
            latencies.append(latency)
            if success:
                embedded += len(batch)
            else:
                failed += len(batch)

        embedding_seconds = sum(latencies)
        documents_per_second = (
            embedded / embedding_seconds if embedding_seconds > 0.0 else 0.0
        )
        peak_bytes = self.peak_memory()
        device_peak_bytes = _provider_peak_device_memory(self.provider)
        estimated_gold60_seconds = _estimated_seconds(
            self.config.gold60_candidate_count, documents_per_second
        )
        estimated_full_seconds = _estimated_seconds(
            self.config.full_corpus_count, documents_per_second
        )
        baseline = self.config.cpu_baseline_documents_per_second
        return {
            "provider": self.provider.config.provider,
            "model": self.provider.config.model,
            "version": self.provider.config.version,
            "dimension": self.provider.config.dimensions,
            "device": self.provider.config.device,
            "max_length": self.provider.config.max_length,
            "batch_size": self.config.batch_size,
            "requested_batch_size": self.provider.config.batch_size,
            "effective_batch_size": int(
                getattr(
                    self.provider,
                    "effective_batch_size",
                    self.provider.config.batch_size,
                )
            ),
            "cuda_oom_retries": int(getattr(self.provider, "oom_retries", 0)),
            "limit": self.config.limit,
            "processed": processed,
            "embedded": embedded,
            "failed": failed,
            "model_load_seconds": round(model_load_seconds, 6),
            "embedding_seconds": round(embedding_seconds, 6),
            "documents_per_second": round(documents_per_second, 6),
            "cpu_baseline_documents_per_second": baseline,
            "speedup_vs_cpu": (
                round(documents_per_second / baseline, 3)
                if documents_per_second > 0.0 and baseline is not None
                else None
            ),
            "latency_seconds": _latency_summary(latencies),
            "estimated_gold60_subset_seconds": estimated_gold60_seconds,
            "estimated_gold60_subset_hours": _seconds_to_hours(
                estimated_gold60_seconds
            ),
            "estimated_full_corpus_seconds": estimated_full_seconds,
            "estimated_full_corpus_hours": _seconds_to_hours(estimated_full_seconds),
            "estimated_full_corpus_days": _seconds_to_days(estimated_full_seconds),
            "peak_memory_bytes": peak_bytes,
            "peak_memory_mib": (
                round(peak_bytes / (1024**2), 3) if peak_bytes is not None else None
            ),
            "peak_device_memory_bytes": device_peak_bytes,
            "peak_device_memory_mib": (
                round(device_peak_bytes / (1024**2), 3)
                if device_peak_bytes is not None
                else None
            ),
            "database_writes": 0,
        }

    def _embed_batch(self, texts: Sequence[str]) -> tuple[bool, float]:
        started = self.clock()
        delay = self.config.retry.initial_delay_seconds
        for attempt in range(1, self.config.retry.max_attempts + 1):
            try:
                vectors = self.provider.embed_documents(texts)
                if len(vectors) != len(texts):
                    raise ValueError("benchmark embedding count mismatch")
                return True, max(self.clock() - started, 0.0)
            except EmbeddingDimensionMismatch:
                raise
            except Exception as error:
                if (
                    attempt >= self.config.retry.max_attempts
                    or not should_retry_embedding_error(error)
                ):
                    return False, max(self.clock() - started, 0.0)
                wait_seconds = embedding_retry_delay(
                    error, delay, self.config.retry.max_delay_seconds
                )
                if wait_seconds:
                    self.sleep(wait_seconds)
                delay = min(
                    max(delay * 2.0, self.config.retry.initial_delay_seconds),
                    self.config.retry.max_delay_seconds,
                )
        raise AssertionError("unreachable retry state")


def _latency_summary(latencies: Sequence[float]) -> dict[str, float]:
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(latencies)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "mean": round(statistics.mean(ordered), 6),
        "p50": round(float(statistics.median(ordered)), 6),
        "p95": round(ordered[p95_index], 6),
    }


def _estimated_seconds(count: int, throughput: float) -> float | None:
    return round(count / throughput, 3) if throughput > 0.0 else None


def _seconds_to_hours(seconds: float | None) -> float | None:
    return round(seconds / 3600.0, 3) if seconds is not None else None


def _seconds_to_days(seconds: float | None) -> float | None:
    return round(seconds / 86400.0, 3) if seconds is not None else None


def _provider_peak_device_memory(provider: EmbeddingProvider) -> int | None:
    reader = getattr(provider, "peak_device_memory_bytes", None)
    if not callable(reader):
        return None
    try:
        return reader()
    except Exception:
        return None


def process_peak_memory_bytes() -> int | None:
    """Return process peak RSS where available; PyTorch allocator details may differ."""

    try:
        import resource
    except ImportError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024
