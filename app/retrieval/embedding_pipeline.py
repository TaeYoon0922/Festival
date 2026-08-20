"""Streaming, resumable subset embedding and PostgreSQL upsert orchestration."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from app.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingDimensionMismatch,
    EmbeddingProvider,
    chunk_embedding_text,
    embedding_retry_delay,
    should_retry_embedding_error,
)


class EmbeddingStore(Protocol):
    def existing_embedding_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        embedding_model: str,
        embedding_version: str,
        embedding_dimensions: int,
    ) -> set[str]: ...

    def upsert_embeddings(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        embedding_model: str,
        embedding_version: str,
        force: bool = False,
    ) -> int: ...


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts <= 0:
            raise ValueError("retry max_attempts must be positive")
        if self.initial_delay_seconds < 0.0 or self.max_delay_seconds < 0.0:
            raise ValueError("retry delays must be non-negative")


@dataclass(frozen=True)
class EmbeddingPipelineConfig:
    batch_size: int = 32
    limit: int | None = None
    force: bool = False
    dry_run: bool = False
    resume: bool = True
    inter_batch_delay_seconds: float = 0.0
    retry: RetryConfig = RetryConfig()

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.limit is not None and self.limit <= 0:
            raise ValueError("limit must be positive")
        if self.inter_batch_delay_seconds < 0.0:
            raise ValueError("inter-batch delay must be non-negative")


ProgressCallback = Callable[[Mapping[str, Any]], None]


class SubsetEmbeddingPipeline:
    def __init__(
        self,
        provider: EmbeddingProvider,
        store: EmbeddingStore,
        embedding_config: EmbeddingConfig,
        *,
        pipeline_config: EmbeddingPipelineConfig | None = None,
        error_path: Path | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if provider.config != embedding_config:
            raise ValueError("provider config must match embedding pipeline config")
        self.provider = provider
        self.store = store
        self.embedding_config = embedding_config
        self.pipeline_config = pipeline_config or EmbeddingPipelineConfig(
            batch_size=embedding_config.batch_size
        )
        self.error_path = error_path
        self.sleep = sleep
        self.clock = clock

    def run(
        self,
        candidates: Iterable[Mapping[str, Any]],
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        started = self.clock()
        state: dict[str, Any] = {
            "status": "completed",
            "input_rows": 0,
            "unique_chunk_ids": 0,
            "duplicate_chunk_ids": 0,
            "already_embedded": 0,
            "embedded": 0,
            "would_embed": 0,
            "failed": 0,
            "batches": 0,
            "embedding_batches": 0,
            "embedding_time_seconds": 0.0,
            "dry_run": self.pipeline_config.dry_run,
            "force": self.pipeline_config.force,
            "model": self.embedding_config.model,
            "version": self.embedding_config.version,
            "dimensions": self.embedding_config.dimensions,
        }
        seen: set[str] = set()
        batch: list[dict[str, str]] = []
        errors = _ErrorSink(self.error_path, append=self.pipeline_config.resume)
        try:
            with errors:
                for candidate in candidates:
                    if (
                        self.pipeline_config.limit is not None
                        and state["unique_chunk_ids"] >= self.pipeline_config.limit
                    ):
                        break
                    state["input_rows"] += 1
                    chunk_id = str(candidate.get("chunk_id") or "").strip()
                    if not chunk_id:
                        state["failed"] += 1
                        errors.write("", "candidate_validation", "missing chunk_id")
                        continue
                    if chunk_id in seen:
                        state["duplicate_chunk_ids"] += 1
                        continue
                    seen.add(chunk_id)
                    state["unique_chunk_ids"] += 1
                    try:
                        text = chunk_embedding_text(candidate)
                    except Exception as error:
                        state["failed"] += 1
                        errors.write(chunk_id, "candidate_validation", str(error))
                        continue
                    batch.append({"chunk_id": chunk_id, "retrieval_text": text})
                    if len(batch) >= self.pipeline_config.batch_size:
                        self._process_batch(batch, state, errors)
                        batch = []
                        if progress:
                            progress(dict(state))
                if batch:
                    self._process_batch(batch, state, errors)
                    if progress:
                        progress(dict(state))
        except KeyboardInterrupt:
            state["status"] = "interrupted"
        elapsed = max(self.clock() - started, 0.0)
        state["elapsed_seconds"] = round(elapsed, 6)
        state["chunks_per_second"] = (
            round(state["embedded"] / elapsed, 6) if elapsed > 0.0 else 0.0
        )
        state["average_embedding_latency_seconds"] = (
            round(
                state["embedding_time_seconds"] / state["embedding_batches"],
                6,
            )
            if state["embedding_batches"]
            else 0.0
        )
        state["embedding_time_seconds"] = round(
            state["embedding_time_seconds"], 6
        )
        state["processed"] = state["unique_chunk_ids"]
        state["skipped"] = state["already_embedded"]
        state["dimension"] = state["dimensions"]
        statistics = getattr(self.provider, "embedding_statistics", None)
        state["provider_statistics"] = (
            dict(statistics()) if callable(statistics) else {}
        )
        return state

    def _process_batch(
        self,
        batch: Sequence[Mapping[str, str]],
        state: dict[str, Any],
        errors: "_ErrorSink",
    ) -> None:
        chunk_ids = [row["chunk_id"] for row in batch]
        existing = set()
        if not self.pipeline_config.force:
            existing = self.store.existing_embedding_chunk_ids(
                chunk_ids,
                embedding_model=self.embedding_config.model,
                embedding_version=self.embedding_config.version,
                embedding_dimensions=self.embedding_config.dimensions,
            )
        missing = [row for row in batch if row["chunk_id"] not in existing]
        state["already_embedded"] += len(batch) - len(missing)
        state["batches"] += 1
        if not missing:
            return
        if self.pipeline_config.dry_run:
            state["would_embed"] += len(missing)
            return

        texts = [row["retrieval_text"] for row in missing]
        embedding_started = self.clock()
        state["embedding_batches"] += 1
        try:
            vectors = self._retry_call(lambda: self.provider.embed_documents(texts))
        except Exception as error:
            state["embedding_time_seconds"] += max(
                self.clock() - embedding_started, 0.0
            )
            state["failed"] += len(missing)
            for row in missing:
                errors.write(row["chunk_id"], "embedding", str(error))
            return
        state["embedding_time_seconds"] += max(
            self.clock() - embedding_started, 0.0
        )
        self._validate_vectors(vectors, len(missing))
        records = [
            {
                "chunk_id": row["chunk_id"],
                "embedding": vector,
                "embedding_dimensions": self.embedding_config.dimensions,
            }
            for row, vector in zip(missing, vectors)
        ]
        try:
            self._retry_call(
                lambda: self.store.upsert_embeddings(
                    records,
                    embedding_model=self.embedding_config.model,
                    embedding_version=self.embedding_config.version,
                    force=self.pipeline_config.force,
                )
            )
        except Exception as error:
            state["failed"] += len(missing)
            for row in missing:
                errors.write(row["chunk_id"], "database_upsert", str(error))
            return
        state["embedded"] += len(missing)
        if self.pipeline_config.inter_batch_delay_seconds:
            self.sleep(self.pipeline_config.inter_batch_delay_seconds)

    def _retry_call(self, operation: Callable[[], Any]) -> Any:
        delay = self.pipeline_config.retry.initial_delay_seconds
        for attempt in range(1, self.pipeline_config.retry.max_attempts + 1):
            try:
                return operation()
            except EmbeddingDimensionMismatch:
                raise
            except Exception as error:
                if (
                    attempt >= self.pipeline_config.retry.max_attempts
                    or not should_retry_embedding_error(error)
                ):
                    raise
                wait_seconds = embedding_retry_delay(
                    error,
                    delay,
                    self.pipeline_config.retry.max_delay_seconds,
                )
                if wait_seconds:
                    self.sleep(wait_seconds)
                delay = min(
                    max(delay * 2.0, self.pipeline_config.retry.initial_delay_seconds),
                    self.pipeline_config.retry.max_delay_seconds,
                )
        raise AssertionError("unreachable retry state")

    def _validate_vectors(
        self, vectors: Sequence[Sequence[float]], expected_count: int
    ) -> None:
        if len(vectors) != expected_count:
            raise EmbeddingDimensionMismatch(
                f"embedding count mismatch: expected {expected_count}, got {len(vectors)}"
            )
        for vector in vectors:
            if len(vector) != self.embedding_config.dimensions:
                raise EmbeddingDimensionMismatch(
                    "embedding dimension mismatch: expected "
                    f"{self.embedding_config.dimensions}, got {len(vector)}"
                )
            if any(not math.isfinite(float(value)) for value in vector):
                raise EmbeddingDimensionMismatch("embedding contains a non-finite value")


def iter_candidate_file(
    path: Path,
    *,
    backend: Any | None = None,
    fetch_batch_size: int = 1000,
) -> Iterator[Mapping[str, Any]]:
    """Stream JSONL rows or hydrate a text file of IDs from ``retrieval_text``."""

    if path.suffix.casefold() == ".jsonl":
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"candidate JSONL line {line_number} is not an object")
                yield value
        return
    if backend is None:
        raise ValueError("a backend is required for candidate chunk ID files")
    if fetch_batch_size <= 0:
        raise ValueError("fetch_batch_size must be positive")
    with path.open("r", encoding="utf-8") as source:
        pending: list[str] = []
        for line in source:
            chunk_id = line.strip()
            if not chunk_id:
                continue
            pending.append(chunk_id)
            if len(pending) >= fetch_batch_size:
                yield from _hydrate_ids(pending, backend)
                pending = []
        if pending:
            yield from _hydrate_ids(pending, backend)


def _hydrate_ids(chunk_ids: Sequence[str], backend: Any) -> Iterator[Mapping[str, Any]]:
    rows = backend.fetch_embedding_source_chunks(chunk_ids)
    rows_by_id = {str(row["chunk_id"]): row for row in rows}
    for chunk_id in chunk_ids:
        yield rows_by_id.get(chunk_id, {"chunk_id": chunk_id})


class _ErrorSink:
    def __init__(self, path: Path | None, *, append: bool) -> None:
        self.path = path
        self.append = append
        self._target: Any | None = None

    def __enter__(self) -> "_ErrorSink":
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._target = self.path.open(
                "a" if self.append else "w", encoding="utf-8"
            )
        return self

    def __exit__(self, *_args: object) -> None:
        if self._target is not None:
            self._target.close()

    def write(self, chunk_id: str, stage: str, error: str) -> None:
        if self._target is None:
            return
        self._target.write(
            json.dumps(
                {"chunk_id": chunk_id, "stage": stage, "error": error},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )
        self._target.flush()
