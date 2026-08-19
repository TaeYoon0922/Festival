"""Dense-only BGE-M3 adapters for local FlagEmbedding and HTTP inference."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

from app.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingDimensionMismatch,
    HttpEmbeddingSettings,
    JsonHttpTransport,
    UrllibJsonTransport,
)


BGE_M3_MODEL = "BAAI/bge-m3"
BGE_M3_DIMENSIONS = 1024
BGE_M3_MAX_LENGTH = 8192


class BgeM3Encoder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs: Any) -> Any: ...


EncoderFactory = Callable[[EmbeddingConfig], BgeM3Encoder]


class BgeM3LocalEmbeddingProvider:
    """Lazy local FlagEmbedding adapter; dense vectors only."""

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        encoder: BgeM3Encoder | None = None,
        encoder_factory: EncoderFactory | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        _validate_bge_config(config)
        self.config = config
        self._encoder = encoder
        self._encoder_factory = encoder_factory or _load_flag_embedding_encoder
        self._clock = clock
        self.load_time_seconds = 0.0

    def load(self) -> None:
        self._ensure_encoder()

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts)

    def _ensure_encoder(self) -> BgeM3Encoder:
        if self._encoder is None:
            started = self._clock()
            self._encoder = self._encoder_factory(self.config)
            self.load_time_seconds = self._clock() - started
        return self._encoder

    def _encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        output = self._ensure_encoder().encode(
            list(texts),
            batch_size=self.config.batch_size,
            max_length=self.config.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return _normalized_dense_vectors(output, self.config.dimensions, len(texts))


class BgeM3HttpEmbeddingProvider:
    """HTTP adapter for a dense-only service wrapping BGE-M3 inference."""

    def __init__(
        self,
        config: EmbeddingConfig,
        settings: HttpEmbeddingSettings,
        *,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        _validate_bge_config(config)
        self.config = config
        self.settings = settings
        self.transport = transport or UrllibJsonTransport()
        self.load_time_seconds = 0.0

    def load(self) -> None:
        """The remote service owns model loading; no request is made here."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.transport.post_json(
            self.settings.endpoint,
            headers=self.settings.request_headers(),
            payload={
                "model": self.config.model,
                "version": self.config.version,
                "texts": list(texts),
                "batch_size": self.config.batch_size,
                "max_length": self.config.max_length,
                "normalize_embeddings": True,
                "return_dense": True,
                "return_sparse": False,
                "return_colbert_vecs": False,
            },
            timeout_seconds=self.settings.timeout_seconds,
        )
        return _normalized_dense_vectors(response, self.config.dimensions, len(texts))


def _load_flag_embedding_encoder(config: EmbeddingConfig) -> BgeM3Encoder:
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as error:
        raise RuntimeError(
            "bge_m3_local requires the optional requirements-embedding.txt dependencies"
        ) from error
    return BGEM3FlagModel(
        config.model,
        normalize_embeddings=True,
        use_fp16=config.device.casefold().startswith("cuda"),
        devices=config.device,
        revision=config.version,
    )


def _validate_bge_config(config: EmbeddingConfig) -> None:
    if config.dimensions != BGE_M3_DIMENSIONS:
        raise EmbeddingDimensionMismatch(
            f"BGE-M3 dense embeddings require {BGE_M3_DIMENSIONS} dimensions"
        )
    if not 1 <= config.max_length <= BGE_M3_MAX_LENGTH:
        raise ValueError(
            f"BGE-M3 max_length must be between 1 and {BGE_M3_MAX_LENGTH}"
        )


def _normalized_dense_vectors(
    output: Any, dimensions: int, expected_count: int
) -> list[list[float]]:
    value = output
    if isinstance(output, Mapping):
        for key in ("dense_vecs", "embeddings", "vectors"):
            if key in output:
                value = output[key]
                break
        else:
            raise ValueError("BGE-M3 response does not contain dense vectors")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("BGE-M3 dense vectors must be an array")
    if len(value) != expected_count:
        raise ValueError(
            f"BGE-M3 response count mismatch: expected {expected_count}, got {len(value)}"
        )
    vectors: list[list[float]] = []
    for item in value:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)):
            raise ValueError("BGE-M3 dense vector is malformed")
        vector = [float(component) for component in item]
        if len(vector) != dimensions:
            raise EmbeddingDimensionMismatch(
                f"BGE-M3 dimension mismatch: expected {dimensions}, got {len(vector)}"
            )
        if any(not math.isfinite(component) for component in vector):
            raise EmbeddingDimensionMismatch("BGE-M3 vector contains a non-finite value")
        norm = math.sqrt(sum(component * component for component in vector))
        if norm <= 0.0:
            raise EmbeddingDimensionMismatch("BGE-M3 returned a zero dense vector")
        vectors.append([component / norm for component in vector])
    return vectors
