"""Provider-neutral embedding contracts and deterministic local test support."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "hash"
    model: str = "festival-hash-embedding"
    version: str = "v1"
    dimensions: int = 1024
    batch_size: int = 32

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip() or not self.version.strip():
            raise ValueError("embedding provider, model, and version must not be empty")
        if not 1 <= self.dimensions <= 2000:
            raise ValueError("embedding dimensions must be between 1 and 2000")
        if self.batch_size <= 0:
            raise ValueError("embedding batch size must be positive")

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "EmbeddingConfig":
        values = os.environ if environment is None else environment
        return cls(
            provider=values.get("FESTIVAL_EMBEDDING_PROVIDER", "hash"),
            model=values.get("FESTIVAL_EMBEDDING_MODEL", "festival-hash-embedding"),
            version=values.get("FESTIVAL_EMBEDDING_VERSION", "v1"),
            dimensions=int(values.get("FESTIVAL_EMBEDDING_DIMENSIONS", "1024")),
            batch_size=int(values.get("FESTIVAL_EMBEDDING_BATCH_SIZE", "32")),
        )


class EmbeddingProvider(Protocol):
    config: EmbeddingConfig

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


class EmbeddingDimensionMismatch(ValueError):
    """Raised when provider output cannot match the configured vector index."""


class JsonHttpTransport(Protocol):
    """Injectable JSON transport so provider tests never need the network."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class UrllibJsonTransport:
    """Small standard-library transport used only by an explicitly configured provider."""

    def post_json(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", **dict(headers)},
            method="POST",
        )
        with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("embedding endpoint must return a JSON object")
        return value


@dataclass(frozen=True)
class HttpEmbeddingSettings:
    endpoint: str
    api_key: str = field(repr=False)
    api_key_header: str = "Authorization"
    api_key_prefix: str = "Bearer"
    timeout_seconds: float = 60.0

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("embedding API endpoint must not be empty")
        if urlparse(self.endpoint).scheme not in {"http", "https"}:
            raise ValueError("embedding API endpoint must use HTTP or HTTPS")
        if not self.api_key.strip():
            raise ValueError("embedding API key must not be empty")
        if not self.api_key_header.strip():
            raise ValueError("embedding API key header must not be empty")
        if self.timeout_seconds <= 0.0:
            raise ValueError("embedding API timeout must be positive")

    @classmethod
    def from_env(
        cls, environment: Mapping[str, str] | None = None
    ) -> "HttpEmbeddingSettings":
        values = os.environ if environment is None else environment
        return cls(
            endpoint=values.get("FESTIVAL_EMBEDDING_API_URL", ""),
            api_key=values.get("FESTIVAL_EMBEDDING_API_KEY", ""),
            api_key_header=values.get(
                "FESTIVAL_EMBEDDING_API_KEY_HEADER", "Authorization"
            ),
            api_key_prefix=values.get("FESTIVAL_EMBEDDING_API_KEY_PREFIX", "Bearer"),
            timeout_seconds=float(
                values.get("FESTIVAL_EMBEDDING_TIMEOUT_SECONDS", "60")
            ),
        )

    def request_headers(self) -> dict[str, str]:
        value = " ".join(
            part for part in (self.api_key_prefix.strip(), self.api_key) if part
        )
        return {self.api_key_header: value}


class OpenAICompatibleEmbeddingProvider:
    """Production-ready adapter for the common ``/embeddings`` JSON contract."""

    def __init__(
        self,
        config: EmbeddingConfig,
        settings: HttpEmbeddingSettings,
        *,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        self.config = config
        self.settings = settings
        self.transport = transport or UrllibJsonTransport()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.transport.post_json(
            self.settings.endpoint,
            headers=self.settings.request_headers(),
            payload={"model": self.config.model, "input": list(texts)},
            timeout_seconds=self.settings.timeout_seconds,
        )
        data = response.get("data")
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            raise ValueError("embedding response must contain a data array")
        ordered: list[tuple[int, list[float]]] = []
        for fallback_index, item in enumerate(data):
            embedding = item.get("embedding") if isinstance(item, Mapping) else None
            if not isinstance(item, Mapping) or not isinstance(
                embedding, Sequence
            ) or isinstance(embedding, (str, bytes)):
                raise ValueError("embedding response item is malformed")
            index = int(item.get("index", fallback_index))
            vector = [float(value) for value in item["embedding"]]
            _validate_embedding(vector, self.config.dimensions)
            ordered.append((index, vector))
        ordered.sort(key=lambda item: item[0])
        if len(ordered) != len(texts):
            raise ValueError("embedding response count does not match input count")
        if [index for index, _ in ordered] != list(range(len(texts))):
            raise ValueError("embedding response indexes do not match input order")
        return [vector for _, vector in ordered]


class DeterministicHashEmbedder:
    """Dependency-free local/mock embedder; not intended as a semantic model."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config or EmbeddingConfig()

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.config.dimensions
        tokens = re.findall(r"[0-9A-Za-z가-힣]+", text.casefold())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:8], "big") % self.config.dimensions
            vector[index] += 1.0 if digest[8] & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


def chunk_embedding_text(chunk: Mapping[str, object]) -> str:
    """Return the frozen retrieval text used for document embeddings."""

    value = str(chunk.get("retrieval_text") or "").strip()
    if not value:
        raise ValueError("chunk retrieval_text must not be empty")
    return value


def create_embedding_provider(
    config: EmbeddingConfig,
    *,
    environment: Mapping[str, str] | None = None,
    transport: JsonHttpTransport | None = None,
) -> EmbeddingProvider:
    """Create a configured adapter without embedding any text."""

    provider = config.provider.strip().casefold().replace("-", "_")
    if provider == "hash":
        return DeterministicHashEmbedder(config)
    if provider in {"openai", "openai_compatible", "http"}:
        return OpenAICompatibleEmbeddingProvider(
            config,
            HttpEmbeddingSettings.from_env(environment),
            transport=transport,
        )
    raise ValueError(f"unsupported embedding provider: {config.provider}")


def _validate_embedding(vector: Sequence[float], dimensions: int) -> None:
    if len(vector) != dimensions:
        raise EmbeddingDimensionMismatch(
            f"embedding dimension mismatch: expected {dimensions}, got {len(vector)}"
        )
    if any(not math.isfinite(float(value)) for value in vector):
        raise EmbeddingDimensionMismatch("embedding values must be finite")
