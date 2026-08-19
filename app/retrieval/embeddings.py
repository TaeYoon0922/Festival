"""Provider-neutral embedding contracts and deterministic local test support."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "hash"
    model: str = "festival-hash-embedding"
    version: str = "v1"
    dimensions: int = 1024

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip() or not self.version.strip():
            raise ValueError("embedding provider, model, and version must not be empty")
        if not 1 <= self.dimensions <= 2000:
            raise ValueError("embedding dimensions must be between 1 and 2000")

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None) -> "EmbeddingConfig":
        values = os.environ if environment is None else environment
        return cls(
            provider=values.get("FESTIVAL_EMBEDDING_PROVIDER", "hash"),
            model=values.get("FESTIVAL_EMBEDDING_MODEL", "festival-hash-embedding"),
            version=values.get("FESTIVAL_EMBEDDING_VERSION", "v1"),
            dimensions=int(values.get("FESTIVAL_EMBEDDING_DIMENSIONS", "1024")),
        )


class EmbeddingProvider(Protocol):
    config: EmbeddingConfig

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...


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
