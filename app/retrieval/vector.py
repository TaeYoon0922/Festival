"""Vector retrieval result and backend contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.retrieval.interfaces import CandidateChunk


@dataclass(frozen=True)
class VectorRetrievalResult:
    chunk_id: str
    doc_id: str
    vector_score: float
    rank: int

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "vector_score": self.vector_score,
            "rank": self.rank,
        }


class VectorRetriever(Protocol):
    def vector_search(
        self,
        query_embedding: Sequence[float],
        candidates: Sequence[CandidateChunk],
        *,
        embedding_model: str,
        embedding_version: str,
        top_k: int = 50,
    ) -> list[VectorRetrievalResult]: ...


class VectorSearchUnavailable(RuntimeError):
    """Raised by optional vector backends when vector search cannot be used."""
