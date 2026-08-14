"""Backend-independent metadata and retrieval contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class MetadataMatch:
    hard_filters: Mapping[str, Any] = field(default_factory=dict)
    soft_boosts: Mapping[str, bool] = field(default_factory=dict)
    soft_inputs: Mapping[str, Any] = field(default_factory=dict)
    soft_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hard_filters": dict(self.hard_filters),
            "soft_boosts": dict(self.soft_boosts),
            "soft_inputs": dict(self.soft_inputs),
            "soft_score": self.soft_score,
        }


@dataclass(frozen=True)
class CandidateDocument:
    doc_id: str
    metadata: Mapping[str, Any]
    metadata_match: MetadataMatch


@dataclass(frozen=True)
class CandidateChunk:
    chunk_id: str
    doc_id: str
    chunk: Mapping[str, Any]
    metadata_match: MetadataMatch


@dataclass(frozen=True)
class RetrievalResult:
    chunk_id: str
    doc_id: str
    bm25_score: float
    rank: int
    metadata_match: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "bm25_score": self.bm25_score,
            "rank": self.rank,
            "metadata_match": dict(self.metadata_match),
        }


class MetadataBackend(Protocol):
    def get_candidate_documents(
        self,
        company: str | Sequence[str] | None = None,
        year: int | Sequence[int] | None = None,
        period: int | tuple[int, int] | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        is_correction: bool | None = None,
        *,
        corp_code: str | Sequence[str] | None = None,
        section_path: str | None = None,
    ) -> list[CandidateDocument]: ...


class ChunkBackend(Protocol):
    def get_candidate_chunks(
        self, documents: Iterable[CandidateDocument]
    ) -> list[CandidateChunk]: ...


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        candidates: Sequence[CandidateChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]: ...
