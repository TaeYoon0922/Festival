"""Metadata-scoped lexical/vector retrieval with RRF and deterministic reranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.reasoning.router import QueryRouter
from app.retrieval.embeddings import EmbeddingConfig, EmbeddingProvider
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    ChunkBackend,
    MetadataBackend,
    RetrievalResult,
    Retriever,
)
from app.retrieval.vector import VectorRetrievalResult, VectorRetriever


DIAGNOSTIC_WEIGHT_GRID: tuple[tuple[float, float], ...] = (
    (0.60, 0.40),
    (0.70, 0.30),
    (0.75, 0.25),
    (0.80, 0.20),
)


@dataclass(frozen=True)
class RRFConfig:
    k: int = 60
    lexical_weight: float = 1.0
    vector_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError("RRF k must be positive")
        if self.lexical_weight < 0.0 or self.vector_weight < 0.0:
            raise ValueError("RRF weights must be non-negative")
        if self.lexical_weight == 0.0 and self.vector_weight == 0.0:
            raise ValueError("at least one RRF weight must be positive")


@dataclass(frozen=True)
class HybridRetrievalConfig:
    lexical_top_n: int = 50
    vector_top_n: int = 50
    final_top_k: int = 10
    fusion_weight: float = 0.60
    deterministic_weight: float = 0.40
    rerank_window_size: int = 2
    fallback_on_vector_error: bool = True
    rrf: RRFConfig = field(default_factory=RRFConfig)

    def __post_init__(self) -> None:
        if min(self.lexical_top_n, self.vector_top_n, self.final_top_k) <= 0:
            raise ValueError("hybrid retrieval limits must be positive")
        if self.rerank_window_size <= 0:
            raise ValueError("hybrid rerank window size must be positive")
        if self.fusion_weight < 0.0 or self.deterministic_weight < 0.0:
            raise ValueError("hybrid final weights must be non-negative")
        total = self.fusion_weight + self.deterministic_weight
        if total <= 0.0:
            raise ValueError("at least one hybrid final weight must be positive")
        object.__setattr__(self, "fusion_weight", self.fusion_weight / total)
        object.__setattr__(self, "deterministic_weight", self.deterministic_weight / total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lexical_top_n": self.lexical_top_n,
            "vector_top_n": self.vector_top_n,
            "final_top_k": self.final_top_k,
            "fusion_weight": self.fusion_weight,
            "deterministic_weight": self.deterministic_weight,
            "rerank_window_size": self.rerank_window_size,
            "fallback_on_vector_error": self.fallback_on_vector_error,
            "diagnostic_weight_grid": [list(pair) for pair in DIAGNOSTIC_WEIGHT_GRID],
            "rrf": {
                "k": self.rrf.k,
                "lexical_weight": self.rrf.lexical_weight,
                "vector_weight": self.rrf.vector_weight,
            },
        }


@dataclass(frozen=True)
class FusedCandidate:
    chunk_id: str
    doc_id: str
    rrf_score: float
    fusion_rank: int
    lexical_rank: int | None = None
    lexical_score: float | None = None
    vector_rank: int | None = None
    vector_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "lexical_rank": self.lexical_rank,
            "lexical_score": self.lexical_score,
            "vector_rank": self.vector_rank,
            "vector_score": self.vector_score,
            "rrf_score": self.rrf_score,
            "fusion_rank": self.fusion_rank,
        }


@dataclass(frozen=True)
class HybridQueryExecution:
    plan: Any
    documents: Sequence[CandidateDocument]
    chunks: Sequence[CandidateChunk]
    lexical_results: Sequence[RetrievalResult]
    lexical_final_results: Sequence[RetrievalResult]
    vector_results: Sequence[VectorRetrievalResult]
    fused_candidates: Sequence[FusedCandidate]
    results: Sequence[RetrievalResult]
    routing: Mapping[str, Any]
    vector_status: str
    vector_error: str | None = None
    vector_coverage: Mapping[str, Any] = field(default_factory=dict)
    embedded_candidate_ids: Sequence[str] = ()
    rerank_diagnostics: Sequence[Mapping[str, Any]] = ()


def reciprocal_rank_fusion(
    lexical_results: Sequence[RetrievalResult],
    vector_results: Sequence[VectorRetrievalResult],
    config: RRFConfig | None = None,
) -> list[FusedCandidate]:
    settings = config or RRFConfig()
    rows: dict[str, dict[str, Any]] = {}
    for result in sorted(lexical_results, key=lambda item: (item.rank, item.chunk_id)):
        row = rows.setdefault(result.chunk_id, {"doc_id": result.doc_id})
        if row.get("lexical_rank") is None:
            row["lexical_rank"] = result.rank
            row["lexical_score"] = float(result.bm25_score)
    for result in sorted(vector_results, key=lambda item: (item.rank, item.chunk_id)):
        row = rows.setdefault(result.chunk_id, {"doc_id": result.doc_id})
        if row.get("vector_rank") is None:
            row["vector_rank"] = result.rank
            row["vector_score"] = float(result.vector_score)

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for chunk_id, row in rows.items():
        score = 0.0
        if row.get("lexical_rank") is not None:
            score += settings.lexical_weight / (settings.k + row["lexical_rank"])
        if row.get("vector_rank") is not None:
            score += settings.vector_weight / (settings.k + row["vector_rank"])
        scored.append((score, chunk_id, row))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [
        FusedCandidate(
            chunk_id=chunk_id,
            doc_id=str(row["doc_id"]),
            rrf_score=score,
            fusion_rank=rank,
            lexical_rank=row.get("lexical_rank"),
            lexical_score=row.get("lexical_score"),
            vector_rank=row.get("vector_rank"),
            vector_score=row.get("vector_score"),
        )
        for rank, (score, chunk_id, row) in enumerate(scored, start=1)
    ]


class HybridQueryExecutor:
    """Run lexical and vector search in one routed candidate universe."""

    def __init__(
        self,
        metadata_backend: MetadataBackend,
        embedder: EmbeddingProvider,
        embedding_config: EmbeddingConfig,
        *,
        chunk_backend: ChunkBackend | None = None,
        lexical_retriever: Retriever | None = None,
        vector_retriever: VectorRetriever | None = None,
        router: QueryRouter | None = None,
        config: HybridRetrievalConfig | None = None,
    ) -> None:
        self._metadata_backend = metadata_backend
        self._chunk_backend = (
            chunk_backend
            if chunk_backend is not None
            else _require_method(metadata_backend, "get_candidate_chunks")
        )
        self._lexical_retriever = (
            lexical_retriever
            if lexical_retriever is not None
            else _require_method(metadata_backend, "retrieve")
        )
        self._vector_retriever = (
            vector_retriever
            if vector_retriever is not None
            else _require_method(metadata_backend, "vector_search")
        )
        self._embedder = embedder
        self.embedding_config = embedding_config
        if embedder.config != embedding_config:
            raise ValueError("embedder config must match the vector index config")
        self.router = router or QueryRouter()
        self.config = config or HybridRetrievalConfig()

    def execute(self, plan: Any) -> HybridQueryExecution:
        route = self.router.route(plan)
        documents = self._metadata_backend.get_candidate_documents(**route.backend_filters)
        documents = self.router.filter_documents(documents, route)
        chunks = self._chunk_backend.get_candidate_chunks(documents)
        chunks = self.router.prepare_chunks(chunks, route)
        document_metadata = {document.doc_id: document.metadata for document in documents}
        embedded_candidate_ids, vector_coverage = self._vector_coverage(chunks)

        lexical_results = self._lexical_retriever.retrieve(
            plan.lexical_query,
            chunks,
            top_k=self.config.lexical_top_n,
        )
        lexical_final = self.router.rerank(
            lexical_results,
            route,
            chunks=chunks,
            document_metadata=document_metadata,
            top_k=min(plan.top_k, self.config.final_top_k),
        )

        vector_results: list[VectorRetrievalResult] = []
        vector_status = "ok"
        vector_error: str | None = None
        if vector_coverage.get("available") and not vector_coverage.get(
            "embedded_count"
        ):
            vector_status = "no_coverage"
        else:
            try:
                query_embedding = self._embedder.embed_query(plan.lexical_query)
                vector_results = self._vector_retriever.vector_search(
                    query_embedding,
                    chunks,
                    embedding_model=self.embedding_config.model,
                    embedding_version=self.embedding_config.version,
                    top_k=self.config.vector_top_n,
                )
                if not vector_results:
                    vector_status = "empty"
            except Exception as error:
                if not self.config.fallback_on_vector_error:
                    raise
                vector_status = "unavailable"
                vector_error = f"{type(error).__name__}: {error}"

        fused = reciprocal_rank_fusion(lexical_results, vector_results, self.config.rrf)
        rerank_diagnostics: Sequence[Mapping[str, Any]] = ()
        if not vector_results:
            final_results = _annotate_lexical_fallback(lexical_final, fused, vector_status)
        else:
            final_results, rerank_diagnostics = self._hybrid_rerank(
                fused,
                route,
                chunks,
                document_metadata,
                top_k=min(plan.top_k, self.config.final_top_k),
            )
        routing = {
            **route.to_dict(),
            "hybrid": {
                "config": self.config.to_dict(),
                "embedding": {
                    "provider": self.embedding_config.provider,
                    "model": self.embedding_config.model,
                    "version": self.embedding_config.version,
                    "dimensions": self.embedding_config.dimensions,
                },
                "vector_status": vector_status,
                "vector_error": vector_error,
                "coverage": vector_coverage,
            },
        }
        return HybridQueryExecution(
            plan=plan,
            documents=documents,
            chunks=chunks,
            lexical_results=lexical_results,
            lexical_final_results=lexical_final,
            vector_results=vector_results,
            fused_candidates=fused,
            results=final_results,
            routing=routing,
            vector_status=vector_status,
            vector_error=vector_error,
            vector_coverage=vector_coverage,
            embedded_candidate_ids=embedded_candidate_ids,
            rerank_diagnostics=rerank_diagnostics,
        )

    def _vector_coverage(
        self, chunks: Sequence[CandidateChunk]
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        candidate_ids = tuple(dict.fromkeys(chunk.chunk_id for chunk in chunks))
        lookup = getattr(self._vector_retriever, "existing_embedding_chunk_ids", None)
        if not callable(lookup):
            return (), {
                "available": False,
                "candidate_count": len(candidate_ids),
                "embedded_count": None,
                "ratio": None,
            }
        try:
            embedded = set(
                lookup(
                    candidate_ids,
                    embedding_model=self.embedding_config.model,
                    embedding_version=self.embedding_config.version,
                    embedding_dimensions=self.embedding_config.dimensions,
                )
            )
        except Exception as error:
            return (), {
                "available": False,
                "candidate_count": len(candidate_ids),
                "embedded_count": None,
                "ratio": None,
                "error": f"{type(error).__name__}: {error}",
            }
        embedded_ids = tuple(sorted(embedded.intersection(candidate_ids)))
        return embedded_ids, {
            "available": True,
            "candidate_count": len(candidate_ids),
            "embedded_count": len(embedded_ids),
            "ratio": (
                round(len(embedded_ids) / len(candidate_ids), 6)
                if candidate_ids
                else 0.0
            ),
        }

    def _hybrid_rerank(
        self,
        fused: Sequence[FusedCandidate],
        route: Any,
        chunks: Sequence[CandidateChunk],
        document_metadata: Mapping[str, Mapping[str, Any]],
        *,
        top_k: int,
    ) -> tuple[list[RetrievalResult], list[dict[str, Any]]]:
        chunks_by_id = {candidate.chunk_id: candidate for candidate in chunks}
        max_rrf = max((candidate.rrf_score for candidate in fused), default=0.0)
        scored: list[dict[str, Any]] = []
        for candidate in fused:
            source = chunks_by_id.get(candidate.chunk_id)
            if source is None:
                continue
            components = self.router.deterministic_components(
                route,
                chunk=source.chunk,
                metadata_match=source.metadata_match.to_dict(),
                document_metadata=document_metadata.get(source.doc_id, {}),
            )
            deterministic_score = self.router.deterministic_score(components)
            normalized_rrf = candidate.rrf_score / max_rrf if max_rrf > 0.0 else 0.0
            source_rank_score = _best_source_rank_score(candidate, self.config.rrf.k)
            retrieval_score = max(normalized_rrf, source_rank_score)
            final_score = (
                self.config.fusion_weight * retrieval_score
                + self.config.deterministic_weight * deterministic_score
            )
            legacy_final_score = (
                self.config.fusion_weight * normalized_rrf
                + self.config.deterministic_weight * deterministic_score
            )
            scored.append(
                {
                    "candidate": candidate,
                    "normalized_rrf_score": normalized_rrf,
                    "source_rank_score": source_rank_score,
                    "retrieval_score": retrieval_score,
                    "deterministic_score": deterministic_score,
                    "final_score": final_score,
                    "legacy_final_score": legacy_final_score,
                    "components": components,
                }
            )

        retrieval_order = sorted(
            scored,
            key=lambda row: (
                -float(row["retrieval_score"]),
                -float(row["normalized_rrf_score"]),
                row["candidate"].fusion_rank,
                row["candidate"].chunk_id,
            ),
        )
        for rank, row in enumerate(retrieval_order, start=1):
            row["preservation_rank"] = rank

        legacy_order = sorted(
            scored,
            key=lambda row: (
                -float(row["legacy_final_score"]),
                row["candidate"].fusion_rank,
                row["candidate"].chunk_id,
            ),
        )
        for rank, row in enumerate(legacy_order, start=1):
            row["legacy_final_rank"] = rank

        unbounded_order = sorted(
            scored,
            key=lambda row: (
                -float(row["final_score"]),
                int(row["preservation_rank"]),
                row["candidate"].chunk_id,
            ),
        )
        for rank, row in enumerate(unbounded_order, start=1):
            row["unbounded_final_rank"] = rank

        final_order = _windowed_order(
            retrieval_order,
            score_key="final_score",
            window_size=self.config.rerank_window_size,
        )
        for rank, row in enumerate(final_order, start=1):
            row["final_rank"] = rank
        _attach_weight_grid(scored, retrieval_order, self.config.rerank_window_size)

        output: list[RetrievalResult] = []
        for row in final_order[:top_k]:
            fused_candidate = row["candidate"]
            final_rank = int(row["final_rank"])
            source = chunks_by_id[fused_candidate.chunk_id]
            match = source.metadata_match.to_dict()
            match["hybrid"] = {
                **fused_candidate.to_dict(),
                "normalized_rrf_score": row["normalized_rrf_score"],
                "source_rank_score": row["source_rank_score"],
                "retrieval_score": row["retrieval_score"],
                "preservation_rank": row["preservation_rank"],
                "deterministic_rerank_score": row["deterministic_score"],
                "legacy_final_score": row["legacy_final_score"],
                "legacy_final_rank": row["legacy_final_rank"],
                "unbounded_final_rank": row["unbounded_final_rank"],
                "final_score": row["final_score"],
                "final_rank": final_rank,
                "fusion_weight": self.config.fusion_weight,
                "deterministic_weight": self.config.deterministic_weight,
                "rerank_window_size": self.config.rerank_window_size,
            }
            match["score_components"] = row["components"]
            output.append(
                RetrievalResult(
                    chunk_id=fused_candidate.chunk_id,
                    doc_id=fused_candidate.doc_id,
                    bm25_score=float(fused_candidate.lexical_score or 0.0),
                    rank=final_rank,
                    metadata_match=match,
                )
            )
        diagnostics = [_rerank_diagnostic(row) for row in final_order]
        return output, diagnostics


def _best_source_rank_score(candidate: FusedCandidate, rrf_k: int) -> float:
    ranks = [
        rank
        for rank in (candidate.lexical_rank, candidate.vector_rank)
        if rank is not None
    ]
    if not ranks:
        return 0.0
    return (rrf_k + 1.0) / (rrf_k + min(ranks))


def _windowed_order(
    retrieval_order: Sequence[dict[str, Any]],
    *,
    score_key: str,
    window_size: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for start in range(0, len(retrieval_order), window_size):
        window = retrieval_order[start : start + window_size]
        output.extend(
            sorted(
                window,
                key=lambda row: (
                    -float(row[score_key]),
                    int(row["preservation_rank"]),
                    row["candidate"].chunk_id,
                ),
            )
        )
    return output


def _attach_weight_grid(
    rows: Sequence[dict[str, Any]],
    retrieval_order: Sequence[dict[str, Any]],
    window_size: int,
) -> None:
    for row in rows:
        row["weight_grid"] = []
    for fusion_weight, deterministic_weight in DIAGNOSTIC_WEIGHT_GRID:
        for row in rows:
            row["_grid_score"] = (
                fusion_weight * float(row["retrieval_score"])
                + deterministic_weight * float(row["deterministic_score"])
            )
        unbounded = sorted(
            rows,
            key=lambda row: (
                -float(row["_grid_score"]),
                int(row["preservation_rank"]),
                row["candidate"].chunk_id,
            ),
        )
        unbounded_ranks = {
            row["candidate"].chunk_id: rank
            for rank, row in enumerate(unbounded, start=1)
        }
        bounded = _windowed_order(
            retrieval_order,
            score_key="_grid_score",
            window_size=window_size,
        )
        bounded_ranks = {
            row["candidate"].chunk_id: rank
            for rank, row in enumerate(bounded, start=1)
        }
        for row in rows:
            chunk_id = row["candidate"].chunk_id
            row["weight_grid"].append(
                {
                    "fusion_weight": fusion_weight,
                    "deterministic_weight": deterministic_weight,
                    "final_score": row["_grid_score"],
                    "unbounded_rank": unbounded_ranks[chunk_id],
                    "bounded_rank": bounded_ranks[chunk_id],
                }
            )
    for row in rows:
        row.pop("_grid_score", None)


def _rerank_diagnostic(row: Mapping[str, Any]) -> dict[str, Any]:
    candidate = row["candidate"]
    return {
        **candidate.to_dict(),
        "normalized_rrf_score": row["normalized_rrf_score"],
        "source_rank_score": row["source_rank_score"],
        "retrieval_score": row["retrieval_score"],
        "preservation_rank": row["preservation_rank"],
        "deterministic_rerank_score": row["deterministic_score"],
        "legacy_final_score": row["legacy_final_score"],
        "legacy_final_rank": row["legacy_final_rank"],
        "unbounded_final_rank": row["unbounded_final_rank"],
        "final_score": row["final_score"],
        "final_rank": row["final_rank"],
        "score_components": dict(row["components"]),
        "weight_grid": list(row["weight_grid"]),
    }


def _annotate_lexical_fallback(
    results: Sequence[RetrievalResult],
    fused: Sequence[FusedCandidate],
    vector_status: str,
) -> list[RetrievalResult]:
    fused_by_id = {candidate.chunk_id: candidate for candidate in fused}
    output: list[RetrievalResult] = []
    for result in results:
        match = dict(result.metadata_match)
        score_components = dict(match.get("score_components") or {})
        fused_candidate = fused_by_id.get(result.chunk_id)
        match["hybrid"] = {
            **(fused_candidate.to_dict() if fused_candidate else {}),
            "vector_status": vector_status,
            "deterministic_rerank_score": score_components.get("final_score"),
            "final_score": score_components.get("final_score"),
            "final_rank": result.rank,
            "fallback": "lexical_only",
        }
        output.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                bm25_score=result.bm25_score,
                rank=result.rank,
                metadata_match=match,
            )
        )
    return output


def _require_method(backend: object, method: str) -> Any:
    if not callable(getattr(backend, method, None)):
        raise TypeError(f"{type(backend).__name__} must implement {method}()")
    return backend
