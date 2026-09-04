"""Metadata-scoped lexical/vector retrieval with RRF and deterministic reranking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.reasoning.router import QueryRouter
from app.retrieval.correction_expansion import (
    CorrectionExpansion,
    apply_expansion,
)
from app.retrieval.event_expansion import EventExpansion
from app.retrieval.filter_relaxation import relax_when_strict_zero
from app.reasoning.periodic_metric_view import has_exact_metric_row
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
_BALANCE_SHEET_METRICS = frozenset({"자산총계", "부채총계", "자본총계"})
_INCOME_STATEMENT_METRICS = frozenset({"매출액", "영업이익", "당기순이익"})
_STATEMENT_METRICS = _BALANCE_SHEET_METRICS | _INCOME_STATEMENT_METRICS


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
    rerank_mode: str = "legacy"
    diagnostic_top_n: int | None = None

    def __post_init__(self) -> None:
        if min(self.lexical_top_n, self.vector_top_n, self.final_top_k) <= 0:
            raise ValueError("hybrid retrieval limits must be positive")
        if self.rerank_window_size <= 0:
            raise ValueError("hybrid rerank window size must be positive")
        if self.rerank_mode not in {"legacy", "bounded"}:
            raise ValueError("hybrid rerank mode must be 'legacy' or 'bounded'")
        if self.diagnostic_top_n is not None and self.diagnostic_top_n <= 0:
            raise ValueError("hybrid diagnostic top-n must be positive when set")
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
            "rerank_mode": self.rerank_mode,
            "diagnostic_top_n": self.diagnostic_top_n,
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
    diagnostic_lexical_results: Sequence[RetrievalResult] = ()
    diagnostic_vector_results: Sequence[VectorRetrievalResult] = ()
    correction_expansion: Mapping[str, Any] = field(default_factory=dict)
    event_expansion: Mapping[str, Any] = field(default_factory=dict)


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
        correction_expander: Any | None = None,
        event_expander: Any | None = None,
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
        self._correction_expander = correction_expander
        self._event_expander = event_expander

    def execute(self, plan: Any) -> HybridQueryExecution:
        route = self.router.route(plan)
        documents = self._metadata_backend.get_candidate_documents(**route.backend_filters)
        # Routing can empty the candidate set entirely when it inferred a
        # narrower event than the question asked for.  P1-B deferred relaxation
        # until a real strict zero was demonstrated; this recovers that one case
        # and leaves every non-empty candidate set exactly as it was.
        strict_documents = self.router.filter_documents(documents, route)
        documents, relaxation = relax_when_strict_zero(
            self.router, documents, strict_documents, route
        )
        chunks = self._chunk_backend.get_candidate_chunks(documents)
        chunks = self.router.prepare_chunks(chunks, route)
        document_metadata = {document.doc_id: document.metadata for document in documents}
        embedded_candidate_ids, vector_coverage = self._vector_coverage(chunks)

        lexical_request_limit = max(
            self.config.lexical_top_n,
            self.config.diagnostic_top_n or self.config.lexical_top_n,
        )
        diagnostic_lexical_results = self._lexical_retriever.retrieve(
            plan.lexical_query,
            chunks,
            top_k=lexical_request_limit,
        )
        lexical_results = list(diagnostic_lexical_results[: self.config.lexical_top_n])
        lexical_final = self.router.rerank(
            lexical_results,
            route,
            chunks=chunks,
            document_metadata=document_metadata,
            top_k=min(plan.top_k, self.config.final_top_k),
        )

        vector_results: list[VectorRetrievalResult] = []
        diagnostic_vector_results: Sequence[VectorRetrievalResult] = ()
        vector_status = "ok"
        vector_error: str | None = None
        if vector_coverage.get("available") and not vector_coverage.get(
            "embedded_count"
        ):
            vector_status = "no_coverage"
        else:
            try:
                query_embedding = self._embedder.embed_query(plan.lexical_query)
                vector_request_limit = max(
                    self.config.vector_top_n,
                    self.config.diagnostic_top_n or self.config.vector_top_n,
                )
                diagnostic_vector_results = self._vector_retriever.vector_search(
                    query_embedding,
                    chunks,
                    embedding_model=self.embedding_config.model,
                    embedding_version=self.embedding_config.version,
                    top_k=vector_request_limit,
                )
                vector_results = list(
                    diagnostic_vector_results[: self.config.vector_top_n]
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
        scored_tail: ScoredTail | None = None
        if not vector_results:
            final_results = _annotate_lexical_fallback(lexical_final, fused, vector_status)
        else:
            final_results, rerank_diagnostics, scored_tail = self._hybrid_rerank(
                fused,
                route,
                chunks,
                document_metadata,
                top_k=min(plan.top_k, self.config.final_top_k),
            )
        if not final_results and chunks:
            final_results = self._filtered_candidate_fallback(
                chunks,
                route,
                document_metadata,
                top_k=min(plan.top_k, self.config.final_top_k),
            )
            vector_status = (
                vector_status
                if vector_status not in {"ok", "empty"}
                else "filtered_candidates"
            )
        final_results = self._rescue_latest_event_candidates(
            final_results,
            chunks,
            route,
            document_metadata,
            top_k=min(plan.top_k, self.config.final_top_k),
        )
        final_results = self._rescue_statement_metric_candidates(
            final_results,
            chunks,
            route,
            document_metadata,
            top_k=min(plan.top_k, self.config.final_top_k),
        )
        # The correction graph knows documents this question's own metadata
        # window excluded. Add them after ranking so the retrieved order is
        # untouched and the added evidence stays identifiable.
        expansion = (
            self._correction_expander.expand(
                plan, documents=documents, chunks=chunks, results=final_results
            )
            if self._correction_expander is not None
            else CorrectionExpansion()
        )
        chunks, final_results = apply_expansion(expansion, chunks, final_results)
        # The event graph knows the rest of a contract's lifecycle: the
        # termination of a contract that was found, or the contract behind a
        # termination that was found. Added after ranking for the same reason.
        event_expansion = (
            self._event_expander.expand(
                plan, documents=documents, chunks=chunks, results=final_results
            )
            if self._event_expander is not None
            else EventExpansion()
        )
        chunks, final_results = apply_expansion(
            event_expansion, chunks, final_results
        )

        # A contract-amount delta names two filing-date scopes.  Ordinary
        # retrieval may rank only one side, and ordinary event expansion
        # deliberately requires a resolved lifecycle.  A delta does not need
        # a terminated lifecycle: it only needs the two explicitly dated
        # operands.  Recover those operands narrowly and additively.
        chunks, final_results, amount_change_recovery = (
            self._recover_amount_change_operands(
                plan,
                documents=documents,
                chunks=chunks,
                results=final_results,
                event_trace=event_expansion.to_dict(),
            )
        )

        # Last, so the document set it inspects is the one actually emitted:
        # the rescues and expansions above may already have supplied the very
        # document a crowded list was missing.
        final_results, document_recovery = _additive_document_rescue(
            final_results, scored_tail
        )

        correction = self.router.correction_summary(documents, route)
        routing = {
            **route.to_dict(),
            **({"correction": correction} if correction else {}),
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
                "additive_document_recovery": document_recovery,
                "amount_change_operand_recovery": amount_change_recovery,
            },
            **(
                {"filter_relaxation": relaxation.to_dict()}
                if relaxation.applied
                else {}
            ),
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
            correction_expansion=expansion.to_dict(),
            event_expansion=event_expansion.to_dict(),
            rerank_diagnostics=rerank_diagnostics,
            diagnostic_lexical_results=diagnostic_lexical_results,
            diagnostic_vector_results=diagnostic_vector_results,
        )

    def _recover_amount_change_operands(
        self,
        plan: Any,
        *,
        documents: Sequence[CandidateDocument],
        chunks: Sequence[CandidateChunk],
        results: Sequence[RetrievalResult],
        event_trace: Mapping[str, Any],
    ) -> tuple[
        list[CandidateChunk],
        list[RetrievalResult],
        dict[str, Any],
    ]:
        """Add only the two explicitly dated operands of an amount-change query.

        This is deliberately narrower than corporate-event expansion.

        A delta question already names two receipt-date scopes.  First reuse
        matching chunks from the routed candidate universe.  Only when a scope
        is absent there may the carried event graph supply a canonical
        correction document, and that document is accepted only when its
        receipt date and corp_code satisfy the missing scope.

        No fuzzy lookalike search and no lifecycle-status relaxation occurs.
        """

        evidence = getattr(plan, "evidence", None)
        if not isinstance(evidence, Mapping):
            return list(chunks), list(results), {
                "requested": False,
                "applied": False,
                "reason": "not_requested",
            }

        request = evidence.get("contract_amount_change")
        if not isinstance(request, Mapping):
            return list(chunks), list(results), {
                "requested": False,
                "applied": False,
                "reason": "not_requested",
            }

        initial_on = str(request.get("initial_on") or "").strip()
        final_on = str(request.get("final_on") or "").strip()
        final_field = str(
            request.get("final_field") or "contract_amount"
        ).strip()

        if not initial_on or not final_on:
            return list(chunks), list(results), {
                "requested": True,
                "applied": False,
                "reason": "invalid_scope",
            }

        corp_code = str(getattr(plan, "corp_code", None) or "").strip()

        scopes = (
            ("initial", initial_on, "contract_amount"),
            ("final", final_on, final_field),
        )

        merged_chunks = list(chunks)
        merged_results = list(results)

        known_chunk_ids = {
            str(candidate.chunk_id)
            for candidate in merged_chunks
        }
        ranked_chunk_ids = {
            str(result.chunk_id)
            for result in merged_results
        }

        metadata_by_doc = {
            str(document.doc_id): dict(document.metadata or {})
            for document in documents
        }

        fetched_doc_ids: list[str] = []
        added_doc_ids: list[str] = []
        added_chunk_ids: list[str] = []
        role_trace: dict[str, dict[str, Any]] = {}

        def scope_digits(value: Any) -> str:
            return "".join(
                ch for ch in str(value or "") if ch.isdigit()
            )

        def candidate_payload(
            candidate: CandidateChunk,
        ) -> Mapping[str, Any]:
            raw = getattr(candidate, "chunk", None)
            return raw if isinstance(raw, Mapping) else {}

        def candidate_date(candidate: CandidateChunk) -> str:
            raw = candidate_payload(candidate)
            value = (
                raw.get("rcept_dt")
                or metadata_by_doc.get(str(candidate.doc_id), {}).get(
                    "rcept_dt"
                )
            )
            return _compact_date(value)

        def candidate_corp(candidate: CandidateChunk) -> str:
            raw = candidate_payload(candidate)
            value = (
                raw.get("corp_code")
                or metadata_by_doc.get(str(candidate.doc_id), {}).get(
                    "corp_code"
                )
            )
            return str(value or "").strip()

        def amount_score(
            candidate: CandidateChunk,
            *,
            field: str,
        ) -> int:
            raw = candidate_payload(candidate)
            text = str(
                raw.get("retrieval_text")
                or raw.get("content")
                or ""
            )

            label = (
                "해지금액"
                if field == "termination_amount"
                else "계약금액"
            )

            if label not in text:
                return 0

            score = 100

            # Prefer the filing's formal amount row over a correction
            # before/after comparison table.  The formal row is the value
            # in force for that filing date and is exactly what the scoped
            # operand represents.
            if (
                field == "termination_amount"
                and "해지내역" in text
                and "해지금액" in text
            ):
                score += 50

            if (
                field != "termination_amount"
                and "계약내역" in text
                and "계약금액" in text
            ):
                score += 50

            if "정정전" in text and "정정후" in text:
                score -= 25

            return score

        def select_candidate(
            on_date: str,
            field: str,
        ) -> tuple[CandidateChunk | None, str]:
            wanted = scope_digits(on_date)

            eligible = [
                candidate
                for candidate in merged_chunks
                if candidate_date(candidate).startswith(wanted)
                and (
                    not corp_code
                    or not candidate_corp(candidate)
                    or candidate_corp(candidate) == corp_code
                )
                and amount_score(candidate, field=field) > 0
            ]

            doc_ids = sorted(
                {
                    str(candidate.doc_id)
                    for candidate in eligible
                }
            )

            if not eligible:
                return None, "scope_not_present"

            # The explicit company + receipt-date scope must identify one
            # filing.  Do not choose between multiple same-date filings.
            if len(doc_ids) != 1:
                return None, "scope_ambiguous"

            selected = sorted(
                eligible,
                key=lambda candidate: (
                    -amount_score(candidate, field=field),
                    str(candidate.chunk_id),
                ),
            )[0]

            return selected, "candidate_scope"

        # -------------------------------------------------------------
        # First pass: use the already-routed candidate universe.
        # C068 is recovered entirely here.
        # -------------------------------------------------------------

        selected: dict[str, CandidateChunk] = {}
        missing_roles: list[str] = []

        for role, on_date, field in scopes:
            candidate, source = select_candidate(on_date, field)

            if candidate is None:
                missing_roles.append(role)
                role_trace[role] = {
                    "scope": on_date,
                    "field": field,
                    "status": source,
                }
                continue

            selected[role] = candidate
            role_trace[role] = {
                "scope": on_date,
                "field": field,
                "status": source,
                "doc_id": str(candidate.doc_id),
                "chunk_id": str(candidate.chunk_id),
            }

        # -------------------------------------------------------------
        # Second pass: only for a still-missing explicit date scope,
        # consult document identities already carried by the event graph.
        #
        # C067/C071 reach this path: the graph already records
        # origin -> canonical correction, but ordinary lifecycle expansion
        # correctly declined because the contract remains open.
        # -------------------------------------------------------------

        if missing_roles:
            graph_doc_ids: list[str] = []

            block = (
                event_trace.get("corporate_event_expansion")
                if isinstance(event_trace, Mapping)
                else None
            )

            if isinstance(block, Mapping):
                mapping = block.get("seed_member_doc_ids")
                if isinstance(mapping, Mapping):
                    for value in mapping.values():
                        doc_id = str(value or "").strip()
                        if doc_id and doc_id not in graph_doc_ids:
                            graph_doc_ids.append(doc_id)

            states = (
                event_trace.get("event_member_states")
                if isinstance(event_trace, Mapping)
                else None
            )

            if isinstance(states, Mapping):
                for state in states.values():
                    if not isinstance(state, Mapping):
                        continue
                    doc_id = str(
                        state.get("canonical_doc_id")
                        or state.get("doc_id")
                        or ""
                    ).strip()
                    if doc_id and doc_id not in graph_doc_ids:
                        graph_doc_ids.append(doc_id)

            present_docs = {
                str(candidate.doc_id)
                for candidate in merged_chunks
            }

            wanted_graph_ids = [
                doc_id
                for doc_id in graph_doc_ids
                if doc_id not in present_docs
            ]

            reader = getattr(
                self._metadata_backend,
                "fetch_documents",
                None,
            )

            fetched_documents: list[CandidateDocument] = []

            if wanted_graph_ids and callable(reader):
                fetched_documents = list(reader(wanted_graph_ids))

            # Accept a graph document only if it matches one of the still
            # missing explicit receipt-date scopes and the same company.
            missing_scope_digits = {
                role: scope_digits(on_date)
                for role, on_date, _field in scopes
                if role in missing_roles
            }

            bounded_documents: list[CandidateDocument] = []

            for document in fetched_documents:
                metadata = dict(document.metadata or {})
                receipt = _compact_date(metadata.get("rcept_dt"))
                doc_corp = str(metadata.get("corp_code") or "").strip()

                matches_missing_scope = any(
                    receipt.startswith(value)
                    for value in missing_scope_digits.values()
                )

                same_company = (
                    not corp_code
                    or not doc_corp
                    or doc_corp == corp_code
                )

                if not matches_missing_scope or not same_company:
                    continue

                bounded_documents.append(document)
                metadata_by_doc[str(document.doc_id)] = metadata
                fetched_doc_ids.append(str(document.doc_id))

            if bounded_documents:
                fetched_chunks = list(
                    self._chunk_backend.get_candidate_chunks(
                        bounded_documents
                    )
                )

                for candidate in fetched_chunks:
                    if str(candidate.chunk_id) in known_chunk_ids:
                        continue
                    known_chunk_ids.add(str(candidate.chunk_id))
                    merged_chunks.append(candidate)

            # Retry only the roles that were absent on the first pass.
            for role, on_date, field in scopes:
                if role not in missing_roles:
                    continue

                candidate, source = select_candidate(on_date, field)

                if candidate is None:
                    role_trace[role] = {
                        "scope": on_date,
                        "field": field,
                        "status": source,
                    }
                    continue

                selected[role] = candidate
                role_trace[role] = {
                    "scope": on_date,
                    "field": field,
                    "status": "graph_scope",
                    "doc_id": str(candidate.doc_id),
                    "chunk_id": str(candidate.chunk_id),
                }

        # -------------------------------------------------------------
        # Add the exact amount-bearing chunks after ranking.
        # No existing result is reordered or removed.
        # -------------------------------------------------------------

        for role, on_date, field in scopes:
            candidate = selected.get(role)
            if candidate is None:
                continue

            chunk_id = str(candidate.chunk_id)

            if chunk_id in ranked_chunk_ids:
                role_trace[role]["served"] = "already_ranked"
                continue

            match = dict(candidate.metadata_match.to_dict())
            match["amount_change_recovery"] = {
                "role": role,
                "on_date": on_date,
                "field": field,
                "retrieval_source": "explicit_operand_scope",
            }

            merged_results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    doc_id=candidate.doc_id,
                    bm25_score=0.0,
                    rank=len(merged_results) + 1,
                    metadata_match=match,
                )
            )

            ranked_chunk_ids.add(chunk_id)
            added_chunk_ids.append(chunk_id)

            doc_id = str(candidate.doc_id)
            if doc_id not in added_doc_ids:
                added_doc_ids.append(doc_id)

            role_trace[role]["served"] = "additive_recovery"

        both_resolved = all(
            role in selected
            for role, _on_date, _field in scopes
        )

        return merged_chunks, merged_results, {
            "requested": True,
            "applied": bool(added_chunk_ids),
            "reason": (
                "resolved"
                if both_resolved
                else "operand_scope_unresolved"
            ),
            "initial_on": initial_on,
            "final_on": final_on,
            "corp_code": corp_code or None,
            "roles": role_trace,
            "graph_fetched_doc_ids": list(
                dict.fromkeys(fetched_doc_ids)
            ),
            "added_doc_ids": list(added_doc_ids),
            "added_chunk_ids": list(added_chunk_ids),
        }

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

    def _filtered_candidate_fallback(
        self,
        chunks: Sequence[CandidateChunk],
        route: Any,
        document_metadata: Mapping[str, Mapping[str, Any]],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Keep metadata-filtered chunks when FTS and vector both miss.

        PostgreSQL ``websearch_to_tsquery`` ANDs every token. A compound
        question can therefore score zero against the already-correct event
        document. Returning nothing in that case throws away evidence the
        router already isolated.
        """

        synthetic = [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                bm25_score=0.0,
                rank=index,
                metadata_match=chunk.metadata_match.to_dict(),
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        reranked = self.router.rerank(
            synthetic,
            route,
            chunks=chunks,
            document_metadata=document_metadata,
            top_k=top_k,
        )
        output: list[RetrievalResult] = []
        for result in reranked:
            match = dict(result.metadata_match)
            match["hybrid"] = {
                **dict(match.get("hybrid") or {}),
                "fallback": "filtered_candidates",
                "final_rank": result.rank,
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

    def _rescue_latest_event_candidates(
        self,
        results: Sequence[RetrievalResult],
        chunks: Sequence[CandidateChunk],
        route: Any,
        document_metadata: Mapping[str, Mapping[str, Any]],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Prefer the newest routed event document for latest-event questions."""

        if route.ranking_context.get("task_type") != "corporate_event":
            return list(results)
        if route.ranking_context.get("period_type") != "latest_event":
            return list(results)
        if not chunks or top_k <= 0:
            return list(results)

        dates_by_doc = {
            chunk.doc_id: _compact_date(
                chunk.chunk.get("rcept_dt")
                or document_metadata.get(chunk.doc_id, {}).get("rcept_dt")
            )
            for chunk in chunks
        }
        latest = max((value for value in dates_by_doc.values() if value), default="")
        if not latest:
            return list(results)
        latest_chunks = [
            chunk for chunk in chunks if dates_by_doc.get(chunk.doc_id) == latest
        ]
        if not latest_chunks:
            return list(results)

        synthetic = [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                bm25_score=0.0,
                rank=index,
                metadata_match=chunk.metadata_match.to_dict(),
            )
            for index, chunk in enumerate(latest_chunks, start=1)
        ]
        ranked = self.router.rerank(
            synthetic,
            route,
            chunks=latest_chunks,
            document_metadata=document_metadata,
            top_k=top_k,
        )
        priority: list[RetrievalResult] = []
        for result in ranked:
            match = dict(result.metadata_match)
            match["hybrid"] = {
                **dict(match.get("hybrid") or {}),
                "fallback": "latest_event_document_rescue",
                "final_rank": result.rank,
            }
            priority.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    bm25_score=result.bm25_score,
                    rank=result.rank,
                    metadata_match=match,
                )
            )
        return _merge_priority_results(priority, results, top_k=top_k)

    def _rescue_statement_metric_candidates(
        self,
        results: Sequence[RetrievalResult],
        chunks: Sequence[CandidateChunk],
        route: Any,
        document_metadata: Mapping[str, Mapping[str, Any]],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Promote exact financial statement rows that retrieval missed.

        Balance-sheet labels are often spaced out in DART tables, for example
        ``자 산 총 계``.  PostgreSQL FTS can rank broader note tables above the
        actual statement row.  The same happens for standalone income-statement
        rows when business-segment notes contain repeated sales terms.  Use the
        already-filtered candidate universe as a narrow rescue source, then keep
        only exact metric rows in the expected statement section.
        """

        if route.ranking_context.get("task_type") != "financial_metric":
            return list(results)
        metric = str(route.ranking_context.get("metric") or "")
        if metric not in _STATEMENT_METRICS:
            return list(results)
        if not chunks or top_k <= 0:
            return list(results)

        synthetic = [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                bm25_score=0.0,
                rank=index,
                metadata_match=chunk.metadata_match.to_dict(),
            )
            for index, chunk in enumerate(chunks, start=1)
        ]
        ranked = self.router.rerank(
            synthetic,
            route,
            chunks=chunks,
            document_metadata=document_metadata,
            top_k=top_k,
        )
        rescue: list[RetrievalResult] = []
        for result in ranked:
            components = dict(result.metadata_match.get("score_components") or {})
            chunk = next(
                (candidate.chunk for candidate in chunks if candidate.chunk_id == result.chunk_id),
                {},
            )
            has_metric_row = has_exact_metric_row(
                str(chunk.get("content") or chunk.get("retrieval_text") or ""),
                metric,
            )
            if (
                not has_metric_row
                and float(components.get("exact_term", 0.0)) < 0.65
                or float(components.get("section", 0.0)) < 0.85
                or not _statement_section_matches_metric(chunk, metric)
            ):
                continue
            match = dict(result.metadata_match)
            match["hybrid"] = {
                **dict(match.get("hybrid") or {}),
                "fallback": "statement_metric_rescue",
                "final_rank": result.rank,
            }
            rescue.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    bm25_score=result.bm25_score,
                    rank=result.rank,
                    metadata_match=match,
                )
            )

        if not rescue:
            return list(results)
        return _merge_priority_results(rescue, results, top_k=top_k)

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
            bounded_final_score = (
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
                    "bounded_final_score": bounded_final_score,
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
                -float(row["bounded_final_score"]),
                int(row["preservation_rank"]),
                row["candidate"].chunk_id,
            ),
        )
        for rank, row in enumerate(unbounded_order, start=1):
            row["unbounded_final_rank"] = rank

        bounded_order = _windowed_order(
            retrieval_order,
            score_key="bounded_final_score",
            window_size=self.config.rerank_window_size,
        )
        for rank, row in enumerate(bounded_order, start=1):
            row["bounded_final_rank"] = rank

        if self.config.rerank_mode == "legacy":
            final_order = legacy_order
            selected_score_key = "legacy_final_score"
        else:
            final_order = bounded_order
            selected_score_key = "bounded_final_score"
        for rank, row in enumerate(final_order, start=1):
            row["final_rank"] = rank
            row["final_score"] = row[selected_score_key]
            row["rerank_mode"] = self.config.rerank_mode
        _attach_weight_grid(scored, retrieval_order, self.config.rerank_window_size)

        output = [
            self._result_from_scored_row(row, chunks_by_id)
            for row in final_order[:top_k]
        ]
        diagnostics = [_rerank_diagnostic(row) for row in final_order]
        # Rows past the budget keep their scores and ranks. They are not emitted,
        # and nothing downstream may treat them as retrieved; they exist so a
        # later stage can consult the order that was already computed instead of
        # querying again.
        tail = ScoredTail(rows=tuple(final_order[top_k:]), chunks_by_id=chunks_by_id,
                          builder=self._result_from_scored_row)
        return output, diagnostics, tail

    def _result_from_scored_row(
        self, row: Mapping[str, Any], chunks_by_id: Mapping[str, CandidateChunk]
    ) -> RetrievalResult:
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
            "bounded_final_score": row["bounded_final_score"],
            "bounded_final_rank": row["bounded_final_rank"],
            "unbounded_final_rank": row["unbounded_final_rank"],
            "final_score": row["final_score"],
            "final_rank": final_rank,
            "rerank_mode": self.config.rerank_mode,
            "fusion_weight": self.config.fusion_weight,
            "deterministic_weight": self.config.deterministic_weight,
            "rerank_window_size": self.config.rerank_window_size,
        }
        match["score_components"] = row["components"]
        return RetrievalResult(
            chunk_id=fused_candidate.chunk_id,
            doc_id=fused_candidate.doc_id,
            bm25_score=float(fused_candidate.lexical_score or 0.0),
            rank=final_rank,
            metadata_match=match,
        )


#: A document occupying this many of the emitted slots is crowding the list: the
#: budget is being spent re-showing one filing while others retrieved just below
#: it are never seen at all.
DOCUMENT_CROWDING_THRESHOLD = 3

#: How many unseen documents a crowded result may recover. One, deliberately:
#: the recovery is a correction for crowding, not a wider retrieval budget.
ADDITIVE_DOCUMENT_RESCUE_LIMIT = 1


@dataclass(frozen=True)
class ScoredTail:
    """Candidates that were scored and ranked but fell outside the budget.

    Keeping them costs nothing -- they were already ordered -- and lets a later
    stage consult that order instead of running retrieval a second time or
    widening the budget that produced it.
    """

    rows: tuple[Mapping[str, Any], ...] = ()
    chunks_by_id: Mapping[str, CandidateChunk] = field(default_factory=dict)
    builder: Any = None

    def result_for(self, row: Mapping[str, Any]) -> RetrievalResult:
        return self.builder(row, self.chunks_by_id)


def _additive_document_rescue(
    results: Sequence[RetrievalResult],
    tail: ScoredTail | None,
    *,
    crowding_threshold: int = DOCUMENT_CROWDING_THRESHOLD,
    limit: int = ADDITIVE_DOCUMENT_RESCUE_LIMIT,
) -> tuple[list[RetrievalResult], dict[str, Any]]:
    """Append an unseen document when one filing has crowded out the rest.

    Ranking scores chunks, so a filing whose table splits into several similar
    chunks can take much of the list while a document ranked just below it never
    appears.  Replacing a crowding chunk was tried and rejected: it removed
    evidence other answers depended on.  So nothing is removed here.  The
    emitted results keep their exact identity and order, and at most one already
    scored candidate from a document the list never showed is appended after
    them.

    The rescue reads only retrieval structure -- how many chunks each document
    holds, and the order the reranker already produced.  It knows nothing about
    what the chunks contain.
    """

    emitted = list(results)
    trace: dict[str, Any] = {
        "attempted": False,
        "crowding_detected": False,
        "max_chunks_from_document": 0,
        "appended": False,
        "appended_chunk_id": None,
        "appended_doc_id": None,
        "original_candidate_rank": None,
    }
    if not emitted or tail is None or not tail.rows or limit < 1:
        return emitted, trace

    trace["attempted"] = True
    counts: dict[str, int] = {}
    for result in emitted:
        # A result without a document identity is not evidence that some other
        # filing is crowding the list, so it never contributes to the count.
        if result.doc_id:
            counts[result.doc_id] = counts.get(result.doc_id, 0) + 1
    crowding = max(counts.values(), default=0)
    trace["max_chunks_from_document"] = crowding
    if crowding < crowding_threshold:
        return emitted, trace

    trace["crowding_detected"] = True
    seen_docs = set(counts)
    seen_chunks = {result.chunk_id for result in emitted}
    added = 0
    for row in tail.rows:
        candidate = row["candidate"]
        doc_id = getattr(candidate, "doc_id", None)
        if not doc_id or doc_id in seen_docs:
            continue
        if candidate.chunk_id in seen_chunks:
            continue
        emitted.append(tail.result_for(row))
        trace.update({
            "appended": True,
            "appended_chunk_id": candidate.chunk_id,
            "appended_doc_id": doc_id,
            "original_candidate_rank": int(row["final_rank"]),
        })
        added += 1
        if added >= limit:
            break
    return emitted, trace


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
        "bounded_final_score": row["bounded_final_score"],
        "bounded_final_rank": row["bounded_final_rank"],
        "unbounded_final_rank": row["unbounded_final_rank"],
        "final_score": row["final_score"],
        "final_rank": row["final_rank"],
        "rerank_mode": row["rerank_mode"],
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


def _merge_priority_results(
    priority: Sequence[RetrievalResult],
    existing: Sequence[RetrievalResult],
    *,
    top_k: int,
) -> list[RetrievalResult]:
    rows: list[RetrievalResult] = []
    seen: set[str] = set()
    for result in (*priority, *existing):
        if result.chunk_id in seen:
            continue
        seen.add(result.chunk_id)
        rows.append(result)
        if len(rows) >= top_k:
            break

    output: list[RetrievalResult] = []
    for rank, result in enumerate(rows, start=1):
        match = dict(result.metadata_match)
        hybrid = dict(match.get("hybrid") or {})
        if hybrid:
            hybrid["final_rank"] = rank
            match["hybrid"] = hybrid
        output.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                bm25_score=result.bm25_score,
                rank=rank,
                metadata_match=match,
            )
        )
    return output


def _compact_date(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _statement_section_matches_metric(chunk: Mapping[str, Any], metric: str) -> bool:
    section_text = " ".join(str(value) for value in chunk.get("section_path") or ())
    normalized = re.sub(r"[^0-9a-z가-힣]+", "", section_text.casefold())
    if metric in _INCOME_STATEMENT_METRICS:
        return "손익계산서" in normalized or "포괄손익계산서" in normalized
    if metric in _BALANCE_SHEET_METRICS:
        return (
            "재무상태표" in normalized
            or "첨부연결재무제표" in normalized
            or "첨부재무제표" in normalized
        )
    return False


def _require_method(backend: object, method: str) -> Any:
    if not callable(getattr(backend, method, None)):
        raise TypeError(f"{type(backend).__name__} must implement {method}()")
    return backend
