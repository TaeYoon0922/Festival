"""Deterministic correction-chain expansion of an existing retrieval result.

Metadata retrieval answers "which disclosures match this question".  For a
question about a correction that is the wrong universe: a question anchored on
the original's receipt date can never reach the correction filed years later,
because the anchor date is a hard filter on the candidate set.  The correction
graph already knows those documents belong to one chain, so this layer adds
them back after retrieval instead of loosening the filter that found the
original.

Nothing here searches, plans, or asks a model anything.  It reads the chain the
graph already resolved, fetches those documents by identity, and picks chunks
inside them with the same lexical query the original retrieval used.  Expansion
happens only when the question asks for the final version or the history of a
report, and only for a group the graph marked ``resolved``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.reasoning.correction_graph import CorrectionGraphUnavailable
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


logger = logging.getLogger(__name__)

#: Question intents that need documents beyond the ones retrieval found.
INTENT_LATEST = "latest"
INTENT_HISTORY = "history"
INTENT_ORIGINAL = "original"
EXPANDING_INTENTS = frozenset({INTENT_LATEST, INTENT_HISTORY})

#: Marker written into an added result's ``metadata_match`` so relation-derived
#: evidence stays distinguishable from what retrieval itself ranked.
PROVENANCE_KEY = "correction_expansion"


@dataclass(frozen=True)
class CorrectionExpansion:
    """Documents the correction graph contributed to one execution."""

    added_chunks: tuple[CandidateChunk, ...] = ()
    added_results: tuple[RetrievalResult, ...] = ()
    added_documents: tuple[CandidateDocument, ...] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)

    @property
    def expanded(self) -> bool:
        return bool(self.added_results)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.trace)


def _seed_doc_ids(results: Sequence[RetrievalResult], limit: int) -> list[str]:
    """Doc ids retrieval actually ranked, best first, without duplicates."""

    ordered: list[str] = []
    for result in results:
        doc_id = str(result.doc_id)
        if doc_id and doc_id not in ordered:
            ordered.append(doc_id)
        if len(ordered) >= limit:
            break
    return ordered


class CorrectionExpander:
    """Add the rest of a resolved correction chain to a retrieval result.

    ``max_documents`` is a safety cap, not a chain limit: the longest legitimate
    chain in the corpus holds fifteen documents, so the default leaves room for
    it and still refuses a pathological group.
    """

    def __init__(
        self,
        repository: Any,
        metadata_backend: Any,
        chunk_backend: Any,
        retriever: Any,
        *,
        max_documents: int = 32,
        chunks_per_document: int = 2,
        history_chunks_per_document: int = 1,
        seed_limit: int = 10,
    ) -> None:
        if min(
            max_documents,
            chunks_per_document,
            history_chunks_per_document,
            seed_limit,
        ) <= 0:
            raise ValueError("correction expansion limits must be positive")
        self._repository = repository
        self._metadata_backend = metadata_backend
        self._chunk_backend = chunk_backend
        self._retriever = retriever
        self.max_documents = max_documents
        self.chunks_per_document = chunks_per_document
        # A history question wants breadth across the chain, not depth inside
        # each filing, so each added document contributes its single best chunk.
        self.history_chunks_per_document = history_chunks_per_document
        self.seed_limit = seed_limit

    # ------------------------------------------------------------------ api

    def expand(
        self,
        plan: Any,
        *,
        documents: Sequence[CandidateDocument],
        chunks: Sequence[CandidateChunk],
        results: Sequence[RetrievalResult],
    ) -> CorrectionExpansion:
        intent = _correction_intent(plan)
        if intent not in EXPANDING_INTENTS:
            # An ordinary question, or one that explicitly wants the original,
            # is answered from what retrieval already found.
            return CorrectionExpansion(trace=_trace(intent, "not_requested"))
        if not results:
            return CorrectionExpansion(trace=_trace(intent, "no_seed_results"))

        seeds = _seed_doc_ids(results, self.seed_limit)
        try:
            states = self._repository.document_states(seeds)
        except CorrectionGraphUnavailable as error:
            logger.warning("correction expansion skipped: %s", error)
            return CorrectionExpansion(trace=_trace(intent, "graph_unavailable"))
        resolved = {
            doc_id: state
            for doc_id, state in states.items()
            if getattr(state, "is_resolved", False)
        }
        if not resolved:
            # Includes every ambiguous or unresolved correction: those are stored
            # as a group of one and must never be expanded into a chain.
            return CorrectionExpansion(trace=_trace(intent, "no_resolved_group"))

        present = {str(result.doc_id) for result in results}
        metadata_by_doc = {
            document.doc_id: document.metadata for document in documents
        }
        wanted: list[str] = []
        groups: list[dict[str, Any]] = []
        for doc_id in seeds:
            state = resolved.get(doc_id)
            if state is None or any(
                group["correction_group_id"] == state.correction_group_id
                for group in groups
            ):
                continue
            selection = self._chain_selection(intent, doc_id, state)
            if selection is None:
                continue
            chain, targets = selection
            seed_corp = _corp_code(metadata_by_doc.get(doc_id))
            groups.append(
                {
                    "correction_group_id": state.correction_group_id,
                    "root_doc_id": state.root_doc_id,
                    "latest_doc_id": chain[-1],
                    "seed_doc_id": doc_id,
                    "seed_corp_code": seed_corp,
                    "chain_length": len(chain),
                }
            )
            for target in targets:
                if target not in present and target not in wanted:
                    wanted.append(target)

        if not wanted:
            return CorrectionExpansion(
                trace=_trace(intent, "already_retrieved", groups=groups)
            )
        if len(wanted) > self.max_documents:
            logger.warning(
                "correction expansion skipped: %d documents exceeds the cap of %d",
                len(wanted),
                self.max_documents,
            )
            return CorrectionExpansion(
                trace=_trace(intent, "too_many_documents", groups=groups)
            )

        added_documents = self._fetch_documents(wanted)
        added_documents = _same_company_only(added_documents, groups, metadata_by_doc)
        if not added_documents:
            return CorrectionExpansion(
                trace=_trace(intent, "documents_unavailable", groups=groups)
            )

        added_chunks, added_results = self._select_chunks(
            plan,
            added_documents,
            start_rank=len(results) + 1,
            per_document=(
                self.history_chunks_per_document
                if intent == INTENT_HISTORY
                else self.chunks_per_document
            ),
        )
        if not added_results:
            return CorrectionExpansion(
                trace=_trace(intent, "no_chunks", groups=groups)
            )
        return CorrectionExpansion(
            added_chunks=tuple(added_chunks),
            added_results=tuple(added_results),
            added_documents=tuple(added_documents),
            trace=_trace(
                intent,
                "expanded",
                groups=groups,
                added_doc_ids=[document.doc_id for document in added_documents],
                added_result_count=len(added_results),
            ),
        )

    # -------------------------------------------------------------- helpers

    def _chain_selection(
        self, intent: str, doc_id: str, state: Any
    ) -> tuple[list[str], list[str]] | None:
        """Which documents of this chain the intent needs."""

        try:
            chain = self._repository.get_correction_chain(doc_id)
        except CorrectionGraphUnavailable as error:
            logger.warning("correction chain unavailable: %s", error)
            return None
        members = [str(member.doc_id) for member in chain]
        if len(members) < 2:
            return None
        latest = next(
            (str(member.doc_id) for member in chain if member.is_latest), None
        )
        if latest is None:
            return None
        if intent == INTENT_LATEST:
            return members, [latest]
        # History wants the original, every correction between, and the final
        # version, in the receipt order the chain already carries.
        return members, list(members)

    def _fetch_documents(self, doc_ids: Sequence[str]) -> list[CandidateDocument]:
        reader = getattr(self._metadata_backend, "fetch_documents", None)
        if not callable(reader):
            logger.warning(
                "correction expansion skipped: %s cannot fetch documents by id",
                type(self._metadata_backend).__name__,
            )
            return []
        return list(reader(doc_ids))

    def _select_chunks(
        self,
        plan: Any,
        documents: Sequence[CandidateDocument],
        *,
        start_rank: int,
        per_document: int,
    ) -> tuple[list[CandidateChunk], list[RetrievalResult]]:
        """Rank each added document's chunks with the question's own query."""

        added_chunks: list[CandidateChunk] = []
        added_results: list[RetrievalResult] = []
        rank = start_rank
        query = str(getattr(plan, "lexical_query", "") or "")
        for document in documents:
            candidates = list(self._chunk_backend.get_candidate_chunks([document]))
            if not candidates:
                continue
            selected = self._rank_within_document(query, candidates, per_document)
            for candidate in selected:
                match = dict(candidate.metadata_match.to_dict())
                match[PROVENANCE_KEY] = {
                    "doc_id": document.doc_id,
                    "relation": "correction_group_member",
                }
                added_chunks.append(
                    CandidateChunk(
                        chunk_id=candidate.chunk_id,
                        doc_id=candidate.doc_id,
                        chunk=candidate.chunk,
                        metadata_match=MetadataMatch(
                            hard_filters=candidate.metadata_match.hard_filters,
                            soft_boosts=candidate.metadata_match.soft_boosts,
                            soft_inputs=candidate.metadata_match.soft_inputs,
                            soft_score=candidate.metadata_match.soft_score,
                        ),
                    )
                )
                added_results.append(
                    RetrievalResult(
                        chunk_id=candidate.chunk_id,
                        doc_id=candidate.doc_id,
                        bm25_score=0.0,
                        rank=rank,
                        metadata_match=match,
                    )
                )
                rank += 1
        return added_chunks, added_results

    def _rank_within_document(
        self, query: str, candidates: Sequence[CandidateChunk], per_document: int
    ) -> list[CandidateChunk]:
        """Prefer the chunks the question's own terms match.

        Whole documents are never added: only the best few chunks per document,
        chosen with the same lexical ranker the initial retrieval used.  When the
        ranker returns nothing the document's leading chunks are used, so an
        added document always contributes readable evidence.
        """

        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        ranked: list[CandidateChunk] = []
        if query.strip():
            try:
                results = self._retriever.retrieve(
                    query, candidates, top_k=per_document
                )
            except Exception:  # noqa: BLE001 - fall back to document order
                results = []
            for result in results:
                candidate = by_id.get(result.chunk_id)
                if candidate is not None and candidate not in ranked:
                    ranked.append(candidate)
        for candidate in candidates:
            if len(ranked) >= per_document:
                break
            if candidate not in ranked:
                ranked.append(candidate)
        return ranked[:per_document]


def _correction_intent(plan: Any) -> str | None:
    evidence = getattr(plan, "evidence", None)
    if isinstance(evidence, Mapping):
        intent = evidence.get("correction_intent")
        if intent:
            return str(intent)
    return None


def _corp_code(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("corp_code")
    return str(value) if value else None


def _same_company_only(
    documents: Sequence[CandidateDocument],
    groups: Sequence[Mapping[str, Any]],
    metadata_by_doc: Mapping[str, Mapping[str, Any]],
) -> list[CandidateDocument]:
    """Drop anything that is not the seed's own company.

    Every resolution rule already keys on ``corp_code``, so this cannot trigger
    on a correctly built graph.  It is here so a damaged row can never pull
    another company's disclosure into an answer.
    """

    expected = {
        str(group["seed_corp_code"])
        for group in groups
        if group.get("seed_corp_code")
    }
    if not expected:
        return list(documents)
    kept: list[CandidateDocument] = []
    for document in documents:
        corp_code = _corp_code(document.metadata)
        if corp_code is None or corp_code in expected:
            kept.append(document)
        else:
            logger.warning(
                "correction expansion dropped %s: corp_code %s is not %s",
                document.doc_id,
                corp_code,
                sorted(expected),
            )
    return kept


def _trace(
    intent: str | None,
    status: str,
    *,
    groups: Sequence[Mapping[str, Any]] = (),
    added_doc_ids: Sequence[str] = (),
    added_result_count: int = 0,
) -> dict[str, Any]:
    """Execution summary only: what was expanded, from which chain, and why."""

    first = dict(groups[0]) if groups else {}
    return {
        "correction_intent": intent,
        "correction_expanded": status == "expanded",
        "correction_status": status,
        "correction_group_id": first.get("correction_group_id"),
        "correction_root_doc_id": first.get("root_doc_id"),
        "correction_latest_doc_id": first.get("latest_doc_id"),
        "correction_group_count": len(groups),
        "correction_added_doc_ids": list(added_doc_ids),
        "correction_added_result_count": int(added_result_count),
    }


def apply_expansion(
    expansion: CorrectionExpansion,
    chunks: Sequence[CandidateChunk],
    results: Sequence[RetrievalResult],
) -> tuple[list[CandidateChunk], list[RetrievalResult]]:
    """Append the added evidence, leaving the retrieved ranking untouched."""

    if not expansion.expanded:
        return list(chunks), list(results)
    known = {candidate.chunk_id for candidate in chunks}
    merged_chunks = list(chunks)
    for candidate in expansion.added_chunks:
        if candidate.chunk_id not in known:
            known.add(candidate.chunk_id)
            merged_chunks.append(candidate)
    ranked = {result.chunk_id for result in results}
    merged_results = list(results)
    for result in expansion.added_results:
        if result.chunk_id not in ranked:
            ranked.add(result.chunk_id)
            merged_results.append(result)
    return merged_chunks, merged_results


def build_default_expander(
    repository: Any,
    metadata_backend: Any,
    chunk_backend: Any,
    retriever: Any,
    **kwargs: Any,
) -> CorrectionExpander | None:
    """Build an expander, or None when a required capability is missing."""

    if repository is None or not callable(
        getattr(metadata_backend, "fetch_documents", None)
    ):
        return None
    return CorrectionExpander(
        repository, metadata_backend, chunk_backend, retriever, **kwargs
    )
