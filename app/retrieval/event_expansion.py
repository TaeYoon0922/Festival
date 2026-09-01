"""Deterministic lifecycle expansion of an existing retrieval result.

Metadata retrieval answers "which disclosures match this question".  For a
question about a contract's fate that is the wrong universe: a question anchored
on the year the contract was signed can never reach the termination filed two
years later, because the anchor date is a hard filter on the candidate set.  The
event graph already knows those filings are one contract, so this layer adds them
back after retrieval instead of loosening the filter that found the first one.

Nothing here searches, plans, iterates, or asks a model anything.  It reads the
lifecycle the graph already resolved, fetches those documents by identity, and
picks chunks inside them with the same lexical query the original retrieval used.
This is deliberately one pass: following the graph once, not searching again.

Expansion is gated on the question's own route.  Only a question the router
already classified as a contract event expands, so a periodic, holding, or
unrelated major question is answered from exactly what retrieval found.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.reasoning.corporate_event_graph import CorporateEventGraphUnavailable
from app.reasoning.corporate_event_resolver import (
    CorporateEventResolver,
    seed_expansion_targets,
)
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


logger = logging.getLogger(__name__)

#: Routes whose questions are about a contract's life.  These are the
#: ``event_type`` values the query understanding layer already produces, so the
#: gate reuses the existing routing vocabulary instead of inventing a second one.
CONTRACT_EVENT_TYPES = frozenset(
    {
        "supply_contract",
        "contract_termination",
        "treasury_share_trust_contract",
        "treasury_share_trust_termination",
    }
)

#: Marker written into an added result's ``metadata_match`` so relation-derived
#: evidence stays distinguishable from what retrieval itself ranked.
PROVENANCE_KEY = "event_expansion"

#: How many lifecycle filings one question may pull in. The largest logical
#: lifecycle in the corpus holds three members, so this is a guard against a
#: damaged graph rather than a working limit.
DEFAULT_EVENT_EXPANSION_LIMIT = 8
#: Total chunks the whole expansion may contribute, across every added document.
DEFAULT_EVENT_EVIDENCE_LIMIT = 12

#: Labels the corpus audit found on contract and termination filings. A chunk
#: carrying these is the structured lifecycle row an answer needs to cite, so it
#: is preferred over prose from the same filing. Nothing is invented here: every
#: label below was observed in the frozen tables.
_SUPPLY_EVIDENCE_LABELS = (
    "해지계약명", "체결계약명", "계약상대", "해지금액", "계약금액",
    "해지일자", "해지 주요사유", "해지주요사유", "시작일", "종료일",
    "계약(수주)일자", "관련공시", "최근매출액",
)
_TRUST_EVIDENCE_LABELS = (
    "계약체결기관", "해지기관", "해지예정일자", "계약금액",
    "시작일", "종료일", "해지전", "해지후", "신탁계약의 계약금액",
)
_EVIDENCE_LABELS = _SUPPLY_EVIDENCE_LABELS + _TRUST_EVIDENCE_LABELS


def _lifecycle_field_hits(candidate: CandidateChunk) -> int:
    """How many audited lifecycle labels this chunk states."""

    chunk = candidate.chunk if isinstance(candidate.chunk, Mapping) else {}
    text = str(chunk.get("retrieval_text") or chunk.get("content") or "")
    if not text:
        return 0
    return sum(1 for label in _EVIDENCE_LABELS if label in text)


@dataclass(frozen=True)
class EventExpansion:
    """Documents the event graph contributed to one execution."""

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


def _event_type(plan: Any) -> str | None:
    value = getattr(plan, "event_type", None)
    return str(value) if value else None


def _corp_code(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("corp_code")
    return str(value) if value else None


class EventExpander:
    """Add the rest of a resolved contract lifecycle to a retrieval result.

    ``max_documents`` is a safety cap, not a working limit: membership is stored
    per logical contract, so the largest lifecycle in the corpus holds three
    members.  The default leaves generous room for that and still refuses a
    pathological group.  ``max_evidence`` bounds the total context the expansion
    may contribute regardless of how many documents it reached.
    """

    def __init__(
        self,
        repository: Any,
        metadata_backend: Any,
        chunk_backend: Any,
        retriever: Any,
        *,
        max_documents: int = DEFAULT_EVENT_EXPANSION_LIMIT,
        chunks_per_document: int = 2,
        seed_limit: int = 10,
        max_evidence: int = DEFAULT_EVENT_EVIDENCE_LIMIT,
    ) -> None:
        if min(max_documents, chunks_per_document, seed_limit, max_evidence) <= 0:
            raise ValueError("event expansion limits must be positive")
        self._resolver = CorporateEventResolver(repository)
        self._repository = repository
        self._metadata_backend = metadata_backend
        self._chunk_backend = chunk_backend
        self._retriever = retriever
        self.max_documents = max_documents
        self.chunks_per_document = chunks_per_document
        self.seed_limit = seed_limit
        self.max_evidence = max_evidence

    # ------------------------------------------------------------------ api

    def expand(
        self,
        plan: Any,
        *,
        documents: Sequence[CandidateDocument],
        chunks: Sequence[CandidateChunk],
        results: Sequence[RetrievalResult],
    ) -> EventExpansion:
        started = time.perf_counter()
        event_type = _event_type(plan)
        if event_type not in CONTRACT_EVENT_TYPES:
            # Not a contract question.  Retrieval's own result stands untouched.
            return EventExpansion(trace=_trace(event_type, "not_requested"))
        if not results:
            return EventExpansion(trace=_trace(event_type, "no_seed_results"))

        seeds = _seed_doc_ids(results, self.seed_limit)
        try:
            wanted, events, diagnostics = seed_expansion_targets(
                self._repository, seeds, limit=self.max_documents
            )
        except CorporateEventGraphUnavailable as error:
            # db/007 not applied, or the database is unreachable.  Retrieval's own
            # result stands; a programming error would have propagated instead.
            logger.warning("event expansion skipped: %s", error)
            return EventExpansion(
                trace=_trace(event_type, "graph_unavailable", seeds=seeds)
            )

        common = {
            "seeds": seeds,
            "events": events,
            "diagnostics": diagnostics,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
        if not events:
            # Includes every ambiguous or unresolved termination: those are stored
            # as an event of their own and must never pull in a lookalike contract.
            return EventExpansion(trace=_trace(event_type, "no_resolved_event", **common))
        if not wanted:
            return EventExpansion(trace=_trace(event_type, "already_retrieved", **common))

        metadata_by_doc = {
            document.doc_id: document.metadata for document in documents
        }
        added_documents = self._fetch_documents(wanted)
        added_documents = self._same_company_only(
            added_documents, events, metadata_by_doc
        )
        if not added_documents:
            return EventExpansion(
                trace=_trace(event_type, "documents_unavailable", **common)
            )

        added_chunks, added_results = self._select_chunks(
            plan, added_documents, start_rank=len(results) + 1
        )
        common["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
        if not added_results:
            return EventExpansion(trace=_trace(event_type, "no_chunks", **common))
        return EventExpansion(
            added_chunks=tuple(added_chunks),
            added_results=tuple(added_results),
            added_documents=tuple(added_documents),
            trace=_trace(
                event_type,
                "expanded",
                **{**common, "added_documents": added_documents,
                   "added_result_count": len(added_results)},
            ),
        )

    # -------------------------------------------------------------- helpers

    def _fetch_documents(self, doc_ids: Sequence[str]) -> list[CandidateDocument]:
        reader = getattr(self._metadata_backend, "fetch_documents", None)
        if not callable(reader):
            logger.warning(
                "event expansion skipped: %s cannot fetch documents by id",
                type(self._metadata_backend).__name__,
            )
            return []
        return list(reader(doc_ids))

    def _same_company_only(
        self,
        documents: Sequence[CandidateDocument],
        events: Sequence[Mapping[str, Any]],
        metadata_by_doc: Mapping[str, Mapping[str, Any]],
    ) -> list[CandidateDocument]:
        """Drop anything that is not the seed's own company.

        Every matching rule already keys on ``corp_code``, so this cannot trigger
        on a correctly built graph.  It is here so a damaged row can never pull
        another company's disclosure into an answer.
        """

        expected = {
            str(event["seed_corp_code"])
            for event in events
            if event.get("seed_corp_code")
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
                    "event expansion dropped %s: corp_code %s is not %s",
                    document.doc_id,
                    corp_code,
                    sorted(expected),
                )
        return kept

    def _select_chunks(
        self,
        plan: Any,
        documents: Sequence[CandidateDocument],
        *,
        start_rank: int,
    ) -> tuple[list[CandidateChunk], list[RetrievalResult]]:
        """Rank each added document's chunks with the question's own query."""

        added_chunks: list[CandidateChunk] = []
        added_results: list[RetrievalResult] = []
        rank = start_rank
        query = str(getattr(plan, "lexical_query", "") or "")
        for document in documents:
            if len(added_results) >= self.max_evidence:
                # Whole-corpus safety: the expansion never grows the context
                # without a ceiling, however many documents a lifecycle holds.
                break
            candidates = list(self._chunk_backend.get_candidate_chunks([document]))
            if not candidates:
                continue
            state = self._resolver.get_event_state(document.doc_id)
            room = self.max_evidence - len(added_results)
            for candidate in self._rank_within_document(query, candidates)[:room]:
                match = dict(candidate.metadata_match.to_dict())
                match[PROVENANCE_KEY] = {
                    "doc_id": document.doc_id,
                    "relation": "corporate_event_member",
                    "retrieval_source": PROVENANCE_KEY,
                    "event_id": _enum_text(getattr(state, "event_id", None)),
                    "member_role": _enum_text(getattr(state, "member_role", None)),
                    "lifecycle_status": _enum_text(
                        getattr(state, "lifecycle_status", None)
                    ),
                    "canonical_doc_id": getattr(state, "canonical_doc_id", None),
                    "correction_group_id": getattr(
                        state, "correction_group_id", None
                    ),
                    "lifecycle_field_hits": _lifecycle_field_hits(candidate),
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
        self, query: str, candidates: Sequence[CandidateChunk]
    ) -> list[CandidateChunk]:
        """Prefer the chunks the question's own terms match.

        Whole documents are never added: only the best few chunks per document,
        chosen with the same lexical ranker the initial retrieval used.  When the
        ranker returns nothing the document's leading chunks are used, so an
        added document always contributes readable evidence.
        """

        per_document = self.chunks_per_document
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        ranked: list[CandidateChunk] = []
        # A lifecycle answer is made of the filing's structured rows -- the
        # counterparty, the amount, the dates, the stated reason.  Chunks
        # carrying those labels come first; the lexical ranker then orders the
        # rest.  Both passes are deterministic.
        structured = sorted(
            (item for item in candidates if _lifecycle_field_hits(item) > 0),
            key=lambda item: (-_lifecycle_field_hits(item), item.chunk_id),
        )
        ranked.extend(structured[:per_document])
        if len(ranked) < per_document and query.strip():
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


def _enum_text(value: Any) -> Any:
    """Enum members serialize as their value, so a trace stays JSON-clean."""

    return getattr(value, "value", value)


def _trace(
    event_type: str | None,
    status: str,
    *,
    seeds: Sequence[str] = (),
    events: Sequence[Mapping[str, Any]] = (),
    diagnostics: Mapping[str, Any] | None = None,
    added_documents: Sequence[CandidateDocument] = (),
    added_result_count: int = 0,
    elapsed_ms: float = 0.0,
) -> dict[str, Any]:
    """Execution summary only: what was expanded, from which lifecycle, and why.

    Shaped like the correction trace beside it so the pipeline renders both the
    same way, with one nested block carrying the lifecycle detail.
    """

    first = dict(events[0]) if events else {}
    info = dict(diagnostics or {})
    added_doc_ids = [document.doc_id for document in added_documents]
    return {
        "event_type": event_type,
        "event_expanded": status == "expanded",
        "event_status": status,
        # P0-B state for the seeds, carried whole rather than flattened.  A
        # sibling of the public block below, never inside it: the pipeline
        # renders ``corporate_event_expansion`` and nothing else from here, so
        # this stays an internal seam and no identifier reaches the API.
        "event_member_states": dict(info.get("member_states") or {}),
        "event_id": first.get("event_id"),
        "event_family": _enum_text(first.get("event_family")),
        "event_lifecycle_status": _enum_text(first.get("lifecycle_status")),
        "event_resolution_source": first.get("resolution_source"),
        "event_count": len(events),
        "event_added_doc_ids": list(added_doc_ids),
        "event_added_result_count": int(added_result_count),
        "corporate_event_expansion": {
            "applied": status == "expanded",
            "reason": status,
            "seed_doc_ids": list(seeds),
            "matched_event_ids": [
                event.get("event_id") for event in events if event.get("event_id")
            ],
            "added_doc_ids": list(added_doc_ids),
            "skipped_doc_ids": [
                item.get("seed_doc_id") for item in info.get("skipped") or ()
            ],
            "skipped": list(info.get("skipped") or ()),
            "truncated": bool(info.get("truncated")),
            "seed_member_doc_ids": dict(info.get("seed_member_doc_ids") or {}),
            "events": [
                {
                    key: _enum_text(value)
                    for key, value in event.items()
                    if key != "skipped"
                }
                for event in events
            ],
            "metrics": {
                "seed_count": len(seeds),
                "graph_lookup_count": int(info.get("events_considered") or 0),
                "added_doc_count": len(added_doc_ids),
                "added_evidence_count": int(added_result_count),
                "elapsed_ms": round(float(elapsed_ms), 3),
            },
        },
    }


def build_default_event_expander(
    repository: Any,
    metadata_backend: Any,
    chunk_backend: Any,
    retriever: Any,
    **kwargs: Any,
) -> EventExpander | None:
    """Build an expander, or None when a required capability is missing."""

    if repository is None or not callable(
        getattr(metadata_backend, "fetch_documents", None)
    ):
        return None
    return EventExpander(
        repository, metadata_backend, chunk_backend, retriever, **kwargs
    )
