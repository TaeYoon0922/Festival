"""Hydrate an executed ``MultiDocumentPlan`` into evidence (P0-C Step 5).

Two layers are kept deliberately separate:

*Completeness* is measured over the whole logical set -- all 14 contracts -- and
survives only as counts.  *Presentation* hydrates the few filings an answer must
actually cite.  Conflating them would either blow the context budget or make a
count look like it came from the handful of documents that happened to fit.

Nothing here re-reads the question, re-derives a set, or walks a graph of its
own: the ids come from Step 4 and the filings come from the P0-B public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.reasoning.multi_document_plan import SlotStatus, SlotType
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


#: Provenance key, matching the ``correction_expansion`` / ``event_expansion``
#: convention so every added row is attributable the same way.
PROVENANCE_KEY = "multi_document_planner"

#: Hydration ceilings.  Mirrors ``DEFAULT_EVENT_EVIDENCE_LIMIT`` (12) so P0-C
#: cannot contribute more context than P0-B expansion already may.  These bound
#: *presentation* only -- completeness is always measured over the full set.
MAX_MULTI_DOC_EVIDENCE = 12
MAX_PER_DOCUMENT_CHUNKS = 2

#: What the deterministic set says about a lifecycle question.
LIFECYCLE_EXISTS = "exists"
LIFECYCLE_NONE = "none"
LIFECYCLE_UNDETERMINED = "undetermined"
LIFECYCLE_NO_MEMBERS = "no_members"


@dataclass(frozen=True)
class MultiDocumentFacts:
    """Deterministic answer material.  The model never recomputes these.

    Every number here came from a metadata enumeration, not from counting the
    documents that fit in the context window.
    """

    plan_type: str
    complete: bool
    logical_count: int = 0
    terminated_count: int = 0
    open_count: int = 0
    unresolved_count: int = 0
    lifecycle_answer: str | None = None
    family_resolution: str | None = None
    stop_reason: str | None = None

    def statement(self) -> str:
        """The deterministic sentence the answer states, in Korean.

        Every number is read from this object, never computed by a model.  The
        four lifecycle states are worded so they cannot collapse into each
        other: "checked and found none" and "could not finish checking" are
        different claims, and only the first may be phrased as a negative.
        """

        count = self.logical_count
        unresolved = self.unresolved_count
        if self.lifecycle_answer == LIFECYCLE_EXISTS:
            return (
                f"조건에 해당하는 계약 {count}건을 확인했으며, "
                f"이 중 이후 해지된 계약은 {self.terminated_count}건입니다."
            )
        if self.lifecycle_answer == LIFECYCLE_NONE:
            return (
                f"조건에 해당하는 계약 {count}건을 모두 확인했으며, "
                "이후 해지된 계약은 확인되지 않았습니다."
            )
        if self.lifecycle_answer == LIFECYCLE_UNDETERMINED:
            # Never a negative: the set was not fully verified, so "없다" would
            # assert something the evidence does not support.
            return (
                f"조건에 해당하는 계약 {count}건 중 {unresolved}건은 "
                "후속 공시와의 연결 근거를 확정하지 못해, "
                "해지 여부를 단정할 수 없습니다."
            )
        if self.lifecycle_answer == LIFECYCLE_NO_MEMBERS:
            return "해당 기간에 조건에 맞는 계약이 확인되지 않았습니다."
        # Cardinality only.  An incomplete set still reports its count, but
        # never as if the whole set had been verified.
        if unresolved or not self.complete:
            return (
                f"조건에 해당하는 계약은 {count}건으로 확인되며, "
                f"이 중 {unresolved}건은 관련 공시 연결을 확정하지 못했습니다."
            )
        if count == 0:
            return "해당 기간에 조건에 맞는 계약이 확인되지 않았습니다."
        return f"조건에 해당하는 계약은 모두 {count}건입니다."

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "plan_type": self.plan_type,
            "complete": self.complete,
            "logical_count": self.logical_count,
            "unresolved_count": self.unresolved_count,
        }
        if self.lifecycle_answer is not None:
            payload.update(
                {
                    "lifecycle_answer": self.lifecycle_answer,
                    "terminated_count": self.terminated_count,
                    "open_count": self.open_count,
                }
            )
        if self.family_resolution:
            payload["family_resolution"] = self.family_resolution
        if self.stop_reason:
            payload["stop_reason"] = self.stop_reason
        return payload


@dataclass(frozen=True)
class MultiDocumentEvidence:
    """Chunks and results to append, plus the facts the answer states."""

    facts: MultiDocumentFacts | None = None
    added_chunks: tuple[CandidateChunk, ...] = ()
    added_results: tuple[RetrievalResult, ...] = ()
    added_doc_ids: tuple[str, ...] = ()
    #: The executed plan, so the trace reuses ``execution.to_dict()`` rather
    #: than re-deriving per-slot counts.
    execution: Any = None

    def trace(self) -> dict[str, Any]:
        """Counts and statuses only; identifiers never leave this object."""

        payload: dict[str, Any] = {"applied": True}
        if self.execution is not None:
            payload.update(self.execution.to_dict())
        if self.facts is not None:
            payload.update(self.facts.to_dict())
        payload["evidence_count"] = len(self.added_results)
        return payload

    @property
    def expanded(self) -> bool:
        return bool(self.added_results)


def lifecycle_answer(
    *, logical_count: int, terminated_count: int, unresolved_count: int, complete: bool
) -> str:
    """Decide a lifecycle question from the deterministic set alone.

    "Checked every contract and found no termination" and "could not check every
    contract" are different answers, and only the first may be stated as a
    negative.  An unresolved member or an incomplete plan always downgrades to
    ``undetermined`` rather than to "none".
    """

    if terminated_count > 0:
        return LIFECYCLE_EXISTS
    if unresolved_count > 0 or not complete:
        return LIFECYCLE_UNDETERMINED
    if logical_count == 0:
        # No contracts in the period at all, so no terminated ones either.
        return LIFECYCLE_NO_MEMBERS
    return LIFECYCLE_NONE


def build_facts(execution: Any) -> MultiDocumentFacts:
    """Summarize an executed plan into answer material."""

    plan = execution.plan
    enumeration = next(
        (
            slot
            for slot in plan.slots
            if slot.slot_type
            in (SlotType.ENUMERATE_EVENTS, SlotType.ENUMERATE_DOCUMENTS)
        ),
        None,
    )
    lifecycle_slot = next(
        (slot for slot in plan.slots if slot.slot_type is SlotType.EVENT_STATE), None
    )
    logical_count = enumeration.expected_count if enumeration is not None else 0
    unresolved = sum(len(slot.unresolved_ids) for slot in plan.slots)
    terminated = opened = 0
    answer = None
    if lifecycle_slot is not None:
        outcome = execution.outcome(lifecycle_slot.slot_id)
        terminated = outcome.terminated_count if outcome else 0
        opened = outcome.open_count if outcome else 0
        answer = lifecycle_answer(
            logical_count=logical_count,
            terminated_count=terminated,
            unresolved_count=unresolved,
            complete=plan.complete,
        )
    return MultiDocumentFacts(
        plan_type=plan.plan_type,
        complete=plan.complete,
        logical_count=logical_count,
        terminated_count=terminated,
        open_count=opened,
        unresolved_count=unresolved,
        lifecycle_answer=answer,
        family_resolution=plan.family_resolution,
        stop_reason=plan.stop_reason,
    )


class MultiDocumentEvidenceBuilder:
    """Turn Step 4's ids into citable filings, within a fixed budget."""

    def __init__(
        self,
        *,
        event_repository: Any = None,
        metadata_backend: Any = None,
        chunk_backend: Any = None,
        retriever: Any = None,
        max_evidence: int = MAX_MULTI_DOC_EVIDENCE,
        chunks_per_document: int = MAX_PER_DOCUMENT_CHUNKS,
    ) -> None:
        if min(max_evidence, chunks_per_document) <= 0:
            raise ValueError("multi-document evidence limits must be positive")
        self._events = event_repository
        self._metadata_backend = metadata_backend
        self._chunk_backend = chunk_backend
        self._retriever = retriever
        self.max_evidence = max_evidence
        self.chunks_per_document = chunks_per_document

    def build(
        self, execution: Any, *, plan: Any = None, start_rank: int = 1
    ) -> MultiDocumentEvidence:
        """Hydrate the evidence an applied plan justifies citing."""

        if execution is None or not execution.applied:
            return MultiDocumentEvidence()
        facts = build_facts(execution)
        if self._metadata_backend is None or self._chunk_backend is None:
            # Facts alone are still worth carrying: the counts are deterministic
            # even when no evidence store is wired (unit and diagnostic paths).
            return MultiDocumentEvidence(facts=facts, execution=execution)

        doc_ids = self._ordered_doc_ids(execution)
        if not doc_ids:
            return MultiDocumentEvidence(facts=facts, execution=execution)
        documents = list(self._metadata_backend.fetch_documents(doc_ids))
        order = {doc_id: index for index, doc_id in enumerate(doc_ids)}
        documents.sort(key=lambda item: order.get(item.doc_id, len(order)))
        chunks, results = self._select_chunks(
            documents, plan=plan, start_rank=start_rank, execution=execution
        )
        return MultiDocumentEvidence(
            facts=facts,
            execution=execution,
            added_chunks=tuple(chunks),
            added_results=tuple(results),
            added_doc_ids=tuple(document.doc_id for document in documents),
        )

    # ------------------------------------------------------------- selection

    def _ordered_doc_ids(self, execution: Any) -> list[str]:
        """Answer-critical filings first, then representative members.

        A termination claim must be able to cite the termination filing *and*
        the contract it closed, so those two lead. Whatever budget is left goes
        to the rest of the enumerated set.
        """

        plan = execution.plan
        terminated_events: list[str] = []
        for slot in plan.slots:
            if slot.slot_type is not SlotType.EVENT_STATE:
                continue
            outcome = execution.outcome(slot.slot_id)
            if outcome is not None:
                terminated_events.extend(outcome.terminated_ids)

        ordered: list[str] = []
        seen: set[str] = set()

        def add(value: Any) -> None:
            text = str(value or "")
            if text and text not in seen:
                seen.add(text)
                ordered.append(text)

        # 1. termination filings, and 2. the openings they closed
        for doc_id in self._termination_documents(plan, terminated_events):
            add(doc_id)
        openings = self._opening_documents(execution)
        for event_id in terminated_events:
            add(openings.get(event_id))
        # 3. openings whose lifecycle evidence is unresolved.  This is the
        # provenance behind an ``undetermined`` answer, so it must not lose its
        # place merely because representative open contracts filled the budget.
        unresolved_events = {
            str(event_id)
            for slot in plan.slots
            for event_id in slot.unresolved_ids
        }
        document_members = {
            str(doc_id)
            for doc_ids in execution.document_ids.values()
            for doc_id in doc_ids
        }
        for event_id in sorted(unresolved_events):
            # Tier 1 carries event ids and resolves them through ``openings``;
            # Tier 2 carries disclosure ids directly after P0-A collapse.
            add(openings.get(event_id) or (
                event_id if event_id in document_members else None
            ))
        # 4. the remaining enumerated members
        for slot in plan.slots:
            for doc_id in execution.document_ids.get(slot.slot_id, ()):  # type: ignore[union-attr]
                add(doc_id)
        return ordered[: self.max_evidence]

    def _opening_documents(self, execution: Any) -> dict[str, str]:
        """event_id -> opening filing, read from the explicit mapping.

        Never derived by pairing two sorted sequences: that would attach the
        wrong contract to a lifecycle the moment either order changed, and the
        error would be silent.
        """

        return dict(getattr(execution, "opening_documents", {}) or {})

    def _termination_documents(
        self, plan: Any, terminated_events: Sequence[str]
    ) -> list[str]:
        """The filings that closed the terminated lifecycles.

        One batch call against the same window the enumeration used, reusing the
        P0-B enumeration primitive rather than walking the graph per event.
        """

        if not terminated_events or self._events is None:
            return []
        source = next(
            (
                slot
                for slot in plan.slots
                if slot.slot_type is SlotType.ENUMERATE_EVENTS
            ),
            None,
        )
        if source is None:
            return []
        wanted = set(terminated_events)
        states = self._events.enumerate_events(
            corp_code=source.corp_code,
            event_family=source.event_family,
            member_role="termination",
            date_field=source.date_field or "opened_at",
            date_from=source.date_from,
            date_to=source.date_to,
        )
        return [
            str(state.doc_id)
            for state in states
            if str(state.event_id) in wanted and state.doc_id
        ]

    def _select_chunks(
        self,
        documents: Sequence[CandidateDocument],
        *,
        plan: Any,
        start_rank: int,
        execution: Any,
    ) -> tuple[list[CandidateChunk], list[RetrievalResult]]:
        added_chunks: list[CandidateChunk] = []
        added_results: list[RetrievalResult] = []
        rank = start_rank
        query = str(getattr(plan, "lexical_query", "") or "")
        slot_by_doc = self._slot_by_document(execution)
        ranked_by_document: list[tuple[CandidateDocument, list[CandidateChunk]]] = []
        for document in documents:
            candidates = list(self._chunk_backend.get_candidate_chunks([document]))
            if not candidates:
                continue
            ranked_by_document.append(
                (
                    document,
                    self._rank_within_document(query, candidates)[
                        : self.chunks_per_document
                    ],
                )
            )

        # Give every answer-critical document one citation before spending a
        # second chunk on any document.  Sequential per-document filling made a
        # five-termination trust case consume all 12 rows on termination
        # filings plus one opening, leaving four opening identity claims
        # uncited even though all ten documents fit the document budget.
        for chunk_index in range(self.chunks_per_document):
            for document, candidates in ranked_by_document:
                if len(added_results) >= self.max_evidence:
                    break
                if chunk_index >= len(candidates):
                    continue
                candidate = candidates[chunk_index]
                slot = slot_by_doc.get(document.doc_id)
                match = dict(candidate.metadata_match.to_dict())
                match[PROVENANCE_KEY] = {
                    "doc_id": document.doc_id,
                    "relation": "logical_set_member",
                    "retrieval_source": PROVENANCE_KEY,
                    "planner_slot_id": slot.slot_id if slot else None,
                    "planner_slot_type": slot.slot_type.value if slot else None,
                    "event_family": slot.event_family if slot else None,
                    "family_resolution": execution.plan.family_resolution,
                    "logical_set_member": True,
                }
                added_chunks.append(
                    CandidateChunk(
                        chunk_id=candidate.chunk_id,
                        doc_id=candidate.doc_id,
                        chunk=candidate.chunk,
                        metadata_match=candidate.metadata_match,
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
            if len(added_results) >= self.max_evidence:
                break
        return added_chunks, added_results

    @staticmethod
    def _slot_by_document(execution: Any) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for slot in execution.plan.slots:
            for doc_id in execution.document_ids.get(slot.slot_id, ()):
                mapping.setdefault(str(doc_id), slot)
        return mapping

    def _rank_within_document(
        self, query: str, candidates: Sequence[CandidateChunk]
    ) -> list[CandidateChunk]:
        """Same lexical ranker retrieval used, so added chunks stay readable."""

        if not query or self._retriever is None:
            return list(candidates)
        try:
            ranked = self._retriever.retrieve(
                query, candidates, top_k=self.chunks_per_document
            )
        except Exception:  # noqa: BLE001 - ranking is a preference, not a contract
            return list(candidates)
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        ordered = [by_id[item.chunk_id] for item in ranked if item.chunk_id in by_id]
        return ordered or list(candidates)


__all__ = [
    "LIFECYCLE_EXISTS",
    "LIFECYCLE_NONE",
    "LIFECYCLE_NO_MEMBERS",
    "LIFECYCLE_UNDETERMINED",
    "MAX_MULTI_DOC_EVIDENCE",
    "MAX_PER_DOCUMENT_CHUNKS",
    "PROVENANCE_KEY",
    "MultiDocumentEvidence",
    "MultiDocumentEvidenceBuilder",
    "MultiDocumentFacts",
    "build_facts",
    "lifecycle_answer",
]
