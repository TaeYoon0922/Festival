"""Execute a ``MultiDocumentPlan`` deterministically (P0-C Step 4).

Step 3 decided *what* set the answer needs; this module finds out *whether the
set is covered*.  It never re-reads the question -- every input it uses is
already on a slot -- and it writes no SQL of its own, calling only the Step 2
repository primitives.

Two passes, and only two:

    Pass 1   set enumeration        ENUMERATE_EVENTS / ENUMERATE_DOCUMENTS
    Pass 2   dependent state checks EVENT_STATE

There is no third pass and no similarity rescue.  P0-A and P0-B expansion
already ran inside retrieval; re-running them here would duplicate frozen
behaviour, and a similarity pass would be exactly the false-positive path P0-C
exists to avoid.  When a set cannot be covered, the honest output is an
``UNRESOLVED`` slot handed to the answer layer -- never a guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from app.reasoning.corporate_event_graph import CorporateEventGraphUnavailable
from app.reasoning.correction_graph import CorrectionGraphUnavailable
from app.reasoning.multi_document_plan import (
    REASON_EMPTY_SET,
    EvidenceSlot,
    MultiDocumentPlan,
    SlotStatus,
    SlotType,
)
from app.retrieval.enumeration import collapse_logical_documents


#: Pass 1 enumerates, pass 2 checks dependent state.  Fixed by Step 1's audit:
#: no third deterministic action exists.
MAX_PLANNER_PASSES = 2

STOP_ALL_SLOTS_COMPLETE = "all_slots_complete"
STOP_NO_DETERMINISTIC_ACTION = "no_deterministic_action"
STOP_MAX_PASSES = "max_passes"
STOP_REPOSITORY_UNAVAILABLE = "repository_unavailable"

#: An id whose lifecycle could not be determined.  Distinct from "checked and
#: still open": the first is missing evidence, the second is an answer.
REASON_NO_LIFECYCLE_EVIDENCE = "no_lifecycle_evidence"

_ENUMERATION_TYPES = (SlotType.ENUMERATE_EVENTS, SlotType.ENUMERATE_DOCUMENTS)


@dataclass(frozen=True)
class LifecycleOutcome:
    """What one ``EVENT_STATE`` slot found, as answer material.

    Kept out of :class:`EvidenceSlot` on purpose.  A slot answers "is the set
    covered"; this answers "what is the set's state".  Folding the second into
    the first would overload ``found_ids`` with two different meanings.
    """

    slot_id: str
    open_ids: tuple[str, ...] = ()
    terminated_ids: tuple[str, ...] = ()

    @property
    def terminated_count(self) -> int:
        return len(self.terminated_ids)

    @property
    def open_count(self) -> int:
        return len(self.open_ids)

    def to_dict(self) -> dict[str, Any]:
        """Counts only.  Raw identifiers stay inside the execution object."""

        return {
            "slot_id": self.slot_id,
            "open_count": self.open_count,
            "terminated_count": self.terminated_count,
        }


@dataclass(frozen=True)
class MultiDocumentExecution:
    """The executed plan plus the lifecycle detail Step 5 needs.

    ``plan`` remains the single source of truth for coverage; ``outcomes`` adds
    only what coverage cannot express.
    """

    plan: MultiDocumentPlan
    outcomes: tuple[LifecycleOutcome, ...] = ()
    unavailable_reason: str | None = None
    #: slot_id -> the disclosure ids that stand for that slot's logical members.
    #: Tier 1 logical ids are ``event_id``s, which are not disclosures, so this
    #: is what evidence hydration reads instead of re-querying. Never traced.
    document_ids: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    #: event_id -> the opening filing that represents it. An explicit mapping,
    #: never an ordering coincidence: hydration must attach the right contract
    #: to the right lifecycle even if either sequence is reordered.
    opening_documents: Mapping[str, str] = field(default_factory=dict)

    @property
    def applied(self) -> bool:
        return self.plan.applied

    @property
    def complete(self) -> bool:
        return self.plan.complete

    def outcome(self, slot_id: str) -> LifecycleOutcome | None:
        for item in self.outcomes:
            if item.slot_id == slot_id:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.plan.to_dict())
        if self.outcomes:
            payload["lifecycle"] = [item.to_dict() for item in self.outcomes]
        if self.unavailable_reason:
            payload["unavailable_reason"] = self.unavailable_reason
        return payload


class MultiDocumentExecutor:
    """Run a plan's slots against the Step 2 enumeration primitives."""

    def __init__(
        self,
        *,
        event_repository: Any = None,
        correction_repository: Any = None,
        disclosure_backend: Any = None,
        event_resolver: Any = None,
    ) -> None:
        self._events = event_repository
        self._corrections = correction_repository
        self._disclosures = disclosure_backend
        # Tier 2 lifecycle goes through the resolver when one is supplied, so
        # the same public surface P0-B already exposes is the only one used.
        self._resolver = event_resolver or event_repository

    def execute(self, plan: MultiDocumentPlan) -> MultiDocumentExecution:
        """Execute ``plan`` and return a new one; the input is never mutated."""

        if not plan.applied:
            # A declined plan reaches no repository at all. Gold60 relies on
            # this: 60 questions, zero database calls.
            return MultiDocumentExecution(plan=plan)

        executed: dict[str, EvidenceSlot] = {}
        # Tier 1 enumeration already returns lifecycle_status, so a dependent
        # EVENT_STATE slot reads this cache instead of issuing a second query.
        # Execution-local: nothing survives the call.
        states: dict[str, dict[str, Any]] = {}
        passes = 0
        try:
            for phase, types in (
                (1, _ENUMERATION_TYPES),
                (2, (SlotType.EVENT_STATE,)),
            ):
                slots = [
                    slot
                    for slot in plan.execution_order()
                    if slot.slot_type in types
                ]
                if not slots:
                    continue
                passes = phase
                for slot in slots:
                    executed[slot.slot_id] = self._run(slot, executed, states)
        except (CorporateEventGraphUnavailable, CorrectionGraphUnavailable) as error:
            # The graph is not there or the database is unreachable. Both are
            # degradable: the answer layer is told the set is uncovered rather
            # than being handed a partial set as if it were whole.
            return MultiDocumentExecution(
                plan=plan.with_slots(
                    plan.slots, passes=passes, stop_reason=STOP_REPOSITORY_UNAVAILABLE
                ),
                unavailable_reason=type(error).__name__,
            )

        ordered = tuple(
            executed.get(slot.slot_id, slot) for slot in plan.slots
        )
        outcomes = tuple(
            sorted(
                (
                    LifecycleOutcome(
                        slot_id=slot_id,
                        open_ids=tuple(sorted(value["open"])),
                        terminated_ids=tuple(sorted(value["terminated"])),
                    )
                    for slot_id, value in states.items()
                    if "open" in value
                ),
                key=lambda item: item.slot_id,
            )
        )
        result = plan.with_slots(
            ordered, passes=passes, stop_reason=_stop_reason(ordered, passes)
        )
        return MultiDocumentExecution(
            plan=result,
            outcomes=outcomes,
            document_ids={
                slot_id: tuple(value["documents"])
                for slot_id, value in states.items()
                if value.get("documents")
            },
            opening_documents={
                event_id: doc_id
                for value in states.values()
                for event_id, doc_id in (value.get("openings") or {}).items()
            },
        )

    # --------------------------------------------------------------- dispatch

    def _run(
        self,
        slot: EvidenceSlot,
        executed: Mapping[str, EvidenceSlot],
        states: dict[str, dict[str, Any]],
    ) -> EvidenceSlot:
        if not slot.is_definable:
            return slot.resolve_status()
        if slot.slot_type is SlotType.ENUMERATE_EVENTS:
            return self._enumerate_events(slot, states)
        if slot.slot_type is SlotType.ENUMERATE_DOCUMENTS:
            return self._enumerate_documents(slot, states)
        return self._event_state(slot, executed, states)

    # ------------------------------------------------------------ enumeration

    def _enumerate_events(
        self, slot: EvidenceSlot, states: dict[str, dict[str, Any]]
    ) -> EvidenceSlot:
        """Tier 1: one query returns the logical set *and* its lifecycle."""

        found = self._events.enumerate_events(
            corp_code=slot.corp_code,
            event_family=slot.event_family,
            member_role=slot.member_role or "contract",
            date_field=slot.date_field or "opened_at",
            date_from=slot.date_from,
            date_to=slot.date_to,
        )
        # The repository is the source of truth for this set, so what it
        # returned *is* what exists: expected and found cannot diverge.
        event_ids = tuple(sorted({str(state.event_id) for state in found}))
        # ``resolution_status != 'resolved'`` is NOT an evidence gap: 732 of the
        # corpus's 941 events are 'unresolved' only because a contract was filed
        # once with nothing to link to. Only a reference pointing outside the
        # corpus is a real gap, and that is what has_dangling_reference means.
        dangling = tuple(
            sorted(
                {
                    str(state.event_id)
                    for state in found
                    if state.has_dangling_reference
                }
            )
        )
        states[slot.slot_id] = {
            "by_id": {str(state.event_id): state for state in found},
            "cached": True,
            # The opening filing that represents each lifecycle, carried out of
            # the same query rather than looked up again per event.
            "documents": tuple(
                dict.fromkeys(
                    str(state.canonical_doc_id)
                    for state in found
                    if state.canonical_doc_id
                )
            ),
            "openings": {
                str(state.event_id): str(state.canonical_doc_id)
                for state in found
                if state.canonical_doc_id
            },
        }
        return slot.resolve_status_with(
            expected_ids=event_ids, found_ids=event_ids, unresolved_ids=dangling
        )

    def _enumerate_documents(
        self, slot: EvidenceSlot, states: dict[str, dict[str, Any]]
    ) -> EvidenceSlot:
        """Tier 2: enumerate raw filings, then collapse them through P0-A.

        Two statements regardless of set size -- never one per document.
        """

        documents = self._disclosures.enumerate_disclosures(
            corp_code=slot.corp_code,
            doc_group=slot.doc_group,
            doc_subtype=slot.doc_subtype,
            date_from=slot.date_from,
            date_to=slot.date_to,
        )
        doc_ids = [document.doc_id for document in documents]
        correction_states = (
            self._corrections.document_states(doc_ids) if doc_ids else {}
        )
        logical = collapse_logical_documents(doc_ids, correction_states)
        states[slot.slot_id] = {"documents": logical.representative_ids}
        return slot.resolve_status_with(
            expected_ids=logical.representative_ids,
            found_ids=logical.representative_ids,
            unresolved_ids=logical.unresolved_doc_ids,
        )

    # ---------------------------------------------------------- lifecycle

    def _event_state(
        self,
        slot: EvidenceSlot,
        executed: Mapping[str, EvidenceSlot],
        states: dict[str, dict[str, Any]],
    ) -> EvidenceSlot:
        """Attach lifecycle state to the ids a dependency enumerated."""

        sources = [executed.get(name) for name in slot.depends_on]
        if any(source is None for source in sources):
            raise ValueError(
                f"slot {slot.slot_id} ran before its dependency; "
                "execution_order is not a topological order"
            )
        expected = tuple(
            sorted({item for source in sources for item in source.expected_ids})
        )
        # A dependency that did not cover its own set cannot hand over a
        # complete input, so this slot inherits that gap instead of masking it.
        inherited = tuple(
            sorted({item for source in sources for item in source.missing_ids})
        )

        cached = self._cached_states(slot, states)
        if cached is not None:
            by_id = cached
        else:
            by_id = self._resolver.event_states(expected) if expected else {}

        checked: list[str] = []
        unresolved: list[str] = list(inherited)
        open_ids: list[str] = []
        terminated: list[str] = []
        for item in expected:
            state = by_id.get(item)
            if state is None:
                # The lookup ran and came back empty. That counts as checked --
                # UNRESOLVED, not INCOMPLETE -- because INCOMPLETE promises a
                # deterministic next action and there is none: P0-B simply holds
                # no lifecycle for this document. In v1 an EVENT_STATE slot is
                # only ever built for a contract family, so this is a corpus
                # boundary rather than a document that has no lifecycle at all.
                checked.append(item)
                unresolved.append(item)
                continue
            checked.append(item)
            if getattr(state, "has_dangling_reference", False):
                unresolved.append(item)
            if getattr(state, "is_terminated", False):
                terminated.append(item)
            else:
                open_ids.append(item)
        states.setdefault(slot.slot_id, {})
        states[slot.slot_id]["open"] = open_ids
        states[slot.slot_id]["terminated"] = terminated
        return slot.resolve_status_with(
            expected_ids=expected,
            found_ids=tuple(sorted(set(checked))),
            unresolved_ids=tuple(sorted(set(unresolved))),
        )

    def _cached_states(
        self, slot: EvidenceSlot, states: Mapping[str, dict[str, Any]]
    ) -> dict[str, Any] | None:
        """Reuse Tier 1 enumeration output instead of re-querying.

        ``enumerate_events`` already returned ``lifecycle_status`` and
        ``resolution_source`` for every member, so a Tier 1 lifecycle check
        costs zero extra round trips. Only a dependency that did not carry
        states -- Tier 2 -- reaches the resolver.
        """

        merged: dict[str, Any] = {}
        for name in slot.depends_on:
            entry = states.get(name)
            if not entry or not entry.get("cached"):
                return None
            merged.update(entry["by_id"])
        return merged or None


def _stop_reason(slots: Sequence[EvidenceSlot], passes: int) -> str:
    if all(
        slot.status in {SlotStatus.COMPLETE, SlotStatus.NOT_APPLICABLE}
        for slot in slots
    ):
        return STOP_ALL_SLOTS_COMPLETE
    if passes >= MAX_PLANNER_PASSES and any(
        slot.status is SlotStatus.INCOMPLETE for slot in slots
    ):
        return STOP_MAX_PASSES
    # Enumeration is authoritative and P0-A/P0-B already reported what they
    # could not resolve, so nothing further is deterministically available.
    return STOP_NO_DETERMINISTIC_ACTION


__all__ = [
    "MAX_PLANNER_PASSES",
    "REASON_NO_LIFECYCLE_EVIDENCE",
    "STOP_ALL_SLOTS_COMPLETE",
    "STOP_MAX_PASSES",
    "STOP_NO_DETERMINISTIC_ACTION",
    "STOP_REPOSITORY_UNAVAILABLE",
    "LifecycleOutcome",
    "MultiDocumentExecution",
    "MultiDocumentExecutor",
]
