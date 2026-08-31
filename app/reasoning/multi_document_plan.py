"""Domain model for the deterministic multi-document planner (P0-C).

A question like "OO기업이 2025년에 체결한 주요 계약 중 이후 해지된 계약이 있는가"
is not answered by ranking chunks: it is answered by *defining a set*, listing
that set deterministically, and then checking a property of every member.  This
module holds the vocabulary for that -- the slots, their statuses, and the plan
that groups them.

Nothing here performs I/O or decides *which* slots a question needs.  Step 2
supplies the model and the enumeration primitives only; the planner that turns a
question into slots (Step 3) and the bounded execution loop (Step 4) are built
on top of these types and must not define their own.

The status vocabulary here is deliberately a **different axis** from the
``resolution_status`` P0-A and P0-B already publish.  Those describe whether the
evidence itself could be linked; :class:`SlotStatus` describes whether retrieval
covered the set.  A contract that P0-B stores as ``unresolved`` because it is a
single filing with nothing to link to is perfectly complete evidence for an
enumeration, and conflating the two axes would mark most of the corpus missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence


#: Enumeration anchors on the date the filing puts on the *contract*, not on the
#: date the filing was received.  See :mod:`app.reasoning.corporate_event_graph`
#: (``ContractDocument.event_date``) for the precedence P0-B already applied.
DATE_FIELD_OPENED_AT = "opened_at"
DATE_FIELD_EVENT_DATE = "event_date"
DATE_FIELD_RCEPT_DT = "rcept_dt"
DATE_FIELDS = frozenset(
    {DATE_FIELD_OPENED_AT, DATE_FIELD_EVENT_DATE, DATE_FIELD_RCEPT_DT}
)

#: The opening role of a lifecycle.  A termination filing is evidence *about* a
#: contract, never a member of the contract list itself.
MEMBER_ROLE_CONTRACT = "contract"


class SlotType(str, Enum):
    """What kind of evidence one slot stands for."""

    #: Tier 1 -- list logical lifecycles out of the P0-B event timeline.
    ENUMERATE_EVENTS = "enumerate_events"
    #: Tier 2 -- list raw disclosures, then collapse them through P0-A.
    ENUMERATE_DOCUMENTS = "enumerate_documents"
    #: Attach lifecycle state to the members another slot enumerated.
    EVENT_STATE = "event_state"


class SlotStatus(str, Enum):
    """Whether retrieval has covered this slot's set."""

    PENDING = "pending"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNRESOLVED = "unresolved"
    NOT_APPLICABLE = "not_applicable"


#: Reasons attached to a status.  Kept as constants so the trace never carries a
#: free-form string a downstream consumer would have to parse.
REASON_ALL_MEMBERS_FOUND = "all_members_found"
REASON_MISSING_MEMBERS = "missing_members"
REASON_UNRESOLVED_MEMBERS = "unresolved_members"
REASON_NO_CORP_CODE = "no_corp_code"
REASON_NO_FAMILY = "no_family"
REASON_NO_DATE_RANGE = "no_date_range"
REASON_EMPTY_SET = "empty_set"
#: Step 3 (planner) non-engagement reasons.  Kept beside the slot reasons so
#: there is one vocabulary for "why is there no covered set here".
REASON_NO_SET_INTENT = "no_set_intent"
REASON_UNRESOLVED_DATE_BASIS = "unresolved_date_basis"
REASON_MIXED_DATE_BASIS = "mixed_date_basis"
REASON_UNSUPPORTED_CALCULATION = "unsupported_calculation"
REASON_UNSUPPORTED_TRUST_CONTRACT_BASIS = "unsupported_trust_contract_basis"

#: ``plan_type`` for a question P0-C declines.  Distinct from a plan with slots
#: so a caller can tell "declined" from "planned and found nothing".
PLAN_NOT_APPLICABLE = "not_applicable"


class MultiDocumentIntent(str, Enum):
    """What kind of set question P0-C v1 supports.

    Comparison, aggregation, and growth-rate intents are deliberately absent:
    they need a typed numeric fact layer that does not exist yet, and the
    planner declines them explicitly rather than enumerating and calling the
    result complete.
    """

    ENUMERATION = "enumeration"
    ENUMERATION_PLUS_EVENT = "enumeration_plus_event"



def _ids(value: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize an id collection: de-duplicated, sorted, no blanks."""

    if not value:
        return ()
    return tuple(sorted({str(item) for item in value if str(item).strip()}))


@dataclass(frozen=True)
class EvidenceSlot:
    """One set of documents the answer needs, and how completely it was found.

    ``expected_ids`` is what deterministic enumeration said the set contains;
    ``found_ids`` is what retrieval actually holds.  Completeness is the
    relationship between the two -- never a similarity score, and never a
    judgement made by a model.
    """

    slot_id: str
    slot_type: SlotType

    corp_code: str | None = None

    event_family: str | None = None
    doc_group: str | None = None
    doc_subtype: str | None = None
    member_role: str | None = None

    date_field: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    depends_on: tuple[str, ...] = ()

    expected_ids: tuple[str, ...] = ()
    found_ids: tuple[str, ...] = ()
    unresolved_ids: tuple[str, ...] = ()

    status: SlotStatus = SlotStatus.PENDING
    completeness_reason: str | None = None

    def __post_init__(self) -> None:
        if not str(self.slot_id).strip():
            raise ValueError("slot_id must not be empty")
        object.__setattr__(self, "slot_type", SlotType(self.slot_type))
        object.__setattr__(self, "status", SlotStatus(self.status))
        if self.date_field is not None and self.date_field not in DATE_FIELDS:
            raise ValueError(
                "date_field must be one of " + ", ".join(sorted(DATE_FIELDS))
            )
        # A half-open interval is the only shape enumeration accepts, so an
        # inverted one is a defect rather than an empty result.
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("date_from must not be after date_to")
        if self.slot_type is SlotType.EVENT_STATE and not self.depends_on:
            raise ValueError("an event_state slot must depend on an enumeration slot")
        object.__setattr__(self, "depends_on", _ids(self.depends_on))
        object.__setattr__(self, "expected_ids", _ids(self.expected_ids))
        object.__setattr__(self, "found_ids", _ids(self.found_ids))
        object.__setattr__(self, "unresolved_ids", _ids(self.unresolved_ids))

    # ------------------------------------------------------------- properties

    @property
    def missing_ids(self) -> tuple[str, ...]:
        """Expected members retrieval does not hold yet."""

        return tuple(sorted(set(self.expected_ids) - set(self.found_ids)))

    @property
    def expected_count(self) -> int:
        return len(self.expected_ids)

    @property
    def found_count(self) -> int:
        return len(self.found_ids)

    @property
    def is_definable(self) -> bool:
        """Whether this slot names a set deterministic enumeration can list.

        An enumeration slot needs a company, a family, and a bounded interval.
        Without all three the honest answer is ``NOT_APPLICABLE``: P0-C declines
        the question rather than guessing a set boundary.
        """

        if self.slot_type is SlotType.EVENT_STATE:
            return bool(self.depends_on)
        if not self.corp_code:
            return False
        if self.slot_type is SlotType.ENUMERATE_EVENTS and not self.event_family:
            return False
        if self.slot_type is SlotType.ENUMERATE_DOCUMENTS and not (
            self.doc_group or self.doc_subtype
        ):
            return False
        return bool(self.date_from and self.date_to)

    # ---------------------------------------------------------------- status

    def undefinable_reason(self) -> str | None:
        """Which part of the set definition is missing, if any."""

        if self.is_definable:
            return None
        if self.slot_type is SlotType.EVENT_STATE:
            return REASON_NO_FAMILY
        if not self.corp_code:
            return REASON_NO_CORP_CODE
        if self.slot_type is SlotType.ENUMERATE_EVENTS and not self.event_family:
            return REASON_NO_FAMILY
        if self.slot_type is SlotType.ENUMERATE_DOCUMENTS and not (
            self.doc_group or self.doc_subtype
        ):
            return REASON_NO_FAMILY
        return REASON_NO_DATE_RANGE

    def resolve_status(self) -> "EvidenceSlot":
        """Return this slot with ``status`` derived from its own ids.

        Precedence is deliberate.  An undefinable set is ``NOT_APPLICABLE``
        before anything else is considered; a missing member outranks an
        unresolved one, because more retrieval can still fix the first and
        cannot fix the second.

        An empty expected set is ``COMPLETE``, not ``INCOMPLETE``: "this company
        signed no contracts that year" is an answer, and the distinction between
        *nothing exists* and *nothing was found* is exactly what enumeration
        exists to make.
        """

        reason = self.undefinable_reason()
        if reason is not None:
            return replace(
                self, status=SlotStatus.NOT_APPLICABLE, completeness_reason=reason
            )
        if self.missing_ids:
            return replace(
                self, status=SlotStatus.INCOMPLETE, completeness_reason=REASON_MISSING_MEMBERS
            )
        if self.unresolved_ids:
            return replace(
                self,
                status=SlotStatus.UNRESOLVED,
                completeness_reason=REASON_UNRESOLVED_MEMBERS,
            )
        return replace(
            self,
            status=SlotStatus.COMPLETE,
            completeness_reason=(
                REASON_EMPTY_SET if not self.expected_ids else REASON_ALL_MEMBERS_FOUND
            ),
        )

    def resolve_status_with(
        self,
        *,
        expected_ids: Sequence[str] = (),
        found_ids: Sequence[str] = (),
        unresolved_ids: Sequence[str] = (),
    ) -> "EvidenceSlot":
        """Attach execution results, then derive status through the same rules.

        Execution never decides a status directly: it supplies the three id sets
        and :meth:`resolve_status` applies the one precedence the domain owns.
        """

        return replace(
            self,
            expected_ids=tuple(expected_ids),
            found_ids=tuple(found_ids),
            unresolved_ids=tuple(unresolved_ids),
        ).resolve_status()

    def to_dict(self) -> dict[str, Any]:
        """Trace shape: counts and statuses, never deliberation."""

        return {
            "slot_id": self.slot_id,
            "slot_type": self.slot_type.value,
            "corp_code": self.corp_code,
            "event_family": self.event_family,
            "doc_group": self.doc_group,
            "doc_subtype": self.doc_subtype,
            "member_role": self.member_role,
            "date_field": self.date_field,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "depends_on": list(self.depends_on),
            "expected_count": self.expected_count,
            "found_count": self.found_count,
            "missing_count": len(self.missing_ids),
            "unresolved_count": len(self.unresolved_ids),
            "status": self.status.value,
            "completeness_reason": self.completeness_reason,
        }



def _reject_cycles(slots: Sequence["EvidenceSlot"]) -> None:
    """A dependency cycle is unexecutable, so it is unconstructable.

    Kept in the domain rather than in the executor: a cyclic plan is not a plan
    that fails to run, it is not a plan at all.
    """

    dependencies = {slot.slot_id: set(slot.depends_on) for slot in slots}
    resolved: set[str] = set()
    while True:
        ready = {
            slot_id
            for slot_id, needs in dependencies.items()
            if slot_id not in resolved and needs <= resolved
        }
        if not ready:
            break
        resolved |= ready
    remaining = sorted(set(dependencies) - resolved)
    if remaining:
        raise ValueError(
            "slot dependency cycle involving: " + ", ".join(remaining)
        )


@dataclass(frozen=True)
class MultiDocumentPlan:
    """The slots one question needs, and why the planner stopped."""

    plan_type: str
    slots: tuple[EvidenceSlot, ...] = ()
    passes: int = 0
    stop_reason: str | None = None
    #: Diagnostic only -- how the lifecycle family was resolved. Never read by
    #: execution; present so a plan that engaged through a fallback is
    #: identifiable in a trace.
    family_resolution: str | None = None
    aggregate_field: str | None = None
    aggregate_ops: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.plan_type).strip():
            raise ValueError("plan_type must not be empty")
        slots = tuple(self.slots)
        slot_ids = [slot.slot_id for slot in slots]
        if len(slot_ids) != len(set(slot_ids)):
            raise ValueError("duplicate slot_id is not allowed")
        known = set(slot_ids)
        for slot in slots:
            unknown = sorted(set(slot.depends_on) - known)
            if unknown:
                raise ValueError(
                    f"slot {slot.slot_id} depends on unknown slot(s): "
                    + ", ".join(unknown)
                )
            if slot.slot_id in slot.depends_on:
                raise ValueError(f"slot {slot.slot_id} depends on itself")
        _reject_cycles(slots)
        if not isinstance(self.passes, int) or isinstance(self.passes, bool):
            raise ValueError("passes must be an integer")
        if self.passes < 0:
            raise ValueError("passes must not be negative")
        object.__setattr__(self, "slots", slots)

    @property
    def applied(self) -> bool:
        """Whether P0-C engaged at all.

        A declined plan carries ``stop_reason`` explaining which part of the set
        definition was missing.  ``complete`` is meaningless for it, so callers
        must check this first.
        """

        return self.plan_type != PLAN_NOT_APPLICABLE

    @property
    def complete(self) -> bool:
        """Every slot that applies is covered.

        ``NOT_APPLICABLE`` does not block completeness -- a slot P0-C declined
        was never part of the required set.
        """

        return all(
            slot.status in {SlotStatus.COMPLETE, SlotStatus.NOT_APPLICABLE}
            for slot in self.slots
        )

    def execution_order(self) -> tuple[EvidenceSlot, ...]:
        """Slots in dependency order, ties broken by ``slot_id``.

        Declaration order is never assumed to be dependency order; the same plan
        always yields the same sequence.
        """

        by_id = {slot.slot_id: slot for slot in self.slots}
        pending = {slot.slot_id: set(slot.depends_on) for slot in self.slots}
        ordered: list[EvidenceSlot] = []
        done: set[str] = set()
        while pending:
            ready = sorted(
                slot_id for slot_id, needs in pending.items() if needs <= done
            )
            if not ready:  # pragma: no cover - __post_init__ rejects cycles
                raise ValueError("slot dependency cycle")
            for slot_id in ready:
                ordered.append(by_id[slot_id])
                done.add(slot_id)
                pending.pop(slot_id)
        return tuple(ordered)

    def with_slots(self, slots: Sequence[EvidenceSlot], **changes: Any) -> "MultiDocumentPlan":
        """A copy carrying executed slots.  The input plan is never mutated."""

        return MultiDocumentPlan(
            plan_type=changes.get("plan_type", self.plan_type),
            slots=tuple(slots),
            passes=changes.get("passes", self.passes),
            stop_reason=changes.get("stop_reason", self.stop_reason),
            family_resolution=changes.get("family_resolution", self.family_resolution),
            aggregate_field=changes.get("aggregate_field", self.aggregate_field),
            aggregate_ops=changes.get("aggregate_ops", self.aggregate_ops),
        )

    def slot(self, slot_id: str) -> EvidenceSlot | None:
        for candidate in self.slots:
            if candidate.slot_id == slot_id:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "plan_type": self.plan_type,
            "applied": self.applied,
            "family_resolution": self.family_resolution,
            "passes": self.passes,
            "slots": [slot.to_dict() for slot in self.slots],
            "complete": self.complete,
            "stop_reason": self.stop_reason,
        }
        if self.aggregate_field:
            payload["aggregate_field"] = self.aggregate_field
        if self.aggregate_ops:
            payload["aggregate_ops"] = list(self.aggregate_ops)
        return payload


__all__ = [
    "PLAN_NOT_APPLICABLE",
    "REASON_MIXED_DATE_BASIS",
    "REASON_NO_SET_INTENT",
    "REASON_UNRESOLVED_DATE_BASIS",
    "REASON_UNSUPPORTED_CALCULATION",
    "REASON_UNSUPPORTED_TRUST_CONTRACT_BASIS",
    "MultiDocumentIntent",
    "DATE_FIELDS",
    "DATE_FIELD_EVENT_DATE",
    "DATE_FIELD_OPENED_AT",
    "DATE_FIELD_RCEPT_DT",
    "MEMBER_ROLE_CONTRACT",
    "REASON_ALL_MEMBERS_FOUND",
    "REASON_EMPTY_SET",
    "REASON_MISSING_MEMBERS",
    "REASON_NO_CORP_CODE",
    "REASON_NO_DATE_RANGE",
    "REASON_NO_FAMILY",
    "REASON_UNRESOLVED_MEMBERS",
    "EvidenceSlot",
    "MultiDocumentPlan",
    "SlotStatus",
    "SlotType",
]
