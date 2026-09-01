"""One normalized statement about one requested field, and its provenance.

STEP 11-C.  The guard used to decide whether a requested field was supported by
reading the generated answer and counting citations.  Citable evidence is not a
supported field value, and the two failure shapes this closes both prove it: a
supply contract whose formal 계약금액 cell reads ``-`` is fully citable and
states no amount, and a holding report that records its 취득단가 as omitted is
citable for exactly that omission.  The guard could not tell either apart from a
real value, because it never saw a field -- it saw prose and a count.

This module is the vocabulary the domain producers speak in.  A producer owns
the semantic identity work its own domain requires: which corporate event, which
correction member, which holding report and which row.  It emits
:class:`FieldEvidence`, which carries the decision it reached plus the
provenance proving where that decision was read.  The guard consumes those
records and decides answerability from them.

Nothing here parses a document, selects an event, resolves a correction, or
knows what a disclosure looks like.  Those questions belong to the producers,
and a record that reaches this module has already had them answered.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence


class FieldStatus(str, Enum):
    """What the authoritative source says about one requested field."""

    #: The source states a value for this field.
    AVAILABLE = "available"
    #: The field exists in the source and carries no value.
    UNAVAILABLE = "unavailable"
    #: The authoritative source does not carry this field at all, or was never
    #: served.  There is nothing to cite either way.
    MISSING = "missing"
    #: More than one authoritative statement, and nothing chooses between them.
    CONFLICT = "conflict"


class FieldReason(str, Enum):
    """Why an ``UNAVAILABLE`` field carries no value.

    Only a field-bound source may strengthen ``NOT_STATED``.  A document-level
    remark about the filing as a whole says nothing about which field it covers,
    so it never reaches this enum.
    """

    #: The field is blank and the source says nothing further about it.
    NOT_STATED = "not_stated"
    #: The source states that this field was left unrecorded.
    OMITTED = "omitted"
    #: The source states that this field is withheld now or disclosed later.
    WITHHELD_OR_DEFERRED = "withheld_or_deferred"


DOMAIN_CORPORATE_EVENT = "corporate_event"
DOMAIN_HOLDING = "holding"

#: Statuses that assert what a source says, and therefore need one to say it.
_GROUNDED_STATUSES = frozenset({FieldStatus.AVAILABLE, FieldStatus.UNAVAILABLE})


@dataclass(frozen=True)
class FieldEvidence:
    """One producer's finding about one field instance.

    ``semantic_key`` is the producer's own identity for the thing the field
    belongs to -- a corporate event member, a holding report row.  It is what
    keeps two same-named fields apart, and it is deliberately not ``doc_id``:
    one filing can carry two different instances of the same field, and merging
    them because they share a document is exactly the mistake this closes.

    A record without ``chunk_id`` asserts an absence.  It can never assert a
    value, and it can never be cited.
    """

    field: str
    status: FieldStatus
    domain: str
    semantic_key: str
    reason: FieldReason | None = None
    value: str | None = None
    corp_code: str | None = None
    doc_id: str | None = None
    member_role: str | None = None
    chunk_id: str | None = None
    table_id: str | None = None
    row_start: int | None = None
    row_end: int | None = None
    #: False for a source the producer identified but ruled out -- a superseded
    #: correction member, say.  Kept for the trace; never decides anything.
    authoritative: bool = True

    def __post_init__(self) -> None:
        if self.chunk_id is None and self.status in _GROUNDED_STATUSES:
            # A stated value and a stated blank are both readings of a source.
            # Only an absence -- nothing carries this field, or the filing that
            # would was never served -- may be asserted without one.
            raise ValueError(f"a {self.status.value} field must name its source chunk")
        if self.status is not FieldStatus.UNAVAILABLE and self.reason is not None:
            raise ValueError("only an unavailable field carries a reason")

    @property
    def instance_key(self) -> tuple[Any, ...]:
        """This exact field instance, down to the row it was read from."""

        return (
            self.semantic_key,
            self.field,
            self.table_id or "",
            -1 if self.row_start is None else int(self.row_start),
            -1 if self.row_end is None else int(self.row_end),
        )

    @property
    def outcome(self) -> tuple[Any, ...]:
        """What this record asserts, ignoring where it was read."""

        return (
            self.status.value,
            self.reason.value if self.reason is not None else None,
            self.value,
        )

    @property
    def citable(self) -> bool:
        return self.chunk_id is not None

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "status": self.status.value,
            "reason": self.reason.value if self.reason is not None else None,
            "domain": self.domain,
            "doc_id": self.doc_id,
            "member_role": self.member_role,
            "chunk_id": self.chunk_id,
            "table_id": self.table_id,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "authoritative": self.authoritative,
        }


def served_chunk_ids(execution: Any) -> frozenset[str]:
    """The chunk ids of the evidence the answer was actually served from."""

    return frozenset(
        chunk_id
        for result in (getattr(execution, "results", ()) or ())
        if (chunk_id := str(getattr(result, "chunk_id", "") or ""))
    )


def accepted_field_evidence(
    records: Iterable[FieldEvidence], served: Iterable[str]
) -> tuple[FieldEvidence, ...]:
    """The records allowed to affect the final answer.

    Membership is tested against the served set unconditionally.  Guarding that
    test with ``if served:`` would accept every candidate in exactly the one
    case where nothing can be grounded at all, so an empty served set accepts
    nothing here -- including an absence, which has no citation to lose but also
    no served evidence to have been read from.
    """

    served_ids = frozenset(str(value) for value in served if str(value))
    if not served_ids:
        return ()
    accepted: list[FieldEvidence] = []
    for record in records:
        if not record.authoritative:
            continue
        if record.chunk_id is not None and str(record.chunk_id) not in served_ids:
            # Read from something the caller was never shown.  A value it
            # supports would be ungrounded, and a refusal it supports would
            # cite a chunk that is not in the response.
            continue
        accepted.append(record)
    return tuple(accepted)


def resolve_field_states(
    records: Iterable[FieldEvidence], *, served: Iterable[str]
) -> dict[str, FieldEvidence]:
    """One state per field, or ``CONFLICT`` when the accepted records disagree.

    Records are keyed on the producer's semantic identity, never on the alias
    alone and never on the document: two instances of one field name are two
    findings unless the producer said they are the same instance.
    """

    states: dict[str, FieldEvidence] = {}
    by_field: dict[str, list[FieldEvidence]] = {}
    for record in accepted_field_evidence(records, served):
        by_field.setdefault(record.field, []).append(record)

    for field, found in by_field.items():
        ordered = sorted(found, key=lambda record: record.instance_key)
        if len({record.semantic_key for record in ordered}) > 1:
            # Two identities answered for one field and nothing here may choose
            # between them; choosing would be the guard selecting an event.
            states[field] = _conflict(field, ordered)
            continue
        if len({record.outcome for record in ordered}) > 1:
            states[field] = _conflict(field, ordered)
            continue
        states[field] = ordered[0]
    return states


def _conflict(field: str, found: Sequence[FieldEvidence]) -> FieldEvidence:
    first = found[0]
    return FieldEvidence(
        field=field,
        status=FieldStatus.CONFLICT,
        domain=first.domain,
        semantic_key=first.semantic_key,
        corp_code=first.corp_code,
    )


def field_evidence_trace(states: Mapping[str, FieldEvidence]) -> list[dict[str, Any]]:
    """The normalized diagnostics the public trace may carry."""

    return [state.to_public_dict() for _field, state in sorted(states.items())]


__all__ = [
    "DOMAIN_CORPORATE_EVENT",
    "DOMAIN_HOLDING",
    "FieldEvidence",
    "FieldReason",
    "FieldStatus",
    "accepted_field_evidence",
    "field_evidence_trace",
    "resolve_field_states",
    "served_chunk_ids",
]
