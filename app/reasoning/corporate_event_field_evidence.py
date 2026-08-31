"""What the selected corporate-event filing says about one requested field.

STEP 11-C, producer A.  A supply-contract question whose formal 계약금액 cell
reads ``-`` is still full of real, citable numbers -- 최근매출액, 매출액대비
비율, a period, a counterparty.  Answering from those is how a blank contract
amount became ``answerable=True``.

So this reads exactly one cell: the formal contract-amount cell of the one
filing P0-B authorised, from the structured table rows the chunker already
persisted.  It does not search served text for a label, it does not read a
neighbouring cell, and it does not accept a narrative approximation in place of
the formal field -- an approximate amount in prose is not the exact field the
question asked for, whatever words surround it.

STEP 11-C.2 removed this module's own answer to "which filing?".  Company plus
receipt date is not an event, a group count is not correction finality, and
both were weaker than authority that already existed upstream.  Which filing
may be read is now :mod:`app.reasoning.corporate_event_authority`'s answer,
consumed whole.  What remains here is activation, one structured cell, and the
provenance that proves where it was read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from app.reasoning.corporate_event_authority import (
    CorporateSelectionIntent,
    selected_corporate_member,
)
from app.reasoning.corporate_event_graph import _AMOUNT_LABELS
from app.reasoning.correction_graph import _cells, _compact, _label_key
from app.reasoning.field_evidence import (
    DOMAIN_CORPORATE_EVENT,
    FieldEvidence,
    FieldReason,
    FieldStatus,
)


#: The one canonical corporate-event field STEP 11-C answers.
CONTRACT_AMOUNT = "contract_amount"

#: Generic wording that requests the contract amount itself.  Deliberately one
#: term: the fielded lane exists to make a specific formal cell authoritative,
#: and a question about a start date, a counterparty or a reason is not asking
#: about that cell.  Whitespace is compacted, so "계약 금액" reads the same.
_CONTRACT_AMOUNT_QUERY_TERM = "계약금액"

#: A cell that exists and states nothing.  This is the corpus's own blank.
_BLANK_CELLS = frozenset({"-", "--", "―", "—", ""})

#: Field-bound wording that upgrades a blank to a stated deferral.  A note only
#: counts when it names the field in the same table, so a document-level
#: 공시유보 remark can never speak for the contract amount on its own.
_DEFERRAL_MARKERS = ("유보", "추후공시", "공시예정", "추후기재", "종료후공시")

_DIGITS = re.compile(r"\d")


def requested_corporate_fields(question: Any) -> tuple[str, ...]:
    """The canonical corporate-event fields this question asks for.

    Read from the question alone.  Retrieved evidence and generated prose are
    not intent: a document that happens to contain 계약금액 does not make the
    asker have wanted it, and an answer that mentions it does not either.
    """

    compact = _compact(question or "")
    if _CONTRACT_AMOUNT_QUERY_TERM in compact:
        return (CONTRACT_AMOUNT,)
    return ()


def corporate_event_field_evidence(
    *,
    question: Any,
    plan: Any,
    execution: Any,
    evidence_items: Sequence[Any],
    multi_document: Any = None,
) -> tuple[FieldEvidence, ...]:
    """Field findings for the corporate event P0-B selected.

    Returns an empty tuple whenever this producer has no authority to speak: a
    question that requests no canonical field, a request P0-C is answering as a
    document set, or filings P0-B cannot single one out of.  Declining is not
    the same as finding nothing -- a decline leaves the request on the lane it
    was always answered by.
    """

    fields = requested_corporate_fields(question)
    if not fields:
        return ()
    if multi_document is not None:
        # P0-C is answering this as a set of filings.  A single-event contract
        # amount is not that answer, and STEP 11-C defines no multi-document
        # field contract, so this lane does not claim the request.
        return ()

    authority = selected_corporate_member(
        execution=execution,
        corp_code=getattr(plan, "corp_code", None),
        served_doc_ids=[
            str(getattr(item, "doc_id", "") or "") for item in evidence_items
        ],
        # Which served filings could answer this field at all.  A filing that
        # carries no formal amount cell is not competing for this selection, so
        # requiring authority for it would fail closed on ordinary neighbouring
        # evidence rather than on a real rival.
        candidate_doc_ids=_field_bearing_docs(evidence_items),
        selection_intent=_selection_intent(plan),
    )
    if authority.conflict:
        return tuple(_conflict(field, authority.reason) for field in fields)
    member = authority.member
    if member is None:
        return ()

    # One field, one reader.  A second canonical field would add its own entry
    # here rather than widen the reader this one uses.
    readers = {CONTRACT_AMOUNT: _contract_amount_evidence}
    records: list[FieldEvidence] = []
    for field in fields:
        records.extend(readers[field](member, evidence_items=evidence_items))
    return tuple(records)


def _selection_intent(plan: Any) -> CorporateSelectionIntent:
    """Reduce the plan to the one authority distinction this producer needs."""

    period = getattr(plan, "period", None)
    exact = bool(
        period is not None
        and getattr(period, "from_date", None)
        and getattr(period, "from_date", None) == getattr(period, "to_date", None)
        and getattr(period, "period_type", None) == "receipt_date"
    )
    date_basis = getattr(plan, "date_basis", None)
    date_basis = getattr(date_basis, "value", date_basis)

    evidence = getattr(plan, "evidence", None)
    evidence = evidence if isinstance(evidence, Mapping) else {}
    route_evidence = getattr(plan, "route_evidence", None)
    route_evidence = route_evidence if isinstance(route_evidence, Mapping) else {}
    correction_intent = str(evidence.get("correction_intent") or "")
    correction_policy = str(getattr(plan, "correction_policy", "") or "")
    explicit_latest_or_corrected = (
        correction_intent in {"latest", "history"}
        or correction_policy == "corrected_only"
        or (
            correction_policy == "latest_preferred"
            and bool(route_evidence.get("is_correction"))
        )
    )
    return CorporateSelectionIntent(
        exact_historical_receipt_date=(
            exact
            and date_basis == "receipt_date"
            and not explicit_latest_or_corrected
        )
    )


def _field_bearing_docs(evidence_items: Sequence[Any]) -> tuple[str, ...]:
    """Served filings whose own structured rows carry the formal amount field.

    Read with the same labelled-cell lookup the extraction uses, so "could this
    filing answer the question" and "what does this filing say" are the same
    test asked twice.  Anything else served -- a different disclosure type, a
    narrative chunk, a neighbouring filing -- carries no such cell and is not a
    candidate for this selection.
    """

    found: list[str] = []
    for item in evidence_items:
        doc_id = str(getattr(item, "doc_id", "") or "")
        if not doc_id or doc_id in found:
            continue
        chunk = _source_chunk(item)
        if chunk is not None and _amount_cells(chunk):
            found.append(doc_id)
    return tuple(found)


# ------------------------------------------------------------------ extraction


def _contract_amount_evidence(
    member: Any, *, evidence_items: Sequence[Any]
) -> list[FieldEvidence]:
    """Read the formal contract-amount cell of the authoritative filing."""

    records: list[FieldEvidence] = []
    found = False
    for item in evidence_items:
        doc_id = str(getattr(item, "doc_id", "") or "")
        authoritative = doc_id == member.authoritative_doc_id
        if not authoritative and doc_id not in member.superseded_doc_ids:
            continue
        chunk = _source_chunk(item)
        chunk_id = str(getattr(item, "chunk_id", "") or "")
        if chunk is None or not chunk_id:
            continue
        for cell in _amount_cells(chunk):
            found = found or authoritative
            records.append(_record(member, doc_id, chunk_id, cell, authoritative))
    if not found:
        # Either the authorised filing was never served, or it carries no formal
        # contract-amount field.  Both are an absence with nothing to cite, and
        # neither may be softened into a value.
        records.append(
            FieldEvidence(
                field=CONTRACT_AMOUNT,
                status=FieldStatus.MISSING,
                domain=DOMAIN_CORPORATE_EVENT,
                semantic_key=_semantic_key(member, member.authoritative_doc_id),
                corp_code=member.corp_code or None,
                doc_id=member.authoritative_doc_id,
                member_role=member.member_role,
            )
        )
    return records


@dataclass(frozen=True)
class _Cell:
    """One labelled contract-amount cell, and the row it was read from."""

    value: str
    table_id: str | None
    row_start: int | None
    row_end: int | None
    deferred: bool


def _amount_cells(chunk: Mapping[str, Any]) -> list[_Cell]:
    """Every formal contract-amount cell this chunk's own table rows carry.

    Rows come from ``table_rows``, which the chunker persisted alongside the
    chunk: the label and its value were adjacent cells of one physical row when
    the document was parsed, so pairing them here is a lookup, not a parse.

    Each cell is bound to the physical row it sits in rather than to the
    chunk's whole span, so two instances of one field inside one chunk stay two
    instances.
    """

    rows = list(chunk.get("table_rows") or [])
    if not rows:
        return []
    table_id = _text(chunk.get("source_table_id") or chunk.get("table_id")) or None
    base = _row_offset(chunk)
    deferred = _states_deferral(rows)
    cells: list[_Cell] = []
    for offset, row in enumerate(rows):
        texts = [value.strip() for value in _cells(row)]
        if len(texts) < 2:
            continue
        if _label_key(texts[-2]) not in _AMOUNT_LABELS:
            continue
        index = None if base is None else base + offset
        cells.append(
            _Cell(
                value=texts[-1],
                table_id=table_id,
                row_start=index,
                row_end=index,
                deferred=deferred,
            )
        )
    return cells


def _states_deferral(rows: Iterable[Any]) -> bool:
    """Whether these rows say the contract amount itself is disclosed later.

    The note has to name the field.  A filing may withhold anything or nothing,
    and a remark that does not say which field it covers proves neither.
    """

    for row in rows:
        for text in _cells(row):
            compact = _compact(text)
            if _CONTRACT_AMOUNT_QUERY_TERM not in compact:
                continue
            if any(marker in compact for marker in _DEFERRAL_MARKERS):
                return True
    return False


def _record(
    member: Any, doc_id: str, chunk_id: str, cell: _Cell, authoritative: bool
) -> FieldEvidence:
    value = cell.value.strip()
    if value in _BLANK_CELLS or not _DIGITS.search(value):
        status = FieldStatus.UNAVAILABLE
        reason = (
            FieldReason.WITHHELD_OR_DEFERRED if cell.deferred else FieldReason.NOT_STATED
        )
        stated: str | None = None
    else:
        status, reason, stated = FieldStatus.AVAILABLE, None, value
    return FieldEvidence(
        field=CONTRACT_AMOUNT,
        status=status,
        domain=DOMAIN_CORPORATE_EVENT,
        semantic_key=_semantic_key(member, doc_id),
        reason=reason,
        value=stated,
        corp_code=member.corp_code or None,
        doc_id=doc_id,
        member_role=member.member_role,
        chunk_id=chunk_id,
        table_id=cell.table_id,
        row_start=cell.row_start,
        row_end=cell.row_end,
        authoritative=authoritative,
    )


def _conflict(field: str, reason: str) -> FieldEvidence:
    return FieldEvidence(
        field=field,
        status=FieldStatus.CONFLICT,
        domain=DOMAIN_CORPORATE_EVENT,
        semantic_key=f"{DOMAIN_CORPORATE_EVENT}:{reason}",
    )


def _semantic_key(member: Any, doc_id: str) -> str:
    return ":".join((DOMAIN_CORPORATE_EVENT, member.event_id, doc_id))


def _source_chunk(item: Any) -> Mapping[str, Any] | None:
    provenance = getattr(item, "provenance", None)
    if not isinstance(provenance, Mapping):
        return None
    chunk = provenance.get("source_chunk")
    return chunk if isinstance(chunk, Mapping) else None


def _row_offset(chunk: Mapping[str, Any]) -> int | None:
    """Where this chunk's first persisted row sits in its source table."""

    try:
        start = chunk.get("row_start")
        return None if start is None else int(start)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    return "" if value is None else str(value)


__all__ = [
    "CONTRACT_AMOUNT",
    "corporate_event_field_evidence",
    "requested_corporate_fields",
]
