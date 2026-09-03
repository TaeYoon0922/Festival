"""What the selected holding row says about the acquisition unit price.

STEP 11-C, producer B.  A holding filing that acquired shares and recorded no
취득단가 is not short of evidence -- it carries the reporter, the transaction
method, the change date, the quantity and the resulting position, all citable,
and some of its rows say in writing that the unit price was left unrecorded.
Counting those citations is how a missing unit price became ``answerable=True``
with no confirmed field at all.

This is a sibling of :mod:`app.reasoning.holding_acquisition`: it reads one more
column of the one row that module already proved an acquisition from.  Every
identity question -- which issuer, which reporter, which report, which row -- is
answered upstream by the frozen holding stack, and this producer consumes those
answers without widening or re-deciding any of them.  It never scans holding
chunks for an alias, and it never pairs a blank cell in one report with an
explanation printed in another.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from app.reasoning.field_evidence import (
    DOMAIN_HOLDING,
    FieldEvidence,
    FieldReason,
    FieldStatus,
)
from app.reasoning.holding_acquisition import (
    DETAIL_PROJECTION,
    _cell_text,
    classify_transaction_method,
)
from app.reasoning.holding_previous_report import previous_report_baseline_evidence


#: The one canonical holding field STEP 11-C answers.
ACQUISITION_UNIT_PRICE = "acquisition_unit_price"

#: Wording that asks for the unit price itself.  Bare 취득 is an acquisition
#: question and already has its own frozen semantics; only the price vocabulary
#: opens this lane.
_UNIT_PRICE_QUERY_TERM = "단가"

#: The acquisition semantics that must also be present.  A disposal unit price
#: is a different field and this producer does not answer for it.
_ACQUISITION_QUERY_TERMS = ("취득", "매수")

#: Disposal vocabulary, used only to tell a header that names one direction
#: from one that names both.  "취득/처분단가" contains an acquisition term and a
#: disposal term, so it distinguishes neither column it spans.
_DISPOSAL_HEADER_TERMS = ("처분", "매도", "양도")

#: The frozen field whose provenance names the row the acquisition was proved
#: from.  The unit price has to be read from that row and no other.
_ANCHOR_FIELDS = ("acquisition_date", "acquired_shares")

#: A cell that exists and states nothing.
_BLANK_CELLS = frozenset({"-", "--", "―", "—", ""})

#: Field-bound wording that upgrades a blank to a stated omission.  It has to
#: sit in the same row as the blank, so a neighbouring row's remark and another
#: report's explanation both fail to reach it.
_OMISSION_MARKERS = ("미기재", "기재생략", "기재하지", "기재되지", "미기입", "기재없")

_DIGITS = re.compile(r"\d")
_WHITESPACE = re.compile(r"\s+")


def requested_holding_fields(question: Any) -> tuple[str, ...]:
    """The canonical holding fields this question asks for.

    Both halves are required.  "취득단가" asks for this field; "취득일" and
    "취득 수량" ask for fields the frozen resolver already answers, and neither
    may be pulled into this lane by the word they share.
    """

    compact = _WHITESPACE.sub("", str(question or ""))
    if _UNIT_PRICE_QUERY_TERM not in compact:
        return ()
    if not any(term in compact for term in _ACQUISITION_QUERY_TERMS):
        return ()
    return (ACQUISITION_UNIT_PRICE,)


def holding_field_evidence(
    *,
    question: Any,
    resolution: Any,
    evidence_items: Sequence[Any],
) -> tuple[FieldEvidence, ...]:
    """Every holding field finding for this question.

    Two producers live behind this one entry point, and they answer about
    different fields of different rows: the acquisition unit price of the row an
    acquisition was proved from, and the previous state of the report a
    ``직전 보고`` question is relative to.  Neither can claim the other's field,
    so their outputs are concatenated rather than ranked.
    """

    return (
        *_acquisition_unit_price_evidence(
            question=question, resolution=resolution, evidence_items=evidence_items
        ),
        *previous_report_baseline_evidence(
            question=question, resolution=resolution, evidence_items=evidence_items
        ),
    )


def _acquisition_unit_price_evidence(
    *,
    question: Any,
    resolution: Any,
    evidence_items: Sequence[Any],
) -> tuple[FieldEvidence, ...]:
    """Field findings for the holding row the frozen stack already selected.

    Returns an empty tuple whenever the frozen stack did not single out one
    acquisition row: no matching event, more than one, a temporal ambiguity, a
    reporter that does not match, or a group that never proved an acquisition.
    Declining leaves the request exactly where it was answered before.
    """

    fields = requested_holding_fields(question)
    if not fields:
        return ()
    event = _selected_event(resolution)
    if event is None:
        return ()

    anchor = _anchor(event)
    if anchor is None:
        # The row that proved the acquisition is not identified, so there is no
        # row to read a unit price from.  This is a decline, not a finding.
        return ()

    item = _served_item(anchor.chunk_id, evidence_items)
    if item is None:
        return (
            FieldEvidence(
                field=ACQUISITION_UNIT_PRICE,
                status=FieldStatus.MISSING,
                domain=DOMAIN_HOLDING,
                semantic_key=_semantic_key(event, anchor),
                corp_code=_text(getattr(event, "corp_code", None)) or None,
                doc_id=_text(getattr(event, "doc_id", None)) or None,
                member_role=_text(getattr(event, "reporter", None)) or None,
            ),
        )
    return (_classify(event, anchor, item),)


class _Anchor:
    """Where the frozen resolver read this event's acquisition from."""

    __slots__ = ("chunk_id", "doc_id", "table_id", "row_start", "row_end")

    def __init__(
        self,
        *,
        chunk_id: str,
        doc_id: str | None,
        table_id: str | None,
        row_start: int | None,
        row_end: int | None,
    ) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.table_id = table_id
        self.row_start = row_start
        self.row_end = row_end


def _is_acquisition_transaction(event: Any) -> bool:
    """Whether this event is a proven acquisition, on the frozen classifier.

    ``transaction_method`` is populated only from a detail row that proved its
    own acquisition, so a position snapshot, a disposal and an unexplained
    increase all arrive here carrying nothing.  The frozen classifier is asked
    anyway rather than trusting that construction, so the one thing this
    producer may answer for stays named by the same vocabulary everywhere.
    """

    method = _text(getattr(event, "transaction_method", None))
    return bool(method) and classify_transaction_method(method) == "acquisition"


def _selected_event(resolution: Any) -> Any | None:
    """The one acquisition this question is about, or nothing.

    ``matches_query`` is the frozen resolver's verdict on which events the
    question reaches, and it is not re-derived here.  What this narrows is
    *which of those* can answer for an acquisition unit price: a filing states
    a position on the day it was filed and the transactions that moved it on
    the days they happened, so one filing legitimately yields a snapshot dated
    later than the acquisition it reports.  The resolver is right to call those
    two events, and right to report the temporal ambiguity between them -- but a
    snapshot proves no acquisition and so was never a rival for this field.
    Ambiguity is therefore measured among the acquisitions themselves.

    Two acquisitions still decline.  So does an event whose method is stated but
    does not name an acquisition: the classifier reports only that it is not
    one, never whether it is a disposal that could not compete or something
    unread that might, and guessing between those is not this producer's to do.
    """

    events = tuple(getattr(resolution, "events", ()) or ())
    if not events:
        return None
    matching = [
        event for event in events
        if getattr(event, "matches_query", None) is True
    ]
    if any(
        _text(getattr(event, "transaction_method", None))
        and not _is_acquisition_transaction(event)
        for event in matching
    ):
        # Unreachable from the frozen resolver, which populates a method only
        # for a proven acquisition.  Kept because the safe reading of a method
        # this producer cannot classify is to answer nothing.
        return None
    acquisitions = [event for event in matching if _is_acquisition_transaction(event)]
    if len(acquisitions) != 1:
        return None
    event = acquisitions[0]
    if bool(getattr(event, "field_conflict", False)):
        return None
    return event


def _anchor(event: Any) -> _Anchor | None:
    provenance = getattr(event, "field_provenance", None)
    if not isinstance(provenance, Mapping):
        return None
    for field in _ANCHOR_FIELDS:
        entry = provenance.get(field)
        if entry is None or bool(getattr(entry, "field_conflict", False)):
            continue
        sources = tuple(getattr(entry, "sources", ()) or ())
        if len(sources) != 1:
            # Two rows proved the same fact.  Which row carries the unit price
            # is then a choice, and this producer does not make choices.
            continue
        source = sources[0]
        if not bool(getattr(source, "direct_field_ref", False)):
            continue
        chunk_id = _text(getattr(source, "chunk_id", None))
        if not chunk_id:
            continue
        refs = tuple(getattr(source, "source_refs", ()) or ())
        if len(refs) != 1 or not isinstance(refs[0], Mapping):
            continue
        ref = refs[0]
        return _Anchor(
            chunk_id=chunk_id,
            doc_id=_text(getattr(source, "doc_id", None)) or None,
            table_id=_text(ref.get("table_id")) or None,
            row_start=_int(ref.get("row_start")),
            row_end=_int(ref.get("row_end", ref.get("row_start"))),
        )
    return None


def _served_item(chunk_id: str, evidence_items: Sequence[Any]) -> Any | None:
    for item in evidence_items:
        if _text(getattr(item, "chunk_id", None)) == chunk_id:
            return item
    return None


def _classify(event: Any, anchor: _Anchor, item: Any) -> FieldEvidence:
    """Read the unit-price cell of the anchored row and say what it holds."""

    chunk = _source_chunk(item)
    cells, headers, header_rows = _anchored_row(chunk, anchor)
    column = _unit_price_column(headers, header_rows)
    common = {
        "field": ACQUISITION_UNIT_PRICE,
        "domain": DOMAIN_HOLDING,
        "semantic_key": _semantic_key(event, anchor),
        "corp_code": _text(getattr(event, "corp_code", None)) or None,
        "doc_id": anchor.doc_id or _text(getattr(event, "doc_id", None)) or None,
        "member_role": _text(getattr(event, "reporter", None)) or None,
        "table_id": anchor.table_id,
        "row_start": anchor.row_start,
        "row_end": anchor.row_end,
    }
    if cells is None or column is None or column >= len(cells):
        # The selected row does not carry this field.  Nothing states it and
        # nothing denies it, so there is no negative source to cite either.
        return FieldEvidence(status=FieldStatus.MISSING, **common)

    value = _cell_text(cells[column])
    if value not in _BLANK_CELLS and _DIGITS.search(value):
        return FieldEvidence(
            status=FieldStatus.AVAILABLE,
            value=value,
            chunk_id=anchor.chunk_id,
            **common,
        )
    return FieldEvidence(
        status=FieldStatus.UNAVAILABLE,
        reason=(
            FieldReason.OMITTED
            if _states_omission(cells, column)
            else FieldReason.NOT_STATED
        ),
        chunk_id=anchor.chunk_id,
        **common,
    )


def _anchored_row(
    chunk: Mapping[str, Any] | None, anchor: _Anchor
) -> tuple[Sequence[Any] | None, list[str], list[Any]]:
    """The one physical row the acquisition was proved from, and its headers.

    A projection holds exactly one row and records which row that was.  Both
    have to agree with the anchor, so a chunk that was re-projected or that
    carries a different row cannot stand in for the selected one.
    """

    if chunk is None:
        return None, [], []
    if _text(chunk.get("projection_type")) != DETAIL_PROJECTION:
        return None, [], []
    rows = list(chunk.get("table_rows") or [])
    if len(rows) != 1:
        return None, [], []
    table_id = _text(chunk.get("source_table_id") or chunk.get("table_id")) or None
    if anchor.table_id is not None and table_id != anchor.table_id:
        return None, [], []
    if anchor.row_start is not None and _int(chunk.get("row_start")) != anchor.row_start:
        return None, [], []
    headers = [str(value or "") for value in chunk.get("column_headers") or []]
    return list(rows[0] or []), headers, list(chunk.get("header_rows") or [])


def _unit_price_column(
    headers: Sequence[str], header_rows: Sequence[Any] = ()
) -> int | None:
    """Which column holds the acquisition unit price, by header semantics.

    Header identity decides this, never position.  The chunker joins a
    multi-row header down its column with " / ", so a table that separates the
    two directions gives one column an acquisition path and the other a
    disposal path, and those are distinguishable no matter which order the
    columns appear in.

    A single unit-price column is read as the acquisition's own only because
    the frozen resolver already proved this row is an acquisition -- one
    undivided 단가 column on an acquisition row is that acquisition's price.
    Several columns that nothing distinguishes are genuinely ambiguous, and
    picking the leftmost would be inventing an answer the table does not give.
    """

    priced = [
        (index, _WHITESPACE.sub("", header))
        for index, header in enumerate(headers)
        if _UNIT_PRICE_QUERY_TERM in _WHITESPACE.sub("", header)
    ]
    if not priced:
        return None

    # A header path that names the acquisition side and not the disposal side.
    # "취득/처분단가" names both and so distinguishes nothing.
    acquisition = [
        index
        for index, header in priced
        if any(term in header for term in _ACQUISITION_QUERY_TERMS)
        and not any(term in header for term in _DISPOSAL_HEADER_TERMS)
    ]
    if len(acquisition) == 1:
        return acquisition[0]
    if acquisition:
        return None

    # No column names a direction.  One undivided price column on a row the
    # frozen layer proved an acquisition is that acquisition's price.
    if len(priced) == 1:
        return priced[0][0]

    # Several columns carry the same price header.  That is one merged header
    # cell repeated across the columns it spans -- one logical field, whose
    # value the row states once -- or it is several separate fields the table
    # failed to distinguish.  Only the physical header cell can tell those
    # apart, so a span this cannot verify is ambiguous and fails closed.
    indexes = [index for index, _header in priced]
    if len({header for _index, header in priced}) != 1:
        return None
    return indexes[0] if _is_one_merged_cell(header_rows, indexes) else None


def _is_one_merged_cell(header_rows: Sequence[Any], indexes: Sequence[int]) -> bool:
    """Whether one physical header cell spans exactly these columns.

    ``header_rows`` keeps the header exactly as the document wrote it, spans
    included, so this is a reading of the table's own structure rather than an
    inference from repeated text.
    """

    wanted = (indexes[0], indexes[-1] + 1)
    for row in header_rows or ():
        column = 0
        for cell in row or ():
            span = _colspan(cell)
            if (column, column + span) == wanted and span > 1:
                return True
            column += span
    return False


def _colspan(cell: Any) -> int:
    value = cell.get("colspan") if isinstance(cell, Mapping) else None
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _states_omission(cells: Sequence[Any], column: int) -> bool:
    """Whether this row itself records that its unit price was not written.

    Scoped to the row the blank sits in.  A remark elsewhere in the table, or in
    another report entirely, explains some other blank.
    """

    for index, cell in enumerate(cells):
        if index == column:
            continue
        compact = _WHITESPACE.sub("", _cell_text(cell))
        if not compact or _UNIT_PRICE_QUERY_TERM not in compact:
            continue
        if any(marker in compact for marker in _OMISSION_MARKERS):
            return True
    return False


def _semantic_key(event: Any, anchor: _Anchor) -> str:
    return ":".join(
        (
            DOMAIN_HOLDING,
            _text(getattr(event, "corp_code", None)),
            _text(getattr(event, "reporter", None)),
            anchor.doc_id or _text(getattr(event, "doc_id", None)),
            anchor.table_id or "",
            "" if anchor.row_start is None else str(anchor.row_start),
        )
    )


def _source_chunk(item: Any) -> Mapping[str, Any] | None:
    provenance = getattr(item, "provenance", None)
    if not isinstance(provenance, Mapping):
        return None
    chunk = provenance.get("source_chunk")
    return chunk if isinstance(chunk, Mapping) else None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ACQUISITION_UNIT_PRICE",
    "holding_field_evidence",
    "requested_holding_fields",
]
