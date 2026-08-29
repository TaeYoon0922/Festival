"""Acquisition semantics for a single holding detail row.

A holding disclosure records *why* each position changed in a 취득/처분방법 cell
sitting in the same physical row as the change date and the signed quantity.
That cell is the only thing separating an acquisition from a disposal, and
without it a question about an acquisition date cannot be answered safely:
``reference_date`` carries the transaction date on a detail row but the report or
base date on a report projection, so aliasing the acquisition date onto it would
answer with the filing date instead of the date the shares were acquired.

Nothing here re-reads the corpus.  A projection chunk already stores its own
``column_headers`` and the single ``table_rows`` entry it was built from, so the
method, the date and the quantity are recovered from one row that was persisted
together -- which makes the same-row guarantee structural rather than a
reconciliation between separately resolved fields.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

#: Only detail rows carry a transaction method.  A holding_report summarises a
#: position and can never prove how it changed.
DETAIL_PROJECTION = "holding_detail_row"

#: "매수선택권"/"매수청구권" are *rights*, not purchases: granting or cancelling a
#: stock option moves no shares, so the 매수 inside them must not read as one.
_PURCHASE_RIGHT = re.compile(r"매수[가-힣]*권")

#: Words that name an acquisition outright.  A method must say what happened --
#: "기타(+)" and "신규보고(+)" are increases that explain nothing, and neither may
#: stand in for proof.
_ACQUISITION_TOKENS = ("취득", "매수")

#: If a method mentions any of these it is not a clean acquisition, whatever
#: else it says.
_DISPOSAL_TOKENS = ("처분", "매도", "양도", "반환", "해소", "퇴임", "취소", "상실")

#: Disclosures mark the direction of every method with a trailing sign.
_POSITIVE_MARKER = re.compile(r"\(\s*\+\s*\)\s*$")

_NUMERIC = re.compile(r"^-?[\d,]+$")
_DATE = re.compile(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})")


def _cell_text(cell: Any) -> str:
    if isinstance(cell, Mapping):
        return str(cell.get("text") or "").strip()
    return str(cell or "").strip()


def _column(headers: Sequence[Any], predicate) -> int | None:
    matches = [index for index, header in enumerate(headers)
               if predicate(str(header or ""))]
    # Ambiguity is not resolved by position; if the table does not name exactly
    # one such column the row cannot be read safely.
    return matches[0] if len(matches) == 1 else None


def _method_column(headers):
    # "취득/처분단가" also contains 취득; requiring 방법 keeps the unit price out.
    return _column(headers, lambda h: "취득" in h and "방법" in h)


def _date_column(headers):
    return _column(headers, lambda h: "변동일" in h)


def _change_column(headers):
    return _column(
        headers,
        lambda h: "증감" in h and "비율" not in h and "단가" not in h,
    )


def classify_transaction_method(method: str | None) -> str | None:
    """``"acquisition"`` only when the method itself says so.

    Deterministic and vocabulary-driven rather than a list of known strings: the
    method must name an acquisition, must not also name a disposal, and must
    carry the disclosure's own positive direction marker.
    """

    text = str(method or "").strip()
    if not text or text == "-":
        return None
    if not _POSITIVE_MARKER.search(text):
        return "other"
    without_rights = _PURCHASE_RIGHT.sub("", text)
    if any(token in without_rights for token in _DISPOSAL_TOKENS):
        return "other"
    if any(token in without_rights for token in _ACQUISITION_TOKENS):
        return "acquisition"
    return "other"


def parse_quantity(value: str | None) -> int | None:
    text = str(value or "").strip().replace(" ", "")
    if not text or not _NUMERIC.match(text):
        return None
    try:
        return int(text.replace(",", ""))
    except ValueError:  # pragma: no cover - guarded by the pattern
        return None


def normalize_date(value: str | None) -> str | None:
    match = _DATE.search(str(value or ""))
    if not match:
        return None
    year, month, day = match.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def acquisition_facts(chunk: Mapping[str, Any]) -> dict[str, Any] | None:
    """The acquisition a detail row proves, or ``None`` when it proves none.

    Every returned value comes from the one row this projection was built from,
    so the method that proves the acquisition, the date and the quantity cannot
    be drawn from different rows even when a table repeats a reporter.
    """

    if str(chunk.get("projection_type") or "") != DETAIL_PROJECTION:
        return None
    headers = list(chunk.get("column_headers") or [])
    rows = list(chunk.get("table_rows") or [])
    # One projection is one physical row; anything else is not a same-row proof.
    if not headers or len(rows) != 1:
        return None
    cells = list(rows[0] or [])

    method_at = _method_column(headers)
    change_at = _change_column(headers)
    if method_at is None or change_at is None:
        return None
    if method_at >= len(cells) or change_at >= len(cells):
        return None

    method = _cell_text(cells[method_at])
    if classify_transaction_method(method) != "acquisition":
        return None

    quantity = parse_quantity(_cell_text(cells[change_at]))
    # The method and the arithmetic must agree; neither alone is proof, and a
    # disposal's magnitude must never be rescued by dropping its sign.
    if quantity is None or quantity <= 0:
        return None

    date_at = _date_column(headers)
    date = (normalize_date(_cell_text(cells[date_at]))
            if date_at is not None and date_at < len(cells) else None)

    source_ref = _row_source_ref(chunk)
    if source_ref is None:
        return None

    return {
        "transaction_method": method,
        "acquisition_date": date,
        "acquired_shares": _cell_text(cells[change_at]),
        "acquired_shares_value": quantity,
        "source_ref": source_ref,
    }


def _row_source_ref(chunk: Mapping[str, Any]) -> dict[str, Any] | None:
    """The reference to this row, so a fact can be cited where it was read."""

    table_id = chunk.get("source_table_id") or chunk.get("table_id")
    row_start = chunk.get("row_start")
    row_end = chunk.get("row_end", row_start)
    if table_id is None or row_start is None:
        return None
    return {"table_id": str(table_id), "row_start": int(row_start),
            "row_end": int(row_end if row_end is not None else row_start)}
