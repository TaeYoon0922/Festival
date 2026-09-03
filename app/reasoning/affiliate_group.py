"""How many affiliates a 기업집단 has, read from one persisted table row.

A periodic report states this as a 계열회사 현황 summary: one row per group,
with the listed and unlisted counts in their own columns and, usually, a stated
total beside them.  The question C086 asks -- "상장과 비상장을 합쳐 몇 개" -- is
that row read whole.

Nothing here re-reads the corpus or interprets prose.  A table chunk already
stores its own ``column_headers`` and the ``table_rows`` it was built from, so
the listed count, the unlisted count and the total come out of one physical row
that was persisted together.  That makes the composition structural rather than
a reconciliation between separately resolved numbers, exactly as
:mod:`app.reasoning.holding_acquisition` does for a holding detail row.

Two rules keep it honest.  The total is composed from the two counts, and when
the table also states a total the two must agree -- a disagreement is a table
this reader does not understand, not a number to pick from.  And a table whose
rows would give more than one such answer is ambiguous rather than guessed:
taking the first row would be inventing which group was meant.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


#: A count cell.  Thousands separators appear even here, and a blank or ``-``
#: is not a count.
_COUNT = re.compile(r"^\d[\d,]*$")

#: The column that holds the listed count, the one that holds the unlisted
#: count, and the one that holds the stated total.  Header identity decides
#: each, never position: the chunker joins a multi-row header down its column
#: with " / ", so ``계열회사의 수 / 상장`` and ``계열회사의 수 / 비상장`` stay
#: distinguishable whichever order the columns appear in.
_UNLISTED_HEADER = "비상장"
_LISTED_HEADER = "상장"
#: A total column names itself with its last header segment.  ``계`` is matched
#: whole rather than by containment -- ``계열회사의 수`` contains it and is not
#: a total.
_TOTAL_SEGMENTS = frozenset({"계", "합계", "총계", "소계"})

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class AffiliateCounts:
    """One group's affiliate counts, and the row they were read from."""

    listed: int
    unlisted: int
    total: int
    stated_total: int | None
    source_ref: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "listed": self.listed,
            "unlisted": self.unlisted,
            "total": self.total,
            "stated_total": self.stated_total,
            "source_ref": dict(self.source_ref),
        }


def affiliate_counts(chunk: Mapping[str, Any]) -> AffiliateCounts | None:
    """The affiliate counts this table row proves, or ``None`` for no row.

    ``None`` covers every shape this reader may not answer from: a table
    without both columns, a table stating a total that disagrees with its own
    parts, more than one countable row, and a row that cannot be cited.
    """

    headers = [_text(value) for value in chunk.get("column_headers") or []]
    rows = list(chunk.get("table_rows") or [])
    if not headers or not rows:
        return None

    unlisted_at = _column(headers, lambda header: _UNLISTED_HEADER in header)
    listed_at = _column(
        headers,
        lambda header: _LISTED_HEADER in header and _UNLISTED_HEADER not in header,
    )
    if unlisted_at is None or listed_at is None:
        return None
    total_at = _column(headers, _is_total_header)

    found: list[tuple[int, int, int, int | None]] = []
    for offset, row in enumerate(rows):
        cells = [_cell_text(cell) for cell in row or ()]
        listed = _count(cells, listed_at)
        unlisted = _count(cells, unlisted_at)
        if listed is None or unlisted is None:
            continue
        found.append(
            (offset, listed, unlisted, None if total_at is None else _count(cells, total_at))
        )
    # One group's counts is what this metric is.  Several rows that each look
    # like one is a table naming several groups, and nothing in the row itself
    # says which the question meant.
    if len(found) != 1:
        return None

    offset, listed, unlisted, stated_total = found[0]
    total = listed + unlisted
    if stated_total is not None and stated_total != total:
        # The table's own arithmetic disagrees with itself.  Reporting either
        # number would be choosing one without a reason to.
        return None

    source_ref = _row_source_ref(chunk, offset)
    if source_ref is None:
        return None
    return AffiliateCounts(
        listed=listed,
        unlisted=unlisted,
        total=total,
        stated_total=stated_total,
        source_ref=source_ref,
    )


def _column(headers: Sequence[str], predicate) -> int | None:
    matches = [index for index, header in enumerate(headers) if predicate(header)]
    # Ambiguity is not resolved by position; a table that does not name exactly
    # one such column cannot be read safely.
    return matches[0] if len(matches) == 1 else None


def _is_total_header(header: str) -> bool:
    segment = _WHITESPACE.sub("", header.rsplit("/", 1)[-1])
    return segment in _TOTAL_SEGMENTS


def _count(cells: Sequence[str], index: int) -> int | None:
    if index >= len(cells):
        return None
    text = _WHITESPACE.sub("", cells[index])
    if not _COUNT.match(text):
        return None
    return int(text.replace(",", ""))


def _cell_text(cell: Any) -> str:
    if isinstance(cell, Mapping):
        return str(cell.get("text") or "").strip()
    return str(cell or "").strip()


def _text(value: Any) -> str:
    return str(value or "")


def _row_source_ref(chunk: Mapping[str, Any], offset: int) -> dict[str, Any] | None:
    """The reference to this row, so the counts can be cited where they sit."""

    table_id = chunk.get("source_table_id") or chunk.get("table_id")
    row_start = chunk.get("row_start")
    if table_id is None or row_start is None:
        return None
    try:
        index = int(row_start) + offset
    except (TypeError, ValueError):
        return None
    return {"table_id": str(table_id), "row_start": index, "row_end": index}


__all__ = ["AffiliateCounts", "affiliate_counts"]
