"""Rank business-segment rows in periodic tables.

Values are read from the source table; only the argmax over existing numeric
cells is derived.  Used when the question asks which segment contributed most.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_TABLE_SEP = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_NUMERIC = re.compile(r"^-?\(?[\d,]+\)?$")

_METRIC_HEADER_ALIASES: dict[str, tuple[str, ...]] = {
    "영업이익": ("영업이익", "영업손익", "segment영업이익"),
    "매출액": ("매출액", "매출", "영업수익", "수익"),
    "당기순이익": ("당기순이익", "당기순손익", "순이익"),
}


def segment_ranking_requested(request: Mapping[str, Any] | None) -> bool:
    if not request:
        return False
    if request.get("segment_ranking"):
        return True
    evidence = request.get("evidence")
    return isinstance(evidence, Mapping) and bool(evidence.get("segment_ranking"))


def project_segment_ranking_table(
    text: str,
    *,
    metric: str | None,
) -> str | None:
    """Return a compact table with the largest segment row, or None."""

    metric_key = str(metric or "").strip()
    if not metric_key or not str(text or "").strip():
        return None
    headers = _METRIC_HEADER_ALIASES.get(metric_key, (metric_key,))
    rows = [line for line in str(text).splitlines() if line.strip().startswith("|")]
    if len(rows) < 2:
        return None
    header_cells = _cells(rows[0])
    if len(header_cells) < 2:
        return None
    separator = rows[1] if _TABLE_SEP.fullmatch(rows[1].strip()) else None
    data_rows = rows[2:] if separator is not None else rows[1:]
    metric_col = _metric_column_index(header_cells, headers)
    if metric_col is None:
        return None
    best: tuple[float, str, list[str]] | None = None
    for row in data_rows:
        cells = _cells(row)
        if len(cells) <= metric_col:
            continue
        label = cells[0].strip()
        if not label or _is_total_row(label):
            continue
        value = _parse_amount(cells[metric_col])
        if value is None:
            continue
        if best is None or value > best[0]:
            best = (value, label, cells)
    if best is None:
        return None
    _, label, cells = best
    parts = [rows[0]]
    if separator is not None:
        parts.append(_separator_for(rows[0]))
    parts.append("| " + " | ".join(cells) + " |")
    parts.append(f"(부문별 {metric_key} 최대: {label})")
    return "\n".join(parts)


def _metric_column_index(header_cells: Sequence[str], aliases: Sequence[str]) -> int | None:
    normalized_aliases = {_normalize(alias) for alias in aliases}
    for index, cell in enumerate(header_cells):
        if _normalize(cell) in normalized_aliases:
            return index
    for index, cell in enumerate(header_cells):
        compact = _normalize(cell)
        if any(alias in compact for alias in normalized_aliases):
            return index
    return None


def _is_total_row(label: str) -> bool:
    compact = _normalize(label)
    return any(term in compact for term in ("합계", "총계", "total", "소계", "계"))


def _parse_amount(raw: str) -> float | None:
    text = str(raw or "").strip().replace(",", "")
    if not text or text in {"-", "—", "–"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    if not _NUMERIC.fullmatch(text.replace(".", "", 1) if "." in text else text):
        # Allow plain integers only — conservative.
        if not re.fullmatch(r"-?\d+", text):
            return None
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def _separator_for(header_row: str) -> str:
    count = max(1, len(_cells(header_row)))
    return "| " + " | ".join(["---"] * count) + " |"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text or "").casefold()
