"""Project a periodic table down to the metric the question named.

The selector and the renderer share this so a 연결 매출액 question does not
dump 매출원가 or 주당이익 just because they sat in the same retrieved table.
Values are copied from the source table; nothing is computed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


_INCOME_STATEMENT_MARKERS = (
    "포괄손익계산서",
    "손익계산서",
    "연결손익",
    "연결포괄손익",
)
_TABLE_SEP = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$")


def project_periodic_metric_table(text: str, *, metric: str | None) -> str | None:
    """Return header plus exact metric rows, or None when the table has no such row."""

    label = _normalize(metric)
    if not label or not str(text or "").strip():
        return None
    rows = [line for line in str(text).splitlines() if line.strip().startswith("|")]
    if len(rows) < 2:
        return None
    header = rows[0]
    separator = rows[1] if _TABLE_SEP.fullmatch(rows[1].strip()) else None
    data = rows[2:] if separator is not None else rows[1:]
    matched = [row for row in data if _row_label(row) == label]
    if not matched:
        return None
    parts = [header]
    if separator is not None:
        parts.append(separator)
    parts.extend(matched)
    return "\n".join(parts)


def has_exact_metric_row(text: str, metric: str | None) -> bool:
    return project_periodic_metric_table(text, metric=metric) is not None


def is_income_statement_section(section_path: Sequence[Any] | None) -> bool:
    joined = " ".join(str(part) for part in (section_path or ()) if part)
    return any(marker in joined for marker in _INCOME_STATEMENT_MARKERS)


def source_chunk_view(source: Any) -> dict[str, Any]:
    """Rebuild a chunk-like mapping for basis classification."""

    provenance = {}
    period: Mapping[str, Any] = {}
    section_path: Sequence[Any] = ()
    fact_text = ""
    if hasattr(source, "provenance"):
        provenance = dict(getattr(source, "provenance") or {})
        period = dict(getattr(source, "reporting_period") or {})
        section_path = tuple(getattr(source, "section_path") or ())
        fact_text = str(getattr(source, "fact_text") or "")
    elif isinstance(source, Mapping):
        provenance = dict(source.get("provenance") or {})
        period = dict(source.get("reporting_period") or {})
        section_path = tuple(source.get("section_path") or ())
        fact_text = str(source.get("fact_text") or "")
    extra = dict(provenance.get("source_chunk") or {})
    scope = extra.get("statement_scope") or period.get("statement_scope")
    chunk = {
        **extra,
        "section_path": list(section_path),
        "content": fact_text or extra.get("content") or "",
    }
    if scope:
        chunk["statement_scope"] = scope
    return chunk


def _row_label(row: str) -> str:
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    return _normalize(cells[0] if cells else "")


def _normalize(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()
