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
_METRIC_LABEL_ALIASES = {
    "매출액": {
        "매출액",
        "매출",
        "영업수익",
    },
    "당기순이익": {
        "당기순이익",
        "당기순이익손실",
        "당기순손익",
        "분기순이익",
        "분기순이익손실",
        "분기순손익",
        "연결분기순이익",
        "연결분기순이익손실",
        "연결분기순손익",
        "연결당기순이익",
        "연결당기순이익손실",
        "연결당기순손익",
    },
    "자산총계": {
        "자산총계",
        "자산총액",
        "자산계",
        "총자산",
    },
    "부채총계": {
        "부채총계",
        "부채총액",
        "부채계",
        "총부채",
    },
    "자본총계": {
        "자본총계",
        "자본총액",
        "자본계",
        "총자본",
    },
}


def project_periodic_metric_table(
    text: str,
    *,
    metric: str | None,
    period: Mapping[str, Any] | None = None,
    comparison: Mapping[str, Any] | None = None,
    raw_query: str | None = None,
) -> str | None:
    """Return header plus exact metric rows, or None when the table has no such row."""

    labels = _metric_labels(metric)
    if not labels or not str(text or "").strip():
        return None
    rows = [line for line in str(text).splitlines() if line.strip().startswith("|")]
    if len(rows) < 2:
        return None
    header = rows[0]
    separator = rows[1] if _TABLE_SEP.fullmatch(rows[1].strip()) else None
    data = rows[2:] if separator is not None else rows[1:]
    matched = [row for row in data if _row_label(row) in labels]
    if not matched:
        return None
    rows_to_render = [header, *(matched)]
    rows_to_render = _project_period_columns(
        rows_to_render,
        period=period,
        comparison=comparison,
        raw_query=raw_query,
    )
    if not rows_to_render:
        return None
    parts = [rows_to_render[0]]
    if separator is not None:
        parts.append(_separator_for(rows_to_render[0]))
    parts.extend(rows_to_render[1:])
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
    return _normalize(_strip_footnote_suffix(cells[0] if cells else ""))


def _project_period_columns(
    rows: Sequence[str],
    *,
    period: Mapping[str, Any] | None,
    comparison: Mapping[str, Any] | None,
    raw_query: str | None,
) -> list[str]:
    if not rows or _is_comparison(comparison):
        return list(rows)
    year = _int_value((period or {}).get("year"))
    quarter = _int_value((period or {}).get("quarter"))
    if year is None and quarter is None:
        return list(rows)
    header_cells = _cells(rows[0])
    if len(header_cells) < 3:
        return list(rows)
    selected = [0]
    has_explicit_year = year is not None and any(
        str(year) in cell for cell in header_cells[1:]
    )
    current_term = None if has_explicit_year else _max_fiscal_term(header_cells[1:])
    for index, cell in enumerate(header_cells[1:], start=1):
        if _column_matches_period(
            cell,
            year=year,
            quarter=quarter,
            has_explicit_year=has_explicit_year,
            current_term=current_term,
        ):
            selected.append(index)
    if len(selected) == 1:
        return list(rows)
    selected = _prefer_quarter_duration(
        header_cells,
        selected,
        period=period,
        raw_query=raw_query,
    )
    return [
        _render_cells([cell for idx, cell in enumerate(_cells(row)) if idx in selected])
        for row in rows
    ]


def _column_matches_period(
    cell: str,
    *,
    year: int | None,
    quarter: int | None,
    has_explicit_year: bool,
    current_term: int | None,
) -> bool:
    text = str(cell or "")
    if year is not None:
        if has_explicit_year and str(year) not in text:
            return False
        if not has_explicit_year and current_term is not None:
            term = _fiscal_term(text)
            if term != current_term:
                return False
    if quarter is not None and f"{quarter}분기" not in re.sub(r"\s+", "", text):
        return False
    return True


def _max_fiscal_term(cells: Sequence[str]) -> int | None:
    terms = [_fiscal_term(cell) for cell in cells]
    found = [term for term in terms if term is not None]
    return max(found) if found else None


def _fiscal_term(value: str) -> int | None:
    match = re.search(r"제\s*(\d+)\s*(?:\([^)]*\)\s*)?기", str(value or ""))
    return int(match.group(1)) if match else None


def _is_comparison(comparison: Mapping[str, Any] | None) -> bool:
    comparison_type = str((comparison or {}).get("type") or "")
    return comparison_type in {
        "period_comparison",
        "year_over_year",
        "company_comparison",
        "trend",
        "before_after",
    }


def _prefer_quarter_duration(
    header_cells: Sequence[str],
    selected: list[int],
    *,
    period: Mapping[str, Any] | None,
    raw_query: str | None,
) -> list[int]:
    if _int_value((period or {}).get("quarter")) is None or len(selected) <= 2:
        return selected
    compact_query = re.sub(r"\s+", "", str(raw_query or ""))
    preferred = "누적" if any(term in compact_query for term in ("누적", "누계")) else "3개월"
    narrowed = [
        index for index in selected[1:] if preferred in str(header_cells[index] or "")
    ]
    return [0, *narrowed] if narrowed else selected


def _separator_for(header: str) -> str:
    return _render_cells(["---" for _ in _cells(header)])


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in str(row or "").strip().strip("|").split("|")]


def _render_cells(cells: Sequence[str]) -> str:
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _strip_footnote_suffix(value: str) -> str:
    return re.sub(r"\s*[\(\[]\s*주\s*\d+\s*[\)\]]\s*$", "", str(value or "")).strip()


def _metric_labels(metric: str | None) -> set[str]:
    label = _normalize(metric)
    if not label:
        return set()
    return {label, *(_METRIC_LABEL_ALIASES.get(label) or set())}


def _normalize(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").casefold())
