"""Fail-closed arithmetic over one periodic statement row.

The periodic selector already preserves the exact metric row and the two
comparable period columns a question requested.  This module adds the missing
typed boundary: it turns those two cells into source-bound operands, and only
then computes ``final - initial`` and the relative change from the initial
value.

It is deliberately not a general table calculator.  A request resolves only
when one selected source contains one exact metric row, exactly two requested
years, comparable period headers, one known monetary unit, and non-negative
operands with a positive baseline.  Anything else is left unresolved.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Mapping, Sequence

from app.reasoning.periodic_fact_resolver import (
    PeriodicFactResolution,
    PeriodicFactSource,
)
from app.reasoning.periodic_metric_view import project_periodic_metric_table


PERIODIC_METRIC_CHANGE_KEY = "periodic_metric_change"

_RATE_REQUEST = re.compile(
    r"증감\s*률|증가\s*율|감소\s*율|"
    r"몇\s*(?:퍼센트|%)|"
    r"(?:퍼센트|%)\s*(?:나|만큼)?\s*(?:증가|감소|늘|줄)|"
    r"(?:증가|감소|늘|줄)\s*(?:한|한가|했|했나|했어|했나요|했습니까)?\s*"
    r"(?:퍼센트|%)"
)
_CHANGE_QUERY_NOISE = re.compile(
    r"(?:" + _RATE_REQUEST.pattern + r")(?:은|는|이|가|을|를)?|"
    r"전년\s*동기\s*대비|전년\s*대비|대비|얼마나|"
    r"(?:증가|감소)(?:했(?:어|나|나요|습니까|는지)?|한(?:가|지)?|하였(?:어|나)?)?|"
    r"늘(?:었(?:어|나|나요|습니까|는지)?|어났(?:어|나|나요)?|어난)?|"
    r"줄(?:었(?:어|나|나요|습니까|는지)?|어들었(?:어|나|나요)?|어든)?"
)
_TABLE_SEPARATOR = re.compile(r"^\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_FISCAL_TERM = re.compile(r"제\s*(\d+)\s*(?:\([^)]*\)\s*)?기")
_CALENDAR_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_ROW_UNIT = re.compile(
    r"단위\s*[:：]?\s*(억\s*원|백\s*만\s*원|천\s*원|원)",
    re.IGNORECASE,
)
_MONETARY_UNITS = ("백만원", "억원", "천원", "원")
_TWO_DECIMALS = Decimal("0.01")


@dataclass(frozen=True)
class PeriodicMetricChangeRequest:
    """The metric and chronological years named by a rate question."""

    metric: str
    years: tuple[int, ...]
    comparison_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "years": list(self.years),
            "comparison_type": self.comparison_type,
        }


@dataclass(frozen=True)
class PeriodicMetricOperand:
    """One value that retains its period column and source identity."""

    year: int
    value: Decimal
    raw_value: str
    unit: str
    column_label: str
    chunk_id: str
    doc_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "value": _decimal_text(self.value),
            "raw_value": self.raw_value,
            "unit": self.unit,
            "column_label": self.column_label,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
        }


@dataclass(frozen=True)
class PeriodicMetricChange:
    """The two source-bound operands and their verified arithmetic."""

    metric: str
    initial: PeriodicMetricOperand
    final: PeriodicMetricOperand
    difference: Decimal
    pct_change: Decimal

    @property
    def unit(self) -> str:
        return self.initial.unit

    @property
    def direction(self) -> str:
        if self.difference > 0:
            return "increase"
        if self.difference < 0:
            return "decrease"
        return "unchanged"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "unit": self.unit,
            "initial": self.initial.to_dict(),
            "final": self.final.to_dict(),
            "difference": _decimal_text(self.difference),
            "pct_change": _decimal_text(self.pct_change),
            "direction": self.direction,
        }


def periodic_metric_change_requested(
    query: str,
    *,
    metric: str | None,
    comparison: Mapping[str, Any] | None,
    mentioned_years: Sequence[int] = (),
) -> PeriodicMetricChangeRequest | None:
    """Record an explicit percentage-change request without inventing a baseline."""

    text = str(query or "")
    if not metric or not _RATE_REQUEST.search(text):
        return None
    comparison_type = str((comparison or {}).get("type") or "") or None
    values = (
        (comparison or {}).get("years")
        if comparison_type in {"period_comparison", "year_over_year"}
        else mentioned_years
    )
    years = _years(values or mentioned_years)
    return PeriodicMetricChangeRequest(
        metric=str(metric),
        years=years,
        comparison_type=comparison_type,
    )


def periodic_metric_change_spans(query: str) -> list[tuple[int, int]]:
    """Search-noise spans for a query already recognized as a rate request."""

    return [match.span() for match in _CHANGE_QUERY_NOISE.finditer(str(query or ""))]


def requested_periodic_metric_change(
    plan: Any,
) -> PeriodicMetricChangeRequest | None:
    """Read the request query understanding recorded; never re-read prose."""

    evidence = _plan_mapping(plan).get("evidence")
    value = (
        dict(evidence).get(PERIODIC_METRIC_CHANGE_KEY)
        if isinstance(evidence, Mapping)
        else None
    )
    if not isinstance(value, Mapping):
        return None
    metric = str(value.get("metric") or "").strip()
    if not metric:
        return None
    return PeriodicMetricChangeRequest(
        metric=metric,
        years=_years(value.get("years") or ()),
        comparison_type=(
            str(value.get("comparison_type"))
            if value.get("comparison_type")
            else None
        ),
    )


def resolve_periodic_metric_change(
    request: PeriodicMetricChangeRequest,
    resolution: PeriodicFactResolution,
    *,
    query_plan: Any,
) -> PeriodicMetricChange | None:
    """Resolve one exact two-period row and calculate its relative change."""

    if len(request.years) != 2 or request.years[0] >= request.years[1]:
        return None
    if resolution.unresolved_requirements or resolution.temporal_ambiguity:
        return None
    if len(resolution.facts) != 1:
        return None
    fact = resolution.facts[0]
    if fact.fact_conflict or len(fact.sources) != 1:
        return None
    source = fact.sources[0]
    if not source.chunk_id or not source.doc_id:
        return None
    plan = _plan_mapping(query_plan)
    projected = project_periodic_metric_table(
        source.fact_text,
        metric=request.metric,
        period=_mapping(plan.get("period")),
        comparison=_mapping(plan.get("comparison")),
        raw_query=str(plan.get("raw_query") or resolution.question or ""),
    )
    parsed = _two_period_row(
        projected,
        source=source,
        requested_years=request.years,
    )
    if parsed is None:
        return None
    unit, values = parsed
    initial_value = values[request.years[0]]
    final_value = values[request.years[1]]
    # Relative growth from zero is undefined, while a loss/profit sign change
    # cannot honestly be summarized as an ordinary percentage growth rate.
    if initial_value[0] <= 0 or final_value[0] < 0:
        return None
    initial = PeriodicMetricOperand(
        year=request.years[0],
        value=initial_value[0],
        raw_value=initial_value[1],
        unit=unit,
        column_label=initial_value[2],
        chunk_id=source.chunk_id,
        doc_id=source.doc_id,
    )
    final = PeriodicMetricOperand(
        year=request.years[1],
        value=final_value[0],
        raw_value=final_value[1],
        unit=unit,
        column_label=final_value[2],
        chunk_id=source.chunk_id,
        doc_id=source.doc_id,
    )
    difference = final.value - initial.value
    pct_change = (
        (difference / initial.value) * Decimal("100")
    ).quantize(_TWO_DECIMALS, rounding=ROUND_HALF_UP)
    return PeriodicMetricChange(
        metric=request.metric,
        initial=initial,
        final=final,
        difference=difference,
        pct_change=pct_change,
    )


def periodic_metric_change_claims(
    value: Mapping[str, Any],
) -> tuple[tuple[str, tuple[str, ...]], ...] | None:
    """Validate a serialized change and return its deterministic cited lines."""

    try:
        metric = str(value["metric"]).strip()
        unit = _normalize_unit(value["unit"])
        initial = dict(value["initial"])
        final = dict(value["final"])
        initial_year = int(initial["year"])
        final_year = int(final["year"])
        initial_value = Decimal(str(initial["value"]))
        final_value = Decimal(str(final["value"]))
        difference = Decimal(str(value["difference"]))
        pct_change = Decimal(str(value["pct_change"]))
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return None
    if (
        not metric
        or unit is None
        or initial_year >= final_year
        or initial_value <= 0
        or final_value < 0
        or _normalize_unit(initial.get("unit")) != unit
        or _normalize_unit(final.get("unit")) != unit
    ):
        return None
    calculated_difference = final_value - initial_value
    calculated_pct = (
        (calculated_difference / initial_value) * Decimal("100")
    ).quantize(_TWO_DECIMALS, rounding=ROUND_HALF_UP)
    if difference != calculated_difference or pct_change != calculated_pct:
        return None
    initial_ids = _source_ids(initial)
    final_ids = _source_ids(final)
    if not initial_ids or not final_ids:
        return None
    both_ids = tuple(dict.fromkeys((*initial_ids, *final_ids)))
    return (
        (
            f"{initial_year}년 {metric}: {_format_number(initial_value)}{unit}",
            initial_ids,
        ),
        (
            f"{final_year}년 {metric}: {_format_number(final_value)}{unit}",
            final_ids,
        ),
        (
            f"증감액: {_format_signed(difference)}{unit}",
            both_ids,
        ),
        (
            f"증감률: {_format_signed(pct_change, decimals=2)}%",
            both_ids,
        ),
    )


def _two_period_row(
    projected: str | None,
    *,
    source: PeriodicFactSource,
    requested_years: tuple[int, ...],
) -> tuple[str, dict[int, tuple[Decimal, str, str]]] | None:
    if not projected:
        return None
    rows = [
        line
        for line in projected.splitlines()
        if line.strip().startswith("|")
        and not _TABLE_SEPARATOR.fullmatch(line.strip())
    ]
    if len(rows) != 2:
        return None
    headers = _cells(rows[0])
    values = _cells(rows[1])
    if len(headers) != 3 or len(values) != 3 or len(headers) != len(values):
        return None
    period_headers = headers[1:]
    if _period_signature(period_headers[0]) != _period_signature(period_headers[1]):
        return None
    source_year = _source_year(source)
    years = _header_years(period_headers, source_year=source_year)
    if years is None or set(years) != set(requested_years):
        return None
    parsed_values = [_numeric_cell(cell) for cell in values[1:]]
    if any(parsed is None for parsed in parsed_values):
        return None
    numeric_values = [parsed for parsed in parsed_values if parsed is not None]
    cell_units = {unit for _number, unit in numeric_values if unit}
    if len(cell_units) > 1:
        return None
    row_unit = _unit_from_row_label(values[0])
    chunk_unit = _normalize_unit(
        dict(source.provenance.get("source_chunk") or {}).get("unit")
    )
    cell_unit = next(iter(cell_units), None)
    specific_units = [unit for unit in (cell_unit, row_unit) if unit]
    if len(set(specific_units)) > 1:
        return None
    unit = cell_unit or row_unit or chunk_unit
    if unit is None:
        return None
    if chunk_unit is not None and not specific_units and chunk_unit != unit:
        return None
    output: dict[int, tuple[Decimal, str, str]] = {}
    for year, header, raw, parsed in zip(
        years, period_headers, values[1:], numeric_values, strict=True
    ):
        number, parsed_unit = parsed
        if parsed_unit is not None and parsed_unit != unit:
            return None
        output[year] = (number, raw, header)
    return unit, output


def _header_years(
    headers: Sequence[str], *, source_year: int | None
) -> tuple[int, int] | None:
    explicit: list[int | None] = []
    for header in headers:
        matches = {int(value) for value in _CALENDAR_YEAR.findall(header)}
        explicit.append(next(iter(matches)) if len(matches) == 1 else None)
    if all(year is not None for year in explicit) and len(set(explicit)) == 2:
        return int(explicit[0]), int(explicit[1])

    terms = [_fiscal_term(header) for header in headers]
    if source_year is not None and all(term is not None for term in terms):
        maximum = max(int(term) for term in terms if term is not None)
        inferred = tuple(
            source_year - (maximum - int(term))
            for term in terms
            if term is not None
        )
        if len(inferred) == 2 and len(set(inferred)) == 2:
            return inferred[0], inferred[1]

    if source_year is not None:
        markers = [_current_prior_marker(header) for header in headers]
        if set(markers) == {"current", "prior"}:
            return tuple(
                source_year if marker == "current" else source_year - 1
                for marker in markers
            )
    return None


def _period_signature(header: str) -> str:
    text = _CALENDAR_YEAR.sub("", str(header or ""))
    text = _FISCAL_TERM.sub("", text)
    text = re.sub(r"\(?\s*(?:당|전)\s*\)?\s*기", "", text)
    return re.sub(r"[^0-9a-z가-힣]+", "", text.casefold())


def _fiscal_term(value: str) -> int | None:
    match = _FISCAL_TERM.search(str(value or ""))
    return int(match.group(1)) if match else None


def _current_prior_marker(value: str) -> str | None:
    normalized = re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").casefold())
    if "당기" in normalized:
        return "current"
    if "전기" in normalized:
        return "prior"
    return None


def _source_year(source: PeriodicFactSource) -> int | None:
    for key in ("year", "fiscal_year", "base_year"):
        try:
            value = source.reporting_period.get(key)
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _numeric_cell(value: str) -> tuple[Decimal, str | None] | None:
    text = re.sub(r"\s+", "", str(value or ""))
    if not text or text in {"-", "--", "해당없음", "없음"}:
        return None
    unit = None
    for candidate in _MONETARY_UNITS:
        if text.endswith(candidate):
            unit = candidate
            text = text[: -len(candidate)]
            break
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    if text.startswith(("-", "△")):
        negative = True
        text = text[1:]
    elif text.startswith(("+", "▲")):
        text = text[1:]
    text = text.replace(",", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", text):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return (-number if negative else number), unit


def _unit_from_row_label(value: str) -> str | None:
    match = _ROW_UNIT.search(str(value or ""))
    return _normalize_unit(match.group(1)) if match else None


def _normalize_unit(value: Any) -> str | None:
    text = re.sub(r"[\s()\[\]]+", "", str(value or ""))
    text = re.sub(r"^단위[:：]?", "", text)
    return text if text in _MONETARY_UNITS else None


def _format_number(value: Decimal) -> str:
    rendered = f"{value:,f}"
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _format_signed(value: Decimal, *, decimals: int | None = None) -> str:
    if decimals is None:
        body = _format_number(abs(value))
    else:
        body = f"{abs(value):,.{decimals}f}"
    if value > 0:
        return f"+{body}"
    if value < 0:
        return f"-{body}"
    return body


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _source_ids(value: Mapping[str, Any]) -> tuple[str, ...]:
    chunk_id = str(value.get("chunk_id") or "")
    doc_id = str(value.get("doc_id") or "")
    return (chunk_id,) if chunk_id and doc_id else ()


def _years(values: Sequence[Any]) -> tuple[int, ...]:
    years: list[int] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            year = int(value)
        except (TypeError, ValueError):
            continue
        if 1900 <= year <= 2100:
            years.append(year)
    return tuple(sorted(set(years)))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _plan_mapping(plan: Any) -> dict[str, Any]:
    if isinstance(plan, Mapping):
        return copy.deepcopy(dict(plan))
    if hasattr(plan, "to_dict"):
        return copy.deepcopy(dict(plan.to_dict()))
    return {}


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in str(row or "").strip().strip("|").split("|")]


__all__ = [
    "PERIODIC_METRIC_CHANGE_KEY",
    "PeriodicMetricChange",
    "PeriodicMetricChangeRequest",
    "PeriodicMetricOperand",
    "periodic_metric_change_claims",
    "periodic_metric_change_requested",
    "periodic_metric_change_spans",
    "requested_periodic_metric_change",
    "resolve_periodic_metric_change",
]
