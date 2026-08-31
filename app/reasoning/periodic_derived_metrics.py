"""Deterministic derived metrics over periodic table cells.

Values are read from disclosed table cells; rates, ratios, and %p deltas are
derived arithmetically, mirroring ``periodic_segment_ranking``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.reasoning.periodic_metric_view import project_periodic_metric_table
from app.reasoning.periodic_segment_ranking import _cells, _parse_amount, _TABLE_SEP


def derived_metric_requested(request: Mapping[str, Any] | None) -> bool:
    if not request:
        return False
    if request.get("derived_metric"):
        return True
    evidence = request.get("evidence")
    return isinstance(evidence, Mapping) and bool(evidence.get("derived_metric"))


def _derived_kind(request: Mapping[str, Any] | None) -> str | None:
    if not request:
        return None
    value = request.get("derived_metric")
    if value:
        return str(value)
    evidence = request.get("evidence")
    if isinstance(evidence, Mapping) and evidence.get("derived_metric"):
        return str(evidence["derived_metric"])
    return None


def project_derived_metric_display(
    text: str,
    *,
    metric: str | None,
    comparison: Mapping[str, Any] | None = None,
    request: Mapping[str, Any] | None = None,
    raw_query: str | None = None,
) -> str | None:
    kind = _derived_kind(request)
    if not kind or not str(text or "").strip():
        return None
    if kind == "balance_ratio":
        return _append_balance_sheet_ratio(text)
    base = project_periodic_metric_table(
        text,
        metric=metric,
        period=(request or {}).get("period"),
        comparison=comparison,
        raw_query=raw_query,
    )
    if not base:
        return None
    if kind == "rate":
        return _append_rate(base, metric=metric)
    if kind == "ratio":
        return _append_operating_margin_delta_pp(base, raw_query=raw_query)
    if kind == "delta_pp":
        return _append_breakdown_delta_pp(base, raw_query=raw_query)
    if kind == "compare_rates":
        return _append_compare_rates(
            text,
            raw_query=raw_query,
            comparison=comparison,
        )
    if kind == "breakdown_share":
        return _append_breakdown_share(base, raw_query=raw_query)
    return None


def _append_rate(table: str, *, metric: str | None) -> str | None:
    values = _metric_row_values(table)
    if len(values) < 2:
        return None
    old_value, new_value = values[-2], values[-1]
    rate = _pct_change(old_value, new_value)
    if rate is None:
        return None
    label = metric or "지표"
    return (
        f"{table}\n"
        f"({label} 증가율(파생): {_format_pct(rate)} — "
        f"{_format_amount(old_value)} → {_format_amount(new_value)})"
    )


def _append_balance_sheet_ratio(table: str) -> str | None:
    liability = _metric_values_from_table(table, ("부채총계", "총부채", "부채"))
    equity = _metric_values_from_table(table, ("자본총계", "총자본", "자기자본"))
    if not liability or not equity:
        return None
    lines = [table]
    period_labels = ("전기", "당기")
    for index in range(min(len(liability), len(equity))):
        ratio = _safe_ratio(liability[index], equity[index])
        if ratio is None:
            continue
        label = period_labels[index] if index < len(period_labels) else f"#{index + 1}"
        lines.append(
            f"(부채비율(파생) {label}: {_format_ratio_pct(ratio)} — "
            f"부채 {_format_amount(liability[index])} ÷ "
            f"자본 {_format_amount(equity[index])})"
        )
    if len(lines) == 1:
        return None
    return "\n".join(lines)


def _append_operating_margin_delta_pp(table: str, *, raw_query: str | None) -> str | None:
    revenue = _metric_values_from_table(table, ("매출", "매출액", "영업수익"))
    profit = _metric_values_from_table(table, ("영업이익", "영업손익"))
    if len(revenue) < 2 or len(profit) < 2:
        return None
    old_margin = _safe_ratio(profit[-2], revenue[-2])
    new_margin = _safe_ratio(profit[-1], revenue[-1])
    if old_margin is None or new_margin is None:
        return None
    delta = (new_margin - old_margin) * 100.0
    return (
        f"{table}\n"
        f"(영업이익률(파생): 전기 {_format_pct(old_margin * 100)} · "
        f"당기 {_format_pct(new_margin * 100)} · "
        f"변화 {_format_pp(delta)})"
    )


def _append_breakdown_share(table: str, *, raw_query: str | None) -> str | None:
    rows = _data_rows(table)
    if not rows:
        return None
    domestic = _row_values_for_label(rows, ("국내",))
    overseas = _row_values_for_label(rows, ("해외", "수출"))
    if not domestic or not overseas:
        return None
    lines = [table]
    for period_index in range(min(len(domestic), len(overseas))):
        total = domestic[period_index] + overseas[period_index]
        if not total:
            continue
        domestic_share = domestic[period_index] / total * 100.0
        overseas_share = overseas[period_index] / total * 100.0
        lines.append(
            f"(매출 비중(파생) #{period_index + 1}: "
            f"국내 {_format_pct(domestic_share)} · 해외 {_format_pct(overseas_share)})"
        )
    if len(lines) == 1:
        return None
    if len(domestic) >= 2 and len(overseas) >= 2:
        old_total = domestic[0] + overseas[0]
        new_total = domestic[1] + overseas[1]
        if old_total and new_total:
            old_overseas = overseas[0] / old_total * 100.0
            new_overseas = overseas[1] / new_total * 100.0
            lines.append(
                f"(해외 비중 변화(파생): {_format_pp(new_overseas - old_overseas)})"
            )
    return "\n".join(lines)


def _append_compare_rates(
    table: str,
    *,
    raw_query: str | None,
    comparison: Mapping[str, Any] | None,
) -> str | None:
    revenue = _metric_values_from_table(table, ("매출", "매출액", "영업수익"))
    profit = _metric_values_from_table(table, ("영업이익", "영업손익"))
    if len(revenue) < 2 or len(profit) < 2:
        return None
    revenue_rate = _pct_change(revenue[-2], revenue[-1])
    profit_rate = _pct_change(profit[-2], profit[-1])
    if revenue_rate is None or profit_rate is None:
        return None
    winner = "영업이익 증가율" if profit_rate > revenue_rate else "매출 증가율"
    if profit_rate == revenue_rate:
        winner = "동일"
    return (
        f"{table}\n"
        f"(증가율(파생): 매출 {_format_pct(revenue_rate)} · "
        f"영업이익 {_format_pct(profit_rate)} · 더 큰 쪽: {winner})"
    )


def _append_breakdown_delta_pp(table: str, *, raw_query: str | None) -> str | None:
    return _append_breakdown_share(table, raw_query=raw_query)


def _metric_row_values(table: str) -> list[float]:
    rows = _data_rows(table)
    if not rows:
        return []
    values: list[float] = []
    for row in rows[:1]:
        for cell in row[1:]:
            parsed = _parse_amount(cell)
            if parsed is not None:
                values.append(parsed)
    return values


def _metric_values_from_table(
    table: str, labels: Sequence[str]
) -> list[float]:
    rows = _data_rows(table)
    normalized = {re.sub(r"\s+", "", label).casefold() for label in labels}
    for row in rows:
        label = re.sub(r"\s+", "", row[0]).casefold()
        if any(item in label for item in normalized):
            values = [_parse_amount(cell) for cell in row[1:]]
            return [value for value in values if value is not None]
    return []


def _row_values_for_label(
    rows: Sequence[Sequence[str]], labels: Sequence[str]
) -> list[float]:
    normalized = {re.sub(r"\s+", "", label).casefold() for label in labels}
    for row in rows:
        label = re.sub(r"\s+", "", row[0]).casefold()
        if any(item in label for item in normalized):
            return [
                value
                for value in (_parse_amount(cell) for cell in row[1:])
                if value is not None
            ]
    return []


def _data_rows(table: str) -> list[list[str]]:
    rows = [line for line in str(table).splitlines() if line.strip().startswith("|")]
    if len(rows) < 2:
        return []
    separator = rows[1] if _TABLE_SEP.fullmatch(rows[1].strip()) else None
    data = rows[2:] if separator is not None else rows[1:]
    return [_cells(row) for row in data if _cells(row)]


def _pct_change(old_value: float, new_value: float) -> float | None:
    if old_value == 0:
        return None
    return (new_value - old_value) / abs(old_value) * 100.0


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _format_ratio_pct(ratio: float) -> str:
    pct = ratio * 100.0
    rounded = round(pct, 2)
    if float(rounded).is_integer():
        return f"{int(rounded)}%"
    return f"{rounded:.2f}%"


def _format_pct(value: float) -> str:
    rounded = round(value, 2)
    if float(rounded).is_integer():
        return f"{int(rounded)}%"
    return f"{rounded:.2f}%"


def _format_pp(value: float) -> str:
    rounded = round(value, 2)
    sign = "+" if rounded > 0 else ""
    if float(rounded).is_integer():
        return f"{sign}{int(rounded)}%p"
    return f"{sign}{rounded:.2f}%p"


def _format_amount(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"
