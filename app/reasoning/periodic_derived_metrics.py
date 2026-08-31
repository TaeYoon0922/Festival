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


def _metric_fallback(request: Mapping[str, Any] | None) -> str | None:
    if not request:
        return None
    evidence = request.get("evidence")
    if isinstance(evidence, Mapping) and evidence.get("metric_fallback"):
        return str(evidence["metric_fallback"])
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
    if kind == "segment_rate":
        return _append_segment_rate(text, raw_query=raw_query, metric=metric)
    if kind == "quarter_timeseries":
        return _append_quarter_timeseries(text, raw_query=raw_query, metric=metric)
    if kind == "quarter_sum_vs_annual":
        return _append_quarter_sum_vs_annual(text, raw_query=raw_query, metric=metric)
    if kind == "sign_flip":
        return _append_sign_flip(text, metric=metric)
    fallback = _metric_fallback(request)
    base = project_periodic_metric_table(
        text,
        metric=metric,
        period=(request or {}).get("period"),
        comparison=comparison,
        raw_query=raw_query,
        metric_fallback=fallback,
    )
    if not base:
        return None
    if kind == "quarter_compare":
        return _append_quarter_compare(base, raw_query=raw_query)
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
    if kind == "peer_rate":
        base = project_periodic_metric_table(
            text,
            metric=metric,
            period=(request or {}).get("period"),
            comparison=comparison,
            raw_query=raw_query,
            metric_fallback=fallback,
        )
        if not base:
            return None
        return _append_rate(base, metric=metric)
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


def _append_quarter_compare(table: str, *, raw_query: str | None) -> str | None:
    values = _metric_row_values(table)
    if len(values) < 2:
        return None
    headers = _table_column_headers(table)
    left_label = headers[0] if headers else "전기"
    right_label = headers[1] if len(headers) > 1 else "당기"
    left_value, right_value = values[0], values[1]
    if right_value > left_value:
        winner = right_label
        delta = right_value - left_value
    elif left_value > right_value:
        winner = left_label
        delta = left_value - right_value
    else:
        winner = "동일"
        delta = 0.0
    return (
        f"{table}\n"
        f"(분기 비교(파생): {left_label} {_format_amount(left_value)} · "
        f"{right_label} {_format_amount(right_value)} · "
        f"더 큰 쪽: {winner} · 차이 {_format_amount(delta)})"
    )


def _table_column_headers(table: str) -> list[str]:
    rows = [line for line in str(table).splitlines() if line.strip().startswith("|")]
    if not rows:
        return []
    cells = _cells(rows[0])
    return cells[1:] if len(cells) > 1 else []


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


def _append_sign_flip(table: str, *, metric: str | None) -> str | None:
    values = _metric_row_values(table)
    if len(values) < 2:
        return None
    old_value, new_value = values[-2], values[-1]
    old_label = _sign_label(old_value)
    new_label = _sign_label(new_value)
    if old_label == new_label:
        verdict = f"{new_label} 지속"
    elif old_label == "적자" and new_label == "흑자":
        verdict = "흑자 전환"
    elif old_label == "흑자" and new_label == "적자":
        verdict = "적자 전환"
    else:
        verdict = f"{old_label} → {new_label}"
    label = metric or "당기순이익"
    return (
        f"{table}\n"
        f"({label} 부호(파생): 전기 {_format_amount(old_value)}({old_label}) · "
        f"당기 {_format_amount(new_value)}({new_label}) · 판정: {verdict})"
    )


def _sign_label(value: float) -> str:
    if value > 0:
        return "흑자"
    if value < 0:
        return "적자"
    return "0"


def _segment_labels_from_query(raw_query: str | None) -> list[str]:
    if not raw_query:
        return []
    match = re.search(r"([^(]+)\(또는([^)]+)\)", raw_query)
    if not match:
        return []
    return [
        re.sub(r"\s+", "", match.group(1)).casefold(),
        re.sub(r"\s+", "", match.group(2)).casefold(),
    ]


def _append_segment_rate(
    table: str,
    *,
    raw_query: str | None,
    metric: str | None,
) -> str | None:
    labels = _segment_labels_from_query(raw_query)
    rows = _data_rows(table)
    if not rows:
        return None
    for row in rows:
        row_label = re.sub(r"\s+", "", row[0]).casefold()
        if labels and not any(label in row_label for label in labels):
            continue
        values = [
            value
            for value in (_parse_amount(cell) for cell in row[1:])
            if value is not None
        ]
        if len(values) < 2:
            continue
        rate = _pct_change(values[-2], values[-1])
        if rate is None:
            continue
        label = metric or row[0].strip() or "segment"
        return (
            f"{table}\n"
            f"({label} 증감률(파생): {_format_pct(rate)} — "
            f"{_format_amount(values[-2])} → {_format_amount(values[-1])})"
        )
    return None


def _append_quarter_timeseries(
    table: str,
    *,
    raw_query: str | None,
    metric: str | None,
) -> str | None:
    values = _metric_row_values(table)
    headers = _table_column_headers(table)
    if len(values) < 3:
        return None
    max_value = max(values)
    min_value = min(values)
    max_index = values.index(max_value)
    min_index = values.index(min_value)
    max_label = headers[max_index] if max_index < len(headers) else f"#{max_index + 1}"
    min_label = headers[min_index] if min_index < len(headers) else f"#{min_index + 1}"
    delta = max_value - min_value
    label = metric or "매출액"
    return (
        f"{table}\n"
        f"({label} 분기 추이(파생): 최고 {max_label} {_format_amount(max_value)} · "
        f"최저 {min_label} {_format_amount(min_value)} · "
        f"차이 {_format_amount(delta)})"
    )


def _append_quarter_sum_vs_annual(
    table: str,
    *,
    raw_query: str | None,
    metric: str | None,
) -> str | None:
    values = _metric_row_values(table)
    headers = _table_column_headers(table)
    if len(values) < 2:
        return None
    quarter_indices = [
        index
        for index, header in enumerate(headers)
        if "분기" in str(header)
    ]
    if len(quarter_indices) >= 2:
        quarter_sum = sum(values[index] for index in quarter_indices)
        annual_indices = [
            index
            for index, header in enumerate(headers)
            if "분기" not in str(header) and re.search(r"\d{4}", str(header))
        ]
        annual_value = values[annual_indices[-1]] if annual_indices else values[-1]
    elif len(values) >= 5:
        quarter_sum = sum(values[:-1])
        annual_value = values[-1]
    else:
        return None
    delta = quarter_sum - annual_value
    same = abs(delta) < 1e-6
    label = metric or "영업이익"
    verdict = "일치" if same else "불일치"
    return (
        f"{table}\n"
        f"({label} 분기 합 vs annual(파생): "
        f"분기 합 {_format_amount(quarter_sum)} · "
        f"annual {_format_amount(annual_value)} · "
        f"차이 {_format_amount(delta)} · {verdict})"
    )


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


def peer_rate_from_table(table: str, *, metric: str | None = None) -> float | None:
    """Return YoY percent change from a two-period metric table."""

    values = _metric_row_values(table)
    if len(values) < 2:
        return None
    return _pct_change(values[-2], values[-1])


def peer_rate_compare_statement(
    rows: Sequence[tuple[str, float]],
    *,
    metric: str | None = None,
) -> str | None:
    """Compare YoY rates across named companies."""

    if len(rows) < 2:
        return None
    label = metric or "지표"
    parts = [f"{name} {_format_pct(rate)}" for name, rate in rows]
    winner = max(rows, key=lambda item: item[1])
    if len({rate for _, rate in rows}) == 1:
        verdict = "동일"
    else:
        verdict = f"더 높은 쪽: {winner[0]}"
    return (
        f"(peer {label} 증감률 비교(파생): "
        + " · ".join(parts)
        + f" · {verdict})"
    )
