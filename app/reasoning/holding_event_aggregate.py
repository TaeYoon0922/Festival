"""Deterministic cross-event holding arithmetic from verified event fields."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any


def holding_multi_event_requested(plan: Mapping[str, Any] | None) -> bool:
    if not plan:
        return False
    evidence = plan.get("evidence")
    return isinstance(evidence, Mapping) and bool(
        evidence.get("holding_multi_event_compute")
    )


def aggregate_holding_change_statement(
    events: Sequence[Mapping[str, Any]],
) -> str | None:
    if len(events) < 2:
        return None
    deltas: list[float] = []
    baseline_before: float | None = None
    for event in events:
        change = _field_number(event.get("change_shares"))
        if change is not None:
            deltas.append(change)
        before = _field_number(event.get("before_shares"))
        if before is not None and baseline_before is None:
            baseline_before = before
    if len(deltas) < 2:
        return None
    net = sum(deltas)
    parts = [f"순증가 주식수(파생): {_format_shares(net)}"]
    if baseline_before not in (None, 0):
        rate = net / baseline_before * 100.0
        parts.append(f"증가율(파생): {_format_pct(rate)}")
    return " · ".join(parts)


def _field_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        normalized = value.get("normalized")
        if isinstance(normalized, (int, float)):
            return float(normalized)
        raw = value.get("raw")
        if raw is not None:
            return _parse_number(str(raw))
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return _parse_number(str(value))


def _parse_number(text: str) -> float | None:
    cleaned = re.sub(r"[^\d.\-+]", "", str(text or "").strip())
    if not cleaned or cleaned in {"-", "+", ".", "-.", "+."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_shares(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}주"
    return f"{value:,.2f}주"


def _format_pct(value: float) -> str:
    rounded = round(value, 2)
    sign = "+" if rounded > 0 else ""
    if float(rounded).is_integer():
        return f"{sign}{int(rounded)}%"
    return f"{sign}{rounded:.2f}%"
