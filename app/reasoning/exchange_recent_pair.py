"""Compare the two most recent exchange filings by receipt date.

Amounts are parsed from disclosed fields; deltas and equity ratios are derived.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.reasoning.exchange_field_aggregate import (
    ParsedExchangeAmount,
    format_aggregate_amount,
    parse_labeled_amount,
    receipt_date_from_doc_id,
)


@dataclass(frozen=True)
class ExchangeRecentPairCompare:
    field: str
    newer_doc_id: str
    older_doc_id: str
    newer_amount: ParsedExchangeAmount
    older_amount: ParsedExchangeAmount
    newer_equity_ratio_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "field": self.field,
            "newer_doc_id": self.newer_doc_id,
            "older_doc_id": self.older_doc_id,
            "newer_amount": self.newer_amount.value,
            "older_amount": self.older_amount.value,
            "derived": True,
        }
        if self.newer_equity_ratio_pct is not None:
            payload["newer_equity_ratio_pct"] = self.newer_equity_ratio_pct
        return payload


def exchange_recent_pair_requested(plan: Mapping[str, Any] | None) -> bool:
    if not plan:
        return False
    evidence = plan.get("evidence")
    if isinstance(evidence, Mapping) and evidence.get("exchange_recent_pair"):
        return True
    return getattr(plan, "recent_pair_limit", None) is not None


def select_recent_documents(doc_ids: Sequence[str], *, limit: int) -> list[str]:
    scored: list[tuple[str, str]] = []
    for doc_id in doc_ids:
        receipt = receipt_date_from_doc_id(doc_id)
        if receipt is not None:
            scored.append((receipt, doc_id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [doc_id for _, doc_id in scored[:limit]]


def build_recent_pair_compare(
    doc_texts: Sequence[tuple[str, str]],
    *,
    field: str,
    include_equity_ratio: bool = False,
) -> ExchangeRecentPairCompare | None:
    if len(doc_texts) < 2:
        return None
    newer_doc_id, newer_text = doc_texts[0]
    older_doc_id, older_text = doc_texts[1]
    newer_amount = parse_labeled_amount(newer_text, field)
    older_amount = parse_labeled_amount(older_text, field)
    if newer_amount is None or older_amount is None:
        return None
    equity_ratio: float | None = None
    if include_equity_ratio:
        equity = parse_labeled_amount(newer_text, "equity_capital")
        if equity is not None and equity.value:
            equity_ratio = newer_amount.value / equity.value * 100.0
    return ExchangeRecentPairCompare(
        field=field,
        newer_doc_id=newer_doc_id,
        older_doc_id=older_doc_id,
        newer_amount=newer_amount,
        older_amount=older_amount,
        newer_equity_ratio_pct=equity_ratio,
    )


def recent_pair_statement(compare: ExchangeRecentPairCompare) -> str:
    field_label = "투자금액" if compare.field == "investment_amount" else "계약금액"
    parts = [
        (
            f"최근 {field_label}(파생): "
            f"{format_aggregate_amount(compare.newer_amount.value)}"
        ),
    ]
    if compare.newer_equity_ratio_pct is not None:
        parts.append(
            f"자기자본 대비(파생): {_format_pct(compare.newer_equity_ratio_pct)}"
        )
    delta = compare.newer_amount.value - compare.older_amount.value
    if delta > 0:
        direction = "증가"
    elif delta < 0:
        direction = "감소"
    else:
        direction = "변동 없음"
    parts.append(
        f"직전 공시 대비(파생): {format_aggregate_amount(abs(delta))} {direction} "
        f"({format_aggregate_amount(compare.older_amount.value)} → "
        f"{format_aggregate_amount(compare.newer_amount.value)})"
    )
    return " ".join(parts)


def _format_pct(value: float) -> str:
    rounded = round(value, 2)
    if float(rounded).is_integer():
        return f"{int(rounded)}%"
    return f"{rounded:.2f}%"
