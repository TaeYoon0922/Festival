"""Aggregate labeled amounts from exchange disclosure text.

Values are parsed from disclosed fields only; sum and average are derived.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.reasoning.periodic_segment_ranking import _parse_amount

_DOC_RECEIPT = re.compile(r"^exchange_(\d{8})")


FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "contract_amount": ("계약금액", "계약금", "계약금액(원)"),
    "investment_amount": ("투자금액", "투자금", "투자금액(원)"),
    "equity_capital": ("자기자본", "자기자본(원)"),
}

_UNIT_MULTIPLIERS = (
    ("조원", 1_000_000_000_000.0),
    ("억원", 100_000_000.0),
    ("백만원", 1_000_000.0),
    ("천원", 1_000.0),
    ("원", 1.0),
)


@dataclass(frozen=True)
class ParsedExchangeAmount:
    value: float
    raw: str
    label: str


@dataclass(frozen=True)
class ExchangeYearAggregate:
    field: str
    year: int
    amount_sum: float
    document_count: int
    parsed_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "year": self.year,
            "amount_sum": self.amount_sum,
            "document_count": self.document_count,
            "parsed_count": self.parsed_count,
            "derived": True,
        }


@dataclass(frozen=True)
class ExchangeFieldAggregate:
    field: str
    document_count: int
    parsed_count: int
    amount_sum: float | None = None
    amount_average: float | None = None
    amounts: tuple[ParsedExchangeAmount, ...] = ()
    by_year: tuple[ExchangeYearAggregate, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "field": self.field,
            "document_count": self.document_count,
            "parsed_count": self.parsed_count,
            "derived": True,
        }
        if self.amount_sum is not None:
            payload["amount_sum"] = self.amount_sum
        if self.amount_average is not None:
            payload["amount_average"] = self.amount_average
        if self.by_year:
            payload["by_year"] = [item.to_dict() for item in self.by_year]
        return payload


def receipt_date_from_doc_id(doc_id: str) -> str | None:
    match = _DOC_RECEIPT.match(str(doc_id or "").strip())
    if match is None:
        return None
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"


def exchange_aggregate_requested(plan: Mapping[str, Any] | None) -> bool:
    if not plan:
        return False
    evidence = plan.get("evidence")
    if isinstance(evidence, Mapping) and evidence.get("exchange_aggregate"):
        return True
    aggregate_ops = plan.get("aggregate_ops")
    return bool(aggregate_ops)


def aggregate_field_for_event(event_type: str | None) -> str:
    if event_type == "facility_investment":
        return "investment_amount"
    return "contract_amount"


def parse_labeled_amount(text: str, field: str) -> ParsedExchangeAmount | None:
    labels = FIELD_LABELS.get(field, ())
    if not labels or not str(text or "").strip():
        return None
    for label in labels:
        escaped = re.escape(label)
        patterns = (
            rf"{escaped}\s*[:：|]\s*([\d,]+(?:\.\d+)?)\s*({ '|'.join(re.escape(unit) for unit, _ in _UNIT_MULTIPLIERS) })?",
            rf"{escaped}\s*[:：]?\s*([\d,]+(?:\.\d+)?)\s*({ '|'.join(re.escape(unit) for unit, _ in _UNIT_MULTIPLIERS) })",
        )
        for pattern in patterns:
            match = re.search(pattern, str(text))
            if not match:
                continue
            raw_number = match.group(1)
            unit = match.group(2) if match.lastindex and match.lastindex >= 2 else "원"
            value = _scaled_amount(raw_number, unit or "원")
            if value is None:
                continue
            return ParsedExchangeAmount(value=value, raw=match.group(0).strip(), label=label)
        table_match = re.search(
            rf"\|\s*{escaped}\s*\|\s*([\d,]+(?:\.\d+)?)\s*(?:\|\s*([\d,]+(?:\.\d+)?))?",
            str(text),
        )
        if table_match:
            raw_number = table_match.group(1)
            value = _parse_amount(raw_number)
            if value is not None:
                return ParsedExchangeAmount(
                    value=value,
                    raw=table_match.group(0).strip(),
                    label=label,
                )
    return None


def aggregate_exchange_amounts(
    texts: Sequence[str],
    field: str,
    *,
    ops: Sequence[str] = ("sum",),
    doc_ids: Sequence[str] | None = None,
    years: Sequence[int] | None = None,
) -> ExchangeFieldAggregate:
    doc_ids = list(doc_ids or [])
    if doc_ids and len(doc_ids) != len(texts):
        doc_ids = []
    parsed_pairs: list[tuple[str | None, ParsedExchangeAmount]] = []
    for index, text in enumerate(texts):
        amount = parse_labeled_amount(text, field)
        if amount is not None:
            doc_id = doc_ids[index] if index < len(doc_ids) else None
            parsed_pairs.append((doc_id, amount))
    parsed = [amount for _, amount in parsed_pairs]
    amount_sum = sum(item.value for item in parsed) if parsed else None
    amount_average = (amount_sum / len(parsed)) if amount_sum is not None and parsed else None
    if "sum" not in ops:
        amount_sum = None
    if "average" not in ops:
        amount_average = None
    by_year: tuple[ExchangeYearAggregate, ...] = ()
    if years and parsed_pairs and doc_ids:
        by_year = aggregate_amounts_by_receipt_year(parsed_pairs, field=field, years=years)
    return ExchangeFieldAggregate(
        field=field,
        document_count=len(texts),
        parsed_count=len(parsed),
        amount_sum=amount_sum,
        amount_average=amount_average,
        amounts=tuple(parsed),
        by_year=by_year,
    )


def aggregate_amounts_by_receipt_year(
    parsed_pairs: Sequence[tuple[str | None, ParsedExchangeAmount]],
    *,
    field: str,
    years: Sequence[int],
) -> tuple[ExchangeYearAggregate, ...]:
    wanted = {int(year) for year in years}
    buckets: dict[int, list[ParsedExchangeAmount]] = {year: [] for year in sorted(wanted)}
    seen_docs: dict[int, set[str]] = {year: set() for year in buckets}
    doc_counts: dict[int, int] = {year: 0 for year in buckets}
    for doc_id, amount in parsed_pairs:
        receipt = receipt_date_from_doc_id(str(doc_id or ""))
        if receipt is None:
            continue
        year = int(receipt[:4])
        if year not in buckets:
            continue
        doc_counts[year] += 1
        if doc_id and doc_id in seen_docs[year]:
            continue
        if doc_id:
            seen_docs[year].add(str(doc_id))
        buckets[year].append(amount)
    rows: list[ExchangeYearAggregate] = []
    for year in sorted(wanted):
        items = buckets.get(year, [])
        rows.append(
            ExchangeYearAggregate(
                field=field,
                year=year,
                amount_sum=sum(item.value for item in items) if items else 0.0,
                document_count=doc_counts.get(year, 0),
                parsed_count=len(items),
            )
        )
    return tuple(rows)


def year_compare_statement(
    by_year: Sequence[ExchangeYearAggregate],
    *,
    years: Sequence[int],
) -> str:
    ordered = sorted(by_year, key=lambda item: item.year)
    if len(ordered) < 2:
        return ""
    field_label = "투자금액" if ordered[0].field == "investment_amount" else "계약금액"
    parts = [
        (
            f"{item.year}년 {field_label} 합계(파생): "
            f"{format_aggregate_amount(item.amount_sum)} "
            f"({item.parsed_count}/{item.document_count}건)"
        )
        for item in ordered
    ]
    newer, older = ordered[-1], ordered[-2]
    delta = newer.amount_sum - older.amount_sum
    direction = "증가" if delta > 0 else "감소" if delta < 0 else "동일"
    parts.append(
        f"{newer.year}년이 {older.year}년 대비(파생): "
        f"{format_aggregate_amount(abs(delta))} {direction}"
    )
    return " ".join(parts)


def format_aggregate_amount(value: float) -> str:
    if abs(value) >= 100_000_000:
        return f"{value:,.0f}원"
    if float(value).is_integer():
        return f"{int(value):,}원"
    return f"{value:,.2f}원"


def aggregate_statement(
    facts: Mapping[str, Any],
    aggregate: ExchangeFieldAggregate,
) -> str:
    parts: list[str] = []
    logical_count = int(facts.get("logical_count") or aggregate.document_count or 0)
    unresolved = int(facts.get("unresolved_count") or 0)
    if logical_count:
        if unresolved:
            parts.append(
                f"조건에 해당하는 공시는 {logical_count}건으로 확인되며, "
                f"이 중 {unresolved}건은 관련 공시 연결을 확정하지 못했습니다."
            )
        else:
            parts.append(f"조건에 해당하는 공시는 모두 {logical_count}건입니다.")
    field_label = "투자금액" if aggregate.field == "investment_amount" else "계약금액"
    if aggregate.amount_sum is not None:
        parts.append(
            f"{field_label} 합계(파생): {format_aggregate_amount(aggregate.amount_sum)} "
            f"({aggregate.parsed_count}/{aggregate.document_count}건 파싱)"
        )
    if aggregate.amount_average is not None:
        parts.append(
            f"건당 평균 {field_label}(파생): "
            f"{format_aggregate_amount(aggregate.amount_average)}"
        )
    if aggregate.by_year:
        year_text = year_compare_statement(
            aggregate.by_year,
            years=[item.year for item in aggregate.by_year],
        )
        if year_text:
            parts.append(year_text)
    return " ".join(parts) if parts else ""


def _scaled_amount(raw_number: str, unit: str) -> float | None:
    base = _parse_amount(raw_number)
    if base is None:
        return None
    multiplier = next((scale for suffix, scale in _UNIT_MULTIPLIERS if suffix == unit), 1.0)
    return base * multiplier
