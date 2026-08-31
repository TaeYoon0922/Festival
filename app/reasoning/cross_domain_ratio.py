"""Cross-domain ratios between exchange aggregates and periodic metrics."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.reasoning.exchange_field_aggregate import ExchangeFieldAggregate, format_aggregate_amount
from app.reasoning.periodic_derived_metrics import _metric_row_values
from app.reasoning.periodic_metric_view import project_periodic_metric_table


@dataclass(frozen=True)
class CrossDomainRatio:
    numerator_op: str
    numerator_value: float
    denominator_metric: str
    denominator_value: float
    ratio_percent: float
    year: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "numerator_op": self.numerator_op,
            "numerator_value": self.numerator_value,
            "denominator_metric": self.denominator_metric,
            "denominator_value": self.denominator_value,
            "ratio_percent": self.ratio_percent,
            "year": self.year,
            "derived": True,
        }


def cross_domain_ratio_requested(plan: Mapping[str, Any] | None) -> bool:
    if not plan:
        return False
    evidence = plan.get("evidence")
    return isinstance(evidence, Mapping) and bool(evidence.get("cross_domain_ratio"))


def cross_domain_config(plan: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not plan:
        return None
    evidence = plan.get("evidence")
    if not isinstance(evidence, Mapping):
        return None
    config = evidence.get("cross_domain_ratio")
    return config if isinstance(config, Mapping) else None


def extract_periodic_metric_amount(
    text: str,
    *,
    metric: str,
    year: int | None = None,
) -> float | None:
    comparison = (
        {"type": "period_comparison", "years": [year]}
        if year is not None
        else None
    )
    projected = project_periodic_metric_table(
        text,
        metric=metric,
        comparison=comparison,
    )
    if not projected:
        projected = project_periodic_metric_table(text, metric=metric)
    if not projected:
        return None
    values = _metric_row_values(projected)
    if not values:
        return None
    return values[-1]


def compute_cross_domain_ratio(
    aggregate: ExchangeFieldAggregate,
    *,
    config: Mapping[str, Any],
    denominator_texts: Sequence[str],
) -> CrossDomainRatio | None:
    metric = str(config.get("denominator_metric") or "매출액")
    year = config.get("year")
    parsed_year = int(year) if isinstance(year, int) and not isinstance(year, bool) else None
    if parsed_year is None and isinstance(year, str) and year.isdigit():
        parsed_year = int(year)
    numerator_op = str(config.get("numerator_op") or "sum")
    numerator_value = (
        aggregate.amount_average
        if numerator_op == "average"
        else aggregate.amount_sum
    )
    if numerator_value is None:
        return None
    denominator_value = None
    for text in denominator_texts:
        denominator_value = extract_periodic_metric_amount(
            text,
            metric=metric,
            year=parsed_year,
        )
        if denominator_value is not None:
            break
    if denominator_value in (None, 0):
        return None
    ratio_percent = numerator_value / denominator_value * 100.0
    return CrossDomainRatio(
        numerator_op=numerator_op,
        numerator_value=numerator_value,
        denominator_metric=metric,
        denominator_value=denominator_value,
        ratio_percent=ratio_percent,
        year=parsed_year,
    )


def cross_domain_ratio_statement(ratio: CrossDomainRatio) -> str:
    numerator_label = "건당 평균 계약금액" if ratio.numerator_op == "average" else "계약금액 합계"
    year_label = f"{ratio.year}년 " if ratio.year else ""
    return (
        f"{year_label}{numerator_label}(파생) {format_aggregate_amount(ratio.numerator_value)}은 "
        f"연결 {ratio.denominator_metric} {format_aggregate_amount(ratio.denominator_value)} "
        f"대비 {ratio.ratio_percent:.2f}%(파생)입니다."
    )
