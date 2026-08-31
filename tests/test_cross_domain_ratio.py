from __future__ import annotations

import unittest

from app.reasoning.cross_domain_ratio import (
    compute_cross_domain_ratio,
    cross_domain_ratio_statement,
    extract_periodic_metric_amount,
)
from app.reasoning.exchange_field_aggregate import ExchangeFieldAggregate


TABLE = "\n".join(
    [
        "| 구분 | 2024년 |",
        "| --- | --- |",
        "| 매출액 | 1,000,000 |",
    ]
)


class CrossDomainRatioTests(unittest.TestCase):
    def test_extract_periodic_metric_amount(self) -> None:
        value = extract_periodic_metric_amount(TABLE, metric="매출액", year=2024)
        self.assertEqual(value, 1_000_000.0)

    def test_compute_ratio_from_aggregate(self) -> None:
        aggregate = ExchangeFieldAggregate(
            field="contract_amount",
            document_count=2,
            parsed_count=2,
            amount_sum=200_000.0,
        )
        ratio = compute_cross_domain_ratio(
            aggregate,
            config={
                "denominator_metric": "매출액",
                "year": 2024,
                "numerator_op": "sum",
            },
            denominator_texts=[TABLE],
        )
        self.assertIsNotNone(ratio)
        assert ratio is not None
        self.assertAlmostEqual(ratio.ratio_percent, 20.0)
        self.assertIn("20.00%", cross_domain_ratio_statement(ratio))


if __name__ == "__main__":
    unittest.main()
