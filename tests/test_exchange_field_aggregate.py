from __future__ import annotations

import unittest

from app.reasoning.exchange_field_aggregate import (
    aggregate_exchange_amounts,
    parse_labeled_amount,
)


class ExchangeFieldAggregateTests(unittest.TestCase):
    def test_parse_contract_amount_from_inline_text(self) -> None:
        parsed = parse_labeled_amount(
            "계약금액 120,000,000,000원 계약기간 2024-02-15 ~ 2026-06-30",
            "contract_amount",
        )
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.value, 120_000_000_000.0)

    def test_aggregate_sum_and_average(self) -> None:
        texts = [
            "계약금액 100,000,000원",
            "계약금액 300,000,000원",
        ]
        result = aggregate_exchange_amounts(
            texts, "contract_amount", ops=("sum", "average")
        )
        self.assertEqual(result.parsed_count, 2)
        self.assertEqual(result.amount_sum, 400_000_000.0)
        self.assertEqual(result.amount_average, 200_000_000.0)


if __name__ == "__main__":
    unittest.main()
