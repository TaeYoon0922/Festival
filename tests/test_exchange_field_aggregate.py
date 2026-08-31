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

    def test_aggregate_by_receipt_year(self) -> None:
        texts = [
            "투자금액 100억원",
            "투자금액 200억원",
            "투자금액 50억원",
        ]
        doc_ids = [
            "exchange_20240115800001",
            "exchange_20250115800002",
            "exchange_20250120800003",
        ]
        result = aggregate_exchange_amounts(
            texts,
            "investment_amount",
            ops=("sum",),
            doc_ids=doc_ids,
            years=(2024, 2025),
        )
        self.assertEqual(len(result.by_year), 2)
        by_year = {item.year: item.amount_sum for item in result.by_year}
        self.assertEqual(by_year[2024], 10_000_000_000.0)
        self.assertEqual(by_year[2025], 25_000_000_000.0)

    def test_aggregate_max_by_year(self) -> None:
        texts = [
            "계약금액 100,000,000원",
            "계약금액 300,000,000원",
            "계약금액 200,000,000원",
        ]
        doc_ids = [
            "exchange_20240115800001",
            "exchange_20250115800002",
            "exchange_20250120800003",
        ]
        result = aggregate_exchange_amounts(
            texts,
            "contract_amount",
            ops=("max",),
            doc_ids=doc_ids,
            years=(2024, 2025),
        )
        self.assertEqual(result.amount_max, 300_000_000.0)
        by_year = {item.year: item.amount_max for item in result.by_year}
        self.assertEqual(by_year[2024], 100_000_000.0)
        self.assertEqual(by_year[2025], 300_000_000.0)

    def test_parse_contract_quantity(self) -> None:
        parsed = parse_labeled_amount("우선협상수량 120", "contract_quantity")
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed.value, 120.0)


if __name__ == "__main__":
    unittest.main()
