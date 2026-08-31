from __future__ import annotations

import unittest

from app.reasoning.exchange_recent_pair import (
    build_recent_pair_compare,
    receipt_date_from_doc_id,
    recent_pair_statement,
    select_recent_documents,
)


class ExchangeRecentPairTests(unittest.TestCase):
    def test_receipt_date_from_doc_id(self) -> None:
        self.assertEqual(
            receipt_date_from_doc_id("exchange_20240521800037"),
            "2024-05-21",
        )

    def test_select_recent_documents(self) -> None:
        doc_ids = [
            "exchange_20230227800485",
            "exchange_20240521800037",
            "exchange_20240813800252",
        ]
        self.assertEqual(
            select_recent_documents(doc_ids, limit=2),
            [
                "exchange_20240813800252",
                "exchange_20240521800037",
            ],
        )

    def test_build_recent_pair_compare(self) -> None:
        compare = build_recent_pair_compare(
            [
                (
                    "exchange_20240813800252",
                    "투자금액: 200억원\n자기자본: 1,000억원",
                ),
                (
                    "exchange_20240521800037",
                    "투자금액: 150억원",
                ),
            ],
            field="investment_amount",
            include_equity_ratio=True,
        )
        self.assertIsNotNone(compare)
        assert compare is not None
        self.assertGreater(compare.newer_amount.value, compare.older_amount.value)
        self.assertIsNotNone(compare.newer_equity_ratio_pct)
        statement = recent_pair_statement(compare)
        self.assertIn("자기자본 대비(파생)", statement)
        self.assertIn("직전 공시 대비(파생)", statement)
        self.assertIn("증가", statement)


if __name__ == "__main__":
    unittest.main()
