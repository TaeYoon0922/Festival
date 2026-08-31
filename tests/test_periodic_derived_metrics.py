from __future__ import annotations

import unittest

from app.reasoning.periodic_derived_metrics import project_derived_metric_display


TABLE = "\n".join(
    [
        "| 구분 | 2023년 | 2024년 |",
        "| --- | --- | --- |",
        "| 매출액 | 100,000 | 120,000 |",
        "| 영업이익 | 10,000 | 15,000 |",
    ]
)


class PeriodicDerivedMetricsTests(unittest.TestCase):
    def test_rate_projection(self) -> None:
        display = project_derived_metric_display(
            TABLE,
            metric="매출액",
            request={"derived_metric": "rate"},
        )
        self.assertIsNotNone(display)
        assert display is not None
        self.assertIn("증가율(파생)", display)
        self.assertIn("20", display)

    def test_compare_rates_projection(self) -> None:
        display = project_derived_metric_display(
            TABLE,
            metric="매출액",
            request={"derived_metric": "compare_rates"},
            raw_query="매출 증가율과 영업이익 증가율 중 어느 쪽",
        )
        self.assertIsNotNone(display)
        assert display is not None
        self.assertIn("영업이익 증가율", display)


if __name__ == "__main__":
    unittest.main()
