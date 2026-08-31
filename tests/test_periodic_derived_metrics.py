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

BALANCE_TABLE = "\n".join(
    [
        "| 구분 | 전기 | 당기 |",
        "| --- | --- | --- |",
        "| 부채총계 | 100,000 | 120,000 |",
        "| 자본총계 | 200,000 | 240,000 |",
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

    def test_balance_ratio_projection(self) -> None:
        display = project_derived_metric_display(
            BALANCE_TABLE,
            metric="부채총계",
            request={"derived_metric": "balance_ratio"},
        )
        self.assertIsNotNone(display)
        assert display is not None
        self.assertIn("부채비율(파생)", display)
        self.assertIn("전기", display)
        self.assertIn("당기", display)
        self.assertIn("50%", display)

    def test_quarter_compare_projection(self) -> None:
        table = "\n".join(
            [
                "| 구분 | 2023년 1분기 | 2024년 1분기 |",
                "| --- | --- | --- |",
                "| 매출액 | 100,000 | 120,000 |",
            ]
        )
        display = project_derived_metric_display(
            table,
            metric="매출액",
            comparison={"type": "period_comparison", "years": [2023, 2024]},
            request={
                "derived_metric": "quarter_compare",
                "period": {"quarter": 1, "period_type": "fiscal_quarter"},
            },
            raw_query="2023년 1분기와 2024년 1분기 중 어느 분기가 더 크고 차이는?",
        )
        self.assertIsNotNone(display)
        assert display is not None
        self.assertIn("분기 비교(파생)", display)
        self.assertIn("더 큰 쪽", display)
        self.assertIn("20000", display.replace(",", ""))


if __name__ == "__main__":
    unittest.main()
