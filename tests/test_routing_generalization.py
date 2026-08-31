from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from app.reasoning.query_understanding import QueryUnderstanding


ROOT = Path(__file__).resolve().parents[1]
ROUTING_MODULES = (
    ROOT / "app/reasoning/query_understanding.py",
    ROOT / "app/reasoning/query_validation.py",
    ROOT / "app/reasoning/multi_document_planner.py",
    ROOT / "app/reasoning/exchange_field_aggregate.py",
    ROOT / "app/reasoning/periodic_derived_metrics.py",
    ROOT / "app/reasoning/cross_domain_ratio.py",
    ROOT / "app/reasoning/exchange_recent_pair.py",
    ROOT / "app/reasoning/periodic_segment_ranking.py",
    ROOT / "app/agent/orchestrator.py",
)

# qa-tool card exemplars — must not appear as literals in routing code.
CARD_COMPANIES = (
    "한미반도체",
    "삼성바이오로직스",
    "현대건설",
    "LG에너지솔루션",
    "NAVER",
    "아모레퍼시픽",
    "LG이노텍",
)


def _aliases(*names: str) -> dict[str, set[str]]:
    return {name: {name} for name in names}


class RoutingGeneralizationTests(unittest.TestCase):
    def test_sp11_cross_domain_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2024년에 공시한 공급계약 금액 합과 "
            "2024 사업보고서 연결 매출액 대비 비율은?"
        )
        signatures: list[tuple[Any, ...]] = []
        for corp in ("삼성바이오로직스", "현대건설", "가상테스트Corp"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            signatures.append(
                (
                    plan.event_type,
                    plan.evidence.get("exchange_aggregate"),
                    plan.evidence.get("cross_domain_ratio"),
                    "periodic" in plan.disclosure_route,
                )
            )
        self.assertEqual(len(set(signatures)), 1)

    def test_sp08_derived_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2023·2024 사업보고서 연결 영업이익을 비교해, "
            "매출액 증가율과 영업이익 증가율 중 어느 쪽이 더 높아?"
        )
        values = []
        for corp in ("NAVER", "크래프톤", "가상상장사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            values.append(plan.evidence.get("derived_metric"))
        self.assertEqual(set(values), {"compare_rates"})

    def test_sp05_balance_ratio_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2024 사업보고서 연결 부채총계와 자본총계로 "
            "부채비율(부채÷자본)을 전기·당기 각각 계산해줘"
        )
        values = []
        for corp in ("POSCO홀딩스", "현대제철", "가상철강사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            values.append(plan.evidence.get("derived_metric"))
        self.assertEqual(set(values), {"balance_ratio"})

    def test_sp06_recent_pair_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 최근 신규시설투자 공시의 투자금액은 자기자본 대비 몇 %이고, "
            "직전 시설투자 공시 대비 금액이 커졌어?"
        )
        signatures: list[tuple[Any, ...]] = []
        for corp in ("고려아연", "풍산", "가상비철금속"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            signatures.append(
                (
                    plan.event_type,
                    plan.date_basis.value,
                    (plan.evidence.get("exchange_recent_pair") or {}).get("limit"),
                )
            )
        self.assertEqual(len(set(signatures)), 1)
        self.assertEqual(signatures[0][0], "facility_investment")
        self.assertEqual(signatures[0][1], "receipt_date")

    def test_sp03_quarter_compare_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} {year_a}년 1분기와 {year_b}년 1분기 연결 매출액 중 "
            "어느 분기가 더 크고 차이는?"
        )
        values = []
        for corp, year_a, year_b in (
            ("레인보우로보틱스", 2023, 2024),
            ("크래프톤", 2023, 2024),
            ("가상로봇사", 2022, 2025),
        ):
            plan = QueryUnderstanding(_aliases(corp)).understand(
                template.format(corp=corp, year_a=year_a, year_b=year_b)
            )
            values.append(plan.evidence.get("derived_metric"))
        self.assertEqual(set(values), {"quarter_compare"})

    def test_sp07_metric_fallback_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2024 사업보고서 연결 보험료수익(또는 영업수익)이 "
            "전기 대비 얼마나 변했는지 증감률까지"
        )
        signatures: list[tuple[Any, ...]] = []
        for corp in ("삼성생명", "한화생명", "가상보험사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            signatures.append(
                (
                    plan.metric,
                    plan.evidence.get("metric_fallback"),
                    plan.evidence.get("derived_metric"),
                )
            )
        self.assertEqual(len(set(signatures)), 1)
        self.assertEqual(signatures[0][0], "보험료수익")
        self.assertEqual(signatures[0][1], "영업수익")

    def test_sp04_year_compare_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} {year_a}년과 {year_b}년에 공시한 신규시설투자 금액 합계를 "
            "비교하고, {year_b}년 합계가 자기자본 대비 더 큰지?"
        )
        signatures: list[tuple[Any, ...]] = []
        for corp in ("LG에너지솔루션", "SK이노베이션", "가상2차전지"):
            plan = QueryUnderstanding(_aliases(corp)).understand(
                template.format(corp=corp, year_a=2024, year_b=2025)
            )
            signatures.append(
                (
                    plan.event_type,
                    bool(plan.evidence.get("exchange_aggregate")),
                    bool(plan.evidence.get("exchange_year_compare")),
                )
            )
        self.assertEqual(len(set(signatures)), 1)
        self.assertEqual(signatures[0][0], "facility_investment")

    def test_routing_modules_do_not_hardcode_card_companies(self) -> None:
        for path in ROUTING_MODULES:
            text = path.read_text(encoding="utf-8")
            for company in CARD_COMPANIES:
                self.assertNotIn(
                    company,
                    text,
                    f"{company} must not be hardcoded in {path.relative_to(ROOT)}",
                )


if __name__ == "__main__":
    unittest.main()
