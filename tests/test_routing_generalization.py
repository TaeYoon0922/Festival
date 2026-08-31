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
