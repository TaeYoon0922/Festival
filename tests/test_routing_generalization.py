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
    ROOT / "app/reasoning/holding_event_aggregate.py",
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
            aggregate = plan.evidence.get("exchange_aggregate") or {}
            ratio = plan.evidence.get("cross_domain_ratio") or {}
            signatures.append(
                (
                    plan.event_type,
                    aggregate.get("field"),
                    tuple(aggregate.get("ops") or ()),
                    ratio.get("numerator_op"),
                    ratio.get("denominator_metric"),
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

    def test_sp04_equity_compare_flag_is_company_agnostic(self) -> None:
        template = (
            "{corp} {year_a}년과 {year_b}년에 공시한 신규시설투자 금액 합계를 "
            "비교하고, {year_b}년 합계가 자기자본 대비 더 큰지?"
        )
        flags = []
        for corp in ("LG에너지솔루션", "SK이노베이션", "가상2차전지"):
            plan = QueryUnderstanding(_aliases(corp)).understand(
                template.format(corp=corp, year_a=2024, year_b=2025)
            )
            year_compare = plan.evidence.get("exchange_year_compare") or {}
            flags.append(bool(year_compare.get("equity_compare")))
        self.assertEqual(set(flags), {True})

    def test_sp09_quarter_timeseries_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2024년 1·2·3분기 연결 매출액 추이에서 "
            "가장 높은 분기와 최저 분기 차이는?"
        )
        values = []
        for corp in ("SK텔레콤", "KT", "가상통신사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            values.append(plan.evidence.get("derived_metric"))
        self.assertEqual(set(values), {"quarter_timeseries"})

    def test_sp10_quarter_sum_vs_annual_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2024년 4개 분기 연결 영업이익 합계와 "
            "2024 사업보고서 연결 영업이익이 같은지, 차이 있으면 얼마야?"
        )
        values = []
        for corp in ("크래프톤", "엔씨소프트", "가상게임사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            values.append(plan.evidence.get("derived_metric"))
        self.assertEqual(set(values), {"quarter_sum_vs_annual"})
        peer_template = (
            "크래프톤과 시프트업 2024년 4분기 영업이익 합계와 각 사 2024 "
            "사업보고서 영업이익을 나란히 비교해 차이는?"
        )
        peer_plan = QueryUnderstanding(
            {**_aliases("크래프톤"), **_aliases("시프트업")}
        ).understand(peer_template)
        self.assertEqual(peer_plan.evidence.get("derived_metric"), "quarter_sum_vs_annual")

    def test_sp12_order_backlog_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2025 사업보고서에서 수주잔고(또는 수주실적) "
            "전기 대비 증감액과 증감률"
        )
        signatures: list[tuple] = []
        for corp in ("HD현대중공업", "삼성중공업", "가상조선사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            signatures.append(
                (
                    plan.metric,
                    plan.evidence.get("metric_fallback"),
                    plan.evidence.get("derived_metric"),
                )
            )
        self.assertEqual(len(set(signatures)), 1)
        self.assertEqual(signatures[0][0], "수주잔고")
        self.assertEqual(signatures[0][1], "수주실적")
        self.assertEqual(signatures[0][2], "rate")

    def test_peer_backlog_compare_prefers_periodic_over_exchange_event(self) -> None:
        query = (
            "HD현대중공업과 한화오션 2024 사업보고서 수주잔고(또는 수주실적) "
            "전기 대비 증감률을 비교하면 어느 조선사가 더 높아?"
        )
        plan = QueryUnderstanding(
            {**_aliases("HD현대중공업"), **_aliases("한화오션")}
        ).understand(query)
        self.assertEqual(plan.task_type, "financial_metric")
        self.assertEqual(plan.metric, "수주잔고")
        self.assertEqual(plan.evidence.get("derived_metric"), "peer_rate")

    def test_sp13_max_aggregate_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2024·2025년 원전 관련 exchange 공급계약 건수와 "
            "최대 계약금액 건 비교"
        )
        signatures: list[tuple] = []
        for corp in ("두산에너빌리티", "한전KPS", "가상원전사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            aggregate = plan.evidence.get("exchange_aggregate") or {}
            signatures.append((plan.event_type, tuple(aggregate.get("ops") or ())))
        self.assertEqual(len(set(signatures)), 1)
        self.assertEqual(signatures[0][0], "supply_contract")
        self.assertIn("count", signatures[0][1])
        self.assertIn("max", signatures[0][1])

    def test_sp14_quantity_compare_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2023년 우선협상 수량과 2024년 공시 수주 계약 중 "
            "수량이 더 큰 건은?"
        )
        signatures: list[tuple] = []
        for corp in ("한화에어로스페이스", "LIG넥스원", "가상방산사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            quantity = plan.evidence.get("exchange_quantity_compare") or {}
            signatures.append((plan.event_type, quantity.get("field")))
        self.assertEqual(len(set(signatures)), 1)
        self.assertEqual(signatures[0][0], "supply_contract")
        self.assertEqual(signatures[0][1], "contract_quantity")

    def test_sp17_sign_flip_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2023·2024 사업보고서 연결 당기순이익을 비교해, "
            "2024년이 흑자 전환인지 적자 지속인지(전기·당기 부호 비교)"
        )
        values = []
        for corp in ("HMM", "팬오션", "가상해운사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            values.append(plan.evidence.get("derived_metric"))
        self.assertEqual(set(values), {"sign_flip"})

    def test_sp19_segment_rate_intent_is_company_agnostic(self) -> None:
        template = (
            "{corp} 2024 사업보고서 태양광(또는 Qcells) segment 매출이 "
            "전기 대비 몇 % 변했는지"
        )
        signatures: list[tuple] = []
        for corp in ("한화솔루션", "OCI", "가상에너지사"):
            plan = QueryUnderstanding(_aliases(corp)).understand(template.format(corp=corp))
            signatures.append(
                (
                    plan.evidence.get("derived_metric"),
                    plan.evidence.get("metric_fallback"),
                )
            )
        self.assertEqual(len(set(signatures)), 1)
        self.assertEqual(signatures[0][0], "segment_rate")

    def test_sp20_holding_multi_event_intent_is_company_agnostic(self) -> None:
        template = (
            "{reporter}의 {corp} 보유비율과 풋옵션 행사 후 증감을 합쳐, "
            "순증가 주식수와 증가율(%)을 계산할 수 있어?"
        )
        values = []
        for reporter, corp in (
            ("에스엠", "하이브"),
            ("카카오", "SM엔터"),
            ("가상보고자", "가상피투자사"),
        ):
            plan = QueryUnderstanding(_aliases(reporter, corp)).understand(
                template.format(reporter=reporter, corp=corp)
            )
            values.append(plan.evidence.get("holding_multi_event_compute"))
        self.assertTrue(all(values))

    def test_dual_corp_exchange_backend_filters_drop_doc_subtype(self) -> None:
        from app.reasoning.query_plan import QueryPlan

        plan = QueryPlan(
            query="LG에너지솔루션과 삼성SDI 2024년 신규시설투자 공시",
            companies=("LG에너지솔루션", "삼성SDI"),
            corp_codes=("01515323", "00126362"),
            years=(2024,),
            disclosure_route=("exchange",),
            doc_subtype="신규시설투자등",
            comparison={"type": "company_comparison", "companies": ["A", "B"]},
        )
        filters = plan.backend_filters()
        self.assertEqual(filters["doc_group"], "exchange")
        self.assertIsNone(filters["doc_subtype"])
        self.assertIsNone(filters["company"])
        self.assertEqual(filters["corp_code"], ["01515323", "00126362"])

    def test_quarter_sum_vs_annual_backend_filters_allow_annual_and_quarter(self) -> None:
        from app.reasoning.query_plan import QueryPeriod, QueryPlan

        plan = QueryPlan(
            query="크래프톤과 시프트업 2024년 4분기 영업이익 합계와 사업보고서 영업이익",
            companies=("크래프톤", "시프트업"),
            corp_codes=("00760971", "01384787"),
            years=(2024,),
            period=QueryPeriod(year=2024, quarter=4, period_type="fiscal_quarter"),
            doc_subtype="quarter",
            evidence={"derived_metric": "quarter_sum_vs_annual"},
        )
        filters = plan.backend_filters()
        self.assertIsNone(filters["doc_subtype"])
        self.assertIsNone(filters["period"])

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
