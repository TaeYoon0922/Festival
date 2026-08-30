import inspect
import unittest

from app.agent.task_router import route_task
from app.reasoning import (
    QueryExecutor,
    QueryPeriod,
    QueryPlan,
    QueryRouter,
    QueryUnderstanding,
)
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataBackend,
    MetadataMatch,
    RetrievalResult,
)
from app.retrieval.postgres_backend import PostgresBackend


class RecordingBackend:
    def __init__(self) -> None:
        self.filters = None
        self.retrieval = None

    def get_candidate_documents(self, **filters):
        self.filters = filters
        return [CandidateDocument("d1", {}, MetadataMatch())]

    def get_candidate_chunks(self, documents):
        self.asserted_documents = list(documents)
        return [CandidateChunk("c1", "d1", {"content": "evidence"}, MetadataMatch())]

    def retrieve(self, query, candidates, *, top_k=None):
        self.retrieval = (query, list(candidates), top_k)
        return [RetrievalResult("c1", "d1", 1.0, 1, {})]


class QueryPlanTests(unittest.TestCase):
    def test_plan_maps_to_existing_backend_contract(self) -> None:
        plan = QueryPlan(
            query="매출액",
            raw_query="삼성전자 2024년 1분기 매출액",
            company="삼성전자",
            corp_code="00126380",
            period=QueryPeriod(
                year=2024, quarter=1, period_type="fiscal_quarter"
            ),
            task_type="financial_metric",
            metric="매출액",
            disclosure_route=("periodic",),
            doc_subtype="quarter",
            basis="consolidated",
            correction_policy="original_only",
            section_path="매출",
            top_k=5,
        )
        self.assertEqual(
            plan.backend_filters(),
            {
                "company": ["삼성전자"],
                "year": [2024],
                "period": 3,
                "doc_group": "periodic",
                "doc_subtype": "quarter",
                "is_correction": False,
                "corp_code": ["00126380"],
                "section_path": "매출",
            },
        )

    def test_executor_sends_only_lexical_query_to_retriever(self) -> None:
        backend = RecordingBackend()
        plan = QueryPlan(
            query="매출액",
            raw_query="고려아연 2024년 매출액",
            corp_code="001",
            top_k=3,
        )
        execution = QueryExecutor(backend).execute(plan)

        self.assertEqual(backend.filters["corp_code"], ["001"])
        self.assertEqual(backend.retrieval[0], "매출액")
        self.assertEqual(backend.retrieval[2], 3)
        self.assertEqual(execution.results[0].chunk_id, "c1")

    def test_invalid_values_fail_before_backend_calls(self) -> None:
        with self.assertRaises(ValueError):
            QueryPlan(query=" ")
        with self.assertRaises(ValueError):
            QueryPlan(query="valid", top_k=0)
        with self.assertRaises(ValueError):
            QueryPeriod(year=2024, quarter=5, period_type="fiscal_quarter")
        with self.assertRaises(ValueError):
            QueryPlan(query="valid", basis="unknown")

    def test_filter_keys_match_protocol_and_postgres_signatures(self) -> None:
        plan_keys = set(QueryPlan(query="valid").backend_filters())
        protocol_keys = set(
            inspect.signature(MetadataBackend.get_candidate_documents).parameters
        ) - {"self"}
        postgres_keys = set(
            inspect.signature(PostgresBackend.get_candidate_documents).parameters
        ) - {"self"}
        self.assertEqual(plan_keys, protocol_keys)
        self.assertEqual(plan_keys, postgres_keys)


class QueryUnderstandingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.aliases = {
            "고려아연": {"고려아연"},
            "삼성전자": {"삼성전자"},
            "NAVER": {"네이버"},
        }

    def test_business_product_intent_preserves_terms_and_adds_section_boosts(self) -> None:
        understanding = QueryUnderstanding(
            {
                **self.aliases,
                "레인보우로보틱스": {"레인보우로보틱스"},
                "한미반도체": {"한미반도체"},
            }
        )
        cases = (
            ("삼성전자 DX 부문의 주요 제품은 무엇인가", "DX"),
            ("레인보우로보틱스 HUBO 이족보행 로봇 사업 설명", "HUBO"),
            ("한미반도체 TC BONDER 반도체 제조 장비 사업", "TC BONDER"),
        )
        for query, retained_term in cases:
            with self.subTest(query=query):
                plan = understanding.understand(query)
                self.assertEqual(plan.evidence["periodic_intent"], "business_product")
                self.assertIn(retained_term, plan.lexical_query)
                self.assertEqual(plan.section_boosts["사업의 개요"], 1.0)
                self.assertEqual(plan.section_boosts["주요 제품 및 서비스"], 1.0)
                self.assertIsNone(plan.backend_filters()["section_path"])

    def test_financial_metric_lexical_query_strips_metric_particle(self) -> None:
        plan = QueryUnderstanding({"현대자동차": {"현대자동차"}}).understand(
            "현대자동차 2025년 1분기 별도 매출액은?"
        )

        self.assertEqual(plan.task_type, "financial_metric")
        self.assertEqual(plan.metric, "매출액")
        self.assertEqual(plan.basis, "standalone")
        self.assertEqual(plan.lexical_query, "매출액")

    def test_listing_history_intent_uses_company_overview_soft_boost(self) -> None:
        plan = QueryUnderstanding({"시프트업": {"시프트업"}}).understand(
            "시프트업 유가증권시장 상장일"
        )
        route = QueryRouter().route(plan)

        self.assertEqual(plan.evidence["periodic_intent"], "listing_history")
        self.assertEqual(plan.section_boosts, {"회사의 개요": 1.0})
        self.assertEqual(route.section_boosts, plan.section_boosts)
        self.assertNotIn("section_path", route.hard_routes)

    def test_periodic_intent_section_relevance_is_soft_and_discriminative(self) -> None:
        plan = QueryUnderstanding({"한미반도체": {"한미반도체"}}).understand(
            "한미반도체 TC BONDER 반도체 제조 장비 사업"
        )
        router = QueryRouter()
        route = router.route(plan)
        relevant = router.deterministic_components(
            route,
            chunk={"section_path": ["II. 사업의 내용", "1. 사업의 개요"]},
            metadata_match={},
        )
        unrelated = router.deterministic_components(
            route,
            chunk={"section_path": ["III. 재무에 관한 사항", "재무제표 주석"]},
            metadata_match={},
        )

        self.assertEqual(relevant["section"], 1.0)
        self.assertEqual(unrelated["section"], 0.0)
        self.assertNotIn("section_path", route.hard_routes)

    def test_merger_history_intent_keeps_existing_ambiguous_routes(self) -> None:
        plan = QueryUnderstanding({"시프트업": {"시프트업"}}).understand(
            "시프트업 테이블원 흡수합병 합병기일"
        )

        self.assertEqual(plan.evidence["periodic_intent"], "merger_history")
        self.assertEqual(plan.task_type, "corporate_event")
        self.assertEqual(plan.disclosure_route, ("major", "periodic"))
        self.assertEqual(plan.section_boosts["회사의 연혁"], 1.0)

    def test_periodic_intent_does_not_fire_for_unrelated_or_other_routes(self) -> None:
        understanding = QueryUnderstanding(
            {
                "고려아연": {"고려아연"},
                "파마리서치": {"파마리서치"},
            }
        )
        unrelated = understanding.understand("고려아연 감사의견 조회")
        holding = understanding.understand(
            "파마리서치 국민연금 2022년 12월 5일 현재 보유 비율"
        )
        holding_overlap = understanding.understand(
            "파마리서치 주요 제품 관련 국민연금 보유 비율"
        )
        major = understanding.understand("고려아연 유상증자 신주 수")
        exchange = understanding.understand("고려아연 단일판매 공급계약 금액")
        major_overlap = understanding.understand("고려아연 유상증자 장비 사업 설명")
        exchange_overlap = understanding.understand(
            "고려아연 장비 사업 단일판매 공급계약"
        )

        self.assertIsNone(unrelated.evidence["periodic_intent"])
        self.assertEqual(unrelated.section_boosts, {})
        self.assertEqual(holding.disclosure_route, ("holding",))
        self.assertIsNone(holding.evidence["periodic_intent"])
        self.assertEqual(holding_overlap.disclosure_route, ("holding",))
        self.assertIsNone(holding_overlap.evidence["periodic_intent"])
        self.assertEqual(holding_overlap.section_boosts["보유 주식"], 1.0)
        self.assertNotIn("사업의 개요", holding_overlap.section_boosts)
        self.assertEqual(major.disclosure_route, ("major",))
        self.assertIsNone(major.evidence["periodic_intent"])
        self.assertEqual(exchange.disclosure_route, ("exchange",))
        self.assertIsNone(exchange.evidence["periodic_intent"])
        self.assertEqual(major_overlap.disclosure_route, ("major",))
        self.assertIsNone(major_overlap.evidence["periodic_intent"])
        self.assertEqual(major_overlap.section_boosts, {})
        self.assertEqual(exchange_overlap.disclosure_route, ("exchange",))
        self.assertIsNone(exchange_overlap.evidence["periodic_intent"])
        self.assertEqual(exchange_overlap.section_boosts, {})

    def test_fixed_korea_zinc_revenue_regression_plan(self) -> None:
        plan = QueryUnderstanding(self.aliases).understand(
            "고려아연 2024년 매출액"
        )

        self.assertEqual(plan.company, "고려아연")
        self.assertEqual(plan.base_year, 2024)
        self.assertEqual(plan.period.period_type, "fiscal_year")
        self.assertEqual(plan.task_type, "financial_metric")
        self.assertEqual(plan.metric, "매출액")
        self.assertEqual(plan.disclosure_route, ("periodic",))
        self.assertEqual(plan.basis, "unspecified")
        self.assertEqual(plan.raw_query, "고려아연 2024년 매출액")
        self.assertEqual(plan.lexical_query, "매출액")
        self.assertEqual(plan.evidence["report_preference"], "annual")
        self.assertIsNone(plan.backend_filters()["period"])
        self.assertIsNone(plan.backend_filters()["doc_subtype"])

        route = QueryRouter().route(plan)
        self.assertEqual(route.hard_filters["company"], ["고려아연"])
        self.assertEqual(route.hard_filters["year"], [2024])
        self.assertNotIn("doc_group", route.hard_routes)
        self.assertEqual(route.soft_boosts["doc_group"], "periodic")
        self.assertEqual(route.ranking_context["task_type"], "financial_metric")
        self.assertEqual(route.ranking_context["metric"], "매출액")
        self.assertEqual(route.ranking_context["period_type"], "fiscal_year")

    def test_company_prefix_fallback_handles_regression_without_aliases(self) -> None:
        plan = QueryUnderstanding().understand("고려아연 2024년 매출액")
        self.assertEqual(plan.company, "고려아연")
        self.assertEqual(plan.lexical_query, "매출액")

    def test_net_income_query_expands_statement_row_aliases(self) -> None:
        plan = QueryUnderstanding().understand(
            "현대자동차 2025년 1분기 연결 당기순이익"
        )

        self.assertEqual(plan.task_type, "financial_metric")
        self.assertEqual(plan.metric, "당기순이익")
        self.assertIn("당기순이익", plan.lexical_query)
        self.assertIn("분기순이익", plan.lexical_query)
        self.assertIn("연결분기순이익", plan.lexical_query)

    def test_receipt_year_is_not_a_fiscal_year_filter(self) -> None:
        plan = QueryUnderstanding(self.aliases).understand(
            "고려아연이 2024년에 공시한 유상증자"
        )
        self.assertEqual(plan.event_type, "capital_increase")
        self.assertEqual(plan.period.period_type, "receipt_date")
        self.assertEqual(plan.period.from_date, "2024-01-01")
        self.assertEqual(plan.period.to_date, "2024-12-31")
        self.assertIsNone(plan.backend_filters()["year"])
        self.assertEqual(
            QueryRouter().route(plan).hard_filters["rcept_dt"],
            ("2024-01-01", "2024-12-31"),
        )

    def test_holding_query_extracts_reporter_metric_and_before_after(self) -> None:
        plan = QueryUnderstanding(self.aliases).understand(
            "NAVER 국민연금 2024년 보유비율 변동 전후"
        )
        self.assertEqual(plan.company, "네이버")
        self.assertEqual(plan.task_type, "holding_change")
        self.assertEqual(plan.metric, "holding_ratio")
        self.assertEqual(plan.reporter, "국민연금")
        self.assertEqual(plan.disclosure_route, ("holding",))
        self.assertEqual(plan.comparison, {"type": "before_after"})
        self.assertEqual(plan.period.from_date, "2024-01-01")

    def test_executive_ownership_report_routes_to_holding_not_periodic(self) -> None:
        aliases = {**self.aliases, "기아": {"기아"}}
        query = (
            "기아의 가장 최근 임원ㆍ주요주주특정증권등소유상황보고서에서 "
            "특수관계자가 한 명 더 늘어난 이유는?"
        )
        plan = QueryUnderstanding(aliases).understand(query)
        decision = route_task(query, plan)

        self.assertEqual(plan.company, "기아")
        self.assertEqual(plan.task_type, "holding_change")
        self.assertEqual(plan.disclosure_route, ("holding",))
        self.assertEqual(plan.period.period_type, "latest_holding")
        self.assertEqual(decision.task_type, "holding_event")
        self.assertEqual(decision.resolver_type, "holding_event_resolver")

    def test_holding_reference_dates_do_not_become_receipt_filters(self) -> None:
        queries = (
            "파마리서치 국민연금 2022년 12월 5일 현재 보유 비율",
            "파마리서치 국민연금 2023년 3월 10일 기준 보유 주식수",
            "파마리서치 국민연금 변동일 2024년 1월 2일 보유 비율",
            "파마리서치 국민연금 2024년 1월 2일 현재 보유 비율",
            "파마리서치 국민연금 기준일 현재 보유 비율",
            "파마리서치 국민연금 보유일 기준 보유 주식수",
        )
        for query in queries:
            with self.subTest(query=query):
                plan = QueryUnderstanding().understand(query)
                route = QueryRouter().route(plan)
                self.assertEqual(plan.task_type, "holding_change")
                self.assertNotIn("rcept_dt", route.hard_filters)
                self.assertIsNone(route.date_range)
                self.assertEqual(
                    plan.evidence["date_semantics"]["role"],
                    "holding_reference",
                )

    def test_explicit_disclosure_dates_keep_receipt_filtering(self) -> None:
        cases = (
            (
                "2023년에 공시한 국민연금 보고서",
                ("2023-01-01", "2023-12-31"),
            ),
            (
                "파마리서치 2023년 제출된 보고서",
                ("2023-01-01", "2023-12-31"),
            ),
            (
                "LS ELECTRIC 2024년 3월 공시 신규시설투자",
                ("2024-03-01", "2024-03-31"),
            ),
            (
                "두산로보틱스 2024년 1월 2일 접수된 유상증자 보고서",
                ("2024-01-02", "2024-01-02"),
            ),
        )
        for query, expected in cases:
            with self.subTest(query=query):
                plan = QueryUnderstanding().understand(query)
                route = QueryRouter().route(plan)
                self.assertEqual(plan.period.period_type, "receipt_date")
                self.assertEqual(route.hard_filters["rcept_dt"], expected)
                self.assertEqual(plan.evidence["date_semantics"]["role"], "receipt")

    def test_ambiguous_merger_keeps_multiple_routes(self) -> None:
        plan = QueryUnderstanding(self.aliases).understand("삼성전자 흡수합병 합병기일")
        self.assertEqual(plan.disclosure_route, ("major", "periodic"))
        route = QueryRouter().route(plan)
        self.assertFalse(route.hard_routes)
        self.assertEqual(route.route_candidates["major"], 0.58)
        self.assertEqual(route.route_candidates["periodic"], 0.42)

    def test_explicit_basis_is_a_hard_route_but_unspecified_is_not(self) -> None:
        consolidated = QueryUnderstanding(self.aliases).understand(
            "삼성전자 2024년 연결 기준 매출액"
        )
        unspecified = QueryUnderstanding(self.aliases).understand(
            "삼성전자 2024년 매출액"
        )
        self.assertEqual(consolidated.basis, "consolidated")
        self.assertEqual(QueryRouter().route(consolidated).hard_routes["basis"], "consolidated")
        self.assertNotIn("basis", QueryRouter().route(unspecified).hard_routes)

    def test_correction_is_soft_unless_user_requests_only_or_exclusion(self) -> None:
        preferred = QueryUnderstanding(self.aliases).understand(
            "삼성전자 2024년 정정 매출액"
        )
        only = QueryUnderstanding(self.aliases).understand(
            "삼성전자 2024년 정정 공시만 매출액"
        )
        original = QueryUnderstanding(self.aliases).understand(
            "삼성전자 2024년 정정 제외 매출액"
        )
        self.assertEqual(preferred.correction_policy, "latest_preferred")
        self.assertEqual(QueryRouter().route(preferred).soft_boosts["is_correction"], True)
        self.assertEqual(QueryRouter().route(only).hard_routes["is_correction"], True)
        self.assertEqual(QueryRouter().route(original).hard_routes["is_correction"], False)

    def test_backend_company_resolution_adds_corp_code(self) -> None:
        calls = []

        def resolver(query):
            calls.append(query)
            return {
                "corp_code": "00126380",
                "corp_name": "삼성전자",
                "listed_name": "삼성전자",
            }

        plan = QueryUnderstanding(company_resolver=resolver).understand(
            "삼성전자 3분기 매출"
        )
        self.assertEqual(calls, ["삼성전자 3분기 매출"])
        self.assertEqual(plan.corp_code, "00126380")
        self.assertEqual(plan.company, "삼성전자")
        self.assertEqual(plan.period.quarter, 3)
        self.assertEqual(plan.lexical_query, "매출액")

    def test_cumulative_period_column_word_is_not_kept_as_search_noise(self) -> None:
        plan = QueryUnderstanding(self.aliases).understand(
            "삼성전자 2025년 1분기 누적 연결 매출액"
        )

        self.assertEqual(plan.metric, "매출액")
        self.assertEqual(plan.period.year, 2025)
        self.assertEqual(plan.period.quarter, 1)
        self.assertEqual(plan.basis, "consolidated")
        self.assertEqual(plan.lexical_query, "매출액")
        self.assertIn("누적", plan.raw_query)

    def test_balance_sheet_metric_expands_spaced_statement_label(self) -> None:
        plan = QueryUnderstanding(self.aliases).understand(
            "삼성전자 2024년 사업보고서 연결 자산총계는 얼마야?"
        )

        self.assertEqual(plan.metric, "자산총계")
        self.assertIn("자산 총 계", plan.lexical_query)
        self.assertIn("자 산 총 계", plan.lexical_query)
        self.assertEqual(plan.section_boosts["첨부연결재무제표"], 1.0)

    def test_multi_year_financial_comparison_keeps_all_explicit_fiscal_years(self) -> None:
        plan = QueryUnderstanding(self.aliases).understand(
            "삼성전자 2022년부터 2024년까지 매출액 추이"
        )
        self.assertEqual(plan.years, (2022, 2023, 2024))
        self.assertEqual(plan.backend_filters()["year"], [2022, 2023, 2024])
        self.assertEqual(
            plan.comparison,
            {"type": "period_comparison", "years": [2022, 2023, 2024]},
        )

    def test_missing_period_uses_task_specific_non_excluding_policy(self) -> None:
        financial = QueryUnderstanding(self.aliases).understand("삼성전자 매출액")
        holding = QueryUnderstanding(self.aliases).understand("삼성전자 보유비율")
        event = QueryUnderstanding(self.aliases).understand("삼성전자 유상증자")
        latest_event = QueryUnderstanding(self.aliases).understand("삼성전자 최근 유상증자")
        self.assertEqual(financial.period.period_type, "latest_valid_periodic")
        self.assertEqual(holding.period.period_type, "latest_holding")
        self.assertIsNone(event.period.period_type)
        self.assertEqual(latest_event.period.period_type, "latest_event")
        self.assertIsNone(financial.backend_filters()["year"])

    def test_facility_investment_with_cash_on_hand_stays_an_exchange_event(self) -> None:
        understanding = QueryUnderstanding(
            {
                "고려아연": {"고려아연"},
                "엘에스일렉트릭": {"엘에스일렉트릭"},
                "LS ELECTRIC": {"엘에스일렉트릭"},
                "삼성전자": {"삼성전자"},
            }
        )
        queries = (
            "고려아연이 최근 공시한 신규시설투자 금액은 자기자본 대비 어느 정도 수준이며, "
            "현재 보유 중인 현금성 자산으로 자체 조달이 가능한가요?",
            "엘에스일렉트릭이 최근 공시한 신규시설투자 금액은 자기자본 대비 비율이며 "
            "현재 보유 중인 현금성 자산으로 자체 조달이 가능한가",
            "삼성전자 최근 공시 시설투자 금액과 자기자본 대비 수준",
        )
        for query in queries:
            with self.subTest(query=query):
                plan = understanding.understand(query)
                route = QueryRouter().route(plan)
                decision = route_task(query, plan)

                self.assertEqual(plan.task_type, "corporate_event")
                self.assertEqual(plan.event_type, "facility_investment")
                self.assertIsNone(plan.metric)
                self.assertEqual(plan.disclosure_route, ("exchange",))
                self.assertEqual(plan.doc_subtype, "신규시설투자등")
                self.assertEqual(plan.period.period_type, "latest_event")
                self.assertIsNone(plan.comparison)
                self.assertEqual(route.hard_routes["doc_group"], "exchange")
                self.assertEqual(route.hard_routes["event_type"], "facility_investment")
                self.assertEqual(route.soft_boosts.get("doc_subtype"), "신규시설투자등")
                self.assertNotIn("doc_subtype", route.hard_routes)
                self.assertEqual(decision.task_type, "corporate_event")
                self.assertIsNone(decision.resolver_type)

    def test_supply_contract_with_sales_ratio_stays_an_exchange_event(self) -> None:
        plan = QueryUnderstanding({"한국항공우주": {"한국항공우주"}}).understand(
            "한국항공우주 최근 공급계약 금액과 매출액 대비 비율은?"
        )
        route = QueryRouter().route(plan)

        self.assertEqual(plan.task_type, "corporate_event")
        self.assertEqual(plan.event_type, "supply_contract")
        self.assertIsNone(plan.metric)
        self.assertEqual(plan.disclosure_route, ("exchange",))
        self.assertEqual(route.hard_routes["doc_group"], "exchange")
        self.assertEqual(route.hard_routes["event_type"], "supply_contract")

    def test_treasury_share_trust_termination_is_a_major_event(self) -> None:
        plan = QueryUnderstanding({"하나금융지주": {"하나금융지주"}}).understand(
            "하나금융지주 자기주식 취득 신탁계약 해지 내용 알려줘"
        )
        route = QueryRouter().route(plan)

        self.assertEqual(plan.task_type, "corporate_event")
        self.assertEqual(plan.event_type, "treasury_share_trust_termination")
        self.assertEqual(plan.disclosure_route, ("major",))
        self.assertEqual(
            plan.section_boosts["자기주식취득 신탁계약 해지 결정"], 1.0
        )
        self.assertEqual(route.hard_routes["event_type"], "treasury_share_trust_termination")

    def test_write_down_capital_security_is_a_major_event(self) -> None:
        plan = QueryUnderstanding().understand(
            "상각형 조건부자본증권 발행결정 최근 공시 금액은?"
        )
        route = QueryRouter().route(plan)

        self.assertEqual(plan.task_type, "corporate_event")
        self.assertEqual(plan.event_type, "write_down_contingent_capital_security")
        self.assertEqual(plan.disclosure_route, ("major",))
        self.assertEqual(
            plan.section_boosts["상각형 조건부자본증권 발행결정"], 1.0
        )
        self.assertEqual(route.hard_routes["event_type"], "write_down_contingent_capital_security")

    def test_share_holding_questions_are_not_reclassified_as_facility_investment(
        self,
    ) -> None:
        plan = QueryUnderstanding({"파마리서치": {"파마리서치"}}).understand(
            "파마리서치 국민연금 2022년 12월 5일 현재 보유 비율"
        )
        self.assertEqual(plan.task_type, "holding_change")
        self.assertEqual(plan.disclosure_route, ("holding",))
        self.assertIsNone(plan.event_type)
        self.assertIsNone(plan.doc_subtype)


class BoundedOwnershipIntentTests(unittest.TestCase):
    HOLDER = "가상투자사"
    ISSUER = "가상발행사"
    THIRD = "가상제삼사"

    def setUp(self) -> None:
        self.understanding = QueryUnderstanding(
            {
                name: {name}
                for name in (self.HOLDER, self.ISSUER, self.THIRD)
            }
        )

    def plan(self, query: str) -> QueryPlan:
        return self.understanding.understand(query)

    def assert_bounded_holding(
        self,
        query: str,
        *,
        metric: str | None,
        family: str,
    ) -> QueryPlan:
        plan = self.plan(query)
        self.assertEqual(plan.task_type, "holding_change")
        self.assertEqual(plan.metric, metric)
        self.assertEqual(plan.disclosure_route, ("holding",))
        self.assertEqual(len(plan.companies), 2)
        self.assertIsNone(plan.company)
        self.assertEqual(plan.evidence["holding_ownership_intent"], family)
        return plan

    def test_closed_two_company_ownership_families_activate(self) -> None:
        cases = (
            (
                f"{self.HOLDER}가 보유한 {self.ISSUER} 주식은 몇 주인가?",
                "holding_shares",
                "company_holds_company_shares",
            ),
            (
                f"{self.HOLDER}가 {self.ISSUER} 주식을 얼마나 들고 있어?",
                None,
                "company_has_company_shares",
            ),
            (
                f"{self.HOLDER}의 {self.ISSUER} 지분은 얼마나 되나?",
                None,
                "company_ownership_interest",
            ),
        )
        for query, metric, family in cases:
            with self.subTest(query=query):
                self.assert_bounded_holding(query, metric=metric, family=family)

    def test_existing_strong_holding_forms_keep_their_metrics(self) -> None:
        cases = (
            (f"{self.ISSUER} {self.HOLDER} 보유주식수", "holding_shares"),
            (f"{self.ISSUER} {self.HOLDER} 보유비율", "holding_ratio"),
            (
                f"{self.ISSUER} {self.HOLDER} 대량보유상황보고서 내용",
                "holding_shares",
            ),
        )
        for query, metric in cases:
            with self.subTest(query=query):
                plan = self.plan(query)
                self.assertEqual(plan.task_type, "holding_change")
                self.assertEqual(plan.metric, metric)
                self.assertEqual(plan.disclosure_route, ("holding",))
                self.assertIsNone(plan.evidence["holding_ownership_intent"])

    def test_new_rule_assigns_only_explicit_metrics(self) -> None:
        shares = self.assert_bounded_holding(
            f"{self.HOLDER}가 보유한 {self.ISSUER} 주식은 몇 주야?",
            metric="holding_shares",
            family="company_holds_company_shares",
        )
        ratio = self.assert_bounded_holding(
            f"{self.HOLDER}의 {self.ISSUER} 지분 비율은 몇 %야?",
            metric="holding_ratio",
            family="company_ownership_interest",
        )
        ambiguous = self.assert_bounded_holding(
            f"{self.HOLDER}가 {self.ISSUER} 주식을 얼마나 들고 있어?",
            metric=None,
            family="company_has_company_shares",
        )

        self.assertEqual(shares.evidence["metric"], "몇 주")
        self.assertEqual(ratio.evidence["metric"], "지분 비율")
        self.assertIsNone(ambiguous.evidence["metric"])

    def test_new_ownership_reference_date_keeps_the_exact_holding_axis(self) -> None:
        plan = self.assert_bounded_holding(
            f"{self.HOLDER}가 보유한 {self.ISSUER} 주식은 "
            "2024년 2월 3일 기준 몇 주인가?",
            metric="holding_shares",
            family="company_holds_company_shares",
        )

        self.assertEqual(plan.period.period_type, "holding_reference_date")
        self.assertEqual(plan.period.from_date, "2024-02-03")
        self.assertEqual(plan.period.to_date, "2024-02-03")
        self.assertEqual(plan.evidence["date_semantics"]["role"], "holding_reference")
        self.assertEqual(
            plan.evidence["holding_report_relative"]["selector"],
            "exact_reference_date",
        )
        self.assertIsNone(plan.backend_filters()["year"])

    def test_new_ownership_receipt_date_keeps_the_exact_receipt_axis(self) -> None:
        plan = self.assert_bounded_holding(
            f"{self.HOLDER}가 보유한 {self.ISSUER} 주식을 "
            "2024년 2월 3일 접수된 보고서 기준 몇 주인가?",
            metric="holding_shares",
            family="company_holds_company_shares",
        )

        self.assertEqual(plan.period.period_type, "receipt_date")
        self.assertEqual(plan.period.from_date, "2024-02-03")
        self.assertEqual(plan.period.to_date, "2024-02-03")
        self.assertEqual(plan.evidence["date_semantics"]["role"], "receipt")
        self.assertEqual(
            plan.evidence["holding_report_relative"]["selector"],
            "exact_receipt_date",
        )

    def test_bare_and_non_security_language_does_not_activate(self) -> None:
        queries = (
            f"{self.ISSUER} 지분은 얼마야?",
            f"{self.ISSUER} 현재 보유 현황 알려줘",
            f"{self.HOLDER}가 {self.ISSUER} 계약서를 들고 있어?",
        )
        for query in queries:
            with self.subTest(query=query):
                plan = self.plan(query)
                self.assertEqual(plan.task_type, "disclosure_lookup")
                self.assertIsNone(plan.evidence["holding_ownership_intent"])

    def test_comparison_firewall_blocks_the_new_ownership_rule(self) -> None:
        cross_company = self.plan(
            f"{self.HOLDER}와 {self.ISSUER} 중 누가 주식을 더 많이 들고 있어?"
        )
        uncertain = self.plan(
            f"{self.HOLDER}가 보유한 {self.ISSUER} 주식을 "
            "더 많이 취득한 시점은 언제야?"
        )

        self.assertEqual(
            cross_company.evidence["comparison_frame"], "cross_company"
        )
        self.assertIsNone(cross_company.evidence["holding_ownership_intent"])
        self.assertEqual(uncertain.evidence["comparison_frame"], "uncertain")
        self.assertEqual(uncertain.task_type, "disclosure_lookup")
        self.assertIsNone(uncertain.evidence["holding_ownership_intent"])

    def test_explicit_holding_comparison_keeps_existing_comparison_semantics(self) -> None:
        plan = self.plan(f"{self.HOLDER}와 {self.ISSUER} 보유비율 비교")

        self.assertEqual(plan.task_type, "holding_change")
        self.assertEqual(plan.metric, "holding_ratio")
        self.assertEqual(plan.evidence["comparison_frame"], "cross_company")
        self.assertEqual(plan.comparison["type"], "company_comparison")
        self.assertIsNone(plan.evidence["holding_ownership_intent"])

    def test_three_companies_do_not_activate_the_new_rule(self) -> None:
        plan = self.plan(
            f"{self.HOLDER}가 보유한 {self.ISSUER} 주식은 몇 주이고 "
            f"{self.THIRD}도 알려줘"
        )

        self.assertEqual(len(plan.companies), 3)
        self.assertEqual(plan.task_type, "disclosure_lookup")
        self.assertIsNone(plan.evidence["holding_ownership_intent"])

    def test_explicit_non_holding_disclosure_or_event_blocks_the_new_rule(self) -> None:
        event = self.plan(
            f"{self.HOLDER}가 보유한 {self.ISSUER} 주식 관련 "
            "신규시설투자 금액 알려줘"
        )
        disclosure = self.plan(
            f"{self.HOLDER}가 보유한 {self.ISSUER} 주식 관련 "
            "사업보고서 내용 알려줘"
        )

        self.assertEqual(event.task_type, "corporate_event")
        self.assertEqual(event.event_type, "facility_investment")
        self.assertEqual(event.disclosure_route, ("exchange",))
        self.assertIsNone(event.evidence["holding_ownership_intent"])
        self.assertEqual(disclosure.task_type, "disclosure_lookup")
        self.assertEqual(disclosure.disclosure_route, ("periodic",))
        self.assertIsNone(disclosure.evidence["holding_ownership_intent"])

    def test_existing_two_company_holding_classes_remain_unchanged(self) -> None:
        cases = (
            (
                f"{self.ISSUER} 대량보유보고에서 {self.HOLDER}가 신고한 "
                "보유주식수는?",
                "holding_shares",
            ),
            (
                f"{self.ISSUER}에 대한 {self.HOLDER}의 보유비율은?",
                "holding_ratio",
            ),
        )
        for query, metric in cases:
            with self.subTest(query=query):
                plan = self.plan(query)
                self.assertEqual(plan.task_type, "holding_change")
                self.assertEqual(plan.metric, metric)
                self.assertEqual(plan.disclosure_route, ("holding",))
                self.assertIsNone(plan.evidence["holding_ownership_intent"])

    def test_existing_holding_period_selectors_are_unchanged(self) -> None:
        cases = (
            ("최신 보고 보유주식수", "latest_holding", "latest"),
            ("현재 기준 보유비율", "holding_reference", "latest"),
            ("이번 보고 보유주식수", "latest_holding", "selected_context"),
        )
        for wording, period_type, selector in cases:
            with self.subTest(wording=wording):
                plan = self.plan(f"{self.ISSUER} {wording}")
                self.assertEqual(plan.period.period_type, period_type)
                self.assertEqual(
                    plan.evidence["holding_report_relative"]["selector"], selector
                )
                self.assertIsNone(plan.evidence["holding_ownership_intent"])


class QueryRouterTests(unittest.TestCase):
    def test_explicit_corporate_event_filters_other_major_event_documents(self) -> None:
        plan = QueryUnderstanding({"삼성전자": {"삼성전자"}}).understand(
            "삼성전자 유상증자 공시 내용"
        )
        router = QueryRouter()
        route = router.route(plan)
        documents = [
            CandidateDocument(
                "rights-offering",
                {
                    "doc_group": "major",
                    "report_nm": "주요사항보고서(유상증자결정)",
                },
                MetadataMatch(),
            ),
            CandidateDocument(
                "treasury-stock",
                {
                    "doc_group": "major",
                    "report_nm": "주요사항보고서(자기주식취득결정)",
                },
                MetadataMatch(),
            ),
        ]

        selected = router.filter_documents(documents, route)

        self.assertEqual(route.hard_routes["event_type"], "capital_increase")
        self.assertEqual([document.doc_id for document in selected], ["rights-offering"])

    def test_explicit_corporate_event_returns_no_unrelated_fallback(self) -> None:
        plan = QueryUnderstanding({"삼성전자": {"삼성전자"}}).understand(
            "삼성전자 유상증자 공시 내용"
        )
        router = QueryRouter()
        route = router.route(plan)
        unrelated = CandidateDocument(
            "treasury-stock",
            {
                "doc_group": "major",
                "report_nm": "주요사항보고서(자기주식취득결정)",
            },
            MetadataMatch(),
        )

        self.assertEqual(router.filter_documents([unrelated], route), [])

    def test_hard_basis_filters_chunks(self) -> None:
        plan = QueryUnderstanding({"삼성전자": {"삼성전자"}}).understand(
            "삼성전자 연결 기준 매출액"
        )
        router = QueryRouter()
        route = router.route(plan)
        chunks = [
            CandidateChunk(
                "connected",
                "d1",
                {"statement_scope": "연결", "section_path": ["손익계산서"]},
                MetadataMatch(),
            ),
            CandidateChunk(
                "standalone",
                "d1",
                {"statement_scope": "별도", "section_path": ["손익계산서"]},
                MetadataMatch(),
            ),
        ]
        selected = router.prepare_chunks(chunks, route)
        self.assertEqual([chunk.chunk_id for chunk in selected], ["connected"])

    def test_standalone_basis_excludes_only_clear_consolidated_chunks(self) -> None:
        plan = QueryUnderstanding({"고려아연": {"고려아연"}}).understand(
            "고려아연 2024년 별도기준 매출액"
        )
        router = QueryRouter()
        route = router.route(plan)
        chunks = [
            *[
                CandidateChunk(
                    f"neutral-{index}",
                    "neutral-doc",
                    {"section_path": ["재무제표 주석", "수익"]},
                    MetadataMatch(),
                )
                for index in range(140)
            ],
            *[
                CandidateChunk(
                    f"standalone-{index}",
                    "standalone-doc",
                    {
                        "statement_scope": "별도",
                        "section_path": ["재무제표", "손익계산서"],
                    },
                    MetadataMatch(),
                )
                for index in range(5)
            ],
            *[
                CandidateChunk(
                    f"consolidated-{index}",
                    "consolidated-doc",
                    {"section_path": ["연결재무제표 주석", "수익"]},
                    MetadataMatch(),
                )
                for index in range(10)
            ],
            CandidateChunk(
                "mixed-1",
                "mixed-doc",
                {
                    "statement_scope": "연결/별도",
                    "section_path": ["요약재무정보"],
                },
                MetadataMatch(),
            ),
            CandidateChunk(
                "mixed-2",
                "mixed-doc",
                {"section_path": ["연결/별도 요약재무정보"]},
                MetadataMatch(),
            ),
        ]

        selected = router.prepare_chunks(chunks, route)
        selected_ids = {chunk.chunk_id for chunk in selected}
        self.assertEqual(len(selected), 147)
        self.assertGreater(len(selected), 133)
        self.assertIn("neutral-0", selected_ids)
        self.assertIn("standalone-0", selected_ids)
        self.assertIn("mixed-1", selected_ids)
        self.assertFalse(any(value.startswith("consolidated-") for value in selected_ids))

    def test_explicit_consolidated_query_keeps_consolidated_top_five(self) -> None:
        plan = QueryUnderstanding({"고려아연": {"고려아연"}}).understand(
            "고려아연 2024년 연결기준 매출액", top_k=5
        )
        router = QueryRouter()
        route = router.route(plan)
        chunks = [
            *[
                CandidateChunk(
                    f"consolidated-{index}",
                    f"c-doc-{index}",
                    {
                        "statement_scope": "연결",
                        "doc_group": "periodic",
                        "section_path": ["연결포괄손익계산서"],
                        "content": "매출액 100억원",
                    },
                    MetadataMatch(),
                )
                for index in range(6)
            ],
            CandidateChunk(
                "standalone",
                "s-doc",
                {
                    "statement_scope": "별도",
                    "section_path": ["재무제표", "손익계산서"],
                    "content": "매출액 100억원",
                },
                MetadataMatch(),
            ),
            CandidateChunk(
                "neutral",
                "n-doc",
                {
                    "section_path": ["재무제표 주석", "수익"],
                    "content": "매출액 100억원",
                },
                MetadataMatch(),
            ),
        ]
        selected = router.prepare_chunks(chunks, route)
        results = [
            RetrievalResult(chunk.chunk_id, chunk.doc_id, 1.0, rank, {})
            for rank, chunk in enumerate(selected, start=1)
        ]
        reranked = router.rerank(results, route, chunks=selected, top_k=5)
        self.assertEqual(len(reranked), 5)
        self.assertTrue(
            all(result.chunk_id.startswith("consolidated-") for result in reranked)
        )

    def test_explicit_standalone_query_keeps_plain_financial_chunks_in_top_five(self) -> None:
        plan = QueryUnderstanding({"고려아연": {"고려아연"}}).understand(
            "고려아연 2024년 별도기준 매출액", top_k=5
        )
        router = QueryRouter()
        route = router.route(plan)
        specs = [
            ("standalone-statement", "별도", ["재무제표", "손익계산서"]),
            ("standalone-note", "별도", ["재무제표 주석", "수익의 인식"]),
            ("plain-statement", None, ["재무제표", "포괄손익계산서"]),
            (
                "plain-note",
                None,
                ["재무제표 주석", "고객과의 계약에서 생기는 수익"],
            ),
            ("plain-income", None, ["손익계산서"]),
            ("mixed-summary", "연결/별도", ["요약재무정보"]),
            ("consolidated-note", "연결", ["연결재무제표 주석", "수익"]),
        ]
        chunks = [
            CandidateChunk(
                chunk_id,
                chunk_id,
                {
                    "statement_scope": scope,
                    "doc_group": "periodic",
                    "section_path": section,
                    "content": "매출액 100억원",
                },
                MetadataMatch(),
            )
            for chunk_id, scope, section in specs
        ]
        selected = router.prepare_chunks(chunks, route)
        results = [
            RetrievalResult(chunk.chunk_id, chunk.doc_id, 1.0, rank, {})
            for rank, chunk in enumerate(selected, start=1)
        ]
        reranked = router.rerank(results, route, chunks=selected, top_k=5)
        top_ids = {result.chunk_id for result in reranked}

        self.assertNotIn("consolidated-note", {chunk.chunk_id for chunk in selected})
        self.assertIn("standalone-statement", top_ids)
        self.assertTrue({"plain-statement", "plain-note", "plain-income"} & top_ids)
        explicit = next(
            result for result in reranked if result.chunk_id == "standalone-statement"
        )
        self.assertEqual(
            explicit.metadata_match["score_components"]["basis_relevance"], 1.0
        )

    def test_soft_correction_preference_is_annotated_without_exclusion(self) -> None:
        plan = QueryUnderstanding({"삼성전자": {"삼성전자"}}).understand(
            "삼성전자 2024년 정정 매출액"
        )
        router = QueryRouter()
        route = router.route(plan)
        chunks = [
            CandidateChunk(
                "corrected", "d1", {"is_correction": True}, MetadataMatch()
            ),
            CandidateChunk(
                "original", "d2", {"is_correction": False}, MetadataMatch()
            ),
        ]
        selected = router.prepare_chunks(chunks, route)
        self.assertEqual(len(selected), 2)
        matches = {
            chunk.chunk_id: chunk.metadata_match.soft_boosts["is_correction"]
            for chunk in selected
        }
        self.assertEqual(matches, {"corrected": True, "original": False})

    def test_revenue_section_boost_promotes_rank_ten_into_top_five(self) -> None:
        plan = QueryUnderstanding({"고려아연": {"고려아연"}}).understand(
            "고려아연 2024년 매출액", top_k=5
        )
        router = QueryRouter()
        route = router.route(plan)
        chunks = []
        results = []
        for index in range(1, 10):
            chunks.append(
                CandidateChunk(
                    f"c{index}",
                    "d1",
                    {
                        "doc_group": "periodic",
                        "section_path": ["II. 사업의 내용", "주요 제품 및 서비스"],
                        "content": "제품 가격 변동",
                    },
                    MetadataMatch(),
                )
            )
            results.append(
                RetrievalResult(f"c{index}", "d1", 1.02 - index * 0.02, index, {})
            )
        chunks.append(
            CandidateChunk(
                "target",
                "d1",
                {
                    "doc_group": "periodic",
                    "section_path": ["II. 사업의 내용", "4. 매출 및 수주상황"],
                    "content": "2024년 매출액 12,345백만원",
                },
                MetadataMatch(),
            )
        )
        results.append(RetrievalResult("target", "d1", 0.82, 10, {}))

        prepared = router.prepare_chunks(chunks, route)
        reranked = router.rerank(results, route, chunks=prepared, top_k=5)
        target = next(result for result in reranked if result.chunk_id == "target")
        self.assertLessEqual(target.rank, 5)
        components = target.metadata_match["score_components"]
        self.assertEqual(components["exact_term"], 1.0)
        self.assertEqual(components["section"], 1.0)
        self.assertIn("final_score", components)

    def test_fiscal_year_revenue_prefers_annual_report_without_filtering_interims(self) -> None:
        plan = QueryUnderstanding({"고려아연": {"고려아연"}}).understand(
            "고려아연 2024년 매출액", top_k=5
        )
        router = QueryRouter()
        route = router.route(plan)
        report_specs = [
            ("q1", "quarter", "분기보고서 (2024.03)", 3),
            ("half", "half", "반기보고서 (2024.06)", 6),
            ("q3", "quarter", "분기보고서 (2024.09)", 9),
            ("q1-note", "quarter", "분기보고서 (2024.03)", 3),
            ("half-note", "half", "반기보고서 (2024.06)", 6),
            ("q3-note", "quarter", "분기보고서 (2024.09)", 9),
            ("annual", "annual", "사업보고서 (2024.12)", 12),
        ]
        chunks = []
        results = []
        document_metadata = {}
        for rank, (doc_id, subtype, report_name, month) in enumerate(
            report_specs, start=1
        ):
            section = (
                ["III. 재무에 관한 사항", "연결포괄손익계산서"]
                if doc_id == "annual"
                else ["III. 재무에 관한 사항", "연결재무제표 주석", "영업이익"]
            )
            chunks.append(
                CandidateChunk(
                    doc_id,
                    doc_id,
                    {
                        "doc_group": "periodic",
                        "section_path": section,
                        "content": "매출액 12,345백만원",
                    },
                    MetadataMatch(),
                )
            )
            results.append(
                RetrievalResult(
                    doc_id,
                    doc_id,
                    1.001 - rank * 0.001,
                    rank,
                    {},
                )
            )
            document_metadata[doc_id] = {
                "doc_group": "periodic",
                "doc_subtype": subtype,
                "report_nm": report_name,
                "base_year": 2024,
                "base_month": month,
            }

        reranked = router.rerank(
            results,
            route,
            chunks=router.prepare_chunks(chunks, route),
            document_metadata=document_metadata,
            top_k=5,
        )
        annual = next(result for result in reranked if result.chunk_id == "annual")
        self.assertLessEqual(annual.rank, 3)
        self.assertEqual(
            annual.metadata_match["score_components"]["period_relevance"], 1.0
        )
        self.assertEqual(annual.metadata_match["score_components"]["section"], 0.98)
        self.assertTrue(any(result.chunk_id != "annual" for result in reranked))

    def test_revenue_section_does_not_boost_unrelated_financial_statement_note(self) -> None:
        plan = QueryUnderstanding({"고려아연": {"고려아연"}}).understand(
            "고려아연 2024년 매출액"
        )
        router = QueryRouter()
        route = router.route(plan)
        chunks = [
            CandidateChunk(
                "unrelated-note",
                "d1",
                {
                    "section_path": ["연결재무제표 주석", "영업이익"],
                    "content": "매출액",
                },
                MetadataMatch(),
            ),
            CandidateChunk(
                "revenue-note",
                "d1",
                {
                    "section_path": [
                        "연결재무제표 주석",
                        "고객과의 계약에서 생기는 수익",
                    ],
                    "content": "매출액",
                },
                MetadataMatch(),
            ),
        ]
        results = [
            RetrievalResult("unrelated-note", "d1", 1.0, 1, {}),
            RetrievalResult("revenue-note", "d1", 1.0, 2, {}),
        ]
        reranked = router.rerank(results, route, chunks=chunks, top_k=2)
        by_id = {result.chunk_id: result for result in reranked}
        self.assertEqual(
            by_id["unrelated-note"].metadata_match["score_components"]["section"],
            0.0,
        )
        self.assertEqual(
            by_id["revenue-note"].metadata_match["score_components"]["section"],
            1.0,
        )
        self.assertEqual(reranked[0].chunk_id, "revenue-note")

    def test_period_relevance_uses_task_metric_and_explicit_quarter(self) -> None:
        plan = QueryUnderstanding({"삼성전자": {"삼성전자"}}).understand(
            "삼성전자 2024년 1분기 자산총계"
        )
        router = QueryRouter()
        route = router.route(plan)
        chunks = [
            CandidateChunk("quarter", "q", {"doc_group": "periodic"}, MetadataMatch()),
            CandidateChunk("annual", "a", {"doc_group": "periodic"}, MetadataMatch()),
        ]
        results = [
            RetrievalResult("annual", "a", 1.0, 1, {}),
            RetrievalResult("quarter", "q", 1.0, 2, {}),
        ]
        reranked = router.rerank(
            results,
            route,
            chunks=chunks,
            document_metadata={
                "q": {"doc_subtype": "quarter", "base_year": 2024, "base_month": 3},
                "a": {"doc_subtype": "annual", "base_year": 2024, "base_month": 12},
            },
            top_k=2,
        )
        by_id = {result.chunk_id: result for result in reranked}
        self.assertEqual(
            by_id["quarter"].metadata_match["score_components"]["period_relevance"],
            1.0,
        )
        self.assertEqual(
            by_id["annual"].metadata_match["score_components"]["period_relevance"],
            0.0,
        )
        self.assertEqual(reranked[0].chunk_id, "quarter")

def _and(name: str) -> str:
    """Attach 와/과 to a company name, picking the form Korean requires."""

    last = name[-1]
    if "가" <= last <= "힣" and (ord(last) - 0xAC00) % 28:
        return name + "과"
    return name + "와"


class ComparisonFrameTests(unittest.TestCase):
    """The cross-company comparison firewall.

    ``comparison_frame`` says whether a question relates several named
    companies to each other.  It is a firewall signal: it decides whether
    reading one company as a disclosure issuer and another as its reporter
    would be unsafe, and says nothing about whether a cross-company comparison
    can be executed.  Companies appear here only as test data -- the classifier
    itself carries no company, question, or corpus-relation special case.
    """

    #: Companies from the corpus universe.  Any pair would do; the rule must
    #: not depend on which one, nor on any relation between them.
    A = "한화오션"
    B = "한화에어로스페이스"
    C = "에스엠"
    D = "하이브"

    def understanding(self) -> QueryUnderstanding:
        return QueryUnderstanding(
            {name: {name} for name in (self.A, self.B, self.C, self.D)}
        )

    def frame(self, query: str):
        return self.understanding().understand(query).evidence["comparison_frame"]

    def comparison(self, query: str):
        return self.understanding().understand(query).comparison

    # ------------------------------------------------------------------ cross
    def test_cross_company_frames_are_recognized(self) -> None:
        for query in (
            # explicit comparison nouns
            f"{_and(self.A)} {self.B} 보유 비율 비교",
            f"{_and(self.A)} {self.B} 보유 주식수를 비교해줘",
            f"{_and(self.A)} {self.B}의 지분율 차이는 얼마야?",
            # the operator taking companies as its operands
            f"{self.A} 대비 {self.B} 지분율",
            f"{self.A}보다 {self.B}의 보유 비율이 높아?",
            # choice frames
            f"{_and(self.A)} {self.B} 중 어디가 보유 주식수가 더 많아?",
            f"{_and(self.A)} {self.B} 중 누가 보유 비율이 더 높아?",
            f"{_and(self.A)} {self.B} 중 어느 회사가 보유 비율이 더 적어?",
            f"{_and(self.A)} {self.B} 어느 쪽이 보유 비율이 더 낮아?",
            f"{_and(self.A)} {self.B} 중 더 많이 보유한 곳은 어디야?",
            # per-company enumeration
            f"{_and(self.A)} {self.B} 각각 보유 주식수 알려줘",
            f"{self.A}, {self.B}의 보유 비율을 각 회사 별로 알려줘",
            f"{_and(self.A)} {self.B} 둘 다 보유 주식수 알려줘",
            f"{_and(self.A)} {self.B} 양사 보유 비율 알려줘",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.frame(query), "cross_company")

    def test_frame_does_not_depend_on_company_order(self) -> None:
        for template in (
            "{x} {y} 중 어디가 보유 비율이 높아?",
            "{x} {y} 중 누가 보유 비율이 높아?",
            "{x} {y} 각각 보유 비율",
            "{x} {y}의 지분율 차이",
        ):
            with self.subTest(template=template):
                self.assertEqual(
                    self.frame(template.format(x=_and(self.A), y=self.B)),
                    self.frame(template.format(x=_and(self.B), y=self.A)),
                )
        for template in ("{x}보다 {y}의 보유 비율이 높아?", "{x} 대비 {y} 보유 비율"):
            with self.subTest(template=template):
                self.assertEqual(
                    self.frame(template.format(x=self.A, y=self.B)),
                    self.frame(template.format(x=self.B, y=self.A)),
                )

    def test_frame_ignores_the_route_the_question_takes(self) -> None:
        """One construction, four routes, one classification."""

        for query in (
            f"{_and(self.A)} {self.B} 중 어디가 보유 주식수가 더 많아?",
            f"{_and(self.A)} {self.B} 중 어디가 2023년 매출액이 더 많아?",
            f"{_and(self.A)} {self.B} 중 어디가 유상증자 규모가 더 커?",
            f"{_and(self.A)} {self.B} 중 어디가 공급계약 금액이 더 커?",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.frame(query), "cross_company")

    def test_three_companies_still_produce_a_frame(self) -> None:
        self.assertEqual(
            self.frame(f"{self.A}, {self.B}, {self.C} 중 보유 비율이 가장 높은 회사는?"),
            "cross_company",
        )

    def test_a_choice_frame_alone_is_enough(self) -> None:
        """No comparative predicate, so the choice frame is the only evidence.

        Without this the frame could be recognized purely from ``더 높아``, and
        losing a choice marker would go unnoticed.
        """

        for query in (
            f"{_and(self.A)} {self.B} 중 어디가 보유 비율이 높아?",
            f"{_and(self.A)} {self.B} 중 누가 보유 비율이 높아?",
            f"{_and(self.A)} {self.B} 중 어느 회사가 보유 비율이 높아?",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.frame(query), "cross_company")

    def test_a_period_reference_alone_resolves_a_comparative_predicate(self) -> None:
        """Two companies and ``더 많이``, held non-comparative by the period.

        The companies are not coordinated and the operator does not take one as
        its operand, so the period reference is the only thing establishing
        that this compares one holding across time rather than two companies.
        """

        for query in (
            f"{self.C}에서 {self.D}가 작년보다 더 많이 보유한 주식수",
            f"{self.C}에서 {self.D}의 보유 주식수가 작년보다 늘었어?",
            f"{self.C}에서 {self.D}의 지분율이 지난해보다 높아졌어?",
        ):
            with self.subTest(query=query):
                self.assertIsNone(self.frame(query))

    # -------------------------------------------------------------- uncertain
    def test_comparative_predicate_without_structure_fails_closed(self) -> None:
        """Comparative wording between two companies that nothing resolves.

        This reads as temporal, but carries no period reference to prove it and
        no structure making the two companies operands of one comparison.
        Neither reading is safe, so the frame declines rather than guessing.
        """

        self.assertEqual(
            self.frame(f"{self.D}가 {self.C} 주식을 더 많이 취득한 시점은 언제야?"),
            "uncertain",
        )

    # ----------------------------------------------------------------- absent
    def test_a_single_company_never_produces_a_frame(self) -> None:
        for query in (
            f"{self.C} 보유 비율이 더 높아진 이유는?",
            f"{self.C} 보유 주식수의 전년 대비 차이는?",
            f"{self.C}의 이전 공시와 비교해줘",
            f"{self.C} 보유 주식수 추이 비교",
        ):
            with self.subTest(query=query):
                self.assertIsNone(self.frame(query))

    def test_period_bound_comparison_is_not_a_company_frame(self) -> None:
        """A comparison anchored to a period compares one subject over time."""

        for query in (
            f"{self.C}에서 {self.D}의 보유 비율 전년 대비 차이",
            f"{self.C}에서 {self.D}가 직전 보고 대비 늘린 주식수",
            f"{self.C} 공시에서 {self.D}의 직전보고 대비 증감",
            f"{self.A}에서 {self.B}의 직전 보고 대비 지분율",
            f"{self.C}의 {self.D} 보유 비율이 이전보다 높아졌어?",
            f"{self.C}에서 {self.D} 지분 변동 전후 비교",
            f"{self.C}에서 {self.D}의 지분율 변화가 얼마나 커?",
            f"{self.C} {self.D} 보유 주식수 추이",
        ):
            with self.subTest(query=query):
                self.assertIsNone(self.frame(query))

    def test_issuer_reporter_questions_are_never_framed(self) -> None:
        """The controls that must survive the firewall untouched."""

        for query in (
            f"{self.C}에서 {self.D}가 보유한 주식수 알려줘",
            f"{self.C} 공시에서 {self.D}의 보유 비율 알려줘",
            f"{self.D}가 {self.C} 주식을 얼마나 보유하고 있어?",
            f"{self.C}에 대한 {self.D}의 지분 변동 알려줘",
            f"{self.A}에서 {self.B}가 보유한 주식수 알려줘",
            f"{self.B}가 {self.A} 주식을 얼마나 보유하고 있어?",
            f"{self.C} {self.D} 이번 보고 보유 주식수와 비율",
            f"{self.C} {self.D} 직전보고 보유주식 수 비율",
        ):
            with self.subTest(query=query):
                self.assertIsNone(self.frame(query))

    def test_juxtaposed_companies_are_not_coordinated(self) -> None:
        """Two names side by side name a subject; they do not compare it."""

        self.assertIsNone(self.frame(f"{self.C} {self.D} 보유주식 증가 수량 증가 비율"))

    # ------------------------------------- the operator, resolved structurally
    def test_the_operator_is_read_from_its_left_operand(self) -> None:
        """``대비``/``보다`` attach to what precedes them, so that decides."""

        self.assertEqual(self.frame(f"{self.A} 대비 {self.B} 보유 비율"), "cross_company")
        self.assertIsNone(self.frame(f"{self.A}에서 {self.B}의 직전 보고 대비 지분율"))
        self.assertEqual(
            self.frame(f"{self.A}보다 {self.B}가 보유 비율이 높아?"), "cross_company"
        )
        self.assertIsNone(self.frame(f"{self.C}의 보유 주식수가 작년보다 얼마나 늘었어?"))

    # -------------------------------------- the frame does not widen execution
    def test_a_frame_does_not_make_a_question_an_executable_comparison(self) -> None:
        """Recognizing the frame must not promote it to ``company_comparison``.

        These questions stay exactly as unanswerable as they were.  The firewall
        records only that they must not be reinterpreted as one company plus a
        role; it claims no ability to answer them.
        """

        for query in (
            f"{_and(self.A)} {self.B} 중 어디가 보유 주식수가 더 많아?",
            f"{_and(self.A)} {self.B} 중 누가 보유 비율이 더 높아?",
            f"{_and(self.A)} {self.B} 각각 보유 주식수 알려줘",
            f"{self.A}보다 {self.B}의 보유 비율이 높아?",
            f"{_and(self.A)} {self.B}의 지분율 차이는 얼마야?",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.frame(query), "cross_company")
                self.assertIsNone(self.comparison(query))

    def test_explicit_comparison_keeps_its_existing_reach(self) -> None:
        for query in (
            f"{_and(self.A)} {self.B} 보유 비율 비교",
            f"{_and(self.A)} {self.B} 보유 주식수를 비교해줘",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.comparison(query)["type"], "company_comparison")

    def test_period_bound_operator_is_no_longer_a_company_comparison(self) -> None:
        """The defect this reopen fixes.

        ``직전 보고 대비`` is how the corpus itself words a change against the
        previous filing.  Reading it as a comparison between the two companies
        the question names made a holding fact request look like something it
        is not.
        """

        for query in (
            f"{self.C}에서 {self.D}가 직전 보고 대비 늘린 주식수",
            f"{self.C} 공시에서 {self.D}의 직전보고 대비 증감",
            f"{self.A}에서 {self.B}의 직전 보고 대비 지분율",
        ):
            with self.subTest(query=query):
                self.assertIsNone(self.comparison(query))

    def test_temporal_comparison_payloads_are_unchanged(self) -> None:
        self.assertEqual(
            self.comparison(f"{self.C}에서 {self.D} 지분 변동 전후 비교"),
            {"type": "before_after"},
        )
        self.assertEqual(
            self.comparison(f"{self.C}에서 {self.D}의 보유 비율 전년 대비 차이"),
            {"type": "year_over_year"},
        )
        self.assertEqual(
            self.comparison(f"{self.C}에서 {self.D}의 지분율 변화가 얼마나 커?"),
            {"type": "trend"},
        )

if __name__ == "__main__":
    unittest.main()
