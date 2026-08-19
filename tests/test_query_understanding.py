import inspect
import unittest

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
        self.assertEqual(financial.period.period_type, "latest_valid_periodic")
        self.assertEqual(holding.period.period_type, "latest_holding")
        self.assertEqual(event.period.period_type, "latest_event")
        self.assertIsNone(financial.backend_filters()["year"])


class QueryRouterTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
