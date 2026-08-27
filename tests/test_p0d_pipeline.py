from __future__ import annotations

import json
import unittest

from app.api.pipeline import AnswerPipeline
from app.api.schemas import AnswerResponse
from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.multi_document_planner import MultiDocumentPlanner
from app.reasoning.query_plan import QueryPlan, QueryPeriod
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope, QueryState, QueryValidator
from app.reasoning.semantic_query_fallback import HcxSemanticQueryFallback
from tests.test_agent_end_to_end_smoke import _StaticExecutor, _StaticUnderstanding, _execution
from tests.test_evidence_builder import _holding_pair


class _Understanding:
    def __init__(self, plan: QueryPlan) -> None:
        self.plan = plan

    def understand(self, question: str, *, top_k: int):
        del question, top_k
        return self.plan


class _Executor:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, plan):
        del plan
        self.calls += 1
        raise AssertionError("retrieval must be blocked")


class _Transport:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def post_json(self, url, *, headers, payload, timeout_seconds):
        del url, headers, payload, timeout_seconds
        self.calls += 1
        return {"choices": [{"message": {"content": self.content}}]}


class _EventScope:
    def __init__(self, bounds=("2021-01-01", "2026-06-30")) -> None:
        self.bounds = bounds
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.bounds


def _validator() -> QueryValidator:
    return QueryValidator(
        corpus_scope=CorpusScope(
            companies={
                "삼성중공업": ("삼성중공업", "00126478"),
                "효성중공업": ("효성중공업", "01316245"),
            },
            fiscal_years=(2023, 2024, 2025, 2026),
            receipt_from="2023-01-02",
            receipt_to="2026-06-19",
            event_from="2022-01-01",
            event_to="2026-06-30",
        )
    )


def _pipeline(plan: QueryPlan, *, semantic_fallback=None):
    executor = _Executor()
    pipeline = AnswerPipeline(
        understanding=_Understanding(plan),
        executor=executor,
        query_validator=_validator(),
        semantic_fallback=semantic_fallback,
        answerability_guard=AnswerabilityGuard(),
    )
    return pipeline, executor


class RetrievalFirewallTests(unittest.TestCase):
    def test_all_non_resolved_states_make_zero_retrieval_calls(self) -> None:
        cases = (
            QueryPlan(
                query="취소",
                raw_query="삼성중공업이 작년에 취소한 건 있어?",
                company="삼성중공업",
                task_type="disclosure_lookup",
                evidence={"operation": "lookup_disclosure"},
            ),
            QueryPlan(
                query="공급계약",
                raw_query="그 회사 공급계약 알려줘",
                task_type="corporate_event",
                event_type="supply_contract",
                evidence={"operation": "inspect_event"},
            ),
            QueryPlan(
                query="예측",
                raw_query="삼성중공업 주가를 예측해줘",
                company="삼성중공업",
                task_type="disclosure_lookup",
                evidence={"operation": "lookup_disclosure"},
            ),
            QueryPlan(
                query="매출액",
                raw_query="애플 2025년 매출액",
                company="애플",
                task_type="financial_metric",
                metric="매출액",
                period=QueryPeriod(year=2025, period_type="fiscal_year"),
                evidence={"operation": "lookup_metric"},
            ),
        )

        states = []
        for index, plan in enumerate(cases):
            pipeline, executor = _pipeline(plan)
            payload = pipeline.answer(f"P0D-{index}", plan.raw_query)
            validated = AnswerResponse.model_validate(payload).model_dump()
            states.append(validated["think_trace"]["query_understanding"]["status"])
            self.assertEqual(executor.calls, 0)
            self.assertEqual(validated["retrieved_context"], [])
            self.assertEqual(validated["think_trace"]["retrieval_count"], 0)
            self.assertIsNotNone(validated["think_trace"]["query_validation"])
            self.assertFalse(
                validated["think_trace"]["query_validation"]["retrieval_allowed"]
            )

        self.assertEqual(
            states,
            [
                QueryState.AMBIGUOUS.value,
                QueryState.INCOMPLETE.value,
                QueryState.UNSUPPORTED.value,
                QueryState.OUT_OF_SCOPE.value,
            ],
        )

    def test_malformed_hcx_is_called_once_then_clarifies(self) -> None:
        plan = QueryPlan(
            query="취소",
            raw_query="삼성중공업이 작년에 취소한 건 있어?",
            company="삼성중공업",
            task_type="disclosure_lookup",
            evidence={"operation": "lookup_disclosure"},
        )
        transport = _Transport("not-json")
        fallback = HcxSemanticQueryFallback(
            HcxSettings(
                enabled=True,
                endpoint="https://clova.example/v1/chat/completions",
                api_key="test-key",
            ),
            transport=transport,
        )
        pipeline, executor = _pipeline(plan, semantic_fallback=fallback)

        payload = pipeline.answer("P0D-HCX", plan.raw_query)

        self.assertEqual(transport.calls, 1)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(executor.calls, 0)
        self.assertTrue(
            payload["think_trace"]["query_understanding"]["clarification_required"]
        )
        self.assertEqual(
            payload["think_trace"]["query_understanding"]["hcx_fallback_status"],
            "malformed_response",
        )
        diagnostic = payload["think_trace"]["query_understanding"][
            "hcx_diagnostic"
        ]
        self.assertEqual(diagnostic["transport_status"], "success")
        self.assertEqual(
            diagnostic["response_shape"],
            "choices[0].message.content:string",
        )
        self.assertTrue(diagnostic["content_present"])
        self.assertEqual(diagnostic["parse_status"], "invalid_json")

    def test_schema_invalid_hcx_exposes_sanitized_reason_and_blocks_retrieval(self) -> None:
        plan = QueryPlan(
            query="취소",
            raw_query="삼성중공업이 작년에 취소한 건 있어?",
            company="삼성중공업",
            task_type="disclosure_lookup",
            evidence={"operation": "lookup_disclosure"},
        )
        content = json.dumps(
            {
                "task_type": "corporate_event",
                "event_family": "contract_termination",
                "operation": "unsupported_operation_value",
                "set_intent": False,
                "requested_state": None,
                "ambiguity": False,
                "possible_interpretations": [],
            }
        )
        transport = _Transport(content)
        fallback = HcxSemanticQueryFallback(
            HcxSettings(
                enabled=True,
                endpoint="https://clova.example/v1/chat/completions",
                api_key="test-key",
            ),
            transport=transport,
        )
        pipeline, executor = _pipeline(plan, semantic_fallback=fallback)

        payload = AnswerResponse.model_validate(
            pipeline.answer("P0D-HCX-SCHEMA", plan.raw_query)
        ).model_dump()

        diagnostic = payload["think_trace"]["query_understanding"][
            "hcx_diagnostic"
        ]
        self.assertEqual(transport.calls, 1)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(payload["retrieved_context"], [])
        self.assertEqual(diagnostic["parse_status"], "schema_invalid")
        self.assertEqual(diagnostic["schema_error_code"], "invalid_enum")
        self.assertEqual(diagnostic["schema_error_fields"], ["operation"])
        self.assertNotIn("unsupported_operation_value", str(payload))

    def test_valid_hcx_wrapper_is_parsed_and_revalidated(self) -> None:
        plan = QueryPlan(
            query="취소",
            raw_query="삼성중공업이 작년에 취소한 건 있어?",
            company="삼성중공업",
            task_type="disclosure_lookup",
            evidence={"operation": "lookup_disclosure"},
        )
        content = json.dumps(
            {
                "task_type": "corporate_event",
                "event_family": "contract_termination",
                "operation": "find_terminated",
                "set_intent": False,
                "requested_state": None,
                "ambiguity": False,
                "possible_interpretations": [],
            }
        )
        transport = _Transport(f"```json\n{content}\n```")
        fallback = HcxSemanticQueryFallback(
            HcxSettings(
                enabled=True,
                endpoint="https://clova.example/v1/chat/completions",
                api_key="test-key",
            ),
            transport=transport,
        )
        pipeline, executor = _pipeline(plan, semantic_fallback=fallback)

        validated_plan, validation = pipeline._validated_understanding(
            plan.raw_query
        )

        self.assertEqual(executor.calls, 0)
        self.assertIs(validation.state, QueryState.RESOLVED)
        self.assertEqual(validated_plan.event_type, "contract_termination")
        self.assertEqual(validation.fallback_status, "success")
        self.assertEqual(validation.hcx_diagnostic["parse_status"], "success")
        self.assertEqual(
            validation.hcx_diagnostic["response_shape"],
            "choices[0].message.content:string",
        )

    def test_reference_year_before_corpus_blocks_retrieval(self) -> None:
        plan = QueryPlan(
            query="계약",
            raw_query="삼성중공업의 2021년 계약을 알려줘",
            company="삼성중공업",
            period=QueryPeriod(year=2021, period_type="reference_year"),
            task_type="disclosure_lookup",
            evidence={"operation": "lookup_disclosure"},
        )
        pipeline, executor = _pipeline(plan)

        payload = pipeline.answer("P0D-REFERENCE-2021", plan.raw_query)

        self.assertEqual(executor.calls, 0)
        self.assertEqual(payload["retrieved_context"], [])
        self.assertEqual(payload["think_trace"]["retrieval_count"], 0)
        self.assertEqual(
            payload["think_trace"]["query_understanding"]["status"],
            "out_of_scope",
        )
        self.assertEqual(
            payload["think_trace"]["query_validation"],
            {
                "status": "out_of_scope",
                "retrieval_allowed": False,
                "reason": "period_out_of_corpus",
            },
        )
        self.assertEqual(
            payload["answer"],
            "현재 제공된 공시 데이터 범위에서는 2021년 내용을 확인할 수 없습니다.",
        )
        self.assertNotIn("계약이 있습니다", payload["answer"])

    def test_repository_runtime_path_does_not_widen_generic_reference_year(self) -> None:
        scope = CorpusScope.repository_default()
        self.assertIsNotNone(scope)
        assert scope is not None
        self.assertEqual(scope.receipt_from, "2023-01-02")
        self.assertEqual(scope.receipt_to, "2026-06-19")
        resolved = scope.resolve_company("삼성중공업")
        self.assertIsNotNone(resolved)
        assert resolved is not None

        def company_resolver(question: str):
            if "삼성중공업" not in question:
                return None
            return {
                "corp_name": resolved[0],
                "listed_name": resolved[0],
                "corp_code": resolved[1],
            }

        event_scope = _EventScope()
        executor = _Executor()
        pipeline = AnswerPipeline(
            understanding=QueryUnderstanding(
                scope.company_aliases(), company_resolver=company_resolver
            ),
            executor=executor,
            query_validator=QueryValidator(
                corpus_scope=scope,
                multi_document_planner=MultiDocumentPlanner(),
                event_scope_provider=event_scope,
            ),
            answerability_guard=AnswerabilityGuard(),
        )

        payload = pipeline.answer(
            "P0D-RUNTIME-REFERENCE-2021",
            "삼성중공업의 2021년 계약을 알려줘",
        )

        self.assertEqual(event_scope.calls, 0)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(payload["retrieved_context"], [])
        self.assertEqual(
            payload["think_trace"]["query_understanding"]["status"],
            QueryState.OUT_OF_SCOPE.value,
        )
        self.assertEqual(
            payload["think_trace"]["query_validation"],
            {
                "status": "out_of_scope",
                "retrieval_allowed": False,
                "reason": "period_out_of_corpus",
            },
        )
        self.assertEqual(
            payload["answer"],
            "현재 제공된 공시 데이터 범위에서는 2021년 내용을 확인할 수 없습니다.",
        )

    def test_repository_runtime_path_blocks_in_scope_semantically_incomplete_contract(self) -> None:
        scope = CorpusScope.repository_default()
        self.assertIsNotNone(scope)
        assert scope is not None
        resolved = scope.resolve_company("삼성중공업")
        self.assertIsNotNone(resolved)
        assert resolved is not None

        def company_resolver(question: str):
            if "삼성중공업" not in question:
                return None
            return {
                "corp_name": resolved[0],
                "listed_name": resolved[0],
                "corp_code": resolved[1],
            }

        event_scope = _EventScope()
        executor = _Executor()
        pipeline = AnswerPipeline(
            understanding=QueryUnderstanding(
                scope.company_aliases(), company_resolver=company_resolver
            ),
            executor=executor,
            query_validator=QueryValidator(
                corpus_scope=scope,
                multi_document_planner=MultiDocumentPlanner(),
                event_scope_provider=event_scope,
            ),
            answerability_guard=AnswerabilityGuard(),
        )

        payload = pipeline.answer(
            "P0D-RUNTIME-REFERENCE-2025",
            "삼성중공업의 2025년 계약을 알려줘",
        )

        understanding = payload["think_trace"]["query_understanding"]
        self.assertEqual(understanding["status"], QueryState.AMBIGUOUS.value)
        self.assertIn("event_family", understanding["ambiguous_slots"])
        self.assertIn("date_basis", understanding["ambiguous_slots"])
        self.assertFalse(
            payload["think_trace"]["query_validation"]["retrieval_allowed"]
        )
        self.assertEqual(executor.calls, 0)
        self.assertEqual(payload["retrieved_context"], [])
        self.assertEqual(event_scope.calls, 0)

    def test_in_scope_contract_hcx_runs_once_but_cannot_guess_date_basis(self) -> None:
        scope = CorpusScope.repository_default()
        self.assertIsNotNone(scope)
        assert scope is not None
        resolved = scope.resolve_company("삼성중공업")
        self.assertIsNotNone(resolved)
        assert resolved is not None

        def company_resolver(question: str):
            if "삼성중공업" not in question:
                return None
            return {
                "corp_name": resolved[0],
                "listed_name": resolved[0],
                "corp_code": resolved[1],
            }

        content = json.dumps(
            {
                "task_type": "corporate_event",
                "event_family": "supply_contract",
                "operation": "inspect_event",
                "set_intent": None,
                "requested_state": None,
                "ambiguity": False,
                "possible_interpretations": [],
            }
        )
        transport = _Transport(content)
        fallback = HcxSemanticQueryFallback(
            HcxSettings(
                enabled=True,
                endpoint="https://clova.example/v1/chat/completions",
                api_key="test-key",
            ),
            transport=transport,
        )
        executor = _Executor()
        pipeline = AnswerPipeline(
            understanding=QueryUnderstanding(
                scope.company_aliases(), company_resolver=company_resolver
            ),
            executor=executor,
            query_validator=QueryValidator(
                corpus_scope=scope,
                multi_document_planner=MultiDocumentPlanner(),
                event_scope_provider=_EventScope(),
            ),
            semantic_fallback=fallback,
            answerability_guard=AnswerabilityGuard(),
        )

        payload = pipeline.answer(
            "P0D-RUNTIME-HCX-REFERENCE-2025",
            "삼성중공업의 2025년 계약을 알려줘",
        )

        understanding = payload["think_trace"]["query_understanding"]
        self.assertEqual(transport.calls, 1)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(payload["retrieved_context"], [])
        self.assertEqual(understanding["status"], QueryState.AMBIGUOUS.value)
        self.assertEqual(understanding["hcx_fallback_status"], "success")
        self.assertEqual(understanding["ambiguous_slots"], ["date_basis"])
        self.assertEqual(
            [
                option["id"]
                for option in understanding["clarification"]["options"]
            ],
            ["contract_date", "receipt_date"],
        )

    def test_clarification_metadata_is_ui_safe(self) -> None:
        plan = QueryPlan(
            query="취소",
            raw_query="삼성중공업이 작년에 취소한 건 있어?",
            company="삼성중공업",
            task_type="disclosure_lookup",
            evidence={"operation": "lookup_disclosure"},
        )
        pipeline, _ = _pipeline(plan)

        payload = pipeline.answer("P0D-UI", plan.raw_query)
        clarification = payload["think_trace"]["query_understanding"]["clarification"]

        self.assertEqual(
            [option["id"] for option in clarification["options"]],
            ["contract_termination", "investment_cancellation"],
        )
        serialized = str(payload["think_trace"])
        for forbidden in ("system_prompt", "chain_of_thought", "sql", "password"):
            self.assertNotIn(forbidden, serialized.casefold())


class ResolvedPipelineTests(unittest.TestCase):
    def test_clear_query_skips_semantic_hcx_and_reports_answerability(self) -> None:
        question = "효성중공업 국민연금기금 변동일 변동후 주식수"
        plan = QueryPlan(
            query=question,
            raw_query=question,
            company="효성중공업",
            corp_code="01316245",
            task_type="holding_change",
            metric="holding_shares",
            reporter="국민연금기금",
            disclosure_route=("holding",),
            evidence={
                "operation": "lookup_holding",
                "requested_holding_fields": ["reference_date", "after_shares"],
            },
        )
        first = _holding_pair(
            "h23:ch_report",
            "h23",
            rank=1,
            date="2023-06-30",
            projection_type="holding_report",
            table_id="t23",
        )
        execution = _execution(plan, first)
        semantic_transport = _Transport("{}")
        semantic = HcxSemanticQueryFallback(
            HcxSettings(
                enabled=True,
                endpoint="https://clova.example/v1/chat/completions",
                api_key="test-key",
            ),
            transport=semantic_transport,
        )
        executor = _StaticExecutor(plan, execution)
        pipeline = AnswerPipeline(
            understanding=_StaticUnderstanding(question, plan),
            executor=executor,
            verbalizer=HcxVerbalizer(HcxSettings(enabled=False)),
            query_validator=_validator(),
            semantic_fallback=semantic,
            answerability_guard=AnswerabilityGuard(),
        )

        payload = pipeline.answer("P0D-CLEAR", question)

        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(semantic_transport.calls, 0)
        self.assertEqual(semantic.call_count, 0)
        self.assertEqual(
            payload["think_trace"]["query_understanding"]["status"], "resolved"
        )
        self.assertEqual(
            payload["think_trace"]["answerability"]["status"], "answerable"
        )
        self.assertEqual(
            payload["think_trace"]["query_validation"],
            {"status": "valid", "retrieval_allowed": True},
        )
        self.assertEqual(
            set(payload),
            {"question_id", "question", "retrieved_context", "think_trace", "answer"},
        )
        self.assertEqual(pipeline.query_metrics["deterministic_resolved_count"], 1)


if __name__ == "__main__":
    unittest.main()
