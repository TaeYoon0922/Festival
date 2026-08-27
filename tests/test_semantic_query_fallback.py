from __future__ import annotations

import json
import unittest

from app.generation.hcx_verbalizer import HcxSettings
from app.reasoning.query_plan import QueryPlan, QueryPeriod
from app.reasoning.query_validation import (
    CorpusScope,
    QueryState,
    QueryValidator,
)
from app.reasoning.semantic_query_fallback import (
    HcxSemanticQueryFallback,
    SEMANTIC_QUERY_SYSTEM_PROMPT,
    SemanticSchemaError,
    parse_semantic_query_result,
)


class _Transport:
    def __init__(self, content: str, *, wrapper: str = "choices") -> None:
        self.content = content
        self.wrapper = wrapper
        self.calls = 0
        self.payload = None

    def post_json(self, url, *, headers, payload, timeout_seconds):
        del url, headers, timeout_seconds
        self.calls += 1
        self.payload = payload
        if self.wrapper == "result":
            return {"result": {"message": {"content": self.content}}}
        return {"choices": [{"message": {"content": self.content}}]}


def _settings() -> HcxSettings:
    return HcxSettings(
        enabled=True,
        endpoint="https://clova.example/v1/chat/completions",
        api_key="test-key",
    )


def _validator() -> QueryValidator:
    return QueryValidator(
        corpus_scope=CorpusScope(
            companies={"삼성중공업": ("삼성중공업", "00126478")},
            fiscal_years=(2023, 2024, 2025, 2026),
        )
    )


def _ambiguous_plan() -> QueryPlan:
    return QueryPlan(
        query="취소",
        raw_query="삼성중공업이 작년에 취소한 건 있어?",
        company="삼성중공업",
        task_type="disclosure_lookup",
        evidence={"operation": "lookup_disclosure"},
    )


def _semantic_payload(**overrides):
    payload = {
        "task_type": "corporate_event",
        "event_family": "contract_termination",
        "operation": "find_terminated",
        "set_intent": False,
        "requested_state": None,
        "ambiguity": False,
        "possible_interpretations": [],
    }
    payload.update(overrides)
    return payload


class SemanticFallbackTests(unittest.TestCase):
    def test_prompt_requires_strict_json_syntax_and_forbids_python_literals(self) -> None:
        self.assertIn("RFC 8259 JSON", SEMANTIC_QUERY_SYSTEM_PROMPT)
        self.assertIn("no prose or Markdown fence", SEMANTIC_QUERY_SYSTEM_PROMPT)
        self.assertIn("double quotes", SEMANTIC_QUERY_SYSTEM_PROMPT)
        self.assertIn("lowercase JSON literals", SEMANTIC_QUERY_SYSTEM_PROMPT)
        self.assertIn("True, False, or None", SEMANTIC_QUERY_SYSTEM_PROMPT)
        self.assertIn("comments, or trailing commas", SEMANTIC_QUERY_SYSTEM_PROMPT)
        self.assertIn('"possible_interpretations":[]', SEMANTIC_QUERY_SYSTEM_PROMPT)
        self.assertIn("Do not add any other top-level key", SEMANTIC_QUERY_SYSTEM_PROMPT)
        self.assertIn("array of at most five", SEMANTIC_QUERY_SYSTEM_PROMPT)

    def test_prompt_skeleton_matches_required_nullable_schema(self) -> None:
        result = parse_semantic_query_result(
            {
                "task_type": None,
                "event_family": None,
                "operation": None,
                "set_intent": None,
                "requested_state": None,
                "ambiguity": False,
                "possible_interpretations": [],
            }
        )

        self.assertIsNone(result.task_type)
        self.assertIsNone(result.event_family)
        self.assertIsNone(result.operation)
        self.assertIsNone(result.set_intent)
        self.assertIsNone(result.requested_state)
        self.assertFalse(result.ambiguity)
        self.assertEqual(result.possible_interpretations, ())

    def test_clear_query_does_not_call_hcx(self) -> None:
        plan = QueryPlan(
            query="매출액",
            raw_query="삼성중공업 2025년 매출액",
            company="삼성중공업",
            task_type="financial_metric",
            metric="매출액",
            period=QueryPeriod(year=2025, period_type="fiscal_year"),
            evidence={"operation": "lookup_metric"},
        )
        validation = _validator().validate(plan)
        transport = _Transport("{}")
        fallback = HcxSemanticQueryFallback(_settings(), transport=transport)

        outcome = fallback.interpret(plan.raw_query, validation)

        self.assertEqual(outcome.status, "not_needed")
        self.assertEqual(transport.calls, 0)
        self.assertEqual(fallback.call_count, 0)

    def test_clear_correction_query_does_not_call_hcx(self) -> None:
        plan = QueryPlan(
            query="단일판매 공급계약체결 계약 내용",
            raw_query=(
                "현대건설의 2023년 6월 26일 단일판매·공급계약체결 공시는 "
                "최종 정정 기준으로 계약 내용이 어떻게 되어 있어?"
            ),
            company="현대건설",
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
            period=QueryPeriod(
                year=2023,
                from_date="2023-06-26",
                to_date="2023-06-26",
                period_type="receipt_date",
            ),
            evidence={
                "operation": "correction_lookup",
                "correction_intent": "latest",
            },
        )
        validation = QueryValidator(
            corpus_scope=CorpusScope(
                companies={"현대건설": ("현대건설", "00164478")},
                receipt_from="2023-01-02",
                receipt_to="2026-06-19",
            )
        ).validate(plan)
        transport = _Transport("{}")
        fallback = HcxSemanticQueryFallback(_settings(), transport=transport)

        outcome = fallback.interpret(plan.raw_query, validation)

        self.assertIs(validation.state, QueryState.RESOLVED)
        self.assertEqual(outcome.status, "not_needed")
        self.assertEqual(transport.calls, 0)
        self.assertEqual(fallback.call_count, 0)

    def test_structured_result_is_revalidated_before_resolution(self) -> None:
        validator = _validator()
        initial = validator.validate(_ambiguous_plan())
        transport = _Transport(
            json.dumps(
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
        )
        fallback = HcxSemanticQueryFallback(_settings(), transport=transport)

        outcome = fallback.interpret(_ambiguous_plan().raw_query, initial)
        result = validator.validate(
            _ambiguous_plan(),
            semantic=outcome.result,
            fallback_used=True,
            fallback_status=outcome.status,
            hcx_elapsed_ms=outcome.elapsed_ms,
        )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(fallback.call_count, 1)
        self.assertIs(result.state, QueryState.RESOLVED)
        self.assertEqual(result.plan.event_type, "contract_termination")
        self.assertEqual(result.plan.disclosure_route, ("exchange",))

    def test_openai_wrapper_with_explicit_json_fence_is_normalized(self) -> None:
        validator = _validator()
        initial = validator.validate(_ambiguous_plan())
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
        transport = _Transport(
            f"분류 결과입니다.\n```json\n{content}\n```\n이상입니다."
        )
        fallback = HcxSemanticQueryFallback(_settings(), transport=transport)

        outcome = fallback.interpret(_ambiguous_plan().raw_query, initial)

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.response_shape, "choices[0].message.content:string")
        self.assertEqual(outcome.content_format, "json_fence")
        self.assertTrue(outcome.prefix_text_present)
        self.assertTrue(outcome.suffix_text_present)
        self.assertEqual(outcome.parse_status, "success")
        self.assertNotIn("response_format", transport.payload)

    def test_native_wrapper_reuses_the_same_content_contract(self) -> None:
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
        fallback = HcxSemanticQueryFallback(
            _settings(), transport=_Transport(content, wrapper="result")
        )

        outcome = fallback.interpret(
            _ambiguous_plan().raw_query,
            _validator().validate(_ambiguous_plan()),
        )

        self.assertTrue(outcome.succeeded)
        self.assertEqual(outcome.response_shape, "result.message.content:string")
        self.assertEqual(outcome.content_format, "json_object")

    def test_locked_deterministic_slot_cannot_be_overwritten(self) -> None:
        plan = QueryPlan(
            query="매출액",
            raw_query="삼성중공업 2025년 매출액",
            company="삼성중공업",
            task_type="financial_metric",
            metric="매출액",
            period=QueryPeriod(year=2025, period_type="fiscal_year"),
            evidence={"operation": "lookup_metric"},
        )

        result = _validator().validate(
            plan,
            semantic={
                "task_type": "corporate_event",
                "event_family": "supply_contract",
                "operation": "inspect_event",
            },
            fallback_used=True,
            fallback_status="success",
        )

        self.assertIs(result.state, QueryState.AMBIGUOUS)
        self.assertEqual(result.plan.task_type, "financial_metric")
        self.assertIn("locked_slot_conflict:task_type", result.issues)

    def test_unknown_enum_and_malformed_json_fail_closed(self) -> None:
        with self.assertRaises(SemanticSchemaError) as raised:
            parse_semantic_query_result(_semantic_payload(operation="made_up"))
        self.assertEqual(raised.exception.code, "invalid_enum")
        self.assertEqual(raised.exception.fields, ("operation",))
        fallback = HcxSemanticQueryFallback(
            _settings(), transport=_Transport("not-json")
        )

        outcome = fallback.interpret(
            _ambiguous_plan().raw_query,
            _validator().validate(_ambiguous_plan()),
        )

        self.assertEqual(outcome.status, "malformed_response")
        self.assertIsNone(outcome.result)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(outcome.transport_status, "success")
        self.assertTrue(outcome.content_present)
        self.assertEqual(outcome.parse_status, "invalid_json")

    def test_schema_rejections_expose_only_sanitized_code_and_fields(self) -> None:
        missing = _semantic_payload()
        missing.pop("operation")
        unexpected = _semantic_payload(extra_reason="not public")
        cases = (
            (missing, "missing_required", ("operation",)),
            (
                _semantic_payload(operation="not_in_vocabulary"),
                "invalid_enum",
                ("operation",),
            ),
            (
                _semantic_payload(set_intent="false"),
                "invalid_type",
                ("set_intent",),
            ),
            (unexpected, "unexpected_field", ("extra_reason",)),
            (
                _semantic_payload(possible_interpretations=[{"reason": "guess"}]),
                "invalid_interpretation_shape",
                ("possible_interpretations",),
            ),
        )

        for payload, code, fields in cases:
            with self.subTest(code=code):
                fallback = HcxSemanticQueryFallback(
                    _settings(), transport=_Transport(json.dumps(payload))
                )

                outcome = fallback.interpret(
                    _ambiguous_plan().raw_query,
                    _validator().validate(_ambiguous_plan()),
                )
                diagnostic = outcome.diagnostic()

                self.assertEqual(outcome.status, "malformed_response")
                self.assertEqual(outcome.parse_status, "schema_invalid")
                self.assertEqual(outcome.schema_error_code, code)
                self.assertEqual(outcome.schema_error_fields, fields)
                self.assertEqual(diagnostic["schema_error_code"], code)
                self.assertEqual(diagnostic["schema_error_fields"], list(fields))
                self.assertNotIn("not_in_vocabulary", str(diagnostic))
                self.assertNotIn("not public", str(diagnostic))
                self.assertNotIn("guess", str(diagnostic))

    def test_fenced_python_literals_match_live_failure_and_fail_closed(self) -> None:
        transport = _Transport(
            "```json\n"
            "{'task_type': 'corporate_event', "
            "'event_family': 'contract_termination', "
            "'operation': 'find_terminated', 'set_intent': False, "
            "'requested_state': None, 'ambiguity': True, "
            "'possible_interpretations': []}\n"
            "```"
        )
        fallback = HcxSemanticQueryFallback(_settings(), transport=transport)

        outcome = fallback.interpret(
            _ambiguous_plan().raw_query,
            _validator().validate(_ambiguous_plan()),
        )

        self.assertEqual(transport.calls, 1)
        self.assertEqual(fallback.call_count, 1)
        self.assertEqual(outcome.status, "malformed_response")
        self.assertEqual(outcome.transport_status, "success")
        self.assertEqual(outcome.response_shape, "choices[0].message.content:string")
        self.assertTrue(outcome.content_present)
        self.assertEqual(outcome.parse_status, "invalid_json")
        self.assertEqual(outcome.content_format, "json_fence")
        self.assertFalse(outcome.prefix_text_present)
        self.assertFalse(outcome.suffix_text_present)
        self.assertIsNone(outcome.result)

    def test_free_form_prose_with_unfenced_json_is_not_interpreted(self) -> None:
        fallback = HcxSemanticQueryFallback(
            _settings(),
            transport=_Transport(
                '분류 결과는 {"task_type": "corporate_event"} 입니다.'
            ),
        )

        outcome = fallback.interpret(
            _ambiguous_plan().raw_query,
            _validator().validate(_ambiguous_plan()),
        )

        self.assertEqual(outcome.status, "malformed_response")
        self.assertEqual(outcome.parse_status, "invalid_json")
        self.assertIsNone(outcome.result)

    def test_semantic_task_resolution_still_requires_missing_metric(self) -> None:
        plan = QueryPlan(
            query="실적",
            raw_query="삼성중공업 2025년 실적 알려줘",
            company="삼성중공업",
            task_type="disclosure_lookup",
            period=QueryPeriod(year=2025, period_type="reference_year"),
            evidence={"operation": "lookup_disclosure"},
        )
        validator = _validator()
        initial = validator.validate(plan)

        result = validator.validate(
            plan,
            semantic={
                "task_type": "financial_metric",
                "operation": "lookup_metric",
                "ambiguity": False,
            },
            fallback_used=True,
            fallback_status="success",
        )

        self.assertIs(initial.state, QueryState.AMBIGUOUS)
        self.assertIs(result.state, QueryState.INCOMPLETE)
        self.assertEqual(result.missing_slots, ("metric",))
        self.assertIn("재무 지표", result.clarification.question)


if __name__ == "__main__":
    unittest.main()
