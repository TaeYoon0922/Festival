from __future__ import annotations

import json
import unittest

import psycopg
from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.pipeline import (
    EMPTY_ANSWER_FALLBACK,
    AnswerPipeline,
    AnswerPipelineError,
    _non_empty,
)
from app.generation.hcx_verbalizer import (
    HcxSettings,
    HcxVerbalizer,
    VerbalizationOutcome,
)
from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.generation.compact_claim import build_compact_claim
from app.generation.protected_literals import (
    PLACEHOLDER_PATTERN,
    protect_literals,
)
from app.reasoning.query_plan import QueryPlan
from app.retrieval.embeddings import EmbeddingHttpError
from tests.test_agent_end_to_end_smoke import (
    _StaticExecutor,
    _StaticUnderstanding,
    _execution,
)
from tests.test_evidence_builder import _holding_pair


QUESTION = "효성중공업 국민연금기금 변동일 변동후 주식수"
QUESTION_ID = "HX08"

#: ``think_trace`` is an execution summary; anything outside this set would risk
#: exposing intermediate reasoning.
THINK_TRACE_KEYS = {
    "task_type",
    "route",
    "stages",
    "retrieval_count",
    "selected_evidence_count",
    "answerable",
    "warnings",
    "hcx_status",
}


def _plan_and_execution():
    first = _holding_pair(
        "h23:ch_report",
        "h23",
        rank=1,
        date="2023-06-30",
        projection_type="holding_report",
        table_id="t23",
    )
    second = _holding_pair(
        "h24:ch_report",
        "h24",
        rank=2,
        date="2024-06-30",
        projection_type="holding_report",
        table_id="t24",
    )
    plan = QueryPlan(
        query=QUESTION,
        task_type="holding_change",
        metric="holding_shares",
        reporter="국민연금기금",
        disclosure_route=("holding",),
        evidence={"requested_holding_fields": ["reference_date", "after_shares"]},
    )
    return plan, _execution(plan, first, second)


class _FailingExecutor:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def execute(self, plan):
        del plan
        raise self.error


class _StubTransport:
    def __init__(self, content: str) -> None:
        self.content = content

    def post_json(self, url, *, headers, payload, timeout_seconds):
        del url, headers, payload, timeout_seconds
        return {"choices": [{"message": {"content": self.content}}]}


def _pipeline(*, executor=None, verbalizer=None) -> AnswerPipeline:
    plan, execution = _plan_and_execution()
    return AnswerPipeline(
        understanding=_StaticUnderstanding(QUESTION, plan),
        executor=executor or _StaticExecutor(plan, execution),
        verbalizer=verbalizer or HcxVerbalizer(HcxSettings(enabled=False)),
    )


def _client(pipeline_factory=None, **kwargs) -> TestClient:
    app = create_app(pipeline_factory=pipeline_factory or _pipeline)
    return TestClient(app, **kwargs)


def _claim_protection():
    """Rebuild the compact claim the pipeline will produce for this fixture."""

    plan, execution = _plan_and_execution()
    result = AgentOrchestrator().run(QUESTION, plan, execution)
    claim = build_compact_claim(
        result.answer_draft,
        result.resolution,
        task_type=result.task_decision.task_type,
    )
    generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
    return claim, protect_literals(claim.deterministic_text), generated.answer_text


def _hcx_factory(reply: str):
    def factory() -> AnswerPipeline:
        return _pipeline(
            verbalizer=HcxVerbalizer(
                HcxSettings(
                    enabled=True,
                    endpoint="https://clova.example/v1/chat/completions",
                    api_key="key",
                ),
                transport=_StubTransport(reply),
            )
        )

    return factory


def _ask(client: TestClient, **params):
    query = {"question_id": QUESTION_ID, "question": QUESTION}
    query.update(params)
    return client.get("/answer", params=query)


class AnswerResponseContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _client()

    def test_returns_exactly_the_five_contract_fields(self) -> None:
        response = _ask(self.client)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            set(response.json()),
            {"question_id", "question", "retrieved_context", "think_trace", "answer"},
        )

    def test_echoes_the_request(self) -> None:
        payload = _ask(self.client).json()

        self.assertEqual(payload["question_id"], QUESTION_ID)
        self.assertEqual(payload["question"], QUESTION)

    def test_answer_is_non_empty_text(self) -> None:
        payload = _ask(self.client).json()

        self.assertIsInstance(payload["answer"], str)
        self.assertTrue(payload["answer"].strip())

    def test_retrieved_context_reports_the_served_ranking(self) -> None:
        context = _ask(self.client).json()["retrieved_context"]

        self.assertEqual([row["rank"] for row in context], [1, 2])
        self.assertEqual(
            [row["chunk_id"] for row in context], ["h23:ch_report", "h24:ch_report"]
        )
        self.assertEqual(context[0]["doc_id"], "h23")
        self.assertTrue(context[0]["source_refs"])

    def test_retrieved_context_omits_gold_only_fields(self) -> None:
        context = _ask(self.client).json()["retrieved_context"]

        for row in context:
            self.assertNotIn("is_gold_relevant", row)
            self.assertNotIn("score_metadata", row)


class ThinkTraceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.trace = _ask(_client()).json()["think_trace"]

    def test_exposes_only_the_execution_summary(self) -> None:
        self.assertEqual(set(self.trace), THINK_TRACE_KEYS)

    def test_reports_route_and_counts(self) -> None:
        self.assertEqual(self.trace["task_type"], "holding_event")
        self.assertEqual(self.trace["route"], "holding_event_resolver")
        self.assertEqual(self.trace["retrieval_count"], 2)
        self.assertIsInstance(self.trace["selected_evidence_count"], int)
        self.assertIsInstance(self.trace["answerable"], bool)

    def test_stages_are_component_names_not_deliberation(self) -> None:
        for stage in self.trace["stages"]:
            self.assertRegex(stage, r"^[a-z0-9_]+$")
        self.assertIn("answer_generator", self.trace["stages"])


class HcxIntegrationTests(unittest.TestCase):
    def test_disabled_hcx_serves_the_deterministic_answer(self) -> None:
        payload = _ask(_client()).json()

        self.assertEqual(payload["think_trace"]["hcx_status"], "disabled")
        self.assertNotIn("hcx_verbalizer", payload["think_trace"]["stages"])

    def test_faithful_hcx_text_is_served(self) -> None:
        claim, protection, _ = _claim_protection()

        payload = _ask(_client(_hcx_factory(f"{protection.masked}입니다"))).json()

        self.assertEqual(payload["think_trace"]["hcx_status"], "success")
        self.assertEqual(payload["answer"], f"{claim.deterministic_text}입니다")
        self.assertIn("hcx_verbalizer", payload["think_trace"]["stages"])

    def test_served_answer_is_shorter_than_the_full_report(self) -> None:
        _, protection, deterministic = _claim_protection()

        payload = _ask(_client(_hcx_factory(f"{protection.masked}입니다"))).json()

        self.assertLess(len(payload["answer"]), len(deterministic))

    def test_served_answer_never_leaks_a_placeholder(self) -> None:
        _, protection, _ = _claim_protection()

        payload = _ask(_client(_hcx_factory(f"{protection.masked}입니다"))).json()

        self.assertIsNone(PLACEHOLDER_PATTERN.search(payload["answer"]))
        self.assertNotIn("__FESTIVAL_", json.dumps(payload, ensure_ascii=False))

    def test_normalized_literals_fall_back(self) -> None:
        """The live failure mode: HCX writes its own prose instead of restating."""

        deterministic = _ask(_client()).json()["answer"]

        payload = _ask(_client(_hcx_factory(deterministic))).json()

        self.assertEqual(
            payload["think_trace"]["hcx_status"],
            "fallback_placeholder_integrity_failed",
        )
        self.assertEqual(payload["answer"], deterministic)

    def test_hallucinating_hcx_falls_back_to_the_deterministic_answer(self) -> None:
        _, protection, deterministic = _claim_protection()

        payload = _ask(
            _client(_hcx_factory(protection.masked + " 총 12,345주 늘었습니다."))
        ).json()

        self.assertEqual(
            payload["think_trace"]["hcx_status"], "fallback_validation_failed"
        )
        self.assertEqual(payload["answer"], deterministic)


class NonEmptyAnswerTests(unittest.TestCase):
    """A 200 always carries answer text, whatever the verbalizer did."""

    def _answer(self, verbalizer) -> str:
        def factory() -> AnswerPipeline:
            return _pipeline(verbalizer=verbalizer)

        payload = _ask(_client(factory)).json()
        return payload["answer"]

    def test_blank_hcx_reply_still_yields_text(self) -> None:
        verbalizer = HcxVerbalizer(
            HcxSettings(enabled=True, api_key="key"),
            transport=_StubTransport("   "),
        )

        self.assertTrue(self._answer(verbalizer).strip())

    def test_blank_verbalizer_output_falls_back_to_the_deterministic_text(
        self,
    ) -> None:
        class _BlankVerbalizer:
            def verbalize(self, generated, **kwargs):
                del generated, kwargs
                return VerbalizationOutcome("", "fallback_error", "blank")

        deterministic = _ask(_client()).json()["answer"]

        self.assertEqual(self._answer(_BlankVerbalizer()), deterministic)

    def test_last_resort_fallback_is_used_when_nothing_has_text(self) -> None:
        self.assertEqual(_non_empty("", "   "), EMPTY_ANSWER_FALLBACK)
        self.assertEqual(_non_empty("", "답변"), "답변")


class RequestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _client()

    def test_blank_question_is_rejected(self) -> None:
        self.assertEqual(_ask(self.client, question="   ").status_code, 422)

    def test_empty_question_is_rejected(self) -> None:
        self.assertEqual(_ask(self.client, question="").status_code, 422)

    def test_blank_question_id_is_rejected(self) -> None:
        self.assertEqual(_ask(self.client, question_id="  ").status_code, 422)

    def test_missing_question_is_rejected(self) -> None:
        response = self.client.get("/answer", params={"question_id": QUESTION_ID})

        self.assertEqual(response.status_code, 422)

    def test_missing_question_id_is_rejected(self) -> None:
        response = self.client.get("/answer", params={"question": QUESTION})

        self.assertEqual(response.status_code, 422)


class FailureHandlingTests(unittest.TestCase):
    def _failure(self, error: Exception):
        def factory() -> AnswerPipeline:
            return _pipeline(executor=_FailingExecutor(error))

        return _ask(_client(factory))

    def test_database_outage_is_reported_as_unavailable(self) -> None:
        response = self._failure(
            psycopg.OperationalError(
                "connection to server at 10.0.0.4, user festival failed"
            )
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["reason"], "database_unavailable")

    def test_database_outage_never_leaks_connection_details(self) -> None:
        response = self._failure(
            psycopg.OperationalError(
                "connection to server at 10.0.0.4, user festival failed"
            )
        )

        body = response.text
        self.assertNotIn("10.0.0.4", body)
        self.assertNotIn("festival", body)
        self.assertNotIn("Traceback", body)

    def test_embedding_outage_is_reported_as_unavailable(self) -> None:
        response = self._failure(
            EmbeddingHttpError("boom", status_code=503, transient=True)
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["reason"], "embedding_unavailable")

    def test_unexpected_failure_is_sanitized(self) -> None:
        response = self._failure(ValueError("internal detail: secret-token-42"))

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["reason"], "internal_error")
        self.assertNotIn("secret-token-42", response.text)

    def test_pipeline_construction_failure_is_reported(self) -> None:
        def factory() -> AnswerPipeline:
            raise AnswerPipelineError("embedding_unavailable")

        response = _ask(_client(factory))

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["reason"], "embedding_unavailable")

    def test_unexpected_construction_failure_is_sanitized(self) -> None:
        def factory() -> AnswerPipeline:
            raise RuntimeError("dsn=postgresql://user:pw@host/db")

        response = _ask(_client(factory, raise_server_exceptions=False))

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["reason"], "internal_error")
        self.assertNotIn("postgresql://", response.text)


class HealthTests(unittest.TestCase):
    def test_healthz_does_not_touch_the_pipeline(self) -> None:
        def factory() -> AnswerPipeline:
            raise AssertionError("healthz must not build the pipeline")

        response = _client(factory).get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
