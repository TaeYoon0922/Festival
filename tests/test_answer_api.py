from __future__ import annotations

import json
import re
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
from app.generation.lossless_verbalization import (
    detach_claim_citations,
    expected_attached_answer,
)
from app.generation.protected_literals import PLACEHOLDER_PATTERN
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.retrieval.embeddings import EmbeddingHttpError
from tests.test_agent_end_to_end_smoke import (
    _StaticExecutor,
    _StaticUnderstanding,
    _execution,
)
from tests.test_evidence_builder import _holding_pair
from tests.test_multi_document_serving import (
    LIFECYCLE_Q as P0C_LIFECYCLE_Q,
    _EventRepo as _P0CEventRepo,
    _pipeline as _p0c_pipeline,
    _state as _p0c_state,
)


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
    "correction",
}

#: The identifiers the correction graph reports when it supplied evidence.
CORRECTION_TRACE_KEYS = {
    "correction_intent",
    "correction_group_id",
    "correction_root_doc_id",
    "correction_latest_doc_id",
    "correction_added_doc_ids",
}


def _p0c_unresolved_pipeline() -> AnswerPipeline:
    contracts = [_p0c_state(f"event-secret-{index:02d}") for index in range(3)]
    contracts.append(
        _p0c_state(
            "event-secret-03",
            source="related_reference_not_in_corpus",
            doc_id="document-secret-03",
        )
    )
    return _p0c_pipeline(_P0CEventRepo(contracts))


def _p0c_bare_contract_pipeline() -> AnswerPipeline:
    return _p0c_pipeline(_P0CEventRepo([_p0c_state("event-secret-bare")]))


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


#: What ``CorrectionExpander`` records on an execution after it adds documents.
CORRECTION_EXPANSION = {
    "correction_intent": "latest",
    "correction_expanded": True,
    "correction_status": "expanded",
    "correction_group_id": "exchange_20230626800002",
    "correction_root_doc_id": "exchange_20230626800002",
    "correction_latest_doc_id": "exchange_20260120800597",
    "correction_group_count": 1,
    "correction_added_doc_ids": ["exchange_20260120800597"],
    "correction_added_result_count": 1,
}


def _correction_pipeline() -> AnswerPipeline:
    """The pipeline as it behaves after correction expansion has fired."""

    plan, execution = _plan_and_execution()
    execution.correction_expansion = dict(CORRECTION_EXPANSION)
    return AnswerPipeline(
        understanding=_StaticUnderstanding(QUESTION, plan),
        executor=_StaticExecutor(plan, execution),
        verbalizer=HcxVerbalizer(HcxSettings(enabled=False)),
    )


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


def _single_event_plan_and_execution():
    """One holding event: the only shape HCX is asked to verbalize."""

    pair = _holding_pair(
        "h23:ch_report",
        "h23",
        rank=1,
        date="2023-06-30",
        projection_type="holding_report",
        table_id="t23",
    )
    # An exact reference date: HCX may only restate answers whose question
    # named one event (P1-A5-A), and these tests exercise the round trip.
    plan = QueryPlan(
        query=QUESTION,
        task_type="holding_change",
        metric="holding_shares",
        reporter="국민연금기금",
        disclosure_route=("holding",),
        period=QueryPeriod(
            year=2023,
            from_date="2023-06-30",
            to_date="2023-06-30",
            period_type="holding_reference_date",
        ),
        evidence={"requested_holding_fields": ["reference_date", "after_shares"]},
    )
    return plan, _execution(plan, pair)


def _single_event_pipeline(*, verbalizer=None) -> AnswerPipeline:
    plan, execution = _single_event_plan_and_execution()
    return AnswerPipeline(
        understanding=_StaticUnderstanding(QUESTION, plan),
        executor=_StaticExecutor(plan, execution),
        verbalizer=verbalizer or HcxVerbalizer(HcxSettings(enabled=False)),
    )


def _claim_detachment():
    """Rebuild what the pipeline sends the model for the single-event fixture."""

    plan, execution = _single_event_plan_and_execution()
    result = AgentOrchestrator().run(QUESTION, plan, execution)
    claim = build_compact_claim(
        result.answer_draft,
        result.resolution,
        task_type=result.task_decision.task_type,
    )
    detached = detach_claim_citations(claim)
    generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
    return detached, expected_attached_answer(detached), generated.answer_text


def _hcx_factory(reply: str, *, single_event: bool = True):
    build = _single_event_pipeline if single_event else _pipeline

    def factory() -> AnswerPipeline:
        return build(
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


class CorrectionTraceTests(unittest.TestCase):
    """The correction block must survive response serialization.

    ``think_trace`` is validated through ``AnswerResponse``, so a key the model
    does not declare is dropped on the way out no matter what the pipeline put
    there.  These tests hold that contract open.
    """

    def setUp(self) -> None:
        self.payload = _ask(_client(_correction_pipeline)).json()
        self.trace = self.payload["think_trace"]

    def test_the_correction_block_reaches_the_response(self) -> None:
        self.assertIsNotNone(self.trace["correction"])
        self.assertEqual(set(self.trace["correction"]), CORRECTION_TRACE_KEYS)

    def test_the_reported_identifiers_are_the_ones_the_graph_supplied(self) -> None:
        correction = self.trace["correction"]

        self.assertEqual(correction["correction_intent"], "latest")
        self.assertEqual(
            correction["correction_group_id"], "exchange_20230626800002"
        )
        self.assertEqual(
            correction["correction_root_doc_id"], "exchange_20230626800002"
        )
        self.assertEqual(
            correction["correction_latest_doc_id"], "exchange_20260120800597"
        )
        self.assertEqual(
            correction["correction_added_doc_ids"], ["exchange_20260120800597"]
        )

    def test_expansion_is_named_among_the_stages(self) -> None:
        self.assertIn("correction_expansion", self.trace["stages"])
        for stage in self.trace["stages"]:
            self.assertRegex(stage, r"^[a-z0-9_]+$")

    def test_the_block_carries_identifiers_only(self) -> None:
        """No deliberation, no free text, no chunk contents."""

        correction = self.trace["correction"]
        self.assertIsInstance(correction["correction_added_doc_ids"], list)
        for value in correction.values():
            for item in value if isinstance(value, list) else [value]:
                self.assertIsInstance(item, str)
                self.assertRegex(item, r"^[A-Za-z0-9_.:-]+$")

    def test_the_response_still_has_exactly_the_five_top_level_fields(self) -> None:
        self.assertEqual(
            set(self.payload),
            {"question_id", "question", "retrieved_context", "think_trace", "answer"},
        )

    def test_think_trace_gains_no_key_beyond_the_declared_summary(self) -> None:
        self.assertEqual(set(self.trace), THINK_TRACE_KEYS)

    def test_an_ordinary_response_reports_no_correction(self) -> None:
        ordinary = _ask(_client()).json()

        self.assertIsNone(ordinary["think_trace"]["correction"])
        self.assertNotIn("correction_expansion", ordinary["think_trace"]["stages"])
        self.assertEqual(
            set(ordinary),
            {"question_id", "question", "retrieved_context", "think_trace", "answer"},
        )
        self.assertEqual(set(ordinary["think_trace"]), THINK_TRACE_KEYS)


class MultiDocumentHttpTraceTests(unittest.TestCase):
    """P0-C trace must survive the actual FastAPI response-model boundary."""

    def test_engaged_unresolved_trace_survives_http_serialization(self) -> None:
        response = _ask(
            _client(_p0c_unresolved_pipeline),
            question_id="P0C-HTTP-U",
            question=P0C_LIFECYCLE_Q,
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            set(payload),
            {"question_id", "question", "retrieved_context", "think_trace", "answer"},
        )
        trace = payload["think_trace"]["multi_document_planner"]
        self.assertIsInstance(trace, dict)
        self.assertEqual(
            set(trace),
            {
                "applied",
                "plan_type",
                "family_resolution",
                "passes",
                "complete",
                "stop_reason",
                "logical_count",
                "unresolved_count",
                "lifecycle_answer",
                "terminated_count",
                "open_count",
                "evidence_count",
            },
        )
        self.assertIs(trace["applied"], True)
        self.assertEqual(trace["plan_type"], "enumeration_plus_event")
        self.assertIs(trace["complete"], False)
        self.assertEqual(trace["logical_count"], 4)
        self.assertEqual(trace["unresolved_count"], 2)
        self.assertEqual(trace["lifecycle_answer"], "undetermined")
        self.assertEqual(trace["terminated_count"], 0)
        self.assertEqual(trace["open_count"], 4)
        self.assertGreater(trace["evidence_count"], 0)

    def test_p0c_trace_contains_no_document_or_event_identifiers(self) -> None:
        payload = _ask(
            _client(_p0c_unresolved_pipeline),
            question_id="P0C-HTTP-ID",
            question=P0C_LIFECYCLE_Q,
        ).json()
        serialized = json.dumps(
            payload["think_trace"]["multi_document_planner"],
            ensure_ascii=False,
            sort_keys=True,
        )

        for forbidden_key in (
            "event_id",
            "doc_id",
            "chunk_id",
            "corp_code",
            "slot_id",
        ):
            self.assertNotIn(f'"{forbidden_key}"', serialized, forbidden_key)
        for forbidden_value in (
            "event-secret",
            "document-secret",
        ):
            self.assertNotIn(forbidden_value, serialized, forbidden_value)
        self.assertNotIn("slots", payload["think_trace"]["multi_document_planner"])
        self.assertNotIn("lifecycle", payload["think_trace"]["multi_document_planner"])

    def test_family_resolution_survives_but_never_reaches_answer(self) -> None:
        question = "삼성중공업이 2025년에 체결한 주요 계약은 모두 몇 건인가?"
        payload = _ask(
            _client(_p0c_bare_contract_pipeline),
            question_id="P0C-HTTP-FAMILY",
            question=question,
        ).json()

        trace = payload["think_trace"]["multi_document_planner"]
        self.assertEqual(trace["family_resolution"], "bare_contract_fallback")
        for lifecycle_only in (
            "slots",
            "lifecycle",
            "lifecycle_answer",
            "terminated_count",
            "open_count",
        ):
            self.assertNotIn(lifecycle_only, trace, lifecycle_only)
        for internal in (
            "family_resolution",
            "bare_contract_fallback",
            "slot_id",
            "corp_code",
        ):
            self.assertNotIn(internal, payload["answer"], internal)

    def test_non_engagement_omits_the_optional_block(self) -> None:
        payload = _ask(_client()).json()

        self.assertNotIn("multi_document_planner", payload["think_trace"])
        self.assertEqual(set(payload["think_trace"]), THINK_TRACE_KEYS)
        self.assertIsNone(payload["think_trace"]["correction"])


class HcxIntegrationTests(unittest.TestCase):
    def test_disabled_hcx_serves_the_deterministic_answer(self) -> None:
        payload = _ask(_client()).json()

        self.assertEqual(payload["think_trace"]["hcx_status"], "disabled")
        self.assertNotIn("hcx_verbalizer", payload["think_trace"]["stages"])

    def test_faithful_hcx_text_is_served(self) -> None:
        detached, expected_final, _ = _claim_detachment()

        payload = _ask(_client(_hcx_factory(detached.protection.masked))).json()

        self.assertEqual(payload["think_trace"]["hcx_status"], "success")
        self.assertEqual(payload["answer"], expected_final)
        self.assertIn("hcx_verbalizer", payload["think_trace"]["stages"])

    def test_served_answer_is_shorter_than_the_full_report(self) -> None:
        detached, _, deterministic = _claim_detachment()

        payload = _ask(_client(_hcx_factory(detached.protection.masked))).json()

        self.assertLess(len(payload["answer"]), len(deterministic))

    def test_served_answer_carries_its_deterministic_citation(self) -> None:
        detached, _, _ = _claim_detachment()

        payload = _ask(_client(_hcx_factory(detached.protection.masked))).json()

        self.assertIn("[1]", payload["answer"])

    def test_served_answer_never_leaks_a_placeholder(self) -> None:
        detached, _, _ = _claim_detachment()

        payload = _ask(_client(_hcx_factory(detached.protection.masked))).json()

        self.assertIsNone(PLACEHOLDER_PATTERN.search(payload["answer"]))
        self.assertNotIn("__FESTIVAL_", json.dumps(payload, ensure_ascii=False))

    def test_a_multi_event_claim_skips_hcx(self) -> None:
        """The two-event fixture is served deterministically, unchanged."""

        deterministic = _ask(_client()).json()["answer"]

        payload = _ask(
            _client(_hcx_factory("무엇이든", single_event=False))
        ).json()

        self.assertEqual(
            payload["think_trace"]["hcx_status"],
            "skipped_multi_event_compact_claim",
        )
        self.assertEqual(payload["answer"], deterministic)
        self.assertNotIn("hcx_verbalizer", payload["think_trace"]["stages"])

    def test_a_redundant_unit_never_reaches_the_client(self) -> None:
        """The reported defect, at the boundary a user actually sees."""

        detached, _, deterministic = _claim_detachment()
        placeholder = detached.protection.literals[-1].placeholder
        reply = detached.protection.masked.replace(
            placeholder, placeholder + "주", 1
        )

        payload = _ask(_client(_hcx_factory(reply))).json()

        self.assertEqual(
            payload["think_trace"]["hcx_status"], "fallback_redundant_unit_suffix"
        )
        self.assertEqual(payload["answer"], deterministic)
        self.assertTrue(payload["answer"].strip())
        self.assertNotIn("%%", payload["answer"])
        self.assertNotIn("주주", payload["answer"])
        self.assertIsNone(
            # Same line only: the defect is a unit glued onto a citation
            # marker, not a section heading that happens to start with 주.
            re.search(r"\[\d+\][ 	]*[%주원배]", payload["answer"])
        )

    def test_normalized_literals_fall_back(self) -> None:
        """The live failure mode: HCX writes its own prose instead of restating."""

        _, _, deterministic = _claim_detachment()

        payload = _ask(_client(_hcx_factory(deterministic))).json()

        self.assertEqual(
            payload["think_trace"]["hcx_status"],
            "fallback_placeholder_integrity_failed",
        )
        self.assertEqual(payload["answer"], deterministic)

    def test_hallucinating_hcx_falls_back_to_the_deterministic_answer(self) -> None:
        detached, _, deterministic = _claim_detachment()

        payload = _ask(
            _client(
                _hcx_factory(detached.protection.masked + " 총 12,345주 늘었습니다.")
            )
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
