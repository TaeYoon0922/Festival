from __future__ import annotations

import json
import unittest

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import (
    CitationAwareAnswerGenerator,
    GeneratedAnswer,
    GeneratedCitation,
    GeneratedSection,
)
from app.generation.compact_claim import build_compact_claim
from app.generation.hcx_verbalizer import (
    CITATION_ATTACHMENT_FAILED_STATUS,
    LOSSLESS_VERBALIZER_SYSTEM_PROMPT,
    PLACEHOLDER_INTEGRITY_FAILED,
    SKIPPED_MULTI_EVENT_CLAIM,
    SKIPPED_NO_COMPACT_CLAIM,
    HcxSettings,
    HcxVerbalizer,
)
from app.generation.lossless_verbalization import (
    claim_event_count,
    detach_claim_citations,
    expected_attached_answer,
)
from app.reasoning.answer_composer import compose_periodic_answer
from app.reasoning.periodic_fact_resolver import resolve_periodic_facts
from app.reasoning.query_plan import QueryPlan
from app.retrieval.embeddings import EmbeddingHttpError
from tests.test_agent_end_to_end_smoke import _execution
from tests.test_evidence_builder import _candidate, _holding_pair
from tests.test_periodic_fact_resolver import (
    _evidence as _periodic_evidence,
    _group as _periodic_group,
    _item as _periodic_item,
)


def _holding_context(events: int = 1):
    """Run the production holding path and keep what the verbalizer needs."""

    pairs = [
        _holding_pair(
            f"h2{index}:ch",
            f"h2{index}",
            rank=index + 1,
            date=f"202{3 + index}-06-30",
            projection_type="holding_report",
            table_id=f"t2{index}",
        )
        for index in range(events)
    ]
    plan = QueryPlan(
        query="효성중공업 국민연금기금 변동일 변동후 주식수",
        task_type="holding_change",
        metric="holding_shares",
        reporter="국민연금기금",
        disclosure_route=("holding",),
        evidence={"requested_holding_fields": ["reference_date", "after_shares"]},
    )
    result = AgentOrchestrator().run(plan.raw_query, plan, _execution(plan, *pairs))
    generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
    return result, generated


RESULT, GENERATED = _holding_context(events=1)

#: The full deterministic answer, served whenever HCX is skipped or rejected.
DETERMINISTIC_TEXT = GENERATED.answer_text

CLAIM = build_compact_claim(
    RESULT.answer_draft,
    RESULT.resolution,
    task_type=RESULT.task_decision.task_type,
)
DETACHED = detach_claim_citations(CLAIM)

#: What the model is shown: citation-free, every verified value masked.
MASKED = DETACHED.protection.masked

#: What a perfectly faithful reply becomes once citations are reattached.
EXPECTED_FINAL = expected_attached_answer(DETACHED)


def _kwargs(result=RESULT) -> dict:
    return {
        "draft": result.answer_draft,
        "resolution": result.resolution,
        "task_type": result.task_decision.task_type,
    }


def _settings(**overrides) -> HcxSettings:
    values = {
        "enabled": True,
        "endpoint": "https://clova.example/v1/chat/completions",
        "api_key": "test-key",
    }
    values.update(overrides)
    return HcxSettings(**values)


class _Transport:
    """Injected JSON transport; the tests never touch the network."""

    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


def _reply(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _verbalize(reply: str, *, generated=GENERATED, result=RESULT, **overrides):
    transport = _Transport(_reply(reply))
    verbalizer = HcxVerbalizer(_settings(**overrides), transport=transport)
    return verbalizer.verbalize(generated, **_kwargs(result)), transport


class SingleEventGateTests(unittest.TestCase):
    """HCX is asked only about the claim shape live runs proved it can restate."""

    def test_a_single_event_claim_reaches_the_model(self) -> None:
        outcome, transport = _verbalize(MASKED)

        self.assertEqual(claim_event_count(CLAIM), 1)
        self.assertEqual(outcome.status, "success")
        self.assertEqual(len(transport.calls), 1)

    def test_a_two_event_claim_never_reaches_the_model(self) -> None:
        result, generated = _holding_context(events=2)
        transport = _Transport(_reply(MASKED))

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            generated, **_kwargs(result)
        )

        self.assertEqual(outcome.status, SKIPPED_MULTI_EVENT_CLAIM)
        self.assertEqual(outcome.text, generated.answer_text)
        self.assertEqual(transport.calls, [])

    def test_a_three_event_claim_never_reaches_the_model(self) -> None:
        result, generated = _holding_context(events=3)
        transport = _Transport(_reply(MASKED))

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            generated, **_kwargs(result)
        )

        self.assertEqual(outcome.status, SKIPPED_MULTI_EVENT_CLAIM)
        self.assertEqual(transport.calls, [])

    def test_the_multi_event_deterministic_answer_is_returned_unchanged(self) -> None:
        result, generated = _holding_context(events=3)

        outcome = HcxVerbalizer(_settings(), transport=_Transport()).verbalize(
            generated, **_kwargs(result)
        )

        self.assertEqual(outcome.text, generated.answer_text)

    def test_the_gate_does_not_change_compact_claim_eligibility(self) -> None:
        """A skipped claim is still a claim; only the HCX call is narrowed."""

        result, _ = _holding_context(events=3)

        claim = build_compact_claim(
            result.answer_draft,
            result.resolution,
            task_type=result.task_decision.task_type,
        )

        self.assertIsNotNone(claim)
        self.assertEqual(claim_event_count(claim), 3)


class CitationDetachmentTests(unittest.TestCase):
    def test_the_model_never_receives_a_citation_marker(self) -> None:
        _, transport = _verbalize(MASKED)

        body = json.dumps(transport.calls[0]["payload"], ensure_ascii=False)
        self.assertNotIn("[1]", body)
        self.assertNotIn("CITATION", body)

    def test_the_model_never_receives_a_raw_verified_value(self) -> None:
        _, transport = _verbalize(MASKED)

        body = json.dumps(transport.calls[0]["payload"], ensure_ascii=False)
        self.assertNotIn("2023-06-30", body)
        self.assertNotIn("1,000", body)

    def test_the_model_is_sent_the_masked_claim_not_the_report(self) -> None:
        _, transport = _verbalize(MASKED)

        user = transport.calls[0]["payload"]["messages"][1]["content"]
        self.assertEqual(user, MASKED)
        self.assertNotIn(DETERMINISTIC_TEXT, user)

    def test_the_lossless_prompt_is_sent(self) -> None:
        _, transport = _verbalize(MASKED)

        system = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertEqual(system, LOSSLESS_VERBALIZER_SYSTEM_PROMPT)

    def test_citations_are_reattached_after_a_successful_reply(self) -> None:
        outcome, _ = _verbalize(MASKED)

        self.assertEqual(outcome.status, "success")
        self.assertIn("[1]", outcome.text)
        self.assertEqual(outcome.text, EXPECTED_FINAL)


class LiteralPreservationTests(unittest.TestCase):
    def test_a_faithful_reply_restores_every_literal_verbatim(self) -> None:
        outcome, _ = _verbalize(MASKED)

        self.assertEqual(outcome.status, "success")
        self.assertIn("2023-06-30", outcome.text)
        self.assertIn("1,000주", outcome.text)

    def test_wording_may_change_around_the_placeholders(self) -> None:
        date, number = DETACHED.protection.placeholders
        reply = f"국민연금기금 테스트회사 변동일 {date}, 변동 후 주식수 {number}주입니다"

        outcome, _ = _verbalize(reply)

        self.assertEqual(outcome.status, "success")
        self.assertIn("2023-06-30", outcome.text)
        self.assertIn("1,000주", outcome.text)

    def test_the_native_clova_envelope_is_accepted(self) -> None:
        transport = _Transport({"result": {"message": {"content": MASKED}}})

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            GENERATED, **_kwargs()
        )

        self.assertEqual(outcome.status, "success")


class FailClosedTests(unittest.TestCase):
    """Every rejection serves the deterministic answer, byte for byte."""

    def _rejected(self, reply: str) -> tuple[str, str]:
        outcome, _ = _verbalize(reply)
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)
        return outcome.status, outcome.reason or ""

    def test_a_missing_placeholder_falls_back(self) -> None:
        date = DETACHED.protection.placeholders[0]

        status, reason = self._rejected(MASKED.replace(date, ""))

        self.assertEqual(status, PLACEHOLDER_INTEGRITY_FAILED)
        self.assertEqual(reason, "placeholder_missing")

    def test_a_reordered_placeholder_falls_back(self) -> None:
        date, number = DETACHED.protection.placeholders

        status, reason = self._rejected(f"{number}주 그리고 {date}")

        self.assertEqual(status, PLACEHOLDER_INTEGRITY_FAILED)
        self.assertEqual(reason, "placeholder_reordered")

    def test_a_duplicated_placeholder_falls_back(self) -> None:
        number = DETACHED.protection.placeholders[1]

        status, reason = self._rejected(MASKED + number)

        self.assertEqual(status, PLACEHOLDER_INTEGRITY_FAILED)
        self.assertEqual(reason, "placeholder_duplicated")

    def test_an_unexpected_placeholder_falls_back(self) -> None:
        status, reason = self._rejected(MASKED + " __FESTIVAL_NUMBER_Z__")

        self.assertEqual(status, PLACEHOLDER_INTEGRITY_FAILED)
        self.assertEqual(reason, "placeholder_unexpected")

    def test_an_invented_number_falls_back(self) -> None:
        """The HX09 failure: a prior value the claim never carried."""

        status, reason = self._rejected(f"이전에는 2,000주였고 {MASKED}")

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "unprotected_numeric_generation")

    def test_a_model_written_citation_falls_back(self) -> None:
        status, reason = self._rejected(MASKED + "[1]")

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "generated_citation")

    def test_investment_language_falls_back(self) -> None:
        status, reason = self._rejected(MASKED + " 매수를 추천합니다")

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "forbidden_investment_language")

    def test_an_added_conclusion_falls_back(self) -> None:
        status, reason = self._rejected(MASKED + " 이를 통해 추이를 알 수 있습니다")

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "inference_marker_added")

    def test_a_missing_entity_falls_back(self) -> None:
        date, number = DETACHED.protection.placeholders
        reply = f"변동일 {date}, 변동 후 주식수 {number}주"

        status, reason = self._rejected(reply)

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "entity_missing")

    def test_an_overlong_reply_falls_back(self) -> None:
        status, reason = self._rejected(MASKED + " 그리고" * 200)

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "length_exceeded")

    def test_an_empty_reply_falls_back(self) -> None:
        status, _ = self._rejected("   ")

        self.assertEqual(status, "fallback_invalid_response")


class StructuredTextLeakageTests(unittest.TestCase):
    """A TEXT value must appear only inside its placeholder."""

    def setUp(self) -> None:
        pair = _holding_pair(
            "h23:ch", "h23", rank=1, date="2023-06-30",
            projection_type="holding_report", table_id="t23",
        )
        plan = QueryPlan(
            query="효성중공업 국민연금기금 변동일 변동방향",
            task_type="holding_change",
            metric="holding_shares",
            reporter="국민연금기금",
            disclosure_route=("holding",),
            evidence={"requested_holding_fields": ["reference_date"]},
        )
        self.result = AgentOrchestrator().run(
            plan.raw_query, plan, _execution(plan, pair)
        )
        self.generated = CitationAwareAnswerGenerator().generate(
            self.result.answer_draft
        )
        self.claim = build_compact_claim(
            self.result.answer_draft,
            self.result.resolution,
            task_type=self.result.task_decision.task_type,
        )
        self.detached = detach_claim_citations(self.claim)

    def test_a_faithful_reply_succeeds(self) -> None:
        outcome, _ = _verbalize(
            self.detached.protection.masked,
            generated=self.generated,
            result=self.result,
        )

        self.assertEqual(outcome.status, "success")

    def test_an_invented_number_still_falls_back_here(self) -> None:
        """TEXT-literal leakage itself is covered in test_lossless_verbalization."""

        outcome, _ = _verbalize(
            f"이전 2,000주 {self.detached.protection.masked}",
            generated=self.generated,
            result=self.result,
        )

        self.assertEqual(outcome.reason, "unprotected_numeric_generation")
        self.assertEqual(outcome.text, self.generated.answer_text)


class TransportFailureTests(unittest.TestCase):
    def _failure(self, error: Exception) -> tuple[str, str]:
        transport = _Transport(error=error)
        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            GENERATED, **_kwargs()
        )
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)
        return outcome.status, outcome.reason or ""

    def test_a_timeout_falls_back(self) -> None:
        status, _ = self._failure(
            EmbeddingHttpError("timed out", status_code=None, transient=True)
        )

        self.assertEqual(status, "fallback_timeout")

    def test_an_http_error_falls_back(self) -> None:
        status, reason = self._failure(
            EmbeddingHttpError("boom", status_code=500, transient=True)
        )

        self.assertEqual(status, "fallback_http_error")
        self.assertEqual(reason, "HTTP 500")

    def test_an_unexpected_exception_falls_back(self) -> None:
        status, _ = self._failure(ValueError("not JSON"))

        self.assertEqual(status, "fallback_error")

    def test_a_missing_message_falls_back(self) -> None:
        transport = _Transport({"choices": []})

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            GENERATED, **_kwargs()
        )

        self.assertEqual(outcome.status, "fallback_invalid_response")
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)


class NeverEmptyTests(unittest.TestCase):
    def test_every_outcome_carries_text(self) -> None:
        replies = [
            MASKED,
            MASKED + "[1]",
            MASKED + " 매수",
            MASKED.replace(DETACHED.protection.placeholders[0], ""),
            "   ",
        ]

        for reply in replies:
            with self.subTest(reply=reply[:24]):
                outcome, _ = _verbalize(reply)
                self.assertTrue(outcome.text.strip())


class SkipTests(unittest.TestCase):
    def test_no_compact_claim_skips_without_calling_the_model(self) -> None:
        older = _periodic_item(
            "p23:ch", "p23", rank=1, text="연료전지 매출액은 1,234억원입니다.", year=2023
        )
        newer = _periodic_item(
            "p24:ch", "p24", rank=2, text="연료전지 매출액은 1,234억원입니다.", year=2024
        )
        evidence = _periodic_evidence(
            [_periodic_group("g1", older, newer, group_type="periodic_repeated_fact")]
        )
        resolution = resolve_periodic_facts(evidence)
        draft = compose_periodic_answer(resolution, evidence)
        generated = CitationAwareAnswerGenerator().generate(draft)
        transport = _Transport(_reply(MASKED))

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            generated, draft=draft, resolution=resolution, task_type="periodic_fact"
        )

        self.assertEqual(outcome.status, SKIPPED_NO_COMPACT_CLAIM)
        self.assertEqual(outcome.text, generated.answer_text)
        self.assertEqual(transport.calls, [])

    def test_general_evidence_skips_without_calling_the_model(self) -> None:
        pair = _candidate(
            "m23:ch", "m23", rank=1, doc_group="major",
            content="유상증자 결정. 신주 1,000,000주.",
            section="주요사항보고서", report_nm="유상증자결정",
        )
        plan = QueryPlan(
            query="삼성전자 유상증자 공시 내용",
            task_type="corporate_event",
            event_type="capital_increase",
            disclosure_route=("major",),
        )
        result = AgentOrchestrator().run(plan.raw_query, plan, _execution(plan, pair))
        generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
        transport = _Transport(_reply(MASKED))

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            generated, **_kwargs(result)
        )

        self.assertEqual(outcome.status, SKIPPED_NO_COMPACT_CLAIM)
        self.assertEqual(transport.calls, [])

    def test_disabled_returns_the_deterministic_answer(self) -> None:
        transport = _Transport(_reply(MASKED))

        outcome = HcxVerbalizer(
            _settings(enabled=False), transport=transport
        ).verbalize(GENERATED, **_kwargs())

        self.assertEqual(outcome.status, "disabled")
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)
        self.assertEqual(transport.calls, [])

    def test_missing_credentials_return_the_deterministic_answer(self) -> None:
        transport = _Transport(_reply(MASKED))

        outcome = HcxVerbalizer(_settings(api_key=""), transport=transport).verbalize(
            GENERATED, **_kwargs()
        )

        self.assertEqual(outcome.status, "not_configured")
        self.assertEqual(transport.calls, [])

    def test_an_unanswerable_result_never_reaches_the_model(self) -> None:
        unanswerable = GeneratedAnswer(
            question="확인되지 않는 질문",
            answer_text="확인되지 않은 정보가 있습니다.",
            citations=(
                GeneratedCitation(
                    citation_id="[1]", chunk_id="c", doc_id="d",
                    source_refs=(), section="s", evidence_type="table",
                ),
            ),
            sections=(
                GeneratedSection(
                    title="확인 필요",
                    content="확인되지 않은 정보가 있습니다.",
                    citations=(),
                ),
            ),
            warnings=("answer_not_supported",),
            confidence={"level": "low", "display_text": "낮음"},
            answerable=False,
        )
        transport = _Transport(_reply(MASKED))

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            unanswerable, **_kwargs()
        )

        self.assertEqual(outcome.status, "skipped_not_answerable")
        self.assertEqual(outcome.text, unanswerable.answer_text)
        self.assertEqual(transport.calls, [])


class OpenAiCompatibleContractTests(unittest.TestCase):
    """The CLOVA Studio OpenAI compatibility contract is unchanged."""

    def setUp(self) -> None:
        _, transport = _verbalize(MASKED)
        self.call = transport.calls[0]

    def test_sends_only_supported_fields(self) -> None:
        self.assertEqual(
            set(self.call["payload"]),
            {"model", "messages", "temperature", "max_tokens"},
        )

    def test_omits_the_unsupported_top_p_field(self) -> None:
        self.assertNotIn("top_p", self.call["payload"])

    def test_sends_no_native_camelcase_fields(self) -> None:
        for field in ("maxTokens", "topP", "topK", "repeatPenalty", "stopBefore"):
            self.assertNotIn(field, self.call["payload"])

    def test_pins_deterministic_sampling(self) -> None:
        self.assertEqual(self.call["payload"]["temperature"], 0.0)

    def test_uses_a_system_and_user_message_pair(self) -> None:
        roles = [message["role"] for message in self.call["payload"]["messages"]]

        self.assertEqual(roles, ["system", "user"])

    def test_sends_bearer_authorization(self) -> None:
        self.assertEqual(self.call["headers"], {"Authorization": "Bearer test-key"})

    def test_defaults_to_the_official_compatibility_endpoint(self) -> None:
        settings = HcxSettings.from_env({})

        self.assertEqual(
            settings.endpoint,
            "https://clovastudio.stream.ntruss.com/v1/openai/chat/completions",
        )
        self.assertEqual(settings.model, "HCX-005")

    def test_default_endpoint_alone_does_not_enable_the_verbalizer(self) -> None:
        self.assertFalse(HcxSettings.from_env({}).configured)


class HcxSettingsTests(unittest.TestCase):
    def test_enabled_by_default(self) -> None:
        self.assertTrue(HcxSettings.from_env({}).enabled)

    def test_reads_environment(self) -> None:
        settings = HcxSettings.from_env(
            {
                "FESTIVAL_HCX_ENABLED": "false",
                "FESTIVAL_HCX_API_URL": "https://clova.example/v1/chat/completions",
                "FESTIVAL_HCX_API_KEY": "key",
                "FESTIVAL_HCX_MODEL": "HCX-DASH-002",
                "FESTIVAL_HCX_TIMEOUT_SECONDS": "5",
                "FESTIVAL_HCX_MAX_TOKENS": "256",
            }
        )

        self.assertFalse(settings.enabled)
        self.assertTrue(settings.configured)
        self.assertEqual(settings.model, "HCX-DASH-002")
        self.assertEqual(settings.timeout_seconds, 5.0)
        self.assertEqual(settings.max_tokens, 256)

    def test_api_key_stays_out_of_repr(self) -> None:
        text = repr(_settings())

        self.assertNotIn("test-key", text)
        self.assertIn("clova.example", text)

    def test_request_headers_still_carry_the_key(self) -> None:
        self.assertEqual(
            _settings().request_headers(), {"Authorization": "Bearer test-key"}
        )

    def test_the_citation_attachment_status_is_distinct(self) -> None:
        self.assertNotEqual(
            CITATION_ATTACHMENT_FAILED_STATUS, PLACEHOLDER_INTEGRITY_FAILED
        )


if __name__ == "__main__":
    unittest.main()
