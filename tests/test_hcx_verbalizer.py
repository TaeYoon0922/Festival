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
    PLACEHOLDER_INTEGRITY_FAILED,
    SKIPPED_NO_COMPACT_CLAIM,
    HcxSettings,
    HcxVerbalizer,
)
from app.generation.protected_literals import protect_literals, restore_literals
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


def _holding_context():
    """Run the production holding path once and keep what the verbalizer needs."""

    pair = _holding_pair(
        "h23:ch", "h23", rank=1, date="2023-06-30",
        projection_type="holding_report", table_id="t23",
    )
    plan = QueryPlan(
        query="효성중공업 국민연금기금 변동일 변동후 주식수",
        task_type="holding_change",
        metric="holding_shares",
        reporter="국민연금기금",
        disclosure_route=("holding",),
        evidence={"requested_holding_fields": ["reference_date", "after_shares"]},
    )
    result = AgentOrchestrator().run(plan.raw_query, plan, _execution(plan, pair))
    generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
    claim = build_compact_claim(
        result.answer_draft, result.resolution, task_type=result.task_decision.task_type
    )
    return result, generated, claim


RESULT, GENERATED, CLAIM = _holding_context()

#: The full deterministic answer, served whenever HCX is skipped or rejected.
DETERMINISTIC_TEXT = GENERATED.answer_text

#: The compact claim is what HCX is asked to restate and is judged against.
CLAIM_TEXT = CLAIM.deterministic_text
PROTECTION = protect_literals(CLAIM_TEXT)

MASKED_FAITHFUL = f"{PROTECTION.masked}입니다"
FAITHFUL_TEXT = restore_literals(MASKED_FAITHFUL, PROTECTION)


def _verbalizer_kwargs() -> dict:
    return {
        "draft": RESULT.answer_draft,
        "resolution": RESULT.resolution,
        "task_type": RESULT.task_decision.task_type,
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


def _verbalize(reply: str, **overrides):
    transport = _Transport(_reply(reply))
    verbalizer = HcxVerbalizer(_settings(**overrides), transport=transport)
    return verbalizer.verbalize(GENERATED, **_verbalizer_kwargs()), transport


class CompactClaimInputTests(unittest.TestCase):
    """HCX restates a short claim, never the whole evidence report."""

    def test_the_claim_is_far_smaller_than_the_full_answer(self) -> None:
        self.assertLess(len(CLAIM_TEXT), len(DETERMINISTIC_TEXT))
        self.assertLess(len(PROTECTION.placeholders), 10)

    def test_the_request_carries_the_masked_claim_not_the_report(self) -> None:
        _, transport = _verbalize(MASKED_FAITHFUL)

        body = json.dumps(transport.calls[0]["payload"], ensure_ascii=False)
        self.assertIn(PROTECTION.masked, body)
        self.assertNotIn(DETERMINISTIC_TEXT, body)

    def test_the_request_never_carries_raw_literals_or_chunks(self) -> None:
        _, transport = _verbalize(MASKED_FAITHFUL)

        body = json.dumps(transport.calls[0]["payload"], ensure_ascii=False)
        self.assertNotIn("2023-06-30", body)
        self.assertNotIn("1,000", body)
        self.assertNotIn("[1]", body)
        self.assertNotIn("chunk_id", body)
        self.assertNotIn("source_refs", body)


class VerbalizerSuccessTests(unittest.TestCase):
    def test_faithful_reply_is_used(self) -> None:
        outcome, _ = _verbalize(MASKED_FAITHFUL)

        self.assertEqual(outcome.status, "success")
        self.assertTrue(outcome.used_hcx)
        self.assertEqual(outcome.text, FAITHFUL_TEXT)

    def test_native_clova_response_shape_is_accepted(self) -> None:
        transport = _Transport({"result": {"message": {"content": MASKED_FAITHFUL}}})
        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            GENERATED, **_verbalizer_kwargs()
        )

        self.assertEqual(outcome.status, "success")
        self.assertEqual(outcome.text, FAITHFUL_TEXT)

    def test_literals_are_restored_verbatim(self) -> None:
        outcome, _ = _verbalize(MASKED_FAITHFUL)

        self.assertIn("2023-06-30", outcome.text)
        self.assertIn("1,000주", outcome.text)
        self.assertIn("[1]", outcome.text)

    def test_wording_only_change_succeeds(self) -> None:
        date, first_marker, number, second_marker = PROTECTION.placeholders
        reply = (
            f"국민연금기금이 보유한 테스트회사 주식은 {date} 기준{first_marker} "
            f"{number}주입니다{second_marker}"
        )

        outcome, _ = _verbalize(reply)

        self.assertEqual(outcome.status, "success")
        self.assertIn("2023-06-30", outcome.text)
        self.assertIn("1,000주", outcome.text)


class OpenAiCompatibleContractTests(unittest.TestCase):
    """Lock the CLOVA Studio OpenAI compatibility contract.

    Two ways to break this request are guarded here.  The native
    ``/v3/chat-completions`` route takes camelCase fields such as ``maxTokens``
    and returns a different envelope.  And the compatibility layer does not
    support every OpenAI sampling field: ``top_p`` in particular is unsupported,
    so it must not be sent even though plain OpenAI would accept it.
    """

    def setUp(self) -> None:
        _, transport = _verbalize(MASKED_FAITHFUL)
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

    def test_reads_content_from_the_openai_envelope(self) -> None:
        transport = _Transport(
            {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "HCX-005",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": MASKED_FAITHFUL},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            GENERATED, **_verbalizer_kwargs()
        )

        self.assertEqual(outcome.status, "success")


class VerbalizerFallbackTests(unittest.TestCase):
    """Every failure serves the full deterministic answer."""

    def _fallback(self, **kwargs) -> tuple[str, str]:
        transport = _Transport(**kwargs)
        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            GENERATED, **_verbalizer_kwargs()
        )
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)
        return outcome.status, outcome.reason or ""

    def test_timeout_falls_back(self) -> None:
        status, _ = self._fallback(
            error=EmbeddingHttpError("timed out", status_code=None, transient=True)
        )

        self.assertEqual(status, "fallback_timeout")

    def test_raw_timeout_error_falls_back(self) -> None:
        status, _ = self._fallback(error=TimeoutError("timed out"))

        self.assertEqual(status, "fallback_timeout")

    def test_http_error_falls_back(self) -> None:
        status, reason = self._fallback(
            error=EmbeddingHttpError("boom", status_code=500, transient=True)
        )

        self.assertEqual(status, "fallback_http_error")
        self.assertEqual(reason, "HTTP 500")

    def test_unexpected_exception_falls_back(self) -> None:
        status, _ = self._fallback(error=ValueError("not JSON"))

        self.assertEqual(status, "fallback_error")

    def test_missing_content_falls_back(self) -> None:
        status, _ = self._fallback(response={"choices": []})

        self.assertEqual(status, "fallback_invalid_response")

    def test_empty_reply_falls_back(self) -> None:
        status, _ = self._fallback(response=_reply("   "))

        self.assertEqual(status, "fallback_invalid_response")

    def test_hallucinated_number_falls_back(self) -> None:
        """A number typed fresh is not a placeholder, so the validator catches it."""

        status, reason = self._fallback(
            response=_reply(MASKED_FAITHFUL + " 전년 대비 99,999주 증가했습니다.")
        )

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "numeric_token_changed")

    def test_added_citation_falls_back(self) -> None:
        status, reason = self._fallback(response=_reply(MASKED_FAITHFUL + "[9]"))

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "citation_marker_changed")

    def test_investment_language_falls_back(self) -> None:
        status, reason = self._fallback(
            response=_reply(MASKED_FAITHFUL + " 매수를 추천합니다.")
        )

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "forbidden_investment_language")


class PlaceholderIntegrityFallbackTests(unittest.TestCase):
    """A mangled placeholder is never guessed back into a literal."""

    def _integrity_failure(self, reply: str) -> str:
        outcome, _ = _verbalize(reply)
        self.assertEqual(outcome.status, PLACEHOLDER_INTEGRITY_FAILED)
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)
        return outcome.reason or ""

    def test_deleted_placeholder_falls_back(self) -> None:
        reason = self._integrity_failure(
            MASKED_FAITHFUL.replace(PROTECTION.placeholders[0], "")
        )

        self.assertEqual(reason, "placeholder_missing")

    def test_added_placeholder_falls_back(self) -> None:
        reason = self._integrity_failure(MASKED_FAITHFUL + " __FESTIVAL_NUMBER_Z__")

        self.assertEqual(reason, "placeholder_unexpected")

    def test_duplicated_placeholder_falls_back(self) -> None:
        reason = self._integrity_failure(
            MASKED_FAITHFUL + PROTECTION.placeholders[0]
        )

        self.assertEqual(reason, "placeholder_duplicated")

    def test_reordered_placeholders_fall_back(self) -> None:
        date, first_marker, number, second_marker = PROTECTION.placeholders
        reply = f"{number}주{second_marker} 기준일 {date}{first_marker}"

        reason = self._integrity_failure(reply)

        self.assertEqual(reason, "placeholder_reordered")

    def test_placeholder_rewritten_as_a_literal_falls_back(self) -> None:
        reason = self._integrity_failure(
            MASKED_FAITHFUL.replace(PROTECTION.placeholders[0], "2023년 6월 30일")
        )

        self.assertEqual(reason, "placeholder_missing")

    def test_the_live_failure_mode_can_no_longer_corrupt_literals(self) -> None:
        """The model writing its own prose instead of restating the claim."""

        outcome, _ = _verbalize(
            "국민연금기금의 테스트회사 보유주식 수는 **2023년 6월 30일** 기준 "
            "**1,000주**입니다."
        )

        self.assertEqual(outcome.status, PLACEHOLDER_INTEGRITY_FAILED)
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)


class CompactClaimSkipTests(unittest.TestCase):
    """No compact claim means no model call, and the deterministic answer stands."""

    def _skip(self, generated, **kwargs) -> tuple[str, list]:
        transport = _Transport(_reply(MASKED_FAITHFUL))
        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            generated, **kwargs
        )
        return outcome, transport.calls

    def test_missing_draft_skips_without_calling_the_model(self) -> None:
        outcome, calls = self._skip(GENERATED)

        self.assertEqual(outcome.status, SKIPPED_NO_COMPACT_CLAIM)
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)
        self.assertEqual(calls, [])

    def test_answerable_periodic_fact_still_skips(self) -> None:
        """P07's shape: the answer is sound, but its value lives in free text."""

        older = _periodic_item(
            "p23:ch", "p23", rank=1, text="연료전지 주기기 매출액은 1,234억원입니다.", year=2023
        )
        newer = _periodic_item(
            "p24:ch", "p24", rank=2, text="연료전지 주기기 매출액은 1,234억원입니다.", year=2024
        )
        evidence = _periodic_evidence(
            [_periodic_group("g1", older, newer, group_type="periodic_repeated_fact")]
        )
        resolution = resolve_periodic_facts(evidence)
        draft = compose_periodic_answer(resolution, evidence)
        generated = CitationAwareAnswerGenerator().generate(draft)
        self.assertTrue(generated.answerable)

        outcome, calls = self._skip(
            generated,
            draft=draft,
            resolution=resolution,
            task_type="periodic_fact",
        )

        self.assertEqual(outcome.status, SKIPPED_NO_COMPACT_CLAIM)
        self.assertEqual(outcome.text, generated.answer_text)
        self.assertEqual(calls, [])

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

        outcome, calls = self._skip(
            generated,
            draft=result.answer_draft,
            resolution=result.resolution,
            task_type=result.task_decision.task_type,
        )

        self.assertEqual(outcome.status, SKIPPED_NO_COMPACT_CLAIM)
        self.assertEqual(calls, [])


class VerbalizerGatingTests(unittest.TestCase):
    def test_disabled_returns_the_deterministic_answer(self) -> None:
        transport = _Transport(_reply(MASKED_FAITHFUL))
        outcome = HcxVerbalizer(
            _settings(enabled=False), transport=transport
        ).verbalize(GENERATED, **_verbalizer_kwargs())

        self.assertEqual(outcome.status, "disabled")
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)
        self.assertEqual(transport.calls, [])

    def test_missing_credentials_return_the_deterministic_answer(self) -> None:
        transport = _Transport(_reply(MASKED_FAITHFUL))
        outcome = HcxVerbalizer(
            _settings(api_key=""), transport=transport
        ).verbalize(GENERATED, **_verbalizer_kwargs())

        self.assertEqual(outcome.status, "not_configured")
        self.assertEqual(transport.calls, [])

    def test_unanswerable_result_never_reaches_the_model(self) -> None:
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
                GeneratedSection(title="확인 필요", content="확인되지 않은 정보가 있습니다.", citations=()),
            ),
            warnings=("answer_not_supported",),
            confidence={"level": "low", "display_text": "낮음"},
            answerable=False,
        )
        transport = _Transport(_reply(MASKED_FAITHFUL))

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            unanswerable, **_verbalizer_kwargs()
        )

        self.assertEqual(outcome.status, "skipped_not_answerable")
        self.assertEqual(outcome.text, unanswerable.answer_text)
        self.assertEqual(transport.calls, [])


class HcxSettingsTests(unittest.TestCase):
    def test_enabled_by_default(self) -> None:
        self.assertTrue(HcxSettings.from_env({}).enabled)

    def test_unconfigured_without_credentials(self) -> None:
        self.assertFalse(HcxSettings.from_env({}).configured)

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


if __name__ == "__main__":
    unittest.main()
