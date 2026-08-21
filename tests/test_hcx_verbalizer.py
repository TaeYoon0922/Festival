from __future__ import annotations

import json
import unittest

from app.generation.answer_generator import (
    GeneratedAnswer,
    GeneratedCitation,
    GeneratedSection,
)
from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer
from app.retrieval.embeddings import EmbeddingHttpError


DETERMINISTIC_TEXT = (
    "국민연금기금의 효성중공업 보유주식수는 2023년 03월 07일 기준 655,490주입니다.[1]"
)

FAITHFUL_TEXT = (
    "효성중공업에 대한 국민연금기금의 보유주식수는 2023년 03월 07일 기준으로 "
    "655,490주입니다.[1]"
)


def _generated(*, answerable: bool = True) -> GeneratedAnswer:
    return GeneratedAnswer(
        question="효성중공업 국민연금기금 변동일 변동후 주식수",
        answer_text=DETERMINISTIC_TEXT,
        citations=(
            GeneratedCitation(
                citation_id="1",
                chunk_id="h23:ch_report",
                doc_id="h23",
                source_refs=({"table_id": "t23", "row_start": 1, "row_end": 1},),
                section="보유주식등의 수 및 보유비율",
                evidence_type="table",
            ),
        ),
        sections=(
            GeneratedSection(
                title="보유 현황",
                content=DETERMINISTIC_TEXT,
                citations=("1",),
                metadata=("보고자: 국민연금기금",),
            ),
        ),
        warnings=(),
        confidence={"level": "high", "display_text": "높음"},
        answerable=answerable,
    )


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


class VerbalizerSuccessTests(unittest.TestCase):
    def test_faithful_reply_is_used(self) -> None:
        transport = _Transport(_reply(FAITHFUL_TEXT))
        verbalizer = HcxVerbalizer(_settings(), transport=transport)

        outcome = verbalizer.verbalize(_generated())

        self.assertEqual(outcome.status, "success")
        self.assertTrue(outcome.used_hcx)
        self.assertEqual(outcome.text, FAITHFUL_TEXT)

    def test_native_clova_response_shape_is_accepted(self) -> None:
        transport = _Transport({"result": {"message": {"content": FAITHFUL_TEXT}}})
        verbalizer = HcxVerbalizer(_settings(), transport=transport)

        outcome = verbalizer.verbalize(_generated())

        self.assertEqual(outcome.status, "success")
        self.assertEqual(outcome.text, FAITHFUL_TEXT)

    def test_request_uses_deterministic_generation_parameters(self) -> None:
        transport = _Transport(_reply(FAITHFUL_TEXT))
        HcxVerbalizer(_settings(model="HCX-005"), transport=transport).verbalize(
            _generated()
        )

        payload = transport.calls[0]["payload"]
        self.assertEqual(payload["model"], "HCX-005")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(transport.calls[0]["headers"]["Authorization"], "Bearer test-key")

    def test_request_never_carries_retrieved_chunks(self) -> None:
        transport = _Transport(_reply(FAITHFUL_TEXT))
        HcxVerbalizer(_settings(), transport=transport).verbalize(_generated())

        body = json.dumps(transport.calls[0]["payload"], ensure_ascii=False)
        self.assertNotIn("retrieval_text", body)
        self.assertNotIn("chunk_id", body)
        self.assertNotIn("source_refs", body)
        self.assertNotIn("provenance", body)


class OpenAiCompatibleContractTests(unittest.TestCase):
    """Lock the CLOVA Studio OpenAI compatibility contract.

    Two ways to break this request are guarded here.  The native
    ``/v3/chat-completions`` route takes camelCase fields such as ``maxTokens``
    and returns a different envelope.  And the compatibility layer does not
    support every OpenAI sampling field: ``top_p`` in particular is unsupported,
    so it must not be sent even though plain OpenAI would accept it.
    """

    def setUp(self) -> None:
        self.transport = _Transport(_reply(FAITHFUL_TEXT))
        HcxVerbalizer(_settings(), transport=self.transport).verbalize(_generated())
        self.call = self.transport.calls[0]

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
                        "message": {"role": "assistant", "content": FAITHFUL_TEXT},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )

        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(
            _generated()
        )

        self.assertEqual(outcome.status, "success")
        self.assertEqual(outcome.text, FAITHFUL_TEXT)


class VerbalizerFallbackTests(unittest.TestCase):
    def _fallback(self, **kwargs) -> tuple[str, str]:
        transport = _Transport(**kwargs)
        outcome = HcxVerbalizer(_settings(), transport=transport).verbalize(_generated())
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

    def test_hallucinated_number_falls_back(self) -> None:
        status, reason = self._fallback(
            response=_reply(DETERMINISTIC_TEXT + " 전년 대비 99,999주 증가했습니다.")
        )

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "numeric_token_changed")

    def test_changed_number_format_falls_back(self) -> None:
        status, reason = self._fallback(
            response=_reply(DETERMINISTIC_TEXT.replace("655,490주", "약 65만 주"))
        )

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "numeric_token_changed")

    def test_removed_citation_falls_back(self) -> None:
        status, reason = self._fallback(
            response=_reply(DETERMINISTIC_TEXT.replace("[1]", ""))
        )

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "citation_marker_changed")

    def test_added_citation_falls_back(self) -> None:
        status, reason = self._fallback(response=_reply(DETERMINISTIC_TEXT + "[2]"))

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "citation_marker_changed")

    def test_investment_language_falls_back(self) -> None:
        status, reason = self._fallback(
            response=_reply(DETERMINISTIC_TEXT + " 매수를 추천합니다.")
        )

        self.assertEqual(status, "fallback_validation_failed")
        self.assertEqual(reason, "forbidden_investment_language")

    def test_empty_reply_falls_back(self) -> None:
        status, _ = self._fallback(response=_reply("   "))

        self.assertEqual(status, "fallback_invalid_response")


class VerbalizerGatingTests(unittest.TestCase):
    def test_disabled_returns_the_deterministic_answer(self) -> None:
        transport = _Transport(_reply(FAITHFUL_TEXT))
        verbalizer = HcxVerbalizer(_settings(enabled=False), transport=transport)

        outcome = verbalizer.verbalize(_generated())

        self.assertEqual(outcome.status, "disabled")
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)
        self.assertEqual(transport.calls, [])

    def test_missing_credentials_return_the_deterministic_answer(self) -> None:
        transport = _Transport(_reply(FAITHFUL_TEXT))
        verbalizer = HcxVerbalizer(_settings(api_key=""), transport=transport)

        outcome = verbalizer.verbalize(_generated())

        self.assertEqual(outcome.status, "not_configured")
        self.assertEqual(transport.calls, [])

    def test_unanswerable_result_never_reaches_the_model(self) -> None:
        transport = _Transport(_reply(FAITHFUL_TEXT))
        verbalizer = HcxVerbalizer(_settings(), transport=transport)

        outcome = verbalizer.verbalize(_generated(answerable=False))

        self.assertEqual(outcome.status, "skipped_not_answerable")
        self.assertEqual(outcome.text, DETERMINISTIC_TEXT)
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
