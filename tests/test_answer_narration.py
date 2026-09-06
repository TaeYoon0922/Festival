"""The rewrite may change wording, and may change nothing else.

Every test here is about that boundary. The model is never shown a figure, so
these check what happens to a reply: one that restated the masked text is
restored exactly, and one that wrote a number, dropped a token, named another
issuer or drew a conclusion is discarded and the answer is served as built.
"""

from __future__ import annotations

import unittest

from app.generation.answer_narration import (
    STATUS_DISABLED,
    STATUS_NOT_CONFIGURED,
    STATUS_NOT_ELIGIBLE,
    STATUS_SUCCESS,
    STATUS_TOO_LONG,
    AnswerNarrator,
    NarrationRejected,
    accept_narration,
    narration_source,
    split_citation_block,
)
from app.generation.hcx_verbalizer import HcxSettings
from app.generation.protected_literals import protect_literals
from app.retrieval.embeddings import EmbeddingHttpError


ANSWER = """정기공시 근거 1
근거 1 보고 기간: 2025년
근거 1 보고서: 사업보고서 (2025.12)
근거 1 재무제표 기준: 연결
확인된 사업 또는 공시 내용:
1.
내용: | 매출액 (주30) | 333,605,938 | [1]

신뢰도
답변 신뢰도: 높음

인용
[1]
doc_id: periodic_20260310002820
chunk_id: periodic_20260310002820:ch_70f7
공시: 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10"""


def _protection(answer: str = ANSWER):
    body, citations = split_citation_block(answer)
    return protect_literals(narration_source(body)), citations


def _faithful_reply(protection) -> str:
    """A reply that restates the masked text, keeping every token in order."""

    return " ".join(
        f"{token}에 관한 내용입니다." for token in protection.placeholders
    )


class _StubTransport:
    def __init__(self, content: str | None = None, error: Exception | None = None):
        self.content = content
        self.error = error
        self.payloads: list[dict] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        del url, headers, timeout_seconds
        self.payloads.append(payload)
        if self.error is not None:
            raise self.error
        return {"choices": [{"message": {"content": self.content}}]}


def _settings(**kwargs) -> HcxSettings:
    defaults = {
        "enabled": True,
        "endpoint": "https://clova.example/v1/chat/completions",
        "api_key": "secret",
    }
    defaults.update(kwargs)
    return HcxSettings(**defaults)


class SplitTests(unittest.TestCase):
    def test_the_citation_block_is_held_back(self) -> None:
        body, citations = split_citation_block(ANSWER)

        self.assertNotIn("doc_id", body)
        self.assertTrue(citations.startswith("인용"))
        self.assertIn("chunk_id", citations)

    def test_an_answer_without_citations_is_all_body(self) -> None:
        body, citations = split_citation_block("한 줄짜리 답변입니다.")

        self.assertEqual(body, "한 줄짜리 답변입니다.")
        self.assertEqual(citations, "")

    def test_the_last_heading_wins(self) -> None:
        """A filing that says 인용 in its prose is not the citation block."""

        answer = "본문에서 인용\n한 문장입니다.\n\n인용\ndoc_id: d1"
        body, citations = split_citation_block(answer)

        self.assertIn("본문에서 인용", body)
        self.assertEqual(citations, "인용\ndoc_id: d1")


class NarrationSourceTests(unittest.TestCase):
    def test_scaffolding_ordinals_leave_the_placeholder_set(self) -> None:
        body, _ = split_citation_block(ANSWER)

        before = len(protect_literals(body).literals)
        after = len(protect_literals(narration_source(body)).literals)

        self.assertLess(after, before)

    def test_no_figure_is_removed(self) -> None:
        body, _ = split_citation_block(ANSWER)

        self.assertIn("333,605,938", narration_source(body))
        self.assertIn("2025.12", narration_source(body))

    def test_the_bare_ordinal_and_content_label_are_dropped(self) -> None:
        source = narration_source("1.\n내용: 어떤 표\n확인된 내용")

        self.assertEqual(source, "어떤 표\n확인된 내용")


class AcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protection, _ = _protection()

    def _reject(self, reply: str) -> str:
        with self.assertRaises(NarrationRejected) as raised:
            accept_narration(reply, self.protection)
        return raised.exception.reason

    def test_a_faithful_restatement_is_restored_exactly(self) -> None:
        restored = accept_narration(_faithful_reply(self.protection), self.protection)

        self.assertIn("333,605,938", restored)
        self.assertIn("[1]", restored)
        self.assertNotIn("__FESTIVAL", restored)

    def test_a_written_digit_is_refused(self) -> None:
        reply = _faithful_reply(self.protection) + " 매출은 12,345입니다."

        self.assertEqual(self._reject(reply), "digit")

    def test_a_dropped_token_is_refused(self) -> None:
        reply = _faithful_reply(self.protection).replace(
            self.protection.placeholders[0], ""
        )

        self.assertEqual(self._reject(reply), "placeholder_missing")

    def test_a_reordered_token_run_is_refused(self) -> None:
        first, second = self.protection.placeholders[:2]
        reply = _faithful_reply(self.protection)
        reply = reply.replace(first, "@@").replace(second, first).replace("@@", second)

        self.assertEqual(self._reject(reply), "placeholder_reordered")

    def test_an_invented_citation_marker_is_refused(self) -> None:
        reply = _faithful_reply(self.protection) + " 근거는 [2]입니다."

        self.assertEqual(self._reject(reply), "citation_marker")

    def test_a_conclusion_is_refused(self) -> None:
        reply = _faithful_reply(self.protection) + " 이 회사가 가장 큽니다."

        self.assertEqual(self._reject(reply), "evaluative_wording")

    def test_an_unsupplied_issuer_is_refused(self) -> None:
        reply = _faithful_reply(self.protection) + " 현대자동차의 자료입니다."

        self.assertEqual(
            self._reject(reply),
            "unsupplied_company",
        )

    def test_an_issuer_the_answer_names_is_allowed(self) -> None:
        """삼성전자 is in the text being rewritten, so restating it is not new."""

        protection, _ = _protection("삼성전자의 매출액은 333,605,938입니다. " * 3)
        reply = " ".join(
            f"삼성전자 {token}" for token in protection.placeholders
        )

        self.assertIn("삼성전자", accept_narration(reply, protection))

    def test_an_empty_reply_is_refused(self) -> None:
        self.assertEqual(self._reject("   "), "empty")

    def test_a_markdown_fence_is_refused(self) -> None:
        reply = "```\n" + _faithful_reply(self.protection) + "\n```"

        self.assertEqual(self._reject(reply), "markdown_fence")

    def test_a_reply_that_grew_is_refused(self) -> None:
        reply = _faithful_reply(self.protection) + " 설명을 덧붙입니다." * 200

        self.assertEqual(self._reject(reply), "expanded")


class NarratorTests(unittest.TestCase):
    def test_a_faithful_reply_is_served_with_its_citations(self) -> None:
        protection, citations = _protection()
        transport = _StubTransport(_faithful_reply(protection))

        outcome = AnswerNarrator(_settings(), transport=transport).narrate(ANSWER)

        self.assertEqual(outcome.status, STATUS_SUCCESS)
        self.assertTrue(outcome.succeeded)
        self.assertTrue(outcome.text.endswith(citations))
        self.assertIn("333,605,938", outcome.text)

    def test_the_model_is_never_shown_a_figure(self) -> None:
        protection, _ = _protection()
        transport = _StubTransport(_faithful_reply(protection))

        AnswerNarrator(_settings(), transport=transport).narrate(ANSWER)

        sent = transport.payloads[0]["messages"][-1]["content"]
        self.assertNotIn("333,605,938", sent)
        self.assertNotIn("2025", sent)
        self.assertIn("__FESTIVAL_", sent)

    def test_the_citation_block_is_never_sent(self) -> None:
        protection, _ = _protection()
        transport = _StubTransport(_faithful_reply(protection))

        AnswerNarrator(_settings(), transport=transport).narrate(ANSWER)

        sent = transport.payloads[0]["messages"][-1]["content"]
        self.assertNotIn("doc_id", sent)
        self.assertNotIn("chunk_id", sent)

    def test_a_refused_reply_returns_no_narration(self) -> None:
        transport = _StubTransport("매출액은 999,999입니다.")

        outcome = AnswerNarrator(_settings(), transport=transport).narrate(ANSWER)

        self.assertFalse(outcome.succeeded)
        self.assertTrue(outcome.status.startswith("rejected:"))
        self.assertIsNone(outcome.text)

    def test_a_transport_failure_returns_no_narration(self) -> None:
        transport = _StubTransport(
            error=EmbeddingHttpError("boom", status_code=503, transient=True)
        )

        outcome = AnswerNarrator(_settings(), transport=transport).narrate(ANSWER)

        self.assertEqual(outcome.status, "transport_failure")
        self.assertIsNone(outcome.text)

    def test_a_short_answer_is_not_eligible(self) -> None:
        outcome = AnswerNarrator(_settings(), transport=_StubTransport("x")).narrate(
            "확인할 수 없습니다."
        )

        self.assertEqual(outcome.status, STATUS_NOT_ELIGIBLE)

    def test_a_long_answer_is_skipped(self) -> None:
        transport = _StubTransport("x")

        outcome = AnswerNarrator(_settings(), transport=transport).narrate(
            "가" * 7000
        )

        self.assertEqual(outcome.status, STATUS_TOO_LONG)
        self.assertEqual(transport.payloads, [])

    def test_a_disabled_model_is_not_called(self) -> None:
        transport = _StubTransport("x")

        outcome = AnswerNarrator(
            _settings(enabled=False), transport=transport
        ).narrate(ANSWER)

        self.assertEqual(outcome.status, STATUS_DISABLED)
        self.assertEqual(transport.payloads, [])

    def test_a_missing_key_is_not_called(self) -> None:
        transport = _StubTransport("x")

        outcome = AnswerNarrator(
            _settings(api_key=None), transport=transport
        ).narrate(ANSWER)

        self.assertEqual(outcome.status, STATUS_NOT_CONFIGURED)
        self.assertEqual(transport.payloads, [])


if __name__ == "__main__":
    unittest.main()
