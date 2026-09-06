"""The model may write the answer; it may not write a figure of its own.

Handing HyperCLOVA X the served filings is the ordinary way to answer from
retrieved documents, and it is safe here only because of what happens next:
every figure in the reply is looked up in those filings, and a reply stating a
number they do not contain is discarded whole. These tests are that boundary.
"""

from __future__ import annotations

import unittest

from app.generation.answer_synthesis import (
    STATUS_DISABLED,
    STATUS_NOT_CONFIGURED,
    STATUS_NOT_ELIGIBLE,
    STATUS_SUCCESS,
    AnswerSynthesizer,
    SynthesisRejected,
    accept_synthesis,
    citation_block,
    evidence_extracts,
    synthesis_prompt,
)
from app.generation.hcx_verbalizer import HcxSettings
from app.retrieval.embeddings import EmbeddingHttpError


ROWS = [
    {
        "rank": 1,
        "chunk_id": "periodic_a:ch_1",
        "doc_id": "periodic_a",
        "corp_name": "삼성전자",
        "report_nm": "사업보고서 (2025.12)",
        "rcept_dt": "2026-03-10",
        "section_path": ["III. 재무에 관한 사항", "2-2. 연결 손익계산서"],
        "content": "| 매출액 (주30) | 333,605,938 |",
    },
    {
        "rank": 2,
        "chunk_id": "periodic_b:ch_1",
        "doc_id": "periodic_b",
        "corp_name": "SK하이닉스",
        "report_nm": "사업보고서 (2025.12)",
        "rcept_dt": "2026-03-19",
        "section_path": [],
        "content": "| 매출액 | 66,192,960 |",
    },
]

CORPUS = ("삼성전자", "SK하이닉스", "현대자동차", "LG에너지솔루션")

GOOD = (
    "삼성전자의 2025년 매출액은 333,605,938입니다. [1] "
    "SK하이닉스의 매출액은 66,192,960입니다. [2]"
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


class PromptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extracts = evidence_extracts(ROWS)

    def test_each_filing_is_numbered_and_named(self) -> None:
        prompt = synthesis_prompt("삼성전자 매출액은?", self.extracts)

        self.assertIn("[1] 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10", prompt)
        self.assertIn("[2] SK하이닉스", prompt)
        self.assertIn("333,605,938", prompt)

    def test_the_question_leads(self) -> None:
        self.assertTrue(
            synthesis_prompt("삼성전자 매출액은?", self.extracts).startswith(
                "질문: 삼성전자 매출액은?"
            )
        )

    def test_a_chunk_without_content_is_not_an_extract(self) -> None:
        extracts = evidence_extracts([{"chunk_id": "c", "content": ""}])

        self.assertEqual(extracts, [])


class AcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extracts = evidence_extracts(ROWS)

    def _reject(self, reply: str) -> str:
        with self.assertRaises(SynthesisRejected) as raised:
            accept_synthesis(reply, self.extracts, corpus_companies=CORPUS)
        return raised.exception.reason

    def _accept(self, reply: str) -> str:
        return accept_synthesis(reply, self.extracts, corpus_companies=CORPUS)

    def test_an_answer_from_the_filings_is_accepted(self) -> None:
        self.assertEqual(self._accept(GOOD), GOOD)

    def test_a_figure_the_filings_do_not_state_is_refused(self) -> None:
        reply = "삼성전자의 매출액은 999,999,999입니다. [1]"

        self.assertEqual(self._reject(reply), "unsupported_number")

    def test_a_rescaled_figure_is_refused(self) -> None:
        """333,605,938 백만원 restated as 333조원 is a different claim."""

        reply = "삼성전자의 매출액은 약 333조원입니다. [1]"

        self.assertEqual(self._reject(reply), "rescaled_number")

    def test_an_answer_with_no_citation_is_refused(self) -> None:
        self.assertEqual(
            self._reject("삼성전자의 매출액은 333,605,938입니다."), "no_citation"
        )

    def test_a_citation_to_a_filing_not_served_is_refused(self) -> None:
        reply = "삼성전자의 매출액은 333,605,938입니다. [7]"

        self.assertEqual(self._reject(reply), "unknown_citation")

    def test_investment_advice_is_refused(self) -> None:
        reply = "삼성전자의 매출액은 333,605,938입니다. [1] 매수를 추천합니다."

        self.assertEqual(self._reject(reply), "evaluative_wording")

    def test_an_issuer_the_filings_do_not_name_is_refused(self) -> None:
        reply = "현대자동차의 매출액은 333,605,938입니다. [1]"

        self.assertEqual(self._reject(reply), "unsupplied_company")

    def test_a_markdown_fence_is_refused(self) -> None:
        self.assertEqual(self._reject(f"```\n{GOOD}\n```"), "markdown_fence")

    def test_an_empty_reply_is_refused(self) -> None:
        self.assertEqual(self._reject("  "), "empty")

    def test_saying_the_filings_do_not_answer_is_allowed(self) -> None:
        reply = "제공된 공시에서는 영업이익을 확인할 수 없습니다. [1]"

        self.assertEqual(self._accept(reply), reply)

    def test_a_comparison_the_figures_support_is_allowed(self) -> None:
        reply = (
            "삼성전자 333,605,938, SK하이닉스 66,192,960으로 "
            "삼성전자가 더 큽니다. [1] [2]"
        )

        self.assertIn("더 큽니다", self._accept(reply))

    def test_a_citation_number_is_not_read_as_a_figure(self) -> None:
        """[2] is the answer's own numbering, not a claim that 2 was filed."""

        self.assertEqual(self._accept(GOOD), GOOD)


class CitationBlockTests(unittest.TestCase):
    def test_only_the_cited_filings_are_listed(self) -> None:
        block = citation_block(
            "삼성전자의 매출액은 333,605,938입니다. [1]", evidence_extracts(ROWS)
        )

        self.assertIn("공시: 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10", block)
        self.assertNotIn("SK하이닉스", block)

    def test_an_uncited_answer_gets_no_block(self) -> None:
        self.assertEqual(citation_block("근거 없음", evidence_extracts(ROWS)), "")


class SynthesizerTests(unittest.TestCase):
    def test_an_accepted_answer_comes_back_with_its_citations(self) -> None:
        transport = _StubTransport(GOOD)

        outcome, citations = AnswerSynthesizer(
            _settings(), transport=transport
        ).synthesize("매출액 비교", ROWS, corpus_companies=CORPUS)

        self.assertEqual(outcome.status, STATUS_SUCCESS)
        self.assertEqual(outcome.text, GOOD)
        self.assertIn("doc_id: periodic_a", citations)

    def test_a_refused_answer_returns_nothing(self) -> None:
        transport = _StubTransport("삼성전자의 매출액은 12,345입니다. [1]")

        outcome, citations = AnswerSynthesizer(
            _settings(), transport=transport
        ).synthesize("매출액", ROWS, corpus_companies=CORPUS)

        self.assertTrue(outcome.status.startswith("rejected:"))
        self.assertIsNone(outcome.text)
        self.assertEqual(citations, "")

    def test_a_transport_failure_returns_nothing(self) -> None:
        transport = _StubTransport(
            error=EmbeddingHttpError("boom", status_code=503, transient=True)
        )

        outcome, _ = AnswerSynthesizer(
            _settings(), transport=transport
        ).synthesize("매출액", ROWS)

        self.assertEqual(outcome.status, "transport_failure")

    def test_no_evidence_is_not_eligible(self) -> None:
        transport = _StubTransport(GOOD)

        outcome, _ = AnswerSynthesizer(_settings(), transport=transport).synthesize(
            "매출액", []
        )

        self.assertEqual(outcome.status, STATUS_NOT_ELIGIBLE)
        self.assertEqual(transport.payloads, [])

    def test_a_disabled_model_is_not_called(self) -> None:
        transport = _StubTransport(GOOD)

        outcome, _ = AnswerSynthesizer(
            _settings(enabled=False), transport=transport
        ).synthesize("매출액", ROWS)

        self.assertEqual(outcome.status, STATUS_DISABLED)
        self.assertEqual(transport.payloads, [])

    def test_the_layer_switch_alone_turns_it_off(self) -> None:
        transport = _StubTransport(GOOD)

        outcome, _ = AnswerSynthesizer(
            _settings(), transport=transport, enabled=False
        ).synthesize("매출액", ROWS)

        self.assertEqual(outcome.status, STATUS_DISABLED)
        self.assertEqual(transport.payloads, [])

    def test_a_missing_key_is_not_called(self) -> None:
        transport = _StubTransport(GOOD)

        outcome, _ = AnswerSynthesizer(
            _settings(api_key=None), transport=transport
        ).synthesize("매출액", ROWS)

        self.assertEqual(outcome.status, STATUS_NOT_CONFIGURED)
        self.assertEqual(transport.payloads, [])


if __name__ == "__main__":
    unittest.main()
