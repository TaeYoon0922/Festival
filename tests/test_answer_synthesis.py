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
        """The rule that refused it.

        A reason now carries what it caught -- ``unsupported_number:12,345`` --
        so a live rejection can be diagnosed from the trace without the reply
        itself. The rule is the part before the colon.
        """

        with self.assertRaises(SynthesisRejected) as raised:
            accept_synthesis(reply, self.extracts, corpus_companies=CORPUS)
        return raised.exception.reason.split(":", 1)[0]

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

    def test_a_missing_citation_is_traced_to_the_filing(self) -> None:
        """Attribution is a lookup, not something only the model can supply."""

        accepted = self._accept("삼성전자의 매출액은 333,605,938입니다.")

        self.assertIn("[1]", accepted)
        self.assertNotIn("[2]", accepted)

    def test_an_answer_with_no_traceable_figure_is_refused(self) -> None:
        self.assertEqual(
            self._reject("두 회사의 실적을 확인했습니다."), "no_citation"
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

    def test_markup_is_removed_rather_than_refused(self) -> None:
        """Bold is formatting. Refusing over it lost 24 of 30 live replies.

        The run discarded four fifths of its own good answers for arriving in
        bold with a bulleted calculation under them, so the markup comes off
        and the sentences stay. None of it could carry a fact.
        """

        reply = (
            "삼성전자 매출액은 **333,605,938**입니다. [1]\n"
            "- SK하이닉스는 66,192,960입니다. [2]"
        )

        accepted = self._accept(reply)

        self.assertNotIn("**", accepted)
        self.assertNotIn("- SK", accepted)
        self.assertIn("333,605,938", accepted)
        self.assertIn("[2]", accepted)

    def test_an_empty_reply_is_refused(self) -> None:
        self.assertEqual(self._reject("  "), "empty")

    def test_saying_the_filings_do_not_answer_is_allowed(self) -> None:
        reply = "제공된 공시에서는 영업이익을 확인할 수 없습니다. [1]"

        self.assertEqual(self._accept(reply), reply)

    def test_a_difference_worked_out_from_the_figures_is_allowed(self) -> None:
        """The gap is in no filing, and is still checkable arithmetic."""

        reply = (
            "삼성전자 333,605,938, SK하이닉스 66,192,960으로 "
            "두 수치의 차이는 267,412,978입니다. [1] [2]"
        )

        self.assertIn("267,412,978", self._accept(reply))

    def test_a_difference_that_is_not_the_difference_is_refused(self) -> None:
        reply = "삼성전자가 SK하이닉스보다 999,999,999 더 큽니다. [1] [2]"

        self.assertEqual(self._reject(reply), "unsupported_number")

    def test_a_total_written_at_a_scale_is_allowed(self) -> None:
        """A 758,900,000,000 total is written 7,589억, and that is the answer.

        The operand cap used to stop at the first extract, so a sum whose
        second figure sat in the next one was refused, and the scale word made
        the digits look like a different number again.
        """

        extracts = evidence_extracts(
            [
                {"chunk_id": "a", "doc_id": "a", "corp_name": "LG이노텍",
                 "report_nm": "신규시설투자등", "rcept_dt": "2024-11-21",
                 "section_path": [],
                 "content": "| 투자금액(원) | 375,900,000,000 |"},
                {"chunk_id": "b", "doc_id": "b", "corp_name": "LG이노텍",
                 "report_nm": "신규시설투자등", "rcept_dt": "2024-02-20",
                 "section_path": [],
                 "content": "| 투자금액(원) | 383,000,000,000 |"},
            ]
        )
        reply = "LG이노텍의 2024년 시설투자 합계는 7,589억원입니다. [1] [2]"

        self.assertIn("7,589억", accept_synthesis(reply, extracts))

    def test_an_abbreviation_of_a_named_issuer_is_allowed(self) -> None:
        """SKT beside an extract headed SK텔레콤 is not another company."""

        extracts = evidence_extracts(
            [
                {"chunk_id": "a", "doc_id": "a", "corp_name": "SK텔레콤",
                 "report_nm": "사업보고서 (2024.12)", "rcept_dt": "2025-03-11",
                 "section_path": [],
                 "content": "| 매출액 | 17,940,860 |"},
            ]
        )
        reply = "SKT의 2024년 매출액은 17,940,860입니다. [1]"

        self.assertIn("SKT", accept_synthesis(reply, extracts, corpus_companies=CORPUS))

    def test_describing_which_is_stronger_is_allowed(self) -> None:
        """"어느 게임사가 더 수익성이 좋아" asks for exactly that word."""

        reply = "삼성전자의 매출액 333,605,938이 더 우수한 수준입니다. [1]"

        self.assertIn("우수한", self._accept(reply))

    def test_a_share_worked_out_from_the_figures_is_allowed(self) -> None:
        reply = "SK하이닉스 매출액은 삼성전자의 19.8% 수준입니다. [1] [2]"

        self.assertIn("19.8%", self._accept(reply))

    def test_a_comparison_the_figures_support_is_still_refused(self) -> None:
        reply = (
            "삼성전자 333,605,938, SK하이닉스 66,192,960으로 "
            "삼성전자가 더 큽니다. [1] [2]"
        )

        self.assertEqual(self._reject(reply), "comparison_verdict")

    def test_verdicts_are_refused_even_when_the_filing_uses_them(self) -> None:
        verdicts = (
            ("더 크다", "더 크"), ("더 큰 쪽입니다", "더 큰"),
            ("더 큽니다", "더 큽"), ("더 커요", "더 커"),
            ("더 높습니다", "더 높"), ("더 많습니다", "더 많"),
            ("더 적습니다", "더 적"), ("더 낮습니다", "더 낮"),
            ("가장 크다", "가장 크"), ("가장 큰 쪽입니다", "가장 큰"),
            ("가장 큽니다", "가장 큽"), ("가장 커요", "가장 커"),
            ("가장 높습니다", "가장 높"), ("증가했습니다", "증가했"),
            ("감소했습니다", "감소했"), ("상회합니다", "상회"),
            ("하회합니다", "하회"),
        )
        for phrase, reason in verdicts:
            for source_contains_verdict in (False, True):
                with self.subTest(phrase=phrase, in_source=source_contains_verdict):
                    extracts = evidence_extracts(ROWS)
                    if source_contains_verdict:
                        extracts[0]["content"] += f" {phrase}."
                    with self.assertRaises(SynthesisRejected) as raised:
                        accept_synthesis(f"{GOOD} {phrase}.", extracts)
                    self.assertEqual(raised.exception.reason, f"comparison_verdict:{reason}")

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
    def test_a_verdict_is_discarded_without_retrying(self) -> None:
        transport = _StubTransport(f"{GOOD} SK하이닉스가 더 큽니다. [2]")
        synthesizer = AnswerSynthesizer(_settings(), transport=transport)

        outcome, citations = synthesizer.synthesize("매출액 비교", ROWS)

        self.assertEqual(outcome.status, "rejected:comparison_verdict:더 큽")
        self.assertIsNone(outcome.text)
        self.assertEqual(citations, "")
        self.assertEqual(len(transport.payloads), 1)
        self.assertEqual(synthesizer.call_count, 1)

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
