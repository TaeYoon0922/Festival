"""The lead may frame the answer; it may never make a claim.

The model is handed company names, filing kinds, the asker's own subject words
and placeholders -- no figure, no citation, no filing text. So anything factual
in its reply was invented, and these tests pin the checks that discard it. The
acceptance rules are tested without a transport, because a safety rule that can
only be exercised through an HTTP call is a safety rule nobody runs.
"""

from __future__ import annotations

import unittest

from app.generation.answer_lead import (
    MAX_LEAD_CHARS,
    STATUS_DISABLED,
    STATUS_NOT_CONFIGURED,
    STATUS_NOT_ELIGIBLE,
    AnswerLeadWriter,
    LeadRejected,
    LeadRequest,
    accept_lead,
    lead_request,
    question_topic,
    with_lead,
)
from app.generation.hcx_verbalizer import HcxSettings


QUESTION = "LG에너지솔루션과 삼성SDI 중 2025년 설비투자 규모가 더 큰 기업은 어디인가?"

PRESENTED = """공시 근거
1. LG에너지솔루션 · 반기보고서 (2025.06)
   II. 사업의 내용 > 3. 원재료 및 생산설비

| 합 계 | 57,694 | [1]
2. 삼성SDI · 분기보고서 (2025.09)
   II. 사업의 내용 > 3. 원재료 및 생산설비

당사는 2조 3,421억원을 시설 투자에 사용하였습니다. [2]"""


def _request() -> LeadRequest:
    request = lead_request(PRESENTED, period="2025", topic=question_topic(QUESTION))
    assert request is not None
    return request


class RequestTests(unittest.TestCase):
    def test_it_reads_the_companies_and_filing_kinds_from_the_answer(self) -> None:
        request = _request()
        self.assertEqual(request.companies, ("LG에너지솔루션", "삼성SDI"))
        self.assertEqual(request.reports, ("반기보고서", "분기보고서"))

    def test_the_filing_period_is_not_handed_to_the_model(self) -> None:
        # "반기보고서 (2025.06)" carries a figure; only the kind may be sent.
        payload = _request().to_payload()
        self.assertNotIn("2025.06", str(payload))
        self.assertEqual(payload["placeholders"], ["{{PERIOD_1}}"])

    def test_no_figure_from_the_answer_reaches_the_payload(self) -> None:
        payload = str(_request().to_payload())
        for figure in ("57,694", "2조 3,421억원", "2025"):
            with self.subTest(figure=figure):
                self.assertNotIn(figure, payload)

    def test_an_answer_without_evidence_headings_is_not_eligible(self) -> None:
        for answer in (
            "현재 확보된 공시 근거만으로는 해당 내용을 확인하기 어렵습니다.",
            "어느 회사에 대한 공시를 확인할까요?",
            "",
        ):
            with self.subTest(answer=answer):
                self.assertIsNone(lead_request(answer))

    def test_the_topic_drops_the_interrogative_half(self) -> None:
        topic = question_topic(QUESTION)
        self.assertIn("설비투자", topic)
        for noise in ("중", "더", "큰", "어디"):
            with self.subTest(noise=noise):
                self.assertNotIn(noise, topic)


class AcceptanceTests(unittest.TestCase):
    def test_a_framing_sentence_is_accepted_and_its_period_restored(self) -> None:
        text = accept_lead(
            "LG에너지솔루션과 삼성SDI의 {{PERIOD_1}}년 설비투자 관련 공시 근거입니다.",
            _request(),
        )
        self.assertIn("2025년", text)
        self.assertNotIn("{{PERIOD_1}}", text)

    def test_a_figure_is_refused(self) -> None:
        with self.assertRaises(LeadRejected) as caught:
            accept_lead("LG에너지솔루션은 5.8조원을 투자했습니다.", _request())
        self.assertEqual(caught.exception.reason, "digit")

    def test_a_comparison_is_refused(self) -> None:
        for reply in (
            "LG에너지솔루션이 삼성SDI보다 더 큰 규모입니다.",
            "삼성SDI의 투자가 가장 많습니다.",
            "설비투자가 증가한 것으로 확인됩니다.",
        ):
            with self.subTest(reply=reply):
                with self.assertRaises(LeadRejected):
                    accept_lead(reply, _request())

    def test_a_forecast_is_refused(self) -> None:
        with self.assertRaises(LeadRejected):
            accept_lead("향후 설비투자가 확대될 전망입니다.", _request())

    def test_a_company_nobody_supplied_is_refused(self) -> None:
        with self.assertRaises(LeadRejected) as caught:
            accept_lead("현대차의 설비투자 공시 근거입니다.", _request())
        self.assertEqual(caught.exception.reason, "unsupplied_wording")

    def test_a_citation_marker_is_refused(self) -> None:
        with self.assertRaises(LeadRejected) as caught:
            accept_lead("근거 [1] 을 참고하세요.", _request())
        self.assertEqual(caught.exception.reason, "citation_marker")

    def test_an_invented_placeholder_is_refused(self) -> None:
        with self.assertRaises(LeadRejected) as caught:
            accept_lead("설비투자 {{NUM_9}} 공시 근거입니다.", _request())
        self.assertEqual(caught.exception.reason, "unknown_placeholder")

    def test_a_summary_length_reply_is_refused(self) -> None:
        with self.assertRaises(LeadRejected) as caught:
            accept_lead("설비투자 공시 근거입니다. " * 20, _request())
        self.assertEqual(caught.exception.reason, "too_long")

    def test_an_empty_reply_is_refused(self) -> None:
        with self.assertRaises(LeadRejected):
            accept_lead("   ", _request())

    def test_the_length_cap_is_the_stated_one(self) -> None:
        self.assertLessEqual(
            len(
                accept_lead(
                    "LG에너지솔루션과 삼성SDI의 설비투자 공시 근거입니다.", _request()
                )
            ),
            MAX_LEAD_CHARS,
        )


class WriterTests(unittest.TestCase):
    class _Explode:
        def post_json(self, *args, **kwargs):
            raise AssertionError("no request may be made")

    class _Reply:
        def __init__(self, content: str) -> None:
            self.content = content
            self.calls = 0

        def post_json(self, *args, **kwargs):
            self.calls += 1
            return {"choices": [{"message": {"content": self.content}}]}

    class _Fail:
        def post_json(self, *args, **kwargs):
            raise TimeoutError("slow")

    def _settings(self, **changes) -> HcxSettings:
        return HcxSettings(api_key="k", **changes)

    def test_an_ineligible_answer_makes_no_request(self) -> None:
        writer = AnswerLeadWriter(self._settings(), transport=self._Explode())
        outcome = writer.write(None)
        self.assertEqual(outcome.status, STATUS_NOT_ELIGIBLE)
        self.assertEqual(writer.call_count, 0)

    def test_a_disabled_model_makes_no_request(self) -> None:
        writer = AnswerLeadWriter(
            self._settings(enabled=False), transport=self._Explode()
        )
        self.assertEqual(writer.write(_request()).status, STATUS_DISABLED)

    def test_an_unconfigured_model_makes_no_request(self) -> None:
        writer = AnswerLeadWriter(
            HcxSettings(api_key=""), transport=self._Explode()
        )
        self.assertEqual(writer.write(_request()).status, STATUS_NOT_CONFIGURED)

    def test_a_timeout_yields_no_lead_rather_than_an_error(self) -> None:
        outcome = AnswerLeadWriter(self._settings(), transport=self._Fail()).write(
            _request()
        )
        self.assertFalse(outcome.succeeded)
        self.assertIsNone(outcome.text)

    def test_a_refused_reply_reports_which_rule_refused_it(self) -> None:
        transport = self._Reply("LG에너지솔루션은 5.8조원을 투자했습니다.")
        outcome = AnswerLeadWriter(self._settings(), transport=transport).write(
            _request()
        )
        self.assertFalse(outcome.succeeded)
        self.assertEqual(outcome.status, "rejected:digit")

    def test_an_accepted_reply_comes_back_restored(self) -> None:
        transport = self._Reply(
            "LG에너지솔루션과 삼성SDI의 {{PERIOD_1}}년 설비투자 공시 근거입니다."
        )
        outcome = AnswerLeadWriter(self._settings(), transport=transport).write(
            _request()
        )
        self.assertTrue(outcome.succeeded)
        self.assertIn("2025년", outcome.text or "")


class AttachmentTests(unittest.TestCase):
    def test_the_lead_sits_above_the_answer(self) -> None:
        output = with_lead(PRESENTED, "설비투자 공시 근거입니다.")
        self.assertTrue(output.startswith("설비투자 공시 근거입니다.\n\n"))
        self.assertIn(PRESENTED, output)

    def test_no_lead_leaves_the_answer_byte_identical(self) -> None:
        for lead in (None, "", "   "):
            with self.subTest(lead=lead):
                self.assertEqual(with_lead(PRESENTED, lead), PRESENTED)


if __name__ == "__main__":
    unittest.main()
