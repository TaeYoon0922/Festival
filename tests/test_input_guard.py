"""The first gate: refuse before the corpus or the model sees the question.

Two halves matter equally. A question carrying a private identifier must stop
before retrieval, and a question about the people disclosure filings name --
executives, largest shareholders, reporting parties -- must not stop at all.
A guard that refuses the second is worse than no guard: it declines the
questions this system exists to answer.
"""

from __future__ import annotations

import unittest

from app.api.pipeline import AnswerPipeline
from app.reasoning.input_guard import (
    CATEGORY_CONTACT_REQUEST,
    CATEGORY_IDENTIFIER,
    inspect_question,
)


class BlockedTests(unittest.TestCase):
    def test_a_resident_registration_number_is_refused(self) -> None:
        decision = inspect_question("김철수 900101-1234567 의 보유 주식 알려줘")
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.category, CATEGORY_IDENTIFIER)

    def test_a_mobile_number_is_refused(self) -> None:
        self.assertTrue(inspect_question("담당자 010-1234-5678 로 연락처 확인").blocked)

    def test_an_email_address_is_refused(self) -> None:
        self.assertTrue(inspect_question("ir@example.com 계정의 공시 이력").blocked)

    def test_a_payment_card_number_is_refused(self) -> None:
        self.assertTrue(inspect_question("1234-5678-9012-3456 결제 내역").blocked)

    def test_asking_for_a_personal_contact_is_refused(self) -> None:
        decision = inspect_question("삼성전자 대표이사 개인 연락처 알려줘")
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.category, CATEGORY_CONTACT_REQUEST)

    def test_asking_for_a_home_address_is_refused(self) -> None:
        self.assertTrue(inspect_question("최대주주의 자택 주소를 확인해줘").blocked)

    def test_asking_for_a_resident_number_is_refused(self) -> None:
        self.assertTrue(inspect_question("보고자의 주민등록번호가 뭐야").blocked)

    def test_the_detected_value_never_reaches_the_trace(self) -> None:
        decision = inspect_question("900101-1234567 조회")
        payload = decision.to_dict()
        self.assertNotIn("900101", str(payload))
        self.assertEqual(payload["category"], CATEGORY_IDENTIFIER)


class AllowedTests(unittest.TestCase):
    """Filings name people by law; questions about them must pass."""

    def test_a_question_about_an_executive_passes(self) -> None:
        self.assertFalse(inspect_question("삼성전자 대표이사가 누구인가").blocked)

    def test_a_question_about_the_largest_shareholder_passes(self) -> None:
        self.assertFalse(inspect_question("카카오의 최대주주 현황 알려줘").blocked)

    def test_a_question_about_a_reporting_party_passes(self) -> None:
        self.assertFalse(
            inspect_question("국민연금공단이 보고한 지분 변동 알려줘").blocked
        )

    def test_an_executive_roster_question_passes(self) -> None:
        self.assertFalse(inspect_question("임원 명단을 정리해줘").blocked)

    def test_ordinary_disclosure_questions_pass(self) -> None:
        for question in (
            "삼성전자의 2025년 연결기준 매출액은 얼마인가?",
            "한화오션이 2025년에 체결한 계약 이후 해지된 계약이 있는가?",
            "LG에너지솔루션과 삼성SDI 중 2025년 설비투자가 더 큰 곳은?",
            "카카오가 2025년에 실시한 자금조달 내역을 유형별로 정리해줘",
        ):
            with self.subTest(question=question):
                self.assertFalse(inspect_question(question).blocked)

    def test_figures_that_look_like_identifiers_pass(self) -> None:
        # Share counts, amounts and dates must not trip the identifier patterns.
        for question in (
            "변동 후 보유주식수 1,037,916주",
            "계약금액 1234-5678 억원 규모",
            "2025-03-14 에 공시된 내용",
            "매출액 333,605,938 백만원",
        ):
            with self.subTest(question=question):
                self.assertFalse(inspect_question(question).blocked)

    def test_an_empty_question_is_not_blocked_here(self) -> None:
        # Blank input is the API's validation error, not this guard's refusal.
        self.assertFalse(inspect_question("   ").blocked)


class PipelineTests(unittest.TestCase):
    class _Explode:
        def understand(self, *args, **kwargs):
            raise AssertionError("understanding must not run on a refused question")

        def execute(self, *args, **kwargs):
            raise AssertionError("retrieval must not run on a refused question")

    def _pipeline(self) -> AnswerPipeline:
        explode = self._Explode()
        return AnswerPipeline(understanding=explode, executor=explode)

    def test_a_refused_question_never_reaches_retrieval(self) -> None:
        payload = self._pipeline().answer("PII-1", "900101-1234567 보유 내역")

        self.assertEqual(
            sorted(payload),
            ["answer", "question", "question_id", "retrieved_context", "think_trace"],
        )
        self.assertEqual(payload["retrieved_context"], [])
        self.assertTrue(payload["answer"])

    def test_the_trace_says_which_gate_ended_it(self) -> None:
        trace = self._pipeline().answer("PII-2", "대표이사 개인 연락처 알려줘")[
            "think_trace"
        ]
        self.assertEqual(trace["route"], "sensitive_input")
        self.assertEqual(trace["stages"], ["input_guard"])
        self.assertFalse(trace["answerable"])
        self.assertEqual(trace["hcx_status"], "skipped_input_guard")

    def test_the_refused_question_is_returned_but_its_evidence_is_not(self) -> None:
        question = "김철수 900101-1234567 의 지분"
        payload = self._pipeline().answer("PII-3", question)
        # The contract returns the question as received; what must not appear is
        # corpus evidence gathered on the strength of it.
        self.assertEqual(payload["question"], question)
        self.assertEqual(payload["retrieved_context"], [])


if __name__ == "__main__":
    unittest.main()
