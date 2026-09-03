"""Naming which served filing began a contract and which one ended it."""

import unittest
from types import SimpleNamespace

from app.reasoning.contract_lifecycle import (
    LIFECYCLE_OUTCOME_KEY,
    ROLE_ORIGIN,
    ROLE_TERMINATION,
    compose_lifecycle_text,
    lifecycle_items,
    lifecycle_outcome,
    lifecycle_outcome_requested,
    lifecycle_role,
    requested_lifecycle_outcome,
)
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope, QueryValidator

ISSUER = "가상발행사"
ISSUER_CODE = "00000001"


def item(doc_id, report_nm, rcept_dt, text="", rank=1):
    return SimpleNamespace(
        chunk_id=f"{doc_id}:c",
        doc_id=doc_id,
        report_nm=report_nm,
        rcept_dt=rcept_dt,
        evidence_text=text,
        retrieval_rank=rank,
    )


ORIGIN = item("exchange_a", "단일판매ㆍ공급계약체결", "20240618")
TERMINATION = item(
    "exchange_b", "단일판매ㆍ공급계약해지", "20241128", "해지일자 : 2024-11-28", rank=7
)
CORRECTION = item("exchange_c", "[기재정정]단일판매ㆍ공급계약체결", "20241205")
UNRELATED = item("periodic_d", "분기보고서", "20241001")


class LifecycleRoleTest(unittest.TestCase):
    def test_roles_are_read_from_the_filings_own_name(self):
        self.assertEqual(lifecycle_role(ORIGIN), ROLE_ORIGIN)
        self.assertEqual(lifecycle_role(TERMINATION), ROLE_TERMINATION)

    def test_a_corrected_reprint_keeps_its_family(self):
        """``[기재정정]`` marks a reprint, not a different kind of document."""

        self.assertEqual(lifecycle_role(CORRECTION), ROLE_ORIGIN)

    def test_a_correction_is_never_a_termination(self):
        outcome = lifecycle_outcome([ORIGIN, CORRECTION])
        self.assertFalse(outcome.terminated)
        self.assertIsNone(outcome.terminal)

    def test_an_unrelated_filing_has_no_lifecycle_role(self):
        self.assertIsNone(lifecycle_role(UNRELATED))
        self.assertIsNone(lifecycle_outcome([UNRELATED]))


class LifecycleOutcomeTest(unittest.TestCase):
    def test_origin_and_termination_are_bound(self):
        outcome = lifecycle_outcome([UNRELATED, TERMINATION, ORIGIN])

        self.assertTrue(outcome.terminated)
        self.assertEqual(outcome.origin.doc_id, "exchange_a")
        self.assertEqual(outcome.terminal.doc_id, "exchange_b")
        self.assertEqual(outcome.effective_date, "2024-11-28")

    def test_the_earliest_origin_and_latest_termination_win(self):
        later_origin = item("exchange_x", "단일판매ㆍ공급계약체결", "20250101")
        earlier_end = item("exchange_y", "단일판매ㆍ공급계약해지", "20240101")
        outcome = lifecycle_outcome([later_origin, ORIGIN, earlier_end, TERMINATION])

        self.assertEqual(outcome.origin.doc_id, "exchange_a")
        self.assertEqual(outcome.terminal.doc_id, "exchange_b")

    def test_no_termination_is_not_read_as_continuation(self):
        outcome = lifecycle_outcome([ORIGIN])
        text = compose_lifecycle_text(outcome)

        self.assertFalse(outcome.terminated)
        self.assertIn("확인되지 않습니다", text)
        self.assertNotIn("해지되었습니다", text)

    def test_a_missing_effective_date_is_simply_absent(self):
        bare = item("exchange_z", "단일판매ㆍ공급계약해지", "20241128")
        outcome = lifecycle_outcome([ORIGIN, bare])
        text = compose_lifecycle_text(outcome)

        self.assertIsNone(outcome.effective_date)
        self.assertIn("해지되었습니다", text)
        self.assertNotIn("해지일자", text)


class LifecycleTextTest(unittest.TestCase):
    def test_the_outcome_and_both_dates_are_stated(self):
        text = compose_lifecycle_text(lifecycle_outcome([ORIGIN, TERMINATION]))

        self.assertIn("해지되었습니다", text)
        self.assertIn("2024년 11월 28일", text)
        self.assertIn("2024년 6월 18일", text)


class LifecycleCitationTest(unittest.TestCase):
    def test_the_terminal_document_is_cited_whatever_its_rank(self):
        """It ranked seventh on the server; a limit of five must not drop it."""

        served = [item(f"filler_{n}", "분기보고서", "2024") for n in range(6)]
        served.insert(0, ORIGIN)
        served.append(TERMINATION)
        outcome = lifecycle_outcome(served)

        cited = lifecycle_items(outcome, served, limit=5)
        ids = [entry.doc_id for entry in cited]

        self.assertIn("exchange_a", ids)
        self.assertIn("exchange_b", ids)
        self.assertEqual(ids[:2], ["exchange_a", "exchange_b"])

    def test_both_ends_are_kept_even_when_the_limit_is_smaller(self):
        outcome = lifecycle_outcome([ORIGIN, TERMINATION])
        cited = lifecycle_items(outcome, [ORIGIN, TERMINATION], limit=1)

        self.assertEqual(len(cited), 2)

    def test_remaining_room_keeps_served_order(self):
        served = [ORIGIN, UNRELATED, TERMINATION]
        cited = lifecycle_items(lifecycle_outcome(served), served, limit=3)

        self.assertEqual(
            [entry.doc_id for entry in cited],
            ["exchange_a", "exchange_b", "periodic_d"],
        )


class LifecycleIntentTest(unittest.TestCase):
    def setUp(self):
        self.scope = CorpusScope(
            companies={ISSUER: (ISSUER, ISSUER_CODE)},
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.understanding = QueryUnderstanding(self.scope.company_aliases())
        self.validator = QueryValidator(corpus_scope=self.scope)

    def plan_for(self, question):
        return self.validator.validate(self.understanding.understand(question)).plan

    def test_each_way_of_asking_the_outcome_is_recognized(self):
        for wording in (
            "계약은 해지됐나?",
            "계약은 이후 해지 공시가 있었는지 알려줘",
            "계약의 최종 상태는?",
            "공급계약은 이후 어떻게 됐어?",
        ):
            with self.subTest(wording=wording):
                self.assertIsNotNone(lifecycle_outcome_requested(wording))

    def test_the_plan_records_the_intent_once(self):
        plan = self.plan_for(f"{ISSUER}이 2024년 6월 18일 공시한 계약은 해지됐나?")

        self.assertTrue(requested_lifecycle_outcome(plan))
        self.assertIn(LIFECYCLE_OUTCOME_KEY, dict(plan.evidence))

    def test_a_change_question_is_not_a_lifecycle_question(self):
        """Asking how much an amount moved wants the number, not the state."""

        for wording in (
            "계약의 최초 계약금액과 해지 공시의 해지금액은 얼마나 차이가 나?",
            "계약금액은 정정에서 얼마나 감소했나?",
            "계약금액이 정정에서 얼마나 변경됐는지 증감액으로 알려줘",
        ):
            with self.subTest(wording=wording):
                self.assertIsNone(lifecycle_outcome_requested(wording))

    def test_a_contract_period_end_date_is_not_a_lifecycle_question(self):
        self.assertIsNone(
            lifecycle_outcome_requested("계약의 계약기간 종료일은?")
        )

    def test_an_ordinary_holding_question_is_not_one(self):
        plan = self.plan_for(f"{ISSUER} 국민연금공단 보유주식수는?")
        self.assertIsNone(requested_lifecycle_outcome(plan))


if __name__ == "__main__":
    unittest.main()
