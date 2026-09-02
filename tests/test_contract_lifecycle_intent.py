"""A contract named plainly, then asked what became of it."""

import unittest

from app.agent.task_router import route_task
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope, QueryValidator

ISSUER = "가상발행사"
ISSUER_CODE = "00000001"


class ContractLifecycleIntentTest(unittest.TestCase):
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

    def assert_contract_event(self, question):
        plan = self.plan_for(question)
        self.assertEqual(plan.event_type, "supply_contract")
        self.assertEqual(tuple(plan.disclosure_route), ("exchange",))
        self.assertEqual(plan.task_type, "corporate_event")
        self.assertEqual(
            route_task(question, query_plan=plan).task_type, "corporate_event"
        )
        return plan

    def assert_not_a_contract_event(self, question):
        plan = self.plan_for(question)
        self.assertIsNone(plan.event_type)
        return plan

    # ------------------------------------------------------------ recognized
    def test_a_contract_asked_whether_it_was_terminated(self):
        self.assert_contract_event(
            f"{ISSUER}가 2024년 6월 18일 체결 공시한 계약은 해지됐나?"
        )

    def test_a_contract_asked_whether_a_termination_filing_followed(self):
        self.assert_contract_event(
            f"{ISSUER}이 2023년 3월 31일 공시한 계약은 이후 해지 공시가 있었는지 알려줘"
        )

    def test_a_contract_asked_for_its_final_state(self):
        self.assert_contract_event(
            f"{ISSUER}이 2023년 2월 28일 공시한 사업 계약의 최종 상태는?"
        )

    def test_the_named_family_still_wins_when_the_question_states_it(self):
        """A question naming its own filing family keeps that family's routing."""

        plan = self.assert_contract_event(
            f"{ISSUER}이 2024년 11월 4일 공시한 변압기 공급계약은 이후 어떻게 됐어?"
        )
        self.assertEqual(plan.evidence.get("event_type"), "공급계약")

    def test_an_explicit_termination_filing_keeps_its_own_event_type(self):
        plan = self.plan_for(f"{ISSUER}의 계약해지 공시 내용 알려줘")
        self.assertEqual(plan.event_type, "contract_termination")

    # ------------------------------------------------------------ declined
    def test_a_contract_period_end_date_is_a_field_not_a_lifecycle(self):
        """``계약기간 종료일`` is stated inside one filing; nothing was terminated."""

        self.assert_not_a_contract_event(
            f"{ISSUER}가 맺은 기체구조물 계약의 계약기간 종료일은?"
        )

    def test_a_bare_contract_amount_question_is_not_rerouted(self):
        self.assert_not_a_contract_event(f"{ISSUER} 2024년 계약금액은 얼마야?")

    def test_an_unrelated_metric_question_is_not_rerouted(self):
        self.assert_not_a_contract_event(f"{ISSUER}의 2024년 매출액은 얼마야?")

    def test_a_holding_termination_word_far_from_any_contract_is_not_rerouted(self):
        self.assert_not_a_contract_event(
            f"{ISSUER} 국민연금공단 보유주식수는? 신탁 해지 여부는 무관하다"
        )

    def test_the_contract_noun_and_the_outcome_must_be_about_each_other(self):
        """Distance is bounded, so a termination in another clause is not this."""

        self.assert_not_a_contract_event(
            f"{ISSUER}의 계약 내용과 무관하게 별도로 진행된 신탁이 해지된 시점은?"
        )


if __name__ == "__main__":
    unittest.main()
