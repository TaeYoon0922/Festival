"""Reporter recovery stands down for a correction *pair*, and only for that.

A history question asks for both versions of one filing at once, and the pair
resolver already owns which documents those are.  Narrowing its evidence to one
named holder can drop the superseded half.  ``latest`` and ``original`` each
want a single version, so reporter scope is as safe for them as anywhere else.
"""

import unittest

from app.reasoning.correction_pair_roles import PAIR_INTENT, correction_intent
from app.reasoning.holding_company_role_resolution import (
    ROLE_PROVENANCE_KEY,
    HoldingCompanyRoleResolver,
)
from app.reasoning.holding_report_index import HoldingReportIndex
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope, QueryValidator
from tests.test_query_grounded_reporter import (
    ISSUER,
    ISSUER_CODE,
    PERSON,
    record,
)

#: One holder of the issuer, written in a shape no deterministic parser binds,
#: so every question below reaches the recovery unless the guard stops it.
HOLDER = PERSON


class CorrectionPairScopeTest(unittest.TestCase):
    def setUp(self):
        self.scope = CorpusScope(
            companies={ISSUER: (ISSUER, ISSUER_CODE)},
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.understanding = QueryUnderstanding(self.scope.company_aliases())
        self.validator = QueryValidator(
            corpus_scope=self.scope,
            holding_company_role_resolver=HoldingCompanyRoleResolver(
                HoldingReportIndex(
                    (record(HOLDER, "holder"),),
                    complete=True,
                    correction_finality_available=True,
                )
            ),
        )

    def plan_for(self, question):
        return self.validator.validate(self.understanding.understand(question)).plan

    # ------------------------------------------------------------ suppressed
    def test_a_pair_question_keeps_its_reporter_unscoped(self):
        plan = self.plan_for(
            f"{ISSUER} {HOLDER} 2024년 2월 3일 기준 보유주식수는 정정 전과 정정 후가 각각 얼마야?"
        )

        self.assertEqual(correction_intent(plan), PAIR_INTENT)
        self.assertEqual(plan.evidence.get("operation"), "correction_lookup")
        self.assertIsNone(plan.reporter)
        self.assertNotIn(ROLE_PROVENANCE_KEY, dict(plan.evidence))

    def test_every_pair_wording_is_covered(self):
        for wording in ("정정 전후로", "정정 전과 정정 후가", "정정 전 대비 정정 후가"):
            with self.subTest(wording=wording):
                plan = self.plan_for(
                    f"{ISSUER} {HOLDER} 2024년 2월 3일 기준 보유주식수는 {wording} 어떻게 되나?"
                )
                self.assertEqual(correction_intent(plan), PAIR_INTENT)
                self.assertIsNone(plan.reporter)

    # ------------------------------------------------------------ unaffected
    def test_the_same_question_without_pair_intent_still_recovers(self):
        """Identical shape and holder; only the correction pair wording is gone."""

        plan = self.plan_for(f"{ISSUER} {HOLDER} 2024년 2월 3일 기준 보유주식수는?")

        self.assertIsNone(correction_intent(plan))
        self.assertEqual(plan.reporter, HOLDER)

    def test_latest_correction_intent_still_recovers(self):
        """``최종 정정`` wants one version, so scoping it is safe."""

        plan = self.plan_for(
            f"{ISSUER} {HOLDER} 2024년 2월 3일 기준 보유주식수는 최종 정정 기준으로 몇 주야?"
        )

        self.assertEqual(correction_intent(plan), "latest")
        self.assertNotEqual(correction_intent(plan), PAIR_INTENT)
        self.assertEqual(plan.reporter, HOLDER)

    def test_original_correction_intent_still_recovers(self):
        plan = self.plan_for(
            f"{ISSUER} {HOLDER} 2024년 2월 3일 기준 최초 공시 보유주식수는?"
        )

        self.assertEqual(correction_intent(plan), "original")
        self.assertEqual(plan.reporter, HOLDER)

    def test_a_latest_holding_question_still_recovers(self):
        plan = self.plan_for(f"{ISSUER} {HOLDER} 이번 보고 보유주식수는?")
        self.assertEqual(plan.reporter, HOLDER)

    def test_an_exact_reference_date_question_still_recovers(self):
        plan = self.plan_for(f"{ISSUER} {HOLDER} 2024년 2월 3일 기준일 보유비율은?")
        self.assertEqual(plan.reporter, HOLDER)

    def test_a_reporter_the_wording_bound_is_unchanged_by_the_guard(self):
        """A pair question whose holder the existing parser binds on its own.

        The guard turns recovery off; it does not turn reporters off.  A
        reporter the question's own wording produced was never this path's to
        give, and is not this path's to take away.
        """

        plan = self.plan_for(f"{ISSUER} {HOLDER}의 정정 전후 보유주식수는?")

        self.assertEqual(correction_intent(plan), PAIR_INTENT)
        self.assertEqual(plan.reporter, HOLDER)
        self.assertNotIn(ROLE_PROVENANCE_KEY, dict(plan.evidence))


if __name__ == "__main__":
    unittest.main()
