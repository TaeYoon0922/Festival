"""Recovering a holder the question writes out and the parsers could not bind.

The shapes exercised here are the ones the existing reporter rules decline: a
holder written with no particle after it, and a holder whose name is more than
one token.  Shapes those rules already bind keep binding through them, which is
asserted rather than assumed.
"""

import unittest

from app.reasoning.holding_company_role_resolution import (
    ROLE_PATH_QUERY_GROUNDED,
    ROLE_PROVENANCE_KEY,
    HoldingCompanyRoleResolver,
)
from app.reasoning.holding_report_index import HoldingReportIndex, HoldingReportRecord
from app.reasoning.holding_reporter import (
    canonical_reporter_key,
    reporter_surface_spans,
)
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope, QueryValidator

ISSUER = "가상발행사"
ISSUER_CODE = "00000001"
#: Holders of that issuer, in the shapes that defeat the deterministic parsers:
#: a person named without a particle, a multi-word foreign institution, and a
#: name carrying its legal form.
PERSON = "장가람"
FOREIGN = "Meridian Asset Partners"
CORPORATE = "가상캐피탈(주)"
#: A second holder, so the singleton rule can be shown declining rather than
#: choosing.
OTHER = "가상자산운용"
#: Shares a prefix with CORPORATE.  The corpus knows only the shorter name, so
#: a containment rule would bind the wrong holder here.
PREFIXED = "가상캐피탈홀딩스"
#: A holder of some other issuer entirely.
OUTSIDER = "무관투자사"


def record(reporter, suffix, issuer_code=ISSUER_CODE):
    return HoldingReportRecord(
        issuer_corp_code=issuer_code,
        reporter_key=canonical_reporter_key(reporter),
        raw_reporter=reporter,
        doc_id=f"holding_{suffix}",
        projection_chunk_id=f"holding_{suffix}:projection",
        reference_date="20240203",
        receipt_date="20240205",
        after_shares="1000",
        after_ratio="1.00",
        source_refs=({"table_id": "t-qur", "row_start": 1, "row_end": 1},),
    )


class Harness:
    """Understanding, validation and a corpus index, wired as the server wires them."""

    HOLDERS = (PERSON, FOREIGN, CORPORATE, OTHER)

    def setUp(self):
        self.index = HoldingReportIndex(
            (
                *(record(holder, f"h{n}") for n, holder in enumerate(self.HOLDERS)),
                record(OUTSIDER, "outsider", issuer_code="00000009"),
            ),
            complete=True,
            correction_finality_available=True,
        )
        self.scope = CorpusScope(
            companies={ISSUER: (ISSUER, ISSUER_CODE)},
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.understanding = QueryUnderstanding(self.scope.company_aliases())
        self.validator = self.validator_for(self.index)

    def validator_for(self, index, scope=None):
        return QueryValidator(
            corpus_scope=scope or self.scope,
            holding_company_role_resolver=HoldingCompanyRoleResolver(index),
        )

    def plan_for(self, question, validator=None, understanding=None):
        parser = understanding or self.understanding
        return (validator or self.validator).validate(parser.understand(question)).plan

    def assert_recovered(self, question, expected):
        plan = self.plan_for(question)
        self.assertEqual(plan.reporter, expected)
        self.assertEqual(
            dict(plan.evidence)[ROLE_PROVENANCE_KEY]["path"], ROLE_PATH_QUERY_GROUNDED
        )
        return plan

    def assert_unbound(self, question, **kwargs):
        plan = self.plan_for(question, **kwargs)
        self.assertIsNone(plan.reporter)
        self.assertNotIn(ROLE_PROVENANCE_KEY, dict(plan.evidence))
        return plan


class RecoveredShapeTest(Harness, unittest.TestCase):
    def test_holder_first_subject(self):
        """<holder>가 <issuer> 주식을 ..."""

        self.assert_recovered(
            f"{FOREIGN}가 {ISSUER} 주식을 2024년 2월 3일에 변동한 내역의 보유주식수는?",
            FOREIGN,
        )

    def test_issuer_then_holder_without_a_particle(self):
        """<issuer> <holder> 이번 보고 ... -- no particle for the parsers to key on."""

        self.assert_recovered(f"{ISSUER} {PERSON} 이번 보고 보유주식수는?", PERSON)

    def test_issuer_relation_then_holder(self):
        """<issuer>에 대해 <holder>가 ..."""

        self.assert_recovered(
            f"{ISSUER}에 대해 {FOREIGN}가 2024년 2월 3일 기준으로 보고한 보유주식수는?",
            FOREIGN,
        )

    def test_holder_possessive_then_issuer(self):
        """<holder>의 <issuer> ... 보고 ..."""

        self.assert_recovered(
            f"{CORPORATE}의 {ISSUER} 2024년 2월 3일 기준 보고 보유비율은?", CORPORATE
        )

    def test_a_person_is_recovered(self):
        self.assert_recovered(f"{ISSUER} {PERSON} 금번 보고 보유비율은?", PERSON)

    def test_a_foreign_institution_is_recovered(self):
        """Multi-word, matched through the existing canonical key -- no alias list."""

        plan = self.assert_recovered(
            f"{ISSUER}에 대한 {FOREIGN}의 2024년 2월 3일 기준 보유비율은?", FOREIGN
        )
        self.assertEqual(plan.companies, (ISSUER,))
        self.assertEqual(plan.corp_codes, (ISSUER_CODE,))

    def test_recovery_binds_the_corpus_spelling(self):
        """The question writes a name; the corpus decides how it is recorded."""

        self.assert_recovered(f"{ISSUER} 가상캐피탈 이번 보고 보유주식수는?", CORPORATE)


class DeclinedTest(Harness, unittest.TestCase):
    def test_issuer_only_question_binds_nobody(self):
        self.assert_unbound(f"{ISSUER}의 최근 보유주식수는?")

    def test_holder_absent_from_the_question_binds_nobody(self):
        self.assert_unbound(f"{ISSUER} 2024년 2월 3일 기준 보유주식수는?")

    def test_two_named_holders_are_not_chosen_between(self):
        self.assert_unbound(
            f"{ISSUER}에 대한 {PERSON}와 {OTHER}의 2024년 2월 3일 보유주식수는?"
        )

    def test_a_holder_of_another_issuer_binds_nobody(self):
        self.assert_unbound(f"{ISSUER} {OUTSIDER} 이번 보고 보유주식수는?")

    def test_a_name_the_corpus_cannot_validate_binds_nobody(self):
        self.assert_unbound(f"{ISSUER} 가상없는투자 이번 보고 보유주식수는?")

    def test_an_unusable_index_binds_nobody(self):
        for label, index in (
            ("absent", None),
            (
                "incomplete",
                HoldingReportIndex((record(PERSON, "p"),), complete=False),
            ),
        ):
            with self.subTest(index=label):
                self.assert_unbound(
                    f"{ISSUER} {PERSON} 이번 보고 보유주식수는?",
                    validator=self.validator_for(index),
                )

    def test_a_non_holding_question_is_never_asked(self):
        self.assert_unbound(f"{ISSUER}의 2024년 매출액은 얼마야?")


class ExistingRulesKeepPriorityTest(Harness, unittest.TestCase):
    def test_a_reporter_the_wording_already_binds_is_not_reconsidered(self):
        """The single-token possessive shape stays with the parser that owns it."""

        plan = self.plan_for(f"{ISSUER} {PERSON}의 최신 보고 보유주식수는?")

        self.assertEqual(plan.reporter, PERSON)
        self.assertNotIn(ROLE_PROVENANCE_KEY, dict(plan.evidence))

    def test_a_labelled_reporter_is_not_reconsidered(self):
        plan = self.plan_for(f"{ISSUER} 국민연금공단의 최신 보고 보유주식수는?")

        self.assertEqual(plan.reporter, "국민연금공단")
        self.assertNotIn(ROLE_PROVENANCE_KEY, dict(plan.evidence))


class IssuerSafetyTest(Harness, unittest.TestCase):
    def test_the_issuer_is_never_rebound_as_its_own_reporter(self):
        """A company is often both; its own name in its own question proves nothing."""

        index = HoldingReportIndex(
            (record(ISSUER, "self"), record(PERSON, "person")),
            complete=True,
            correction_finality_available=True,
        )
        self.assert_unbound(
            f"{ISSUER}의 최근 보유주식수는?", validator=self.validator_for(index)
        )

    def test_a_holder_name_inside_the_issuer_name_is_not_a_mention(self):
        """``가상캐피탈`` inside ``가상캐피탈전자`` is the issuer, not its holder."""

        issuer = "가상캐피탈전자"
        scope = CorpusScope(
            companies={issuer: (issuer, ISSUER_CODE)},
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.assert_unbound(
            f"{issuer}의 최근 보유주식수는?",
            validator=self.validator_for(self.index, scope=scope),
            understanding=QueryUnderstanding(scope.company_aliases()),
        )


class SurfaceBoundaryTest(unittest.TestCase):
    """The containment guard: a longer name is a different holder."""

    def test_a_holder_name_is_not_matched_inside_a_longer_name(self):
        self.assertEqual(
            reporter_surface_spans(f"{PREFIXED}의 보유주식수는?", CORPORATE), ()
        )

    def test_a_holder_name_is_matched_before_a_postposition(self):
        for particle in ("의", "이", "가", "은", "는", "에", ""):
            with self.subTest(particle=particle):
                self.assertEqual(
                    reporter_surface_spans(f"가상캐피탈{particle} 보유주식수는?", CORPORATE),
                    ((0, 5),),
                )

    def test_spacing_and_legal_form_differences_are_absorbed(self):
        self.assertEqual(
            reporter_surface_spans("Meridian Asset Partners가 보유한", FOREIGN),
            ((0, 23),),
        )

    def test_a_name_written_twice_yields_both_mentions(self):
        self.assertEqual(len(reporter_surface_spans(f"{PERSON}와 {PERSON}의 보유", PERSON)), 2)

    def test_an_absent_name_yields_nothing(self):
        self.assertEqual(reporter_surface_spans("보유주식수는?", PERSON), ())

    def test_an_empty_name_or_question_yields_nothing(self):
        self.assertEqual(reporter_surface_spans("", PERSON), ())
        self.assertEqual(reporter_surface_spans("보유", ""), ())


if __name__ == "__main__":
    unittest.main()
