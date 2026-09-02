"""One field compared across N companies, each answered from its own scope."""

import unittest
from types import SimpleNamespace

from app.reasoning.company_comparison import (
    CompanyComparisonRequest,
    comparison_requested,
    compose_comparison_text,
    execute_company_scopes,
    resolve_comparison,
)
from tests.test_scoped_operands import chunk

A, B, C, D = "00000001", "00000002", "00000003", "00000004"
NAMES = {A: "가상항공", B: "가상로템", C: "가상중공업", D: "가상디펜스"}


def request(*codes, ordered=False):
    return CompanyComparisonRequest(
        companies=tuple((NAMES[code], code) for code in codes), ordered=ordered
    )


class RecognitionTest(unittest.TestCase):
    def test_two_companies_and_a_larger_side(self):
        found = comparison_requested(
            "가상항공의 수출계약과 가상로템의 부품 계약 중 계약금액이 더 큰 쪽은?",
            ["가상항공", "가상로템"],
            [A, B],
        )

        self.assertIsNotNone(found)
        self.assertEqual(found.size, 2)
        self.assertFalse(found.ordered)

    def test_four_companies_and_an_ordering(self):
        found = comparison_requested(
            "가상항공, 가상로템, 가상중공업, 가상디펜스가 각각 공시한 계약 중 "
            "계약금액이 큰 순서대로 나열해줘",
            list(NAMES.values()),
            [A, B, C, D],
        )

        self.assertEqual(found.size, 4)
        self.assertTrue(found.ordered)

    def test_a_difference_question_is_a_comparison(self):
        self.assertIsNotNone(
            comparison_requested(
                "가상항공의 계약과 가상로템의 계약 계약금액은 차이가 얼마나 돼?",
                ["가상항공", "가상로템"],
                [A, B],
            )
        )

    # ----------------------------------------------------------- declined
    def test_one_company_is_not_a_comparison(self):
        self.assertIsNone(
            comparison_requested("가상항공 계약금액이 가장 큰 건은?", ["가상항공"], [A])
        )

    def test_companies_without_comparison_intent_stay_ambiguous(self):
        self.assertIsNone(
            comparison_requested(
                "가상항공과 가상로템의 계약금액을 알려줘", ["가상항공", "가상로템"], [A, B]
            )
        )

    def test_a_holding_role_pair_is_not_a_comparison(self):
        """Two companies, one holding question: an issuer and its filer."""

        self.assertIsNone(
            comparison_requested(
                "가상로템이 보유한 가상항공 주식은 몇 주인가?",
                ["가상항공", "가상로템"],
                [A, B],
            )
        )

    def test_a_comparison_of_a_non_comparable_field_is_declined(self):
        self.assertIsNone(
            comparison_requested(
                "가상항공과 가상로템 중 보유주식수가 더 큰 쪽은?",
                ["가상항공", "가상로템"],
                [A, B],
            )
        )

    def test_an_unresolved_company_cannot_own_an_operand(self):
        self.assertIsNone(
            comparison_requested(
                "가상항공과 가상로템 중 계약금액이 더 큰 쪽은?",
                ["가상항공", "가상로템"],
                [A],
            )
        )

    def test_duplicate_codes_are_refused(self):
        self.assertIsNone(
            comparison_requested(
                "가상항공과 가상로템 중 계약금액이 더 큰 쪽은?",
                ["가상항공", "가상로템"],
                [A, A],
            )
        )


class ScopedExecutionTest(unittest.TestCase):
    def test_each_company_is_retrieved_against_its_own_plan(self):
        seen = []

        def execute(plan):
            seen.append((plan.companies, plan.corp_codes))
            return SimpleNamespace(chunks=(chunk(plan.corp_codes[0], "100",
                                                 corp_code=plan.corp_codes[0]),))

        plan = SimpleNamespace(companies=("가", "나"), corp_codes=(A, B))
        # ``replace`` needs a real dataclass; a namespace stands in via a stub.
        from dataclasses import dataclass, replace as _r

        @dataclass(frozen=True)
        class Plan:
            companies: tuple
            corp_codes: tuple

        per_company = execute_company_scopes(
            request(A, B), Plan(companies=("가", "나"), corp_codes=(A, B)), execute
        )

        self.assertEqual([codes for _names, codes in seen], [(A,), (B,)])
        self.assertEqual(sorted(per_company), [A, B])

    def test_one_companys_failure_does_not_borrow_another(self):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Plan:
            companies: tuple
            corp_codes: tuple

        def execute(plan):
            if plan.corp_codes[0] == B:
                raise RuntimeError("retrieval failed")
            return SimpleNamespace(chunks=(chunk("a", "100", corp_code=A),))

        per_company = execute_company_scopes(
            request(A, B), Plan(companies=("가", "나"), corp_codes=(A, B)), execute
        )

        self.assertEqual(per_company[B], ())
        self.assertIsNone(resolve_comparison(request(A, B), per_company))


class ResolutionTest(unittest.TestCase):
    def per_company(self, *pairs):
        return {
            code: (chunk(code, amount, corp_code=code),) for code, amount in pairs
        }

    def test_two_companies_rank_by_their_own_amounts(self):
        order = resolve_comparison(
            request(A, B), self.per_company((A, "1,195,242,120,000"), (B, "300"))
        )

        self.assertEqual(order.largest.scope.company, NAMES[A])
        self.assertEqual(order.spread, 1195242119700)

    def test_four_companies_order_completely(self):
        order = resolve_comparison(
            request(A, B, C, D),
            self.per_company((A, "10"), (B, "40"), (C, "30"), (D, "20")),
        )

        self.assertEqual(
            [operand.scope.company for operand in order.operands],
            [NAMES[B], NAMES[C], NAMES[D], NAMES[A]],
        )

    def test_a_company_with_no_evidence_declines_the_whole_ranking(self):
        per_company = self.per_company((A, "10"), (B, "40"))
        per_company[C] = ()

        self.assertIsNone(resolve_comparison(request(A, B, C), per_company))

    def test_one_companys_evidence_never_satisfies_another_scope(self):
        """B's slice is empty; A's document must not fill it."""

        per_company = {A: (chunk("a", "10", corp_code=A),), B: ()}

        self.assertIsNone(resolve_comparison(request(A, B), per_company))

    def test_a_foreign_document_in_a_slice_is_ignored(self):
        """A chunk carrying another company's code cannot answer this scope."""

        per_company = {
            A: (chunk("a", "10", corp_code=A),),
            B: (chunk("stray", "40", corp_code=A),),
        }

        self.assertIsNone(resolve_comparison(request(A, B), per_company))

    def test_every_ranked_member_keeps_its_own_document(self):
        order = resolve_comparison(request(A, B), self.per_company((A, "10"), (B, "40")))
        docs = {o.scope.company: o.source.doc_id for o in order.operands}

        self.assertEqual(docs, {NAMES[A]: A, NAMES[B]: B})


class CompositionTest(unittest.TestCase):
    def per_company(self, *pairs):
        return {code: (chunk(code, amount, corp_code=code),) for code, amount in pairs}

    def test_two_companies_state_the_winner_and_the_difference(self):
        req = request(A, B)
        text = compose_comparison_text(
            req, resolve_comparison(req, self.per_company((A, "300"), (B, "100")))
        )

        self.assertIn(NAMES[A], text)
        self.assertIn("차이", text)
        self.assertIn("200", text)

    def test_an_ordering_lists_every_company(self):
        req = request(A, B, C, D, ordered=True)
        text = compose_comparison_text(
            req,
            resolve_comparison(
                req, self.per_company((A, "10"), (B, "40"), (C, "30"), (D, "20"))
            ),
        )

        for name in NAMES.values():
            self.assertIn(name, text)
        self.assertIn(">", text)


if __name__ == "__main__":
    unittest.main()
