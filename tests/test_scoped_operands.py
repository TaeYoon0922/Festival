"""Values kept attached to their scope and their source document."""

import unittest

from app.reasoning.corporate_event_field_evidence import CONTRACT_AMOUNT
from app.reasoning.scoped_operands import (
    OP_RANKING,
    ROLE_FINAL,
    ROLE_INITIAL,
    ROLE_MEMBER,
    TERMINATION_AMOUNT,
    OperandScope,
    amount_source,
    amount_sources,
    difference,
    ranking,
    resolve_operands,
)

CODE_A = "00000001"
CODE_B = "00000002"


def chunk(doc_id, amount, *, corp_code=CODE_A, rcept_dt="20240315",
          report_nm="단일판매ㆍ공급계약체결", chunk_id=None):
    """A served chunk shaped the way the chunker persists contract tables."""

    return {
        "doc_id": doc_id,
        "chunk_id": chunk_id or f"{doc_id}:c",
        "corp_code": corp_code,
        "rcept_dt": rcept_dt,
        "report_nm": report_nm,
        "source_table_id": "t1",
        "row_start": 0,
        "table_rows": [
            [{"text": "계약상대"}, {"text": "상대사"}],
            [{"text": "계약금액(원)"}, {"text": amount}],
        ],
    }


class AmountReadingTest(unittest.TestCase):
    def test_a_formal_amount_cell_is_read(self):
        source = amount_source(chunk("d1", "4,150,000,000"))

        self.assertEqual(source.value_for(CONTRACT_AMOUNT), 4150000000)
        self.assertEqual(source.doc_id, "d1")
        self.assertEqual(source.receipt_date, "2024-03-15")

    def test_the_corpus_blank_never_becomes_a_zero(self):
        """A blank cell must not enter arithmetic as a number."""

        self.assertIsNone(amount_source(chunk("d1", "-")))

    def test_a_termination_filing_states_a_termination_amount(self):
        source = amount_source(
            chunk("d2", "355,000,000", report_nm="단일판매ㆍ공급계약해지")
        )

        self.assertEqual(source.value_for(TERMINATION_AMOUNT), 355000000)

    def test_one_source_per_document(self):
        sources = amount_sources(
            [chunk("d1", "100"), chunk("d1", "200", chunk_id="d1:c2"), chunk("d2", "300")]
        )
        self.assertEqual([s.doc_id for s in sources], ["d1", "d2"])


class ScopeResolutionTest(unittest.TestCase):
    def test_a_date_scope_selects_its_own_filing(self):
        sources = amount_sources(
            [
                chunk("origin", "4,150,000,000", rcept_dt="20240315"),
                chunk("later", "9,999,999,999", rcept_dt="20241205"),
            ]
        )
        operands = resolve_operands(
            [
                OperandScope(role=ROLE_INITIAL, on_date="2024-03-15"),
                OperandScope(role=ROLE_FINAL, on_date="2024-12"),
            ],
            sources,
        )

        self.assertEqual(operands[0].source.doc_id, "origin")
        self.assertEqual(operands[1].source.doc_id, "later")

    def test_an_unrelated_contract_leaves_the_scope_unresolved(self):
        """Two filings satisfy the scope, so neither is chosen for it."""

        sources = amount_sources(
            [chunk("a", "100", rcept_dt="20240315"), chunk("b", "200", rcept_dt="20240315")]
        )
        operands = resolve_operands([OperandScope(role=ROLE_INITIAL)], sources)

        self.assertFalse(operands[0].resolved)
        self.assertIsNone(operands[0].value)

    def test_one_document_cannot_satisfy_two_scopes(self):
        sources = amount_sources([chunk("only", "100")])
        operands = resolve_operands(
            [OperandScope(role=ROLE_INITIAL), OperandScope(role=ROLE_FINAL)], sources
        )

        self.assertTrue(operands[0].resolved)
        self.assertFalse(operands[1].resolved)

    def test_a_company_scope_never_takes_another_companys_filing(self):
        sources = amount_sources(
            [chunk("a", "100", corp_code=CODE_A), chunk("b", "200", corp_code=CODE_B)]
        )
        operands = resolve_operands(
            [
                OperandScope(role=ROLE_MEMBER, corp_code=CODE_A, company="가"),
                OperandScope(role=ROLE_MEMBER, corp_code=CODE_B, company="나"),
            ],
            sources,
        )

        self.assertEqual(operands[0].source.corp_code, CODE_A)
        self.assertEqual(operands[1].source.corp_code, CODE_B)

    def test_a_company_with_no_filing_stays_unresolved(self):
        sources = amount_sources([chunk("a", "100", corp_code=CODE_A)])
        operands = resolve_operands(
            [
                OperandScope(role=ROLE_MEMBER, corp_code=CODE_A, company="가"),
                OperandScope(role=ROLE_MEMBER, corp_code=CODE_B, company="나"),
            ],
            sources,
        )

        self.assertTrue(operands[0].resolved)
        self.assertFalse(operands[1].resolved)

    def test_every_resolved_operand_keeps_its_source(self):
        sources = amount_sources([chunk("a", "100", corp_code=CODE_A)])
        operand = resolve_operands(
            [OperandScope(role=ROLE_MEMBER, corp_code=CODE_A, company="가")], sources
        )[0]

        self.assertEqual(operand.to_dict()["doc_id"], "a")
        self.assertEqual(operand.to_dict()["chunk_id"], "a:c")
        self.assertEqual(operand.to_dict()["label"], "가")


class ArithmeticTest(unittest.TestCase):
    def operands(self, initial, final, **kwargs):
        sources = amount_sources(
            [
                chunk("i", initial, rcept_dt="20240315"),
                chunk("f", final, rcept_dt="20241205", **kwargs),
            ]
        )
        return resolve_operands(
            [
                OperandScope(role=ROLE_INITIAL, on_date="2024-03-15"),
                OperandScope(role=ROLE_FINAL, on_date="2024-12-05"),
            ],
            sources,
        )

    def test_a_positive_delta(self):
        delta = difference(self.operands("4,150,000,000", "4,250,000,000"))

        self.assertEqual(delta.difference, 100000000)
        self.assertEqual(delta.pct_change, 2.41)

    def test_a_negative_delta(self):
        delta = difference(self.operands("1,000,000", "750,000"))

        self.assertEqual(delta.difference, -250000)
        self.assertEqual(delta.pct_change, -25.0)

    def test_the_delta_carries_both_sources(self):
        delta = difference(self.operands("100", "150")).to_dict()

        self.assertEqual([o["doc_id"] for o in delta["operands"]], ["i", "f"])

    def test_a_missing_operand_fails_closed(self):
        sources = amount_sources([chunk("i", "100", rcept_dt="20240315")])
        operands = resolve_operands(
            [
                OperandScope(role=ROLE_INITIAL, on_date="2024-03-15"),
                OperandScope(role=ROLE_FINAL, on_date="2024-12-05"),
            ],
            sources,
        )

        self.assertIsNone(difference(operands))

    def test_a_zero_start_reports_no_proportion(self):
        delta = difference(self.operands("0", "500"))

        self.assertEqual(delta.difference, 500)
        self.assertIsNone(delta.pct_change)

    def test_a_termination_amount_can_be_the_final_operand(self):
        sources = amount_sources(
            [
                chunk("i", "352,000,000", rcept_dt="20230228"),
                chunk(
                    "t", "362,424,124", rcept_dt="20241115",
                    report_nm="단일판매ㆍ공급계약해지",
                ),
            ]
        )
        operands = resolve_operands(
            [
                OperandScope(role=ROLE_INITIAL, on_date="2023-02-28"),
                OperandScope(
                    role=ROLE_FINAL, on_date="2024-11", field=TERMINATION_AMOUNT
                ),
            ],
            sources,
        )
        delta = difference(operands)

        self.assertEqual(delta.difference, 10424124)
        self.assertEqual(delta.final.source.doc_id, "t")


class RankingTest(unittest.TestCase):
    def build(self, *pairs):
        sources = amount_sources(
            [chunk(code, amount, corp_code=code) for code, amount in pairs]
        )
        return resolve_operands(
            [
                OperandScope(role=ROLE_MEMBER, corp_code=code, company=code)
                for code, _ in pairs
            ],
            sources,
        )

    def test_two_companies_order_by_amount(self):
        order = ranking(self.build(("A", "100"), ("B", "300")))

        self.assertEqual(order.largest.scope.company, "B")
        self.assertEqual([o.scope.company for o in order.operands], ["B", "A"])

    def test_two_companies_report_their_difference(self):
        self.assertEqual(ranking(self.build(("A", "100"), ("B", "300"))).spread, 200)

    def test_four_companies_order_by_amount(self):
        order = ranking(
            self.build(("A", "10"), ("B", "40"), ("C", "30"), ("D", "20"))
        )

        self.assertEqual([o.scope.company for o in order.operands], ["B", "C", "D", "A"])
        self.assertEqual(order.to_dict()["operation"], OP_RANKING)

    def test_a_partial_ranking_is_refused(self):
        """A missing member changes the order, so nothing is stated."""

        sources = amount_sources([chunk("A", "100", corp_code="A")])
        operands = resolve_operands(
            [
                OperandScope(role=ROLE_MEMBER, corp_code="A", company="A"),
                OperandScope(role=ROLE_MEMBER, corp_code="B", company="B"),
            ],
            sources,
        )

        self.assertIsNone(ranking(operands))

    def test_every_ranked_member_keeps_its_own_provenance(self):
        order = ranking(self.build(("A", "100"), ("B", "300")))
        docs = {o.scope.company: o.source.doc_id for o in order.operands}

        self.assertEqual(docs, {"A": "A", "B": "B"})


if __name__ == "__main__":
    unittest.main()


class AmountChangeRecognitionTest(unittest.TestCase):
    """Which two filings a change question names, read from the question alone."""

    def request(self, question):
        from app.reasoning.amount_change import amount_change_requested

        return amount_change_requested(question)

    def test_two_dates_and_a_change_are_required(self):
        found = self.request(
            "가상사가 2024년 3월 15일 공시한 계약의 최초 계약금액과 "
            "2024년 12월 정정 후 계약금액의 차이는 얼마인가?"
        )

        self.assertEqual(found.initial_on, "2024-03-15")
        self.assertEqual(found.final_on, "2024-12")
        self.assertEqual(found.final_field, CONTRACT_AMOUNT)

    def test_a_termination_amount_is_requested_by_name(self):
        found = self.request(
            "가상사가 2023년 2월 28일 공시한 계약의 최초 계약금액과 "
            "2024년 11월 해지 공시의 해지금액은 얼마나 차이가 나?"
        )

        self.assertEqual(found.final_field, TERMINATION_AMOUNT)

    def test_every_direction_of_change_is_recognized(self):
        for wording in ("얼마나 늘었어?", "얼마나 감소했나?", "얼마나 변경됐는지 증감액으로 알려줘"):
            with self.subTest(wording=wording):
                self.assertIsNotNone(
                    self.request(
                        f"가상사가 2024년 7월 3일 공시한 계약의 계약금액은 "
                        f"2025년 1월 정정에서 {wording}"
                    )
                )

    def test_one_date_cannot_pose_the_question(self):
        self.assertIsNone(
            self.request("가상사가 2024년 7월 3일 공시한 계약의 계약금액 차이는?")
        )

    def test_an_amount_without_change_intent_is_not_this(self):
        self.assertIsNone(
            self.request("가상사가 2024년 7월 3일 공시한 계약의 계약금액은 얼마야?")
        )

    def test_a_lifecycle_question_is_not_this(self):
        self.assertIsNone(self.request("2024년 6월 18일 체결 공시한 계약은 해지됐나?"))

    def test_a_contract_period_end_date_is_not_this(self):
        self.assertIsNone(self.request("계약의 계약기간 종료일은?"))
