# -*- coding: utf-8 -*-
"""Synthetic tests for the Independent Eval v2 scorer.

Evaluation-only. Every payload here is hand-built; the agent is never called and no
prediction from the frozen system is used.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluator import (  # noqa: E402
    DERIVED_PCT_TOLERANCE,
    IndependentV2Evaluator,
    normalize_numbers,
    numeric_match,
)


def ctx(*docs, chunk_suffix="ch_x"):
    return [{"rank": i, "doc_id": d, "chunk_id": f"{d}:{chunk_suffix}"}
            for i, d in enumerate(docs, 1)]


def payload(answer, *, docs=(), answerable=True, warnings=(), cites=None):
    """cites: 1-based ranks the answer references. Defaults to every served row."""
    ranks = list(range(1, len(docs) + 1)) if cites is None else list(cites)
    marked = answer + "".join(f"[{r}]" for r in ranks)
    return {"answer": marked,
            "retrieved_context": ctx(*docs),
            "think_trace": {"answerable": answerable, "stages": [],
                            "selected_evidence_count": len(docs), "warnings": list(warnings)}}


NUMERIC_GOLD = {
    "id": "S1", "category": "T9_holding_latest", "expected_behavior": "answer",
    "expected_answer": "1,914,033주", "gold_numeric": {"value": 1914033, "unit": "shares"},
    "gold_doc_ids": ["holding_A"], "gold_chunk_ids": ["holding_A:ch_x"],
    "citation_policy": "single", "required_gold_doc_ids": ["holding_A"],
    "acceptable_gold_doc_ids": [], "derived_values": {},
}
PERCENT_GOLD = dict(NUMERIC_GOLD, id="S2", expected_answer="6.36%",
                    gold_numeric={"value": 6.36, "unit": "percent"})
ROLE_GOLD = dict(NUMERIC_GOLD, id="S3",
                 expected_answer="정정 전 183,202,431주 / 정정 후 183,323,596주",
                 gold_numeric={"value": 183323596, "unit": "shares",
                               "secondary": {"value": 183202431, "unit": "shares",
                                             "role": "superseded"}})
REFUSAL_GOLD = {
    "id": "S4", "category": "T15_first_report_no_previous",
    "expected_behavior": "insufficient_evidence", "expected_answer": None, "gold_numeric": None,
    "gold_doc_ids": ["holding_B"], "gold_chunk_ids": ["holding_B:ch_x"],
    "citation_policy": "single", "required_gold_doc_ids": ["holding_B"],
    "acceptable_gold_doc_ids": [], "derived_values": {},
    "source_values": {"보유주식수": "15,041,797", "직전 보유주식수": "-"},
    "ambiguity_reason": "first filing", "forbidden_fallbacks": ["answer 0 shares", "restate 15,041,797"],
}
CLARIFY_GOLD = {
    "id": "S5", "category": "T18_negative_insufficient", "expected_behavior": "clarify",
    "expected_answer": None, "gold_numeric": None,
    "gold_doc_ids": ["ex_A", "ex_B"], "gold_chunk_ids": [],
    "citation_policy": "any_of", "required_gold_doc_ids": [],
    "acceptable_gold_doc_ids": ["ex_A", "ex_B"], "derived_values": {},
    "source_values": {"c1": "240,993,039,040", "c2": "922,746,708,000"},
    "ambiguity_reason": "two contracts share a title",
    "clarification_requirements": ["어느 계약", "공시일"],
    "forbidden_fallbacks": ["240,993,039,040", "922,746,708,000"],
}
ANY_OF_GOLD = {
    "id": "S6", "category": "T20_same_day_two_filings", "expected_behavior": "answer",
    "expected_answer": "934,443주", "gold_numeric": {"value": 934443, "unit": "shares"},
    "gold_doc_ids": ["h_A", "h_B"], "gold_chunk_ids": [],
    "citation_policy": "any_of", "required_gold_doc_ids": [],
    "acceptable_gold_doc_ids": ["h_A", "h_B"], "derived_values": {},
}
DERIVED_GOLD = {
    "id": "S7", "category": "T5_multi_doc_synthesis", "expected_behavior": "answer",
    "expected_answer": "+100,000,000원 (+2.41%)",
    "gold_numeric": {"value": 100000000, "unit": "KRW",
                     "secondary": {"value": 2.41, "unit": "percent"}},
    "gold_doc_ids": ["ex_1", "ex_2"], "gold_chunk_ids": [],
    "citation_policy": "all_required", "required_gold_doc_ids": ["ex_1", "ex_2"],
    "acceptable_gold_doc_ids": [],
    "source_values": {"initial_contract_amount": 4150000000, "final_contract_amount": 4250000000},
    "derived_values": {"difference": 100000000, "pct_change": 2.4096},
    "derivation": "final - initial",
}
DEICTIC_GOLD = {
    "id": "S8", "category": "T19_deictic_context_dependent",
    "expected_behavior": "insufficient_evidence", "expected_answer": None, "gold_numeric": None,
    "gold_doc_ids": [], "gold_chunk_ids": [], "citation_policy": "none",
    "required_gold_doc_ids": [], "acceptable_gold_doc_ids": [], "derived_values": {},
    "source_values": {}, "ambiguity_reason": "no report named",
    "forbidden_fallbacks": ["answer from the latest filing"],
}
ORDER_GOLD = {
    "id": "S9", "category": "T7_multi_company_comparison", "expected_behavior": "answer",
    "expected_answer": "A > B > C", "gold_numeric": None,
    "gold_doc_ids": ["d1", "d2", "d3"], "gold_chunk_ids": [],
    "citation_policy": "all_required", "required_gold_doc_ids": ["d1", "d2", "d3"],
    "acceptable_gold_doc_ids": [],
    "derived_values": {"descending_order": ["HD현대중공업", "현대로템", "LIG"]},
    "derivation": "sort desc",
}


class NumericNormalisation(unittest.TestCase):
    def test_thousands_and_korean_scale(self):
        self.assertIn(120000000.0, normalize_numbers("1억 2,000만원"))
        self.assertIn(4250000000.0, normalize_numbers("계약금액은 4,250,000,000원"))

    def test_percent_tolerance_is_bounded(self):
        gold = {"value": 2.41, "unit": "percent"}
        self.assertTrue(numeric_match(gold, "약 2.42% 증가", derived=True))
        self.assertFalse(numeric_match(gold, "약 2.5% 증가", derived=True))
        self.assertLessEqual(DERIVED_PCT_TOLERANCE, 0.01)

    def test_exact_extraction_has_no_tolerance(self):
        self.assertFalse(numeric_match({"value": 1914033, "unit": "shares"}, "1,914,034주"))


class ScoringMatrix(unittest.TestCase):
    def setUp(self):
        self.ev = IndependentV2Evaluator()

    # 1
    def test_numeric_exact_pass(self):
        r = self.ev.score(NUMERIC_GOLD, payload("보유주식수는 1,914,033주입니다.", docs=["holding_A"]))
        self.assertTrue(r.overall_pass)
        self.assertTrue(r.answer_correct)

    # 2
    def test_numeric_wrong_fail(self):
        r = self.ev.score(NUMERIC_GOLD, payload("보유주식수는 1,905,209주입니다.", docs=["holding_A"]))
        self.assertFalse(r.answer_correct)
        self.assertFalse(r.overall_pass)
        self.assertTrue(r.evidence_correct)  # axes stay independent

    # 3
    def test_percent_normalised_pass(self):
        r = self.ev.score(PERCENT_GOLD, payload("보유비율은 6.36% 입니다.", docs=["holding_A"]))
        self.assertTrue(r.answer_correct)

    # 4
    def test_role_swapped_correction_values_fail(self):
        swapped = payload("정정 전 183,323,596주, 정정 후 183,202,431주입니다.", docs=["holding_A"])
        r = self.ev.score(ROLE_GOLD, swapped)
        self.assertFalse(r.answer_correct)
        self.assertIn("role_swap", r.sub_reasons)

    def test_role_correct_pass(self):
        good = payload("정정 전 183,202,431주, 정정 후 183,323,596주입니다.", docs=["holding_A"])
        self.assertTrue(self.ev.score(ROLE_GOLD, good).answer_correct)

    # 5
    def test_insufficient_correct_refusal_pass(self):
        r = self.ev.score(REFUSAL_GOLD,
                          payload("제공된 공시 근거만으로는 직전 보유주식수를 확인할 수 없습니다.",
                                  docs=["holding_B"], answerable=False))
        self.assertTrue(r.answerability_correct)
        self.assertTrue(r.answer_correct)
        self.assertTrue(r.overall_pass, "a correct refusal must not be scored as a failure")

    # 6
    def test_insufficient_factual_fallback_fail(self):
        r = self.ev.score(REFUSAL_GOLD,
                          payload("직전 보유주식수는 0주입니다.", docs=["holding_B"]))
        self.assertFalse(r.answerability_correct)
        self.assertFalse(r.answer_correct)
        self.assertEqual(r.severity, "P0_false_positive")

    def test_insufficient_restating_current_value_fail(self):
        r = self.ev.score(REFUSAL_GOLD,
                          payload("직전 보유주식수는 15,041,797주입니다.", docs=["holding_B"]))
        self.assertFalse(r.answerability_correct)
        self.assertEqual(r.severity, "P0_false_positive")

    # 7
    def test_clarify_correct_behaviour_pass(self):
        r = self.ev.score(CLARIFY_GOLD,
                          payload("해당 조건에 맞는 계약이 두 건 있습니다. 어느 계약을 말씀하시는지 특정해 주세요.",
                                  docs=["ex_A"], answerable=False))
        self.assertTrue(r.answerability_correct)

    # 8
    def test_clarify_arbitrary_choice_fail(self):
        r = self.ev.score(CLARIFY_GOLD,
                          payload("계약금액은 922,746,708,000원입니다.", docs=["ex_A"]))
        self.assertFalse(r.answerability_correct)
        self.assertEqual(r.severity, "P0_false_positive")

    def test_clarify_flat_refusal_is_not_a_pass(self):
        r = self.ev.score(CLARIFY_GOLD,
                          payload("확인할 수 없습니다.", docs=["ex_A"], answerable=False))
        self.assertFalse(r.answerability_correct)
        self.assertIn("refused_without_disambiguation", r.sub_reasons)

    # 9
    def test_citation_single_pass_and_fail(self):
        ok = self.ev.score(NUMERIC_GOLD, payload("1,914,033주", docs=["holding_A"]))
        bad = self.ev.score(NUMERIC_GOLD, payload("1,914,033주", docs=["holding_Z"]))
        self.assertTrue(ok.citation_correct)
        self.assertFalse(bad.citation_correct)
        self.assertTrue(bad.answer_correct, "answer stays correct when only the citation is wrong")

    # 10 / 11 / 12
    def test_citation_any_of(self):
        a = self.ev.score(ANY_OF_GOLD, payload("934,443주", docs=["h_A"]))
        b = self.ev.score(ANY_OF_GOLD, payload("934,443주", docs=["h_B"]))
        none = self.ev.score(ANY_OF_GOLD, payload("934,443주", docs=["h_Z"]))
        self.assertTrue(a.citation_correct)
        self.assertTrue(b.citation_correct)
        self.assertFalse(none.citation_correct)
        self.assertTrue(a.overall_pass and b.overall_pass)

    # 13 / 14
    def test_citation_all_required(self):
        full = self.ev.score(DERIVED_GOLD,
                             payload("차이는 100,000,000원이며 2.41% 증가했습니다.", docs=["ex_1", "ex_2"]))
        part = self.ev.score(DERIVED_GOLD,
                             payload("차이는 100,000,000원이며 2.41% 증가했습니다.",
                                     docs=["ex_1", "ex_2"], cites=[1]))
        self.assertTrue(full.citation_correct)
        self.assertFalse(part.citation_correct)
        self.assertAlmostEqual(part.citation_recall, 0.5)

    # 15
    def test_derived_correct_but_citation_incomplete(self):
        # both filings served, but the answer references only the first
        r = self.ev.score(DERIVED_GOLD,
                          payload("계약금액 차이는 100,000,000원, 약 2.41% 증가입니다.",
                                  docs=["ex_1", "ex_2"], cites=[1]))
        self.assertTrue(r.answer_correct)
        self.assertTrue(r.evidence_correct, "both documents were served")
        self.assertFalse(r.citation_correct, "only one was cited")
        self.assertAlmostEqual(r.citation_recall, 0.5)
        self.assertFalse(r.overall_pass)
        self.assertEqual(r.first_failing_stage, "C1")

    def test_multi_doc_served_only_one_is_M1(self):
        r = self.ev.score(DERIVED_GOLD,
                          payload("계약금액 차이는 100,000,000원, 약 2.41% 증가입니다.", docs=["ex_2"]))
        self.assertFalse(r.evidence_correct)
        self.assertEqual(r.first_failing_stage, "M1")

    # 16
    def test_answer_wrong_but_citations_correct(self):
        r = self.ev.score(DERIVED_GOLD,
                          payload("차이는 90,000,000원입니다.", docs=["ex_1", "ex_2"]))
        self.assertFalse(r.answer_correct)
        self.assertTrue(r.citation_correct)
        self.assertTrue(r.evidence_correct)

    # 17
    def test_answer_correct_but_evidence_wrong(self):
        r = self.ev.score(NUMERIC_GOLD, payload("1,914,033주", docs=["unrelated_doc"]))
        self.assertTrue(r.answer_correct)
        self.assertFalse(r.evidence_correct)
        self.assertFalse(r.citation_correct)

    # 18
    def test_presentation_only_is_not_a_factual_failure(self):
        verbose = ("보유주식수는 1,914,033주입니다. " + "부연 설명입니다. " * 40)
        r = self.ev.score(NUMERIC_GOLD, payload(verbose, docs=["holding_A"]))
        self.assertTrue(r.overall_pass)
        self.assertTrue(r.presentation_issue)
        self.assertEqual(r.first_failing_stage, "P1")


class DeicticAndAttribution(unittest.TestCase):
    def setUp(self):
        self.ev = IndependentV2Evaluator()

    def test_deictic_fail_closed_with_no_evidence_passes(self):
        r = self.ev.score(DEICTIC_GOLD,
                          payload("현재 확보된 공시 근거만으로는 확인하기 어렵습니다.",
                                  docs=(), answerable=False, cites=[]))
        self.assertTrue(r.overall_pass)

    def test_deictic_leaking_ranked_evidence_fails(self):
        r = self.ev.score(DEICTIC_GOLD,
                          payload("현재 확보된 공시 근거만으로는 확인하기 어렵습니다.",
                                  docs=["holding_latest"], answerable=False, cites=[1]))
        self.assertFalse(r.evidence_correct)
        self.assertFalse(r.citation_correct)

    def test_ordering_requires_correct_sequence(self):
        good = payload("HD현대중공업, 현대로템, LIG 순입니다.", docs=["d1", "d2", "d3"])
        bad = payload("현대로템, HD현대중공업, LIG 순입니다.", docs=["d1", "d2", "d3"])
        self.assertTrue(self.ev.score(ORDER_GOLD, good).answer_correct)
        self.assertFalse(self.ev.score(ORDER_GOLD, bad).answer_correct)

    def test_environment_warning_attributes_to_ENV(self):
        r = self.ev.score(NUMERIC_GOLD,
                          payload("1,905,209주", docs=["holding_A"],
                                  warnings=["vector backend error"]))
        self.assertEqual(r.first_failing_stage, "ENV")

    def test_refusal_on_answerable_item_is_A1(self):
        r = self.ev.score(NUMERIC_GOLD,
                          payload("확인할 수 없습니다.", docs=["holding_A"], answerable=False))
        self.assertFalse(r.answerability_correct)
        self.assertEqual(r.first_failing_stage, "A1")
        self.assertEqual(r.severity, "A1_over_refusal")

    def test_unknown_is_used_rather_than_guessing(self):
        r = self.ev.score(PERCENT_GOLD, payload("보유비율은 7.00%입니다.", docs=["holding_A"]))
        self.assertFalse(r.answer_correct)
        self.assertEqual(r.first_failing_stage, "UNKNOWN")


if __name__ == "__main__":
    unittest.main(verbosity=2)
