"""Generic current-holding amount: a question that names no unit gets both.

The frozen grammar leaves ``metric`` unset when a question says only
"얼마나", because the word selects no unit.  These tests pin the narrow case
where the corpus answers that question with a share count and a ratio
together, and -- more importantly -- pin every neighbouring question that
borrows the same adverb for a different quantity and must stay closed.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.api.pipeline import AnswerPipeline
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.holding_evidence_coverage import (
    CURRENT_HOLDING_STATE_FIELDS,
    requested_holding_fields,
)
from app.reasoning.holding_event_resolver import _requested_fields
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope, QueryState, QueryValidator

import tests.test_holding_company_role_execution as role


ISSUER = role.ISSUER
REPORTER = role.REPORTER
PAIR = ("after_shares", "after_ratio")


def _understanding() -> QueryUnderstanding:
    return QueryUnderstanding({ISSUER: ISSUER, REPORTER: REPORTER})


def fields(question: str) -> tuple[str, ...]:
    plan = _understanding().understand(question)
    return requested_holding_fields(question, plan)


def plan_of(question: str):
    return _understanding().understand(question)


class GenericCurrentAmountTests(unittest.TestCase):
    """The guarded positive case: no unit named, so both are requested."""

    def test_class_a_shares_possession_requests_the_pair(self) -> None:
        question = f"{REPORTER}가 {ISSUER} 주식을 2024년 2월 3일 기준으로 얼마나 들고 있어?"
        plan = plan_of(question)

        self.assertIsNone(plan.metric, "the plan must not name a unit it was never given")
        self.assertEqual(
            plan.evidence["holding_ownership_intent"], "company_has_company_shares"
        )
        self.assertEqual(requested_holding_fields(question, plan), PAIR)

    def test_class_b_ownership_interest_requests_the_pair(self) -> None:
        question = f"{REPORTER}의 {ISSUER} 지분은 2024년 2월 3일 보고 기준 얼마나 되나?"
        plan = plan_of(question)

        self.assertIsNone(plan.metric)
        self.assertEqual(
            plan.evidence["holding_ownership_intent"], "company_ownership_interest"
        )
        self.assertEqual(requested_holding_fields(question, plan), PAIR)

    def test_bare_stative_without_a_date_requests_the_pair(self) -> None:
        question = f"{REPORTER}가 보유한 {ISSUER} 주식은 얼마나 되나?"
        self.assertEqual(fields(question), PAIR)

    def test_pair_ordering_is_shares_then_ratio(self) -> None:
        self.assertEqual(CURRENT_HOLDING_STATE_FIELDS, PAIR)
        self.assertEqual(fields(f"{REPORTER}가 보유한 {ISSUER} 주식은 얼마나 되나?"), PAIR)


class NonStativePredicateTests(unittest.TestCase):
    """Every question below sits in a recognised ownership family and says
    "얼마나".  None of them asks for the current holding, so none may open."""

    def assert_closed(self, question: str) -> None:
        got = fields(question)
        self.assertNotEqual(got, PAIR, f"generic pair leaked into: {question}")

    def test_duration(self) -> None:
        self.assert_closed(f"{REPORTER}가 보유한 {ISSUER} 주식은 얼마나 오래됐어?")

    def test_valuation(self) -> None:
        self.assert_closed(f"{REPORTER}가 보유한 {ISSUER} 주식은 얼마나 가치가 있어?")
        self.assert_closed(f"{REPORTER}의 {ISSUER} 지분은 얼마 짜리야?")
        self.assert_closed(f"{REPORTER}의 {ISSUER} 지분 금액은 얼마나 되나?")

    def test_transaction(self) -> None:
        for verb in ("매각됐어", "처분했어", "취득했어", "인수됐어", "양수했어", "양도했어"):
            with self.subTest(verb=verb):
                self.assert_closed(f"{REPORTER}의 {ISSUER} 지분은 얼마나 {verb}?")

    def test_transaction_inside_a_possession_clause(self) -> None:
        self.assert_closed(f"{REPORTER}가 {ISSUER} 주식을 얼마나 처분했는지 들고 있어?")

    def test_change(self) -> None:
        for verb in ("늘었어", "줄었어", "증가했어", "감소했어", "증감했어", "변동했어"):
            with self.subTest(verb=verb):
                self.assert_closed(f"{REPORTER}의 {ISSUER} 지분은 얼마나 {verb}?")

    def test_ownership_compounds(self) -> None:
        for noun in ("지분증권", "비지배지분", "지분법 적용 지분"):
            with self.subTest(noun=noun):
                self.assert_closed(f"{REPORTER}의 {ISSUER} {noun}은 얼마나 되나?")


class StativeInflectionTests(unittest.TestCase):
    """``되`` opens compound verbs as well as ending ``되다``.

    The positive grammar accepts a closed set of endings and otherwise
    requires the word to stop at the stem, so a following verb stem can never
    be read as a stative ending.
    """

    def stake_question(self, tail: str) -> str:
        return f"{REPORTER}의 {ISSUER} 지분은 {tail}"

    def test_closed_stative_endings_are_accepted(self) -> None:
        for tail in (
            "얼마나 돼?",
            "얼마나 돼요?",
            "얼마나 되나?",
            "얼마나 되나요?",
            "얼마나 되는지 알려줘",
            "얼마나 되죠?",
            "얼마나 됩니까?",
            "얼마나 되냐?",
        ):
            with self.subTest(tail=tail):
                self.assertEqual(fields(self.stake_question(tail)), PAIR)

    def test_compound_verbs_beginning_with_the_stem_are_refused(self) -> None:
        for tail in (
            "얼마나 되팔았어?",       # resell
            "얼마나 되샀어?",         # buy back
            "얼마나 되돌려받았어?",   # take back
            # verbs the review never named: the rule is structural, not a
            # blocklist of the three examples that exposed it
            "얼마나 되찾았어?",
            "얼마나 되넘겼어?",
            "얼마나 되갚았어?",
        ):
            with self.subTest(tail=tail):
                self.assertNotEqual(fields(self.stake_question(tail)), PAIR)

    def test_progressive_is_not_part_of_the_closed_contract(self) -> None:
        self.assertNotEqual(fields(self.stake_question("얼마나 되고 있어?")), PAIR)


class ChangeInflectionTests(unittest.TestCase):
    """Causative change inflections describe a change, not a current state."""

    def test_causative_change_inflections_are_refused(self) -> None:
        for tail in (
            "얼마나 늘려서 들고 있어?",
            "얼마나 줄여서 들고 있어?",
            "얼마나 늘렸다가 들고 있어?",
            "얼마나 줄였다가 들고 있어?",
        ):
            with self.subTest(tail=tail):
                self.assertNotEqual(
                    fields(f"{REPORTER}가 {ISSUER} 주식을 {tail}"), PAIR
                )

    def test_plain_change_inflections_remain_refused(self) -> None:
        for tail in ("얼마나 늘었어?", "얼마나 줄었어?"):
            with self.subTest(tail=tail):
                self.assertNotEqual(
                    fields(f"{REPORTER}의 {ISSUER} 지분은 {tail}"), PAIR
                )


class MarkedFormRegressionTests(unittest.TestCase):
    """Questions that do name a unit keep naming exactly that unit."""

    def test_share_wording_stays_shares_only(self) -> None:
        for question in (
            f"{REPORTER}가 보유한 {ISSUER} 주식은 몇 주야?",
            f"{REPORTER}가 보유한 {ISSUER} 주식수는?",
            f"{REPORTER}가 보유한 {ISSUER} 보유수량은?",
        ):
            with self.subTest(question=question):
                self.assertEqual(fields(question), ("after_shares",))

    def test_ratio_wording_stays_ratio_only(self) -> None:
        for question in (
            f"{REPORTER}의 {ISSUER} 지분율은?",
            f"{REPORTER}가 보유한 {ISSUER} 보유비율은?",
            f"{REPORTER}의 {ISSUER} 지분은 몇 퍼센트야?",
        ):
            with self.subTest(question=question):
                self.assertEqual(fields(question), ("after_ratio",))

    def test_explicitly_asking_for_both_is_unchanged(self) -> None:
        question = f"{ISSUER} {REPORTER} 이번 보고 보유 주식수와 비율"
        plan = plan_of(question)
        self.assertEqual(plan.metric, "holding_shares")
        self.assertEqual(requested_holding_fields(question, plan), PAIR)

    def test_one_company_wording_is_untouched(self) -> None:
        understanding = QueryUnderstanding({ISSUER: ISSUER})
        for question in (
            f"{ISSUER} 주식을 얼마나 들고 있어?",
            f"{ISSUER} 지분은 얼마나 되나?",
        ):
            with self.subTest(question=question):
                plan = understanding.understand(question)
                self.assertIsNone(plan.evidence.get("holding_ownership_intent"))
                self.assertEqual(requested_holding_fields(question, plan), ())

    def test_acquisition_wording_is_untouched(self) -> None:
        question = f"{REPORTER} {ISSUER} 풋옵션 행사 주식 취득일과 취득 수량"
        plan = plan_of(question)
        self.assertNotEqual(requested_holding_fields(question, plan), PAIR)


class ValidationRepresentationTests(unittest.TestCase):
    def setUp(self) -> None:
        harness = role.HoldingCompanyRoleExecutionTests(
            "test_pipeline_uses_one_shared_index_dependency"
        )
        harness.setUp()
        self.harness = harness
        self.validator = harness.validator
        self.understanding = harness.understanding

    def validated(self, question: str):
        return self.validator.validate(self.understanding.understand(question))

    def test_generic_request_resolves_without_naming_a_metric(self) -> None:
        for question in (
            f"{REPORTER}가 {ISSUER} 주식을 2024년 2월 3일 기준으로 얼마나 들고 있어?",
            f"{REPORTER}의 {ISSUER} 지분은 2024년 2월 3일 보고 기준 얼마나 되나?",
        ):
            with self.subTest(question=question):
                result = self.validated(question)
                self.assertIs(result.state, QueryState.RESOLVED)
                self.assertTrue(result.retrieval_allowed)
                self.assertEqual(result.slots["metric"].status.value, "resolved")
                # the whole point: the slot is satisfied, the plan is not relabelled
                self.assertIsNone(result.plan.metric)

    def test_plan_never_gains_a_metric_name(self) -> None:
        result = self.validated(
            f"{REPORTER}의 {ISSUER} 지분은 2024년 2월 3일 보고 기준 얼마나 되나?"
        )
        self.assertIsNone(result.plan.metric)
        self.assertNotIn("holding_both", str(result.plan.to_dict()))
        # the internal slot filler must not reach the public validation contract
        self.assertEqual(
            set(result.to_validation_dict()) - {"reason"},
            {"status", "retrieval_allowed"},
        )

    def test_non_stative_question_stays_incomplete(self) -> None:
        result = self.validated(f"{REPORTER}의 {ISSUER} 지분은 얼마나 매각됐어?")
        self.assertIs(result.state, QueryState.INCOMPLETE)
        self.assertFalse(result.retrieval_allowed)
        self.assertEqual(result.slots["metric"].status.value, "missing")

    def test_other_field_tuples_do_not_satisfy_the_slot(self) -> None:
        """Non-emptiness alone must never fill the metric slot."""

        result = self.validated(f"{REPORTER}의 {ISSUER} 지분은 얼마나 감소했어?")
        requested = requested_holding_fields(str(result.plan.raw_query), result.plan)
        self.assertNotEqual(requested, PAIR)
        if result.plan.metric is None:
            self.assertEqual(result.slots["metric"].status.value, "missing")


class OneSourceOfTruthTests(unittest.TestCase):
    def test_validation_and_resolver_observe_the_same_tuple(self) -> None:
        question = f"{REPORTER}의 {ISSUER} 지분은 2024년 2월 3일 보고 기준 얼마나 되나?"
        plan = plan_of(question)

        via_coverage = requested_holding_fields(question, plan)   # validation path
        via_resolver = _requested_fields(question, plan.to_dict())  # resolver path

        self.assertEqual(via_coverage, via_resolver)
        self.assertEqual(via_coverage, PAIR)

    def test_validation_module_holds_no_stative_predicate_of_its_own(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "app" / "reasoning" / "query_validation.py"
        ).read_text(encoding="utf-8")
        for token in (
            "얼마나", "들고", "가지고", "보유하고",
            "매각", "처분", "양수", "양도", "지분증권", "비지배지분", "지분법",
            "오래", "얼마 짜리", "얼마짜리",
        ):
            with self.subTest(token=token):
                self.assertNotIn(
                    token, source,
                    "the stative decision must live only in the request parser",
                )
        self.assertIsNone(re.search(r"holding_both", source))


class GenericCurrentAmountExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        harness = role.HoldingCompanyRoleExecutionTests(
            "test_pipeline_uses_one_shared_index_dependency"
        )
        harness.setUp()
        self.harness = harness

    def test_both_fields_render_from_one_cited_projection(self) -> None:
        harness = self.harness
        question = f"{REPORTER}가 {ISSUER} 주식을 2024년 2월 3일 기준으로 얼마나 들고 있어?"
        correct = role.candidate(harness.correct)
        wrong = role.candidate(harness.wrong)
        executor = role.StaticExecutor(
            (wrong, correct), (role.ranked(wrong, 1), role.ranked(correct, 2))
        )
        pipeline = AnswerPipeline(
            understanding=harness.understanding,
            executor=executor,
            orchestrator=harness.orchestrator,
            query_validator=harness.validator,
            answerability_guard=AnswerabilityGuard(),
        )

        payload = pipeline.answer("B3-BOTH", question)
        answer = payload["answer"]

        self.assertIsNone(executor.plan.metric, "retrieval must run without a metric")
        self.assertTrue(payload["think_trace"]["answerable"])
        # one projection, and it is the right holder's
        self.assertEqual(
            [row["doc_id"] for row in payload["retrieved_context"]],
            [harness.correct.doc_id],
        )
        self.assertNotIn(harness.wrong.doc_id, answer)
        self.assertNotIn("999", answer)
        # both values, from that one projection
        self.assertIn(harness.correct.after_shares, answer)
        self.assertIn(harness.correct.after_ratio, answer)
        # a single citation covers both
        self.assertIn("[1]", answer)
        self.assertNotIn("[2]", answer)
        self.assertEqual(len(payload["retrieved_context"]), 1)


class BoundaryRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        harness = role.HoldingCompanyRoleExecutionTests(
            "test_pipeline_uses_one_shared_index_dependency"
        )
        harness.setUp()
        self.harness = harness

    def answered(self, question: str):
        harness = self.harness
        correct = role.candidate(harness.correct)
        wrong = role.candidate(harness.wrong)
        executor = role.StaticExecutor(
            (wrong, correct), (role.ranked(wrong, 1), role.ranked(correct, 2))
        )
        pipeline = AnswerPipeline(
            understanding=harness.understanding,
            executor=executor,
            orchestrator=harness.orchestrator,
            query_validator=harness.validator,
            answerability_guard=AnswerabilityGuard(),
        )
        return pipeline.answer("B3-BOUNDARY", question)

    def test_unbound_deictic_still_fails_closed(self) -> None:
        payload = self.answered(
            f"{REPORTER}가 보유한 {ISSUER} 주식은 이번 보고 기준 얼마나 돼?"
        )
        self.assertIn(
            "holding_report_relative_execution", payload["think_trace"]["stages"]
        )
        self.assertEqual(payload["retrieved_context"], [])
        self.assertFalse(payload["think_trace"]["answerable"])

    def test_latest_selector_still_owned_by_phase_three(self) -> None:
        payload = self.answered(
            f"{REPORTER}가 보유한 {ISSUER} 주식은 최신 보고 기준 얼마나 돼?"
        )
        stages = payload["think_trace"]["stages"]
        self.assertIn("holding_report_relative_execution", stages)
        self.assertNotIn("holding_reporter_scope", stages)

    def test_role_resolution_and_reporter_scope_still_apply(self) -> None:
        payload = self.answered(
            f"{REPORTER}가 {ISSUER} 주식을 2024년 2월 3일 기준으로 얼마나 들고 있어?"
        )
        self.assertIn("holding_reporter_scope", payload["think_trace"]["stages"])
        self.assertEqual(
            [row["doc_id"] for row in payload["retrieved_context"]],
            [self.harness.correct.doc_id],
        )

    def test_generic_without_report_wording_uses_the_ordinary_holding_path(self) -> None:
        """No report-relative wording: the ordinary path answers it.

        Requesting both fields makes these questions retrieval-eligible, which
        also makes them visible to Phase 3's own gate.  That gate additionally
        requires report-relative wording, so a bare reference date stays on the
        ordinary path with B.2 reporter scoping.
        """

        payload = self.answered(
            f"{REPORTER}가 {ISSUER} 주식을 2024년 2월 3일 기준으로 얼마나 들고 있어?"
        )
        stages = payload["think_trace"]["stages"]

        self.assertNotIn("holding_report_relative_execution", stages)
        self.assertIn("holding_reporter_scope", stages)
        self.assertTrue(payload["think_trace"]["answerable"])
        self.assertIn(self.harness.correct.after_shares, payload["answer"])
        self.assertIn(self.harness.correct.after_ratio, payload["answer"])

    def test_generic_with_report_selector_is_owned_by_phase_three(self) -> None:
        """Report-relative wording plus an exact date: Phase 3 owns selection.

        This is the consequence the review asked to be pinned rather than
        changed.  The index picks the filing, so reporter scoping is not
        additionally applied, and both fields still come from that one
        projection.
        """

        payload = self.answered(
            f"{REPORTER}의 {ISSUER} 지분은 2024년 2월 3일 보고 기준 얼마나 되나?"
        )
        stages = payload["think_trace"]["stages"]

        self.assertIn("holding_report_relative_execution", stages)
        self.assertNotIn("holding_reporter_scope", stages)
        self.assertTrue(payload["think_trace"]["answerable"])
        self.assertEqual(
            [row["doc_id"] for row in payload["retrieved_context"]],
            [self.harness.correct.doc_id],
        )
        self.assertIn(self.harness.correct.after_shares, payload["answer"])
        self.assertIn(self.harness.correct.after_ratio, payload["answer"])
        self.assertIn("[1]", payload["answer"])
        self.assertNotIn("[2]", payload["answer"])

    def test_generic_with_unbound_selector_stays_fail_closed(self) -> None:
        """STEP 10-A keeps ownership of an unbound deictic selector."""

        payload = self.answered(
            f"{REPORTER}의 {ISSUER} 지분은 이번 보고 기준 얼마나 되나?"
        )
        stages = payload["think_trace"]["stages"]

        self.assertIn("holding_report_relative_execution", stages)
        self.assertNotIn("holding_reporter_scope", stages)
        self.assertEqual(payload["retrieved_context"], [])
        self.assertFalse(payload["think_trace"]["answerable"])
        self.assertNotIn(self.harness.correct.after_shares, payload["answer"])
        self.assertNotIn(self.harness.wrong.after_shares, payload["answer"])

    def test_comparison_questions_never_receive_the_generic_pair(self) -> None:
        """A comparison keeps its own semantics; B.3 must not reinterpret it.

        A genuine ``company_comparison`` legitimately resolves and keeps both
        companies -- that is the comparison path doing its job, not a leak.
        What matters here is that neither shape is turned into a single-issuer
        current-state request.
        """

        scope = CorpusScope(
            companies={ISSUER: (ISSUER, role.ISSUER_CODE),
                       REPORTER: (REPORTER, role.REPORTER_CODE)},
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        understanding = QueryUnderstanding(scope.company_aliases())
        validator = QueryValidator(corpus_scope=scope)
        for question in (
            f"{REPORTER}와 {ISSUER} 보유비율 비교",
            f"{REPORTER}와 {ISSUER} 중 누가 보유 주식수가 더 많아?",
        ):
            with self.subTest(question=question):
                result = validator.validate(understanding.understand(question))
                self.assertEqual(len(result.plan.companies), 2)
                self.assertEqual(
                    result.plan.evidence.get("comparison_frame"), "cross_company"
                )
                self.assertNotEqual(
                    requested_holding_fields(str(result.plan.raw_query), result.plan),
                    PAIR,
                )


if __name__ == "__main__":
    unittest.main()
