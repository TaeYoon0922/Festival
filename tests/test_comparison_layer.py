"""Multi-company comparison: fan out per company, combine deterministically.

The layer must stay invisible to every question it declines, must ask each
named company its own scoped question, and must never present a ranking the
evidence does not carry.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from app.generation.answer_generator import (
    GeneratedAnswer,
    GeneratedCitation,
    GeneratedSection,
)
from app.reasoning.comparison import ComparisonPlanner
from app.reasoning.comparison.execution import (
    ComparisonExecution,
    ComparisonExecutor,
    ComparisonLeg,
)
from app.reasoning.comparison.render import render_comparison
from app.reasoning.query_plan import QueryPlan


def _plan(**changes) -> QueryPlan:
    base = QueryPlan(
        query="설비투자",
        raw_query="LG에너지솔루션과 삼성SDI 중 2025년 설비투자가 더 큰 곳은?",
        companies=("LG에너지솔루션", "삼성SDI"),
        corp_codes=("00121", "00122"),
        event_type="facility_investment",
        evidence={"comparison_frame": "cross_company"},
    )
    return replace(base, **changes) if changes else base


def _answer(company: str, facts: tuple[str, ...], *, answerable: bool = True):
    citations, sections = [], []
    for index, text in enumerate(facts, start=1):
        marker = f"[{index}]"
        citations.append(
            GeneratedCitation(
                citation_id=marker,
                chunk_id=f"{company}:ch_{index}",
                doc_id=f"doc_{company}",
                source_refs=(),
                section="II. 사업의 내용",
                evidence_type="text",
            )
        )
        sections.append(
            GeneratedSection(
                title=f"Periodic fact {index}",
                content=f"{text} {marker}",
                citations=(marker,),
            )
        )
    return GeneratedAnswer(
        question="q",
        answer_text="(leg)",
        citations=tuple(citations),
        sections=tuple(sections),
        warnings=(),
        confidence={"level": "높음"},
        answerable=answerable,
    )


class PlannerTests(unittest.TestCase):
    def test_two_named_companies_with_a_shared_subject_engage(self) -> None:
        decision = ComparisonPlanner().plan("q", _plan())
        self.assertTrue(decision.applied)
        self.assertEqual(decision.company_count, 2)
        self.assertEqual(decision.subject_kind, "event_type")

    def test_a_single_company_question_is_declined(self) -> None:
        decision = ComparisonPlanner().plan(
            "q", _plan(companies=("LG에너지솔루션",), corp_codes=("00121",))
        )
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "fewer_than_two_companies")

    def test_companies_without_comparison_wording_are_declined(self) -> None:
        decision = ComparisonPlanner().plan("q", _plan(evidence={}))
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "no_comparison_intent")

    def test_a_question_without_a_shared_subject_is_declined(self) -> None:
        decision = ComparisonPlanner().plan(
            "q", _plan(event_type=None, metric=None, years=())
        )
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "no_shared_subject")

    def test_an_unresolved_company_is_declined(self) -> None:
        decision = ComparisonPlanner().plan("q", _plan(corp_codes=("00121",)))
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "company_not_resolved_in_corpus")

    def test_more_companies_than_the_cap_are_declined(self) -> None:
        decision = ComparisonPlanner(max_companies=2).plan(
            "q", _plan(companies=("A", "B", "C"), corp_codes=("1", "2", "3"))
        )
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "more_than_max_companies")


class _RecordingExecutor:
    def __init__(self) -> None:
        self.plans = []

    def execute(self, plan):
        self.plans.append(plan)
        return object()


class _StubOrchestrator:
    def run(self, question, plan, execution):
        del question, execution
        return type("R", (), {"answer_draft": plan.companies[0]})()


class _StubGenerator:
    def generate(self, draft):
        return _answer(str(draft), (f"{draft} 2025년 시설투자",))


def _executor(retrieval=None) -> ComparisonExecutor:
    return ComparisonExecutor(
        retrieval_executor=retrieval or _RecordingExecutor(),
        orchestrator=_StubOrchestrator(),
        generator=_StubGenerator(),
    )


class ExecutorTests(unittest.TestCase):
    def test_each_company_gets_its_own_scoped_plan(self) -> None:
        retrieval = _RecordingExecutor()
        plan = _plan()
        execution = _executor(retrieval).execute(
            "q", plan, ComparisonPlanner().plan("q", plan)
        )

        self.assertTrue(execution.applied)
        self.assertEqual(len(execution.legs), 2)
        self.assertEqual(
            [tuple(scoped.companies) for scoped in retrieval.plans],
            [("LG에너지솔루션",), ("삼성SDI",)],
        )
        for scoped in retrieval.plans:
            self.assertEqual(len(scoped.corp_codes), 1)
            self.assertIsNone(scoped.comparison)

    def test_the_firewall_signal_survives_scoping(self) -> None:
        retrieval = _RecordingExecutor()
        plan = _plan()
        _executor(retrieval).execute("q", plan, ComparisonPlanner().plan("q", plan))
        for scoped in retrieval.plans:
            self.assertEqual(scoped.evidence.get("comparison_frame"), "cross_company")

    def test_one_failing_leg_does_not_fail_the_rest(self) -> None:
        class _Flaky(_RecordingExecutor):
            def execute(self, plan):
                super().execute(plan)
                if plan.companies[0] == "삼성SDI":
                    raise RuntimeError("boom")
                return object()

        plan = _plan()
        execution = _executor(_Flaky()).execute(
            "q", plan, ComparisonPlanner().plan("q", plan)
        )

        self.assertTrue(execution.applied)
        self.assertEqual([leg.completed for leg in execution.legs], [True, False])
        self.assertEqual(execution.legs[1].failure, "RuntimeError")

    def test_a_declined_plan_executes_nothing(self) -> None:
        retrieval = _RecordingExecutor()
        declined = ComparisonPlanner().plan("q", _plan(evidence={}))
        execution = _executor(retrieval).execute("q", _plan(), declined)
        self.assertFalse(execution.applied)
        self.assertEqual(retrieval.plans, [])


def _execution(*legs) -> ComparisonExecution:
    return ComparisonExecution(
        plan=ComparisonPlanner().plan("q", _plan()), legs=tuple(legs)
    )


class RenderTests(unittest.TestCase):
    def test_every_fact_names_the_company_it_came_from(self) -> None:
        rendered = render_comparison(
            "q",
            _execution(
                ComparisonLeg("LG에너지솔루션", "1", generated=_answer("LG", ("1.6조원",))),
                ComparisonLeg("삼성SDI", "2", generated=_answer("SDI", ("1조 8,430억원",))),
            ),
        )
        self.assertIn("[LG에너지솔루션]", rendered.answer_text)
        self.assertIn("[삼성SDI]", rendered.answer_text)

    def test_citations_are_renumbered_into_one_sequence(self) -> None:
        rendered = render_comparison(
            "q",
            _execution(
                ComparisonLeg("A", "1", generated=_answer("A", ("가",))),
                ComparisonLeg("B", "2", generated=_answer("B", ("나", "다"))),
            ),
        )
        self.assertEqual(
            [citation.citation_id for citation in rendered.citations],
            ["[1]", "[2]", "[3]"],
        )
        # The second leg's own markers must have moved, not been reused.
        self.assertIn("나 [2]", rendered.answer_text)
        self.assertIn("다 [3]", rendered.answer_text)

    def test_a_company_without_evidence_is_named_not_dropped(self) -> None:
        rendered = render_comparison(
            "q",
            _execution(
                ComparisonLeg("A", "1", generated=_answer("A", ("가",))),
                ComparisonLeg("B", "2", generated=_answer("B", (), answerable=False)),
            ),
        )
        self.assertIn("확인되지 않은 기업", rendered.answer_text)
        self.assertIn("B", rendered.answer_text)

    def test_no_ranking_is_claimed(self) -> None:
        rendered = render_comparison(
            "q",
            _execution(
                ComparisonLeg("A", "1", generated=_answer("A", ("가",))),
                ComparisonLeg("B", "2", generated=_answer("B", ("나",))),
            ),
        )
        self.assertIn("우열은 판단하지 않았습니다", rendered.answer_text)

    def test_nothing_renders_when_no_leg_is_answerable(self) -> None:
        rendered = render_comparison(
            "q",
            _execution(
                ComparisonLeg("A", "1", generated=_answer("A", (), answerable=False))
            ),
        )
        self.assertIsNone(rendered)


if __name__ == "__main__":
    unittest.main()
