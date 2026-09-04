"""Side-by-side comparison evidence: retrieve per company, interleave, rank nothing.

The amount lane owns ranking, and owns it because 계약금액 is comparable by
construction. What these assert is that the evidence path stays inside its own
boundary: it engages only on a question naming several resolvable companies, it
keeps the period every company was asked about, it lets no company crowd out
another, and it never claims one figure exceeds another.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.reasoning.comparison_evidence import (
    MAX_COMPARISON_COMPANIES,
    EvidenceComparison,
    evidence_comparison,
    evidence_subplan,
    execute_per_company,
    merge_executions,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.retrieval.interfaces import RetrievalResult


def _plan(**changes) -> QueryPlan:
    from dataclasses import replace

    base = QueryPlan(
        query="설비투자",
        raw_query="LG에너지솔루션과 삼성SDI 중 2025년 설비투자가 더 큰 곳은?",
        companies=("LG에너지솔루션", "삼성SDI"),
        corp_codes=("00121", "00122"),
        event_type="facility_investment",
        period=QueryPeriod(year=2025, period_type="reference_year"),
        evidence={"comparison_frame": "cross_company"},
    )
    return replace(base, **changes) if changes else base


@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    chunk: Mapping[str, Any]


@dataclass(frozen=True)
class _Document:
    doc_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Execution:
    documents: Sequence[Any]
    chunks: Sequence[Any]
    results: Sequence[Any]


def _execution(company: str, count: int) -> _Execution:
    chunks = [
        _Chunk(f"{company}:{index}", {"chunk_id": f"{company}:{index}", "corp_code": company})
        for index in range(1, count + 1)
    ]
    results = [
        RetrievalResult(
            chunk_id=chunk.chunk_id,
            doc_id=f"doc_{company}_{index}",
            bm25_score=1.0,
            rank=index,
            metadata_match={},
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    documents = [_Document(f"doc_{company}_{index}") for index in range(1, count + 1)]
    return _Execution(documents=documents, chunks=chunks, results=results)


class PlannerTests(unittest.TestCase):
    def test_two_resolved_companies_with_a_shared_subject_engage(self) -> None:
        decision = evidence_comparison(_plan())
        self.assertTrue(decision.applied)
        self.assertEqual(decision.company_count, 2)
        self.assertEqual(decision.subject_kind, "event_type")

    def test_one_company_is_declined(self) -> None:
        decision = evidence_comparison(
            _plan(companies=("LG에너지솔루션",), corp_codes=("00121",))
        )
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "fewer_than_two_companies")

    def test_companies_without_comparison_intent_are_declined(self) -> None:
        decision = evidence_comparison(_plan(evidence={}))
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "no_comparison_intent")

    def test_a_company_the_corpus_did_not_resolve_is_declined(self) -> None:
        decision = evidence_comparison(_plan(corp_codes=("00121",)))
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "company_not_resolved_in_corpus")

    def test_a_question_without_a_shared_subject_is_declined(self) -> None:
        decision = evidence_comparison(
            _plan(event_type=None, metric=None, years=(), period=QueryPeriod())
        )
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "no_shared_subject")

    def test_more_companies_than_the_cap_are_declined(self) -> None:
        names = tuple(f"C{index}" for index in range(MAX_COMPARISON_COMPANIES + 1))
        codes = tuple(f"{index:05d}" for index in range(MAX_COMPARISON_COMPANIES + 1))
        decision = evidence_comparison(_plan(companies=names, corp_codes=codes))
        self.assertFalse(decision.applied)
        self.assertEqual(decision.decline_reason, "more_than_max_companies")


class SubplanTests(unittest.TestCase):
    def test_each_leg_keeps_the_period_the_question_asked_about(self) -> None:
        plan = _plan()
        decision = evidence_comparison(plan)

        scoped = [evidence_subplan(plan, operand) for operand in decision.companies]

        # The amount lane drops the parent period on purpose; here dropping it is
        # what lets one company answer from 2023 while another answers from 2025.
        for subplan in scoped:
            self.assertEqual(subplan.period.year, 2025)
            self.assertEqual(len(subplan.companies), 1)
            self.assertEqual(len(subplan.corp_codes), 1)
            self.assertIsNone(subplan.comparison)

    def test_the_firewall_signal_survives_scoping(self) -> None:
        plan = _plan()
        decision = evidence_comparison(plan)
        for operand in decision.companies:
            subplan = evidence_subplan(plan, operand)
            self.assertEqual(
                subplan.evidence.get("comparison_frame"), "cross_company"
            )

    def test_one_leg_is_retrieved_per_company(self) -> None:
        plan = _plan()
        decision = evidence_comparison(plan)
        seen: list[tuple[str, ...]] = []

        def execute(subplan):
            seen.append(tuple(subplan.companies))
            return _execution(subplan.companies[0], 2)

        executions = execute_per_company(decision, plan, execute)

        self.assertEqual(seen, [("LG에너지솔루션",), ("삼성SDI",)])
        self.assertEqual(len(executions), 2)

    def test_one_failing_leg_does_not_fail_the_rest(self) -> None:
        plan = _plan()
        decision = evidence_comparison(plan)

        def execute(subplan):
            if subplan.companies[0] == "삼성SDI":
                raise RuntimeError("boom")
            return _execution("LG에너지솔루션", 2)

        executions = execute_per_company(decision, plan, execute)

        self.assertIsNotNone(executions["00121"])
        self.assertIsNone(executions["00122"])


class MergeTests(unittest.TestCase):
    def _merged(self, first: int, second: int):
        plan = _plan()
        decision = evidence_comparison(plan)
        executions = {
            "00121": _execution("LG", first),
            "00122": _execution("SDI", second),
        }
        return decision, merge_executions(plan, decision, executions)

    def test_every_company_lands_inside_the_first_positions(self) -> None:
        # Ten results each: concatenating would put SDI's best at position 11,
        # past the evidence limit, and the answer would cite LG ten times.
        _decision, merged = self._merged(10, 10)
        leading = [result.chunk_id for result in merged.results[:4]]
        self.assertTrue(any(chunk_id.startswith("LG:") for chunk_id in leading))
        self.assertTrue(any(chunk_id.startswith("SDI:") for chunk_id in leading))

    def test_each_company_keeps_its_own_order(self) -> None:
        _decision, merged = self._merged(3, 3)
        lg = [r.chunk_id for r in merged.results if r.chunk_id.startswith("LG:")]
        sdi = [r.chunk_id for r in merged.results if r.chunk_id.startswith("SDI:")]
        self.assertEqual(lg, ["LG:1", "LG:2", "LG:3"])
        self.assertEqual(sdi, ["SDI:1", "SDI:2", "SDI:3"])

    def test_ranks_are_renumbered_into_one_sequence(self) -> None:
        _decision, merged = self._merged(2, 2)
        self.assertEqual(
            [result.rank for result in merged.results], [1, 2, 3, 4]
        )

    def test_an_uneven_split_still_serves_both(self) -> None:
        _decision, merged = self._merged(5, 1)
        self.assertEqual(merged.results[0].chunk_id, "LG:1")
        self.assertEqual(merged.results[1].chunk_id, "SDI:1")

    def test_a_company_that_retrieved_nothing_is_simply_absent(self) -> None:
        plan = _plan()
        decision = evidence_comparison(plan)
        merged = merge_executions(
            plan, decision, {"00121": _execution("LG", 2), "00122": None}
        )
        self.assertEqual(
            [result.chunk_id for result in merged.results], ["LG:1", "LG:2"]
        )

    def test_the_trace_reports_the_fan_out(self) -> None:
        _decision, merged = self._merged(2, 2)
        self.assertEqual(
            merged.routing["comparison_evidence"],
            {
                "applied": True,
                "company_count": 2,
                "subject_kind": "event_type",
                "decline_reason": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
