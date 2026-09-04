"""Side-by-side comparison evidence and its fail-closed ranking gate.

The contract-amount lane owns unconditional ranking because 계약금액 is
comparable by construction. What these assert is that the evidence path stays
inside its own boundary: it engages only on a question naming several resolvable
companies, keeps the period every company was asked about, and lets no company
crowd out another. A narrative ranking is added only for same-kind, same-year,
total-marked amounts.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from app.reasoning.comparison_evidence import (
    MAX_COMPARISON_COMPANIES,
    EvidenceComparison,
    evidence_comparison,
    evidence_subplan,
    execute_per_company,
    merge_executions,
)
from app.reasoning.comparison_ranking import (
    ANNUAL,
    COMPARISON_RANKING_KEY,
    HALF_YEAR,
    apply_conditional_ranking,
    conditional_ranking,
    parse_total_amount,
    ranking_from_outcome,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.retrieval.interfaces import (
    CandidateChunk,
    MetadataMatch,
    RetrievalResult,
)


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


def _amount_execution(
    corp_code: str, rows: Sequence[Mapping[str, Any]]
) -> _Execution:
    candidates = []
    results = []
    for index, row in enumerate(rows, start=1):
        chunk_id = str(row.get("chunk_id") or f"{corp_code}:{index}")
        doc_id = str(row.get("doc_id") or f"doc_{corp_code}_{index}")
        payload = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "corp_code": corp_code,
            "corp_name": "LG에너지솔루션" if corp_code == "00121" else "삼성SDI",
            "report_nm": "사업보고서 (2025.12)",
            "base_year": 2025,
            "content": "2025년 당기 중 총 1조원",
            "source_refs": [{"table_id": f"table_{corp_code}_{index}"}],
            **dict(row),
        }
        candidates.append(
            CandidateChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                chunk=payload,
                metadata_match=MetadataMatch(),
            )
        )
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                bm25_score=1.0 / index,
                rank=index,
                metadata_match={},
            )
        )
    return _Execution(documents=(), chunks=candidates, results=results)


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

    def test_a_reference_year_becomes_a_filterable_one(self) -> None:
        # Metadata filtering reads `years` only when the period is fiscal, so a
        # reference year alone leaves every year of that company retrievable.
        plan = _plan()
        self.assertFalse(plan.period.is_fiscal)
        self.assertEqual(plan.years, ())

        subplan = evidence_subplan(plan, evidence_comparison(plan).companies[0])

        self.assertEqual(subplan.years, (2025,))
        self.assertTrue(subplan.period.is_fiscal)
        self.assertEqual(subplan.backend_filters()["year"], [2025])

    def test_a_year_the_question_already_named_is_left_alone(self) -> None:
        plan = _plan(
            years=(2024,),
            period=QueryPeriod(year=2024, period_type="fiscal_year"),
        )
        subplan = evidence_subplan(plan, evidence_comparison(plan).companies[0])
        self.assertEqual(subplan.years, (2024,))

    def test_a_question_without_a_year_stays_unscoped(self) -> None:
        plan = _plan(period=QueryPeriod(), years=(), metric="매출액")
        subplan = evidence_subplan(plan, evidence_comparison(plan).companies[0])
        self.assertEqual(subplan.years, ())

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


class AmountParsingTests(unittest.TestCase):
    def test_decimal_trillion_is_normalized_to_won(self) -> None:
        parsed = parse_total_amount("2025년 당기 중 총 10.5조원")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.value, 10_500_000_000_000)
        self.assertEqual(parsed.text, "10.5조원")

    def test_mixed_trillion_and_hundred_million_units_are_added(self) -> None:
        parsed = parse_total_amount("2025년 상반기 누적 1조 8,430억원")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.value, 1_843_000_000_000)

    def test_compound_thousand_units_are_unambiguous(self) -> None:
        self.assertEqual(parse_total_amount("총 2천억원").value, 200_000_000_000)
        self.assertEqual(parse_total_amount("누적 3만 5천원").value, 35_000)

    def test_unmarked_first_amount_is_not_mistaken_for_a_total(self) -> None:
        text = (
            "당사는 2025년 상반기 중 1조 8,430억원을 사용하였으며, "
            "각 부문별 투자금액은 1조 7,994억원, 436억원입니다."
        )
        self.assertIsNone(parse_total_amount(text))

    def test_several_marked_amounts_in_one_chunk_are_declined(self) -> None:
        self.assertIsNone(parse_total_amount("총 1조원이며 누적 2조원입니다."))

    def test_a_marker_only_applies_to_the_next_amount(self) -> None:
        parsed = parse_total_amount(
            "누적 1조 8,430억원이며 부문별로 1조 7,994억원과 436억원입니다."
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.value, 1_843_000_000_000)


class ConditionalRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _plan()
        self.comparison = evidence_comparison(self.plan)

    def _executions(self) -> dict[str, _Execution]:
        return {
            "00121": _amount_execution(
                "00121",
                (
                    {
                        "chunk_id": "lg-half",
                        "report_nm": "반기보고서 (2025.06)",
                        "content": "2025년 당기 중 총 5.8조원",
                    },
                    {
                        "chunk_id": "lg-annual",
                        "content": "2025년 당기 중 총 10.5조원",
                    },
                ),
            ),
            "00122": _amount_execution(
                "00122",
                (
                    {
                        "chunk_id": "sdi-half",
                        "report_nm": "반기보고서 (2025.06)",
                        "content": "2025년 상반기 누적 1조 8,430억원",
                    },
                    {
                        "chunk_id": "sdi-annual",
                        "content": "2025년 당기 중 총 3조원",
                    },
                ),
            ),
        }

    def test_a_year_only_question_selects_annual_reports_for_every_company(self) -> None:
        ranking = conditional_ranking(
            self.comparison, self.plan, self._executions()
        )
        self.assertIsNotNone(ranking)
        self.assertEqual(ranking.report_kind, ANNUAL)
        self.assertEqual(ranking.base_year, 2025)
        self.assertEqual(
            [operand.chunk_id for operand in ranking.operands],
            ["lg-annual", "sdi-annual"],
        )
        self.assertEqual(
            [operand.value for operand in ranking.operands],
            [10_500_000_000_000, 3_000_000_000_000],
        )

    def test_a_missing_annual_operand_does_not_fall_back_to_common_half_years(self) -> None:
        executions = self._executions()
        executions["00122"] = _amount_execution(
            "00122",
            (
                {
                    "chunk_id": "sdi-half",
                    "report_nm": "반기보고서 (2025.06)",
                    "content": "2025년 상반기 누적 1조 8,430억원",
                },
            ),
        )
        self.assertIsNone(
            conditional_ranking(self.comparison, self.plan, executions)
        )

    def test_a_company_without_a_total_marker_declines_the_whole_ranking(self) -> None:
        executions = self._executions()
        executions["00122"] = _amount_execution(
            "00122",
            (
                {
                    "content": (
                        "2025년 중 1조 8,430억원을 사용했으며 부문별 금액은 "
                        "1조 7,994억원, 436억원입니다."
                    )
                },
            ),
        )
        self.assertIsNone(
            conditional_ranking(self.comparison, self.plan, executions)
        )

    def test_a_lower_clean_amount_cannot_replace_the_first_same_kind_evidence(self) -> None:
        executions = self._executions()
        executions["00122"] = _amount_execution(
            "00122",
            (
                {"chunk_id": "unsafe-first", "content": "부문별 3조원"},
                {"chunk_id": "clean-later", "content": "총 4조원"},
            ),
        )
        self.assertIsNone(
            conditional_ranking(self.comparison, self.plan, executions)
        )

    def test_different_base_years_decline_the_whole_ranking(self) -> None:
        executions = self._executions()
        executions["00122"] = _amount_execution(
            "00122", ({"base_year": 2024, "content": "누적 3조원"},)
        )
        self.assertIsNone(
            conditional_ranking(self.comparison, self.plan, executions)
        )

    def test_a_prior_year_amount_inside_the_selected_report_is_declined(self) -> None:
        executions = self._executions()
        executions["00122"] = _amount_execution(
            "00122", ({"content": "2024년 누적 3조원"},)
        )
        self.assertIsNone(
            conditional_ranking(self.comparison, self.plan, executions)
        )

    def test_a_chunk_owned_by_another_company_cannot_supply_an_operand(self) -> None:
        executions = self._executions()
        wrong = _amount_execution("00999", ({"content": "총 3조원"},))
        executions["00122"] = wrong
        self.assertIsNone(
            conditional_ranking(self.comparison, self.plan, executions)
        )

    def test_an_explicit_half_year_question_selects_half_year_reports(self) -> None:
        plan = replace(
            self.plan,
            raw_query=(
                "LG에너지솔루션과 삼성SDI 중 2025년 상반기 설비투자가 더 큰 곳은?"
            ),
            period=QueryPeriod(year=2025, quarter=2, period_type="fiscal_quarter"),
            years=(2025,),
        )
        ranking = conditional_ranking(
            evidence_comparison(plan), plan, self._executions()
        )
        self.assertIsNotNone(ranking)
        self.assertEqual(ranking.report_kind, HALF_YEAR)
        self.assertEqual(
            [operand.chunk_id for operand in ranking.operands],
            ["lg-half", "sdi-half"],
        )

    def test_a_smaller_amount_question_orders_ascending(self) -> None:
        plan = replace(
            self.plan,
            raw_query=(
                "LG에너지솔루션과 삼성SDI 중 2025년 설비투자 규모가 더 작은 곳은?"
            ),
        )
        ranking = conditional_ranking(
            evidence_comparison(plan), plan, self._executions()
        )
        self.assertIsNotNone(ranking)
        self.assertEqual(ranking.direction, "ascending")
        self.assertEqual(
            [operand.chunk_id for operand in ranking.operands],
            ["sdi-annual", "lg-annual"],
        )

    def test_a_non_amount_comparison_never_ranks_incidental_money(self) -> None:
        plan = replace(
            self.plan,
            raw_query="LG에너지솔루션과 삼성SDI 중 2025년에 더 먼저 공시한 곳은?",
        )
        self.assertIsNone(
            conditional_ranking(
                evidence_comparison(plan), plan, self._executions()
            )
        )


class RankingApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _plan()
        self.comparison = evidence_comparison(self.plan)
        self.executions = ConditionalRankingTests()._executions()
        self.merged = merge_executions(
            self.plan, self.comparison, self.executions
        )

    def test_decline_returns_the_exact_stage_one_execution(self) -> None:
        self.assertIs(apply_conditional_ranking(self.merged, None), self.merged)

    def test_success_promotes_every_cited_operand_and_attaches_the_outcome(self) -> None:
        ranking = conditional_ranking(
            self.comparison, self.plan, self.executions
        )
        applied = apply_conditional_ranking(self.merged, ranking)

        self.assertEqual(
            [result.chunk_id for result in applied.results[:2]],
            ["lg-annual", "sdi-annual"],
        )
        rebuilt = ranking_from_outcome(
            applied.routing[COMPARISON_RANKING_KEY]
        )
        self.assertIsNotNone(rebuilt)
        self.assertEqual(
            [operand.chunk_id for operand in rebuilt.operands],
            ["lg-annual", "sdi-annual"],
        )

    def test_the_composed_ranking_leads_and_cites_both_companies(self) -> None:
        from app.agent.orchestrator import _compose_conditional_comparison_ranking
        from app.generation.answer_generator import CitationAwareAnswerGenerator
        from app.reasoning.evidence_builder import EvidenceBuilder

        ranking = conditional_ranking(
            self.comparison, self.plan, self.executions
        )
        applied = apply_conditional_ranking(self.merged, ranking)
        evidence = EvidenceBuilder().build(applied, question=self.plan.raw_query)
        draft = _compose_conditional_comparison_ranking(
            evidence, ranking, task_type="general_evidence"
        )
        self.assertIsNotNone(draft)
        generated = CitationAwareAnswerGenerator().generate(draft)

        self.assertTrue(generated.answerable)
        self.assertTrue(
            generated.answer_text.startswith("기업별 조건부 금액 비교")
        )
        self.assertIn(
            "LG에너지솔루션의 합계 금액은 10.5조원으로 삼성SDI(3조원)보다 큽니다.",
            generated.answer_text,
        )
        self.assertIn("[1] [2]", generated.answer_text)
        self.assertEqual(
            [citation.chunk_id for citation in generated.citations[:2]],
            ["lg-annual", "sdi-annual"],
        )


class RankingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _plan()
        self.comparison = evidence_comparison(self.plan)
        self.executions = ConditionalRankingTests()._executions()

    def _pipeline(self, executions: Mapping[str, _Execution]):
        from app.api.pipeline import AnswerPipeline
        from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer
        from app.reasoning.query_validation import (
            QueryState,
            QueryValidationResult,
        )

        plan = self.plan

        class Understanding:
            def understand(self, question, *, top_k):
                del question, top_k
                return plan

        class Validator:
            def validate(self, candidate):
                return QueryValidationResult(
                    state=QueryState.AMBIGUOUS,
                    plan=candidate,
                    slots={},
                    required_slots=(),
                    issues=("several_companies_need_separate_scopes",),
                )

        return AnswerPipeline(
            understanding=Understanding(),
            executor=type(
                "Executor",
                (),
                {
                    "execute": lambda _self, subplan: executions[
                        subplan.corp_codes[0]
                    ]
                },
            )(),
            query_validator=Validator(),
            verbalizer=HcxVerbalizer(HcxSettings(enabled=False)),
        )

    def test_the_serving_path_states_the_verified_winner(self) -> None:
        payload = self._pipeline(self.executions).answer(
            "OF3", self.plan.raw_query
        )

        self.assertIn(
            "LG에너지솔루션의 합계 금액은 10.5조원으로 삼성SDI(3조원)보다 큽니다.",
            payload["answer"],
        )
        self.assertIn("[1] [2]", payload["answer"])
        self.assertIn(
            "conditional_comparison_ranking", payload["think_trace"]["stages"]
        )
        self.assertEqual(
            [row["chunk_id"] for row in payload["retrieved_context"][:2]],
            ["lg-annual", "sdi-annual"],
        )

    def test_a_failed_gate_keeps_the_side_by_side_answer_without_a_winner(self) -> None:
        executions = dict(self.executions)
        executions["00122"] = _amount_execution(
            "00122",
            (
                {
                    "content": (
                        "2025년 중 1조 8,430억원을 사용했으며 부문별 금액은 "
                        "1조 7,994억원, 436억원입니다."
                    )
                },
            ),
        )
        payload = self._pipeline(executions).answer("OF3", self.plan.raw_query)

        self.assertNotIn("기업별 조건부 금액 비교", payload["answer"])
        self.assertNotIn(
            "conditional_comparison_ranking", payload["think_trace"]["stages"]
        )
        self.assertIn("총 5.8조원", payload["answer"])
        self.assertIn("1조 8,430억원", payload["answer"])
        self.assertEqual(
            {row["corp_name"] for row in payload["retrieved_context"][:2]},
            {"LG에너지솔루션", "삼성SDI"},
        )


if __name__ == "__main__":
    unittest.main()
