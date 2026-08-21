from __future__ import annotations

import unittest

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.generation.compact_claim import (
    MAX_CLAIM_EVENTS,
    build_compact_claim,
)
from app.reasoning.query_plan import QueryPlan
from tests.test_agent_end_to_end_smoke import _execution
from tests.test_evidence_builder import _candidate, _holding_pair


def _holding_plan(query: str, *, requested: list[str]) -> QueryPlan:
    return QueryPlan(
        query=query,
        task_type="holding_change",
        metric="holding_shares",
        reporter="국민연금기금",
        disclosure_route=("holding",),
        evidence={"requested_holding_fields": requested},
    )


def _run(plan: QueryPlan, *pairs):
    """Run the production reasoning path and return what the claim builder sees."""

    result = AgentOrchestrator().run(plan.raw_query, plan, _execution(plan, *pairs))
    generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
    claim = build_compact_claim(
        result.answer_draft,
        result.resolution,
        task_type=result.task_decision.task_type,
    )
    return result, generated, claim


def _holding_case(*, events: int = 1, requested: list[str] | None = None):
    fields = requested or ["reference_date", "after_shares"]
    pairs = [
        _holding_pair(
            f"h2{index}:ch",
            f"h2{index}",
            rank=index + 1,
            date=f"202{3 + index}-06-30",
            projection_type="holding_report",
            table_id=f"t2{index}",
        )
        for index in range(events)
    ]
    plan = _holding_plan("효성중공업 국민연금기금 변동일 변동후 주식수", requested=fields)
    return _run(plan, *pairs)


class HoldingClaimTests(unittest.TestCase):
    """HX-style questions: the resolver already produced structured fields."""

    def test_hx02_single_event_builds_a_claim(self) -> None:
        _, _, claim = _holding_case(events=1)

        self.assertIsNotNone(claim)
        self.assertEqual(
            [field.name for field in claim.fields],
            ["reference_date", "after_shares"],
        )

    def test_hx07_claim_carries_verbatim_values(self) -> None:
        _, _, claim = _holding_case(events=1)

        values = {field.name: field.value for field in claim.fields}
        self.assertEqual(values["reference_date"], "2023-06-30")
        self.assertEqual(values["after_shares"], "1,000주")

    def test_hx15_claim_is_short_enough_to_verbalize(self) -> None:
        _, generated, claim = _holding_case(events=1)

        self.assertLess(len(claim.deterministic_text), 120)
        self.assertLess(len(claim.deterministic_text), len(generated.answer_text))

    def test_claim_text_contains_every_value_and_marker(self) -> None:
        _, _, claim = _holding_case(events=1)

        for field in claim.fields:
            self.assertIn(field.value, claim.deterministic_text)
            self.assertIn(field.marker, claim.deterministic_text)

    def test_claim_names_the_company_and_reporter(self) -> None:
        _, _, claim = _holding_case(events=1)

        self.assertEqual(claim.reporter, "국민연금기금")
        self.assertIn(claim.reporter, claim.deterministic_text)
        self.assertEqual(set(claim.required_terms), {claim.company, claim.reporter})

    def test_every_field_is_linked_to_a_cited_chunk(self) -> None:
        _, _, claim = _holding_case(events=1)

        cited = {citation.marker: citation for citation in claim.citations}
        for field in claim.fields:
            self.assertIn(field.marker, cited)
            self.assertEqual(cited[field.marker].chunk_id, field.chunk_id)

    def test_citations_carry_field_provenance(self) -> None:
        _, _, claim = _holding_case(events=1)

        for citation in claim.citations:
            self.assertTrue(citation.chunk_id)
            self.assertTrue(citation.doc_id)
            self.assertTrue(citation.source_refs)

    def test_markers_are_numbered_from_one_in_order_of_use(self) -> None:
        _, _, claim = _holding_case(events=2)

        markers = [citation.marker for citation in claim.citations]
        self.assertEqual(markers, [f"[{index + 1}]" for index in range(len(markers))])

    def test_ratio_field_keeps_its_unit(self) -> None:
        _, _, claim = _holding_case(requested=["after_ratio"])

        self.assertEqual(claim.fields[0].value, "7.0%")

    def test_claim_values_are_never_recomputed(self) -> None:
        """The rendered value is the resolver's raw string, not the parsed number."""

        _, _, claim = _holding_case(requested=["after_shares"])

        self.assertEqual(claim.fields[0].value, "1,000주")
        self.assertNotIn("1000주", claim.deterministic_text)


class HoldingClaimSkipTests(unittest.TestCase):
    def test_too_many_events_are_skipped_rather_than_chosen_between(self) -> None:
        _, _, claim = _holding_case(events=MAX_CLAIM_EVENTS + 1)

        self.assertIsNone(claim)

    def test_a_field_the_resolver_did_not_supply_skips_the_claim(self) -> None:
        _, _, claim = _holding_case(requested=["reference_date", "before_shares"])

        self.assertIsNone(claim)

    def test_no_requested_fields_skips_the_claim(self) -> None:
        """The resolver infers fields from the question, so this guard is probed
        directly rather than through a plan that cannot express "nothing asked"."""

        result, _, _ = _holding_case(events=1)
        resolution = dict(result.resolution.to_dict())
        resolution["requested_fields"] = []

        claim = build_compact_claim(
            result.answer_draft, resolution, task_type="holding_event"
        )

        self.assertIsNone(claim)

    def test_unresolved_fields_skip_the_claim(self) -> None:
        result, _, _ = _holding_case(events=1)
        resolution = dict(result.resolution.to_dict())
        resolution["unresolved_fields"] = ["before_shares"]

        claim = build_compact_claim(
            result.answer_draft, resolution, task_type="holding_event"
        )

        self.assertIsNone(claim)

    def test_an_unanswerable_draft_skips_the_claim(self) -> None:
        result, _, _ = _holding_case(events=1)
        draft = result.answer_draft.__class__(
            question=result.answer_draft.question,
            task_type=result.answer_draft.task_type,
            answer_sections=result.answer_draft.answer_sections,
            evidence_references=result.answer_draft.evidence_references,
            citations=result.answer_draft.citations,
            ambiguity=result.answer_draft.ambiguity,
            warnings=result.answer_draft.warnings,
            confidence=result.answer_draft.confidence,
            answerable=False,
        )

        claim = build_compact_claim(
            draft, result.resolution, task_type="holding_event"
        )

        self.assertIsNone(claim)

    def test_wrong_task_type_skips_the_claim(self) -> None:
        result, _, _ = _holding_case(events=1)

        self.assertIsNone(
            build_compact_claim(
                result.answer_draft, result.resolution, task_type="periodic_fact"
            )
        )

    def test_missing_draft_or_resolution_skips_the_claim(self) -> None:
        result, _, _ = _holding_case(events=1)

        self.assertIsNone(
            build_compact_claim(None, result.resolution, task_type="holding_event")
        )
        self.assertIsNone(
            build_compact_claim(result.answer_draft, None, task_type="holding_event")
        )


class PeriodicAndGeneralSkipTests(unittest.TestCase):
    """Only holding is supported; everything else keeps the deterministic answer."""

    def test_p07_periodic_question_has_no_compact_claim(self) -> None:
        text = "DX 부문은 TV, 모니터, 냉장고를 생산합니다."
        older = _candidate(
            "p23:ch", "p23", rank=1, doc_group="periodic", content=text,
            section="주요 제품 및 서비스", fiscal_year=2023, period_type="fiscal_year",
            source_refs=[{"table_id": "t23", "row_start": 1, "row_end": 1}],
        )
        newer = _candidate(
            "p24:ch", "p24", rank=2, doc_group="periodic", content=text,
            section="주요 제품 및 서비스", fiscal_year=2024, period_type="fiscal_year",
            source_refs=[{"table_id": "t24", "row_start": 2, "row_end": 2}],
        )
        plan = QueryPlan(
            query="두산퓨얼셀 2023년 1분기 연료전지 주기기 매출액",
            task_type="disclosure_lookup",
            disclosure_route=("periodic",),
            evidence={"periodic_intent": "business_product"},
        )

        result, generated, claim = _run(plan, older, newer)

        self.assertEqual(result.task_decision.task_type, "periodic_fact")
        self.assertIsNone(claim)
        self.assertTrue(generated.answer_text)

    def test_corporate_event_has_no_compact_claim(self) -> None:
        pair = _candidate(
            "m23:ch", "m23", rank=1, doc_group="major",
            content="유상증자 결정. 신주 1,000,000주.",
            section="주요사항보고서", report_nm="유상증자결정",
        )
        plan = QueryPlan(
            query="삼성전자 유상증자 공시 내용",
            task_type="corporate_event",
            event_type="capital_increase",
            disclosure_route=("major",),
        )

        result, generated, claim = _run(plan, pair)

        self.assertEqual(result.task_decision.task_type, "corporate_event")
        self.assertIsNone(claim)
        self.assertTrue(generated.answer_text)

    def test_general_evidence_has_no_compact_claim(self) -> None:
        pair = _candidate(
            "x23:ch", "x23", rank=1, doc_group="major",
            content="자기주식 취득 결정 공시입니다.",
            section="주요사항보고서", report_nm="자기주식취득결정",
        )
        plan = QueryPlan(
            query="삼성전자 자기주식 취득 공시",
            task_type="disclosure_lookup",
            disclosure_route=("major",),
        )

        result, generated, claim = _run(plan, pair)

        self.assertEqual(result.task_decision.task_type, "general_evidence")
        self.assertIsNone(claim)
        self.assertTrue(generated.answer_text)


if __name__ == "__main__":
    unittest.main()
