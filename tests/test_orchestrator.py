import copy
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import generate_answer
from app.reasoning.answer_composer import AnswerDraft
from app.reasoning.holding_event_resolver import HoldingResolution
from app.reasoning.periodic_fact_resolver import PeriodicFactResolution
from app.reasoning.query_plan import QueryPlan
from tests.test_evidence_builder import _candidate, _holding_pair


def _execution(plan, *pairs):
    return SimpleNamespace(
        plan=plan,
        chunks=[pair[0] for pair in pairs],
        results=[pair[1] for pair in pairs],
    )


def _input_snapshot(plan, execution):
    return (
        copy.deepcopy(plan.to_dict()),
        copy.deepcopy([candidate.chunk for candidate in execution.chunks]),
        copy.deepcopy([result.to_dict() for result in execution.results]),
    )


class AgentOrchestratorTests(unittest.TestCase):
    def test_holding_pipeline_order_and_ambiguity_are_preserved(self):
        first = _holding_pair(
            "h23:ch_report",
            "h23",
            rank=1,
            date="2023-06-30",
            projection_type="holding_report",
            table_id="t23",
        )
        second = _holding_pair(
            "h24:ch_report",
            "h24",
            rank=2,
            date="2024-06-30",
            projection_type="holding_report",
            table_id="t24",
        )
        plan = QueryPlan(
            query="국민연금기금 변동일 변동후 주식수",
            task_type="holding_change",
            metric="holding_shares",
            reporter="국민연금기금",
            disclosure_route=("holding",),
            evidence={
                "requested_holding_fields": ["reference_date", "after_shares"]
            },
        )
        execution = _execution(plan, first, second)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        self.assertEqual(
            result.execution_trace,
            (
                "task_router",
                "evidence_builder",
                "holding_event_resolver",
                "answer_composer",
            ),
        )
        self.assertIsInstance(result.resolution, HoldingResolution)
        self.assertIsInstance(result.answer_draft, AnswerDraft)
        self.assertEqual(
            result.evidence_set.retrieval_order,
            ("h23:ch_report", "h24:ch_report"),
        )
        self.assertTrue(result.resolution.temporal_ambiguity)
        self.assertTrue(result.answer_draft.ambiguity["temporal_ambiguity"])
        events = result.answer_draft.answer_sections[0].content["events"]
        self.assertEqual(
            [event["reference_date"] for event in events],
            ["2023-06-30", "2024-06-30"],
        )
        with self.assertRaises(FrozenInstanceError):
            result.question = "변경"

    def test_periodic_pipeline_preserves_repeated_periods_and_provenance(self):
        text = "동일한 주요 제품 및 서비스의 구체적인 사업 설명입니다."
        older = _candidate(
            "p23:ch_fact",
            "p23",
            rank=1,
            doc_group="periodic",
            content=text,
            section="주요 제품 및 서비스",
            fiscal_year=2023,
            period_type="fiscal_year",
            source_refs=[{"table_id": "t23", "row_start": 1, "row_end": 1}],
        )
        newer = _candidate(
            "p24:ch_fact",
            "p24",
            rank=2,
            doc_group="periodic",
            content=text,
            section="주요 제품 및 서비스",
            fiscal_year=2024,
            period_type="fiscal_year",
            source_refs=[{"table_id": "t24", "row_start": 2, "row_end": 2}],
        )
        plan = QueryPlan(
            query="주요 제품 사업 내용",
            task_type="disclosure_lookup",
            disclosure_route=("periodic",),
            evidence={"periodic_intent": "business_product"},
        )
        execution = _execution(plan, older, newer)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        self.assertEqual(
            result.execution_trace,
            (
                "task_router",
                "evidence_builder",
                "periodic_fact_resolver",
                "periodic_evidence_selector",
                "answer_composer",
            ),
        )
        self.assertIsInstance(result.resolution, PeriodicFactResolution)
        fact = result.resolution.facts[0]
        self.assertTrue(fact.repeated_across_periods)
        self.assertEqual(
            [period["fiscal_year"] for period in fact.reporting_periods],
            [2023, 2024],
        )
        self.assertFalse(result.answer_draft.ambiguity["latest_period_selected"])
        citations = {citation.chunk_id: citation for citation in result.answer_draft.citations}
        self.assertEqual(citations["p23:ch_fact"].source_refs[0]["table_id"], "t23")
        self.assertEqual(citations["p24:ch_fact"].source_refs[0]["table_id"], "t24")

    def test_periodic_selector_keeps_only_question_and_period_aligned_evidence(self):
        correct = _candidate(
            "p23q1:ch_sales",
            "p23q1",
            rank=1,
            doc_group="periodic",
            content="연료전지 주기기 매출액 23,848백만원",
            section="매출 및 수주상황",
            fiscal_year=2023,
            quarter=1,
            period_type="fiscal_quarter",
            report_nm="분기보고서 (2023.03)",
            source_refs=[{"table_id": "sales", "row_start": 2, "row_end": 2}],
        )
        unrelated = _candidate(
            "p23q1:ch_inventory",
            "p23q1",
            rank=2,
            doc_group="periodic",
            content="재고자산과 원재료에 관한 표",
            section="재고자산",
            fiscal_year=2023,
            quarter=1,
            period_type="fiscal_quarter",
            source_refs=[{"table_id": "inventory", "row_start": 1, "row_end": 4}],
        )
        wrong_period = _candidate(
            "p23q2:ch_sales",
            "p23q2",
            rank=3,
            doc_group="periodic",
            content="연료전지 주기기 매출액 48,000백만원",
            section="매출 및 수주상황",
            fiscal_year=2023,
            quarter=2,
            period_type="fiscal_quarter",
            source_refs=[{"table_id": "sales-q2", "row_start": 2, "row_end": 2}],
        )
        wrong_subject = _candidate(
            "p23q1:ch_total_sales",
            "p23q1",
            rank=4,
            doc_group="periodic",
            content="회사 전체 매출액 99,999백만원",
            section="요약재무정보",
            fiscal_year=2023,
            quarter=1,
            period_type="fiscal_quarter",
            source_refs=[{"table_id": "total-sales", "row_start": 1, "row_end": 1}],
        )
        plan = QueryPlan(
            query="연료전지 주기기 매출액",
            raw_query="두산퓨얼셀 2023년 1분기 연료전지 주기기 매출액",
            company="두산퓨얼셀",
            task_type="financial_metric",
            metric="매출액",
            period=(2023, 3),
            disclosure_route=("periodic",),
        )
        execution = _execution(plan, correct, unrelated, wrong_period, wrong_subject)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        self.assertEqual(len(result.resolution.facts), 4)
        self.assertEqual(
            result.answer_draft.evidence_references,
            ("p23q1:ch_sales",),
        )
        self.assertNotIn(
            "p23q1:ch_inventory", result.answer_draft.evidence_references
        )
        self.assertNotIn(
            "p23q2:ch_sales", result.answer_draft.evidence_references
        )
        self.assertEqual(
            result.evidence_set.retrieval_order,
            (
                "p23q1:ch_sales",
                "p23q1:ch_inventory",
                "p23q2:ch_sales",
                "p23q1:ch_total_sales",
            ),
        )
        generated = generate_answer(result.answer_draft)
        fact_section = generated.sections[0]
        self.assertTrue(generated.answerable)
        self.assertIn("연료전지 주기기 매출액 23,848백만원 [1]", fact_section.content)
        self.assertNotIn("재고자산", generated.answer_text)
        self.assertNotIn("48,000", generated.answer_text)
        self.assertNotIn("99,999", generated.answer_text)
        self.assertTrue(any("2023년 1분기" in row for row in fact_section.metadata))
        self.assertTrue(any("분기보고서 (2023.03)" in row for row in fact_section.metadata))
        self.assertTrue(all("[1]" not in row for row in fact_section.metadata))

    def test_general_evidence_uses_no_resolver_and_keeps_retrieval_order(self):
        first = _candidate(
            "m1:ch_a",
            "m1",
            rank=1,
            doc_group="major",
            content="합병 공시 첫 번째 근거",
            section="합병",
        )
        second = _candidate(
            "m1:ch_b",
            "m1",
            rank=2,
            doc_group="major",
            content="합병 공시 두 번째 근거",
            section="합병",
        )
        plan = QueryPlan(
            query="합병 공시 내용",
            task_type="corporate_event",
            disclosure_route=("major",),
        )
        execution = _execution(plan, first, second)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        self.assertEqual(
            result.execution_trace,
            ("task_router", "evidence_builder", "answer_composer"),
        )
        self.assertIsNone(result.resolution)
        self.assertTrue(result.answer_draft.answerable)
        self.assertEqual(
            result.answer_draft.evidence_references,
            ("m1:ch_a", "m1:ch_b"),
        )
        rows = result.answer_draft.answer_sections[0].content["evidence"]
        self.assertEqual([row["retrieval_rank"] for row in rows], [1, 2])

    def test_general_evidence_is_bounded_for_answer_display(self):
        items = [
            _candidate(
                f"m1:ch_{index}",
                "m1",
                rank=index,
                doc_group="major",
                content=("합병 공시 근거 " + str(index) + " ") * 200,
                section="합병",
            )
            for index in range(1, 7)
        ]
        plan = QueryPlan(
            query="합병 공시 내용",
            task_type="general_evidence",
            disclosure_route=("major",),
        )
        execution = _execution(plan, *items)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        self.assertIn("general_evidence_limited:max=5", result.answer_draft.warnings)
        self.assertEqual(
            result.answer_draft.evidence_references,
            ("m1:ch_1", "m1:ch_2", "m1:ch_3", "m1:ch_4", "m1:ch_5"),
        )
        rows = result.answer_draft.answer_sections[0].content["evidence"]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["truncated"] for row in rows))
        self.assertLess(len(rows[0]["evidence_text"]), 1300)
        self.assertTrue(all(len(row["evidence_text"]) < 700 for row in rows[1:]))

    def test_corporate_event_keeps_six_general_evidence_items(self):
        items = [
            _candidate(
                f"m1:ch_{index}",
                "m1",
                rank=index,
                doc_group="major",
                content=f"자기주식 처분 근거 {index}",
                section="자기주식 처분 결정",
            )
            for index in range(1, 8)
        ]
        plan = QueryPlan(
            query="자기주식 처분 예정 보통주 수와 가격",
            task_type="corporate_event",
            disclosure_route=("major",),
        )
        execution = _execution(plan, *items)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        self.assertIn("general_evidence_limited:max=6", result.answer_draft.warnings)
        self.assertEqual(
            result.answer_draft.evidence_references,
            (
                "m1:ch_1",
                "m1:ch_2",
                "m1:ch_3",
                "m1:ch_4",
                "m1:ch_5",
                "m1:ch_6",
            ),
        )

    def test_corporate_event_table_rows_are_focused_by_question(self):
        pair = _candidate(
            "m1:ch_table",
            "m1",
            rank=1,
            doc_group="major",
            content=(
                "[기업명] 케이티\n"
                "[공시명] 주요사항보고서(자기주식처분결정)\n"
                "[Section Path] 자기주식 처분 결정\n"
                "[Table] 자기주식 처분 결정\n"
                "| 1. 처분예정주식(주) | 보통주식 | 131,690 |\n"
                "| 2. 처분 대상 주식가격(원) | 보통주식 | 31,350 |\n"
                "| 5. 처분목적 | 임직원 주식보상 |\n"
                "| 7. 위탁투자중개업자 | NH투자증권 |\n"
                "| - 사외이사참석여부 | 참석(명) | 8 |\n"
            ),
            section="자기주식 처분 결정",
        )
        plan = QueryPlan(
            query="자기주식 처분 예정 보통주 수와 가격",
            task_type="corporate_event",
            disclosure_route=("major",),
        )
        execution = _execution(plan, pair)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        row = result.answer_draft.answer_sections[0].content["evidence"][0]
        self.assertIn("처분예정주식", row["evidence_text"])
        self.assertIn("131,690", row["evidence_text"])
        self.assertIn("처분 대상 주식가격", row["evidence_text"])
        self.assertIn("31,350", row["evidence_text"])
        self.assertNotIn("위탁투자중개업자", row["evidence_text"])
        self.assertNotIn("사외이사참석여부", row["evidence_text"])
        self.assertTrue(row["truncated"])

    def test_corporate_event_focus_ignores_contract_note_rows(self):
        pair = _candidate(
            "x1:ch_contract",
            "x1",
            rank=1,
            doc_group="exchange",
            content=(
                "[기업명] 한국항공우주\n"
                "[공시명] 단일판매ㆍ공급계약체결\n"
                "[Table] 단일판매ㆍ공급계약 체결\n"
                "| - 체결계약명 | 항공기 공급계약 |\n"
                "| 2. 계약내역 | 계약금액(원) | 100,000 |\n"
                "| 2. 계약내역 | 매출액대비(%) | 10.5 |\n"
                "| 3. 계약상대 | AIRBUS |\n"
                "| 5. 계약기간 | 시작일 | 2026-01-01 |\n"
                "| 가. 상기 계약금액은 부가가치세 제외 금액입니다. |  |  |\n"
                "| 1) 상기 계약기간은 변경될 수 있습니다. |  |  |\n"
            ),
            section="단일판매ㆍ공급계약 체결",
        )
        plan = QueryPlan(
            query="최근 공급계약 금액과 계약상대방은?",
            task_type="corporate_event",
            disclosure_route=("exchange",),
        )
        execution = _execution(plan, pair)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        text = result.answer_draft.answer_sections[0].content["evidence"][0]["evidence_text"]
        self.assertIn("계약금액", text)
        self.assertIn("계약상대", text)
        self.assertNotIn("부가가치세 제외", text)
        self.assertNotIn("변경될 수 있습니다", text)

    def test_holding_route_general_evidence_keeps_nine_items_without_resolver(self):
        items = [
            _candidate(
                f"h1:ch_{index}",
                "h1",
                rank=index,
                doc_group="holding",
                content=f"국민연금기금 변동 세부 근거 {index}",
                section="제3부 직전보고일 이후 대량변동 내역",
            )
            for index in range(1, 11)
        ]
        plan = QueryPlan(
            query="국민연금기금 변동일 변동전 변동후",
            task_type="disclosure_lookup",
            disclosure_route=("holding",),
        )
        execution = _execution(plan, *items)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        self.assertIsNone(result.resolution)
        self.assertEqual(result.task_decision.task_type, "general_evidence")
        self.assertIn("general_evidence_limited:max=9", result.answer_draft.warnings)
        self.assertEqual(
            result.answer_draft.evidence_references,
            tuple(f"h1:ch_{index}" for index in range(1, 10)),
        )

    def test_input_plan_candidates_results_and_scores_are_not_mutated(self):
        pair = _candidate(
            "p1:ch_fact",
            "p1",
            rank=1,
            doc_group="periodic",
            content="주요 제품을 생산하는 구체적인 사업 내용입니다.",
            section="주요 제품 및 서비스",
            fiscal_year=2024,
            period_type="fiscal_year",
        )
        plan = QueryPlan(
            query="주요 제품 생산",
            task_type="disclosure_lookup",
            disclosure_route=("periodic",),
            evidence={"periodic_intent": "business_product"},
        )
        execution = _execution(plan, pair)
        before = _input_snapshot(plan, execution)

        result = AgentOrchestrator().run(plan.raw_query, plan, execution)

        self.assertEqual(_input_snapshot(plan, execution), before)
        self.assertEqual(result.evidence_set.retrieval_order, ("p1:ch_fact",))
        source = result.resolution.facts[0].sources[0]
        self.assertEqual(source.retrieval_rank, pair[1].rank)
        self.assertEqual(source.retrieval_score, 0.99)

    def test_result_sequence_path_requires_and_accepts_candidate_chunks(self):
        pair = _candidate(
            "m1:ch_a",
            "m1",
            rank=1,
            doc_group="major",
            content="상장 관련 공시 근거",
            section="상장",
        )
        plan = QueryPlan(query="상장 공시", task_type="corporate_event")

        with self.assertRaises(ValueError):
            AgentOrchestrator().run(plan.raw_query, plan, [pair[1]])

        result = AgentOrchestrator().run(
            plan.raw_query,
            plan,
            [pair[1]],
            candidate_chunks=[pair[0]],
        )
        self.assertEqual(result.evidence_set.retrieval_order, ("m1:ch_a",))


if __name__ == "__main__":
    unittest.main()
