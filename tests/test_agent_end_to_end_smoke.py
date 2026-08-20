import copy
import unittest
from types import SimpleNamespace

from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.reasoning.query_plan import QueryPlan
from scripts.smoke_agent_end_to_end import (
    run_smoke_pipeline,
    validate_completed_execution,
)
from tests.test_evidence_builder import _candidate, _holding_pair


def _execution(plan, *pairs):
    return SimpleNamespace(
        plan=plan,
        chunks=[pair[0] for pair in pairs],
        results=[pair[1] for pair in pairs],
    )


def _snapshot(plan, execution):
    return (
        copy.deepcopy(plan.to_dict()),
        copy.deepcopy([candidate.chunk for candidate in execution.chunks]),
        copy.deepcopy([result.to_dict() for result in execution.results]),
    )


class _MutatingGenerator:
    def generate(self, draft):
        draft.ambiguity["injected_mutation"] = True
        return CitationAwareAnswerGenerator().generate(draft)


class _StaticUnderstanding:
    def __init__(self, question, plan):
        self.question = question
        self.plan = plan
        self.calls = []

    def understand(self, question, *, top_k):
        self.calls.append((question, top_k))
        if question != self.question:
            raise AssertionError("unexpected smoke question")
        return self.plan


class _StaticExecutor:
    def __init__(self, plan, execution):
        self.plan = plan
        self.execution = execution
        self.calls = []

    def execute(self, plan):
        self.calls.append(plan)
        if plan is not self.plan:
            raise AssertionError("query plan identity was not preserved")
        return self.execution


def _run_full_pipeline(question, plan, execution):
    understanding = _StaticUnderstanding(question, plan)
    executor = _StaticExecutor(plan, execution)
    report = run_smoke_pipeline(
        (question,),
        understanding=understanding,
        executor=executor,
        output_path=None,
    )
    if understanding.calls != [(question, 10)]:
        raise AssertionError("query understanding was not called exactly once")
    if executor.calls != [plan]:
        raise AssertionError("hybrid retrieval was not called exactly once")
    return report["queries"][0]


class AgentEndToEndSmokeTests(unittest.TestCase):
    def test_full_holding_pipeline_preserves_order_citations_and_ambiguity(self):
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
            query="효성중공업 국민연금기금 변동일 변동후 주식수",
            task_type="holding_change",
            metric="holding_shares",
            reporter="국민연금기금",
            disclosure_route=("holding",),
            evidence={
                "requested_holding_fields": ["reference_date", "after_shares"]
            },
        )
        execution = _execution(plan, first, second)
        before = _snapshot(plan, execution)

        row = _run_full_pipeline(plan.raw_query, plan, execution)

        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["task_decision"]["task_type"], "holding_event")
        self.assertEqual(
            row["execution_trace"],
            [
                "task_router",
                "evidence_builder",
                "holding_event_resolver",
                "answer_composer",
                "answer_generator",
            ],
        )
        self.assertTrue(row["answer_draft"]["ambiguity"]["temporal_ambiguity"])
        self.assertFalse(row["answer_draft"]["ambiguity"]["latest_event_selected"])
        self.assertIn("multiple_matching_holding_events", row["warnings"])
        self.assertEqual(row["citation_count"], 2)
        self.assertTrue(
            row["validation"]["all_invariants_preserved"], row["validation"]
        )
        self.assertEqual(_snapshot(plan, execution), before)

    def test_full_periodic_pipeline_preserves_repeated_periods(self):
        text = "DX 부문의 주요 제품 및 서비스에 관한 동일한 사업 내용"
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
            query="삼성전자 DX 부문의 주요 제품은 무엇인가",
            task_type="disclosure_lookup",
            disclosure_route=("periodic",),
            evidence={"periodic_intent": "business_product"},
        )
        execution = _execution(plan, older, newer)

        row = _run_full_pipeline(plan.raw_query, plan, execution)

        self.assertEqual(row["task_decision"]["task_type"], "periodic_fact")
        self.assertEqual(
            row["execution_trace"],
            [
                "task_router",
                "evidence_builder",
                "periodic_fact_resolver",
                "answer_composer",
                "answer_generator",
            ],
        )
        self.assertFalse(row["answer_draft"]["ambiguity"]["latest_period_selected"])
        rendered = row["generated_answer"]["answer_text"]
        self.assertIn("2023", rendered)
        self.assertIn("2024", rendered)
        self.assertEqual(row["citation_count"], 2)
        self.assertTrue(row["validation"]["provenance_preserved"])
        self.assertTrue(row["validation"]["all_invariants_preserved"])

    def test_general_evidence_path_uses_no_resolver(self):
        pair = _candidate(
            "m1:ch_event",
            "m1",
            rank=1,
            doc_group="major",
            content="삼성전자의 유상증자 공시 근거",
            section="유상증자",
            source_refs=[{"table_id": "tm1", "row_start": 1, "row_end": 1}],
        )
        plan = QueryPlan(
            query="삼성전자 유상증자 공시 내용",
            task_type="corporate_event",
            disclosure_route=("major",),
        )
        execution = _execution(plan, pair)

        row = _run_full_pipeline(plan.raw_query, plan, execution)

        self.assertEqual(row["task_decision"]["task_type"], "general_evidence")
        self.assertIsNone(row["task_decision"]["resolver_type"])
        self.assertEqual(
            row["execution_trace"],
            ["task_router", "evidence_builder", "answer_composer", "answer_generator"],
        )
        self.assertIsNone(row["resolution"])
        self.assertEqual(row["citation_count"], 1)
        self.assertTrue(row["validation"]["citations_have_provenance"])
        self.assertTrue(row["validation"]["all_invariants_preserved"])

    def test_generator_mutation_is_detected(self):
        pair = _candidate(
            "m1:ch_event",
            "m1",
            rank=1,
            doc_group="major",
            content="검증된 공시 근거",
            section="공시",
            source_refs=[{"table_id": "tm1", "row_start": 1, "row_end": 1}],
        )
        plan = QueryPlan(
            query="일반 공시 내용",
            task_type="corporate_event",
            disclosure_route=("major",),
        )
        execution = _execution(plan, pair)

        row = validate_completed_execution(
            plan.raw_query,
            plan,
            execution,
            generator=_MutatingGenerator(),
        )

        self.assertFalse(row["validation"]["answer_draft_not_mutated"])
        self.assertFalse(row["validation"]["all_invariants_preserved"])
        self.assertTrue(row["validation"]["retrieval_order_unchanged"])
        self.assertTrue(row["validation"]["retrieval_scores_unchanged"])


if __name__ == "__main__":
    unittest.main()
