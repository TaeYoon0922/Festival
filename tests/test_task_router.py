import inspect
import json
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from app.agent.task_router import TaskRouter, route_task
from app.reasoning.query_plan import QueryPlan


class TaskRouterTests(unittest.TestCase):
    def test_holding_question_routes_to_holding_resolver(self):
        decision = route_task("국민연금 변동일 변동 후 보유주식수")

        self.assertEqual(decision.task_type, "holding_event")
        self.assertEqual(decision.resolver_type, "holding_event_resolver")
        self.assertIn("query:국민연금", decision.matched_signals)
        self.assertGreater(decision.confidence, 0.8)
        with self.assertRaises(FrozenInstanceError):
            decision.task_type = "unknown"

    def test_ownership_situation_report_routes_to_holding_resolver(self):
        plan = QueryPlan(
            query="임원주요주주특정증권등소유상황보고서 특수관계자",
            raw_query=(
                "기아의 가장 최근 임원ㆍ주요주주특정증권등소유상황보고서에서 "
                "특수관계자가 한 명 더 늘어난 이유는?"
            ),
            task_type="holding_change",
            metric="holding_shares",
            disclosure_route=("holding",),
        )
        decision = route_task(plan.raw_query, plan)

        self.assertEqual(decision.task_type, "holding_event")
        self.assertEqual(decision.resolver_type, "holding_event_resolver")
        self.assertIn("plan:task_type=holding_change", decision.matched_signals)

    def test_gold60_holding_routes_including_hx12_use_holding_resolver(self):
        artifact = (
            Path(__file__).resolve().parents[1]
            / "reports/evaluation/gold60/2026-08-21-agent-90pct"
            / "gold60_agent_questions.jsonl"
        )
        rows = [
            json.loads(line)
            for line in artifact.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        holding_routes = [
            row
            for row in rows
            if row["query_plan"]["task_type"] == "disclosure_lookup"
            and "holding" in row["query_plan"]["disclosure_route"]
        ]

        self.assertEqual(
            [row["question_id"] for row in holding_routes],
            [
                "H02",
                "H03",
                "H05",
                "H07",
                "H08",
                "HX04",
                "HX12",
                "HX13",
                "HX17",
            ],
        )
        for row in holding_routes:
            with self.subTest(question_id=row["question_id"]):
                decision = route_task(row["question"], row["query_plan"])

                self.assertEqual(decision.task_type, "holding_event")
                self.assertEqual(
                    decision.resolver_type,
                    "holding_event_resolver",
                )
                self.assertIn("plan:route=holding", decision.matched_signals)

    def test_periodic_question_routes_to_periodic_resolver(self):
        decision = TaskRouter().route("주요 제품과 연구개발 사업 내용")

        self.assertEqual(decision.task_type, "periodic_fact")
        self.assertEqual(decision.resolver_type, "periodic_fact_resolver")
        self.assertIn("query:주요제품", decision.matched_signals)
        self.assertIn("query:연구개발", decision.matched_signals)

    def test_query_plan_semantics_take_precedence_deterministically(self):
        plan = QueryPlan(
            query="보유 현황",
            task_type="holding_change",
            metric="holding_shares",
        )

        first = route_task("일반적인 질문", plan)
        second = route_task("일반적인 질문", plan)

        self.assertEqual(first, second)
        self.assertEqual(first.task_type, "holding_event")
        self.assertIn("plan:task_type=holding_change", first.matched_signals)

    def test_general_evidence_and_unknown_are_distinct(self):
        general = route_task(
            "합병 공시 내용",
            QueryPlan(query="합병 공시 내용", task_type="general_evidence"),
        )
        unknown = route_task("오늘 점심은 무엇인가")

        self.assertEqual(general.task_type, "general_evidence")
        self.assertIsNone(general.resolver_type)
        self.assertEqual(unknown.task_type, "unknown")
        self.assertIsNone(unknown.resolver_type)
        self.assertIn("no_task_signal", unknown.warnings)

    def test_corporate_event_query_plan_task_is_preserved(self):
        decision = route_task(
            "삼성전자 유상증자 공시 내용",
            QueryPlan(
                query="삼성전자 유상증자 공시 내용",
                task_type="corporate_event",
                disclosure_route=("major",),
            ),
        )

        self.assertEqual(decision.task_type, "corporate_event")
        self.assertIsNone(decision.resolver_type)
        self.assertIn("plan:task_type=corporate_event", decision.matched_signals)

    def test_router_source_contains_no_evaluation_id_special_cases(self):
        source = inspect.getsource(inspect.getmodule(route_task))

        for forbidden in ("Gold60", "P01", "P06", "P13", "HX09"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
