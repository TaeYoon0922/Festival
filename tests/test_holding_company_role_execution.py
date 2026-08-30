import unittest
from types import SimpleNamespace

from app.agent.orchestrator import AgentOrchestrator, _holding_reporter_scope
from app.api.pipeline import AnswerPipeline
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.holding_company_role_resolution import (
    ROLE_PROVENANCE_KEY,
    HoldingCompanyRoleResolver,
    has_role_provenance,
    role_provenance,
)
from app.reasoning.query_plan import QueryPlan
from app.reasoning.holding_report_index import HoldingReportIndex, HoldingReportRecord
from app.reasoning.holding_report_relative_execution import HoldingReportRelativeExecution
from app.reasoning.holding_reporter import canonical_reporter_key
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope, QueryState, QueryValidator
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


ISSUER = "가상발행사"
REPORTER = "가상투자사"
WRONG_REPORTER = "가상제삼사"
ISSUER_CODE = "00000001"
REPORTER_CODE = "00000002"
WRONG_REPORTER_CODE = "00000003"
SOURCE_REF = {"table_id": "t-role", "row_start": 2, "row_end": 2}


def record(reporter: str, suffix: str, shares: str) -> HoldingReportRecord:
    return HoldingReportRecord(
        issuer_corp_code=ISSUER_CODE,
        reporter_key=canonical_reporter_key(reporter),
        raw_reporter=reporter,
        doc_id=f"holding_{suffix}",
        projection_chunk_id=f"holding_{suffix}:projection",
        reference_date="20240203",
        receipt_date="20240205",
        after_shares=shares,
        after_ratio="1.00",
        source_refs=(SOURCE_REF,),
    )


def candidate(value: HoldingReportRecord) -> CandidateChunk:
    fields = {
        "보고자/보유자": value.raw_reporter,
        "기준일/보고일": value.reference_date,
        "보유주식수": value.after_shares,
        "보유비율": value.after_ratio,
    }
    return CandidateChunk(
        value.projection_chunk_id,
        value.doc_id,
        {
            "chunk_id": value.projection_chunk_id,
            "doc_id": value.doc_id,
            "doc_group": "holding",
            "projection_type": "holding_report",
            "corp_code": value.issuer_corp_code,
            "corp_name": ISSUER,
            "report_nm": "주식등의대량보유상황보고서",
            "rcept_dt": value.receipt_date,
            "reporter": value.raw_reporter,
            "reference_date": value.reference_date,
            "projection_fields": fields,
            "projection_field_refs": {
                label: [dict(SOURCE_REF)] for label in fields
            },
            "source_refs": [dict(SOURCE_REF)],
            "section_path": ["주식등의 대량보유상황보고서"],
            "content": " | ".join(f"{key}: {item}" for key, item in fields.items()),
            "retrieval_text": " ".join(str(item) for item in fields.values()),
        },
        MetadataMatch(),
    )


def ranked(value: CandidateChunk, rank: int) -> RetrievalResult:
    return RetrievalResult(
        value.chunk_id,
        value.doc_id,
        10.0 / rank,
        rank,
        {},
    )


class StaticExecutor:
    def __init__(self, chunks, results) -> None:
        self.chunks = list(chunks)
        self.results = list(results)
        self.calls = 0
        self.plan = None

    def execute(self, plan):
        self.calls += 1
        self.plan = plan
        return SimpleNamespace(
            plan=plan,
            documents=(),
            chunks=list(self.chunks),
            results=list(self.results),
        )


class HoldingCompanyRoleExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.correct = record(REPORTER, "correct", "100")
        self.wrong = record(WRONG_REPORTER, "wrong", "999")
        self.index = HoldingReportIndex(
            (self.correct, self.wrong),
            complete=True,
            correction_finality_available=True,
        )
        self.role_resolver = HoldingCompanyRoleResolver(self.index)
        self.report_relative = HoldingReportRelativeExecution(index=self.index)
        self.scope = CorpusScope(
            companies={
                ISSUER: (ISSUER, ISSUER_CODE),
                REPORTER: (REPORTER, REPORTER_CODE),
                WRONG_REPORTER: (WRONG_REPORTER, WRONG_REPORTER_CODE),
            },
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.understanding = QueryUnderstanding(self.scope.company_aliases())
        self.validator = QueryValidator(
            corpus_scope=self.scope,
            holding_company_role_resolver=self.role_resolver,
        )
        self.orchestrator = AgentOrchestrator(
            report_relative_execution=self.report_relative,
            holding_company_role_resolver=self.role_resolver,
        )

    def test_pipeline_preserves_both_constraints_and_excludes_wrong_reporter(self) -> None:
        question = (
            f"{REPORTER}가 보유한 {ISSUER} 주식은 "
            "2024년 2월 3일 기준 몇 주야?"
        )
        correct = candidate(self.correct)
        wrong = candidate(self.wrong)
        executor = StaticExecutor(
            (wrong, correct),
            (ranked(wrong, 1), ranked(correct, 2)),
        )
        pipeline = AnswerPipeline(
            understanding=self.understanding,
            executor=executor,
            orchestrator=self.orchestrator,
            query_validator=self.validator,
            answerability_guard=AnswerabilityGuard(),
        )

        payload = pipeline.answer("ROLE-SCOPE", question)

        self.assertEqual(executor.calls, 1)
        self.assertEqual(executor.plan.company, ISSUER)
        self.assertEqual(executor.plan.corp_code, ISSUER_CODE)
        self.assertEqual(executor.plan.reporter, REPORTER)
        self.assertEqual(
            payload["think_trace"]["query_understanding"]["status"],
            QueryState.RESOLVED.value,
        )
        self.assertEqual(
            [item["doc_id"] for item in payload["retrieved_context"]],
            [self.correct.doc_id],
        )
        self.assertNotIn(self.wrong.doc_id, payload["answer"])
        self.assertNotIn("999", payload["answer"])
        self.assertIn("holding_reporter_scope", payload["think_trace"]["stages"])

    def test_pipeline_uses_one_shared_index_dependency(self) -> None:
        self.assertIs(self.orchestrator.report_relative_execution.index, self.index)
        self.assertIs(self.orchestrator.holding_company_role_resolver.index, self.index)
        self.assertIs(self.validator.holding_company_role_resolver.index, self.index)

    def test_resolved_role_records_bounded_provenance_and_scopes(self) -> None:
        question = (
            f"{REPORTER}가 보유한 {ISSUER} 주식은 "
            "2024년 2월 3일 기준 몇 주야?"
        )
        validated = self.validator.validate(self.understanding.understand(question))
        marker = validated.plan.evidence[ROLE_PROVENANCE_KEY]

        self.assertTrue(has_role_provenance(validated.plan))
        self.assertEqual(marker["source"], "corpus_relation")
        self.assertTrue(marker["resolved"])
        self.assertEqual(marker["direction_report_count"], 1)
        # Provenance only: the marker never restates who the parties are.
        self.assertNotIn(ISSUER, str(marker))
        self.assertNotIn(REPORTER, str(marker))
        self.assertNotIn(ISSUER_CODE, str(marker))

        scoped = _holding_reporter_scope(
            self.served(self.wrong, self.correct),
            validated.plan,
            routed_task_type="holding_event",
            resolver=self.role_resolver,
        )
        self.assertIsNotNone(scoped)
        self.assertEqual([row.doc_id for row in scoped.results], [self.correct.doc_id])

    def test_reporter_named_by_the_asker_is_left_alone(self) -> None:
        """A one-company question whose reporter the user typed is untouched."""

        plan = QueryPlan(
            query="보유주식수",
            raw_query=f"{ISSUER} {REPORTER} 보유주식수",
            companies=(ISSUER,),
            corp_codes=(ISSUER_CODE,),
            task_type="holding_change",
            disclosure_route=("holding",),
            reporter=REPORTER,
        )
        self.assertFalse(has_role_provenance(plan))

        scoped = _holding_reporter_scope(
            self.served(self.wrong, self.correct),
            plan,
            routed_task_type="holding_event",
            resolver=self.role_resolver,
        )
        self.assertIsNone(scoped, "pre-B.2 reporter behaviour must not change")

    def test_surviving_ranks_are_relabelled_densely(self) -> None:
        second = record(REPORTER, "correct_two", "200")
        index = HoldingReportIndex(
            (self.correct, second, self.wrong),
            complete=True,
            correction_finality_available=True,
        )
        resolver = HoldingCompanyRoleResolver(index)
        first_chunk = candidate(self.correct)
        second_chunk = candidate(second)
        wrong_chunk = candidate(self.wrong)
        execution = SimpleNamespace(
            plan=None,
            chunks=[wrong_chunk, first_chunk, second_chunk],
            # sparse on purpose: the survivors start at 2 and 5
            results=[ranked(wrong_chunk, 1), ranked(first_chunk, 2), ranked(second_chunk, 5)],
        )

        scoped = _holding_reporter_scope(
            execution,
            self.role_plan(),
            routed_task_type="holding_event",
            resolver=resolver,
        )

        self.assertEqual([row.rank for row in scoped.results], [1, 2])
        self.assertEqual(
            [row.doc_id for row in scoped.results],
            [self.correct.doc_id, second.doc_id],
        )
        # relabelling only: identity, scores and provenance survive untouched
        self.assertEqual(
            [row.bm25_score for row in scoped.results], [10.0 / 2, 10.0 / 5]
        )
        self.assertEqual(
            [row.chunk_id for row in scoped.results],
            [self.correct.projection_chunk_id, second.projection_chunk_id],
        )
        self.assertEqual([row.doc_id for row in scoped.chunks],
                         [self.correct.doc_id, second.doc_id])

    def test_single_survivor_is_rank_one(self) -> None:
        execution = self.served(self.wrong, self.correct)
        scoped = _holding_reporter_scope(
            execution,
            self.role_plan(),
            routed_task_type="holding_event",
            resolver=self.role_resolver,
        )
        self.assertEqual([row.rank for row in scoped.results], [1])
        self.assertEqual(execution.results[1].rank, 2, "input must not be mutated")

    def test_multiple_same_direction_documents_all_survive_in_order(self) -> None:
        second = record(REPORTER, "correct_two", "200")
        third = record(REPORTER, "correct_three", "300")
        index = HoldingReportIndex(
            (self.correct, second, third, self.wrong),
            complete=True,
            correction_finality_available=True,
        )
        chunks = [candidate(value) for value in (self.correct, self.wrong, second, third)]
        execution = SimpleNamespace(
            plan=None,
            chunks=chunks,
            results=[ranked(chunk, i) for i, chunk in enumerate(chunks, start=1)],
        )

        scoped = _holding_reporter_scope(
            execution,
            self.role_plan(),
            routed_task_type="holding_event",
            resolver=HoldingCompanyRoleResolver(index),
        )

        self.assertEqual(
            [row.doc_id for row in scoped.results],
            [self.correct.doc_id, second.doc_id, third.doc_id],
        )
        self.assertEqual([row.rank for row in scoped.results], [1, 2, 3])

    def test_empty_reporter_scope_fails_closed(self) -> None:
        question = (
            f"{REPORTER}가 보유한 {ISSUER} 주식은 "
            "2024년 2월 3일 기준 몇 주야?"
        )
        wrong = candidate(self.wrong)
        executor = StaticExecutor((wrong,), (ranked(wrong, 1),))
        pipeline = AnswerPipeline(
            understanding=self.understanding,
            executor=executor,
            orchestrator=self.orchestrator,
            query_validator=self.validator,
            answerability_guard=AnswerabilityGuard(),
        )

        payload = pipeline.answer("ROLE-EMPTY", question)

        self.assertEqual(payload["retrieved_context"], [])
        self.assertFalse(payload["think_trace"]["answerable"])
        self.assertEqual(payload["think_trace"]["selected_evidence_count"], 0)
        self.assertNotIn("999", payload["answer"])
        self.assertNotIn(self.wrong.doc_id, payload["answer"])

    def test_report_relative_owns_latest_and_bypasses_reporter_scope(self) -> None:
        payload = self.answered(
            f"{REPORTER}가 보유한 {ISSUER} 주식은 최신 보고 기준 몇 주야?"
        )
        stages = payload["think_trace"]["stages"]

        self.assertIn("holding_report_relative_execution", stages)
        self.assertNotIn("holding_reporter_scope", stages)
        self.assertEqual(
            [(row["rank"], row["doc_id"]) for row in payload["retrieved_context"]],
            [(1, self.correct.doc_id)],
        )

    def test_unbound_selected_context_stays_fail_closed(self) -> None:
        payload = self.answered(
            f"{REPORTER}가 보유한 {ISSUER} 주식은 이번 보고 기준 몇 주야?"
        )
        stages = payload["think_trace"]["stages"]

        self.assertIn("holding_report_relative_execution", stages)
        self.assertNotIn("holding_reporter_scope", stages)
        self.assertEqual(payload["retrieved_context"], [])
        self.assertFalse(payload["think_trace"]["answerable"])

    # ------------------------------------------------------------- helpers
    def served(self, *records):
        chunks = [candidate(value) for value in records]
        return SimpleNamespace(
            plan=None,
            chunks=chunks,
            results=[ranked(chunk, i) for i, chunk in enumerate(chunks, start=1)],
        )

    def role_plan(self) -> QueryPlan:
        return QueryPlan(
            query="보유주식수",
            raw_query=f"{REPORTER} {ISSUER} 보유주식수",
            companies=(ISSUER,),
            corp_codes=(ISSUER_CODE,),
            task_type="holding_change",
            disclosure_route=("holding",),
            reporter=REPORTER,
            evidence={
                ROLE_PROVENANCE_KEY: role_provenance(
                    self.role_resolver.resolve(
                        ISSUER, ISSUER_CODE, REPORTER, REPORTER_CODE
                    )
                )
            },
        )

    def answered(self, question: str):
        correct = candidate(self.correct)
        wrong = candidate(self.wrong)
        executor = StaticExecutor(
            (wrong, correct), (ranked(wrong, 1), ranked(correct, 2))
        )
        pipeline = AnswerPipeline(
            understanding=self.understanding,
            executor=executor,
            orchestrator=self.orchestrator,
            query_validator=self.validator,
            answerability_guard=AnswerabilityGuard(),
        )
        return pipeline.answer("ROLE-PRECEDENCE", question)


if __name__ == "__main__":
    unittest.main()
