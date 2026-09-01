import unittest
from types import SimpleNamespace

from app.agent.orchestrator import AgentOrchestrator, _holding_reporter_scope
from app.api.pipeline import AnswerPipeline
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.field_evidence import FieldReason, FieldStatus
from app.reasoning.holding_company_role_resolution import (
    AMBIGUOUS_FILER,
    INCOMPLETE_INDEX,
    NO_INDEX,
    RESOLVED,
    ROLE_PATH_FILER,
    ROLE_PROVENANCE_KEY,
    STALE_INDEX,
    UNKNOWN_FILER,
    HoldingCompanyRoleResolver,
    has_role_provenance,
    role_provenance,
)
from app.reasoning.query_plan import QueryPlan
from app.reasoning.holding_report_index import HoldingReportIndex, HoldingReportRecord
from app.reasoning.holding_report_relative_execution import HoldingReportRelativeExecution
from app.reasoning.holding_reporter import canonical_reporter_key
from app.reasoning.holding_evidence_coverage import anchor_tier
from app.reasoning.holding_field_evidence import (
    ACQUISITION_UNIT_PRICE,
    _selected_event,
    holding_field_evidence,
    requested_holding_fields,
)
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


def acquisition_candidate(value: HoldingReportRecord) -> CandidateChunk:
    """A production-shaped acquisition row with an explicitly omitted price."""

    headers = [
        "성명(명칭)", "생년월일 또는사업자등록번호 등", "변동일*", "취득/처분방법",
        "주식등의종류", "변동 내역 / 변동전", "변동 내역 / 증감",
        "변동 내역 / 변동후", "취득/처분단가**", "취득/처분단가**", "비 고",
    ]
    row = [
        value.raw_reporter, "120-00-00000", "2024.02.03", "장내매수(+)",
        "의결권있는 주식", "2,000", "100", "2,100", "-", "-",
        "최초 보고이므로 취득단가 미기재",
    ]
    source_ref = {"table_id": "t-acquisition", "row_start": 2, "row_end": 2}
    fields = {
        "보고자/보유자": row[0],
        "기준일/보고일": row[2],
        "직전 보유주식수": row[5],
        "증감주식수": row[6],
        "보유주식수": row[7],
    }
    chunk_id = f"{value.doc_id}:acquisition"
    return CandidateChunk(
        chunk_id,
        value.doc_id,
        {
            "chunk_id": chunk_id,
            "doc_id": value.doc_id,
            "doc_group": "holding",
            "projection_type": "holding_detail_row",
            "chunk_type": "table_projection",
            "corp_code": value.issuer_corp_code,
            "corp_name": ISSUER,
            "report_nm": "주식등의대량보유상황보고서",
            "rcept_dt": value.receipt_date,
            "reporter": value.raw_reporter,
            "reference_date": value.reference_date,
            "section_path": ["세부변동내역"],
            "content": "세부 변동 내역",
            "retrieval_text": "장내매수 취득단가 미기재",
            "table_id": "t-acquisition",
            "source_table_id": "t-acquisition",
            "row_start": 2,
            "row_end": 2,
            "column_headers": headers,
            "header_rows": [[
                *[
                    {"text": header, "colspan": 1, "rowspan": 1,
                     "is_header": True}
                    for header in headers[:8]
                ],
                {"text": "취득/처분단가**", "colspan": 2, "rowspan": 1,
                 "is_header": True},
                {"text": "비 고", "colspan": 1, "rowspan": 1,
                 "is_header": True},
            ]],
            "table_rows": [[
                {"text": cell, "colspan": 1, "rowspan": 1, "is_header": False}
                for cell in row
            ]],
            "source_refs": [dict(source_ref)],
            "projection_fields": fields,
            "projection_field_refs": {
                label: [dict(source_ref)] for label in fields
            },
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


class RecordingOrchestrator:
    def __init__(self, delegate: AgentOrchestrator) -> None:
        self.delegate = delegate
        self.result = None

    def run(self, *args, **kwargs):
        self.result = self.delegate.run(*args, **kwargs)
        return self.result


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

    def test_raw_acquisition_unit_price_runs_the_production_holding_path(self) -> None:
        question = (
            f"{REPORTER}가 {ISSUER} 주식을 취득할 때의 취득단가는 얼마였어?"
        )
        index = HoldingReportIndex(
            (self.correct,), complete=True, correction_finality_available=True
        )
        resolver = HoldingCompanyRoleResolver(index)
        validator = QueryValidator(
            corpus_scope=self.scope,
            holding_company_role_resolver=resolver,
        )
        orchestrator = AgentOrchestrator(
            report_relative_execution=HoldingReportRelativeExecution(index=index),
            holding_company_role_resolver=resolver,
        )
        recording = RecordingOrchestrator(orchestrator)
        detail = acquisition_candidate(self.correct)
        # A higher-ranked generic holding projection for another reporter is
        # retrieval noise; corpus-proved role scope must keep it from deciding.
        generic = candidate(self.wrong)
        executor = StaticExecutor(
            (generic, detail), (ranked(generic, 1), ranked(detail, 2))
        )

        understood = self.understanding.understand(question)
        self.assertEqual(understood.task_type, "holding_change")
        self.assertEqual(understood.metric, ACQUISITION_UNIT_PRICE)
        self.assertEqual(understood.disclosure_route, ("holding",))
        self.assertEqual(understood.doc_group, "holding")
        self.assertEqual(understood.evidence["operation"], "lookup_holding")

        validated = validator.validate(understood)
        self.assertIs(validated.state, QueryState.RESOLVED)
        self.assertEqual(validated.plan.company, ISSUER)
        self.assertEqual(validated.plan.reporter, REPORTER)
        self.assertTrue(has_role_provenance(validated.plan))

        payload = AnswerPipeline(
            understanding=self.understanding,
            executor=executor,
            orchestrator=recording,
            query_validator=validator,
            answerability_guard=AnswerabilityGuard(),
        ).answer("ACQUISITION-UNIT-PRICE", question)

        self.assertEqual(executor.plan.task_type, "holding_change")
        self.assertEqual(executor.plan.metric, ACQUISITION_UNIT_PRICE)
        self.assertEqual(executor.plan.company, ISSUER)
        self.assertEqual(executor.plan.reporter, REPORTER)
        self.assertIsNotNone(recording.result)
        self.assertEqual(recording.result.resolution.matching_event_count, 1)
        self.assertIn("holding_event_resolver", recording.result.execution_trace)
        self.assertEqual(len(recording.result.field_evidence), 1)
        field = recording.result.field_evidence[0]
        self.assertIs(field.status, FieldStatus.UNAVAILABLE)
        self.assertIs(field.reason, FieldReason.OMITTED)
        self.assertEqual(field.doc_id, self.correct.doc_id)
        self.assertEqual(field.chunk_id, detail.chunk_id)
        self.assertEqual(field.table_id, "t-acquisition")
        self.assertEqual((field.row_start, field.row_end), (2, 2))

        answerability = payload["think_trace"]["answerability"]
        self.assertFalse(payload["think_trace"]["answerable"])
        self.assertEqual(answerability["status"], "insufficient_evidence")
        self.assertEqual(answerability["unavailable_fields"],
                         [ACQUISITION_UNIT_PRICE])
        self.assertTrue(answerability["citable"])
        self.assertEqual(
            answerability["unavailable_evidence"][0]["chunk_id"], detail.chunk_id
        )
        self.assertIn("query_understanding", payload["think_trace"]["stages"])
        self.assertIn("query_validation", payload["think_trace"]["stages"])
        self.assertIn("holding_event_resolver", payload["think_trace"]["stages"])
        self.assertIn("field_evidence", payload["think_trace"]["stages"])
        self.assertNotIn(
            self.wrong.doc_id,
            [context["doc_id"] for context in payload["retrieved_context"]],
        )
        self.assertIn("[", payload["answer"])

    def test_acquisition_unit_price_role_ambiguity_fails_closed(self) -> None:
        question = (
            f"{REPORTER}가 {ISSUER} 주식을 매수한 단가는 얼마였어?"
        )
        empty_index = HoldingReportIndex(
            (), complete=True, correction_finality_available=True
        )
        validator = QueryValidator(
            corpus_scope=self.scope,
            holding_company_role_resolver=HoldingCompanyRoleResolver(empty_index),
        )

        understood = self.understanding.understand(question)
        validated = validator.validate(understood)

        self.assertEqual(understood.metric, ACQUISITION_UNIT_PRICE)
        self.assertIs(validated.state, QueryState.AMBIGUOUS)
        self.assertFalse(validated.retrieval_allowed)
        self.assertIsNone(validated.plan.reporter)

    def test_pipeline_uses_one_shared_index_dependency(self) -> None:
        self.assertIs(self.orchestrator.report_relative_execution.index, self.index)
        self.assertIs(self.orchestrator.holding_company_role_resolver.index, self.index)
        self.assertIs(self.orchestrator.holding_report_index, self.index)
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


def priced_acquisition_candidate(value, price: str) -> CandidateChunk:
    """The same production-shaped row, with the unit price actually stated."""

    base = acquisition_candidate(value)
    chunk = dict(base.chunk)
    rows = [[dict(cell) for cell in row] for row in chunk["table_rows"]]
    for cell in rows[0][8:10]:
        cell["text"] = price
    rows[0][10]["text"] = ""
    chunk["table_rows"] = rows
    chunk["retrieval_text"] = f"장내매수 취득단가 {price}"
    return CandidateChunk(base.chunk_id, base.doc_id, chunk, MetadataMatch())


def dated_acquisition_candidate(
    value, trade_date: str, *, price: str | None = None
) -> CandidateChunk:
    """The same row, dated the day the shares actually changed hands.

    A holding report is filed after the transaction it reports, so the detail
    row's date is ordinarily older than the report's own date.  That gap is the
    shape this fixture exists to carry.
    """

    base = acquisition_candidate(value)
    chunk = dict(base.chunk)
    rows = [[dict(cell) for cell in row] for row in chunk["table_rows"]]
    rows[0][2]["text"] = trade_date
    if price is not None:
        for cell in rows[0][8:10]:
            cell["text"] = price
        rows[0][10]["text"] = ""
    chunk["table_rows"] = rows
    fields = dict(chunk["projection_fields"])
    fields["기준일/보고일"] = trade_date
    chunk["projection_fields"] = fields
    return CandidateChunk(base.chunk_id, base.doc_id, chunk, MetadataMatch())


class FilerIdentityResolutionTests(unittest.TestCase):
    """Proving which of an issuer's own filers a question named.

    ``resolve`` compares two canonical companies because both are in the
    company universe.  A holding report's filer usually is not, so the surface
    a question used is checked against the holders this issuer actually has.
    Every outcome other than exactly one holder fails closed: naming the wrong
    one states some other holder's position as this one's.
    """

    HOLDER = "가상지주"
    HOLDER_LEGAL_FORM = "(주)가상지주"
    PENSION = "가상연금"
    PENSION_AGENCY = "가상연금공단"
    PENSION_FUND = "가상연금기금"
    OTHER_ISSUER_CODE = "00000009"

    def record(
        self,
        reporter: str,
        suffix: str,
        *,
        issuer_code: str = ISSUER_CODE,
        reference: str = "20240203",
    ) -> HoldingReportRecord:
        return HoldingReportRecord(
            issuer_corp_code=issuer_code,
            reporter_key=canonical_reporter_key(reporter),
            raw_reporter=reporter,
            doc_id=f"holding_{suffix}",
            projection_chunk_id=f"holding_{suffix}:projection",
            reference_date=reference,
            receipt_date="20240205",
            after_shares="100",
            after_ratio="1.00",
        )

    def resolver(self, *records, **over) -> HoldingCompanyRoleResolver:
        settings = dict(complete=True, correction_finality_available=True)
        settings.update(over)
        return HoldingCompanyRoleResolver(HoldingReportIndex(records, **settings))

    def resolve(self, surface: str, *records, **over):
        return self.resolver(*records, **over).resolve_filer(
            ISSUER, ISSUER_CODE, surface
        )

    def test_resolve_filer_binds_unique_issuer_reporter(self) -> None:
        result = self.resolve(self.HOLDER, self.record(self.HOLDER, "a1"))

        self.assertEqual(result.status, RESOLVED)
        self.assertEqual(result.reporter, self.HOLDER)
        self.assertEqual(result.issuer_corp_code, ISSUER_CODE)

    def test_resolve_filer_unknown_fails_closed(self) -> None:
        result = self.resolve(self.HOLDER, self.record(REPORTER, "a1"))

        self.assertEqual(result.status, UNKNOWN_FILER)
        self.assertIsNone(result.reporter)

    def test_resolve_filer_ambiguous_fails_closed(self) -> None:
        # Two holders answer to one name through the frozen family-suffix rule
        # and neither is the name itself.  Two holders are not one holder
        # written twice, so neither may be picked.
        result = self.resolve(
            self.PENSION,
            self.record(self.PENSION_AGENCY, "a1"),
            self.record(self.PENSION_FUND, "a2"),
        )

        self.assertEqual(result.status, AMBIGUOUS_FILER)
        self.assertIsNone(result.reporter)

    def test_exact_canonical_key_wins_over_family_suffix(self) -> None:
        result = self.resolve(
            self.PENSION,
            self.record(self.PENSION, "a1"),
            self.record(self.PENSION_AGENCY, "a2"),
        )

        self.assertEqual(result.status, RESOLVED)
        self.assertEqual(result.reporter, self.PENSION)

    def test_same_holder_spelled_two_ways_is_not_ambiguous(self) -> None:
        result = self.resolve(
            self.HOLDER,
            self.record(self.HOLDER, "a1"),
            self.record(self.HOLDER_LEGAL_FORM, "a2", reference="20240301"),
        )

        self.assertEqual(result.status, RESOLVED)
        self.assertEqual(
            canonical_reporter_key(result.reporter),
            canonical_reporter_key(self.HOLDER),
        )

    def test_resolve_filer_never_crosses_issuers(self) -> None:
        result = self.resolve(
            self.HOLDER,
            self.record(self.HOLDER, "b1", issuer_code=self.OTHER_ISSUER_CODE),
        )

        self.assertEqual(result.status, UNKNOWN_FILER)
        self.assertIsNone(result.reporter)

    def test_unusable_index_fails_closed(self) -> None:
        holder = self.record(self.HOLDER, "a1")
        stale = HoldingCompanyRoleResolver(
            HoldingReportIndex(
                (holder,), complete=True, identity={"corpus_manifest_sha256": "one"}
            ),
            active_corpus_identity={"corpus_manifest_sha256": "two"},
        )
        for label, resolver, expected in (
            ("absent", HoldingCompanyRoleResolver(None), NO_INDEX),
            ("incomplete", self.resolver(holder, complete=False), INCOMPLETE_INDEX),
            ("stale", stale, STALE_INDEX),
        ):
            with self.subTest(index=label):
                result = resolver.resolve_filer(ISSUER, ISSUER_CODE, self.HOLDER)
                self.assertEqual(result.status, expected)
                self.assertIsNone(result.reporter)


class DirectedFilerAcquisitionEndToEndTests(unittest.TestCase):
    """The whole path for an acquirer no company universe can name.

    The question names a filer, the corpus proves which holder that is, the
    frozen resolver selects that holder's acquisition row, and the row's blank
    unit price is reported as an absence with a citation rather than filled in
    from whatever else retrieval happened to return.
    """

    HOLDER = "가상지주"
    RIVAL_PRICE = "77,777"

    def setUp(self) -> None:
        self.selected = record(self.HOLDER, "selected", "100")
        self.rival = record(WRONG_REPORTER, "rival", "999")
        self.index = HoldingReportIndex(
            (self.selected, self.rival),
            complete=True,
            correction_finality_available=True,
        )
        self.resolver = HoldingCompanyRoleResolver(self.index)
        # The acquirer is deliberately absent from the universe: that absence
        # is the structural shape this whole path exists for.
        self.scope = CorpusScope(
            companies={ISSUER: (ISSUER, ISSUER_CODE)},
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.understanding = QueryUnderstanding(self.scope.company_aliases())
        self.validator = QueryValidator(
            corpus_scope=self.scope,
            holding_company_role_resolver=self.resolver,
        )
        self.orchestrator = AgentOrchestrator(
            report_relative_execution=HoldingReportRelativeExecution(index=self.index),
            holding_company_role_resolver=self.resolver,
        )

    def test_blank_acquisition_unit_price_is_unavailable_with_grounded_source(
        self,
    ) -> None:
        question = (
            f"{self.HOLDER}가 {ISSUER} 주식을 "
            "취득할 때의 취득단가는 얼마야?"
        )

        understood = self.understanding.understand(question)
        self.assertEqual(understood.companies, (ISSUER,))
        self.assertIsNone(understood.reporter)
        self.assertEqual(understood.metric, ACQUISITION_UNIT_PRICE)

        validated = self.validator.validate(understood)
        self.assertIs(validated.state, QueryState.RESOLVED)
        self.assertEqual(validated.plan.reporter, self.HOLDER)
        self.assertTrue(has_role_provenance(validated.plan))
        self.assertEqual(
            validated.plan.evidence[ROLE_PROVENANCE_KEY]["path"], ROLE_PATH_FILER
        )

        detail = acquisition_candidate(self.selected)
        # Another holder's row, priced and ranked first: proving the answer
        # comes from the selected holder means proving this cannot supply it.
        rival = priced_acquisition_candidate(self.rival, self.RIVAL_PRICE)
        executor = StaticExecutor(
            (rival, detail), (ranked(rival, 1), ranked(detail, 2))
        )
        recording = RecordingOrchestrator(self.orchestrator)

        payload = AnswerPipeline(
            understanding=self.understanding,
            executor=executor,
            orchestrator=recording,
            query_validator=self.validator,
            answerability_guard=AnswerabilityGuard(),
        ).answer("DIRECTED-FILER-UNIT-PRICE", question)

        self.assertEqual(len(recording.result.field_evidence), 1)
        field = recording.result.field_evidence[0]
        self.assertIs(field.status, FieldStatus.UNAVAILABLE)
        self.assertEqual(field.doc_id, self.selected.doc_id)
        self.assertEqual(field.chunk_id, detail.chunk_id)
        self.assertEqual(field.table_id, "t-acquisition")
        self.assertEqual((field.row_start, field.row_end), (2, 2))

        self.assertFalse(payload["think_trace"]["answerable"])
        answerability = payload["think_trace"]["answerability"]
        self.assertEqual(answerability["status"], "insufficient_evidence")
        self.assertEqual(
            answerability["unavailable_fields"], [ACQUISITION_UNIT_PRICE]
        )
        self.assertEqual(
            answerability["unavailable_evidence"][0]["chunk_id"], detail.chunk_id
        )
        # No other holder's price may stand in for the one that was omitted.
        self.assertNotIn(self.RIVAL_PRICE, payload["answer"])
        self.assertNotIn(
            self.rival.doc_id,
            [context["doc_id"] for context in payload["retrieved_context"]],
        )


class RecoveredAcquisitionEvidenceTests(unittest.TestCase):
    """The proving detail row is fetched but unranked, and must still arrive.

    A holding detail projection carries its acquisition columns only inside
    ``column_headers``/``table_rows``, so no acquisition wording reaches its
    lexical surface and a question about a unit price cannot rank it.  The
    position summary ranks instead, the resolver is handed a row that proves no
    acquisition, and it fails closed.  These cases serve exactly that shape --
    summary ranked, detail row present only in the fetched pool -- and pin that
    coverage recovery closes it without any component downstream changing.
    """

    HOLDER = "가상지주"
    PRICE = "77,777"
    #: Both projections of one filing state the reference date the same way, as
    #: the corpus does.  ``record`` carries the compact rcept-style value that
    #: a projection label never holds, so the summary restates it here.
    REFERENCE = "2024.02.03"
    #: A filing is dated after the transaction it reports.
    REPORT_DATE = "2025.02.07"
    TRADE_DATE = "2025.02.05"

    def summary(self, reference=None):
        """The position projection, dated the day the position was reported."""

        base = candidate(self.selected)
        chunk = dict(base.chunk)
        fields = dict(chunk["projection_fields"])
        fields["기준일/보고일"] = reference or self.REFERENCE
        chunk["projection_fields"] = fields
        chunk["projection_field_refs"] = {
            label: [dict(SOURCE_REF)] for label in fields
        }
        return CandidateChunk(base.chunk_id, base.doc_id, chunk, MetadataMatch())

    def setUp(self) -> None:
        # The summary and its detail row are two renderings of one filing,
        # so their position must agree; a disagreement is a real conflict
        # the frozen producer is right to refuse.
        self.selected = record(self.HOLDER, "selected", "2,100")
        self.index = HoldingReportIndex(
            (self.selected,), complete=True, correction_finality_available=True
        )
        self.resolver = HoldingCompanyRoleResolver(self.index)
        # The acquirer is absent from the universe, as a filer normally is.
        self.scope = CorpusScope(
            companies={ISSUER: (ISSUER, ISSUER_CODE)},
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.understanding = QueryUnderstanding(self.scope.company_aliases())
        self.validator = QueryValidator(
            corpus_scope=self.scope,
            holding_company_role_resolver=self.resolver,
        )
        self.orchestrator = AgentOrchestrator(
            report_relative_execution=HoldingReportRelativeExecution(index=self.index),
            holding_company_role_resolver=self.resolver,
        )
        self.question = (
            f"{self.HOLDER}가 {ISSUER} 주식을 "
            "취득할 때의 취득단가는 얼마야?"
        )

    def _run(self, pool, served):
        """Serve ``served`` while ``pool`` is what retrieval already fetched."""

        executor = StaticExecutor(
            pool, [ranked(candidate, index + 1)
                   for index, candidate in enumerate(served)]
        )
        recording = RecordingOrchestrator(self.orchestrator)
        payload = AnswerPipeline(
            understanding=self.understanding,
            executor=executor,
            orchestrator=recording,
            query_validator=self.validator,
            answerability_guard=AnswerabilityGuard(),
        ).answer("RECOVERED-ACQUISITION", self.question)
        return recording.result, payload

    def test_unranked_detail_row_is_recovered_and_grounds_unavailable(self) -> None:
        summary = self.summary()
        detail = acquisition_candidate(self.selected)

        # Baseline: without the detail row in the pool at all, the frozen
        # resolver has nothing that proves an acquisition and fails closed.
        without, _payload = self._run((summary,), (summary,))
        self.assertEqual(without.field_evidence, ())
        self.assertTrue(
            all(event.transaction_method is None
                for event in without.resolution.events)
        )

        result, payload = self._run((summary, detail), (summary,))

        # The frozen resolver now discovers the acquisition for itself.
        proven = [event for event in result.resolution.events
                  if event.transaction_method is not None]
        self.assertEqual(len(proven), 1)
        self.assertIsNotNone(proven[0].acquisition_date)
        self.assertIsNotNone(proven[0].acquired_shares)
        self.assertEqual(result.resolution.matching_event_count, 1)

        # The frozen producer reads the recovered physical row.
        self.assertEqual(len(result.field_evidence), 1)
        field = result.field_evidence[0]
        self.assertIs(field.status, FieldStatus.UNAVAILABLE)
        self.assertEqual(field.doc_id, self.selected.doc_id)
        self.assertEqual(field.chunk_id, detail.chunk_id)
        self.assertEqual(field.table_id, "t-acquisition")
        self.assertEqual((field.row_start, field.row_end), (2, 2))

        # Answerability decides through the ordinary FieldEvidence contract.
        self.assertFalse(payload["think_trace"]["answerable"])
        answerability = payload["think_trace"]["answerability"]
        self.assertEqual(answerability["status"], "insufficient_evidence")
        self.assertEqual(answerability["unavailable_fields"],
                         [ACQUISITION_UNIT_PRICE])
        self.assertEqual(
            answerability["unavailable_evidence"][0]["chunk_id"], detail.chunk_id
        )

    def test_recovered_numeric_unit_price_is_available_from_that_row(self) -> None:
        summary = self.summary()
        detail = priced_acquisition_candidate(self.selected, self.PRICE)

        result, payload = self._run((summary, detail), (summary,))

        self.assertEqual(len(result.field_evidence), 1)
        field = result.field_evidence[0]
        self.assertIs(field.status, FieldStatus.AVAILABLE)
        self.assertEqual(field.value, self.PRICE)
        self.assertEqual(field.doc_id, self.selected.doc_id)
        self.assertEqual(field.chunk_id, detail.chunk_id)
        self.assertEqual(field.table_id, "t-acquisition")
        self.assertEqual((field.row_start, field.row_end), (2, 2))
        # No neighbouring quantity may be read as the unit price.
        for neighbour in ("100", "2,000", "2,100", "1.00"):
            self.assertNotEqual(field.value, neighbour)
        self.assertTrue(payload["think_trace"]["answerable"])



    def test_transaction_older_than_its_report_is_answered(self) -> None:
        """The measured shape: the filing is dated after the trade it reports.

        Two things had to be true for this to work, and both are pinned here.
        Coverage anchors the proof row on the filing rather than on a shared
        event date, so the row arrives at all.  The producer then measures
        ambiguity among the acquisitions rather than among every event, so the
        position snapshot the filing also renders -- dated the day the position
        was reported, proving no acquisition -- stops being treated as a rival
        answer for a unit price it never carried.

        The resolver still reports both facts truthfully, including the temporal
        ambiguity between them.  Nothing here rewrites that.
        """

        summary = self.summary(reference=self.REPORT_DATE)
        detail = dated_acquisition_candidate(self.selected, self.TRADE_DATE)

        # The generic anchor, unchanged, still cannot bridge these two.
        self.assertIsNone(anchor_tier(detail.chunk, summary.chunk, self.HOLDER))

        result, payload = self._run((summary, detail), (summary,))

        # The resolver's own account is untouched: two events, one ambiguity.
        self.assertTrue(result.resolution.temporal_ambiguity)
        self.assertEqual(result.resolution.matching_event_count, 2)
        proven = [event for event in result.resolution.events
                  if event.transaction_method is not None]
        self.assertEqual(len(proven), 1)
        self.assertEqual(proven[0].acquisition_date, "2025-02-05")

        # The producer answers from the one acquisition among them.
        self.assertEqual(len(result.field_evidence), 1)
        field = result.field_evidence[0]
        self.assertEqual(field.field, ACQUISITION_UNIT_PRICE)
        self.assertIs(field.status, FieldStatus.UNAVAILABLE)
        self.assertEqual(field.doc_id, self.selected.doc_id)
        self.assertEqual(field.chunk_id, detail.chunk_id)
        self.assertEqual(field.table_id, "t-acquisition")
        self.assertEqual((field.row_start, field.row_end), (2, 2))

        # Answerability decides through the ordinary FieldEvidence contract.
        self.assertFalse(payload["think_trace"]["answerable"])
        answerability = payload["think_trace"]["answerability"]
        self.assertEqual(answerability["status"], "insufficient_evidence")
        self.assertEqual(answerability["unavailable_fields"],
                         [ACQUISITION_UNIT_PRICE])
        self.assertEqual(
            answerability["unavailable_evidence"][0]["chunk_id"], detail.chunk_id
        )

    def test_transaction_older_than_its_report_reads_a_stated_price(self) -> None:
        """The same shape with the price actually stated."""

        summary = self.summary(reference=self.REPORT_DATE)
        detail = dated_acquisition_candidate(
            self.selected, self.TRADE_DATE, price=self.PRICE
        )

        result, payload = self._run((summary, detail), (summary,))

        self.assertEqual(len(result.field_evidence), 1)
        field = result.field_evidence[0]
        self.assertIs(field.status, FieldStatus.AVAILABLE)
        self.assertEqual(field.value, self.PRICE)
        self.assertEqual(field.chunk_id, detail.chunk_id)
        self.assertEqual((field.row_start, field.row_end), (2, 2))
        # No number the position snapshot carries may stand in for the price.
        for neighbour in ("100", "2,000", "2,100", "1.00"):
            self.assertNotEqual(field.value, neighbour)
        self.assertTrue(payload["think_trace"]["answerable"])

    def test_baseline_served_order_survives_recovery(self) -> None:
        summary = self.summary()
        detail = acquisition_candidate(self.selected)

        result, payload = self._run((summary, detail), (summary,))

        served = [row["chunk_id"] for row in payload["retrieved_context"]]
        # The ranked summary keeps its place; the recovered row is appended.
        self.assertEqual(served[0], summary.chunk_id)
        self.assertIn(detail.chunk_id, served)
        self.assertLess(served.index(summary.chunk_id),
                        served.index(detail.chunk_id))
        self.assertEqual(len(result.field_evidence), 1)


def _event(
    *,
    method=None,
    date=None,
    shares=None,
    matches=True,
    conflict=False,
    reporter="가상지주",
):
    """One resolved event, in the shape the frozen resolver reports it.

    ``transaction_method`` is populated only for a row that proved its own
    acquisition, so a snapshot and a disposal both arrive carrying ``None`` --
    which is what these cases model.
    """

    return SimpleNamespace(
        reporter=reporter,
        transaction_method=method,
        acquisition_date=date,
        acquired_shares=shares,
        matches_query=matches,
        field_conflict=conflict,
        field_provenance={},
    )


def _resolution(*events, ambiguity=None):
    matching = [event for event in events if event.matches_query is True]
    return SimpleNamespace(
        events=tuple(events),
        temporal_ambiguity=(len(matching) > 1) if ambiguity is None else ambiguity,
        matching_event_count=len(matching),
    )


#: A proven acquisition, as the frozen parser reports one.
_ACQUIRED = "장내매수(+)"
#: An increase that explains nothing.  The frozen classifier reports only that
#: it is not an acquisition, never that it is a disposal.
_UNEXPLAINED = "기타(+)"


class AcquisitionEventSelectionTests(unittest.TestCase):
    """Which event may answer for an acquisition unit price.

    A filing states a position on the day it was filed and the transactions
    that moved it on the days they happened, so one filing legitimately yields
    a snapshot dated later than the acquisition it reports.  Ambiguity between
    those two is real and the resolver is right to report it -- but a snapshot
    carries no unit price and was never a rival for this field.
    """

    def test_a_snapshot_does_not_rival_the_one_acquisition(self) -> None:
        snapshot = _event()
        acquisition = _event(method=_ACQUIRED, date="2025-02-05", shares="1,000")

        resolution = _resolution(snapshot, acquisition)

        # The ambiguity the resolver reports is left exactly as it found it.
        self.assertTrue(resolution.temporal_ambiguity)
        self.assertIs(_selected_event(resolution), acquisition)

    def test_two_acquisitions_on_different_days_decline(self) -> None:
        first = _event(method=_ACQUIRED, date="2025-02-03", shares="1,000")
        second = _event(method=_ACQUIRED, date="2025-02-05", shares="2,000")

        self.assertIsNone(_selected_event(_resolution(_event(), first, second)))

    def test_two_acquisitions_on_one_day_still_decline(self) -> None:
        """Two events are two events; one date does not make them one."""

        first = _event(method=_ACQUIRED, date="2025-02-05", shares="1,000")
        second = _event(method=_ACQUIRED, date="2025-02-05", shares="2,000")

        self.assertIsNone(_selected_event(_resolution(first, second)))

    def test_a_disposal_does_not_compete(self) -> None:
        # The frozen resolver gives a disposal no method at all.
        disposal = _event()
        acquisition = _event(method=_ACQUIRED, date="2025-02-05", shares="1,000")

        self.assertIs(_selected_event(_resolution(disposal, acquisition)),
                      acquisition)

    def test_an_unclassifiable_method_fails_closed(self) -> None:
        """Not provably an acquisition is not provably harmless either."""

        unexplained = _event(method=_UNEXPLAINED, date="2025-02-04", shares="500")
        acquisition = _event(method=_ACQUIRED, date="2025-02-05", shares="1,000")

        self.assertIsNone(_selected_event(_resolution(unexplained, acquisition)))

    def test_a_position_only_resolution_declines(self) -> None:
        self.assertIsNone(_selected_event(_resolution(_event())))
        self.assertIsNone(_selected_event(_resolution()))

    def test_a_conflicted_acquisition_declines(self) -> None:
        conflicted = _event(method=_ACQUIRED, date="2025-02-05",
                            shares="1,000", conflict=True)

        self.assertIsNone(_selected_event(_resolution(conflicted)))

    def test_an_unmatched_acquisition_does_not_compete(self) -> None:
        unmatched = _event(method=_ACQUIRED, date="2025-02-01",
                           shares="9,000", matches=False)
        acquisition = _event(method=_ACQUIRED, date="2025-02-05", shares="1,000")

        self.assertIs(_selected_event(_resolution(unmatched, acquisition)),
                      acquisition)
        # ...and alone it answers nothing.
        self.assertIsNone(_selected_event(_resolution(unmatched)))

    def test_another_holders_event_does_not_compete(self) -> None:
        """A row the resolver did not match is not a rival for this field."""

        other_holder = _event(method=_ACQUIRED, date="2025-02-05",
                              shares="8,000", reporter="가상연금",
                              matches=False)
        acquisition = _event(method=_ACQUIRED, date="2025-02-05", shares="1,000")

        self.assertIs(_selected_event(_resolution(other_holder, acquisition)),
                      acquisition)

    def test_ordinary_holding_questions_never_reach_this_producer(self) -> None:
        """Only the acquisition unit price changed; nothing else asks here."""

        for question in (
            f"{ISSUER} 보유주식수는?",
            f"{ISSUER} 보유비율은?",
            f"{ISSUER} 증감주식수는?",
            f"{ISSUER} 기준일은?",
        ):
            with self.subTest(question=question):
                self.assertEqual(requested_holding_fields(question), ())
                self.assertEqual(
                    holding_field_evidence(
                        question=question,
                        resolution=_resolution(
                            _event(method=_ACQUIRED, date="2025-02-05",
                                   shares="1,000")
                        ),
                        evidence_items=(),
                    ),
                    (),
                )
