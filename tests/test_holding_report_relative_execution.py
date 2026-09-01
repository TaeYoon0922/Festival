from __future__ import annotations

import copy
import re
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.agent.orchestrator import AgentOrchestrator
from app.api.pipeline import AnswerPipeline, final_evidence, retrieved_context
from app.api.schemas import AnswerResponse, RetrievedContextItem
from app.reasoning.holding_correction_finality import (
    STATUS_AMBIGUOUS,
    STATUS_RESOLVED,
    CorrectionChain,
    HoldingCorrectionFinality,
)
from app.reasoning.holding_report_index import (
    AMBIGUOUS,
    CORRECTION_AMBIGUOUS,
    NO_INDEX,
    PREVIOUS_UNAVAILABLE,
    RESOLVED,
    STALE_INDEX,
    HoldingReportIndex,
    HoldingReportRecord,
    load_index,
)
from app.reasoning.holding_reporter import canonical_reporter_key
from app.reasoning.holding_report_relative_execution import (
    PROJECTION_CHUNK_AMBIGUOUS,
    PROJECTION_CHUNK_MISSING,
    PROVENANCE_KEY,
    HoldingReportRelativeExecution,
    repository_corpus_identity,
)
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


ISSUER = "01316245"
COMPANY = "효성중공업"
REPORTER = "국민연금공단"
IDENTITY = {
    "corpus_snapshot_id": "phase-3",
    "corpus_manifest_sha256": "phase-3",
    "source_holding_disclosure_count": 2,
}
SOURCE_REF = {"table_id": "t0012", "row_start": 3, "row_end": 3}


def report(
    *,
    doc_id: str,
    chunk_id: str,
    reference_date: str,
    receipt_date: str,
    previous_date: str | None = "20230125",
    before_shares: str | None = "555,510",
    before_ratio: str | None = "5.96",
    change_shares: str | None = "99,980",
    change_ratio: str | None = "1.07",
    change_direction: str | None = "increase",
    after_shares: str | None = "655,490",
    after_ratio: str | None = "7.03",
    is_correction: bool = False,
) -> HoldingReportRecord:
    return HoldingReportRecord(
        issuer_corp_code=ISSUER,
        reporter_key=REPORTER,
        raw_reporter=REPORTER,
        doc_id=doc_id,
        projection_chunk_id=chunk_id,
        reference_date=reference_date,
        receipt_date=receipt_date,
        previous_date=previous_date,
        before_shares=before_shares,
        before_ratio=before_ratio,
        change_shares=change_shares,
        change_ratio=change_ratio,
        change_direction=change_direction,
        after_shares=after_shares,
        after_ratio=after_ratio,
        is_correction=is_correction,
        report_nm="주식등의대량보유상황보고서(약식)",
        source_table_id="t0012",
        source_refs=(SOURCE_REF,),
    )


OLD = report(
    doc_id="holding_old",
    chunk_id="holding_old:report",
    reference_date="20240509",
    receipt_date="20240702",
    previous_date="20231228",
    before_shares="1,045,027",
    before_ratio="11.21",
    change_shares="93,878",
    change_ratio="1.00",
    after_shares="1,138,905",
    after_ratio="12.21",
)
NEW = report(
    doc_id="holding_new",
    chunk_id="holding_new:report",
    reference_date="20250728",
    receipt_date="20251001",
    previous_date="20240509",
    before_shares="1,138,905",
    before_ratio="12.21",
    change_shares="-100,989",
    change_ratio="-1.08",
    change_direction="decrease",
    after_shares="1,037,916",
    after_ratio="11.13",
)


def index_of(
    *records: HoldingReportRecord,
    identity: dict | None = None,
    finality: HoldingCorrectionFinality | None = None,
) -> HoldingReportIndex:
    return HoldingReportIndex(
        records,
        identity=identity or IDENTITY,
        complete=True,
        correction_finality_available=finality is None,
        correction_finality=finality,
    )


def candidate(record: HoldingReportRecord, **overrides) -> CandidateChunk:
    fields = {
        "보고자/보유자": record.raw_reporter,
        "기준일/보고일": record.reference_date,
        "직전 보유주식수": record.before_shares,
        "증감주식수": record.change_shares,
        "보유주식수": record.after_shares,
        "직전 보유비율": record.before_ratio,
        "보유비율": record.after_ratio,
        "증감비율": record.change_ratio,
        "변동방향": record.change_direction,
    }
    fields = {key: value for key, value in fields.items() if value is not None}
    chunk = {
        "chunk_id": record.projection_chunk_id,
        "doc_id": record.doc_id,
        "doc_group": "holding",
        "projection_type": "holding_report",
        "corp_code": record.issuer_corp_code,
        "corp_name": COMPANY,
        "report_nm": record.report_nm,
        "rcept_dt": record.receipt_date,
        "reference_date": record.reference_date,
        "reporter": record.raw_reporter,
        "projection_fields": fields,
        "projection_field_refs": {
            label: [dict(SOURCE_REF)] for label in fields
        },
        "source_table_id": "t0012",
        "source_table_ids": ["t0012"],
        "source_refs": [dict(SOURCE_REF)],
        "section_path": ["주식등의 대량보유상황보고서"],
        "content": " | ".join(f"{key}: {value}" for key, value in fields.items()),
        "retrieval_text": " ".join(str(value) for value in fields.values()),
    }
    chunk.update(overrides)
    return CandidateChunk(
        chunk_id=record.projection_chunk_id,
        doc_id=record.doc_id,
        chunk=chunk,
        metadata_match=MetadataMatch(),
    )


def ranked(value: CandidateChunk, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=value.chunk_id,
        doc_id=value.doc_id,
        bm25_score=10.0 / rank,
        rank=rank,
        metadata_match={},
    )


def plan(question: str):
    return QueryUnderstanding(
        company_resolver=lambda _question: {
            "corp_code": ISSUER,
            "corp_name": COMPANY,
            "listed_name": COMPANY,
        }
    ).understand(question)


def execution(query_plan, *values: CandidateChunk, results=None):
    rows = (
        list(results)
        if results is not None
        else [ranked(value, i) for i, value in enumerate(values, 1)]
    )
    return SimpleNamespace(plan=query_plan, chunks=list(values), results=rows)


def adapter(
    value: HoldingReportIndex | None,
    *,
    active: dict | None = None,
    backend=None,
) -> HoldingReportRelativeExecution:
    return HoldingReportRelativeExecution(
        index=value,
        document_backend=backend,
        chunk_backend=backend,
        active_corpus_identity=IDENTITY if active is None else active,
    )


class _HydrationBackend:
    def __init__(self, value: CandidateChunk | None) -> None:
        self.value = value
        self.document_calls: list[tuple[str, ...]] = []
        self.chunk_calls = 0

    def fetch_documents(self, doc_ids):
        self.document_calls.append(tuple(doc_ids))
        if self.value is None:
            return [
                CandidateDocument(
                    doc_id=str(doc_ids[0]), metadata={}, metadata_match=MetadataMatch()
                )
            ]
        return [
            CandidateDocument(
                doc_id=self.value.doc_id,
                metadata=dict(self.value.chunk),
                metadata_match=MetadataMatch(),
            )
        ]

    def get_candidate_chunks(self, documents):
        self.chunk_calls += 1
        list(documents)
        return [self.value] if self.value is not None else []


class GatingTests(unittest.TestCase):
    def test_intent_absent_keeps_the_existing_execution_unchanged(self) -> None:
        question = f"{REPORTER} {COMPANY} 보유주식수"
        query_plan = plan(question)
        existing = candidate(OLD)
        source = execution(query_plan, existing)
        before = copy.deepcopy((source.chunks, source.results))
        result = AgentOrchestrator(
            report_relative_execution=adapter(index_of(OLD, NEW))
        ).run(question, query_plan, source)

        self.assertIsNone(result.report_relative_execution)
        self.assertFalse(result.evidence_overridden)
        self.assertEqual((source.chunks, source.results), before)
        self.assertEqual(result.evidence_results, tuple(source.results))

    def test_plain_exact_date_is_not_stolen(self) -> None:
        question = f"{REPORTER} {COMPANY} 2024년 5월 9일 보유주식수"
        query_plan = plan(question)
        outcome = adapter(index_of(OLD, NEW)).adapt(
            question,
            query_plan,
            execution(query_plan, candidate(OLD)),
            routed_task_type="holding_event",
        )
        self.assertIsNone(outcome)

    def test_acquisition_plus_latest_uses_the_frozen_field_firewall(self) -> None:
        for wording in ("최신 보고 취득 주식수", "이번 보고 취득 주식수"):
            with self.subTest(wording=wording):
                question = f"{REPORTER} {COMPANY} {wording}"
                query_plan = plan(question)
                outcome = adapter(index_of(OLD, NEW)).adapt(
                    question,
                    query_plan,
                    execution(query_plan, candidate(OLD)),
                    routed_task_type="holding_event",
                )
                self.assertIsNone(outcome)

    def test_selected_context_is_never_promoted_to_latest(self) -> None:
        for wording in ("이번 보고 보유주식수", "직전보고 보유비율"):
            with self.subTest(wording=wording):
                question = f"{REPORTER} {COMPANY} {wording}"
                query_plan = plan(question)
                intent = query_plan.evidence["holding_report_relative"]
                self.assertEqual(intent["selector"], "selected_context")
                outcome = adapter(index_of(OLD, NEW)).adapt(
                    question,
                    query_plan,
                    execution(query_plan, candidate(OLD)),
                    routed_task_type="holding_event",
                )
                # Phase 3 owns the question -- leaving it to ranked retrieval
                # would answer it from whichever report happened to rank first.
                # No report was named, so the execution is authoritatively
                # empty rather than the latest one.
                self.assertIsNotNone(outcome)
                self.assertEqual(outcome.status, "unsupported_selector")
                self.assertFalse(outcome.resolved)
                self.assertEqual(outcome.chunks, ())
                self.assertEqual(outcome.results, ())
                self.assertFalse(outcome.report_execution.executable)

    def test_unbound_deictics_fail_closed_without_a_parsed_reporter(self) -> None:
        for wording in (
            "이번 보고 보유주식수",
            "금번 보고서 보유주식수",
            "직전보고 보유비율",
            "이전 보고 보유비율",
        ):
            with self.subTest(wording=wording):
                question = f"홍길동 {COMPANY} {wording}"
                query_plan = plan(question)
                intent = query_plan.evidence["holding_report_relative"]
                self.assertIsNone(query_plan.reporter)
                self.assertEqual(intent["selector"], "selected_context")

                outcome = adapter(index_of(OLD, NEW)).adapt(
                    question,
                    query_plan,
                    execution(query_plan, candidate(OLD), candidate(NEW)),
                    routed_task_type="holding_event",
                )

                self.assertIsNotNone(outcome)
                self.assertEqual(outcome.status, "unsupported_selector")
                self.assertFalse(outcome.resolved)
                self.assertEqual(outcome.chunks, ())
                self.assertEqual(outcome.results, ())

    def test_non_deictic_report_selectors_keep_the_frozen_contract(self) -> None:
        cases = (
            (f"{REPORTER} {COMPANY} 최신 보고 보유주식수", "latest", "current", NEW),
            (f"{REPORTER} {COMPANY} 현재 기준 보유주식수", "latest", "current", NEW),
            (
                f"{REPORTER} {COMPANY} 2024년 5월 9일 보고서의 보유주식수",
                "exact_reference_date",
                "current",
                OLD,
            ),
            (
                f"{REPORTER} {COMPANY} 2024년 7월 2일 접수된 보고서의 보유주식수",
                "exact_receipt_date",
                "current",
                OLD,
            ),
            (
                f"{REPORTER} {COMPANY} 최신 보고의 직전보고 보유비율",
                "latest",
                "previous",
                NEW,
            ),
        )
        for question, selector, role, selected in cases:
            with self.subTest(question=question):
                query_plan = plan(question)
                intent = query_plan.evidence["holding_report_relative"]
                self.assertEqual(intent["selector"], selector)
                self.assertEqual(intent["projection_role"], role)

                outcome = adapter(index_of(OLD, NEW)).adapt(
                    question,
                    query_plan,
                    execution(query_plan, candidate(OLD), candidate(NEW)),
                    routed_task_type="holding_event",
                )

                self.assertTrue(outcome.resolved)
                self.assertEqual(outcome.selected_chunk_id, selected.projection_chunk_id)

    def test_missing_reporter_and_comparison_firewall_stay_outside_phase_3(self) -> None:
        missing_question = f"{COMPANY} 최신 보고 보유주식수"
        missing = plan(missing_question)
        self.assertIsNone(
            adapter(index_of(OLD, NEW)).adapt(
                missing_question,
                missing,
                execution(missing, candidate(OLD)),
                routed_task_type="holding_event",
            )
        )

        question = f"{REPORTER} {COMPANY} 최신 보고 보유주식수"
        ordinary = plan(question)
        blocked = replace(
            ordinary,
            comparison={"type": "company_comparison"},
            evidence={**ordinary.evidence, "comparison_frame": "cross_company"},
        )
        self.assertIsNone(
            adapter(index_of(OLD, NEW)).adapt(
                question,
                blocked,
                execution(blocked, candidate(OLD)),
                routed_task_type="holding_event",
            )
        )

    def test_short_reporter_is_not_widened_to_the_index_identity(self) -> None:
        question = f"국민연금 {COMPANY} 최신 보고 보유주식수"
        query_plan = plan(question)
        outcome = adapter(index_of(OLD, NEW)).adapt(
            question,
            query_plan,
            execution(query_plan, candidate(OLD)),
            routed_task_type="holding_event",
        )
        self.assertIsNotNone(outcome)
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.status, "no_match")


class SuccessfulExecutionTests(unittest.TestCase):
    def _run(self, question: str, reports=(OLD, NEW)):
        query_plan = plan(question)
        old = candidate(OLD)
        new = candidate(NEW)
        source = execution(
            query_plan,
            old,
            new,
            results=[ranked(old, 1), ranked(new, 2)],
        )
        result = AgentOrchestrator(
            report_relative_execution=adapter(index_of(*reports))
        ).run(question, query_plan, source)
        return query_plan, source, result

    def test_latest_current_ignores_rank_one_and_orders_by_reference_date(self) -> None:
        question = f"{REPORTER} {COMPANY} 최신 보고 보유주식수"
        _plan, source, result = self._run(question)

        self.assertEqual(source.results[0].chunk_id, OLD.projection_chunk_id)
        self.assertTrue(result.report_relative_execution.resolved)
        self.assertEqual(result.evidence_results[0].chunk_id, NEW.projection_chunk_id)
        self.assertEqual(result.evidence_results[0].bm25_score, 0.0)
        self.assertFalse(
            result.evidence_results[0].metadata_match[PROVENANCE_KEY][
                "ranked_retrieval"
            ]
        )
        event = result.resolution.events[0]
        self.assertEqual(event.after_shares.normalized, 1_037_916)
        self.assertEqual(event.reference_date, "2025-07-28")

    def test_latest_uses_reference_date_not_receipt_date(self) -> None:
        late_receipt = replace(
            OLD,
            doc_id="holding_late_receipt",
            projection_chunk_id="holding_late_receipt:report",
            receipt_date="20260101",
        )
        question = f"{REPORTER} {COMPANY} 최신 보고 보유주식수"
        query_plan = plan(question)
        source = execution(query_plan, candidate(late_receipt), candidate(NEW))
        outcome = adapter(index_of(late_receipt, NEW)).adapt(
            question, query_plan, source, routed_task_type="holding_event"
        )
        self.assertTrue(outcome.resolved)
        self.assertEqual(outcome.selected_chunk_id, NEW.projection_chunk_id)

    def test_exact_reference_and_receipt_select_the_same_provenance(self) -> None:
        cases = (
            f"{REPORTER} {COMPANY} 2024년 5월 9일 보고서의 보유주식수",
            f"{REPORTER} {COMPANY} 2024년 7월 2일 접수된 보고서의 보유주식수",
        )
        for question in cases:
            with self.subTest(question=question):
                query_plan = plan(question)
                outcome = adapter(index_of(OLD, NEW)).adapt(
                    question,
                    query_plan,
                    execution(query_plan, candidate(NEW), candidate(OLD)),
                    routed_task_type="holding_event",
                )
                self.assertTrue(outcome.resolved)
                self.assertEqual(outcome.selected_chunk_id, OLD.projection_chunk_id)

    def test_previous_role_reads_the_selected_reports_own_previous_state(self) -> None:
        question = f"{REPORTER} {COMPANY} 최신 보고의 직전보고 보유비율"
        second_latest = replace(OLD, after_ratio="99.99")
        query_plan = plan(question)
        old = candidate(second_latest)
        new = candidate(NEW)
        source = execution(
            query_plan,
            old,
            new,
            results=[ranked(old, 1), ranked(new, 2)],
        )
        result = AgentOrchestrator(
            report_relative_execution=adapter(index_of(second_latest, NEW))
        ).run(question, query_plan, source)

        execution_result = result.report_relative_execution.report_execution
        self.assertEqual(execution_result.projection.role, "previous")
        self.assertEqual(execution_result.projection.values["reference_date"], "20240509")
        self.assertEqual(execution_result.projection.values["ratio"], "12.21")
        self.assertEqual(result.evidence_results[0].chunk_id, NEW.projection_chunk_id)
        event = result.resolution.events[0]
        self.assertEqual(event.before_ratio.normalized, 12.21)
        self.assertNotEqual(event.before_ratio.normalized, 99.99)

    def test_report_reference_date_never_becomes_an_acquisition_fact(self) -> None:
        question = (
            f"{REPORTER} {COMPANY} 2024년 5월 9일 보고서의 보유주식수"
        )
        query_plan = plan(question)
        old = candidate(OLD)
        result = AgentOrchestrator(
            report_relative_execution=adapter(index_of(OLD, NEW))
        ).run(question, query_plan, execution(query_plan, old))

        self.assertTrue(result.report_relative_execution.resolved)
        self.assertNotIn(
            "acquisition_date",
            result.report_relative_execution.requested_fields,
        )
        event = result.resolution.events[0]
        self.assertIsNone(event.acquisition_date)
        self.assertIsNone(event.acquired_shares)
        self.assertIsNone(event.transaction_method)

    def test_change_role_reads_the_selected_reports_change(self) -> None:
        question = (
            f"{REPORTER} {COMPANY} 최신 보고의 직전보고 대비 증감주식수"
        )
        _plan, _source, result = self._run(question)

        execution_result = result.report_relative_execution.report_execution
        self.assertEqual(execution_result.projection.role, "change")
        self.assertEqual(execution_result.projection.values["shares"], "-100,989")
        self.assertEqual(result.resolution.events[0].change_shares.normalized, -100_989)

    def test_citation_and_retrieved_context_are_the_selected_projection(self) -> None:
        question = f"{REPORTER} {COMPANY} 최신 보고 보유비율"
        _plan, source, result = self._run(question)
        citations = result.answer_draft.citations
        self.assertTrue(citations)
        self.assertEqual({value.chunk_id for value in citations}, {NEW.projection_chunk_id})
        self.assertEqual(citations[0].source_refs, (SOURCE_REF,))

        served = final_evidence(source, result)
        context = retrieved_context(served, 10)
        self.assertEqual([row["chunk_id"] for row in context], [NEW.projection_chunk_id])
        self.assertEqual(context[0]["source_refs"], [SOURCE_REF])
        RetrievedContextItem.model_validate(context[0])

    def test_exact_hydration_uses_identity_lookup_without_search(self) -> None:
        question = f"{REPORTER} {COMPANY} 최신 보고 보유주식수"
        query_plan = plan(question)
        old = candidate(OLD)
        selected = candidate(NEW)
        source = execution(query_plan, old, results=[ranked(old)])
        backend = _HydrationBackend(selected)
        result = AgentOrchestrator(
            report_relative_execution=adapter(
                index_of(OLD, NEW), backend=backend
            )
        ).run(question, query_plan, source)

        self.assertTrue(result.report_relative_execution.resolved)
        self.assertTrue(result.report_relative_execution.hydrated)
        self.assertEqual(backend.document_calls, [(NEW.doc_id,)])
        self.assertEqual(backend.chunk_calls, 1)
        served = final_evidence(source, result)
        self.assertEqual([value.chunk_id for value in served.chunks], [NEW.projection_chunk_id])
        self.assertEqual([value.chunk_id for value in served.results], [NEW.projection_chunk_id])


class FailureAuthorityTests(unittest.TestCase):
    def _run_failure(self, value: HoldingReportIndex | None, question: str):
        query_plan = plan(question)
        old = candidate(OLD)
        source = execution(query_plan, old, results=[ranked(old)])
        result = AgentOrchestrator(
            report_relative_execution=adapter(value)
        ).run(question, query_plan, source)
        return source, result

    def test_no_index_is_authoritative(self) -> None:
        question = f"{REPORTER} {COMPANY} 최신 보고 보유주식수"
        source, result = self._run_failure(None, question)
        self.assertEqual(result.report_relative_execution.status, NO_INDEX)
        self.assertFalse(result.answer_draft.answerable)
        self.assertEqual(result.evidence_results, ())
        self.assertEqual(retrieved_context(final_evidence(source, result), 10), [])

    def test_same_date_and_same_document_projection_ambiguity_are_authoritative(self) -> None:
        other_doc = replace(
            OLD,
            doc_id="holding_same_date",
            projection_chunk_id="holding_same_date:report",
        )
        other_projection = replace(
            OLD,
            projection_chunk_id="holding_old:second_projection",
        )
        question = (
            f"{REPORTER} {COMPANY} 2024년 5월 9일 보고서의 보유주식수"
        )
        for value in (
            index_of(OLD, other_doc),
            index_of(OLD, other_projection),
        ):
            with self.subTest(records=value.record_count):
                source, result = self._run_failure(value, question)
                self.assertEqual(result.report_relative_execution.status, AMBIGUOUS)
                self.assertEqual(result.evidence_results, ())
                self.assertEqual(retrieved_context(final_evidence(source, result), 10), [])

    def test_stale_index_is_authoritative(self) -> None:
        question = f"{REPORTER} {COMPANY} 최신 보고 보유주식수"
        query_plan = plan(question)
        source = execution(query_plan, candidate(OLD), results=[ranked(candidate(OLD))])
        result = AgentOrchestrator(
            report_relative_execution=adapter(
                index_of(OLD, NEW), active={"corpus_manifest_sha256": "moved"}
            )
        ).run(question, query_plan, source)
        self.assertEqual(result.report_relative_execution.status, STALE_INDEX)
        self.assertEqual(result.evidence_results, ())

    def test_previous_unavailable_is_authoritative(self) -> None:
        first = replace(
            NEW,
            previous_date=None,
            before_shares=None,
            before_ratio=None,
        )
        question = f"{REPORTER} {COMPANY} 최신 보고의 직전보고 보유비율"
        source, result = self._run_failure(index_of(first), question)
        self.assertEqual(result.report_relative_execution.status, PREVIOUS_UNAVAILABLE)
        self.assertEqual(result.evidence_results, ())
        self.assertEqual(retrieved_context(final_evidence(source, result), 10), [])

    def test_missing_or_duplicate_hydration_never_falls_back(self) -> None:
        question = f"{REPORTER} {COMPANY} 최신 보고 보유주식수"
        query_plan = plan(question)
        old = candidate(OLD)
        source = execution(query_plan, old, results=[ranked(old)])
        missing_backend = _HydrationBackend(None)
        missing = AgentOrchestrator(
            report_relative_execution=adapter(
                index_of(OLD, NEW), backend=missing_backend
            )
        ).run(question, query_plan, source)
        self.assertEqual(missing.report_relative_execution.status, PROJECTION_CHUNK_MISSING)
        self.assertEqual(missing.evidence_results, ())

        duplicate_source = execution(
            query_plan,
            candidate(NEW),
            candidate(NEW),
            results=[ranked(old)],
        )
        duplicate = AgentOrchestrator(
            report_relative_execution=adapter(index_of(OLD, NEW))
        ).run(question, query_plan, duplicate_source)
        self.assertEqual(
            duplicate.report_relative_execution.status,
            PROJECTION_CHUNK_AMBIGUOUS,
        )
        self.assertEqual(duplicate.evidence_results, ())


class CorrectionFinalityTests(unittest.TestCase):
    def _source(self, status: str, *, identity=IDENTITY):
        return HoldingCorrectionFinality(
            [
                CorrectionChain(
                    group_id="g1",
                    root_doc_id="holding_original",
                    members=("holding_original", "holding_correction"),
                    final_doc_id=(
                        "holding_correction" if status == STATUS_RESOLVED else None
                    ),
                    status=status,
                )
            ],
            identity=identity,
            complete=True,
        )

    def _records(self):
        original = replace(
            OLD,
            doc_id="holding_original",
            projection_chunk_id="holding_original:report",
            is_correction=False,
        )
        correction = replace(
            OLD,
            doc_id="holding_correction",
            projection_chunk_id="holding_correction:report",
            after_shares="1,200,000",
            is_correction=True,
        )
        return original, correction

    def test_b3_resolved_pair_selects_only_the_final_projection(self) -> None:
        original, correction = self._records()
        value = index_of(
            original,
            correction,
            finality=self._source(STATUS_RESOLVED),
        )
        question = (
            f"{REPORTER} {COMPANY} 2024년 5월 9일 보고서의 보유주식수"
        )
        query_plan = plan(question)
        outcome = adapter(value).adapt(
            question,
            query_plan,
            execution(query_plan, candidate(original), candidate(correction)),
            routed_task_type="holding_event",
        )
        self.assertTrue(outcome.resolved)
        self.assertEqual(outcome.selected_chunk_id, correction.projection_chunk_id)

    def test_ambiguous_and_stale_finality_are_authoritative(self) -> None:
        original, correction = self._records()
        question = (
            f"{REPORTER} {COMPANY} 2024년 5월 9일 보고서의 보유주식수"
        )
        sources = (
            self._source(STATUS_AMBIGUOUS),
            self._source(
                STATUS_RESOLVED,
                identity={
                    **IDENTITY,
                    "corpus_manifest_sha256": "another-corpus",
                },
            ),
        )
        for source in sources:
            with self.subTest(source=source.identity):
                value = index_of(original, correction, finality=source)
                query_plan = plan(question)
                outcome = adapter(value).adapt(
                    question,
                    query_plan,
                    execution(query_plan, candidate(original), candidate(correction)),
                    routed_task_type="holding_event",
                )
                self.assertEqual(outcome.status, CORRECTION_AMBIGUOUS)
                self.assertEqual(outcome.results, ())


class RepositoryWiringTests(unittest.TestCase):
    def test_production_lane_contains_no_p0d2_or_case_specific_shortcut(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = "\n".join(
            (root / path).read_text(encoding="utf-8")
            for path in (
                "app/reasoning/holding_report_relative_execution.py",
                "app/agent/orchestrator.py",
            )
        )
        compact = source.casefold()
        for forbidden in (
            "p0d2",
            "p0-d.2",
            "h01",
            "hx02",
            "gold",
            "삼성전자",
            "에스엠",
            "효성중공업",
            "국민연금",
        ):
            self.assertNotIn(forbidden, compact)
        self.assertIsNone(re.search(r"(?<!\d)\d{8}(?!\d)", source))
        self.assertIsNone(re.search(r"holding_\d{8,}", source))

    def test_tracked_index_matches_independently_recomputed_manifest_identity(self) -> None:
        value = load_index(
            "data/corpus/holding_report_index.json",
            finality_path="data/corpus/holding_correction_finality.json",
        )
        self.assertIsNotNone(value)
        self.assertTrue(value.matches_corpus(repository_corpus_identity()))
        self.assertEqual(value.correction_finality_status, "attached")

    def test_natural_language_pipeline_keeps_the_public_schema_and_selected_context(self) -> None:
        question = f"{REPORTER} {COMPANY} 최신 보고 보유주식수"
        query_plan = plan(question)
        old = candidate(OLD)
        new = candidate(NEW)
        source = execution(
            query_plan,
            old,
            new,
            results=[ranked(old, 1), ranked(new, 2)],
        )

        class _Understanding:
            def understand(self, _question, *, top_k):
                self.top_k = top_k
                return query_plan

        class _Executor:
            def execute(self, supplied):
                self.plan = supplied
                return source

        pipeline = AnswerPipeline(
            understanding=_Understanding(),
            executor=_Executor(),
            orchestrator=AgentOrchestrator(
                report_relative_execution=adapter(index_of(OLD, NEW))
            ),
        )
        payload = AnswerResponse.model_validate(
            pipeline.answer("PHASE3-E2E", question)
        ).model_dump()

        self.assertEqual(
            [row["chunk_id"] for row in payload["retrieved_context"]],
            [NEW.projection_chunk_id],
        )
        self.assertIn(NEW.projection_chunk_id, payload["answer"])
        self.assertNotIn(OLD.projection_chunk_id, payload["answer"])
        self.assertIn(
            "holding_report_relative_execution",
            payload["think_trace"]["stages"],
        )

    def test_natural_language_selected_context_fails_closed(self) -> None:
        question = f"{REPORTER} {COMPANY} 이번 보고 보유주식수"
        query_plan = plan(question)
        old = candidate(OLD)
        new = candidate(NEW)
        source = execution(
            query_plan,
            old,
            new,
            results=[ranked(old, 1), ranked(new, 2)],
        )

        class _Understanding:
            def understand(self, _question, *, top_k):
                self.top_k = top_k
                return query_plan

        class _Executor:
            def execute(self, supplied):
                self.plan = supplied
                return source

        pipeline = AnswerPipeline(
            understanding=_Understanding(),
            executor=_Executor(),
            orchestrator=AgentOrchestrator(
                report_relative_execution=adapter(index_of(OLD, NEW))
            ),
        )
        payload = AnswerResponse.model_validate(
            pipeline.answer("PHASE3-E2E-SELECTED", question)
        ).model_dump()

        # The ranked pool held both reports.  Neither may reach the public
        # evidence, because "이번 보고" named no report for Phase 3 to select.
        self.assertEqual(payload["retrieved_context"], [])
        self.assertIn(
            "holding_report_relative_execution",
            payload["think_trace"]["stages"],
        )
        self.assertEqual(payload["think_trace"]["selected_evidence_count"], 0)
        self.assertIs(payload["think_trace"]["answerable"], False)

    def test_unparsed_reporter_deictics_never_expose_ranked_evidence(self) -> None:
        for wording in (
            "이번 보고 보유주식수",
            "금번 보고서 보유주식수",
            "직전보고 보유비율",
            "이전 보고 보유비율",
        ):
            with self.subTest(wording=wording):
                question = f"홍길동 {COMPANY} {wording}"
                query_plan = plan(question)
                old = candidate(OLD)
                new = candidate(NEW)
                source = execution(
                    query_plan,
                    old,
                    new,
                    results=[ranked(old, 1), ranked(new, 2)],
                )

                class _Understanding:
                    def understand(self, _question, *, top_k):
                        return query_plan

                class _Executor:
                    def execute(self, supplied):
                        return source

                pipeline = AnswerPipeline(
                    understanding=_Understanding(),
                    executor=_Executor(),
                    orchestrator=AgentOrchestrator(
                        report_relative_execution=adapter(index_of(OLD, NEW))
                    ),
                )
                payload = AnswerResponse.model_validate(
                    pipeline.answer("PHASE3-DEICTIC", question)
                ).model_dump()

                self.assertIsNone(query_plan.reporter)
                self.assertEqual(payload["retrieved_context"], [])
                self.assertIn(
                    "holding_report_relative_execution",
                    payload["think_trace"]["stages"],
                )
                self.assertEqual(payload["think_trace"]["selected_evidence_count"], 0)
                self.assertIs(payload["think_trace"]["answerable"], False)


if __name__ == "__main__":
    unittest.main()


class NaturalReporterEntersPhase3Tests(unittest.TestCase):
    """A naturally named holder supplies the identity this lane already needed.

    The selector, the index, the ordering and the projection are all unchanged
    here.  What changes upstream is only that ``plan.reporter`` is populated for
    a holder the company universe cannot canonicalize, which is the one input
    the lane was missing.  Nothing in this module is modified to make these
    pass.
    """

    HOLDER = "가상보유인"

    def records(self):
        older = replace(
            OLD, reporter_key=self.HOLDER, raw_reporter=self.HOLDER,
            doc_id="holding_natural_old",
            projection_chunk_id="holding_natural_old:report",
        )
        newer = replace(
            NEW, reporter_key=self.HOLDER, raw_reporter=self.HOLDER,
            doc_id="holding_natural_new",
            projection_chunk_id="holding_natural_new:report",
        )
        return older, newer

    def test_a_natural_holder_reaches_the_authoritative_lane(self) -> None:
        older, newer = self.records()
        question = f"{COMPANY}에 대한 {self.HOLDER}의 최신 보고 보유주식수는?"
        query_plan = plan(question)

        # The upstream fix, stated as the precondition this lane depends on.
        self.assertEqual(query_plan.reporter, self.HOLDER)

        outcome = adapter(index_of(older, newer)).adapt(
            question,
            query_plan,
            # Rank order is deliberately the wrong way round: the lane must
            # answer by enumeration, not from what retrieval put first.
            execution(query_plan, candidate(older), candidate(newer)),
            routed_task_type="holding_event",
        )

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.status, RESOLVED)
        self.assertTrue(outcome.resolved)
        self.assertEqual(outcome.report_execution.record.doc_id, newer.doc_id)
        self.assertEqual(outcome.selected_chunk_id, newer.projection_chunk_id)

    def test_the_role_noun_shape_reaches_the_same_report(self) -> None:
        older, newer = self.records()
        question = (
            f"{COMPANY} 최대주주 {self.HOLDER}의 가장 최근 보고 기준 보유주식수는?"
        )
        query_plan = plan(question)

        self.assertEqual(query_plan.reporter, self.HOLDER)

        outcome = adapter(index_of(older, newer)).adapt(
            question,
            query_plan,
            execution(query_plan, candidate(older), candidate(newer)),
            routed_task_type="holding_event",
        )

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.status, RESOLVED)
        self.assertEqual(outcome.report_execution.record.doc_id, newer.doc_id)

    def test_an_issuer_without_a_named_holder_stays_outside_phase_3(self) -> None:
        older, newer = self.records()
        question = f"{COMPANY} 최신 보고 보유주식수는?"
        query_plan = plan(question)

        self.assertIsNone(query_plan.reporter)
        self.assertIsNone(
            adapter(index_of(older, newer)).adapt(
                question,
                query_plan,
                execution(query_plan, candidate(older)),
                routed_task_type="holding_event",
            )
        )

    def test_a_natural_surface_still_has_to_match_the_index_identity(self) -> None:
        """Extraction is not identity.  Strict canonical matching still decides."""

        older, newer = self.records()
        question = f"{COMPANY}에 대한 가상보유{'인이다'}의 최신 보고 보유주식수는?"
        query_plan = plan(question)

        self.assertIsNotNone(query_plan.reporter)
        self.assertNotEqual(
            canonical_reporter_key(query_plan.reporter),
            canonical_reporter_key(self.HOLDER),
        )

        outcome = adapter(index_of(older, newer)).adapt(
            question,
            query_plan,
            execution(query_plan, candidate(older), candidate(newer)),
            routed_task_type="holding_event",
        )

        self.assertIsNotNone(outcome)
        self.assertNotEqual(outcome.status, RESOLVED)
        self.assertFalse(outcome.resolved)
        self.assertEqual(outcome.results, ())
