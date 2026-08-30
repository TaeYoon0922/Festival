"""Contract-D materialisation for ordinary exact-date holding questions.

The fixtures deliberately separate the three proofs: a served text chunk says
the document is relevant, the report index names one event, and an unserved raw
table identifies the selected projection's physical rows.  None is sufficient
without the other two.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.agent.orchestrator import AgentOrchestrator
from app.api.app import create_app
from app.api.pipeline import AnswerPipeline
from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.holding_correction_finality import (
    STATUS_RESOLVED,
    CorrectionChain,
    HoldingCorrectionFinality,
)
from app.reasoning.holding_evidence_coverage import (
    RESCUE_MODE_CONTRACT_D,
    RESCUE_MODE_SERVED_ANCHOR,
    STATUS_RESCUED,
    assess,
)
from app.reasoning.holding_event_resolver import _FIELD_LABELS
from app.reasoning.holding_report_index import (
    HoldingReportIndex,
    HoldingReportRecord,
)
from app.reasoning.holding_report_relative import (
    ROLE_CURRENT,
    SELECTOR_EXACT_RECEIPT_DATE,
    SELECTOR_EXACT_REFERENCE_DATE,
    SELECTOR_LATEST,
    SELECTOR_SELECTED_CONTEXT,
)
from app.reasoning.holding_report_relative_execution import (
    HoldingReportRelativeExecution,
)
from app.reasoning.holding_reporter import canonical_reporter_key
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


ISSUER = "Fixture Issuer"
ISSUER_CODE = "00010001"
REPORTER = "Fixture Holder"
OTHER_REPORTER = "Other Holder"
DOC = "fixture_document"
OTHER_DOC = "other_document"
REFERENCE_DATE = "2024-02-03"
REFERENCE_DIGITS = "20240203"
TABLE = "fixture_table"
IDENTITY = {
    "corpus_snapshot_id": "fixture_snapshot",
    "corpus_manifest_sha256": "fixture_manifest",
    "source_holding_disclosure_count": 3,
}
GENERIC_QUERY = (
    "\uc8fc\uc2dd\uc744 2024\ub144 2\uc6d4 3\uc77c \uae30\uc900\uc73c\ub85c "
    "\uc5bc\ub9c8\ub098 \ub4e4\uace0 \uc788\uc5b4?"
)


def label(field: str) -> str:
    return _FIELD_LABELS[field][0]


def refs(*spans: tuple[int, int], table: str = TABLE) -> list[dict[str, object]]:
    return [
        {"table_id": table, "row_start": start, "row_end": end}
        for start, end in spans
    ]


def record(
    chunk_id: str = "selected_projection",
    *,
    doc_id: str = DOC,
    reporter: str = REPORTER,
    reference_date: str = REFERENCE_DIGITS,
    is_correction: bool = False,
) -> HoldingReportRecord:
    return HoldingReportRecord(
        issuer_corp_code=ISSUER_CODE,
        reporter_key=canonical_reporter_key(reporter),
        raw_reporter=reporter,
        doc_id=doc_id,
        projection_chunk_id=chunk_id,
        reference_date=reference_date,
        receipt_date="20240205",
        after_shares="1,234,567",
        after_ratio="12.34",
        is_correction=is_correction,
        source_table_id=TABLE,
        source_refs=tuple(refs((2, 2), (3, 3), (4, 4))),
    )


def projection(
    value: HoldingReportRecord,
    *,
    spans: tuple[tuple[int, int], ...] = ((2, 2), (3, 3), (4, 4)),
    table: str = TABLE,
) -> CandidateChunk:
    fields = {
        label("reporter"): value.raw_reporter,
        label("reference_date"): (
            f"{value.reference_date[:4]}.{value.reference_date[4:6]}."
            f"{value.reference_date[6:8]}"
        ),
        label("after_shares"): value.after_shares,
        label("after_ratio"): value.after_ratio,
    }
    event_refs = refs(*spans, table=table)
    chunk = {
        "chunk_id": value.projection_chunk_id,
        "doc_id": value.doc_id,
        "doc_group": "holding",
        "corp_code": value.issuer_corp_code,
        "corp_name": ISSUER,
        "chunk_type": "table_projection",
        "projection_type": "holding_report",
        "is_indexable": True,
        "reporter": value.raw_reporter,
        "reference_date": fields[label("reference_date")],
        "projection_fields": fields,
        "projection_field_refs": {
            field_label: [dict(item) for item in event_refs]
            for field_label in fields
        },
        "source_refs": [dict(item) for item in event_refs],
        "content": " ".join(f"[{key}] {item}" for key, item in fields.items()),
        "retrieval_text": " ".join(str(item) for item in fields.values()),
        "section_path": ["fixture holding report"],
    }
    return CandidateChunk(
        value.projection_chunk_id, value.doc_id, chunk, MetadataMatch()
    )


def raw_table(
    chunk_id: str = "raw_counterpart",
    *,
    doc_id: str = DOC,
    spans: tuple[tuple[int, int], ...] = ((2, 4),),
    table: str = TABLE,
    is_indexable: bool = True,
) -> CandidateChunk:
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": "holding",
        "chunk_type": "table",
        "projection_type": None,
        "is_indexable": is_indexable,
        "source_refs": refs(*spans, table=table),
        "content": "raw holding rows",
        "retrieval_text": "raw holding rows",
        "section_path": ["fixture holding report"],
    }
    return CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch())


def served_document(
    chunk_id: str = "served_document_text", *, doc_id: str = DOC
) -> CandidateChunk:
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": "holding",
        "chunk_type": "text",
        "projection_type": None,
        "is_indexable": True,
        "source_refs": [],
        "content": "relevant filing text",
        "retrieval_text": "relevant filing text",
        "section_path": ["fixture overview"],
    }
    return CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch())


def ranked(candidate: CandidateChunk, rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        candidate.chunk_id,
        candidate.doc_id,
        1.0,
        rank,
        MetadataMatch().to_dict(),
    )


def plan(
    *,
    metric: str | None = None,
    raw_query: str = "current holding",
    period: QueryPeriod | None = None,
    evidence: dict[str, object] | None = None,
    comparison: object = None,
    task_type: str = "holding_change",
) -> QueryPlan:
    return QueryPlan(
        query="current holding",
        raw_query=raw_query,
        companies=(ISSUER,),
        corp_codes=(ISSUER_CODE,),
        task_type=task_type,
        metric=metric,
        reporter=REPORTER,
        disclosure_route=("holding",),
        period=period
        or QueryPeriod(
            year=2024,
            from_date=REFERENCE_DATE,
            to_date=REFERENCE_DATE,
            period_type="holding_reference_date",
        ),
        evidence=evidence or {},
        comparison=comparison,
    )


def index_of(
    *records: HoldingReportRecord,
    complete: bool = True,
    identity: dict[str, object] | None = None,
    correction_finality_available: bool = True,
    correction_finality: HoldingCorrectionFinality | None = None,
) -> HoldingReportIndex:
    return HoldingReportIndex(
        records,
        complete=complete,
        identity=IDENTITY if identity is None else identity,
        correction_finality_available=correction_finality_available,
        correction_finality=correction_finality,
    )


def outcome(
    query_plan: QueryPlan,
    chunks: list[CandidateChunk],
    results: list[RetrievalResult],
    report_index: HoldingReportIndex | None,
    *,
    active_identity: dict[str, object] | None = IDENTITY,
    ordinary_lane: bool = True,
):
    return assess(
        str(query_plan.raw_query),
        query_plan,
        chunks,
        results,
        routed_task_type="holding_event",
        report_index=report_index,
        active_corpus_identity=active_identity,
        ordinary_lane=ordinary_lane,
    )


class ContractDFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.record = record()
        self.index = index_of(self.record)
        self.served = served_document()
        self.projection = projection(self.record)
        self.raw = raw_table()
        self.chunks = [self.served, self.raw, self.projection]
        self.results = [ranked(self.served)]

    def assess(self, query_plan: QueryPlan, **overrides):
        return outcome(
            query_plan,
            overrides.get("chunks", self.chunks),
            overrides.get("results", self.results),
            overrides.get("report_index", self.index),
            active_identity=overrides.get("active_identity", IDENTITY),
            ordinary_lane=overrides.get("ordinary_lane", True),
        )

    def orchestrate(self, query_plan: QueryPlan):
        phase_three = HoldingReportRelativeExecution(
            index=self.index,
            active_corpus_identity=IDENTITY,
        )
        orchestrator = AgentOrchestrator(
            report_relative_execution=phase_three,
            holding_report_index=self.index,
            active_corpus_identity=IDENTITY,
        )
        execution = SimpleNamespace(
            plan=query_plan,
            chunks=list(self.chunks),
            results=list(self.results),
        )
        return orchestrator.run(str(query_plan.raw_query), query_plan, execution)


class PositiveMaterializationTests(ContractDFixture):
    def assert_answered(self, query_plan: QueryPlan, expected: tuple[str, ...]) -> None:
        result = self.orchestrate(query_plan)

        self.assertEqual(result.holding_coverage.status, STATUS_RESCUED)
        self.assertEqual(result.holding_coverage.requested, expected)
        self.assertEqual(
            [item.chunk_id for item in result.holding_coverage.selected],
            [self.record.projection_chunk_id],
        )
        self.assertIn("holding_evidence_coverage", result.execution_trace)
        self.assertNotIn(
            "holding_report_relative_execution", result.execution_trace
        )
        self.assertEqual(
            result.holding_coverage.rescue_mode, RESCUE_MODE_CONTRACT_D
        )
        self.assertTrue(result.answer_draft.answerable)
        self.assertTrue(result.resolution.events)
        event = result.resolution.events[0]
        if "after_shares" in expected:
            self.assertEqual(event.after_shares.raw, self.record.after_shares)
        if "after_ratio" in expected:
            self.assertEqual(event.after_ratio.raw, self.record.after_ratio)

    def test_explicit_shares(self) -> None:
        self.assert_answered(plan(metric="holding_shares"), ("after_shares",))

    def test_explicit_ratio(self) -> None:
        self.assert_answered(plan(metric="holding_ratio"), ("after_ratio",))

    def test_generic_current_state_pair(self) -> None:
        query_plan = plan(
            raw_query=GENERIC_QUERY,
            evidence={"holding_ownership_intent": "company_has_company_shares"},
        )
        self.assertIsNone(query_plan.metric)
        self.assert_answered(query_plan, ("after_shares", "after_ratio"))

    def test_first_report_current_state_succeeds_but_previous_does_not(self) -> None:
        current = self.assess(plan(metric="holding_shares"))
        previous = self.assess(
            plan(evidence={"requested_holding_fields": ["before_shares"]})
        )

        self.assertTrue(current.rescued)
        self.assertFalse(previous.rescued)
        self.assertNotIn(self.record.projection_chunk_id,
                         [item.chunk_id for item in previous.results])

    def test_promotion_keeps_result_cap_rank_and_existing_provenance(self) -> None:
        result = self.assess(plan(metric="holding_shares"))

        self.assertEqual(len(result.results), len(self.results))
        self.assertEqual([item.rank for item in result.results], [1])
        self.assertEqual(result.displaced, (self.served.chunk_id,))
        promoted = result.results[0]
        self.assertEqual(
            dict(promoted.metadata_match)["holding_evidence_coverage"],
            {"selected_for": "holding_field_coverage"},
        )


class ThreeProofNegativeTests(ContractDFixture):
    def assert_closed(self, value) -> None:
        self.assertFalse(value.rescued)
        self.assertNotIn(
            self.record.projection_chunk_id,
            [item.chunk_id for item in value.results],
        )

    def test_selected_document_must_be_served(self) -> None:
        other = served_document("other_text", doc_id=OTHER_DOC)
        self.assert_closed(
            self.assess(
                plan(metric="holding_shares"),
                chunks=[other, self.raw, self.projection],
                results=[ranked(other)],
            )
        )

    def test_zero_exact_index_match(self) -> None:
        self.assert_closed(
            self.assess(
                replace(
                    plan(metric="holding_shares"),
                    period=QueryPeriod(
                        year=2024,
                        from_date="2024-02-04",
                        to_date="2024-02-04",
                        period_type="holding_reference_date",
                    ),
                )
            )
        )

    def test_ambiguous_exact_index_match(self) -> None:
        tied = replace(
            self.record,
            doc_id="tied_document",
            projection_chunk_id="tied_projection",
        )
        self.assert_closed(
            self.assess(
                plan(metric="holding_shares"),
                report_index=index_of(self.record, tied),
            )
        )

    def test_unresolved_correction_finality(self) -> None:
        correction = replace(self.record, is_correction=True)
        unresolved = index_of(
            correction,
            correction_finality_available=False,
        )
        self.assert_closed(
            self.assess(
                plan(metric="holding_shares"), report_index=unresolved
            )
        )

    def test_stale_incomplete_or_missing_index(self) -> None:
        cases = (
            {"active_identity": {**IDENTITY, "corpus_snapshot_id": "other"}},
            {"report_index": index_of(self.record, complete=False)},
            {"report_index": index_of(self.record, identity={})},
            {"report_index": None},
            {"active_identity": None},
        )
        for case in cases:
            with self.subTest(case=case):
                self.assert_closed(
                    self.assess(plan(metric="holding_shares"), **case)
                )

    def test_selected_projection_must_exist_uniquely(self) -> None:
        for chunks in (
            [self.served, self.raw],
            [self.served, self.raw, self.projection, self.projection],
        ):
            with self.subTest(count=len(chunks)):
                self.assert_closed(
                    self.assess(plan(metric="holding_shares"), chunks=chunks)
                )

    def test_multiple_pool_projections_for_selected_event_identity_are_rejected(self) -> None:
        duplicate_record = replace(
            self.record, projection_chunk_id="duplicate_event_projection"
        )
        duplicate = projection(
            duplicate_record, spans=((20, 20),)
        )
        duplicate_raw = raw_table(
            "duplicate_event_raw", spans=((20, 20),)
        )
        self.assert_closed(
            self.assess(
                plan(metric="holding_shares"),
                chunks=[
                    self.served,
                    self.raw,
                    duplicate_raw,
                    self.projection,
                    duplicate,
                ],
            )
        )

    def test_exact_raw_counterpart_must_exist_uniquely_and_be_indexable(self) -> None:
        duplicate = raw_table("duplicate_raw")
        cases = (
            [self.served, self.projection],
            [self.served, self.raw, duplicate, self.projection],
            [self.served, raw_table(is_indexable=False), self.projection],
        )
        for chunks in cases:
            with self.subTest(raw_count=len(chunks) - 2):
                self.assert_closed(
                    self.assess(plan(metric="holding_shares"), chunks=chunks)
                )

    def test_non_exact_raw_relations_are_rejected(self) -> None:
        cases = {
            "containment": raw_table(spans=((1, 10),)),
            "partial_overlap": raw_table(spans=((3, 6),)),
            "adjacency": raw_table(spans=((5, 7),)),
            "same_table_disjoint": raw_table(spans=((10, 12),)),
            "different_table": raw_table(table="other_table"),
            "different_document": raw_table(doc_id=OTHER_DOC),
        }
        for relation, raw in cases.items():
            with self.subTest(relation=relation):
                self.assert_closed(
                    self.assess(
                        plan(metric="holding_shares"),
                        chunks=[self.served, raw, self.projection],
                    )
                )

    def test_raw_rows_must_map_only_to_the_selected_projection(self) -> None:
        other_record = replace(
            self.record,
            projection_chunk_id="other_projection",
            raw_reporter=OTHER_REPORTER,
            reporter_key=canonical_reporter_key(OTHER_REPORTER),
        )
        other_projection = projection(other_record)
        self.assert_closed(
            self.assess(
                plan(metric="holding_shares"),
                chunks=[self.served, self.raw, self.projection, other_projection],
            )
        )


class MutationSensitiveGuardTests(ContractDFixture):
    def assert_closed(self, query_plan, **overrides) -> None:
        value = self.assess(query_plan, **overrides)
        self.assertFalse(value.rescued)
        self.assertIsNone(value.rescue_mode)

    def test_multiple_issuers_are_the_only_blocker(self) -> None:
        query_plan = plan(metric="holding_shares")
        attributes = dict(vars(query_plan))
        attributes.update(
            corp_code=ISSUER_CODE,
            corp_codes=(ISSUER_CODE, "00010002"),
        )
        multi_issuer = SimpleNamespace(**attributes)
        multi_issuer.to_dict = query_plan.to_dict

        self.assert_closed(multi_issuer)

    def test_reporter_is_required_even_with_a_preselected_event(self) -> None:
        selection = self.index.select_report(
            ISSUER_CODE,
            REPORTER,
            SELECTOR_EXACT_REFERENCE_DATE,
            reference_date=REFERENCE_DATE,
            active_corpus_identity=IDENTITY,
        )
        self.assertTrue(selection.resolved)
        preselected_index = SimpleNamespace(
            identity=IDENTITY,
            select_report=lambda *args, **kwargs: selection,
        )

        self.assert_closed(
            replace(plan(metric="holding_shares"), reporter=None),
            report_index=preselected_index,
        )

    def test_reporter_mismatch_is_refused_by_exact_index_selection(self) -> None:
        self.assert_closed(
            replace(plan(metric="holding_shares"), reporter=OTHER_REPORTER)
        )

    def test_forbidden_field_is_blocked_even_when_projection_can_supply_it(self) -> None:
        chunk = dict(self.projection.chunk)
        fields = dict(chunk["projection_fields"])
        field_refs = {
            key: [dict(item) for item in values]
            for key, values in chunk["projection_field_refs"].items()
        }
        forbidden_label = label("before_shares")
        fields[forbidden_label] = "765,432"
        field_refs[forbidden_label] = refs((2, 2), (3, 3), (4, 4))
        chunk["projection_fields"] = fields
        chunk["projection_field_refs"] = field_refs
        rich_projection = replace(self.projection, chunk=chunk)

        self.assert_closed(
            plan(evidence={"requested_holding_fields": ["before_shares"]}),
            chunks=[self.served, self.raw, rich_projection],
        )

    def test_exact_row_candidate_must_be_a_raw_table(self) -> None:
        non_table = replace(
            self.raw,
            chunk={**dict(self.raw.chunk), "chunk_type": "text"},
        )

        self.assert_closed(
            plan(metric="holding_shares"),
            chunks=[self.served, non_table, self.projection],
        )

    def test_selected_projection_identity_must_match_the_index_record(self) -> None:
        issuer_chunk = {
            **dict(self.projection.chunk),
            "corp_code": "00019999",
        }
        reporter_chunk = dict(self.projection.chunk)
        reporter_fields = dict(reporter_chunk["projection_fields"])
        reporter_fields[label("reporter")] = OTHER_REPORTER
        reporter_chunk["projection_fields"] = reporter_fields
        reporter_chunk["reporter"] = OTHER_REPORTER
        date_chunk = dict(self.projection.chunk)
        date_fields = dict(date_chunk["projection_fields"])
        date_fields[label("reference_date")] = "2024.02.04"
        date_chunk["projection_fields"] = date_fields
        date_chunk["reference_date"] = "2024.02.04"
        cases = {
            "issuer": replace(self.projection, chunk=issuer_chunk),
            "reporter": replace(self.projection, chunk=reporter_chunk),
            "reference_date": replace(self.projection, chunk=date_chunk),
        }

        for identity, candidate in cases.items():
            with self.subTest(identity=identity):
                self.assert_closed(
                    plan(metric="holding_shares"),
                    chunks=[self.served, self.raw, candidate],
                )


class DocumentAmbiguityTests(ContractDFixture):
    def three_projections(self):
        records = [
            record(f"projection_{number}", reporter=f"Holder {number}")
            for number in range(1, 4)
        ]
        projections = [
            projection(value, spans=((number, number),))
            for number, value in enumerate(records, start=10)
        ]
        raws = [
            raw_table(f"raw_{number}", spans=((number, number),))
            for number in range(10, 13)
        ]
        return records, projections, raws

    def test_document_and_row_bridges_without_unique_index_select_nothing(self) -> None:
        records, projections, raws = self.three_projections()
        query_plan = replace(plan(metric="holding_shares"), reporter="Holder 1")
        value = outcome(
            query_plan,
            [self.served, *raws, *projections],
            self.results,
            None,
        )
        self.assertFalse(value.rescued)

    def test_ambiguous_same_event_index_selects_nothing(self) -> None:
        records = [record(f"projection_{number}") for number in range(1, 4)]
        projections = [
            projection(value, spans=((number, number),))
            for number, value in enumerate(records, start=10)
        ]
        raws = [
            raw_table(f"raw_{number}", spans=((number, number),))
            for number in range(10, 13)
        ]
        value = outcome(
            plan(metric="holding_shares"),
            [self.served, *raws, *projections],
            self.results,
            index_of(*records),
        )
        self.assertFalse(value.rescued)

    def test_unique_index_event_promotes_only_its_own_projection(self) -> None:
        records, projections, raws = self.three_projections()
        selected = records[1]
        query_plan = replace(plan(metric="holding_shares"), reporter=selected.raw_reporter)
        value = outcome(
            query_plan,
            [self.served, *raws, *projections],
            self.results,
            index_of(selected),
        )

        self.assertTrue(value.rescued)
        self.assertEqual(
            [item.chunk_id for item in value.selected],
            [selected.projection_chunk_id],
        )


class CorrectionFinalityTests(ContractDFixture):
    def finality_index(self):
        original = record(
            "original_projection",
            doc_id="original_document",
            reference_date="20240102",
        )
        final = record(
            is_correction=True,
        )
        finality = HoldingCorrectionFinality(
            (
                CorrectionChain(
                    group_id="fixture_chain",
                    root_doc_id=original.doc_id,
                    members=(original.doc_id, final.doc_id),
                    final_doc_id=final.doc_id,
                    status=STATUS_RESOLVED,
                ),
            ),
            identity=IDENTITY,
            complete=True,
        )
        return original, final, index_of(
            original,
            final,
            correction_finality_available=False,
            correction_finality=finality,
        )

    def test_resolved_final_record_is_eligible(self) -> None:
        _original, final, report_index = self.finality_index()
        value = self.assess(
            plan(metric="holding_shares"), report_index=report_index
        )
        self.assertTrue(value.rescued)
        self.assertEqual(value.selected[0].chunk_id, final.projection_chunk_id)

    def test_superseded_exact_date_has_no_active_materialisation(self) -> None:
        original, _final, report_index = self.finality_index()
        original_plan = replace(
            plan(metric="holding_shares"),
            period=QueryPeriod(
                year=2024,
                from_date="2024-01-02",
                to_date="2024-01-02",
                period_type="holding_reference_date",
            ),
        )
        original_projection = projection(original)
        original_raw = raw_table(
            "original_raw", doc_id=original.doc_id
        )
        original_served = served_document(
            "original_text", doc_id=original.doc_id
        )
        value = outcome(
            original_plan,
            [original_served, original_raw, original_projection],
            [ranked(original_served)],
            report_index,
        )
        self.assertFalse(value.rescued)

    def test_same_day_tie_remains_ambiguous(self) -> None:
        tie = replace(
            self.record,
            doc_id="same_day_document",
            projection_chunk_id="same_day_projection",
        )
        self.assertFalse(
            self.assess(
                plan(metric="holding_shares"),
                report_index=index_of(self.record, tie),
            ).rescued
        )


class LaneAndFieldBoundaryTests(ContractDFixture):
    def assert_closed(self, query_plan: QueryPlan, **overrides) -> None:
        self.assertFalse(self.assess(query_plan, **overrides).rescued)

    def test_contract_d_requires_the_ordinary_lane(self) -> None:
        self.assert_closed(
            plan(metric="holding_shares"), ordinary_lane=False
        )

    def test_receipt_date_and_range_are_not_exact_reference_lookups(self) -> None:
        receipt = replace(
            plan(metric="holding_shares"),
            period=QueryPeriod(
                year=2024,
                from_date=REFERENCE_DATE,
                to_date=REFERENCE_DATE,
                period_type="receipt_date",
            ),
        )
        date_range = replace(
            plan(metric="holding_shares"),
            period=QueryPeriod(
                year=2024,
                from_date="2024-02-01",
                to_date="2024-02-03",
                period_type="holding_reference_range",
            ),
        )
        self.assert_closed(receipt)
        self.assert_closed(date_range)

    def test_acquisition_comparison_previous_and_change_requests_stay_closed(self) -> None:
        cases = (
            plan(evidence={"requested_holding_fields": ["acquisition_date"]}),
            plan(
                metric="holding_shares",
                comparison={"type": "company_comparison"},
            ),
            plan(evidence={"requested_holding_fields": ["before_shares"]}),
            plan(evidence={"requested_holding_fields": ["change_shares"]}),
        )
        for query_plan in cases:
            with self.subTest(fields=query_plan.evidence):
                self.assert_closed(query_plan)

    def test_non_holding_execution_stays_closed(self) -> None:
        value = assess(
            "current holding",
            plan(metric="holding_shares", task_type="financial_metric"),
            self.chunks,
            self.results,
            routed_task_type="periodic_fact",
            report_index=self.index,
            active_corpus_identity=IDENTITY,
            ordinary_lane=True,
        )
        self.assertFalse(value.evaluated)
        self.assertFalse(value.rescued)

    def phase_three_result(self, intent: dict[str, object], query_plan: QueryPlan):
        query_plan = replace(
            query_plan,
            evidence={**dict(query_plan.evidence), "holding_report_relative": intent},
        )
        phase_three = HoldingReportRelativeExecution(
            index=self.index,
            active_corpus_identity=IDENTITY,
        )
        orchestrator = AgentOrchestrator(
            report_relative_execution=phase_three,
            holding_report_index=self.index,
            active_corpus_identity=IDENTITY,
        )
        execution = SimpleNamespace(
            plan=query_plan,
            chunks=list(self.chunks),
            results=list(self.results),
        )
        return orchestrator.run(str(query_plan.raw_query), query_plan, execution)

    def test_selected_context_and_latest_remain_phase_three_owned(self) -> None:
        cases = (
            {
                "selector": SELECTOR_SELECTED_CONTEXT,
                "projection_role": ROLE_CURRENT,
                "dynamic": True,
                "executable": False,
                "evidence": "selected report",
            },
            {
                "selector": SELECTOR_LATEST,
                "projection_role": ROLE_CURRENT,
                "dynamic": True,
                "executable": True,
                "evidence": "latest report",
            },
        )
        for intent in cases:
            with self.subTest(selector=intent["selector"]):
                result = self.phase_three_result(
                    intent, plan(metric="holding_shares")
                )
                self.assertIn(
                    "holding_report_relative_execution", result.execution_trace
                )
                self.assertFalse(result.holding_coverage.evaluated)
                self.assertNotIn(
                    "holding_evidence_coverage", result.execution_trace
                )

    def test_report_worded_exact_reference_and_receipt_stay_phase_three_owned(self) -> None:
        cases = (
            (
                SELECTOR_EXACT_REFERENCE_DATE,
                plan(metric="holding_shares"),
            ),
            (
                SELECTOR_EXACT_RECEIPT_DATE,
                replace(
                    plan(metric="holding_shares"),
                    period=QueryPeriod(
                        year=2024,
                        from_date="2024-02-05",
                        to_date="2024-02-05",
                        period_type="receipt_date",
                    ),
                ),
            ),
        )
        for selector, query_plan in cases:
            with self.subTest(selector=selector):
                intent = {
                    "selector": selector,
                    "projection_role": ROLE_CURRENT,
                    "dynamic": False,
                    "executable": True,
                    "evidence": "report date",
                }
                result = self.phase_three_result(intent, query_plan)
                self.assertIn(
                    "holding_report_relative_execution", result.execution_trace
                )
                self.assertFalse(result.holding_coverage.evaluated)


class DependencyAndIoTests(ContractDFixture):
    def test_orchestrator_reuses_the_phase_three_index_instance(self) -> None:
        phase_three = HoldingReportRelativeExecution(
            index=self.index,
            active_corpus_identity=IDENTITY,
        )
        orchestrator = AgentOrchestrator(
            report_relative_execution=phase_three,
        )
        self.assertIs(orchestrator.holding_report_index, self.index)
        self.assertEqual(orchestrator.active_corpus_identity, IDENTITY)

    def test_materialisation_uses_only_the_supplied_execution_and_index(self) -> None:
        class CountingIndex(HoldingReportIndex):
            calls = 0

            def select_report(self, *args, **kwargs):
                self.calls += 1
                return super().select_report(*args, **kwargs)

        counting = CountingIndex(
            (self.record,), identity=IDENTITY, complete=True,
            correction_finality_available=True,
        )
        value = self.assess(
            plan(metric="holding_shares"), report_index=counting
        )

        self.assertTrue(value.rescued)
        self.assertEqual(counting.calls, 1)
        self.assertFalse(hasattr(counting, "get_candidate_chunks"))
        self.assertFalse(hasattr(counting, "retrieve"))


class RescueModeObservabilityTests(ContractDFixture):
    def test_served_anchor_and_contract_d_have_distinct_modes(self) -> None:
        served_anchor = outcome(
            plan(metric="holding_shares"),
            [self.raw, self.projection],
            [ranked(self.raw)],
            self.index,
        )
        contract_d = self.assess(plan(metric="holding_shares"))

        self.assertTrue(served_anchor.rescued)
        self.assertEqual(
            served_anchor.rescue_mode, RESCUE_MODE_SERVED_ANCHOR
        )
        self.assertEqual(contract_d.rescue_mode, RESCUE_MODE_CONTRACT_D)
        self.assertEqual(
            served_anchor.to_dict()["rescue_mode"], RESCUE_MODE_SERVED_ANCHOR
        )
        self.assertEqual(
            contract_d.to_dict()["rescue_mode"], RESCUE_MODE_CONTRACT_D
        )

    def test_contract_d_refusal_never_claims_a_rescue_mode(self) -> None:
        other = served_document("unrelated_text", doc_id=OTHER_DOC)
        refused = self.assess(
            plan(metric="holding_shares"),
            chunks=[other, self.raw, self.projection],
            results=[ranked(other)],
        )

        self.assertFalse(refused.rescued)
        self.assertIsNone(refused.rescue_mode)
        self.assertIsNone(refused.to_dict()["rescue_mode"])


class ApiTracePropagationTests(ContractDFixture):
    def serve(self, *, chunks=None, results=None):
        query_plan = plan(metric="holding_shares")
        execution = SimpleNamespace(
            plan=query_plan,
            chunks=list(self.chunks if chunks is None else chunks),
            results=list(self.results if results is None else results),
        )

        class Understanding:
            def understand(self, question, *, top_k):
                del question, top_k
                return query_plan

        class Executor:
            def execute(self, supplied_plan):
                self.plan = supplied_plan
                return execution

        class CapturingOrchestrator(AgentOrchestrator):
            def run(self, *args, **kwargs):
                self.result = super().run(*args, **kwargs)
                return self.result

        phase_three = HoldingReportRelativeExecution(
            index=self.index,
            active_corpus_identity=IDENTITY,
        )
        orchestrator = CapturingOrchestrator(
            report_relative_execution=phase_three,
            holding_report_index=self.index,
            active_corpus_identity=IDENTITY,
        )
        pipeline = AnswerPipeline(
            understanding=Understanding(),
            executor=Executor(),
            orchestrator=orchestrator,
            verbalizer=HcxVerbalizer(HcxSettings(enabled=False)),
            answerability_guard=AnswerabilityGuard(),
        )
        client = TestClient(create_app(pipeline_factory=lambda: pipeline))
        response = client.get(
            "/answer",
            params={"question_id": "trace-fixture", "question": query_plan.raw_query},
        )
        self.assertEqual(response.status_code, 200)
        return response.json(), orchestrator.result

    def test_contract_d_mode_reaches_http_trace_without_moving_results(self) -> None:
        payload, result = self.serve()

        self.assertEqual(
            payload["think_trace"]["holding_evidence_coverage"],
            {
                "status": STATUS_RESCUED,
                "rescued": True,
                "rescue_mode": RESCUE_MODE_CONTRACT_D,
            },
        )
        self.assertEqual(
            [(row["chunk_id"], row["rank"]) for row in payload["retrieved_context"]],
            [(row.chunk_id, row.rank) for row in result.evidence_results],
        )
        self.assertEqual(
            set(payload),
            {"question_id", "question", "retrieved_context", "think_trace", "answer"},
        )

    def test_served_anchor_mode_reaches_http_trace(self) -> None:
        payload, result = self.serve(
            chunks=[self.raw, self.projection],
            results=[ranked(self.raw)],
        )

        self.assertEqual(
            payload["think_trace"]["holding_evidence_coverage"]["rescue_mode"],
            RESCUE_MODE_SERVED_ANCHOR,
        )
        self.assertEqual(
            [(row["chunk_id"], row["rank"]) for row in payload["retrieved_context"]],
            [(row.chunk_id, row.rank) for row in result.evidence_results],
        )

    def test_refusal_does_not_claim_contract_d_in_http_trace(self) -> None:
        other = served_document("unrelated_trace_text", doc_id=OTHER_DOC)
        payload, result = self.serve(
            chunks=[other, self.raw, self.projection],
            results=[ranked(other)],
        )

        self.assertFalse(result.holding_coverage.rescued)
        self.assertNotIn("holding_evidence_coverage", payload["think_trace"])


if __name__ == "__main__":
    unittest.main()
