import copy
import unittest
from dataclasses import replace

from app.reasoning.evidence_builder import (
    EvidenceGroup,
    EvidenceItem,
    EvidenceSet,
    build_evidence_set,
)
from app.reasoning.holding_event_resolver import (
    HoldingEventResolver,
    NumericValue,
    _requested_fields,
    has_acquisition_semantics,
    resolve_holding_events,
)
from app.reasoning.query_plan import QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


def _item(
    chunk_id: str,
    doc_id: str,
    *,
    rank: int,
    fields: dict | None = None,
    projection_type: str = "holding_detail_row",
    table_id: str | None = None,
    temporal_match: bool | None = None,
    rcept_dt: str = "2024-03-08",
) -> EvidenceItem:
    fields = dict(fields or {})
    labels = {
        "reporter": "보고자/보유자",
        "reference_date": "기준일/보고일",
        "before_shares": "직전 보유주식수",
        "change_shares": "증감주식수",
        "after_shares": "보유주식수",
        "before_ratio": "직전 보유비율",
        "after_ratio": "보유비율",
        "change_ratio": "증감비율",
    }
    projection_fields = {
        labels[key]: value for key, value in fields.items() if key in labels
    }
    ref = {
        "table_id": table_id or f"t{rank}",
        "row_start": rank,
        "row_end": rank,
    }
    field_refs = {label: [ref] for label in projection_fields}
    holding = {
        **fields,
        "projection_type": projection_type,
        "projection_fields": projection_fields,
        "projection_field_refs": field_refs,
    }
    source_chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "projection_type": projection_type,
        "projection_fields": projection_fields,
        "source_refs": [ref],
    }
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        company_id="00123456",
        corp_code="00123456",
        corp_name="테스트회사",
        doc_group="holding",
        chunk_type="table_projection",
        section_path=("보유주식등의 수 및 보유비율",),
        evidence_text="holding evidence",
        retrieval_rank=rank,
        retrieval_score=1.0 - rank / 100,
        rcept_dt=rcept_dt,
        report_nm="주식등의대량보유상황보고서",
        period={},
        source_refs=(ref,),
        provenance={
            "source_chunk_id": chunk_id,
            "source_doc_id": doc_id,
            "table_id": ref["table_id"],
            "source_refs": [ref],
            "projection_field_refs": field_refs,
            "source_chunk": source_chunk,
        },
        holding=holding,
        temporal_match=temporal_match,
    )


def _group(group_id: str, *items: EvidenceItem) -> EvidenceGroup:
    ordered = tuple(sorted(items, key=lambda item: item.retrieval_rank))
    return EvidenceGroup(
        group_id=group_id,
        group_type="holding_event",
        member_chunk_ids=tuple(item.chunk_id for item in ordered),
        primary_evidence=ordered[0],
        supporting_evidence=ordered[1:],
        doc_ids=tuple(dict.fromkeys(item.doc_id for item in ordered)),
        reason="fixture holding event",
    )


def _evidence_set(
    groups: list[EvidenceGroup],
    *,
    question: str = "국민연금 보유주식수",
    period: dict | None = None,
    reporter: str | None = "국민연금기금",
    metric: str = "holding_shares",
    requested: list[str] | None = None,
) -> EvidenceSet:
    period = period or {"period_type": "latest_holding"}
    explicit = any(period.get(key) is not None for key in ("year", "from", "to"))
    plan = {
        "task_type": "holding_change",
        "metric": metric,
        "reporter": reporter,
        "raw_query": question,
        "period": period,
        "evidence": {"requested_holding_fields": requested or []},
    }
    items = [item for group in groups for item in group.items]
    return EvidenceSet(
        question=question,
        query_plan=plan,
        task_type="holding_change",
        evidence_groups=tuple(groups),
        retrieval_order=tuple(item.chunk_id for item in items),
        raw_candidate_count=len(items),
        selected_evidence_count=len(items),
        warnings=(),
        ambiguity={
            "temporal_ambiguity": False,
            "temporal_constraint": {
                "explicit": explicit,
                "year": period.get("year"),
                "from_date": period.get("from"),
                "to_date": period.get("to"),
                "period_type": period.get("period_type"),
            },
        },
    )


class HoldingEventResolverTests(unittest.TestCase):
    def test_single_holding_event_reconstructs_core_fields(self):
        item = _item(
            "h1:ch_report",
            "h1",
            rank=1,
            fields={
                "reporter": "국민연금기금",
                "reference_date": "2024년 03월 07일",
                "after_shares": "1,234,567",
            },
        )

        resolution = resolve_holding_events(_evidence_set([_group("g1", item)]))
        event = resolution.events[0]

        self.assertEqual(event.reporter, "국민연금기금")
        self.assertEqual(event.reference_date, "2024-03-07")
        self.assertEqual(event.after_shares, NumericValue("1,234,567", 1234567))

    def test_projection_merge_combines_original_detail_and_report_fields(self):
        original = _item(
            "h1:ch_table",
            "h1",
            rank=1,
            projection_type="original",
            fields={"reporter": "국민연금기금", "reference_date": "2024-03-07"},
        )
        detail = _item(
            "h1:ch_detail",
            "h1",
            rank=2,
            fields={"before_shares": "900", "change_shares": "100", "after_shares": "1,000"},
        )
        report = _item(
            "h1:ch_report",
            "h1",
            rank=3,
            projection_type="holding_report",
            fields={"after_ratio": "7.25%", "change_ratio": "0.50%"},
        )

        event = resolve_holding_events(
            _evidence_set([_group("g1", original, detail, report)])
        ).events[0]

        self.assertEqual(event.before_shares.normalized, 900)
        self.assertEqual(event.change_shares.normalized, 100)
        self.assertEqual(event.after_shares.normalized, 1000)
        self.assertEqual(event.after_ratio.normalized, 7.25)

    def test_field_provenance_points_to_exact_chunk_and_source_row(self):
        item = _item(
            "h1:ch_detail",
            "h1",
            rank=4,
            table_id="t19",
            fields={"after_shares": "2,202,050"},
        )

        event = resolve_holding_events(
            _evidence_set([_group("g1", item)])
        ).events[0]
        provenance = event.field_provenance["after_shares"]

        self.assertEqual(provenance.sources[0].chunk_id, "h1:ch_detail")
        self.assertEqual(provenance.sources[0].source_refs[0]["table_id"], "t19")
        self.assertTrue(provenance.sources[0].direct_field_ref)

    def test_multiple_dates_remain_distinct_events(self):
        first = _item(
            "h2023:ch_report",
            "h2023",
            rank=1,
            fields={"reporter": "국민연금기금", "reference_date": "2023-06-30"},
        )
        second = _item(
            "h2024:ch_report",
            "h2024",
            rank=2,
            fields={"reporter": "국민연금기금", "reference_date": "2024-06-30"},
        )

        resolution = resolve_holding_events(
            _evidence_set([_group("g2023", first), _group("g2024", second)])
        )

        self.assertEqual(len(resolution.events), 2)
        self.assertEqual(
            [event.reference_date for event in resolution.events],
            ["2023-06-30", "2024-06-30"],
        )

    def test_no_date_with_multiple_matches_sets_temporal_ambiguity(self):
        first = _item(
            "h2023:ch_report",
            "h2023",
            rank=1,
            fields={"reporter": "국민연금기금", "reference_date": "2023-06-30", "after_shares": "100"},
        )
        second = _item(
            "h2024:ch_report",
            "h2024",
            rank=2,
            fields={"reporter": "국민연금기금", "reference_date": "2024-06-30", "after_shares": "120"},
        )

        resolution = resolve_holding_events(
            _evidence_set([_group("g2023", first), _group("g2024", second)])
        )

        self.assertTrue(resolution.temporal_ambiguity)
        self.assertEqual(resolution.matching_event_count, 2)

    def test_explicit_date_marks_match_without_removing_other_events(self):
        first = _item(
            "h2023:ch_report",
            "h2023",
            rank=1,
            temporal_match=False,
            fields={"reporter": "국민연금기금", "reference_date": "2023-06-30", "after_shares": "100"},
        )
        second = _item(
            "h2024:ch_report",
            "h2024",
            rank=2,
            temporal_match=True,
            fields={"reporter": "국민연금기금", "reference_date": "2024-06-30", "after_shares": "120"},
        )
        period = {
            "year": 2024,
            "from": "2024-06-30",
            "to": "2024-06-30",
            "period_type": "holding_reference_date",
        }

        resolution = resolve_holding_events(
            _evidence_set(
                [_group("g2023", first), _group("g2024", second)],
                question="2024년 6월 30일 현재 보유주식수",
                period=period,
            )
        )

        self.assertEqual(len(resolution.events), 2)
        self.assertEqual([event.temporal_match for event in resolution.events], [False, True])
        self.assertEqual(resolution.matching_event_count, 1)

    def test_change_shares_determines_increase_decrease_and_unchanged(self):
        cases = (("100", "increase"), ("-100", "decrease"), ("0", "unchanged"))
        for index, (raw, expected) in enumerate(cases, start=1):
            with self.subTest(raw=raw):
                item = _item(
                    f"h{index}:ch_detail",
                    f"h{index}",
                    rank=index,
                    fields={"change_shares": raw},
                )
                event = resolve_holding_events(
                    _evidence_set([_group(f"g{index}", item)], reporter=None)
                ).events[0]
                self.assertEqual(event.change_direction, expected)

    def test_conflicting_field_values_are_not_overwritten(self):
        first = _item(
            "h1:ch_detail",
            "h1",
            rank=1,
            fields={"after_shares": "1,000", "change_shares": "100", "change_direction": "decrease"},
        )
        second = _item(
            "h1:ch_report",
            "h1",
            rank=2,
            fields={"after_shares": "1,200"},
        )

        event = resolve_holding_events(
            _evidence_set([_group("g1", first, second)], reporter=None)
        ).events[0]

        self.assertIsNone(event.after_shares)
        self.assertTrue(event.field_provenance["after_shares"].field_conflict)
        self.assertEqual(len(event.field_provenance["after_shares"].alternatives), 2)
        self.assertIsNone(event.change_direction)
        self.assertIn("change_direction_metadata_mismatch", event.warnings)

    def test_numeric_normalization_preserves_raw_shares_and_percent(self):
        item = _item(
            "h1:ch_report",
            "h1",
            rank=1,
            fields={"after_shares": "1,234,567", "after_ratio": "7.25%"},
        )

        event = resolve_holding_events(
            _evidence_set([_group("g1", item)], metric="holding_ratio")
        ).events[0]

        self.assertEqual(event.after_shares.raw, "1,234,567")
        self.assertEqual(event.after_shares.normalized, 1234567)
        self.assertEqual(event.after_ratio.raw, "7.25%")
        self.assertEqual(event.after_ratio.normalized, 7.25)

    def test_missing_markers_are_not_normalized_to_zero(self):
        item = _item(
            "h1:ch_detail",
            "h1",
            rank=1,
            fields={"change_shares": "-", "after_ratio": "unknown"},
        )

        event = resolve_holding_events(
            _evidence_set([_group("g1", item)], reporter=None)
        ).events[0]

        self.assertIsNone(event.change_shares)
        self.assertIsNone(event.change_direction)
        self.assertIsNone(event.after_ratio)

    def test_resolution_does_not_mutate_evidence_set(self):
        item = _item(
            "h1:ch_report",
            "h1",
            rank=1,
            fields={"after_shares": "1,000"},
        )
        evidence = _evidence_set([_group("g1", item)])
        before = copy.deepcopy(evidence)

        HoldingEventResolver().resolve(evidence)

        self.assertEqual(evidence, before)

    def test_event_preserves_every_evidence_chunk_and_source_ref(self):
        first = _item(
            "h1:ch_table",
            "h1",
            rank=1,
            table_id="t12",
            fields={"reporter": "국민연금기금", "reference_date": "2024-03-07"},
        )
        second = _item(
            "h1:ch_detail",
            "h1",
            rank=2,
            table_id="t19",
            fields={"after_shares": "1,000"},
        )

        event = resolve_holding_events(
            _evidence_set([_group("g1", first, second)])
        ).events[0]

        self.assertEqual(event.evidence_chunk_ids, ("h1:ch_table", "h1:ch_detail"))
        self.assertEqual(
            {ref["table_id"] for ref in event.source_refs}, {"t12", "t19"}
        )
        serialized = event.to_dict()
        self.assertEqual(
            serialized["field_provenance"]["after_shares"]["chunk_id"],
            "h1:ch_detail",
        )

    def test_query_plan_requested_fields_are_reused_before_alias_fallback(self):
        item = _item(
            "h1:ch_detail",
            "h1",
            rank=1,
            fields={"change_shares": "100"},
        )
        evidence = _evidence_set(
            [_group("g1", item)],
            question="국민연금 현황",
            metric="holding_shares",
            requested=["증감주식수"],
        )

        resolution = resolve_holding_events(evidence)

        self.assertEqual(resolution.requested_fields, ("change_shares",))
        self.assertEqual(resolution.unresolved_fields, ())

    def test_resolver_consumes_evidence_builder_output_contract(self):
        chunk = {
            "chunk_id": "h1:ch_projection",
            "doc_id": "h1",
            "corp_code": "00123456",
            "corp_name": "테스트회사",
            "doc_group": "holding",
            "chunk_type": "table_projection",
            "section_path": ["보유주식등의 수 및 보유비율"],
            "content": "국민연금기금 2024-03-07 보유주식수 1,000",
            "retrieval_text": "국민연금기금 2024-03-07 보유주식수 1,000",
            "projection_type": "holding_report",
            "projection_fields": {
                "보고자/보유자": "국민연금기금",
                "기준일/보고일": "2024-03-07",
                "보유주식수": "1,000",
            },
            "projection_field_refs": {
                "보유주식수": [{"table_id": "t12", "row_start": 3, "row_end": 3}]
            },
            "source_refs": [{"table_id": "t12", "row_start": 3, "row_end": 3}],
        }
        candidate = CandidateChunk(
            "h1:ch_projection", "h1", chunk, MetadataMatch()
        )
        result = RetrievalResult(
            "h1:ch_projection", "h1", 0.9, 1, {"hybrid": {"final_score": 0.95}}
        )
        plan = QueryPlan(
            query="국민연금 보유주식수", task_type="holding_change"
        )
        evidence = build_evidence_set(
            question="국민연금 보유주식수",
            query_plan=plan,
            candidates=[candidate],
            results=[result],
        )

        resolution = resolve_holding_events(evidence)

        self.assertEqual(len(resolution.events), 1)
        self.assertEqual(resolution.events[0].after_shares.normalized, 1000)
        self.assertEqual(
            resolution.events[0].field_provenance["after_shares"].sources[0].chunk_id,
            "h1:ch_projection",
        )

    def test_direct_projection_field_precedes_generic_item_metadata(self):
        item = _item(
            "h1:ch_projection",
            "h1",
            rank=1,
            fields={"after_shares": "1,000"},
        )
        holding = dict(item.holding)
        holding["after_shares"] = "999"
        item = replace(item, holding=holding)

        event = resolve_holding_events(
            _evidence_set([_group("g1", item)], reporter=None)
        ).events[0]

        self.assertEqual(event.after_shares.normalized, 1000)
        self.assertTrue(
            event.field_provenance["after_shares"].sources[0].direct_field_ref
        )


class AcquisitionSemanticAuthorityTests(unittest.TestCase):
    """The acquisition family a question belongs to, answerable or not."""

    def test_answerable_acquisition_wording_is_acquisition_semantics(self):
        for question in ("취득일", "취득 일자", "취득 수량", "취득 주식수", "취득주식수"):
            with self.subTest(question=question):
                self.assertTrue(has_acquisition_semantics(question, {}))
                # These stay answerable request fields; the predicate reads
                # them through the canonical parser rather than restating it.
                self.assertTrue(_requested_fields(question, {}))

    def test_unit_price_wording_is_acquisition_semantics_but_not_a_field(self):
        for question in ("취득단가", "취득 단가", "취득/처분단가"):
            with self.subTest(question=question):
                self.assertTrue(has_acquisition_semantics(question, {}))
                # Recognising the family must not invent an answer field.
                self.assertEqual(_requested_fields(question, {}), ())

    def test_generic_holding_language_is_not_acquisition_semantics(self):
        for question in (
            "보유주식수는?",
            "보유비율은?",
            "지분율은 얼마인가요?",
            "증감주식수는?",
            "보고자는 누구인가요?",
            "주식을 얼마나 들고 있어?",
            "단가는 얼마인가요?",
        ):
            with self.subTest(question=question):
                self.assertFalse(has_acquisition_semantics(question, {}))

    def test_a_requested_acquisition_field_on_the_plan_is_honoured(self):
        plan = {"evidence": {"requested_holding_fields": ["acquired_shares"]}}

        self.assertTrue(has_acquisition_semantics("보유 현황", plan))
        self.assertFalse(has_acquisition_semantics("보유 현황", {}))


if __name__ == "__main__":
    unittest.main()
