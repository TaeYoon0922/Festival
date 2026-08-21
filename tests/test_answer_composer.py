import copy
import unittest
from dataclasses import FrozenInstanceError

from app.reasoning.answer_composer import (
    AnswerComposer,
    compose_holding_answer,
    compose_periodic_answer,
)
from app.reasoning.holding_event_resolver import resolve_holding_events
from app.reasoning.periodic_fact_resolver import resolve_periodic_facts
from tests.test_holding_event_resolver import (
    _evidence_set as _holding_evidence,
    _group as _holding_group,
    _item as _holding_item,
)
from tests.test_periodic_fact_resolver import (
    _evidence as _periodic_evidence,
    _group as _periodic_group,
    _item as _periodic_item,
)


class AnswerComposerTests(unittest.TestCase):
    def test_holding_multiple_events_are_not_collapsed_or_latest_selected(self):
        older = _holding_item(
            "h23:ch_report",
            "h23",
            rank=1,
            fields={
                "reporter": "국민연금기금",
                "reference_date": "2023-06-30",
                "after_shares": "100",
            },
        )
        newer = _holding_item(
            "h24:ch_report",
            "h24",
            rank=2,
            fields={
                "reporter": "국민연금기금",
                "reference_date": "2024-06-30",
                "after_shares": "120",
            },
        )
        evidence = _holding_evidence(
            [
                _holding_group("g23", older),
                _holding_group("g24", newer),
            ],
            question="국민연금기금 변동일 변동후 주식수",
        )
        resolution = resolve_holding_events(evidence)

        draft = compose_holding_answer(resolution, evidence)
        events = draft.answer_sections[0].content["events"]

        self.assertEqual(len(events), 2)
        self.assertEqual(
            [event["reference_date"] for event in events],
            ["2023-06-30", "2024-06-30"],
        )
        self.assertEqual(
            [event["after_shares"]["normalized"] for event in events],
            [100, 120],
        )
        self.assertTrue(draft.ambiguity["temporal_ambiguity"])
        self.assertFalse(draft.ambiguity["latest_event_selected"])
        self.assertIn("multiple_matching_holding_events", draft.warnings)

    def test_holding_provenance_is_preserved_in_citation_path(self):
        item = _holding_item(
            "h1:ch_detail",
            "h1",
            rank=4,
            table_id="t19",
            fields={
                "reporter": "국민연금기금",
                "after_shares": "2,202,050",
            },
        )
        evidence = _holding_evidence(
            [_holding_group("g1", item)],
            question="국민연금기금 변동후 주식수",
        )
        resolution = resolve_holding_events(evidence)

        draft = compose_holding_answer(resolution, evidence)
        citation = draft.citations[0]

        self.assertEqual(citation.chunk_id, "h1:ch_detail")
        self.assertEqual(citation.doc_id, "h1")
        self.assertEqual(citation.source_refs[0]["table_id"], "t19")
        self.assertTrue(
            any(path.get("field") == "after_shares" for path in citation.provenance_path)
        )
        self.assertTrue(draft.answerable)

    def test_periodic_repeated_fact_preserves_all_periods_without_latest(self):
        older = _periodic_item(
            "p23:ch_fact", "p23", rank=1, text="동일 사업 내용 확인", year=2023
        )
        newer = _periodic_item(
            "p24:ch_fact", "p24", rank=2, text="동일 사업 내용 확인", year=2024
        )
        evidence = _periodic_evidence(
            [
                _periodic_group(
                    "g-repeat",
                    older,
                    newer,
                    group_type="periodic_repeated_fact",
                )
            ]
        )
        resolution = resolve_periodic_facts(evidence)

        draft = compose_periodic_answer(resolution, evidence)
        fact = draft.answer_sections[0].content["fact"]

        self.assertEqual(fact["fact_text"], "동일 사업 내용 확인")
        self.assertEqual(
            [period["fiscal_year"] for period in fact["reporting_periods"]],
            [2023, 2024],
        )
        self.assertEqual(
            [source["chunk_id"] for source in fact["sources"]],
            ["p23:ch_fact", "p24:ch_fact"],
        )
        self.assertFalse(draft.ambiguity["latest_period_selected"])
        self.assertEqual(
            draft.evidence_references, ("p23:ch_fact", "p24:ch_fact")
        )
        self.assertTrue(draft.answerable)

    def test_periodic_conflict_alternatives_are_exposed_without_selection(self):
        first = _periodic_item(
            "p1:ch_a", "p1", rank=1, text="제품 수는 10개", year=2024
        )
        second = _periodic_item(
            "p1:ch_b", "p1", rank=2, text="제품 수는 12개", year=2024
        )
        evidence = _periodic_evidence(
            [
                _periodic_group(
                    "g-conflict",
                    first,
                    second,
                    group_type="document_evidence",
                )
            ]
        )
        resolution = resolve_periodic_facts(evidence)

        draft = compose_periodic_answer(resolution, evidence)
        fact = draft.answer_sections[0].content["fact"]

        self.assertIsNone(fact["fact_text"])
        self.assertTrue(fact["fact_conflict"])
        self.assertEqual(fact["conflict_type"], "same_period_source_conflict")
        self.assertEqual(len(fact["alternatives"]), 2)
        self.assertEqual(
            {tuple(value["evidence_chunk_ids"]) for value in fact["alternatives"]},
            {("p1:ch_a",), ("p1:ch_b",)},
        )
        self.assertFalse(draft.answerable)

    def test_missing_required_holding_field_is_not_answerable(self):
        item = _holding_item(
            "h1:ch_report",
            "h1",
            rank=1,
            fields={
                "reporter": "국민연금기금",
                "reference_date": "2024-03-07",
            },
        )
        evidence = _holding_evidence(
            [_holding_group("g1", item)],
            question="국민연금기금 변동일 변동후 주식수",
        )
        resolution = resolve_holding_events(evidence)

        draft = compose_holding_answer(resolution, evidence)

        self.assertFalse(draft.answerable)
        self.assertIn("after_shares", resolution.unresolved_fields)
        self.assertIn("answer_not_supported", draft.warnings)

    def test_complete_holding_report_is_answerable_with_incomplete_detail(self):
        report = _holding_item(
            "h1:ch_report",
            "h1",
            rank=1,
            projection_type="holding_report",
            fields={
                "reporter": "국민연금기금",
                "reference_date": "2024-03-07",
                "change_shares": "100",
                "change_ratio": "1.00",
            },
        )
        detail = _holding_item(
            "h2:ch_detail",
            "h2",
            rank=2,
            projection_type="holding_detail_row",
            fields={
                "reporter": "국민연금기금",
                "reference_date": "2024-04-07",
                "change_shares": "50",
            },
        )
        evidence = _holding_evidence(
            [_holding_group("g-report", report), _holding_group("g-detail", detail)],
            question="국민연금 증가 주식수 증가 비율",
            requested=["change_shares", "change_ratio"],
        )

        resolution = resolve_holding_events(evidence)
        draft = compose_holding_answer(resolution, evidence)

        self.assertEqual(resolution.matching_event_count, 2)
        self.assertEqual(len(resolution.events), 2)
        self.assertTrue(draft.answerable)
        self.assertEqual(
            draft.evidence_references,
            ("h1:ch_report", "h2:ch_detail"),
        )
        self.assertTrue(draft.ambiguity["temporal_ambiguity"])
        self.assertIn("multiple_matching_holding_events", draft.warnings)

    def test_incomplete_holding_events_are_not_combined_for_answerability(self):
        shares_only = _holding_item(
            "h1:ch_shares",
            "h1",
            rank=1,
            fields={
                "reporter": "국민연금기금",
                "change_shares": "100",
            },
        )
        ratio_only = _holding_item(
            "h2:ch_ratio",
            "h2",
            rank=2,
            fields={
                "reporter": "국민연금기금",
                "change_ratio": "1.00",
            },
        )
        evidence = _holding_evidence(
            [
                _holding_group("g-shares", shares_only),
                _holding_group("g-ratio", ratio_only),
            ],
            question="국민연금 증감 주식수 증감 비율",
            requested=["change_shares", "change_ratio"],
        )

        resolution = resolve_holding_events(evidence)
        draft = compose_holding_answer(resolution, evidence)

        self.assertFalse(draft.answerable)
        self.assertIn(
            "complete_requested_holding_fields",
            draft.confidence["unresolved_requirements"],
        )

    def test_no_complete_holding_event_keeps_completeness_unresolved(self):
        item = _holding_item(
            "h1:ch_detail",
            "h1",
            rank=1,
            fields={
                "reporter": "국민연금기금",
                "change_shares": "100",
            },
        )
        evidence = _holding_evidence(
            [_holding_group("g1", item)],
            question="국민연금 증감 주식수 증감 비율",
            requested=["change_shares", "change_ratio"],
        )

        draft = compose_holding_answer(resolve_holding_events(evidence), evidence)

        self.assertFalse(draft.answerable)
        self.assertIn(
            "complete_requested_holding_fields",
            draft.confidence["unresolved_requirements"],
        )

    def test_multiple_complete_holding_events_remain_ambiguous_and_answerable(self):
        events = [
            _holding_item(
                f"h{index}:ch_report",
                f"h{index}",
                rank=index,
                projection_type="holding_report",
                fields={
                    "reporter": "국민연금기금",
                    "reference_date": f"202{index + 2}-06-30",
                    "change_shares": str(index * 100),
                    "change_ratio": f"{index}.00",
                },
            )
            for index in (1, 2)
        ]
        evidence = _holding_evidence(
            [
                _holding_group("g1", events[0]),
                _holding_group("g2", events[1]),
            ],
            question="국민연금 증감 주식수 증감 비율",
            requested=["change_shares", "change_ratio"],
        )

        resolution = resolve_holding_events(evidence)
        draft = compose_holding_answer(resolution, evidence)

        self.assertTrue(draft.answerable)
        self.assertEqual(resolution.matching_event_count, 2)
        self.assertEqual(len(draft.answer_sections[0].content["events"]), 2)
        self.assertTrue(draft.ambiguity["temporal_ambiguity"])
        self.assertFalse(draft.ambiguity["latest_event_selected"])

    def test_complete_event_with_wrong_direction_is_not_answerable(self):
        decrease = _holding_item(
            "h1:ch_report",
            "h1",
            rank=1,
            projection_type="holding_report",
            fields={
                "reporter": "국민연금기금",
                "change_shares": "-100",
                "change_ratio": "-1.00",
            },
        )
        evidence = _holding_evidence(
            [_holding_group("g1", decrease)],
            question="국민연금 증가 주식수 증가 비율",
            requested=["change_shares", "change_ratio", "change_direction"],
        )

        resolution = resolve_holding_events(evidence)
        draft = compose_holding_answer(resolution, evidence)

        self.assertEqual(resolution.matching_event_count, 0)
        self.assertFalse(draft.answerable)
        self.assertIn(
            "matching_holding_event",
            draft.confidence["unresolved_requirements"],
        )

    def test_complete_event_for_different_reporter_is_not_answerable(self):
        other_reporter = _holding_item(
            "h1:ch_report",
            "h1",
            rank=1,
            projection_type="holding_report",
            fields={
                "reporter": "다른 기관",
                "change_shares": "100",
                "change_ratio": "1.00",
            },
        )
        evidence = _holding_evidence(
            [_holding_group("g1", other_reporter)],
            question="국민연금 증감 주식수 증감 비율",
            reporter="국민연금",
            requested=["change_shares", "change_ratio"],
        )

        resolution = resolve_holding_events(evidence)
        draft = compose_holding_answer(resolution, evidence)

        self.assertEqual(resolution.matching_event_count, 0)
        self.assertFalse(draft.answerable)
        self.assertIn(
            "matching_holding_event",
            draft.confidence["unresolved_requirements"],
        )

    def test_hx02_complete_previous_report_survives_partial_new_report(self):
        complete = _holding_item(
            "sm:ch_complete",
            "sm",
            rank=1,
            projection_type="holding_report",
            fields={
                "reporter": "(주)하이브",
                "before_shares": "2,098,811",
                "after_shares": "2,967,759",
            },
        )
        partial = _holding_item(
            "sm-new:ch_partial",
            "sm-new",
            rank=2,
            projection_type="holding_report",
            fields={
                "reporter": "다른 신규 보고자",
                "after_shares": "2,212,237",
            },
        )
        evidence = _holding_evidence(
            [
                _holding_group("g-complete", complete),
                _holding_group("g-partial", partial),
            ],
            question="에스엠 하이브 직전보고 보유주식 수 비율",
            reporter=None,
            requested=["before_shares", "after_shares"],
        )

        resolution = resolve_holding_events(evidence)
        draft = compose_holding_answer(resolution, evidence)

        self.assertEqual(resolution.matching_event_count, 2)
        self.assertTrue(draft.answerable)
        self.assertEqual(len(draft.answer_sections[0].content["events"]), 2)
        self.assertEqual(
            draft.evidence_references,
            ("sm:ch_complete", "sm-new:ch_partial"),
        )
        self.assertIn("multiple_matching_holding_events", draft.warnings)

    def test_no_matching_holding_event_is_not_answerable(self):
        item = _holding_item(
            "h1:ch_report",
            "h1",
            rank=1,
            fields={
                "reporter": "다른 기관",
                "reference_date": "2024-03-07",
                "after_shares": "1,000",
            },
        )
        evidence = _holding_evidence(
            [_holding_group("g1", item)],
            question="국민연금기금 변동후 주식수",
        )
        resolution = resolve_holding_events(evidence)

        draft = compose_holding_answer(resolution, evidence)

        self.assertEqual(resolution.matching_event_count, 0)
        self.assertFalse(draft.answerable)
        self.assertIn(
            "matching_holding_event",
            draft.confidence["unresolved_requirements"],
        )

    def test_composition_does_not_mutate_any_input(self):
        holding_item = _holding_item(
            "h1:ch_report",
            "h1",
            rank=1,
            fields={"after_shares": "1,000"},
        )
        holding_evidence = _holding_evidence(
            [_holding_group("hg1", holding_item)],
            question="국민연금기금 보유주식수",
        )
        holding_resolution = resolve_holding_events(holding_evidence)
        periodic_item = _periodic_item(
            "p1:ch_fact", "p1", rank=1, text="사업 내용", year=2024
        )
        periodic_evidence = _periodic_evidence(
            [_periodic_group("pg1", periodic_item)]
        )
        periodic_resolution = resolve_periodic_facts(periodic_evidence)
        before = (
            copy.deepcopy(holding_evidence.to_dict()),
            copy.deepcopy(holding_resolution.to_dict()),
            copy.deepcopy(periodic_evidence.to_dict()),
            copy.deepcopy(periodic_resolution.to_dict()),
        )

        holding_draft = AnswerComposer().compose(
            holding_evidence, holding_resolution=holding_resolution
        )
        AnswerComposer().compose(
            periodic_evidence, periodic_resolution=periodic_resolution
        )

        after = (
            holding_evidence.to_dict(),
            holding_resolution.to_dict(),
            periodic_evidence.to_dict(),
            periodic_resolution.to_dict(),
        )
        self.assertEqual(after, before)
        with self.assertRaises(FrozenInstanceError):
            holding_draft.answerable = False

    def test_exactly_one_resolver_output_is_required(self):
        item = _periodic_item(
            "p1:ch_fact", "p1", rank=1, text="사업 내용", year=2024
        )
        evidence = _periodic_evidence([_periodic_group("g1", item)])

        with self.assertRaises(ValueError):
            AnswerComposer().compose(evidence)


if __name__ == "__main__":
    unittest.main()
