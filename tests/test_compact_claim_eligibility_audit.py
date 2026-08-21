from __future__ import annotations

import unittest

from app.agent.orchestrator import AgentOrchestrator
from app.generation.compact_claim import MAX_CLAIM_EVENTS, MAX_CLAIM_LITERALS
from app.reasoning.query_plan import QueryPlan
from scripts.audit_compact_claim_eligibility import (
    CITATION_NOT_LINKED,
    NO_REQUESTED_FIELDS,
    NOT_ANSWERABLE,
    TOO_MANY_EVENTS,
    UNRESOLVED_FIELDS,
    UNSUPPORTED_TASK_TYPE,
    _summarize,
    analyze_eligibility,
)
from tests.test_agent_end_to_end_smoke import _execution
from tests.test_evidence_builder import _candidate, _holding_pair


def _holding_result(*, events: int = 1, requested: list[str] | None = None):
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
    plan = QueryPlan(
        query="효성중공업 국민연금기금 변동일 변동후 주식수",
        task_type="holding_change",
        metric="holding_shares",
        reporter="국민연금기금",
        disclosure_route=("holding",),
        evidence={"requested_holding_fields": fields},
    )
    return AgentOrchestrator().run(plan.raw_query, plan, _execution(plan, *pairs))


def _analyze(result, *, task_type: str | None = None, resolution=None):
    return analyze_eligibility(
        result.answer_draft,
        result.resolution if resolution is None else resolution,
        task_type=task_type or result.task_decision.task_type,
    )


class EligibleTests(unittest.TestCase):
    def test_a_single_event_question_is_eligible(self) -> None:
        report = _analyze(_holding_result(events=1))

        self.assertTrue(report["compact_eligible"])
        self.assertIsNone(report["skip_reason"])
        self.assertEqual(report["candidate_event_count"], 1)
        self.assertEqual(report["candidate_literal_count"], 2)

    def test_counts_describe_the_event_funnel(self) -> None:
        report = _analyze(_holding_result(events=2))

        self.assertEqual(report["total_events"], 2)
        self.assertEqual(report["matches_query_events"], 2)
        self.assertEqual(report["complete_matching_events"], 2)
        self.assertEqual(report["fully_requested_field_events"], 2)
        self.assertEqual(report["citation_linked_events"], 2)

    def test_limits_are_reported_so_a_skip_can_be_read_in_context(self) -> None:
        report = _analyze(_holding_result(events=1))

        self.assertEqual(report["MAX_CLAIM_EVENTS"], MAX_CLAIM_EVENTS)
        self.assertEqual(report["MAX_CLAIM_LITERALS"], MAX_CLAIM_LITERALS)


class SkipReasonTests(unittest.TestCase):
    def test_hx07_shape_reports_too_many_events(self) -> None:
        """Four complete matching events, which is one past the cap."""

        report = _analyze(_holding_result(events=MAX_CLAIM_EVENTS + 1))

        self.assertFalse(report["compact_eligible"])
        self.assertEqual(report["skip_reason"], TOO_MANY_EVENTS)
        self.assertEqual(report["candidate_event_count"], MAX_CLAIM_EVENTS + 1)
        self.assertGreater(report["candidate_event_count"], MAX_CLAIM_EVENTS)
        # The literal cap is not what stopped it.
        self.assertLessEqual(report["candidate_literal_count"], MAX_CLAIM_LITERALS)

    def test_unsupported_task_type_is_named(self) -> None:
        report = _analyze(_holding_result(), task_type="periodic_fact")

        self.assertEqual(report["skip_reason"], UNSUPPORTED_TASK_TYPE)

    def test_unanswerable_draft_is_named(self) -> None:
        report = analyze_eligibility(None, None, task_type="holding_event")

        self.assertEqual(report["skip_reason"], NOT_ANSWERABLE)

    def test_no_requested_fields_is_named(self) -> None:
        result = _holding_result(events=1)
        resolution = dict(result.resolution.to_dict())
        resolution["requested_fields"] = []

        report = _analyze(result, resolution=resolution)

        self.assertEqual(report["skip_reason"], NO_REQUESTED_FIELDS)

    def test_unresolved_fields_is_named(self) -> None:
        result = _holding_result(events=1)
        resolution = dict(result.resolution.to_dict())
        resolution["unresolved_fields"] = ["before_shares"]

        report = _analyze(result, resolution=resolution)

        self.assertEqual(report["skip_reason"], UNRESOLVED_FIELDS)

    def test_a_field_the_resolver_never_supplied_stops_at_the_composer(self) -> None:
        """An incomplete holding answer never reaches the compact gates.

        The composer marks the draft unanswerable first, so the audit reports
        ``not_answerable`` rather than a compact-claim gate.
        """

        report = _analyze(
            _holding_result(requested=["reference_date", "before_shares"])
        )

        self.assertFalse(report["compact_eligible"])
        self.assertEqual(report["skip_reason"], NOT_ANSWERABLE)
        self.assertEqual(report["candidate_event_count"], 0)
        self.assertEqual(report["total_events"], 0)

    def test_periodic_question_is_reported_as_unsupported(self) -> None:
        pair = _candidate(
            "p23:ch", "p23", rank=1, doc_group="periodic",
            content="DX 부문은 TV, 모니터를 생산합니다.",
            section="주요 제품 및 서비스", fiscal_year=2023, period_type="fiscal_year",
        )
        plan = QueryPlan(
            query="두산퓨얼셀 2023년 1분기 연료전지 주기기 매출액",
            task_type="disclosure_lookup",
            disclosure_route=("periodic",),
            evidence={"periodic_intent": "business_product"},
        )
        result = AgentOrchestrator().run(plan.raw_query, plan, _execution(plan, pair))

        report = _analyze(result)

        self.assertEqual(report["skip_reason"], UNSUPPORTED_TASK_TYPE)


class BuilderAgreementTests(unittest.TestCase):
    """The audit must never explain a decision the adapter did not make."""

    def test_agreement_is_asserted_on_every_path(self) -> None:
        cases = [
            _analyze(_holding_result(events=1)),
            _analyze(_holding_result(events=MAX_CLAIM_EVENTS + 1)),
            _analyze(_holding_result(), task_type="periodic_fact"),
            _analyze(_holding_result(requested=["reference_date", "before_shares"])),
        ]

        for report in cases:
            with self.subTest(reason=report["skip_reason"]):
                self.assertTrue(report["audit_agrees_with_builder"])
                self.assertEqual(
                    report["compact_eligible"], report["builder_returned_claim"]
                )

    def test_claim_details_appear_only_when_a_claim_exists(self) -> None:
        eligible = _analyze(_holding_result(events=1))
        skipped = _analyze(_holding_result(events=MAX_CLAIM_EVENTS + 1))

        self.assertIn("claim_chars", eligible)
        self.assertNotIn("claim_chars", skipped)


class SummaryTests(unittest.TestCase):
    def _row(self, question_id: str, *, eligible: bool, reason: str | None) -> dict:
        return {
            "question_id": question_id,
            "answerable": True,
            "requested_fields": ["reference_date"],
            "complete_matching_events": 1,
            "candidate_event_count": 1,
            "candidate_literal_count": 1,
            "compact_eligible": eligible,
            "skip_reason": reason,
            "audit_agrees_with_builder": True,
        }

    def test_counts_each_skip_bucket(self) -> None:
        rows = [
            self._row("HX01", eligible=True, reason=None),
            self._row("HX02", eligible=False, reason=TOO_MANY_EVENTS),
            self._row("HX03", eligible=False, reason=TOO_MANY_EVENTS),
            self._row("HX04", eligible=False, reason=CITATION_NOT_LINKED),
        ]

        summary = _summarize(rows)["summary"]

        self.assertEqual(summary["holding_questions"], 4)
        self.assertEqual(summary["compact_eligible_count"], 1)
        self.assertEqual(summary["too_many_events_count"], 2)
        self.assertEqual(summary["too_many_literals_count"], 0)
        self.assertEqual(summary["other_skip_count"], 1)

    def test_disagreements_are_surfaced_not_hidden(self) -> None:
        row = self._row("HX09", eligible=True, reason=None)
        row["audit_agrees_with_builder"] = False

        summary = _summarize([row])["summary"]

        self.assertEqual(summary["audit_disagreements"], ["HX09"])


if __name__ == "__main__":
    unittest.main()
