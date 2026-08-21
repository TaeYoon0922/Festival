from __future__ import annotations

import unittest

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import (
    CitationAwareAnswerGenerator,
    _holding_fact_lines,
)
from app.reasoning.answer_composer import _reported_holding_events
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from tests.test_agent_end_to_end_smoke import _execution
from tests.test_evidence_builder import _holding_pair


#: Synthetic events. No real company, reporter, date, or value is relied on.
EVENTS = (
    ("2022-12-05", "720,039", "7.12"),
    ("2023-06-30", "801,200", "7.90"),
    ("2024-06-30", "650,100", "6.40"),
)

REPORTER = "국민연금기금"


def _pairs(events=EVENTS):
    built = []
    for index, (date, shares, ratio) in enumerate(events):
        pair = _holding_pair(
            f"h{index}:ch",
            f"h{index}",
            rank=index + 1,
            date=date,
            projection_type="holding_report",
            table_id=f"t{index}",
        )
        chunk = pair[0].chunk
        chunk["projection_fields"]["보유주식수"] = shares
        chunk["projection_fields"]["보유비율"] = ratio
        chunk["content"] = f"{REPORTER} {date} 보유주식수 {shares} 보유비율 {ratio}"
        built.append(pair)
    return built


def _ask(question: str, *, period=None, metric=None, events=EVENTS):
    plan = QueryPlan(
        query=question,
        task_type="holding_change",
        metric=metric,
        reporter=REPORTER,
        disclosure_route=("holding",),
        period=period,
    )
    pairs = _pairs(events)
    result = AgentOrchestrator().run(question, plan, _execution(plan, *pairs))
    generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
    return result, generated


def _exact(date: str) -> QueryPeriod:
    return QueryPeriod(year=int(date[:4]), from_date=date, to_date=date)


def _event_rows(result) -> list:
    return [
        row
        for section in result.answer_draft.answer_sections
        for row in dict(section.content).get("events", [])
    ]


class Case1ExactDateSingleFieldTests(unittest.TestCase):
    """"2022년 12월 5일 기준 ... 보유 비율은?" — one event, one figure."""

    def setUp(self) -> None:
        self.result, self.generated = _ask(
            "2022년 12월 5일 기준 국민연금의 파마리서치 보유 비율은?",
            period=_exact("2022-12-05"),
            metric="holding_ratio",
        )

    def test_only_the_requested_field_is_recognised(self) -> None:
        self.assertEqual(
            list(self.result.resolution.requested_fields), ["after_ratio"]
        )

    def test_exactly_one_event_matches_the_date(self) -> None:
        self.assertEqual(self.result.resolution.matching_event_count, 1)

    def test_only_the_matching_event_is_reported(self) -> None:
        rows = _event_rows(self.result)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reference_date"], "2022-12-05")

    def test_the_answer_carries_the_asked_value(self) -> None:
        self.assertIn("7.12%", self.generated.answer_text)

    def test_other_dates_are_absent_from_the_answer(self) -> None:
        for absent in ("2023-06-30", "2024-06-30", "7.90", "6.40", "801,200", "650,100"):
            with self.subTest(absent=absent):
                self.assertNotIn(absent, self.generated.answer_text)

    def test_complementary_facts_of_the_same_event_are_preserved(self) -> None:
        """The asked figure leads, but nothing verified is deleted.

        Deleting them removed facts a reader needs to interpret the one they
        asked for, and cost the internal regression set its evidence-term
        coverage.  The assertion is on values, not labels, because the wording
        is presentation and the values are the contract.
        """

        self.assertIn("720,039주", self.generated.answer_text)

    def test_the_asked_value_leads_the_event(self) -> None:
        text = self.generated.answer_text

        self.assertLess(text.index("7.12%"), text.index("720,039주"))

    def test_the_answer_still_says_whose_holding_and_when(self) -> None:
        self.assertIn(REPORTER, self.generated.answer_text)
        self.assertIn("2022-12-05", self.generated.answer_text)

    def test_only_the_supporting_citation_is_kept(self) -> None:
        self.assertEqual(len(self.generated.citations), 1)
        self.assertEqual(self.generated.citations[0].chunk_id, "h0:ch")

    def test_the_answer_remains_answerable(self) -> None:
        self.assertTrue(self.generated.answerable)


class Case2AlternatePhrasingTests(unittest.TestCase):
    """A different wording of the same fact resolves to the same scope."""

    def setUp(self) -> None:
        self.result, self.generated = _ask(
            "2022-12-05 국민연금공단의 파마리서치 지분율 알려줘",
            period=_exact("2022-12-05"),
        )

    def test_the_ratio_field_is_recognised_from_the_wording(self) -> None:
        self.assertIn("after_ratio", self.result.resolution.requested_fields)

    def test_the_same_single_event_is_reported(self) -> None:
        rows = _event_rows(self.result)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reference_date"], "2022-12-05")

    def test_the_same_value_is_answered(self) -> None:
        self.assertIn("7.12%", self.generated.answer_text)
        self.assertNotIn("2023-06-30", self.generated.answer_text)


class Case3TwoRequestedFieldsTests(unittest.TestCase):
    """Two fields asked, two fields answered, still one event."""

    def setUp(self) -> None:
        self.result, self.generated = _ask(
            "2022년 12월 5일 기준 국민연금의 파마리서치 보유 주식수와 비율은?",
            period=_exact("2022-12-05"),
        )

    def test_both_fields_are_recognised(self) -> None:
        requested = set(self.result.resolution.requested_fields)

        self.assertIn("after_shares", requested)
        self.assertIn("after_ratio", requested)

    def test_one_event_is_reported(self) -> None:
        self.assertEqual(len(_event_rows(self.result)), 1)

    def test_both_values_appear(self) -> None:
        self.assertIn("720,039주", self.generated.answer_text)
        self.assertIn("7.12%", self.generated.answer_text)

    def test_other_events_are_absent(self) -> None:
        self.assertNotIn("2023-06-30", self.generated.answer_text)
        self.assertNotIn("2024-06-30", self.generated.answer_text)


class Case4HistoryQuestionTests(unittest.TestCase):
    """A history question keeps the existing multi-event answer."""

    def setUp(self) -> None:
        self.result, self.generated = _ask(
            "파마리서치 국민연금 보유 변동 내역을 알려줘"
        )

    def test_no_field_is_singled_out(self) -> None:
        self.assertEqual(self.result.resolution.requested_fields, ())

    def test_every_event_is_still_reported(self) -> None:
        self.assertEqual(len(_event_rows(self.result)), len(EVENTS))

    def test_every_date_still_appears(self) -> None:
        for date, _, _ in EVENTS:
            with self.subTest(date=date):
                self.assertIn(date, self.generated.answer_text)

    def test_every_verified_value_is_still_rendered(self) -> None:
        """The compact multi-event form drops labels, never values."""

        text = self.generated.answer_text
        for _, shares, ratio in EVENTS:
            with self.subTest(shares=shares):
                self.assertIn(f"{shares}주", text)
                self.assertIn(f"{ratio}%", text)


class Case5MultiEventSafetyTests(unittest.TestCase):
    """Fields asked but several events match: nothing is picked for the user.

    This is the shape the HCX multi-event safety gate depends on. Narrowing
    must not turn an ambiguous question into a single-event claim.
    """

    def setUp(self) -> None:
        # Fields named, no date given, so every event still satisfies the query.
        self.result, self.generated = _ask("국민연금 보유 비율은?")

    def test_fields_are_recognised(self) -> None:
        self.assertEqual(
            list(self.result.resolution.requested_fields), ["after_ratio"]
        )

    def test_more_than_one_event_matches(self) -> None:
        self.assertGreater(self.result.resolution.matching_event_count, 1)

    def test_all_matching_events_are_kept(self) -> None:
        self.assertEqual(len(_event_rows(self.result)), len(EVENTS))

    def test_the_ambiguity_warning_is_preserved(self) -> None:
        self.assertTrue(self.result.answer_draft.ambiguity["temporal_ambiguity"])
        self.assertIn(
            "multiple_matching_holding_events", self.result.answer_draft.warnings
        )

    def test_the_hcx_multi_event_gate_still_sees_several_events(self) -> None:
        from app.generation.compact_claim import build_compact_claim
        from app.generation.lossless_verbalization import claim_event_count

        claim = build_compact_claim(
            self.result.answer_draft,
            self.result.resolution,
            task_type=self.result.task_decision.task_type,
        )

        self.assertIsNotNone(claim)
        self.assertGreater(claim_event_count(claim), 1)

    def test_an_ambiguous_question_never_narrows_to_one_event(self) -> None:
        dates = {row["reference_date"] for row in _event_rows(self.result)}

        self.assertEqual(dates, {date for date, _, _ in EVENTS})


class Case6NoExactMatchTests(unittest.TestCase):
    """A date nothing satisfies is never rounded to the nearest event."""

    def setUp(self) -> None:
        self.result, self.generated = _ask(
            "2021년 3월 1일 기준 국민연금의 파마리서치 보유 비율은?",
            period=_exact("2021-03-01"),
            metric="holding_ratio",
        )

    def test_no_event_matches(self) -> None:
        self.assertEqual(self.result.resolution.matching_event_count, 0)

    def test_no_nearest_event_is_selected(self) -> None:
        rows = _event_rows(self.result)

        self.assertEqual(len(rows), len(EVENTS))

    def test_the_answer_is_not_supported(self) -> None:
        self.assertFalse(self.generated.answerable)
        self.assertIn("answer_not_supported", self.generated.warnings)


class ReportedEventSelectionTests(unittest.TestCase):
    """The narrowing rule itself, without any corpus fixture."""

    class _Event:
        def __init__(self, matches_query, chunk="c") -> None:
            self.matches_query = matches_query
            self.evidence_chunk_ids = (chunk,)

    class _Resolution:
        def __init__(self, events, requested) -> None:
            self.events = tuple(events)
            self.requested_fields = tuple(requested)

    def test_one_match_with_requested_fields_narrows(self) -> None:
        events = [self._Event(True), self._Event(False), self._Event(False)]

        reported = _reported_holding_events(
            self._Resolution(events, ["after_ratio"])
        )

        self.assertEqual(list(reported), [events[0]])

    def test_no_requested_field_keeps_every_event(self) -> None:
        events = [self._Event(True), self._Event(False)]

        reported = _reported_holding_events(self._Resolution(events, []))

        self.assertEqual(list(reported), events)

    def test_several_matches_keep_every_event(self) -> None:
        events = [self._Event(True), self._Event(True), self._Event(False)]

        reported = _reported_holding_events(
            self._Resolution(events, ["after_ratio"])
        )

        self.assertEqual(list(reported), events)

    def test_no_match_keeps_every_event(self) -> None:
        events = [self._Event(False), self._Event(False)]

        reported = _reported_holding_events(
            self._Resolution(events, ["after_ratio"])
        )

        self.assertEqual(list(reported), events)

    def test_an_undecided_match_is_not_treated_as_a_match(self) -> None:
        events = [self._Event(None), self._Event(True)]

        reported = _reported_holding_events(
            self._Resolution(events, ["after_ratio"])
        )

        self.assertEqual(list(reported), [events[1]])


class HoldingFactLineScopeTests(unittest.TestCase):
    """Field scoping, independent of any resolver or corpus."""

    EVENT = {
        "corp_name": "테스트회사",
        "reporter": "테스트기금",
        "reference_date": "2022-12-05",
        "report_date": "2022-12-07",
        "after_shares": {"raw": "720,039", "normalized": 720039},
        "after_ratio": {"raw": "7.12", "normalized": 7.12},
        "change_direction": "increase",
    }

    def test_no_request_renders_everything(self) -> None:
        lines = _holding_fact_lines(self.EVENT, "[1]")

        rendered = "\n".join(lines)
        self.assertIn("변동 후 주식수", rendered)
        self.assertIn("변동 후 비율", rendered)
        self.assertIn("보고일", rendered)
        self.assertIn("변동 방향", rendered)

    def test_a_request_reorders_but_never_deletes(self) -> None:
        lines = _holding_fact_lines(self.EVENT, "[1]", ["after_ratio"])

        rendered = "\n".join(lines)
        self.assertIn("변동 후 비율: 7.12%", rendered)
        self.assertIn("변동 후 주식수", rendered)
        self.assertIn("보고일", rendered)
        self.assertIn("변동 방향", rendered)

    def test_every_verified_value_survives_any_request(self) -> None:
        full = set(_holding_fact_lines(self.EVENT, "[1]"))

        for requested in (["after_ratio"], ["after_shares"], ["change_direction"]):
            with self.subTest(requested=requested):
                self.assertEqual(
                    set(_holding_fact_lines(self.EVENT, "[1]", requested)), full
                )

    def test_the_requested_field_leads_the_non_identity_lines(self) -> None:
        lines = _holding_fact_lines(self.EVENT, "[1]", ["after_ratio"])

        labels = [line.split(":", 1)[0] for line in lines]
        self.assertEqual(labels[:3], ["회사", "보고자", "변동일"])
        self.assertEqual(labels[3], "변동 후 비율")

    def test_rendering_is_stable_for_the_same_request(self) -> None:
        first = _holding_fact_lines(self.EVENT, "[1]", ["after_ratio"])
        second = _holding_fact_lines(self.EVENT, "[1]", ["after_ratio"])

        self.assertEqual(first, second)

    def test_context_fields_always_lead(self) -> None:
        lines = _holding_fact_lines(self.EVENT, "[1]", ["after_ratio"])

        rendered = "\n".join(lines)
        self.assertIn("회사: 테스트회사", rendered)
        self.assertIn("보고자: 테스트기금", rendered)
        self.assertIn("변동일: 2022-12-05", rendered)

    def test_values_are_never_reformatted(self) -> None:
        lines = _holding_fact_lines(self.EVENT, "[1]", ["after_shares"])

        self.assertIn("변동 후 주식수: 720,039주 [1]", lines)

    def test_every_requested_field_is_rendered(self) -> None:
        lines = _holding_fact_lines(
            self.EVENT, "[1]", ["after_shares", "after_ratio"]
        )

        rendered = "\n".join(lines)
        self.assertIn("720,039주", rendered)
        self.assertIn("7.12%", rendered)

    def test_a_requested_field_the_event_lacks_is_simply_absent(self) -> None:
        lines = _holding_fact_lines(self.EVENT, "[1]", ["before_shares"])

        self.assertNotIn("변동 전 주식수", "\n".join(lines))


if __name__ == "__main__":
    unittest.main()
