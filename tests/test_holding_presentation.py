from __future__ import annotations

import re
import unittest

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import (
    CitationAwareAnswerGenerator,
    _holding_fact_lines,
    _holding_prose_lines,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from tests.test_agent_end_to_end_smoke import _execution
from tests.test_evidence_builder import _holding_pair


#: Synthetic. No real company, reporter, date, or value is depended on.
INCREASE_EVENT = {
    "corp_name": "테스트회사",
    "reporter": "테스트기금",
    "reference_date": "2022-12-05",
    "receipt_date": "2023-01-03",
    "before_shares": {"raw": "613,758", "normalized": 613758},
    "change_shares": {"raw": "106,281", "normalized": 106281},
    "after_shares": {"raw": "720,039", "normalized": 720039},
    "before_ratio": {"raw": "6.07", "normalized": 6.07},
    "after_ratio": {"raw": "7.12", "normalized": 7.12},
    "change_ratio": {"raw": "1.05", "normalized": 1.05},
    "change_direction": "increase",
}

DECREASE_EVENT = {
    **INCREASE_EVENT,
    "reference_date": "2023-06-13",
    "before_shares": {"raw": "2,485,201", "normalized": 2485201},
    "change_shares": {"raw": "-283,151", "normalized": -283151},
    "after_shares": {"raw": "2,202,050", "normalized": 2202050},
    "before_ratio": {"raw": "9.13", "normalized": 9.13},
    "after_ratio": {"raw": "8.09", "normalized": 8.09},
    "change_ratio": {"raw": "1.04", "normalized": 1.04},
    "change_direction": "decrease",
}


def _prose(event, requested, marker: str = "[1]") -> str:
    lines = _holding_prose_lines(event, marker, requested)
    assert lines is not None, "prose renderer declined this event"
    return " ".join(lines)


def _record_values(event) -> list[str]:
    return [
        line.split(": ", 1)[1].rsplit(" [", 1)[0]
        for line in _holding_fact_lines(event, "[1]")
    ]


class CaseALeadsWithTheAskedRatioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _prose(INCREASE_EVENT, ["after_ratio"])

    def test_the_lead_sentence_answers_the_question(self) -> None:
        lead = self.text.split(". ")[0]

        self.assertIn("보유 비율은 7.12%", lead)
        self.assertIn("2022-12-05", lead)

    def test_the_answer_is_prose_not_a_record(self) -> None:
        for label in ("회사:", "보고자:", "변동일:", "변동 후 비율:"):
            with self.subTest(label=label):
                self.assertNotIn(label, self.text)

    def test_no_event_numbering(self) -> None:
        self.assertNotIn("1.\n", self.text)

    def test_every_verified_value_survives(self) -> None:
        for value in _record_values(INCREASE_EVENT):
            with self.subTest(value=value):
                self.assertIn(value, self.text)


class CaseBLeadsWithTheAskedSharesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _prose(INCREASE_EVENT, ["after_shares"])

    def test_shares_lead(self) -> None:
        self.assertIn("보유 주식수는 720,039주", self.text.split(". ")[0])

    def test_the_paired_ratio_context_survives(self) -> None:
        self.assertIn("7.12%", self.text)
        self.assertIn("6.07%", self.text)

    def test_every_verified_value_survives(self) -> None:
        for value in _record_values(INCREASE_EVENT):
            with self.subTest(value=value):
                self.assertIn(value, self.text)


class CaseCBothRequestedValuesLeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _prose(INCREASE_EVENT, ["after_shares", "after_ratio"])

    def test_both_asked_values_appear_before_the_movement(self) -> None:
        lead = self.text.split("보유 주식수는 변동 전 613,758주")[0]

        self.assertIn("720,039주", lead)
        self.assertIn("7.12%", lead)

    def test_every_verified_value_survives(self) -> None:
        for value in _record_values(INCREASE_EVENT):
            with self.subTest(value=value):
                self.assertIn(value, self.text)


class CaseDChangeFieldsLeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _prose(INCREASE_EVENT, ["change_shares", "change_ratio"])

    def test_the_change_leads(self) -> None:
        lead = self.text.split(". ")[0]

        self.assertIn("106,281주", lead)
        self.assertIn("증가", lead)

    def test_the_before_and_after_context_survives(self) -> None:
        self.assertIn("613,758주", self.text)
        self.assertIn("720,039주", self.text)
        self.assertIn("6.07%", self.text)
        self.assertIn("7.12%", self.text)

    def test_the_change_figure_is_not_repeated_in_the_movement(self) -> None:
        self.assertEqual(self.text.count("106,281"), 1)

    def test_every_verified_value_survives(self) -> None:
        for value in _record_values(INCREASE_EVENT):
            with self.subTest(value=value):
                self.assertIn(value, self.text)


class CaseEDecreaseEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = _prose(DECREASE_EVENT, ["after_shares"])

    def test_the_direction_is_stated_as_a_decrease(self) -> None:
        self.assertIn("감소했습니다", self.text)
        self.assertNotIn("증가했습니다", self.text)

    def test_the_signed_value_is_preserved_verbatim(self) -> None:
        self.assertIn("-283,151주", self.text)

    def test_the_movement_direction_matches_the_values(self) -> None:
        self.assertIn("변동 전 2,485,201주에서 변동 후 2,202,050주로", self.text)

    def test_the_ratio_falls(self) -> None:
        self.assertIn("하락했습니다", self.text)

    def test_every_verified_value_survives(self) -> None:
        for value in _record_values(DECREASE_EVENT):
            with self.subTest(value=value):
                self.assertIn(value, self.text)


class UnitTests(unittest.TestCase):
    def test_share_counts_keep_their_unit(self) -> None:
        self.assertIn("720,039주", _prose(INCREASE_EVENT, ["after_shares"]))

    def test_holding_ratios_use_percent(self) -> None:
        self.assertIn("7.12%", _prose(INCREASE_EVENT, ["after_ratio"]))

    def test_a_ratio_change_uses_percentage_points(self) -> None:
        self.assertIn("1.05%p", _prose(INCREASE_EVENT, ["after_ratio"]))

    def test_percentage_points_survive_gold_term_normalisation(self) -> None:
        """Gold terms are compared with every non-alphanumeric char stripped."""

        from app.agent.gold60_evaluation import _normalize_comparison_text

        answer = _normalize_comparison_text(_prose(INCREASE_EVENT, ["after_ratio"]))

        self.assertIn(_normalize_comparison_text("1.05%"), answer)
        self.assertIn(_normalize_comparison_text("1.05"), answer)

    def test_numbers_are_never_rounded_or_reformatted(self) -> None:
        text = _prose(INCREASE_EVENT, ["after_shares"])

        self.assertIn("720,039", text)
        self.assertNotIn("720039", text)
        self.assertNotIn("약 72만", text)


class CompletenessGuardTests(unittest.TestCase):
    """Prose is used only when it provably states every verified value."""

    def test_an_unmapped_direction_falls_back_to_the_record_form(self) -> None:
        event = {**INCREASE_EVENT, "change_direction": "unchanged"}

        self.assertIsNone(_holding_prose_lines(event, "[1]", ["after_ratio"]))

    def test_an_event_without_a_date_falls_back(self) -> None:
        event = {**INCREASE_EVENT}
        event.pop("reference_date")

        self.assertIsNone(_holding_prose_lines(event, "[1]", ["after_ratio"]))

    def test_an_event_without_a_company_falls_back(self) -> None:
        event = {**INCREASE_EVENT}
        event.pop("corp_name")

        self.assertIsNone(_holding_prose_lines(event, "[1]", ["after_ratio"]))

    def test_a_sparse_event_still_states_what_it_has(self) -> None:
        event = {
            "corp_name": "테스트회사",
            "reporter": "테스트기금",
            "reference_date": "2022-12-05",
            "after_ratio": {"raw": "7.12", "normalized": 7.12},
        }

        text = _prose(event, ["after_ratio"])

        self.assertIn("7.12%", text)
        for value in _record_values(event):
            with self.subTest(value=value):
                self.assertIn(value, text)

    def test_every_request_shape_preserves_every_value(self) -> None:
        shapes = (
            ["after_ratio"],
            ["after_shares"],
            ["after_shares", "after_ratio"],
            ["change_shares"],
            ["change_ratio"],
            ["before_shares", "after_shares"],
            [],
        )

        for requested in shapes:
            with self.subTest(requested=requested):
                text = _prose(INCREASE_EVENT, requested)
                for value in _record_values(INCREASE_EVENT):
                    self.assertIn(value, text)


class CaseHCitationTests(unittest.TestCase):
    def test_every_sentence_carries_its_supporting_citation(self) -> None:
        lines = _holding_prose_lines(INCREASE_EVENT, "[1] [2]", ["after_ratio"])

        for line in lines:
            with self.subTest(line=line):
                self.assertTrue(line.endswith("[1] [2]"))

    def test_the_marker_is_not_repeated_after_every_value(self) -> None:
        text = _prose(INCREASE_EVENT, ["after_ratio"], marker="[1]")
        record = "\n".join(_holding_fact_lines(INCREASE_EVENT, "[1]"))

        self.assertLess(text.count("[1]"), record.count("[1]"))

    def test_the_marker_is_never_invented(self) -> None:
        text = _prose(INCREASE_EVENT, ["after_ratio"], marker="[1]")

        self.assertEqual(set(re.findall(r"\[\d+\]", text)), {"[1]"})


def _ask(question: str, *, period=None, metric=None, events=3):
    pairs = []
    values = (
        ("2022-12-05", "720,039", "7.12"),
        ("2023-06-30", "801,200", "7.90"),
        ("2024-06-30", "650,100", "6.40"),
    )[:events]
    for index, (date, shares, ratio) in enumerate(values):
        pair = _holding_pair(
            f"h{index}:ch", f"h{index}", rank=index + 1, date=date,
            projection_type="holding_report", table_id=f"t{index}",
        )
        chunk = pair[0].chunk
        chunk["projection_fields"]["보유주식수"] = shares
        chunk["projection_fields"]["보유비율"] = ratio
        chunk["content"] = f"국민연금기금 {date} 보유주식수 {shares} 보유비율 {ratio}"
        pairs.append(pair)
    plan = QueryPlan(
        query=question,
        task_type="holding_change",
        metric=metric,
        reporter="국민연금기금",
        disclosure_route=("holding",),
        period=period,
    )
    result = AgentOrchestrator().run(question, plan, _execution(plan, *pairs))
    return result, CitationAwareAnswerGenerator().generate(result.answer_draft)


class CaseFNoMatchingEventTests(unittest.TestCase):
    def test_an_unmatched_date_keeps_the_unsupported_behaviour(self) -> None:
        result, generated = _ask(
            "2021년 3월 1일 기준 국민연금의 보유 비율은?",
            period=QueryPeriod(year=2021, from_date="2021-03-01", to_date="2021-03-01"),
            metric="holding_ratio",
        )

        self.assertEqual(result.resolution.matching_event_count, 0)
        self.assertFalse(generated.answerable)
        self.assertIn("answer_not_supported", generated.warnings)


class CaseGMultiEventCompactTests(unittest.TestCase):
    """Multi-event answers keep every event, on one line each."""

    def setUp(self) -> None:
        self.result, self.generated = _ask("국민연금 보유 변동 내역을 알려줘")

    def test_the_repeated_label_blocks_are_gone(self) -> None:
        text = self.generated.answer_text

        for label in ("회사:", "변동일:", "변동 후 주식수:", "변동 후 비율:"):
            with self.subTest(label=label):
                self.assertNotIn(label, text)

    def test_events_are_no_longer_numbered(self) -> None:
        text = self.generated.answer_text

        self.assertNotIn("\n1.\n", text)
        self.assertNotIn("\n2.\n", text)

    def test_every_event_is_still_reported(self) -> None:
        for date in ("2022-12-05", "2023-06-30", "2024-06-30"):
            with self.subTest(date=date):
                self.assertIn(date, self.generated.answer_text)

    def test_one_line_per_event(self) -> None:
        body = self.generated.sections[0].content.splitlines()

        # A header plus one row per event.
        self.assertEqual(len(body), 1 + 3)

    def test_every_verified_value_is_still_present(self) -> None:
        text = self.generated.answer_text

        for shares, ratio in (
            ("720,039", "7.12"),
            ("801,200", "7.90"),
            ("650,100", "6.40"),
        ):
            with self.subTest(shares=shares):
                self.assertIn(f"{shares}주", text)
                self.assertIn(f"{ratio}%", text)

    def test_the_ambiguity_warning_is_kept(self) -> None:
        self.assertIn("특정 시점을 자동 선택하지 않았습니다", self.generated.answer_text)
        self.assertIn(
            "multiple_matching_holding_events", self.result.answer_draft.warnings
        )


class CaseIEndToEndSingleEventTests(unittest.TestCase):
    """The reported question, through the production reasoning path."""

    def setUp(self) -> None:
        self.result, self.generated = _ask(
            "2022년 12월 5일 기준 국민연금의 보유 비율은?",
            period=QueryPeriod(year=2022, from_date="2022-12-05", to_date="2022-12-05"),
            metric="holding_ratio",
        )

    def test_only_the_asked_date_is_reported(self) -> None:
        text = self.generated.answer_text

        self.assertIn("2022-12-05", text)
        self.assertNotIn("2023-06-30", text)
        self.assertNotIn("2024-06-30", text)

    def test_the_answer_reads_as_prose(self) -> None:
        text = self.generated.answer_text

        self.assertIn("보유 비율은 7.12%입니다", text)
        self.assertNotIn("변동 후 비율:", text)

    def test_the_preamble_and_numbering_are_dropped(self) -> None:
        text = self.generated.answer_text

        self.assertNotIn("확인된 보유 변동 내역은 다음과 같습니다.", text)

    def test_the_complementary_share_count_survives(self) -> None:
        self.assertIn("720,039주", self.generated.answer_text)

    def test_warnings_are_untouched(self) -> None:
        self.assertNotIn("answer_not_supported", self.generated.warnings)
        self.assertTrue(self.generated.answerable)

    def test_the_citation_block_is_still_available_to_consumers(self) -> None:
        text = self.generated.answer_text

        self.assertIn("인용", text)
        self.assertIn("chunk_id:", text)
        self.assertEqual(len(self.generated.citations), 1)


class TheMovementSentenceNamesTheRoleOfEachValue(unittest.TestCase):
    """IEV2-C097: "변동 전과 변동 후 주식수는?" -- both numbers, each labelled.

    The movement sentence used to read "보유 주식수는 613,758주에서 720,039주로
    106,281주 증가했습니다".  Word order is the only thing that said which
    figure was the before and which the after: no reader-visible label bound
    either number to its role, and the lead sentence states the after value
    with no role word at all.  A reader who scans for "변동 후" finds nothing,
    and a reader who mis-scans the "에서 ... 로" ordering has nothing to check
    themselves against -- in a sentence whose whole subject is a change, the
    two figures are exactly what must not be confusable.

    The record form has always labelled them ("변동 전 주식수: ...").  Prose,
    which is what a single event actually renders, said the same facts without
    the same labels.  These tests pin the labels into the prose.
    """

    def test_the_before_value_is_named_as_the_before_value(self) -> None:
        text = _prose(INCREASE_EVENT, [])

        self.assertIn("변동 전 613,758주", text)

    def test_the_after_value_is_named_as_the_after_value(self) -> None:
        text = _prose(INCREASE_EVENT, [])

        self.assertIn("변동 후 720,039주", text)

    def test_the_sentence_still_reads_as_one_movement(self) -> None:
        text = _prose(INCREASE_EVENT, [])

        self.assertIn(
            "보유 주식수는 변동 전 613,758주에서 변동 후 720,039주로 "
            "106,281주 증가했습니다.",
            text,
        )

    def test_a_decrease_is_labelled_the_same_way(self) -> None:
        text = _prose(DECREASE_EVENT, [])

        self.assertIn("변동 전 2,485,201주", text)
        self.assertIn("변동 후 2,202,050주", text)

    def test_the_ratio_movement_is_labelled_too(self) -> None:
        text = _prose(INCREASE_EVENT, [])

        self.assertIn("변동 전 6.07%", text)
        self.assertIn("변동 후 7.12%", text)

    def test_no_verified_value_was_lost_to_the_labels(self) -> None:
        for requested in ([], ["after_shares"], ["before_shares", "after_shares"]):
            text = _prose(INCREASE_EVENT, requested)
            for value in _record_values(INCREASE_EVENT):
                with self.subTest(requested=tuple(requested), value=value):
                    self.assertIn(value, text)


def _role_window(text: str, markers, width: int = 60) -> str:
    """The independent evaluator's own reading, restated here rather than imported.

    ``evaluation/`` is not an importable package from the app test suite, and
    copying the predicate keeps this suite honest about what it is asserting:
    a value counts as bound to a role only when it follows that role's own
    word closely enough to be read as belonging to it.
    """

    spans = []
    for marker in markers:
        for match in re.finditer(re.escape(marker), text):
            spans.append(text[match.start() : match.end() + width])
    return " ".join(spans)


_BEFORE_MARKERS = ("변동 전", "변동전", "직전")
_AFTER_MARKERS = ("변동 후", "변동후", "이후")


class EachFigureIsReadableUnderItsOwnRole(unittest.TestCase):
    """The failure IEV2-C097 reported, restated as the property it tests.

    ``role_unbound`` fires when neither figure is stated near a word naming its
    role.  What made it reachable is that the question's own wording -- 변동
    전과 변동 후 -- need not appear anywhere in an answer that reports exactly
    those two numbers.
    """

    def _windows(self, text: str) -> tuple[str, str]:
        return (
            _role_window(text, _BEFORE_MARKERS),
            _role_window(text, _AFTER_MARKERS),
        )

    def test_both_roles_are_bound_when_the_question_named_neither_field(self) -> None:
        before_window, after_window = self._windows(_prose(INCREASE_EVENT, []))

        self.assertIn("613,758", before_window)
        self.assertIn("720,039", after_window)

    def test_both_roles_are_bound_when_the_question_named_both_fields(self) -> None:
        before_window, after_window = self._windows(
            _prose(INCREASE_EVENT, ["before_shares", "after_shares"])
        )

        self.assertIn("613,758", before_window)
        self.assertIn("720,039", after_window)

    def test_the_after_role_window_is_no_longer_empty(self) -> None:
        """The precise shape of the live failure: nothing named the after role."""

        _before_window, after_window = self._windows(_prose(INCREASE_EVENT, []))

        self.assertNotEqual(after_window, "")

    def test_a_decrease_binds_its_roles_the_same_way(self) -> None:
        before_window, after_window = self._windows(_prose(DECREASE_EVENT, []))

        self.assertIn("2,485,201", before_window)
        self.assertIn("2,202,050", after_window)


if __name__ == "__main__":
    unittest.main()
