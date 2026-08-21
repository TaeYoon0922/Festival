from __future__ import annotations

import re
import unittest

from unittest import mock

from app.generation.answer_generator import (
    _holding_compact_lines,
    _holding_compact_row,
    _holding_compact_row_states_every_value,
    _holding_fact_lines,
)


#: Synthetic. No real company, reporter, date, value, or document is used.
def _event(
    date: str,
    *,
    reporter: str = "테스트기금",
    corp: str = "테스트회사",
    before_shares: str | None = "613,758",
    change_shares: str | None = "106,281",
    after_shares: str | None = "720,039",
    before_ratio: str | None = "6.07",
    change_ratio: str | None = "1.05",
    after_ratio: str | None = "7.12",
    direction: str | None = "increase",
    receipt: str | None = "2023-01-03",
    report: str | None = None,
) -> dict:
    def numeric(raw):
        return None if raw is None else {"raw": raw, "normalized": 0}

    event = {
        "corp_name": corp,
        "reporter": reporter,
        "reference_date": date,
        "report_date": report,
        "receipt_date": receipt,
        "before_shares": numeric(before_shares),
        "change_shares": numeric(change_shares),
        "after_shares": numeric(after_shares),
        "before_ratio": numeric(before_ratio),
        "change_ratio": numeric(change_ratio),
        "after_ratio": numeric(after_ratio),
        "change_direction": direction,
    }
    return {key: value for key, value in event.items() if value is not None}


def _markers(count: int) -> list[tuple[str, ...]]:
    return [(f"[{index}]",) for index in range(1, count + 1)]


def _render(events) -> list[str]:
    lines = _holding_compact_lines(events, _markers(len(events)))
    assert lines is not None, "compact renderer declined this event set"
    return lines


def _record_values(event) -> list[str]:
    return [
        line.split(": ", 1)[1].rsplit(" [", 1)[0]
        for line in _holding_fact_lines(event, "[x]")
    ]


def _assert_all_values_present(case, events, body: str) -> None:
    for index, event in enumerate(events, start=1):
        for value in _record_values(event):
            with case.subTest(event=index, value=value):
                case.assertIn(value, body)


TWO_EVENTS = [
    _event("2024-06-27", before_shares="716,596", change_shares="-112,792",
           after_shares="603,804", before_ratio="6.92", change_ratio="-1.12",
           after_ratio="5.80", direction="decrease", receipt="2024-07-02"),
    _event("2024-03-26", before_shares="613,758", change_shares="102,838",
           after_shares="716,596", before_ratio="5.93", change_ratio="0.99",
           after_ratio="6.92", direction="increase", receipt="2024-04-02"),
]


class CaseATwoEventsAreCompactTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lines = _render(TWO_EVENTS)
        self.body = "\n".join(self.lines)

    def test_a_header_plus_one_line_per_event(self) -> None:
        self.assertEqual(len(self.lines), 3)

    def test_the_repeated_label_blocks_are_gone(self) -> None:
        for label in ("회사:", "보고자:", "변동일:", "변동 후 주식수:", "변동 방향:"):
            with self.subTest(label=label):
                self.assertNotIn(label, self.body)

    def test_the_header_names_the_company_once(self) -> None:
        self.assertIn("테스트회사", self.lines[0])
        self.assertEqual(self.body.count("테스트회사"), 1)

    def test_it_is_shorter_than_the_record_form(self) -> None:
        record = sum(len(_holding_fact_lines(event, "[x]")) for event in TWO_EVENTS)

        self.assertLess(len(self.lines), record)


class CaseBTenEventsAllSurviveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            _event(f"2024-{month:02d}-15", after_shares=f"{100 + month},000")
            for month in range(1, 11)
        ]
        self.lines = _render(self.events)
        self.body = "\n".join(self.lines)

    def test_every_event_gets_its_own_line(self) -> None:
        self.assertEqual(len(self.lines), 1 + 10)

    def test_no_event_is_dropped(self) -> None:
        for month in range(1, 11):
            with self.subTest(month=month):
                self.assertIn(f"2024-{month:02d}-15", self.body)

    def test_every_verified_value_survives(self) -> None:
        _assert_all_values_present(self, self.events, self.body)


class CaseCShareFactsTests(unittest.TestCase):
    def test_before_change_and_after_shares_all_survive(self) -> None:
        body = "\n".join(_render(TWO_EVENTS))

        for value in ("716,596주", "-112,792주", "603,804주", "613,758주", "102,838주"):
            with self.subTest(value=value):
                self.assertIn(value, body)

    def test_the_movement_reads_from_before_to_after(self) -> None:
        body = "\n".join(_render(TWO_EVENTS))

        self.assertIn("716,596주 → 603,804주", body)

    def test_a_missing_share_field_does_not_invent_one(self) -> None:
        events = [
            _event("2024-06-27", before_shares=None, change_shares=None),
            _event("2024-03-26"),
        ]

        body = "\n".join(_render(events))

        self.assertIn("720,039주", body)
        _assert_all_values_present(self, events, body)


class CaseDRatioFactsTests(unittest.TestCase):
    def test_before_change_and_after_ratios_all_survive(self) -> None:
        body = "\n".join(_render(TWO_EVENTS))

        for value in ("6.92%", "-1.12%", "5.80%", "5.93%", "0.99%", "6.92%"):
            with self.subTest(value=value):
                self.assertIn(value, body)

    def test_a_ratio_change_is_stated_in_percentage_points(self) -> None:
        body = "\n".join(_render(TWO_EVENTS))

        self.assertIn("(-1.12%p)", body)

    def test_percentage_points_survive_gold_term_normalisation(self) -> None:
        from app.agent.gold60_evaluation import _normalize_comparison_text

        body = _normalize_comparison_text("\n".join(_render(TWO_EVENTS)))

        self.assertIn(_normalize_comparison_text("-1.12%"), body)


class CaseEDirectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.body = "\n".join(_render(TWO_EVENTS))

    def test_a_decrease_is_labelled(self) -> None:
        self.assertIn("감소", self.body)

    def test_an_increase_is_labelled(self) -> None:
        self.assertIn("증가", self.body)

    def test_the_direction_sits_with_its_own_event(self) -> None:
        decrease_line = next(
            line for line in self.body.splitlines() if line.startswith("2024-06-27")
        )

        self.assertIn("감소", decrease_line)
        self.assertNotIn("증가", decrease_line)

    def test_signed_values_are_preserved_verbatim(self) -> None:
        self.assertIn("-112,792주", self.body)

    def test_an_unchanged_direction_is_still_stated(self) -> None:
        events = [
            _event("2024-06-27", change_shares="0", change_ratio="0",
                   direction="unchanged", after_shares="613,758", after_ratio="5.93",
                   before_shares="613,758", before_ratio="5.93"),
            _event("2024-03-26"),
        ]

        body = "\n".join(_render(events))

        self.assertIn("변동 없음", body)
        _assert_all_values_present(self, events, body)


class CaseFDistinctReportersTests(unittest.TestCase):
    """Reporters that merely look similar are never merged."""

    def setUp(self) -> None:
        self.events = [
            _event("2024-06-27", reporter="테스트공단"),
            _event("2024-03-26", reporter="테스트기금"),
        ]
        self.body = "\n".join(_render(self.events))

    def test_each_reporter_is_kept_on_its_own_row(self) -> None:
        self.assertIn("테스트공단", self.body)
        self.assertIn("테스트기금", self.body)

    def test_the_reporter_stays_with_its_event(self) -> None:
        first = next(
            line for line in self.body.splitlines() if line.startswith("2024-06-27")
        )

        self.assertIn("테스트공단", first)
        self.assertNotIn("테스트기금", first)

    def test_distinct_companies_keep_their_names_per_row(self) -> None:
        events = [
            _event("2024-06-27", corp="테스트회사"),
            _event("2024-03-26", corp="다른회사"),
        ]

        body = "\n".join(_render(events))

        self.assertIn("테스트회사", body)
        self.assertIn("다른회사", body)
        _assert_all_values_present(self, events, body)


class CaseGCitationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lines = _render(TWO_EVENTS)

    def test_each_row_ends_with_its_own_citation(self) -> None:
        self.assertTrue(self.lines[1].endswith("[1]"))
        self.assertTrue(self.lines[2].endswith("[2]"))

    def test_a_row_carries_no_other_events_citation(self) -> None:
        self.assertNotIn("[2]", self.lines[1])
        self.assertNotIn("[1]", self.lines[2])

    def test_no_citation_is_invented(self) -> None:
        markers = set(re.findall(r"\[\d+\]", "\n".join(self.lines)))

        self.assertEqual(markers, {"[1]", "[2]"})

    def test_multiple_markers_for_one_event_are_kept(self) -> None:
        lines = _holding_compact_lines(TWO_EVENTS, [("[1]", "[2]"), ("[3]",)])

        self.assertTrue(lines[1].endswith("[1] [2]"))

    def test_an_event_without_provenance_declines_the_compact_form(self) -> None:
        self.assertIsNone(_holding_compact_lines(TWO_EVENTS, [("[1]",), ()]))


class CaseHCompletenessGuardTests(unittest.TestCase):
    """All events compact, or all events in the record form. Never a mix."""

    def test_an_event_without_a_date_declines_the_whole_set(self) -> None:
        broken = {key: value for key, value in TWO_EVENTS[0].items()}
        broken.pop("reference_date")

        self.assertIsNone(
            _holding_compact_lines([broken, TWO_EVENTS[1]], _markers(2))
        )

    def test_a_mismatched_marker_count_declines(self) -> None:
        self.assertIsNone(_holding_compact_lines(TWO_EVENTS, _markers(1)))

    def test_an_empty_event_set_declines(self) -> None:
        self.assertIsNone(_holding_compact_lines([], []))

    def test_every_event_shape_preserves_every_value(self) -> None:
        shapes = (
            TWO_EVENTS,
            [_event("2024-06-27"), _event("2024-03-26", direction="decrease")],
            [_event("2024-06-27", report="2024-07-01"), _event("2024-03-26")],
            [_event("2024-06-27", receipt=None), _event("2024-03-26")],
            [
                _event("2024-06-27", before_ratio=None, change_ratio=None),
                _event("2024-03-26"),
            ],
        )

        for index, events in enumerate(shapes):
            with self.subTest(shape=index):
                _assert_all_values_present(self, events, "\n".join(_render(events)))

    def test_the_guard_is_what_decides_not_the_row_builder(self) -> None:
        """A value the row cannot express must veto the whole set."""

        events = [
            {**TWO_EVENTS[0], "report_date": "2024-07-01"},
            TWO_EVENTS[1],
        ]

        body = "\n".join(_render(events))

        self.assertIn("2024-07-01", body)


class OrderAndSetTests(unittest.TestCase):
    """The event set and its order come from the resolver, untouched."""

    def test_rows_follow_the_given_order(self) -> None:
        lines = _render(TWO_EVENTS)

        self.assertTrue(lines[1].startswith("2024-06-27"))
        self.assertTrue(lines[2].startswith("2024-03-26"))

    def test_reversing_the_input_reverses_the_rows(self) -> None:
        lines = _render(list(reversed(TWO_EVENTS)))

        self.assertTrue(lines[1].startswith("2024-03-26"))
        self.assertTrue(lines[2].startswith("2024-06-27"))

    def test_duplicate_looking_events_are_never_merged(self) -> None:
        events = [
            _event("2024-06-27", reporter="테스트공단"),
            _event("2024-06-27", reporter="테스트기금"),
        ]

        lines = _render(events)

        self.assertEqual(len(lines), 3)


if __name__ == "__main__":
    unittest.main()


class RowLocalCompletenessTests(unittest.TestCase):
    """A value must appear in its own row, not merely somewhere in the answer.

    Holding data reuses figures constantly, so a body-wide substring check
    passes when a neighbouring row happens to carry the missing value.
    """

    def _row(self, event, marker=("[1]",), *, omit_company=True):
        row = _holding_compact_row(event, marker, omit_company=omit_company)
        assert row is not None
        return row

    def _states_every_value(self, event, row, marker=("[1]",), **kwargs):
        options = {"header": "테스트회사에 대해 확인된 보유 변동은 다음과 같습니다.",
                   "company_hoisted": True}
        options.update(kwargs)
        return _holding_compact_row_states_every_value(event, row, marker, **options)

    def test_a_complete_row_passes(self) -> None:
        event = TWO_EVENTS[0]

        self.assertTrue(self._states_every_value(event, self._row(event)))

    def test_case_a_a_duplicate_ratio_cannot_cover_a_missing_one(self) -> None:
        """Both events share change_ratio; A losing it must still fail."""

        shared = "1.05"
        event_a = _event("2024-06-27", change_ratio=shared)
        event_b = _event("2024-03-26", change_ratio=shared)
        row_a = self._row(event_a).replace(f"({shared}%p)", "")
        row_b = self._row(event_b, ("[2]",))

        self.assertIn(f"{shared}%p", row_b)
        self.assertFalse(self._states_every_value(event_a, row_a))

    def test_case_b_a_repeated_share_count_cannot_cover_a_missing_one(self) -> None:
        """event1.after_shares == event2.before_shares, a real data shape."""

        shared = "716,596"
        event_one = _event("2024-06-27", after_shares=shared)
        event_two = _event("2024-03-26", before_shares=shared)
        row_two = self._row(event_two, ("[2]",)).replace(f"{shared}주", "")

        self.assertIn(f"{shared}주", self._row(event_one))
        self.assertFalse(
            self._states_every_value(event_two, row_two, ("[2]",))
        )

    def test_case_c_a_repeated_reporter_is_still_required_per_row(self) -> None:
        reporter = "테스트기금"
        event = _event("2024-06-27", reporter=reporter)
        row = self._row(event).replace(reporter, "")

        self.assertFalse(self._states_every_value(event, row))

    def test_case_d_a_row_must_carry_its_own_citation(self) -> None:
        event = TWO_EVENTS[0]
        row = self._row(event, ("[1]",)).replace("[1]", "[2]")

        self.assertFalse(self._states_every_value(event, row, ("[1]",)))

    def test_case_d_an_empty_marker_fails(self) -> None:
        event = TWO_EVENTS[0]

        self.assertFalse(self._states_every_value(event, self._row(event), ()))

    def test_case_d_every_marker_of_a_multi_cited_event_is_required(self) -> None:
        event = TWO_EVENTS[0]
        row = self._row(event, ("[1]", "[2]")).replace(" [2]", "")

        self.assertFalse(self._states_every_value(event, row, ("[1]", "[2]")))

    def test_case_e_a_hoisted_company_counts_from_the_header(self) -> None:
        event = _event("2024-06-27", corp="테스트회사")
        row = self._row(event)

        self.assertNotIn("테스트회사", row)
        self.assertTrue(self._states_every_value(event, row))

    def test_case_e_a_hoisted_company_absent_from_the_header_fails(self) -> None:
        event = _event("2024-06-27", corp="테스트회사")

        self.assertFalse(
            self._states_every_value(
                event, self._row(event), header="다른 회사의 보유 변동입니다."
            )
        )

    def test_case_f_distinct_companies_must_appear_in_their_own_row(self) -> None:
        event = _event("2024-06-27", corp="테스트회사")
        row = self._row(event, omit_company=False).replace("테스트회사", "")

        self.assertFalse(
            self._states_every_value(event, row, company_hoisted=False)
        )

    def test_case_f_a_company_kept_in_its_row_passes(self) -> None:
        event = _event("2024-06-27", corp="다른회사")

        self.assertTrue(
            self._states_every_value(
                event,
                self._row(event, omit_company=False),
                header="확인된 보유 변동은 다음과 같습니다.",
                company_hoisted=False,
            )
        )


class RowLocalFallbackTests(unittest.TestCase):
    """One failing row discards the compact form for the whole set."""

    def test_case_g_a_lossy_row_falls_back_for_every_event(self) -> None:
        shared = "1.05"
        events = [
            _event("2024-06-27", change_ratio=shared),
            _event("2024-03-26", change_ratio=shared),
        ]
        real_row = _holding_compact_row

        def lossy(event, marker, *, omit_company):
            row = real_row(event, marker, omit_company=omit_company)
            if event is events[0]:
                return row.replace(f"({shared}%p)", "")
            return row

        with mock.patch(
            "app.generation.answer_generator._holding_compact_row", lossy
        ):
            self.assertIsNone(_holding_compact_lines(events, _markers(2)))

    def test_a_lossy_row_reaches_the_record_form_end_to_end(self) -> None:
        events = [_event("2024-06-27"), _event("2024-03-26")]
        real_row = _holding_compact_row

        def lossy(event, marker, *, omit_company):
            row = real_row(event, marker, omit_company=omit_company)
            return row.replace("테스트기금", "") if event is events[1] else row

        with mock.patch(
            "app.generation.answer_generator._holding_compact_row", lossy
        ):
            self.assertIsNone(_holding_compact_lines(events, _markers(2)))

    def test_an_intact_set_still_renders_compactly(self) -> None:
        self.assertIsNotNone(_holding_compact_lines(TWO_EVENTS, _markers(2)))
