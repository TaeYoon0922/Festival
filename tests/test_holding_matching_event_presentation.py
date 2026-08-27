"""P1-A5-B: an answer may only show events that satisfy the question.

The resolver already decides, from the question's own reporter, date and
direction constraints, which retrieved events satisfy it.  The composer used to
throw that away whenever it could not narrow to exactly one, so a question
asking about decreases was answered with increases, and a question naming a
holder was answered with rows belonging to someone else.

These tests pin the corrected contract and the three behaviours around it that
must not move: one match still narrows (P1-A4 depends on it), no match still
falls back, and a history question still keeps its whole timeline.
"""

import unittest

from app.reasoning.answer_composer import _reported_holding_events


class _Event:
    """The only fields the reporting rule is allowed to read."""

    def __init__(self, matches_query, *, chunk="c", reporter=None,
                 reference_date=None, direction=None) -> None:
        self.matches_query = matches_query
        self.evidence_chunk_ids = (chunk,)
        self.reporter = reporter
        self.reference_date = reference_date
        self.change_direction = direction


class _Resolution:
    def __init__(self, events, requested=("after_shares",), reporter=None) -> None:
        self.events = tuple(events)
        self.requested_fields = tuple(requested)
        self.reporter_constraint = reporter


class SingleMatchTests(unittest.TestCase):
    """1: the P1-A4 path, which must not move."""

    def test_one_match_still_narrows_to_that_event(self) -> None:
        events = [_Event(False), _Event(True), _Event(False)]

        reported = _reported_holding_events(_Resolution(events))

        self.assertEqual(list(reported), [events[1]])

    def test_a_lone_match_is_returned_even_when_it_ranks_last(self) -> None:
        events = [_Event(False), _Event(False), _Event(True)]

        self.assertEqual(list(_reported_holding_events(_Resolution(events))),
                         [events[2]])


class SeveralMatchTests(unittest.TestCase):
    """2, 3: the corrected branch."""

    def test_every_match_is_reported_and_nothing_else(self) -> None:
        events = [_Event(True), _Event(False), _Event(True), _Event(True)]

        reported = _reported_holding_events(_Resolution(events))

        self.assertEqual(list(reported), [events[0], events[2], events[3]])
        self.assertTrue(all(e.matches_query is True for e in reported))

    def test_the_matching_subset_keeps_its_original_order(self) -> None:
        """3: filtering must not reorder what survives."""

        events = [_Event(False), _Event(True, chunk="a"), _Event(False),
                  _Event(True, chunk="b"), _Event(True, chunk="c")]

        reported = _reported_holding_events(_Resolution(events))

        self.assertEqual([e.evidence_chunk_ids[0] for e in reported],
                         ["a", "b", "c"])

    def test_an_undecided_event_is_not_a_match(self) -> None:
        events = [_Event(None), _Event(True), _Event(None)]

        self.assertEqual(list(_reported_holding_events(_Resolution(events))),
                         [events[1]])


class NoMatchTests(unittest.TestCase):
    """4: the fallback is unchanged."""

    def test_nothing_matching_still_shows_what_was_retrieved(self) -> None:
        events = [_Event(False), _Event(False)]

        self.assertEqual(list(_reported_holding_events(_Resolution(events))),
                         events)

    def test_all_undecided_still_shows_what_was_retrieved(self) -> None:
        events = [_Event(None), _Event(None)]

        self.assertEqual(list(_reported_holding_events(_Resolution(events))),
                         events)

    def test_a_holder_absent_from_every_event_still_reports_nothing(self) -> None:
        """The pre-existing reporter guard is untouched."""

        events = [_Event(False, reporter="한바다"), _Event(False, reporter="두리안")]

        reported = _reported_holding_events(
            _Resolution(events, reporter="가나연금"))

        self.assertEqual(list(reported), [])


class HistoryPathTests(unittest.TestCase):
    """8: a question naming no field keeps its whole timeline."""

    def test_without_requested_fields_every_event_is_kept(self) -> None:
        events = [_Event(True), _Event(False), _Event(None)]

        self.assertEqual(
            list(_reported_holding_events(_Resolution(events, requested=()))),
            events)


class ConstraintExclusionTests(unittest.TestCase):
    """5, 6, 7: each constraint the resolver applies is honoured downstream.

    The rule reads only ``matches_query``; these fixtures carry the constraint
    that produced it so a failure names the semantics that broke.
    """

    def test_a_holder_mismatch_is_excluded_when_others_match(self) -> None:
        wanted = _Event(True, reporter="가나연금기금")
        stranger = _Event(False, reporter="한바다")
        other = _Event(True, reporter="가나연금공단")

        reported = _reported_holding_events(
            _Resolution([wanted, stranger, other], reporter="가나연금"))

        self.assertEqual(list(reported), [wanted, other])
        self.assertNotIn(stranger, reported)

    def test_a_direction_mismatch_is_excluded_when_others_match(self) -> None:
        down_a = _Event(True, direction="decrease")
        up = _Event(False, direction="increase")
        down_b = _Event(True, direction="decrease")

        reported = _reported_holding_events(_Resolution([down_a, up, down_b]))

        self.assertEqual(list(reported), [down_a, down_b])
        self.assertTrue(all(e.change_direction == "decrease" for e in reported))

    def test_a_date_mismatch_is_excluded_when_others_match(self) -> None:
        inside_a = _Event(True, reference_date="2023-06-13")
        outside = _Event(False, reference_date="2024-08-26")
        inside_b = _Event(True, reference_date="2023-07-12")

        reported = _reported_holding_events(
            _Resolution([inside_a, outside, inside_b]))

        self.assertEqual(list(reported), [inside_a, inside_b])
        self.assertNotIn(outside, reported)


class NoRankSelectionTests(unittest.TestCase):
    """9, 10: position in the list carries no authority."""

    def test_the_first_event_is_dropped_when_it_does_not_match(self) -> None:
        first = _Event(False)
        second = _Event(True)
        third = _Event(True)

        reported = _reported_holding_events(_Resolution([first, second, third]))

        self.assertNotIn(first, reported)
        self.assertEqual(list(reported), [second, third])

    def test_several_matches_are_not_collapsed_to_the_first_of_them(self) -> None:
        events = [_Event(True), _Event(True), _Event(True)]

        reported = _reported_holding_events(_Resolution(events))

        self.assertEqual(len(reported), 3,
                         "no event may be chosen for coming first")

    def test_neither_the_newest_nor_the_oldest_is_preferred(self) -> None:
        oldest = _Event(True, reference_date="2022-01-01")
        middle = _Event(True, reference_date="2023-01-01")
        newest = _Event(True, reference_date="2024-01-01")

        reported = _reported_holding_events(
            _Resolution([newest, oldest, middle]))

        self.assertEqual(list(reported), [newest, oldest, middle],
                         "order is presentation only; nothing is date-ranked")


class NoDomainLiteralTests(unittest.TestCase):
    """The rule may name no question, holder, company, date or identifier."""

    def test_the_reporting_rule_carries_no_domain_literals(self) -> None:
        import inspect

        source = inspect.getsource(_reported_holding_events)
        # The prose deliberately names what the rule refuses to do, so only the
        # executable body is checked for the signals it must never act on.
        body = source.split('"""')[-1]
        for literal in ("HX", "국민연금", "이마트", "LG생활건강", "효성중공업",
                        "파마리서치", "2023-", "2024-", "holding_20",
                        "rank", "latest", "newest", "earliest", "sort"):
            self.assertNotIn(literal, body,
                             f"the rule must not act on {literal!r}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
