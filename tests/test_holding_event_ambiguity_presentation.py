"""P1-A5-A: an answer may only claim to have found *the* event when asked for one.

The resolver reports how many events satisfy the question.  That count is a fact
about what retrieval served, not about what was asked: a holder alone leaves
several events in most of the corpus, and a single observed event is routinely
produced by which projections happened to be returned.  So a single-event answer
requires two things -- the question named a date, and exactly one event answers
to it.  Everything else is shown with a caveat that names what the question left
open, and never claims more than was seen.
"""

import unittest

from app.generation.answer_generator import (
    _HOLDING_EXACT_MANY,
    _HOLDING_OBSERVED_MANY,
    _HOLDING_UNDER_SPECIFIED_MANY,
    _HOLDING_UNDER_SPECIFIED_ONE,
    generate_answer,
)
from app.reasoning.answer_composer import compose_holding_answer
from app.reasoning.holding_event_selection import (
    EXACT,
    FILTERED,
    NOT_APPLICABLE,
    UNSPECIFIED,
    classify_holding_event_selection,
    is_semantically_unique,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan

EXACT_PERIOD = "holding_reference_date"
BLANK_LINE = chr(10) + chr(10)


def _plan(question, *, task_type="disclosure_lookup", reporter=None,
          period=None, route=("holding",), evidence=None):
    return QueryPlan(
        query=question, raw_query=question, company="테스트회사",
        corp_code="00000001", task_type=task_type, reporter=reporter,
        disclosure_route=route, period=period,
        evidence=evidence if evidence is not None else {"operation": "lookup_holding"},
    )


# ------------------------------------------------------------------ the helper


class SelectionModeTests(unittest.TestCase):
    """What the question, on its own, says about which event is wanted."""

    def _mode(self, question, **kwargs):
        return classify_holding_event_selection(
            _plan(question, **kwargs), routed_task_type="holding_event")

    def test_an_exact_date_names_one_event(self) -> None:
        """The helper reads the execution-scoped plan P1-A4 D1 produces."""

        self.assertEqual(
            self._mode("테스트회사 2023년 6월 13일 보유 수와 비율",
                       period=QueryPeriod(year=2023, from_date="2023-06-13",
                                          to_date="2023-06-13",
                                          period_type=EXACT_PERIOD)),
            EXACT)

    def test_a_question_whose_date_was_never_materialised_is_not_exact(self) -> None:
        """Until D1 puts the day on the plan, there is nothing to read."""

        self.assertNotEqual(
            self._mode("테스트회사 2023년 6월 13일 보유 수와 비율"), EXACT)

    def test_a_native_exact_period_is_also_exact(self) -> None:
        plan = _plan("테스트회사 2023년 6월 13일 보유 수량 비율",
                     task_type="holding_change",
                     period=QueryPeriod(year=2023, from_date="2023-06-13",
                                        to_date="2023-06-13",
                                        period_type=EXACT_PERIOD))
        self.assertEqual(
            classify_holding_event_selection(plan,
                                             routed_task_type="holding_event"),
            EXACT)

    def test_a_year_only_narrows_without_naming(self) -> None:
        self.assertEqual(
            self._mode("테스트회사 2023년 보유 수와 비율",
                       period=QueryPeriod(year=2023,
                                          period_type="reference_year")),
            FILTERED)

    def test_a_holder_only_narrows_without_naming(self) -> None:
        self.assertEqual(
            self._mode("테스트회사 가나연금 보유 수와 비율", reporter="가나연금"),
            FILTERED)

    def test_a_direction_only_narrows_without_naming(self) -> None:
        self.assertEqual(self._mode("테스트회사 감소 주식수"), FILTERED)

    def test_a_receipt_date_narrows_without_naming(self) -> None:
        self.assertEqual(
            self._mode("테스트회사 2023년 7월 4일 공시 보유 수와 비율",
                       period=QueryPeriod(year=2023, from_date="2023-07-04",
                                          to_date="2023-07-04",
                                          period_type="receipt_date")),
            FILTERED)

    def test_a_range_narrows_without_naming(self) -> None:
        self.assertEqual(
            self._mode("테스트회사 2023년 6월 1일부터 2023년 6월 30일 보유",
                       period=QueryPeriod(from_date="2023-06-01",
                                          to_date="2023-06-30",
                                          period_type="date_range")),
            FILTERED)

    def test_a_question_naming_nothing_is_unspecified(self) -> None:
        self.assertEqual(self._mode("테스트회사 보유 수와 비율"), UNSPECIFIED)

    def test_a_non_holding_execution_has_no_opinion(self) -> None:
        for routed in ("periodic_fact", "general_evidence", None):
            with self.subTest(routed=routed):
                self.assertEqual(
                    classify_holding_event_selection(
                        _plan("테스트회사 2023년 6월 13일 보유"),
                        routed_task_type=routed),
                    NOT_APPLICABLE)


class InertLabelTests(unittest.TestCase):
    """13-20: words the resolver does not act on cannot select an event."""

    LABELS = ("직전보고", "현재", "최근", "최신", "최초", "마지막")

    def test_no_inert_label_is_ever_exact(self) -> None:
        for label in self.LABELS:
            with self.subTest(label=label):
                mode = classify_holding_event_selection(
                    _plan(f"테스트회사 {label} 보유 수량 비율",
                          task_type="holding_change",
                          period=QueryPeriod(period_type="latest_holding")),
                    routed_task_type="holding_event")
                self.assertNotEqual(mode, EXACT, f"{label} must not select")

    def test_opposite_labels_classify_the_same(self) -> None:
        """최초 and 마지막 are opposites; neither selects anything."""

        first, last = (classify_holding_event_selection(
            _plan(f"테스트회사 {label} 보유 수량 비율", task_type="holding_change",
                  period=QueryPeriod(period_type="latest_holding")),
            routed_task_type="holding_event") for label in ("최초", "마지막"))
        self.assertEqual(first, last)
        self.assertNotEqual(first, EXACT)


class SemanticUniquenessTests(unittest.TestCase):
    """Both halves are required."""

    def test_an_exact_question_with_one_event_is_unique(self) -> None:
        self.assertTrue(is_semantically_unique(EXACT, 1))

    def test_an_exact_question_with_several_events_is_not(self) -> None:
        self.assertFalse(is_semantically_unique(EXACT, 2))

    def test_one_event_without_an_exact_question_is_not(self) -> None:
        for mode in (FILTERED, UNSPECIFIED, NOT_APPLICABLE):
            with self.subTest(mode=mode):
                self.assertFalse(is_semantically_unique(mode, 1))


# ------------------------------------------------- composer and generator


class _Provenance:
    def __init__(self) -> None:
        self.field_conflict = False
        self.sources = ()
        self.alternatives = ()
        self.value = None
        self.field_name = ""

    def to_dict(self):
        return {"field_conflict": False, "sources": [], "alternatives": []}


class _Value:
    def __init__(self, raw, normalized) -> None:
        self.raw, self.normalized = raw, normalized


class _Event:
    """The fields the composer and generator read from a resolved event."""

    _FIELDS = ("reporter", "reference_date", "report_date", "receipt_date",
               "before_shares", "change_shares", "after_shares",
               "before_ratio", "after_ratio", "change_ratio")

    def __init__(self, chunk, *, matches=True, date="2023-06-30",
                 after=1000) -> None:
        self.matches_query = matches
        self.evidence_chunk_ids = (chunk,)
        self.doc_id, self.doc_ids = "d1", ("d1",)
        self.corp_name, self.corp_code, self.company_id = "테스트회사", "1", "1"
        self.reporter = "가나연금기금"
        self.reference_date = date
        self.report_date = self.receipt_date = "2023-07-04"
        self.before_shares = _Value("900", 900)
        self.change_shares = _Value("100", 100)
        self.after_shares = _Value(f"{after:,}", after)
        self.before_ratio = self.after_ratio = self.change_ratio = None
        self.change_direction = "increase"
        self.event_type = "holding_change"
        self.source_refs = ({"table_id": "t1", "row_start": 1, "row_end": 1},)
        self.field_provenance = {f: _Provenance() for f in self._FIELDS}
        self.field_conflict = False
        self.conflicting_fields = ()
        self.temporal_match = self.direction_match = None
        self.confidence = {}
        self.completeness = {}
        self.warnings = ()

    def to_dict(self):
        def value(name):
            held = getattr(self, name)
            return (None if held is None
                    else {"raw": held.raw, "normalized": held.normalized})

        return {
            "reporter": self.reporter,
            "reference_date": self.reference_date,
            "report_date": self.report_date,
            "receipt_date": self.receipt_date,
            "before_shares": value("before_shares"),
            "change_shares": value("change_shares"),
            "after_shares": value("after_shares"),
            "before_ratio": value("before_ratio"),
            "after_ratio": value("after_ratio"),
            "change_ratio": value("change_ratio"),
            "change_direction": self.change_direction,
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "doc_ids": list(self.doc_ids),
            "source_refs": [dict(r) for r in self.source_refs],
            "field_conflict": self.field_conflict,
            "conflicting_fields": list(self.conflicting_fields),
        }


class _Resolution:
    def __init__(self, events, *, requested=("after_shares",)) -> None:
        self.question = "테스트 질문"
        self.events = tuple(events)
        self.requested_fields = tuple(requested)
        matching = [e for e in events if e.matches_query is True]
        self.matching_event_count = len(matching)
        self.temporal_ambiguity = len(matching) > 1
        self.unresolved_fields = ()
        self.warnings = ()
        self.reporter_constraint = None


class _Item:
    def __init__(self, chunk) -> None:
        self.chunk_id, self.doc_id = chunk, "d1"
        self.source_refs = ({"table_id": "t1", "row_start": 1, "row_end": 1},)
        self.provenance = {"source_chunk_id": chunk, "source_doc_id": "d1"}


class _Group:
    def __init__(self, items) -> None:
        self.items = tuple(items)


class _EvidenceSet:
    def __init__(self, chunks) -> None:
        self.evidence_groups = (_Group([_Item(c) for c in chunks]),)
        self.warnings = ()
        self.task_type = "holding_change"
        self.question = "테스트 질문"


def _render(events, *, mode, requested=("after_shares",)):
    resolution = _Resolution(events, requested=requested)
    evidence = _EvidenceSet([c for e in events for c in e.evidence_chunk_ids])
    draft = compose_holding_answer(resolution, evidence, selection_mode=mode)
    return draft, generate_answer(draft)


class NoticeSelectionTests(unittest.TestCase):
    """1-8: which caveat each shape carries."""

    def test_exact_with_one_event_says_nothing(self) -> None:
        """1: the P1-A4 shape, which must stay silent."""

        draft, generated = _render([_Event("c1")], mode=EXACT)

        self.assertTrue(draft.ambiguity["semantic_unique"])
        self.assertFalse(draft.ambiguity["under_specified"])
        self.assertNotIn("주의", generated.answer_text)

    def test_exact_with_several_events_says_the_date_was_shared(self) -> None:
        """2: no arbitrary pick; both are shown."""

        events = [_Event("c1"), _Event("c2", after=2000)]
        draft, generated = _render(events, mode=EXACT)

        self.assertFalse(draft.ambiguity["semantic_unique"])
        self.assertTrue(draft.ambiguity["exact_multi_match"])
        self.assertIn(_HOLDING_EXACT_MANY, generated.answer_text)
        self.assertIn("1,000", generated.answer_text)
        self.assertIn("2,000", generated.answer_text)

    def test_filtered_with_one_event_names_what_the_question_left_open(self) -> None:
        """3: the HX12 shape."""

        draft, generated = _render([_Event("c1")], mode=FILTERED)

        self.assertFalse(draft.ambiguity["semantic_unique"])
        self.assertTrue(draft.ambiguity["under_specified"])
        self.assertIn(_HOLDING_UNDER_SPECIFIED_ONE, generated.answer_text)

    def test_filtered_with_several_events_shows_them_together(self) -> None:
        events = [_Event("c1"), _Event("c2", after=2000)]
        _draft, generated = _render(events, mode=FILTERED)

        self.assertIn(_HOLDING_UNDER_SPECIFIED_MANY, generated.answer_text)

    def test_unspecified_with_one_event_carries_the_same_caveat(self) -> None:
        _draft, generated = _render([_Event("c1")], mode=UNSPECIFIED)

        self.assertIn(_HOLDING_UNDER_SPECIFIED_ONE, generated.answer_text)

    def test_unspecified_with_several_events_carries_the_many_caveat(self) -> None:
        events = [_Event("c1"), _Event("c2", after=2000)]
        _draft, generated = _render(events, mode=UNSPECIFIED)

        self.assertIn(_HOLDING_UNDER_SPECIFIED_MANY, generated.answer_text)

    def test_a_history_question_carries_no_under_specified_caveat(self) -> None:
        """7: asking for the timeline means several events are the answer."""

        events = [_Event("c1"), _Event("c2", after=2000)]
        draft, generated = _render(events, mode=FILTERED, requested=())

        self.assertFalse(draft.ambiguity["under_specified"])
        self.assertNotIn(_HOLDING_UNDER_SPECIFIED_MANY, generated.answer_text)
        self.assertNotIn(_HOLDING_UNDER_SPECIFIED_ONE, generated.answer_text)

    def test_nothing_matching_keeps_the_existing_fallback(self) -> None:
        """8: P1-A5-B's no-match path is untouched."""

        events = [_Event("c1", matches=False), _Event("c2", matches=False)]
        draft, _generated = _render(events, mode=FILTERED)

        self.assertEqual(draft.ambiguity["matching_event_count"], 0)
        self.assertEqual(draft.ambiguity["observable_matching_event_count"], 2)

    def test_a_caller_supplying_no_mode_keeps_the_previous_wording(self) -> None:
        events = [_Event("c1"), _Event("c2", after=2000)]
        _draft, generated = _render(events, mode=NOT_APPLICABLE)

        self.assertIn(_HOLDING_OBSERVED_MANY, generated.answer_text)


class OneObservedEventTests(unittest.TestCase):
    """9-10: a caveat must not invent events, nor change the rendering."""

    def test_the_caveat_never_claims_several_events_were_seen(self) -> None:
        _draft, generated = _render([_Event("c1")], mode=FILTERED)

        self.assertNotIn("여러 변동 이벤트가 확인되어", generated.answer_text)
        self.assertNotIn(_HOLDING_OBSERVED_MANY, generated.answer_text)

    def test_a_single_event_keeps_its_prose_form(self) -> None:
        """10: the notice must not force the verbose record rendering."""

        _draft, with_notice = _render([_Event("c1")], mode=FILTERED)
        _draft2, without = _render([_Event("c1")], mode=EXACT)

        # The record form numbers its events and labels every field; prose does
        # neither.  Both answers must be prose.
        for text in (with_notice.answer_text, without.answer_text):
            self.assertNotIn("보고자:", text)
            self.assertNotIn("변동 후 주식수:", text)
        # The first block is the event rendering; later sections vary for
        # reasons this test is not about.
        first = with_notice.answer_text.split(BLANK_LINE)[0]
        self.assertEqual(first, without.answer_text.split(BLANK_LINE)[0])


class MatchingOnlyTests(unittest.TestCase):
    """22: P1-A5-B is not weakened by adding semantic framing."""

    def test_non_matching_events_are_still_excluded(self) -> None:
        events = [_Event("keep", after=1000), _Event("drop", matches=False,
                                                     after=9999),
                  _Event("keep2", after=2000)]
        draft, generated = _render(events, mode=FILTERED)

        self.assertEqual(draft.ambiguity["observable_matching_event_count"], 2)
        self.assertNotIn("9,999", generated.answer_text)
        self.assertIn("1,000", generated.answer_text)
        self.assertIn("2,000", generated.answer_text)


class OrderAndCitationTests(unittest.TestCase):
    """21, 23: order is presentation; the caveat cites nothing."""

    def test_permuting_the_events_does_not_change_the_mode(self) -> None:
        first = [_Event("a"), _Event("b", after=2000), _Event("c", after=3000)]
        second = [_Event("c", after=3000), _Event("a"), _Event("b", after=2000)]
        one, _g1 = _render(first, mode=FILTERED)
        two, _g2 = _render(second, mode=FILTERED)

        self.assertEqual(one.ambiguity["selection_mode"],
                         two.ambiguity["selection_mode"])
        self.assertEqual(one.ambiguity["under_specified"],
                         two.ambiguity["under_specified"])
        self.assertEqual(one.ambiguity["semantic_unique"],
                         two.ambiguity["semantic_unique"])

    def test_the_caveat_carries_no_citation_of_its_own(self) -> None:
        _draft, generated = _render([_Event("c1")], mode=FILTERED)
        notice = next(s for s in generated.sections if s.title == "주의")

        self.assertEqual(tuple(notice.citations), ())

    def test_every_citation_still_comes_from_the_evidence(self) -> None:
        events = [_Event("c1"), _Event("c2", after=2000)]
        _draft, generated = _render(events, mode=FILTERED)
        served = {c for e in events for c in e.evidence_chunk_ids}

        self.assertTrue(generated.citations)
        self.assertLessEqual({c.chunk_id for c in generated.citations}, served)


class RetrievalShapeStabilityTests(unittest.TestCase):
    """The HX12 requirement: the claim must not move with retrieval."""

    CLAIM = "질문에 특정 변동 시점이 지정되지"

    def test_one_event_and_nine_events_make_the_same_claim(self) -> None:
        one, generated_one = _render([_Event("c0")], mode=FILTERED)
        nine_events = [_Event(f"c{i}", after=1000 + i) for i in range(9)]
        nine, generated_nine = _render(nine_events, mode=FILTERED)

        self.assertIn(self.CLAIM, generated_one.answer_text)
        self.assertIn(self.CLAIM, generated_nine.answer_text)
        self.assertEqual(one.ambiguity["selection_mode"],
                         nine.ambiguity["selection_mode"])
        self.assertFalse(one.ambiguity["semantic_unique"])
        self.assertFalse(nine.ambiguity["semantic_unique"])
        # The row count may differ; the semantic framing may not.
        self.assertIn(_HOLDING_UNDER_SPECIFIED_ONE, generated_one.answer_text)
        self.assertIn(_HOLDING_UNDER_SPECIFIED_MANY, generated_nine.answer_text)


class NoDomainLiteralTests(unittest.TestCase):
    """24, 35: the rules name no question, holder, company, date or model."""

    def test_the_selection_helper_carries_no_domain_literals(self) -> None:
        from pathlib import Path

        text = Path("app/reasoning/holding_event_selection.py").read_text(
            encoding="utf-8")
        # The prose names what the module refuses to read, so only executable
        # code is checked for the signals it must never act on.
        body = "".join(text.split('"""')[::2])
        for literal in ("HX", "국민연금", "이마트", "파마리서치", "효성중공업",
                        "LG생활건강", "2023-", "2024-", "holding_20",
                        "rank", "latest", "newest", "earliest", "hcx", "HCX"):
            self.assertNotIn(literal, body, f"must not act on {literal!r}")

    def test_classification_and_notice_need_no_model(self) -> None:
        """The whole path is deterministic: no HCX import, no call."""

        from pathlib import Path

        for name in ("app/reasoning/holding_event_selection.py",):
            text = Path(name).read_text(encoding="utf-8")
            self.assertNotIn("hcx", text.lower())
        _draft, generated = _render([_Event("c1")], mode=FILTERED)
        self.assertIn(_HOLDING_UNDER_SPECIFIED_ONE, generated.answer_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
