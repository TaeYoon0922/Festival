"""P1-A4: a question that names one day must resolve to one complete event.

Two defects had to be fixed together, and the tests are written to fail if
either half is removed.  D1 recovers the exact date a promoted execution lost.
D2 merges the complementary views one filing can give of a single event, so the
shares and the ratios reach the resolver as one event rather than two.  D2b
keeps holder attribution honest once two holder labels share a group.

Holder names in the unit fixtures are deliberately invented.  The rules may not
depend on any real holder, and a test written with real names could not prove
that.
"""

import unittest
from dataclasses import FrozenInstanceError, replace

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.evidence_builder import build_evidence_set
from app.reasoning.holding_date_intent import (
    EXACT_PERIOD_TYPE,
    derive_exact_period,
    exact_reference_date,
    execution_plan,
)
from app.reasoning.holding_event_fusion import fuse, same_event
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult

# Invented holders: a family whose members share a stem, and two strangers.
FAMILY = "가나연금"
MEMBER_A = "가나연금기금"
MEMBER_B = "가나연금공단"
STRANGER_A = "한바다"
STRANGER_B = "두리안"

DETAIL_TABLE = "tA"
REPORT_TABLE = "tB"
META_TABLE = "tM"


def _projection(
    chunk_id,
    doc_id="d1",
    *,
    kind="holding_detail_row",
    reporter=MEMBER_A,
    date="2023년 06월 13일",
    before="2,485,201",
    change="-283,151",
    after="2,202,050",
    ratios=None,
    table=DETAIL_TABLE,
    row=2,
    rank=1,
):
    """A structured holding projection shaped like the frozen corpus.

    ``ratios`` is the (before, change, after) ratio triple a report carries and a
    detail row does not; every projection also points at the same holding-purpose
    metadata row, which is exactly what must not count as event identity.
    """

    fields = {
        "보고자/보유자": reporter,
        "기준일/보고일": date,
        "직전 보유주식수": before,
        "증감주식수": change,
        "보유주식수": after,
        "보유 목적": "경영권 영향",
    }
    refs = {
        label: [{"table_id": table, "row_start": row, "row_end": row}]
        for label in ("보고자/보유자", "기준일/보고일", "직전 보유주식수",
                      "증감주식수", "보유주식수")
    }
    refs["보유 목적"] = [{"table_id": META_TABLE, "row_start": 8, "row_end": 8}]
    if ratios is not None:
        before_ratio, change_ratio, after_ratio = ratios
        fields.update({
            "직전 보유비율": before_ratio,
            "증감비율": change_ratio,
            "보유비율": after_ratio,
        })
        for label in ("직전 보유비율", "증감비율", "보유비율"):
            refs[label] = [{"table_id": table, "row_start": row + 1,
                            "row_end": row + 1}]
    text = " ".join(f"[{k}] {v}" for k, v in fields.items())
    chunk = {
        "chunk_id": chunk_id, "doc_id": doc_id, "doc_group": "holding",
        "corp_code": "00000001", "corp_name": "테스트회사",
        "report_nm": "주식등의대량보유상황보고서", "rcept_dt": "2023-07-04",
        "chunk_type": "table_projection", "projection_type": kind,
        "projection_fields": fields, "projection_field_refs": refs,
        "source_refs": [{"table_id": table, "row_start": row, "row_end": row}],
        "section_path": ["대량보유상황보고"],
        "content": text, "retrieval_text": text,
    }
    return (CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
            RetrievalResult(chunk_id, doc_id, 1.0, rank, MetadataMatch().to_dict()))


def _plan(question, *, task_type="disclosure_lookup", reporter=FAMILY,
          period=None, route=("holding",), evidence=None):
    return QueryPlan(
        query=question, raw_query=question, company="테스트회사",
        corp_code="00000001", task_type=task_type, reporter=reporter,
        disclosure_route=route, period=period,
        evidence=evidence if evidence is not None else {"operation": "lookup_holding"},
    )


def _evidence(plan, pairs):
    return build_evidence_set(
        question=plan.raw_query, query_plan=dict(plan.to_dict()),
        candidates=[c for c, _r in pairs], results=[r for _c, r in pairs],
        grouping_intent="holding_change",
    )


def _holding_groups(evidence_set):
    return [g for g in evidence_set.evidence_groups if g.group_type == "holding_event"]


def _items(pairs):
    """The EvidenceItems the builder makes from these chunks, in order."""

    plan = _plan("테스트회사 가나연금 보유", period=None)
    groups = _holding_groups(_evidence(plan, pairs))
    return [item for group in groups for item in group.items]


# ---------------------------------------------------------------- D1


class ExactDateIntentTests(unittest.TestCase):
    """1-4: the date a promoted execution should have been given."""

    HX13 = "이마트 국민연금 2023년 6월 13일 보유 수와 비율"

    def _promoted(self):
        # What P0-D produces for this shape: the day is visible, the plan kept
        # only the year, because the plan's own task type is not holding_change.
        return _plan(self.HX13, period=QueryPeriod(year=2023,
                                                   period_type="reference_year"),
                     evidence={"operation": "lookup_disclosure",
                               "date_semantics": {"role": "reference",
                                                  "values": ["2023"]}})

    def test_a_promoted_execution_recovers_the_exact_day(self) -> None:
        plan = self._promoted()
        scoped = execution_plan(self.HX13, plan, routed_task_type="holding_event")

        self.assertIsNot(scoped, plan)
        self.assertEqual(scoped.period.period_type, EXACT_PERIOD_TYPE)
        self.assertEqual(scoped.period.from_date, "2023-06-13")
        self.assertEqual(scoped.period.to_date, "2023-06-13")
        self.assertEqual(exact_reference_date(scoped), "2023-06-13")

    def test_the_original_plan_is_left_exactly_as_p0d_built_it(self) -> None:
        plan = self._promoted()
        before = plan.to_dict()
        execution_plan(self.HX13, plan, routed_task_type="holding_event")

        self.assertEqual(plan.to_dict(), before)
        self.assertEqual(plan.period.period_type, "reference_year")
        self.assertIsNone(plan.period.from_date)

    def test_the_routed_promotion_is_not_written_back_onto_the_plan(self) -> None:
        plan = self._promoted()
        scoped = execution_plan(self.HX13, plan, routed_task_type="holding_event")

        self.assertEqual(plan.task_type, "disclosure_lookup")
        self.assertEqual(scoped.task_type, "disclosure_lookup")

    def test_a_native_holding_question_keeps_the_date_p0d_gave_it(self) -> None:
        """4: an exact date already decided upstream is never re-derived."""

        question = "효성중공업 국민연금 2023년 3월 7일 보유 수량 비율"
        plan = _plan(question, task_type="holding_change",
                     period=QueryPeriod(year=2023, from_date="2023-03-07",
                                        to_date="2023-03-07",
                                        period_type=EXACT_PERIOD_TYPE))
        self.assertIs(execution_plan(question, plan,
                                     routed_task_type="holding_event"), plan)
        self.assertIsNone(derive_exact_period(question, plan))
        self.assertEqual(exact_reference_date(plan), "2023-03-07")


class NoExactDateTests(unittest.TestCase):
    """5-10: shapes that do not name one day must not gain one."""

    def _declines(self, question, *, period=None, evidence=None,
                  routed="holding_event"):
        plan = _plan(question, period=period, evidence=evidence)
        scoped = execution_plan(question, plan, routed_task_type=routed)
        self.assertIs(scoped, plan, f"{question!r} must not gain an exact date")
        self.assertIsNone(exact_reference_date(scoped))

    def test_a_year_alone_is_not_a_day(self) -> None:
        self._declines("테스트회사 가나연금 2023년 보유 수와 비율",
                       period=QueryPeriod(year=2023, period_type="reference_year"))

    def test_a_month_alone_is_not_a_day(self) -> None:
        self._declines("테스트회사 가나연금 2023년 6월 보유 수와 비율",
                       period=QueryPeriod(year=2023, period_type="reference_year"))

    def test_no_date_at_all_stays_undated(self) -> None:
        self._declines("테스트회사 가나연금 보유 수와 비율")

    def test_a_receipt_date_is_not_reread_as_an_event_date(self) -> None:
        """8: 공시/접수 wording names when the filing arrived, not when the
        holding changed."""

        question = "테스트회사 가나연금 2023년 7월 4일 공시에서 보유 수와 비율"
        self._declines(
            question,
            period=QueryPeriod(year=2023, from_date="2023-07-04",
                               to_date="2023-07-04", period_type="receipt_date"),
            evidence={"operation": "lookup_disclosure",
                      "date_semantics": {"role": "receipt", "marker": "공시"}})

    def test_a_receipt_role_blocks_rederivation_even_without_bounds(self) -> None:
        self._declines(
            "테스트회사 가나연금 2023년 7월 4일 공시 보유 수와 비율",
            period=QueryPeriod(year=2023, period_type="reference_year"),
            evidence={"operation": "lookup_disclosure",
                      "date_semantics": {"role": "receipt", "marker": "공시"}})

    def test_a_date_range_names_a_window_not_an_event(self) -> None:
        self._declines(
            "테스트회사 가나연금 2023년 6월 1일부터 2023년 6월 30일 보유",
            period=QueryPeriod(from_date="2023-06-01", to_date="2023-06-30",
                               period_type="date_range"))

    def test_a_non_holding_execution_is_never_touched(self) -> None:
        """10: the lane is gated on the routed shape, not on the wording."""

        question = "테스트회사 2023년 6월 13일 매출액"
        for routed in ("periodic_fact", "general_evidence", "corporate_event", None):
            with self.subTest(routed=routed):
                self._declines(question, routed=routed)


class DateWithOtherConstraintsTests(unittest.TestCase):
    """11-12: the exact date composes with what else the question asked."""

    def test_direction_semantics_survive_the_exact_date(self) -> None:
        question = "테스트회사 가나연금 2023년 6월 13일 감소 주식수"
        plan = _plan(question, period=QueryPeriod(year=2023,
                                                  period_type="reference_year"))
        scoped = execution_plan(question, plan, routed_task_type="holding_event")

        self.assertEqual(exact_reference_date(scoped), "2023-06-13")
        pairs = [_projection("p1", "d1")]
        resolution = _resolve(scoped, pairs)
        event = resolution.events[0]
        self.assertEqual(event.change_direction, "decrease")
        self.assertIs(event.direction_match, True)
        self.assertIs(event.matches_query, True)

    def test_the_reporter_constraint_survives_the_exact_date(self) -> None:
        question = "테스트회사 가나연금 2023년 6월 13일 보유 수와 비율"
        plan = _plan(question, period=QueryPeriod(year=2023,
                                                  period_type="reference_year"))
        scoped = execution_plan(question, plan, routed_task_type="holding_event")

        self.assertEqual(scoped.reporter, FAMILY)
        resolution = _resolve(scoped, [_projection("p1", "d1",
                                                   reporter=STRANGER_A)])
        self.assertEqual(resolution.reporter_constraint, FAMILY)
        self.assertIsNot(resolution.events[0].matches_query, True)


def _resolve(plan, pairs, *, fused=True):
    from app.reasoning.holding_event_resolver import resolve_holding_events

    evidence = _evidence(plan, pairs)
    if fused:
        evidence = fuse(evidence, reference_date=exact_reference_date(plan),
                        reporter=plan.reporter)
    return resolve_holding_events(evidence, query_plan=dict(plan.to_dict()))


# ---------------------------------------------------------------- D2


class FusionKeyTests(unittest.TestCase):
    """13-20: what may and may not be treated as one event."""

    def _pair(self, **overrides):
        detail = _projection("detail", "d1", **overrides)
        report = _projection(
            "report", "d1", kind="holding_report", reporter=MEMBER_B,
            ratios=("8.92", "-1.02", "7.90"), table=REPORT_TABLE, rank=2,
            **overrides)
        return detail, report

    def test_disjoint_event_tables_with_one_transition_are_one_event(self) -> None:
        detail, report = self._pair()
        left, right = _items([detail, report])
        self.assertTrue(same_event(left, right))

    def test_two_rows_of_one_table_are_two_events(self) -> None:
        """14: an enumeration is not a set of views."""

        first = _projection("r1", "d1", row=2)
        second = _projection("r2", "d1", reporter=MEMBER_B, row=3, rank=2)
        left, right = _items([first, second])
        self.assertFalse(same_event(left, right))

    def test_a_different_transition_is_a_different_event(self) -> None:
        detail = _projection("detail", "d1")
        report = _projection("report", "d1", kind="holding_report",
                             reporter=MEMBER_B, before="2,485,201",
                             change="-100,000", after="2,385,201",
                             ratios=("8.92", "-1.02", "7.90"),
                             table=REPORT_TABLE, rank=2)
        left, right = _items([detail, report])
        self.assertFalse(same_event(left, right))

    def test_a_partly_reported_transition_is_not_an_identity(self) -> None:
        """16: two of three is not enough to claim the same event."""

        detail = _projection("detail", "d1", before=None)
        report = _projection("report", "d1", kind="holding_report",
                             reporter=MEMBER_B, before=None,
                             ratios=("8.92", "-1.02", "7.90"),
                             table=REPORT_TABLE, rank=2)
        left, right = _items([detail, report])
        self.assertFalse(same_event(left, right))

    def test_an_all_zero_transition_never_identifies_an_event(self) -> None:
        """17: several holders of one filing can each report 0 -> 0."""

        detail = _projection("detail", "d1", reporter=STRANGER_A,
                             before="0", change="0", after="0")
        report = _projection("report", "d1", kind="holding_report",
                             reporter=STRANGER_B, before="0", change="0",
                             after="0", ratios=("0.00", "0.00", "0.00"),
                             table=REPORT_TABLE, rank=2)
        left, right = _items([detail, report])
        self.assertFalse(same_event(left, right))

    def test_inconsistent_arithmetic_is_refused(self) -> None:
        detail = _projection("detail", "d1", before="100", change="5", after="200")
        report = _projection("report", "d1", kind="holding_report",
                             reporter=MEMBER_B, before="100", change="5",
                             after="200", ratios=("1.00", "0.05", "2.00"),
                             table=REPORT_TABLE, rank=2)
        left, right = _items([detail, report])
        self.assertFalse(same_event(left, right))

    def test_a_conflict_outside_reporter_declines_the_merge(self) -> None:
        """18: a fuse may fill gaps, never create a contradiction."""

        # Same transition and disjoint tables -- so the structural key holds --
        # but the two disagree on a ratio they both populate. Two views of one
        # event cannot report different ratios for it.
        detail = _projection("detail", "d1", ratios=("9.99", "-9.99", "1.11"))
        report = _projection("report", "d1", kind="holding_report",
                             reporter=MEMBER_B, ratios=("8.92", "-1.02", "7.90"),
                             table=REPORT_TABLE, rank=2)
        left, right = _items([detail, report])
        self.assertTrue(same_event(left, right),
                        "the structural key alone must still admit this pair")

        plan = _plan("테스트회사 가나연금 보유", period=QueryPeriod(
            year=2023, from_date="2023-06-13", to_date="2023-06-13",
            period_type=EXACT_PERIOD_TYPE))
        evidence = _evidence(plan, [detail, report])
        fused = fuse(evidence, reference_date="2023-06-13", reporter=FAMILY)

        self.assertIs(fused, evidence, "the field-safety gate must decline")
        self.assertEqual(len(_holding_groups(fused)), 2)

    def test_projection_type_alone_never_fuses(self) -> None:
        """19: detail + report is not the rule; the numbers are."""

        detail = _projection("detail", "d1", after="2,202,050")
        report = _projection("report", "d1", kind="holding_report",
                             reporter=MEMBER_B, before="1", change="1",
                             after="2", ratios=("1.0", "1.0", "2.0"),
                             table=REPORT_TABLE, rank=2)
        left, right = _items([detail, report])
        self.assertFalse(same_event(left, right))

    def test_two_detail_rows_of_separate_tables_may_fuse(self) -> None:
        """19b: nothing about the rule mentions projection type."""

        first = _projection("a", "d1", table=DETAIL_TABLE)
        second = _projection("b", "d1", reporter=MEMBER_B, table=REPORT_TABLE,
                             rank=2)
        left, right = _items([first, second])
        self.assertTrue(same_event(left, right))

    def test_matching_reporter_names_do_not_by_themselves_fuse(self) -> None:
        """20: identical holders on separate events stay separate."""

        first = _projection("a", "d1", row=2)
        second = _projection("b", "d1", row=3, rank=2)   # same holder, same table
        left, right = _items([first, second])
        self.assertFalse(same_event(left, right))

    def test_a_shared_metadata_row_is_not_event_identity(self) -> None:
        """28: every projection of a filing points at 보유 목적."""

        detail, report = self._pair()
        left, right = _items([detail, report])
        # The metadata table is common to both and must not be what links them.
        self.assertTrue(same_event(left, right))
        moved = _projection("report2", "d1", kind="holding_report",
                            reporter=MEMBER_B, ratios=("8.92", "-1.02", "7.90"),
                            table=DETAIL_TABLE, row=2, rank=2)
        left2, right2 = _items([detail, moved])
        self.assertFalse(same_event(left2, right2),
                         "sharing the event table must block the merge even "
                         "though the metadata row is shared either way")


class FusionScopeTests(unittest.TestCase):
    """Fusion is an exact-date feature, not a holding normalisation pass."""

    def _pairs(self):
        return [
            _projection("detail", "d1"),
            _projection("report", "d1", kind="holding_report", reporter=MEMBER_B,
                        ratios=("8.92", "-1.02", "7.90"), table=REPORT_TABLE,
                        rank=2),
        ]

    def test_without_an_exact_date_nothing_is_merged(self) -> None:
        plan = _plan("테스트회사 가나연금 보유 수와 비율")
        evidence = _evidence(plan, self._pairs())
        self.assertIs(fuse(evidence, reference_date=None, reporter=FAMILY),
                      evidence)
        self.assertEqual(len(_holding_groups(evidence)), 2)

    def test_only_groups_on_the_requested_day_are_considered(self) -> None:
        pairs = [
            *self._pairs(),
            _projection("other", "d1", date="2023년 07월 12일", before="2,202,050",
                        change="-285,926", after="1,916,124", rank=3),
        ]
        plan = _plan("테스트회사 가나연금 2023년 6월 13일 보유")
        fused = fuse(_evidence(plan, pairs), reference_date="2023-06-13",
                     reporter=FAMILY)
        groups = _holding_groups(fused)

        self.assertEqual(len(groups), 2)
        self.assertEqual(sorted(len(g.items) for g in groups), [1, 2])


# ---------------------------------------------------------------- D2b


class ReporterAlternativeTests(unittest.TestCase):
    """21-25: holder attribution once a group holds two labels."""

    def _fused_event(self, *, reporter, holders=(MEMBER_A, MEMBER_B)):
        first, second = holders
        pairs = [
            _projection("detail", "d1", reporter=first),
            _projection("report", "d1", kind="holding_report", reporter=second,
                        ratios=("8.92", "-1.02", "7.90"), table=REPORT_TABLE,
                        rank=2),
        ]
        plan = _plan("테스트회사 보유 수와 비율", reporter=reporter,
                     period=QueryPeriod(year=2023, from_date="2023-06-13",
                                        to_date="2023-06-13",
                                        period_type=EXACT_PERIOD_TYPE))
        resolution = _resolve(plan, pairs)
        return resolution, _evidence(plan, pairs)

    def test_every_alternative_answering_the_holder_lets_the_event_match(self) -> None:
        resolution, _e = self._fused_event(reporter=FAMILY)

        self.assertEqual(resolution.matching_event_count, 1)
        event = resolution.events[0]
        self.assertIsNone(event.reporter)
        self.assertEqual(tuple(event.conflicting_fields), ("reporter",))
        alternatives = {
            str(getattr(a.value, "normalized", a.value))
            for a in event.field_provenance["reporter"].alternatives
        }
        self.assertEqual(alternatives, {MEMBER_A, MEMBER_B})
        self.assertIs(event.matches_query, True)

    def test_one_stranger_among_the_alternatives_blocks_the_match(self) -> None:
        """22: a group holding a stranger's label cannot answer for the holder.

        Built directly, because the fusion gate refuses to assemble this group
        in the first place -- this pins the resolver rule that backs it up.
        """

        from app.reasoning.evidence_builder import _make_group
        from app.reasoning.holding_event_resolver import _resolve_group

        pairs = [
            _projection("a", "d1", reporter=STRANGER_A),
            _projection("b", "d1", kind="holding_report", reporter=STRANGER_B,
                        ratios=("8.92", "-1.02", "7.90"), table=REPORT_TABLE,
                        rank=2),
        ]
        group = _make_group(_items(pairs), group_type="holding_event",
                            reason="forced cross-holder group")
        event = _resolve_group(
            group, requested_fields=("after_shares",),
            reporter_constraint=STRANGER_A, direction_constraint=None,
            explicit_temporal=False, temporal_constraint={})

        self.assertIsNone(event.reporter)
        self.assertEqual(tuple(event.conflicting_fields), ("reporter",))
        self.assertIs(event.matches_query, False)

    def test_the_fusion_gate_refuses_to_build_a_stranger_group(self) -> None:
        """The first line of defence: such a group is never assembled."""

        pairs = [
            _projection("a", "d1", reporter=STRANGER_A),
            _projection("b", "d1", kind="holding_report", reporter=STRANGER_B,
                        ratios=("8.92", "-1.02", "7.90"), table=REPORT_TABLE,
                        rank=2),
        ]
        plan = _plan("테스트회사 한바다 보유", reporter=STRANGER_A,
                     period=QueryPeriod(year=2023, from_date="2023-06-13",
                                        to_date="2023-06-13",
                                        period_type=EXACT_PERIOD_TYPE))
        evidence = _evidence(plan, pairs)
        self.assertIs(fuse(evidence, reference_date="2023-06-13",
                           reporter=STRANGER_A), evidence)

        # The holder that was actually asked for still answers on its own row,
        # which is the point: declining to fuse costs nothing here.
        resolution = _resolve(plan, pairs)
        matching = [e for e in resolution.events if e.matches_query is True]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0].reporter, STRANGER_A)
        self.assertEqual(len(matching[0].evidence_chunk_ids), 1)

    def test_without_a_holder_in_the_question_cross_name_views_stay_apart(self) -> None:
        """23: rather than build an event attributed to nobody."""

        pairs = [
            _projection("detail", "d1", reporter=STRANGER_A),
            _projection("report", "d1", kind="holding_report",
                        reporter=STRANGER_B, ratios=("8.92", "-1.02", "7.90"),
                        table=REPORT_TABLE, rank=2),
        ]
        plan = _plan("테스트회사 2023년 6월 13일 보유 수와 비율", reporter=None,
                     period=QueryPeriod(year=2023, from_date="2023-06-13",
                                        to_date="2023-06-13",
                                        period_type=EXACT_PERIOD_TYPE))
        evidence = _evidence(plan, pairs)
        fused = fuse(evidence, reference_date="2023-06-13", reporter=None)

        self.assertIs(fused, evidence)
        self.assertEqual(len(_holding_groups(fused)), 2)

    def test_a_single_holder_group_behaves_exactly_as_before(self) -> None:
        """24: no conflict, no new code path."""

        plan = _plan("테스트회사 가나연금 보유", period=QueryPeriod(
            year=2023, from_date="2023-06-13", to_date="2023-06-13",
            period_type=EXACT_PERIOD_TYPE))
        resolution = _resolve(plan, [_projection("only", "d1")], fused=False)
        event = resolution.events[0]

        self.assertEqual(event.reporter, MEMBER_A)
        self.assertFalse(event.field_conflict)
        self.assertIs(event.matches_query, True)

    def test_no_real_holder_name_is_required_anywhere(self) -> None:
        """25: every rule above was proven with invented holders."""

        for name in (FAMILY, MEMBER_A, MEMBER_B, STRANGER_A, STRANGER_B):
            self.assertNotIn("국민연금", name)


class FalseMergeSafetyTests(unittest.TestCase):
    """26-27: the shape where one filer's total coincides with one holder's row."""

    def _shape(self):
        """A filer's report whose transition equals a different holder's row."""

        return [
            _projection("holder_row", "d1", reporter=STRANGER_A,
                        before="7,960,493", change="-7,960,493", after="0"),
            _projection("filer_report", "d1", kind="holding_report",
                        reporter=STRANGER_B, before="7,960,493",
                        change="-7,960,493", after="0",
                        ratios=("28.56", "-28.56", "0.00"),
                        table=REPORT_TABLE, rank=2),
        ]

    def test_naming_one_party_does_not_attribute_the_other_to_them(self) -> None:
        plan = _plan("테스트회사 두리안 2023년 6월 13일 보유 수와 비율",
                     reporter=STRANGER_B,
                     period=QueryPeriod(year=2023, from_date="2023-06-13",
                                        to_date="2023-06-13",
                                        period_type=EXACT_PERIOD_TYPE))
        evidence = _evidence(plan, self._shape())
        fused = fuse(evidence, reference_date="2023-06-13", reporter=STRANGER_B)

        self.assertIs(fused, evidence, "strangers must not be merged")
        resolution = _resolve(plan, self._shape())
        for event in resolution.events:
            self.assertLessEqual(len(event.evidence_chunk_ids), 1)

    def test_with_no_party_named_the_shape_is_left_alone(self) -> None:
        plan = _plan("테스트회사 2023년 6월 13일 보유 수와 비율", reporter=None,
                     period=QueryPeriod(year=2023, from_date="2023-06-13",
                                        to_date="2023-06-13",
                                        period_type=EXACT_PERIOD_TYPE))
        evidence = _evidence(plan, self._shape())
        self.assertIs(fuse(evidence, reference_date="2023-06-13", reporter=None),
                      evidence)


# ---------------------------------------------------------------- end to end


class _Execution:
    def __init__(self, plan, pairs) -> None:
        self.plan = plan
        self.chunks = [c for c, _r in pairs]
        self.results = [r for _c, r in pairs]


def _run(plan, pairs):
    execution = _Execution(plan, pairs)
    result = AgentOrchestrator().run(plan.raw_query, plan, execution)
    generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
    guard = AnswerabilityGuard().evaluate(
        generated, plan=plan, agent_result=result, execution=execution)
    return execution, result, generated, guard


class EndToEndTests(unittest.TestCase):
    """29-37: the shapes the Gold questions exercise, through the orchestrator."""

    def _complementary(self, date="2023년 06월 13일", after="2,202,050",
                       ratios=("8.92", "-1.02", "7.90")):
        return [
            _projection("detail", "d1", date=date, after=after),
            _projection("report", "d1", kind="holding_report", reporter=MEMBER_B,
                        date=date, after=after, ratios=ratios,
                        table=REPORT_TABLE, rank=2),
        ]

    def test_a_promoted_exact_date_question_resolves_one_complete_event(self) -> None:
        """32/33: the HX13 and HX17 shape."""

        plan = _plan("테스트회사 가나연금 2023년 6월 13일 보유 수와 비율",
                     period=QueryPeriod(year=2023, period_type="reference_year"))
        _e, result, generated, guard = _run(plan, self._complementary())

        self.assertIn("holding_date_intent", result.execution_trace)
        self.assertIn("holding_event_fusion", result.execution_trace)
        self.assertEqual(result.resolution.matching_event_count, 1)
        event = next(e for e in result.resolution.events if e.matches_query is True)
        self.assertEqual(event.reference_date, "2023-06-13")
        self.assertEqual(event.after_shares.normalized, 2202050)
        self.assertEqual(event.after_ratio.normalized, 7.90)
        self.assertEqual(len(event.evidence_chunk_ids), 2)
        self.assertTrue(guard.model_answer_allowed)
        self.assertEqual(list(guard.missing_fields), [])
        self.assertGreaterEqual(guard.citation_count, 2)

    def test_a_native_exact_date_question_fuses_without_a_rewritten_date(self) -> None:
        """30/31: the HX05 and HX09 shape -- D2 without D1."""

        plan = _plan("테스트회사 가나연금 2023년 6월 13일 보유 수량 비율",
                     task_type="holding_change",
                     period=QueryPeriod(year=2023, from_date="2023-06-13",
                                        to_date="2023-06-13",
                                        period_type=EXACT_PERIOD_TYPE))
        _e, result, _g, guard = _run(plan, self._complementary())

        self.assertNotIn("holding_date_intent", result.execution_trace)
        self.assertIn("holding_event_fusion", result.execution_trace)
        self.assertEqual(result.resolution.matching_event_count, 1)
        self.assertTrue(guard.model_answer_allowed)

    def test_a_single_view_exact_date_question_is_untouched(self) -> None:
        """29: the HX01 shape -- already exactly one match, nothing to fuse."""

        plan = _plan("테스트회사 한바다 2023년 6월 13일 보유 수량 비율",
                     task_type="holding_change", reporter=STRANGER_A,
                     period=QueryPeriod(year=2023, from_date="2023-06-13",
                                        to_date="2023-06-13",
                                        period_type=EXACT_PERIOD_TYPE))
        _e, result, _g, guard = _run(plan, [_projection("only", "d1",
                                                        reporter=STRANGER_A)])

        self.assertNotIn("holding_date_intent", result.execution_trace)
        self.assertNotIn("holding_event_fusion", result.execution_trace)
        self.assertEqual(result.resolution.matching_event_count, 1)
        self.assertTrue(guard.model_answer_allowed)

    def test_an_undated_holding_question_keeps_every_event(self) -> None:
        """34/36: HX08/HX10/HX16/HX20 -- no exact date, so no fusion."""

        plan = _plan("테스트회사 가나연금 변동일 변동후 주식수")
        _e, result, _g, guard = _run(plan, self._complementary())

        self.assertNotIn("holding_date_intent", result.execution_trace)
        self.assertNotIn("holding_event_fusion", result.execution_trace)
        self.assertEqual(len(_holding_groups(result.evidence_set)), 2)
        self.assertTrue(guard.model_answer_allowed)

    def test_a_non_holding_execution_never_enters_either_lane(self) -> None:
        """37."""

        plan = _plan("테스트회사 2024년 매출액", task_type="financial_metric",
                     reporter=None, route=("periodic",),
                     evidence={"operation": "lookup_metric"})
        pairs = [_projection("p1", "d1")]
        pairs[0][0].chunk["doc_group"] = "periodic"
        _e, result, _g, _guard = _run(plan, pairs)

        self.assertNotEqual(result.task_decision.task_type, "holding_event")
        self.assertNotIn("holding_date_intent", result.execution_trace)
        self.assertNotIn("holding_event_fusion", result.execution_trace)


class ImmutabilityTests(unittest.TestCase):
    """39-43: nothing upstream is rewritten to make this work."""

    def _fixture(self):
        plan = _plan("테스트회사 가나연금 2023년 6월 13일 보유 수와 비율",
                     period=QueryPeriod(year=2023, period_type="reference_year"))
        pairs = [
            _projection("detail", "d1"),
            _projection("report", "d1", kind="holding_report", reporter=MEMBER_B,
                        ratios=("8.92", "-1.02", "7.90"), table=REPORT_TABLE,
                        rank=2),
        ]
        return plan, pairs

    def test_the_query_plan_is_never_mutated(self) -> None:
        plan, pairs = self._fixture()
        before = plan.to_dict()
        _run(plan, pairs)

        self.assertEqual(plan.to_dict(), before)
        self.assertEqual(plan.period.period_type, "reference_year")
        with self.assertRaises(FrozenInstanceError):
            plan.task_type = "holding_change"

    def test_the_builder_output_is_never_mutated(self) -> None:
        plan, pairs = self._fixture()
        scoped = execution_plan(plan.raw_query, plan,
                                routed_task_type="holding_event")
        evidence = _evidence(scoped, pairs)
        before = evidence.to_dict()
        fused = fuse(evidence, reference_date="2023-06-13", reporter=FAMILY)

        self.assertIsNot(fused, evidence)
        self.assertEqual(evidence.to_dict(), before)
        self.assertEqual(len(_holding_groups(evidence)), 2)
        self.assertEqual(len(_holding_groups(fused)), 1)

    def test_the_retrieval_execution_is_never_mutated(self) -> None:
        plan, pairs = self._fixture()
        execution, _result, _g, _guard = _run(plan, pairs)

        self.assertEqual([r.chunk_id for r in execution.results],
                         ["detail", "report"])
        self.assertEqual([c.chunk_id for c in execution.chunks],
                         ["detail", "report"])

    def test_fusion_moves_no_evidence_chunk(self) -> None:
        """42/43: grouping changes, membership does not."""

        plan, pairs = self._fixture()
        execution, result, generated, _guard = _run(plan, pairs)

        served = {r.chunk_id for r in execution.results}
        self.assertEqual({r.chunk_id for r in result.evidence_results}, served)
        grouped = {item.chunk_id
                   for group in result.evidence_set.evidence_groups
                   for item in group.items}
        self.assertEqual(grouped, served)
        cited = {c.chunk_id for c in generated.citations}
        self.assertTrue(cited)
        self.assertLessEqual(cited, served)

    def test_every_fused_field_keeps_a_real_source(self) -> None:
        plan, pairs = self._fixture()
        _e, result, _g, _guard = _run(plan, pairs)
        event = next(e for e in result.resolution.events if e.matches_query is True)

        for name in ("after_shares", "after_ratio", "before_shares", "reference_date"):
            provenance = event.field_provenance[name]
            self.assertTrue(provenance.sources, f"{name} has no source")
            for source in provenance.sources:
                self.assertIn(source.chunk_id, {"detail", "report"})
        self.assertGreaterEqual(len(event.source_refs), 2)


class NoDomainLiteralTests(unittest.TestCase):
    """The rules must not name a holder, a company, a date, or a table."""

    def test_production_modules_carry_no_domain_literals(self) -> None:
        from pathlib import Path

        forbidden = ("국민연금", "이마트", "LG생활건강", "효성중공업", "파마리서치",
                     "에스엠", "정용진", "이명희", "HX0", "HX1", "HX2",
                     "holding_20", "2023-06-13", "2023-06-30", "2023-03-07",
                     "2022-12-05", "t0019", "t0012", "t0002",
                     "holding_detail_row", "holding_report")
        for name in ("holding_date_intent.py", "holding_event_fusion.py"):
            source = Path("app/reasoning") / name
            text = source.read_text(encoding="utf-8")
            for literal in forbidden:
                self.assertNotIn(literal, text, f"{name} must not name {literal!r}")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
