"""FAIL-8: exact holding reference dates, holder identity, and the report before.

Three failures share one root.  The timeline that decides *which* holding report
a question means is keyed on the holder label the filing was typed with, and a
question almost never writes that label in full: the corpus says ``Fidelity
Management & Research Company LLC`` and ``T. Rowe PriceAssociates, Inc``, the
question says ``Fidelity`` and ``T. Rowe Price Associates``.  The identity check
is exact on the canonical key, so both questions reach an event whose value is
read and whose holder is then judged to be somebody else -- which drops the
event, and with it every citation the value was bound to.

C030 is that break seen from the citation end: the reference date and the share
count both resolve, and nothing is left to cite them from.  C054 is the same
break one stage earlier, where the deterministic index cannot even enumerate the
holder's timeline.  C042 is the opposite failure -- a first report, whose
``직전`` cells are literally ``-``, answered as though the previous state were
zero, with the *next* report shown alongside it as if it were the previous one.

So the identity rule is widened exactly as far as a multi-word foreign holder
name requires and no further: a shorter label answers to a longer one only when
it ends on a whitespace token boundary of that longer name, and only when both
are written in Latin script.  ``영풍`` is a single token and can never be a
prefix of ``영풍정밀``; the firewall that keeps those two apart is untouched.

Holder names below are the corpus's own only where a test is *about* the corpus
artifact.  Every unit fixture uses invented holders, because a rule that only
works on real names has not been proven.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from app.reasoning.answer_composer import compose_holding_answer
from app.reasoning.answerability import AnswerabilityGuard, AnswerabilityStatus
from app.reasoning.evidence_builder import build_evidence_set
from app.reasoning.field_evidence import FieldStatus
from app.reasoning.holding_date_intent import (
    exact_reference_date,
    question_reference_date,
)
from app.reasoning.holding_event_resolver import resolve_holding_events
from app.reasoning.holding_event_selection import EXACT
from app.reasoning.holding_field_evidence import holding_field_evidence
from app.reasoning.holding_previous_report import (
    BASELINE_FIELDS,
    previous_report_baseline_evidence,
)
from app.reasoning.holding_report_index import (
    AMBIGUOUS,
    CHANGE_UNAVAILABLE,
    NO_MATCH,
    RESOLVED,
    ROLE_CHANGE,
    ROLE_CURRENT,
    SELECTOR_EXACT_RECEIPT_DATE,
    SELECTOR_EXACT_REFERENCE_DATE,
    SELECTOR_LATEST,
    HoldingReportIndex,
    HoldingReportRecord,
    execute_report_relative,
    load_index,
    project_role,
)
from app.reasoning.holding_reporter import canonical_reporter_key, reporter_matches
from app.reasoning.holding_report_relative_execution import (
    HoldingReportRelativeExecution,
    ROUTED_TASK_TYPE,
    repository_corpus_identity,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


ROOT = Path(__file__).resolve().parents[1]

# Invented holders prove the general legal-suffix rule.  ``SPLIT_HOLDER`` also
# writes two of its words joined, which canonical spacing already normalizes.
LONG_HOLDER = "Vantis LLC"
LONG_HOLDER_SHORT = "Vantis"
SPLIT_HOLDER = "Q. Meridian CapitalPartners, Inc"
SPLIT_HOLDER_SHORT = "Q. Meridian Capital Partners"
OTHER_LATIN_HOLDER = "Vantis Inc"
KOREAN_STEM = "가상영풍"
KOREAN_LONGER = "가상영풍정밀"

ISSUER = "00000001"
OTHER_ISSUER = "00000002"


# --------------------------------------------------------------- fixtures
def _report_projection(
    chunk_id,
    doc_id,
    *,
    reporter=LONG_HOLDER,
    reference_date="2023년 10월 11일",
    receipt_date="2023-11-10",
    before="6,009,827",
    change="1,036,962",
    after="7,046,789",
    before_ratio="6.17",
    change_ratio="1.06",
    after_ratio="7.23",
    corp_code=ISSUER,
    corp_name="테스트항공",
    table="t0053",
    row=2,
    rank=1,
):
    """A ``holding_report`` projection shaped like the frozen corpus.

    ``reference_date`` and ``receipt_date`` are deliberately different: 62.5% of
    this corpus files on a day other than the one it reports about, and a test
    where they agree cannot tell the two axes apart.
    """

    fields = {
        "보고자/보유자": reporter,
        "기준일/보고일": reference_date,
        "직전 보유주식수": before,
        "증감주식수": change,
        "보유주식수": after,
        "직전 보유비율": before_ratio,
        "증감비율": change_ratio,
        "보유비율": after_ratio,
    }
    refs = {
        label: [{"table_id": table, "row_start": row, "row_end": row}]
        for label in fields
    }
    text = " ".join(f"[{k}] {v}" for k, v in fields.items())
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": "holding",
        "corp_code": corp_code,
        "corp_name": corp_name,
        "report_nm": "주식등의대량보유상황보고서",
        "rcept_dt": receipt_date,
        "chunk_type": "table_projection",
        "projection_type": "holding_report",
        "projection_fields": fields,
        "projection_field_refs": refs,
        "source_refs": [{"table_id": table, "row_start": row, "row_end": row}],
        "section_path": ["대량보유상황보고"],
        "content": text,
        "retrieval_text": text,
    }
    return (
        CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
        RetrievalResult(chunk_id, doc_id, 1.0, rank, MetadataMatch().to_dict()),
    )


def _plan(question, *, reporter, exact=None, corp_code=ISSUER, task="holding_change"):
    period = (
        QueryPeriod(
            year=int(exact[:4]),
            from_date=exact,
            to_date=exact,
            period_type="holding_reference_date",
        )
        if exact
        else None
    )
    return QueryPlan(
        query=question,
        raw_query=question,
        company="테스트항공",
        corp_code=corp_code,
        task_type=task,
        reporter=reporter,
        disclosure_route=("holding",),
        period=period,
        evidence={"operation": "lookup_holding"},
    )


def _resolve(plan, pairs):
    evidence = build_evidence_set(
        question=plan.raw_query,
        query_plan=dict(plan.to_dict()),
        candidates=[c for c, _r in pairs],
        results=[r for _c, r in pairs],
        grouping_intent="holding_change",
    )
    return evidence, resolve_holding_events(evidence, query_plan=dict(plan.to_dict()))


def record(**over) -> HoldingReportRecord:
    base = dict(
        issuer_corp_code=ISSUER,
        reporter_key=canonical_reporter_key(LONG_HOLDER),
        raw_reporter=LONG_HOLDER,
        doc_id="holding_1",
        projection_chunk_id="holding_1:ch_1",
        reference_date="20231011",
        receipt_date="20231110",
        previous_date="20230727",
        before_shares="6,009,827",
        before_ratio="6.17",
        change_shares="1,036,962",
        change_ratio="1.06",
        change_direction="increase",
        after_shares="7,046,789",
        after_ratio="7.23",
        is_canonical_body=True,
    )
    base.update(over)
    return HoldingReportRecord(**base)


def index_of(*records, **over) -> HoldingReportIndex:
    settings = dict(
        complete=True,
        correction_finality_available=True,
        identity={"corpus_manifest_sha256": "abc"},
    )
    settings.update(over)
    return HoldingReportIndex(records, **settings)


def _relative_plan(question, *, reporter, exact, role=ROLE_CURRENT):
    plan = _plan(question, reporter=reporter, exact=exact)
    return replace(
        plan,
        evidence={
            **dict(plan.evidence),
            "holding_report_relative": {
                "selector": SELECTOR_EXACT_REFERENCE_DATE,
                "projection_role": role,
                "dynamic": False,
                "executable": True,
                "evidence": None,
            },
        },
    )


def _execution_pool(*pairs):
    return SimpleNamespace(
        chunks=tuple(candidate for candidate, _result in pairs),
        results=tuple(result for _candidate, result in pairs),
    )


# ----------------------------------------------------- holder identity rule
class BoundedHolderAliasTests(unittest.TestCase):
    """Only legal suffixes and proven aliases widen Latin holder identity."""

    def test_a_latin_legal_suffix_may_be_omitted(self) -> None:
        self.assertTrue(reporter_matches(LONG_HOLDER, LONG_HOLDER_SHORT))
        self.assertTrue(reporter_matches(LONG_HOLDER_SHORT, LONG_HOLDER))

    def test_a_joined_word_is_still_reached_through_its_own_boundary(self) -> None:
        """Spacing normalizes before the filing's ``Inc`` suffix is removed."""

        self.assertTrue(reporter_matches(SPLIT_HOLDER, SPLIT_HOLDER_SHORT))

    def test_a_prefix_ending_inside_a_token_names_nobody(self) -> None:
        for constraint in ("Vantis Man", "Q. Meridian Capital", "Vanti"):
            with self.subTest(constraint=constraint):
                self.assertFalse(reporter_matches(LONG_HOLDER, constraint))
                self.assertFalse(reporter_matches(SPLIT_HOLDER, constraint))

    def test_a_one_or_two_character_prefix_is_not_a_holder(self) -> None:
        for constraint in ("Q", "Q."):
            with self.subTest(constraint=constraint):
                self.assertFalse(reporter_matches(SPLIT_HOLDER, constraint))

    def test_a_brand_is_not_a_generic_prefix_of_an_asset_manager(self) -> None:
        self.assertFalse(reporter_matches("BlackRock Fund Advisors", "BlackRock"))

    def test_the_proven_fidelity_short_name_is_supported(self) -> None:
        self.assertTrue(
            reporter_matches(
                "Fidelity Management & Research Company LLC", "Fidelity"
            )
        )

    def test_a_korean_stem_is_never_a_prefix_of_a_longer_holder(self) -> None:
        """The 영풍 / 영풍정밀 firewall, restated so it cannot be relaxed."""

        self.assertFalse(reporter_matches(KOREAN_LONGER, KOREAN_STEM))
        self.assertFalse(reporter_matches(KOREAN_STEM, KOREAN_LONGER))

    def test_a_spaced_korean_holder_is_not_split_by_its_spaces(self) -> None:
        self.assertFalse(reporter_matches("가나 연금", "가나"))

    def test_the_existing_designator_and_family_rules_are_unchanged(self) -> None:
        self.assertTrue(reporter_matches("(주)가상영풍", KOREAN_STEM))
        self.assertTrue(reporter_matches("가상연금공단", "가상연금"))
        self.assertFalse(reporter_matches("", LONG_HOLDER_SHORT))
        self.assertFalse(reporter_matches("-", "-"))


# ----------------------------------------------- deterministic index lookup
class ReporterKeyRecoveryTests(unittest.TestCase):
    """The index enumerates a holder's timeline from the name as asked."""

    def test_the_full_corpus_label_still_resolves_exactly(self) -> None:
        index = index_of(record())
        self.assertEqual(
            [r.doc_id for r in index.enumerate_reports(ISSUER, LONG_HOLDER)],
            ["holding_1"],
        )

    def test_a_shortened_holder_reaches_the_same_timeline(self) -> None:
        index = index_of(record())
        self.assertEqual(
            [r.doc_id for r in index.enumerate_reports(ISSUER, LONG_HOLDER_SHORT)],
            ["holding_1"],
        )

    def test_two_compatible_holders_under_one_issuer_decline(self) -> None:
        """A shortened name that could be either holder names neither."""

        index = index_of(
            record(),
            record(
                reporter_key=canonical_reporter_key(OTHER_LATIN_HOLDER),
                raw_reporter=OTHER_LATIN_HOLDER,
                doc_id="holding_2",
                projection_chunk_id="holding_2:ch_1",
            ),
        )
        self.assertEqual(index.enumerate_reports(ISSUER, LONG_HOLDER_SHORT), ())
        self.assertEqual(
            index.select_report(
                ISSUER,
                LONG_HOLDER_SHORT,
                SELECTOR_EXACT_REFERENCE_DATE,
                reference_date="20231011",
            ).status,
            NO_MATCH,
        )

    def test_recovery_never_crosses_the_issuer(self) -> None:
        index = index_of(record(issuer_corp_code=OTHER_ISSUER))
        self.assertEqual(index.enumerate_reports(ISSUER, LONG_HOLDER_SHORT), ())

    def test_a_korean_stem_never_recovers_a_longer_holders_timeline(self) -> None:
        index = index_of(
            record(
                reporter_key=canonical_reporter_key(KOREAN_LONGER),
                raw_reporter=KOREAN_LONGER,
            )
        )
        self.assertEqual(index.enumerate_reports(ISSUER, KOREAN_STEM), ())


class ExactReferenceDateSelectionTests(unittest.TestCase):
    """The date in the question is the reference date, never the receipt date."""

    def setUp(self) -> None:
        self.index = index_of(record())

    def _select(self, selector, **dates):
        return self.index.select_report(
            ISSUER, LONG_HOLDER_SHORT, selector, **dates
        )

    def test_the_reference_date_selects_the_report(self) -> None:
        selection = self._select(
            SELECTOR_EXACT_REFERENCE_DATE, reference_date="2023-10-11"
        )
        self.assertEqual(selection.status, RESOLVED)
        self.assertEqual(selection.selected.doc_id, "holding_1")
        self.assertEqual(selection.selected.after_shares, "7,046,789")

    def test_the_receipt_date_never_answers_a_reference_date_question(self) -> None:
        selection = self._select(
            SELECTOR_EXACT_REFERENCE_DATE, reference_date="2023-11-10"
        )
        self.assertEqual(selection.status, NO_MATCH)

    def test_the_two_axes_stay_separately_addressable(self) -> None:
        selection = self._select(
            SELECTOR_EXACT_RECEIPT_DATE, receipt_date="2023-11-10"
        )
        self.assertEqual(selection.status, RESOLVED)
        self.assertEqual(selection.selected.reference_date, "20231011")

    def test_the_selected_report_keeps_its_own_source_identity(self) -> None:
        selected = self._select(
            SELECTOR_EXACT_REFERENCE_DATE, reference_date="20231011"
        ).selected
        self.assertEqual(selected.projection_chunk_id, "holding_1:ch_1")
        self.assertEqual(selected.issuer_corp_code, ISSUER)
        self.assertEqual(selected.reporter_key, canonical_reporter_key(LONG_HOLDER))

    def test_the_latest_selector_is_unaffected_by_key_recovery(self) -> None:
        """T9: the newest report is still the newest by reference date."""

        index = index_of(
            record(),
            record(
                doc_id="holding_2",
                projection_chunk_id="holding_2:ch_1",
                reference_date="20231205",
                receipt_date="20230101",
                previous_date="20231011",
                after_shares="6,981,127",
            ),
        )
        selection = index.select_report(ISSUER, LONG_HOLDER_SHORT, SELECTOR_LATEST)
        self.assertEqual(selection.status, RESOLVED)
        self.assertEqual(selection.selected.doc_id, "holding_2")


class SameDayIdentityTests(unittest.TestCase):
    """One reference date filed twice stays two filings."""

    def _pair(self, **over):
        return (
            record(**over),
            record(
                doc_id="holding_2",
                projection_chunk_id="holding_2:ch_1",
                is_correction=True,
                **over,
            ),
        )

    def test_a_same_day_pair_is_never_collapsed_by_receipt_date(self) -> None:
        index = index_of(*self._pair())
        reports = index.enumerate_reports(ISSUER, LONG_HOLDER_SHORT)

        self.assertEqual(len(reports), 2)
        self.assertEqual(
            sorted(r.doc_id for r in reports), ["holding_1", "holding_2"]
        )
        self.assertEqual({r.receipt_date for r in reports}, {"20231110"})

    def test_an_unproven_same_day_pair_declines_rather_than_picking_one(self) -> None:
        index = index_of(*self._pair(), correction_finality_available=False)
        selection = index.select_report(
            ISSUER,
            LONG_HOLDER_SHORT,
            SELECTOR_EXACT_REFERENCE_DATE,
            reference_date="20231011",
        )
        self.assertNotEqual(selection.status, RESOLVED)
        self.assertIsNone(selection.selected)

    def test_the_existing_tie_break_decides_only_when_several_remain(self) -> None:
        """Two filings on one date are ambiguous, and both are named."""

        index = index_of(*self._pair())
        selection = index.select_report(
            ISSUER,
            LONG_HOLDER_SHORT,
            SELECTOR_EXACT_REFERENCE_DATE,
            reference_date="20231011",
        )
        self.assertEqual(selection.status, AMBIGUOUS)
        self.assertEqual(
            selection.detail["doc_ids"], ["holding_1", "holding_2"]
        )

    def test_one_reference_date_two_holders_stay_distinct(self) -> None:
        index = index_of(
            record(),
            record(
                reporter_key=canonical_reporter_key(OTHER_LATIN_HOLDER),
                raw_reporter=OTHER_LATIN_HOLDER,
                doc_id="holding_2",
                projection_chunk_id="holding_2:ch_1",
                after_shares="1,111,111",
            ),
        )
        chosen = index.select_report(
            ISSUER,
            OTHER_LATIN_HOLDER,
            SELECTOR_EXACT_REFERENCE_DATE,
            reference_date="20231011",
        )
        self.assertEqual(chosen.status, RESOLVED)
        self.assertEqual(chosen.selected.after_shares, "1,111,111")
        self.assertEqual(chosen.selected.doc_id, "holding_2")


class SameDayFactResolutionTests(unittest.TestCase):
    """C054: duplicated provenance may still prove one requested fact."""

    def _records(self, *, second_after="7,046,789"):
        return (
            record(projection_chunk_id="c1"),
            record(
                doc_id="holding_2",
                projection_chunk_id="c2",
                after_shares=second_after,
            ),
        )

    def _execute(self, *, second_after="7,046,789"):
        return execute_report_relative(
            {
                "selector": SELECTOR_EXACT_REFERENCE_DATE,
                "projection_role": ROLE_CURRENT,
            },
            index=index_of(*self._records(second_after=second_after)),
            issuer_corp_code=ISSUER,
            reporter=LONG_HOLDER_SHORT,
            reference_date="20231011",
            requested_fields=("reference_date", "after_shares"),
        )

    def test_equal_requested_values_resolve_without_merging_documents(self) -> None:
        result = self._execute()

        self.assertEqual(result.status, RESOLVED)
        self.assertEqual(result.projection.values["shares"], "7,046,789")
        self.assertEqual(
            result.selection.detail["equivalent_doc_ids"],
            ["holding_1", "holding_2"],
        )
        self.assertEqual(
            {candidate.doc_id for candidate in result.selection.candidates},
            {"holding_1", "holding_2"},
        )

    def test_one_conflicting_requested_value_keeps_the_selection_ambiguous(self) -> None:
        result = self._execute(second_after="7,046,790")

        self.assertEqual(result.status, AMBIGUOUS)
        self.assertIsNone(result.record)

    def test_unrequested_fields_do_not_turn_an_equal_fact_into_a_conflict(self) -> None:
        first, second = self._records()
        second = replace(second, after_ratio="7.24")
        result = execute_report_relative(
            {
                "selector": SELECTOR_EXACT_REFERENCE_DATE,
                "projection_role": ROLE_CURRENT,
            },
            index=index_of(first, second),
            issuer_corp_code=ISSUER,
            reporter=LONG_HOLDER_SHORT,
            reference_date="20231011",
            requested_fields=("reference_date", "after_shares"),
        )

        self.assertEqual(result.status, RESOLVED)

    def test_fact_resolution_keeps_one_proven_citable_source(self) -> None:
        pairs = [
            _report_projection("c1", "holding_1"),
            _report_projection("c2", "holding_2", rank=2),
        ]
        question = (
            "테스트항공에 대해 Vantis가 2023년 10월 11일을 기준일로 "
            "보고한 보유주식수는?"
        )
        adapted = HoldingReportRelativeExecution(
            index=index_of(*self._records())
        ).adapt(
            question,
            _relative_plan(
                question, reporter=LONG_HOLDER_SHORT, exact="2023-10-11"
            ),
            _execution_pool(*pairs),
            routed_task_type=ROUTED_TASK_TYPE,
        )

        self.assertIsNotNone(adapted)
        self.assertTrue(adapted.resolved)
        self.assertEqual(len(adapted.results), 1)
        self.assertIn(adapted.selected_chunk_id, {"c1", "c2"})

    def test_a_present_equivalent_source_carries_provenance(self) -> None:
        pair = _report_projection("c2", "holding_2")
        question = (
            "테스트항공에 대해 Vantis가 2023년 10월 11일을 기준일로 "
            "보고한 보유주식수는?"
        )
        adapted = HoldingReportRelativeExecution(
            index=index_of(*self._records())
        ).adapt(
            question,
            _relative_plan(
                question, reporter=LONG_HOLDER_SHORT, exact="2023-10-11"
            ),
            _execution_pool(pair),
            routed_task_type=ROUTED_TASK_TYPE,
        )

        self.assertIsNotNone(adapted)
        self.assertTrue(adapted.resolved)
        self.assertEqual(adapted.selected_chunk_id, "c2")
        self.assertEqual(adapted.report_execution.record.doc_id, "holding_2")


# ----------------------------------------------- C030: value keeps its source
class ExactEventProvenanceTests(unittest.TestCase):
    """C030: the value and the chunk it was read from stay one event."""

    QUESTION = (
        "테스트항공에 대해 Vantis가 2023년 10월 11일을 기준일로 "
        "신고한 보유주식수는?"
    )

    def setUp(self) -> None:
        self.plan = _plan(
            self.QUESTION, reporter=LONG_HOLDER_SHORT, exact="2023-10-11"
        )
        self.evidence, self.resolution = _resolve(
            self.plan, [_report_projection("c1", "holding_1")]
        )
        self.draft = compose_holding_answer(
            self.resolution, self.evidence, selection_mode=EXACT
        )

    def test_the_question_asked_for_the_reference_date_and_the_share_count(self) -> None:
        self.assertEqual(
            self.resolution.requested_fields, ("reference_date", "after_shares")
        )

    def test_the_shortened_holder_reaches_the_event(self) -> None:
        event = self.resolution.events[0]
        self.assertIs(event.matches_query, True)
        self.assertEqual(self.resolution.matching_event_count, 1)
        self.assertEqual(event.after_shares.normalized, 7046789)

    def test_the_answer_is_citation_capable(self) -> None:
        self.assertTrue(self.draft.citations)
        self.assertTrue(self.draft.answerable)

    def test_the_citation_is_the_chunk_the_value_was_read_from(self) -> None:
        event = self.resolution.events[0]
        source = event.field_provenance["after_shares"].sources[0]

        self.assertEqual(source.chunk_id, "c1")
        self.assertIn(
            source.chunk_id, {citation.chunk_id for citation in self.draft.citations}
        )
        self.assertEqual(source.doc_id, "holding_1")

    def test_the_reference_date_is_the_events_own_not_its_receipt_date(self) -> None:
        event = self.resolution.events[0]
        self.assertEqual(event.reference_date, "2023-10-11")
        self.assertEqual(event.receipt_date, "2023-11-10")

    def test_another_holder_on_the_same_day_is_still_refused(self) -> None:
        plan = _plan(self.QUESTION, reporter="가상연금", exact="2023-10-11")
        _evidence, resolution = _resolve(
            plan, [_report_projection("c1", "holding_1")]
        )
        self.assertIsNot(resolution.events[0].matches_query, True)

    def test_singo_wording_enters_the_authoritative_exact_date_lane(self) -> None:
        pair = _report_projection("c1", "holding_1")
        adapted = HoldingReportRelativeExecution(
            index=index_of(record(projection_chunk_id="c1"))
        ).adapt(
            self.QUESTION,
            _relative_plan(
                self.QUESTION,
                reporter=LONG_HOLDER_SHORT,
                exact="2023-10-11",
            ),
            _execution_pool(pair),
            routed_task_type=ROUTED_TASK_TYPE,
        )

        self.assertIsNotNone(adapted)
        self.assertTrue(adapted.resolved)
        self.assertEqual(adapted.selected_chunk_id, "c1")
        self.assertEqual(adapted.report_execution.record.reference_date, "20231011")
        self.assertEqual(adapted.report_execution.record.receipt_date, "20231110")


# --------------------------------- C042: the report before, or nothing at all
FIRST_REPORT_QUESTION = (
    "가상영풍의 테스트항공 2024년 9월 23일 기준 보고에서 "
    "직전 보고 대비 증감은 어떻게 되나?"
)


def _first_report(chunk_id="f1", doc_id="holding_first", **over):
    return _report_projection(
        chunk_id,
        doc_id,
        reporter=KOREAN_STEM,
        reference_date="2024년 09월 23일",
        receipt_date="2024-09-24",
        before="-",
        before_ratio="-",
        change="6,859,254",
        after="6,859,254",
        after_ratio="33.13",
        **over,
    )


def _next_report(chunk_id="f2", doc_id="holding_next"):
    return _report_projection(
        chunk_id,
        doc_id,
        reporter=KOREAN_STEM,
        reference_date="2024년 10월 03일",
        receipt_date="2024-10-04",
        before="6,859,254",
        before_ratio="33.13",
        change="0",
        after="6,859,254",
        after_ratio="33.13",
        table="t0034",
        rank=2,
    )


class FirstReportHasNoPredecessorTests(unittest.TestCase):
    """C042: a first filing has no previous state, and no zero either."""

    def _findings(self, pairs, *, question=FIRST_REPORT_QUESTION):
        plan = _plan(question, reporter=KOREAN_STEM, exact="2024-09-23")
        evidence, resolution = _resolve(plan, pairs)
        return evidence, resolution, previous_report_baseline_evidence(
            question=question,
            resolution=resolution,
            evidence_items=evidence.served_items,
        )

    def test_the_question_names_the_day_it_is_about(self) -> None:
        self.assertEqual(question_reference_date(FIRST_REPORT_QUESTION), "20240923")

    def test_the_baseline_fields_are_reported_unavailable(self) -> None:
        _evidence, _resolution, findings = self._findings([_first_report()])

        self.assertEqual(
            sorted(finding.field for finding in findings), sorted(BASELINE_FIELDS)
        )
        for finding in findings:
            with self.subTest(field=finding.field):
                self.assertIs(finding.status, FieldStatus.UNAVAILABLE)

    def test_the_missing_previous_state_is_never_read_as_zero(self) -> None:
        _evidence, _resolution, findings = self._findings([_first_report()])

        for finding in findings:
            with self.subTest(field=finding.field):
                self.assertIsNone(finding.value)

    def test_a_change_against_previous_is_unavailable_on_the_first_report(self) -> None:
        projected = project_role(
            record(
                previous_date=None,
                before_shares=None,
                before_ratio=None,
                change_shares="6,859,254",
                change_ratio="33.13",
            ),
            ROLE_CHANGE,
        )

        self.assertEqual(projected.status, CHANGE_UNAVAILABLE)
        self.assertEqual(projected.values, {})

    def test_the_first_report_keeps_its_own_citation(self) -> None:
        _evidence, _resolution, findings = self._findings([_first_report()])

        for finding in findings:
            with self.subTest(field=finding.field):
                self.assertEqual(finding.chunk_id, "f1")
                self.assertEqual(finding.doc_id, "holding_first")
                self.assertEqual(finding.table_id, "t0053")

    def test_a_later_report_is_never_used_as_the_predecessor(self) -> None:
        """The 2024-10-03 filing states a previous state; it is not this one's."""

        _evidence, resolution, findings = self._findings(
            [_first_report(), _next_report()]
        )

        self.assertEqual(len(resolution.events), 2)
        self.assertEqual(len(findings), len(BASELINE_FIELDS))
        for finding in findings:
            with self.subTest(field=finding.field):
                self.assertIs(finding.status, FieldStatus.UNAVAILABLE)
                self.assertEqual(finding.doc_id, "holding_first")

    def test_a_filing_that_records_a_previous_state_is_left_alone(self) -> None:
        question = (
            "가상영풍의 테스트항공 2024년 10월 03일 기준 보고에서 "
            "직전 보고 대비 증감은 어떻게 되나?"
        )
        plan = _plan(question, reporter=KOREAN_STEM, exact="2024-10-03")
        evidence, resolution = _resolve(plan, [_first_report(), _next_report()])

        self.assertEqual(
            previous_report_baseline_evidence(
                question=question,
                resolution=resolution,
                evidence_items=evidence.served_items,
            ),
            (),
        )

    def test_a_question_that_names_no_previous_report_declines(self) -> None:
        question = "테스트항공에 대해 가상영풍이 2024년 9월 23일 기준 보유주식수는?"
        _evidence, _resolution, findings = self._findings(
            [_first_report()], question=question
        )
        self.assertEqual(findings, ())

    def test_another_holders_first_report_is_never_answered_for(self) -> None:
        plan = _plan(
            FIRST_REPORT_QUESTION, reporter="가상연금", exact="2024-09-23"
        )
        evidence, resolution = _resolve(plan, [_first_report()])

        self.assertEqual(
            previous_report_baseline_evidence(
                question=FIRST_REPORT_QUESTION,
                resolution=resolution,
                evidence_items=evidence.served_items,
            ),
            (),
        )

    def test_the_holding_producer_carries_the_finding_forward(self) -> None:
        """The finding reaches the guard through the existing holding lane."""

        plan = _plan(
            FIRST_REPORT_QUESTION, reporter=KOREAN_STEM, exact="2024-09-23"
        )
        evidence, resolution = _resolve(plan, [_first_report()])
        findings = holding_field_evidence(
            question=FIRST_REPORT_QUESTION,
            resolution=resolution,
            evidence_items=evidence.served_items,
        )
        self.assertEqual(
            sorted(finding.field for finding in findings), sorted(BASELINE_FIELDS)
        )

    def test_the_unavailable_role_still_serves_the_first_report_as_evidence(self) -> None:
        pair = _first_report()
        adapted = HoldingReportRelativeExecution(
            index=index_of(
                record(
                    doc_id="holding_first",
                    projection_chunk_id="f1",
                    reference_date="20240923",
                    receipt_date="20240924",
                    previous_date=None,
                    before_shares=None,
                    before_ratio=None,
                    change_shares="6,859,254",
                    change_ratio="33.13",
                    after_shares="6,859,254",
                    after_ratio="33.13",
                    reporter_key=canonical_reporter_key(KOREAN_STEM),
                    raw_reporter=KOREAN_STEM,
                )
            )
        ).adapt(
            FIRST_REPORT_QUESTION,
            _relative_plan(
                FIRST_REPORT_QUESTION,
                reporter=KOREAN_STEM,
                exact="2024-09-23",
                role=ROLE_CHANGE,
            ),
            _execution_pool(pair),
            routed_task_type=ROUTED_TASK_TYPE,
        )

        self.assertIsNotNone(adapted)
        self.assertFalse(adapted.resolved)
        self.assertEqual(adapted.status, CHANGE_UNAVAILABLE)
        self.assertEqual(adapted.selected_chunk_id, "f1")


class FirstReportFailsClosedTests(unittest.TestCase):
    """The verdict the asker sees: insufficient, with the filing still cited."""

    def setUp(self) -> None:
        plan = _plan(
            FIRST_REPORT_QUESTION, reporter=KOREAN_STEM, exact="2024-09-23"
        )
        evidence, resolution = _resolve(plan, [_first_report(), _next_report()])
        self.draft = compose_holding_answer(
            resolution, evidence, selection_mode=EXACT
        )
        self.findings = holding_field_evidence(
            question=FIRST_REPORT_QUESTION,
            resolution=resolution,
            evidence_items=evidence.served_items,
        )
        self.result = AnswerabilityGuard().evaluate(
            self.draft,
            plan=plan,
            agent_result=_AgentResultStub(resolution, self.findings),
            execution=_ExecutionStub(("f1", "f2")),
        )

    def test_the_delta_question_is_not_answerable(self) -> None:
        self.assertIs(self.result.status, AnswerabilityStatus.INSUFFICIENT_EVIDENCE)
        self.assertFalse(self.result.answerable)

    def test_the_absent_baseline_is_named_rather_than_invented(self) -> None:
        self.assertEqual(
            sorted(self.result.unavailable_fields), sorted(BASELINE_FIELDS)
        )
        self.assertEqual(self.result.confirmed_fields, ())

    def test_the_target_report_is_still_cited(self) -> None:
        self.assertIn("f1", {c.chunk_id for c in self.draft.citations})


class _AgentResultStub:
    def __init__(self, resolution, field_evidence) -> None:
        self.resolution = resolution
        self.field_evidence = field_evidence


class _ExecutionStub:
    def __init__(self, chunk_ids) -> None:
        self.results = [
            RetrievalResult(chunk_id, "d", 1.0, rank, {})
            for rank, chunk_id in enumerate(chunk_ids, start=1)
        ]


# ------------------------------------------------ C054: the holder as written
class MultiWordLatinReporterTests(unittest.TestCase):
    """A foreign holder is several words, and the question writes them all."""

    ISSUER_NAME = "가상발행사"
    OTHER = "가상상대사"

    def setUp(self) -> None:
        self.understanding = QueryUnderstanding(
            {name: {name} for name in (self.ISSUER_NAME, self.OTHER)}
        )

    def reporter(self, query):
        return self.understanding.understand(query).reporter

    def test_a_multi_word_latin_holder_is_read_whole(self) -> None:
        query = (
            f"{self.ISSUER_NAME}에 대해 Q. Meridian Capital Partners가 "
            "2024년 3월 19일 기준으로 보고한 보유주식수는?"
        )
        self.assertEqual(self.reporter(query), "Q. Meridian Capital Partners")

    def test_the_exact_reference_date_survives_beside_the_holder(self) -> None:
        query = (
            f"{self.ISSUER_NAME}에 대해 Q. Meridian Capital Partners가 "
            "2024년 3월 19일 기준으로 보고한 보유주식수는?"
        )
        plan = self.understanding.understand(query)

        self.assertEqual(plan.period.period_type, "holding_reference_date")
        self.assertEqual(exact_reference_date(plan), "2024-03-19")
        self.assertEqual(
            plan.evidence["holding_report_relative"]["selector"],
            SELECTOR_EXACT_REFERENCE_DATE,
        )

    def test_a_single_word_latin_holder_is_unchanged(self) -> None:
        query = f"{self.ISSUER_NAME}에 대해 Vantis가 2023년 10월 11일 보유주식수는?"
        self.assertEqual(self.reporter(query), "Vantis")

    def test_a_korean_holder_is_still_one_token(self) -> None:
        query = f"{self.ISSUER_NAME}에 대한 가상보유인의 최신 보고 보유비율은?"
        self.assertEqual(self.reporter(query), "가상보유인")

    def test_a_second_corpus_company_is_still_never_the_holder(self) -> None:
        """T8: two recognizable companies are a role pair, not a holder."""

        for query in (
            f"{self.OTHER}가 보유한 {self.ISSUER_NAME} 주식은 몇 주인가?",
            f"{self.ISSUER_NAME}에 대한 {self.OTHER}의 최신 보고 보유비율은?",
            f"{self.OTHER}가 2024년 7월 29일 기준으로 보고한 "
            f"{self.ISSUER_NAME} 보유주식수는?",
        ):
            with self.subTest(query=query):
                plan = self.understanding.understand(query)
                self.assertEqual(len(plan.companies), 2)
                self.assertIsNone(plan.reporter)

    def test_a_filing_noun_in_the_holder_slot_still_names_nobody(self) -> None:
        query = f"{self.ISSUER_NAME} 대량보유상황보고서의 보유주식수는?"
        self.assertIsNone(self.reporter(query))


# ------------------------------------------------------- corpus verification
class TrackedCorpusRecoveryTests(unittest.TestCase):
    """The three questions, resolved against the tracked artifacts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = load_index(
            ROOT / "data/corpus/holding_report_index.json",
            finality_path=ROOT / "data/corpus/holding_correction_finality.json",
        )
        cls.identity = repository_corpus_identity(ROOT)
        if cls.index is None or not cls.index.matches_corpus(cls.identity):
            raise unittest.SkipTest("the tracked holding index is not this corpus")

    def _select(self, issuer, reporter, date):
        return self.index.select_report(
            issuer,
            reporter,
            SELECTOR_EXACT_REFERENCE_DATE,
            reference_date=date,
            active_corpus_identity=self.identity,
        )

    def test_c030_resolves_from_the_shortened_holder(self) -> None:
        selection = self._select("00309503", "Fidelity", "20231011")

        self.assertEqual(selection.status, RESOLVED)
        self.assertEqual(selection.selected.doc_id, "holding_20231110000478")
        self.assertEqual(selection.selected.after_shares, "7,046,789")
        self.assertEqual(selection.selected.receipt_date, "20231110")

    def test_c054_resolves_from_the_spaced_holder(self) -> None:
        selection = self._select(
            "00190321", "T. Rowe Price Associates", "20240319"
        )

        self.assertEqual(selection.status, RESOLVED)
        self.assertIn(
            selection.selected.doc_id,
            {"holding_20240412003768", "holding_20240412004007"},
        )
        self.assertEqual(selection.selected.after_shares, "12,898,415")

    def test_c042_names_no_previous_report(self) -> None:
        selection = self._select("00102858", "(주)영풍", "20240923")

        self.assertEqual(selection.status, RESOLVED)
        self.assertIsNone(selection.selected.previous_date)
        self.assertIsNone(selection.selected.before_shares)
        self.assertFalse(selection.selected.has_previous_state)

    def test_a_korean_holder_never_reaches_a_longer_name(self) -> None:
        """영풍 files on this issuer; 영풍정밀 is a different holder entirely."""

        keys = {
            canonical_reporter_key(name)
            for name in self.index.enumerate_reporters("00102858")
        }
        self.assertIn("영풍", keys)
        self.assertEqual(
            self.index.resolve_reporter_key("00102858", "(주)영풍"), "영풍"
        )
        self.assertIsNone(self.index.resolve_reporter_key("00102858", "영풍정밀"))
        self.assertEqual(self.index.enumerate_reports("00102858", "영풍정밀"), ())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
