"""T2-E: which state of a corrected filing answers a question about the holding.

A 정정신고 reprints the report it corrects.  The filing therefore states the same
field twice under its own labels -- once as ``정정 전`` and once as ``정정 후`` --
before stating it a third time in its body.  These tests pin the contract that
decides between them: the state is read from the filing's own labels, never from
the values, the ranks or the table order; a question that asks for the prior
state keeps it; and anything the labels do not settle stays a field conflict.
"""

import unittest

from app.reasoning.correction_pair_roles import (
    ROLE_AFTER,
    ROLE_BEFORE,
    decide_correction_pair,
)
from app.reasoning.evidence_builder import build_evidence_set
from app.reasoning.holding_correction_state import (
    CORRECTED_STATE,
    PRIOR_STATE,
    PRIOR_STATE_SUPERSEDED,
    declared_correction_state,
)
from app.reasoning.holding_event_resolver import resolve_holding_events
from app.reasoning.query_plan import QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


QUESTION = "테스트홀딩스가 보고한 테스트회사 보유주식수는 몇 주야?"

#: The body of a report: the region a filing states its own report in, which
#: carries no correction-state label of any kind.
BODY_SECTION = ("제1부 보고의 개요", "3. 보유주식등의 수 및 보유비율")


def _projection(
    chunk_id: str,
    doc_id: str,
    *,
    rank: int,
    table_id: str,
    after_shares: str,
    change_shares: str,
    section: tuple[str, ...] = BODY_SECTION,
    table_title: str | None = None,
    reporter: str = "테스트홀딩스",
    date: str = "2024년 07월 29일",
) -> tuple[CandidateChunk, RetrievalResult]:
    """One holding report projection, as the chunker materializes one."""

    ref = {"table_id": table_id, "row_start": 3, "row_end": 3}
    fields = {
        "보고자/보유자": reporter,
        "기준일/보고일": date,
        "직전 보유주식수": "1,000",
        "증감주식수": change_shares,
        "보유주식수": after_shares,
        "보유비율": "20.08",
    }
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": "00123456",
        "corp_name": "테스트회사",
        "doc_group": "holding",
        "chunk_type": "table_projection",
        "section_title": section[-1],
        "section_path": list(section),
        "table_title": table_title,
        "content": f"[보유주식수] {after_shares}",
        "retrieval_text": f"[보유주식수] {after_shares}",
        "report_nm": "[기재정정]주식등의대량보유상황보고서(일반)",
        "rcept_dt": "20241025",
        "projection_type": "holding_report",
        "table_id": table_id,
        "source_table_id": table_id,
        "source_table_ids": [table_id],
        "source_refs": [ref],
        "projection_fields": dict(fields),
        "projection_field_refs": {label: [ref] for label in fields},
    }
    candidate = CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch())
    result = RetrievalResult(chunk_id, doc_id, 1.0 / rank, rank, {})
    return candidate, result


def _plan(**evidence) -> QueryPlan:
    correction_policy = evidence.pop("correction_policy", "any")
    return QueryPlan(
        query=QUESTION,
        task_type="holding_event",
        correction_policy=correction_policy,
        evidence=dict(evidence),
    )


def _resolve(pairs, *, plan: QueryPlan | None = None, question: str = QUESTION):
    plan = plan or _plan()
    evidence = build_evidence_set(
        question=question,
        query_plan=plan,
        candidates=[pair[0] for pair in pairs],
        results=[pair[1] for pair in pairs],
        grouping_intent="holding_change",
    )
    return evidence, resolve_holding_events(evidence, query_plan=plan)


def _served_chunk_ids(evidence) -> set[str]:
    return {item.chunk_id for group in evidence.evidence_groups for item in group.items}


def _holding_events(evidence):
    return [
        group
        for group in evidence.evidence_groups
        if group.group_type == "holding_event"
    ]


def _event_members(evidence) -> set[str]:
    return {item.chunk_id for group in _holding_events(evidence) for item in group.items}


def _authority_warnings(evidence) -> list[str]:
    return [w for w in evidence.warnings if w.startswith(PRIOR_STATE_SUPERSEDED)]


def _corrected_filing():
    """The 정정 전 / 정정 후 / body triplet one corrected filing carries."""

    return [
        _projection(
            "c1:ch_prior",
            "c1",
            rank=1,
            table_id="t0008",
            after_shares="1,198,888,008",
            change_shares="5,067",
            section=("정정 신고 (주2)", "정정 신고 (주2) 정정 전"),
        ),
        _projection(
            "c1:ch_corrected",
            "c1",
            rank=2,
            table_id="t0009",
            after_shares="1,198,889,258",
            change_shares="6,317",
            section=("정정 신고 (주2)", "정정 신고 (주2) 정정 후"),
        ),
        _projection(
            "c1:ch_body",
            "c1",
            rank=3,
            table_id="t0043",
            after_shares="1,198,889,258",
            change_shares="6,317",
        ),
    ]


def _correction_trace() -> dict:
    return {
        "correction_group_count": 1,
        "correction_groups": [
            {
                "correction_group_id": "g1",
                "root_doc_id": "root",
                "latest_doc_id": "c1",
            }
        ],
    }


class CorrectionStateReadingTests(unittest.TestCase):
    """A. The state is read from the filing's own structural labels."""

    def test_state_is_read_from_section_and_table_labels(self):
        for label_field in ("section_title", "table_title"):
            for text, expected in (
                ("정정 신고 (주2) 정정 전", PRIOR_STATE),
                ("정정 신고 (주2) 정정 후", CORRECTED_STATE),
                ("정정전", PRIOR_STATE),
                ("정정후 기준 보유현황", CORRECTED_STATE),
            ):
                with self.subTest(field=label_field, text=text):
                    self.assertEqual(
                        declared_correction_state({label_field: text}), expected
                    )

    def test_labels_that_declare_no_single_state_declare_none(self):
        for text in (
            "3. 보유주식등의 수 및 보유비율",
            # Names both states, so it describes a comparison, not one side.
            "정정 전후 비교",
            "정정 전과 정정 후",
            # A cell saying a row did not change is not a state heading.
            "정정 전과 동일",
            # The marker has to end its own word.
            "정정 전액 반환",
        ):
            with self.subTest(text=text):
                self.assertIsNone(declared_correction_state({"section_title": text}))

    def test_state_is_carried_on_the_served_evidence(self):
        evidence, _ = _resolve(_corrected_filing())
        states = {
            item.chunk_id: item.holding.get("correction_state")
            for group in evidence.evidence_groups
            for item in group.items
        }
        self.assertEqual(states["c1:ch_prior"], PRIOR_STATE)
        self.assertEqual(states["c1:ch_corrected"], CORRECTED_STATE)
        self.assertIsNone(states["c1:ch_body"])


class CurrentStateAuthorityTests(unittest.TestCase):
    """A. A current-holding question is answered from the corrected state."""

    def test_corrected_state_answers_and_no_false_conflict_is_reported(self):
        _evidence, resolution = _resolve(_corrected_filing())

        events = [event for event in resolution.events if event.matches_query is True]
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.after_shares.raw, "1,198,889,258")
        self.assertEqual(event.change_shares.raw, "6,317")
        self.assertFalse(event.field_conflict)
        self.assertEqual(event.conflicting_fields, ())
        self.assertNotIn("field_conflict:after_shares", resolution.warnings)
        self.assertNotIn("field_conflict:change_shares", resolution.warnings)
        self.assertEqual(resolution.unresolved_fields, ())

    def test_the_superseded_state_is_still_served_evidence(self):
        evidence, _ = _resolve(_corrected_filing())
        self.assertIn("c1:ch_prior", _served_chunk_ids(evidence))
        self.assertNotIn("c1:ch_prior", _event_members(evidence))

    def test_the_authority_decision_is_recorded(self):
        evidence, _ = _resolve(_corrected_filing())
        self.assertIn(f"{PRIOR_STATE_SUPERSEDED}:c1:ch_prior", evidence.warnings)

    def test_a_prior_state_alone_is_never_resolved_away(self):
        """Nothing else states the event, so removing it would lose it entirely."""

        prior, _corrected, _body = _corrected_filing()
        evidence, resolution = _resolve([prior])
        self.assertEqual(len(_holding_events(evidence)), 1)
        self.assertEqual(resolution.events[0].after_shares.raw, "1,198,888,008")
        self.assertEqual(_authority_warnings(evidence), [])


class PriorStateQuestionTests(unittest.TestCase):
    """B. A question about the prior state keeps every bit of it."""

    def test_history_intent_keeps_both_states_in_the_event(self):
        evidence, resolution = _resolve(
            _corrected_filing(), plan=_plan(correction_intent="history")
        )
        members = _event_members(evidence)
        self.assertIn("c1:ch_prior", members)
        self.assertIn("c1:ch_corrected", members)
        self.assertTrue(resolution.events[0].field_conflict)
        self.assertEqual(_authority_warnings(evidence), [])

    def test_original_intent_keeps_the_prior_state(self):
        evidence, _ = _resolve(
            _corrected_filing(), plan=_plan(correction_intent="original")
        )
        self.assertIn("c1:ch_prior", _event_members(evidence))

    def test_original_only_policy_keeps_the_prior_state(self):
        evidence, _ = _resolve(
            _corrected_filing(), plan=_plan(correction_policy="original_only")
        )
        self.assertIn("c1:ch_prior", _event_members(evidence))


class FailClosedTests(unittest.TestCase):
    """C/F. Only a conflict the labels settle is settled."""

    def test_two_unlabelled_values_stay_a_field_conflict(self):
        pairs = [
            _projection(
                "u1:ch_one",
                "u1",
                rank=1,
                table_id="t0008",
                after_shares="1,198,888,008",
                change_shares="5,067",
            ),
            _projection(
                "u1:ch_two",
                "u1",
                rank=2,
                table_id="t0009",
                after_shares="1,198,889,258",
                change_shares="6,317",
            ),
        ]
        evidence, resolution = _resolve(pairs)
        self.assertTrue(resolution.events[0].field_conflict)
        self.assertIn("field_conflict:after_shares", resolution.warnings)
        self.assertEqual(_authority_warnings(evidence), [])

    def test_two_prior_states_and_nothing_else_stay_a_field_conflict(self):
        pairs = [
            _projection(
                "p1:ch_one",
                "p1",
                rank=1,
                table_id="t0008",
                after_shares="1,198,888,008",
                change_shares="5,067",
                section=("정정 신고", "정정 신고 (주2) 정정 전"),
            ),
            _projection(
                "p1:ch_two",
                "p1",
                rank=2,
                table_id="t0010",
                after_shares="1,198,889,258",
                change_shares="6,317",
                section=("정정 신고", "정정 신고 (주3) 정정 전"),
            ),
        ]
        evidence, resolution = _resolve(pairs)
        self.assertTrue(resolution.events[0].field_conflict)
        self.assertEqual(_authority_warnings(evidence), [])

    def test_a_heading_naming_both_states_settles_nothing(self):
        pairs = [
            _projection(
                "b1:ch_one",
                "b1",
                rank=1,
                table_id="t0008",
                after_shares="1,198,888,008",
                change_shares="5,067",
                section=("정정 신고", "정정 전후 대비표"),
            ),
            _projection(
                "b1:ch_two",
                "b1",
                rank=2,
                table_id="t0043",
                after_shares="1,198,889,258",
                change_shares="6,317",
            ),
        ]
        evidence, resolution = _resolve(pairs)
        self.assertTrue(resolution.events[0].field_conflict)
        self.assertEqual(_authority_warnings(evidence), [])

    def test_a_prior_state_never_supersedes_another_filing(self):
        """The declaration is about the filing that wrote it, and no other."""

        pairs = [
            _projection(
                "o1:ch_body",
                "o1",
                rank=1,
                table_id="t0016",
                after_shares="1,198,888,008",
                change_shares="5,067",
            ),
            _projection(
                "c1:ch_prior",
                "c1",
                rank=2,
                table_id="t0008",
                after_shares="1,198,888,008",
                change_shares="5,067",
                section=("정정 신고 (주2)", "정정 신고 (주2) 정정 전"),
            ),
        ]
        evidence, _ = _resolve(pairs)
        members = _event_members(evidence)
        self.assertIn("c1:ch_prior", members)
        self.assertIn("o1:ch_body", members)


class UnchangedBehaviourTests(unittest.TestCase):
    """D. A filing that corrects nothing is untouched."""

    def test_ordinary_holding_document_keeps_its_field_authority(self):
        pairs = [
            _projection(
                "o1:ch_body",
                "o1",
                rank=1,
                table_id="t0016",
                after_shares="1,198,888,008",
                change_shares="5,067",
            ),
        ]
        evidence, resolution = _resolve(pairs)
        self.assertEqual(len(_holding_events(evidence)), 1)
        event = resolution.events[0]
        self.assertEqual(event.after_shares.raw, "1,198,888,008")
        self.assertFalse(event.field_conflict)
        self.assertEqual(_authority_warnings(evidence), [])


class CorrectionPairRolesTests(unittest.TestCase):
    """E. T2-B's document-version pair is unaffected."""

    def test_the_before_after_pair_still_binds_on_a_history_question(self):
        pairs = [
            _projection(
                "root:ch_body",
                "root",
                rank=1,
                table_id="t0016",
                after_shares="1,198,888,008",
                change_shares="5,067",
            ),
            _projection(
                "c1:ch_body",
                "c1",
                rank=2,
                table_id="t0043",
                after_shares="1,198,889,258",
                change_shares="6,317",
            ),
        ]
        plan = _plan(correction_intent="history")
        _evidence, resolution = _resolve(pairs, plan=plan)
        decision = decide_correction_pair(
            resolution, correction_trace=_correction_trace(), query_plan=plan
        )
        self.assertIsNotNone(decision.claim)
        self.assertEqual(decision.claim.before_doc_id, "root")
        self.assertEqual(decision.claim.after_doc_id, "c1")
        self.assertEqual(
            (ROLE_BEFORE, ROLE_AFTER), ("correction_before", "correction_after")
        )

    def test_a_history_question_declines_exactly_as_it_did_before(self):
        """A history question keeps both of the corrected filing's own states,
        so the field stays unresolved and the pair declines on the frozen
        reason -- the behaviour this contract deliberately does not touch."""

        pairs = [
            _projection(
                "root:ch_body",
                "root",
                rank=1,
                table_id="t0016",
                after_shares="1,198,888,008",
                change_shares="5,067",
            ),
            *_corrected_filing(),
        ]
        plan = _plan(correction_intent="history")
        _evidence, resolution = _resolve(pairs, plan=plan)
        decision = decide_correction_pair(
            resolution, correction_trace=_correction_trace(), query_plan=plan
        )
        self.assertIsNone(decision.claim)
        self.assertEqual(decision.reason, "field_missing")


class CitationProvenanceTests(unittest.TestCase):
    """G. The answered value is cited from the state it was read from."""

    def test_provenance_points_at_current_state_evidence_only(self):
        _evidence, resolution = _resolve(_corrected_filing())
        provenance = resolution.events[0].field_provenance["after_shares"]
        self.assertFalse(provenance.field_conflict)
        chunk_ids = {source.chunk_id for source in provenance.sources}
        self.assertEqual(chunk_ids, {"c1:ch_corrected", "c1:ch_body"})
        self.assertNotIn("c1:ch_prior", chunk_ids)
        tables = {
            str(ref.get("table_id"))
            for source in provenance.sources
            for ref in source.source_refs
        }
        self.assertEqual(tables, {"t0009", "t0043"})
        self.assertTrue(all(source.direct_field_ref for source in provenance.sources))


if __name__ == "__main__":
    unittest.main()
