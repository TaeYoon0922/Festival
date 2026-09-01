"""Source-row identity survives how the rows were written down.

A projection records the rows behind each field one ref per row; a served
table chunk records the same rows as one span.  Those are the same source
rows, and ANCHOR_STRONG already means "the same source rows" -- so both
spellings have to read as one.  Everything short of equality (containment,
partial overlap, adjacency, a shared table) stays refused: one table can hold
thousands of holding events, and anchoring on the table would let any of them
answer for any other.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.reasoning.evidence_builder import EvidenceBuilder
from app.reasoning.holding_evidence_coverage import (
    ANCHOR_STRONG,
    PROVENANCE_KEY,
    STATUS_NO_ANCHOR,
    STATUS_RESCUED,
    anchor_tier,
    assess,
)
from app.reasoning.holding_event_resolver import resolve_holding_events
from app.reasoning.query_plan import QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


HOLDER = "테스트투자자"
TABLE = "t0013"


def refs(*spans, table=TABLE):
    return [{"table_id": table, "row_start": a, "row_end": b} for a, b in spans]


def served_table(chunk_id="raw", doc_id="d1", *, spans=((2, 4),), table=TABLE, rank=1):
    """A raw table chunk: source rows only, no projection metadata at all."""

    chunk = {
        "chunk_id": chunk_id, "doc_id": doc_id, "doc_group": "holding",
        "chunk_type": "table", "report_nm": "주식등의대량보유상황보고서(일반)",
        "source_refs": refs(*spans, table=table),
        "section_path": ["제1부 보고의 개요"],
        "content": "표 내용", "retrieval_text": "표 내용",
    }
    return (CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
            RetrievalResult(chunk_id, doc_id, 5.0, rank, MetadataMatch().to_dict()))


def projection(chunk_id="proj", doc_id="d1", *, spans=((2, 2), (3, 3), (4, 4)),
               table=TABLE, shares="1,234,567", ratio="12.34", rank=1,
               reporter=HOLDER, reference_date="2024년 02월 03일"):
    """A holding_report projection whose event fields cite one row each."""

    pf = {
        "보고자/보유자": reporter, "기준일/보고일": reference_date,
        "보유주식수": shares, "보유비율": ratio,
    }
    event = refs(*spans, table=table)
    chunk = {
        "chunk_id": chunk_id, "doc_id": doc_id, "doc_group": "holding",
        "chunk_type": "table_projection", "projection_type": "holding_report",
        "report_nm": "주식등의대량보유상황보고서(일반)",
        "projection_fields": pf,
        "projection_field_refs": {label: [dict(r) for r in event] for label in pf},
        "source_refs": [dict(r) for r in event],
        "section_path": ["제1부 보고의 개요", "3. 보유주식등의 수 및 보유비율"],
        "content": " ".join(f"[{k}] {v}" for k, v in pf.items()),
        "retrieval_text": " ".join(f"[{k}] {v}" for k, v in pf.items()),
    }
    return (CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
            RetrievalResult(chunk_id, doc_id, 1.0, rank, MetadataMatch().to_dict()))


def tier(candidate, served, reporter=HOLDER):
    return anchor_tier(candidate[0].chunk, served[0].chunk, reporter)


class EquivalentRowEncodingTests(unittest.TestCase):
    """The same rows, written two ways, are the same rows."""

    def test_per_row_projection_against_a_served_span(self) -> None:
        self.assertEqual(
            tier(projection(spans=((2, 2), (3, 3), (4, 4))),
                 served_table(spans=((2, 4),))),
            ANCHOR_STRONG,
        )

    def test_span_projection_against_per_row_served_refs(self) -> None:
        self.assertEqual(
            tier(projection(spans=((2, 4),)),
                 served_table(spans=((2, 2), (3, 3), (4, 4)))),
            ANCHOR_STRONG,
        )

    def test_existing_exact_tuple_intersection_still_anchors(self) -> None:
        """The original fast path is untouched: a shared ref triple anchors
        even when the surrounding unions differ."""

        self.assertEqual(
            tier(projection(spans=((2, 2),)),
                 served_table(spans=((2, 2), (9, 9)))),
            ANCHOR_STRONG,
        )

    def test_two_spans_meeting_end_to_end_equal_one_span(self) -> None:
        self.assertEqual(
            tier(projection(spans=((2, 3), (4, 5))), served_table(spans=((2, 5),))),
            ANCHOR_STRONG,
        )

    def test_a_gap_is_never_merged_away(self) -> None:
        """Rows 2-3 and 5 do not cover row 4, so they are not rows 2-5."""

        self.assertIsNone(
            tier(projection(spans=((2, 3), (5, 5))), served_table(spans=((2, 5),)))
        )

    def test_one_matching_table_is_enough(self) -> None:
        """Existential per table: an unrelated second table cannot veto it."""

        candidate = projection(spans=((2, 4),))
        candidate[0].chunk["projection_field_refs"]["보유주식수"].append(
            {"table_id": "t9999", "row_start": 1, "row_end": 1}
        )
        served = served_table(spans=((2, 4),))
        served[0].chunk["source_refs"].append(
            {"table_id": "t9999", "row_start": 7, "row_end": 9}
        )
        self.assertEqual(tier(candidate, served), ANCHOR_STRONG)


class RefusedRelationTests(unittest.TestCase):
    """Everything short of equality stays refused."""

    def test_containment_is_not_an_anchor(self) -> None:
        self.assertIsNone(
            tier(projection(spans=((5, 5),)), served_table(spans=((1, 20),)))
        )

    def test_partial_overlap_is_not_an_anchor(self) -> None:
        self.assertIsNone(
            tier(projection(spans=((2, 5),)), served_table(spans=((4, 7),)))
        )

    def test_adjacency_is_not_an_anchor(self) -> None:
        self.assertIsNone(
            tier(projection(spans=((4, 4),)), served_table(spans=((1, 3),)))
        )

    def test_a_shared_table_alone_is_not_an_anchor(self) -> None:
        self.assertIsNone(
            tier(projection(spans=((2, 4),)), served_table(spans=((10, 12),)))
        )

    def test_a_different_document_never_anchors(self) -> None:
        self.assertIsNone(
            tier(projection(doc_id="d1", spans=((2, 4),)),
                 served_table(doc_id="d2", spans=((2, 4),)))
        )

    def test_a_different_table_never_anchors(self) -> None:
        self.assertIsNone(
            tier(projection(spans=((2, 4),), table="t0013"),
                 served_table(spans=((2, 4),), table="t0044"))
        )


def _plan(**kw):
    base = dict(
        query="보유주식수", raw_query="주식을 얼마나 들고 있어?",
        companies=("테스트회사",), corp_codes=("00000001",),
        task_type="holding_change", disclosure_route=("holding",),
        reporter=HOLDER,
        evidence={"holding_ownership_intent": "company_has_company_shares"},
    )
    base.update(kw)
    return QueryPlan(**base)


class ProjectionRescueTests(unittest.TestCase):
    """The architectural failure, reproduced generically and then closed."""

    def setUp(self) -> None:
        self.raw = served_table("raw", spans=((2, 4),), rank=1)
        self.proj = projection("proj", spans=((2, 2), (3, 3), (4, 4)))
        self.pool = [self.raw[0], self.proj[0]]
        self.served = [self.raw[1]]

    def assess(self, plan=None):
        return assess("주식을 얼마나 들고 있어?", plan or _plan(),
                      self.pool, self.served, routed_task_type="holding_event")

    def test_projection_is_rescued_for_the_current_state_pair(self) -> None:
        outcome = self.assess()

        self.assertEqual(outcome.requested, ("after_shares", "after_ratio"))
        self.assertEqual(outcome.status, STATUS_RESCUED)
        self.assertTrue(outcome.rescued)
        self.assertEqual(outcome.anchored_candidate_count, 1)
        self.assertIn("proj", [row.chunk_id for row in outcome.results])

    def test_promoted_row_keeps_rank_and_provenance_convention(self) -> None:
        outcome = self.assess()
        promoted = next(r for r in outcome.results if r.chunk_id == "proj")

        self.assertEqual([r.rank for r in outcome.results],
                         list(range(1, len(outcome.results) + 1)))
        self.assertEqual(dict(promoted.metadata_match).get(PROVENANCE_KEY),
                         {"selected_for": "holding_field_coverage"})

    def test_explicit_shares_and_ratio_are_rescued_too(self) -> None:
        """Not a BOTH-only repair: the same encoding gap blocked every shape."""

        for metric, field in (("holding_shares", "after_shares"),
                              ("holding_ratio", "after_ratio")):
            with self.subTest(metric=metric):
                outcome = self.assess(_plan(metric=metric, raw_query="보유 현황"))
                self.assertEqual(outcome.requested, (field,))
                self.assertTrue(outcome.rescued)

    def test_a_disjoint_projection_is_still_declined(self) -> None:
        far = projection("far", spans=((30, 32),))
        outcome = assess("주식을 얼마나 들고 있어?", _plan(),
                         [self.raw[0], far[0]], self.served,
                         routed_task_type="holding_event")

        self.assertEqual(outcome.status, STATUS_NO_ANCHOR)
        self.assertFalse(outcome.rescued)

    def test_a_wrong_holder_projection_is_never_rescued(self) -> None:
        other = projection("other", spans=((2, 2), (3, 3), (4, 4)),
                           reporter="다른투자자")
        outcome = assess("주식을 얼마나 들고 있어?", _plan(),
                         [self.raw[0], other[0]], self.served,
                         routed_task_type="holding_event")

        self.assertFalse(outcome.rescued)
        self.assertNotIn("other", [row.chunk_id for row in outcome.results])

    def test_rescued_projection_becomes_one_citable_holding_event(self) -> None:
        """End to end: the rescue is what lets the resolver see an event."""

        outcome = self.assess()
        by_id = {c.chunk_id: c for c in self.pool}
        chunks = [by_id[r.chunk_id] for r in outcome.results]

        plan = _plan()
        execution = SimpleNamespace(
            plan=plan, chunks=chunks, results=list(outcome.results)
        )
        evidence_set = EvidenceBuilder().build(
            execution,
            question=str(plan.raw_query),
            grouping_intent="holding_change",
        )
        groups = [g for g in evidence_set.evidence_groups
                  if g.group_type == "holding_event"]
        self.assertEqual(len(groups), 1, "exactly one holding event")

        resolution = resolve_holding_events(evidence_set, query_plan=_plan())
        self.assertTrue(resolution.events, "the rescue must yield an event")
        event = resolution.events[0]
        self.assertIsNotNone(event.after_shares, "after_shares confirmed")
        self.assertIsNotNone(event.after_ratio, "after_ratio confirmed")
        self.assertTrue(event.source_refs, "the event is citable")
        self.assertNotIn("no_holding_event_groups", resolution.warnings)


if __name__ == "__main__":
    unittest.main()
