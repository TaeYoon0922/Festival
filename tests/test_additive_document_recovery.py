"""One filing must not be able to hide every other document behind it.

Replacing a crowding chunk was tried and rejected: it removed evidence other
answers depended on.  So this recovery only ever *appends*, and the first
assertion of almost every test below is that the emitted results are byte-
identical -- same chunk ids, same order -- before and after.

The retrieval layer here knows only how many chunks each document holds and the
order the reranker already produced.  It never learns what a chunk says.
"""

import unittest

from app.retrieval.hybrid import (
    ADDITIVE_DOCUMENT_RESCUE_LIMIT,
    DOCUMENT_CROWDING_THRESHOLD,
    ScoredTail,
    _additive_document_rescue,
)
from app.retrieval.interfaces import RetrievalResult


def _result(chunk_id, doc_id, rank=1):
    return RetrievalResult(chunk_id=chunk_id, doc_id=doc_id, bm25_score=0.0,
                           rank=rank, metadata_match={})


class _Candidate:
    def __init__(self, chunk_id, doc_id) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id


def _tail(*pairs, start_rank=11):
    """Scored-but-unemitted rows, in the order the reranker produced them."""

    rows = tuple(
        {"candidate": _Candidate(chunk, doc), "final_rank": start_rank + offset}
        for offset, (chunk, doc) in enumerate(pairs)
    )
    return ScoredTail(
        rows=rows,
        chunks_by_id={},
        builder=lambda row, _chunks: _result(
            row["candidate"].chunk_id, row["candidate"].doc_id,
            rank=int(row["final_rank"])),
    )


def _crowded_top10():
    """Document A holds three slots; B and C fill the rest."""

    return [
        _result("a1", "A", 1), _result("a2", "A", 2), _result("a3", "A", 3),
        _result("b1", "B", 4), _result("b2", "B", 5), _result("c1", "C", 6),
        _result("c2", "C", 7), _result("b3", "B", 8), _result("c3", "C", 9),
        _result("a4", "A", 10),
    ]


def _ids(results):
    return [r.chunk_id for r in results]


class PositiveRecoveryTests(unittest.TestCase):
    def test_a_crowded_list_gains_one_unseen_document(self) -> None:
        baseline = _crowded_top10()
        out, trace = _additive_document_rescue(baseline, _tail(("d1", "D")))

        self.assertEqual(_ids(out)[:10], _ids(baseline))
        self.assertEqual(len(out), 11)
        self.assertEqual(out[10].chunk_id, "d1")
        self.assertEqual(out[10].doc_id, "D")
        self.assertTrue(trace["crowding_detected"])
        self.assertTrue(trace["appended"])
        self.assertEqual(trace["appended_doc_id"], "D")
        self.assertEqual(trace["original_candidate_rank"], 11)

    def test_the_appended_candidate_keeps_its_own_scored_rank(self) -> None:
        """A rescued candidate is not renumbered into the emitted sequence."""

        out, trace = _additive_document_rescue(
            _crowded_top10(), _tail(("z1", "Z"), start_rank=42))

        self.assertEqual(out[10].rank, 42)
        self.assertEqual(trace["original_candidate_rank"], 42)

    def test_the_first_unseen_document_in_scored_order_wins(self) -> None:
        out, _ = _additive_document_rescue(
            _crowded_top10(), _tail(("d1", "D"), ("e1", "E")))

        self.assertEqual(out[10].doc_id, "D")


class NoCrowdingTests(unittest.TestCase):
    def test_an_uncrowded_list_is_returned_untouched(self) -> None:
        baseline = [_result(f"x{i}", f"D{i}", i) for i in range(1, 11)]

        out, trace = _additive_document_rescue(baseline, _tail(("d1", "NEW")))

        self.assertEqual(_ids(out), _ids(baseline))
        self.assertEqual(len(out), 10)
        self.assertFalse(trace["crowding_detected"])
        self.assertFalse(trace["appended"])

    def test_two_chunks_from_one_document_is_not_crowding(self) -> None:
        baseline = [_result("a1", "A", 1), _result("a2", "A", 2),
                    *[_result(f"x{i}", f"D{i}", i) for i in range(3, 11)]]

        out, trace = _additive_document_rescue(baseline, _tail(("d1", "NEW")))

        self.assertEqual(_ids(out), _ids(baseline))
        self.assertEqual(trace["max_chunks_from_document"], 2)
        self.assertFalse(trace["appended"])

    def test_the_threshold_is_three(self) -> None:
        self.assertEqual(DOCUMENT_CROWDING_THRESHOLD, 3)


class NoUnseenDocumentTests(unittest.TestCase):
    def test_nothing_is_appended_when_every_tail_document_is_present(self) -> None:
        baseline = _crowded_top10()

        out, trace = _additive_document_rescue(
            baseline, _tail(("a9", "A"), ("b9", "B"), ("c9", "C")))

        self.assertEqual(_ids(out), _ids(baseline))
        self.assertTrue(trace["crowding_detected"])
        self.assertFalse(trace["appended"])

    def test_an_empty_tail_appends_nothing(self) -> None:
        baseline = _crowded_top10()

        out, trace = _additive_document_rescue(baseline, _tail())

        self.assertEqual(_ids(out), _ids(baseline))
        self.assertFalse(trace["appended"])

    def test_a_missing_tail_appends_nothing(self) -> None:
        baseline = _crowded_top10()

        out, trace = _additive_document_rescue(baseline, None)

        self.assertEqual(_ids(out), _ids(baseline))
        self.assertFalse(trace["attempted"])


class OneOnlyBoundTests(unittest.TestCase):
    def test_only_one_unseen_document_is_appended(self) -> None:
        out, _ = _additive_document_rescue(
            _crowded_top10(), _tail(("d1", "D"), ("e1", "E"), ("f1", "F")))

        self.assertEqual(len(out), 11)
        self.assertEqual([r.doc_id for r in out[10:]], ["D"])

    def test_the_limit_is_one(self) -> None:
        self.assertEqual(ADDITIVE_DOCUMENT_RESCUE_LIMIT, 1)


class DocumentIdentityTests(unittest.TestCase):
    """Identity comes from doc_id; nothing is inferred from the chunk id."""

    def test_results_without_a_document_id_never_signal_crowding(self) -> None:
        baseline = [_result(f"x{i}", "", i) for i in range(1, 11)]

        out, trace = _additive_document_rescue(baseline, _tail(("d1", "D")))

        self.assertEqual(trace["max_chunks_from_document"], 0)
        self.assertFalse(trace["appended"])
        self.assertEqual(_ids(out), _ids(baseline))

    def test_a_tail_candidate_without_a_document_id_is_skipped(self) -> None:
        out, trace = _additive_document_rescue(
            _crowded_top10(), _tail(("n1", ""), ("d1", "D")))

        self.assertEqual(out[10].chunk_id, "d1")
        self.assertEqual(trace["appended_doc_id"], "D")

    def test_documents_sharing_a_chunk_id_prefix_stay_distinct(self) -> None:
        baseline = [_result("shared:1", "A", 1), _result("shared:2", "A", 2),
                    _result("shared:3", "A", 3),
                    *[_result(f"x{i}", f"D{i}", i) for i in range(4, 11)]]

        out, trace = _additive_document_rescue(baseline, _tail(("shared:9", "B")))

        self.assertTrue(trace["appended"])
        self.assertEqual(out[10].doc_id, "B")


class DuplicateTests(unittest.TestCase):
    def test_a_chunk_already_emitted_is_never_appended_again(self) -> None:
        baseline = _crowded_top10()

        out, trace = _additive_document_rescue(baseline, _tail(("a1", "A")))

        self.assertEqual(_ids(out), _ids(baseline))
        self.assertFalse(trace["appended"])


class DeterminismTests(unittest.TestCase):
    def test_repeated_calls_produce_the_same_output(self) -> None:
        tail = _tail(("d1", "D"), ("e1", "E"))

        first = _ids(_additive_document_rescue(_crowded_top10(), tail)[0])
        second = _ids(_additive_document_rescue(_crowded_top10(), tail)[0])

        self.assertEqual(first, second)

    def test_the_input_results_are_not_mutated(self) -> None:
        baseline = _crowded_top10()
        before = _ids(baseline)

        _additive_document_rescue(baseline, _tail(("d1", "D")))

        self.assertEqual(_ids(baseline), before)


class BudgetContractTests(unittest.TestCase):
    """The normal retrieval budget must not be widened to make room."""

    def test_the_configured_top_k_default_is_untouched(self) -> None:
        from app.retrieval.hybrid import HybridRetrievalConfig

        self.assertEqual(HybridRetrievalConfig().final_top_k, 10)

    def test_the_rescue_adds_at_most_one_beyond_the_budget(self) -> None:
        out, _ = _additive_document_rescue(
            _crowded_top10(), _tail(*[(f"d{i}", f"D{i}") for i in range(1, 9)]))

        self.assertLessEqual(len(out), 10 + ADDITIVE_DOCUMENT_RESCUE_LIMIT)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
