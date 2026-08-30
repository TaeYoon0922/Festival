"""The runner must not call a fact lost when the run actually delivered it.

Gold marks one chunk that carries a fact.  A structured projection can carry
the same fact, and the frozen holding stack routinely delivers it that way, so
"a gold chunk was not served" and "the answer lost something" are different
claims.  Conflating them attributes a healthy run to P1-C -- the wrong-evidence
reopening this evaluator exists to prevent.
"""

import unittest

import scripts.run_failure_attribution as runner


class _Result:
    def __init__(self, chunk_id, doc_id, rank=1):
        self.chunk_id, self.doc_id, self.rank = chunk_id, doc_id, rank


class _Execution:
    def __init__(self, results):
        self.results = results


def _row(terms, chunk_ids):
    return {"question_id": "Q", "question": "질문",
            "gold": {"evidence_terms": list(terms),
                     "relevant_chunks": [{"chunk_id": c} for c in chunk_ids]}}


class EvidenceMissTests(unittest.TestCase):
    def setUp(self) -> None:
        # one table t1 split into two chunks, plus an unrelated table t9
        self._corpus = {
            "d1": {
                "d1:a": ("t1", "보유주식수 1,000 변동일 2023년 06월 13일"),
                "d1:b": ("t1", "보유주식수 1,000 변동일 2023년 06월 13일"),
                "d1:c": ("t1", "보유비율 7.90"),
                "d1:z": ("t9", "전혀 다른 표"),
            }
        }
        self._real = runner.doc_chunks
        runner.doc_chunks = lambda doc: self._corpus.get(doc, {})

    def tearDown(self) -> None:
        runner.doc_chunks = self._real

    def test_a_redundant_witness_is_not_a_loss(self) -> None:
        """Gold names d1:b, the run served d1:a -- same fact, nothing lost."""

        siblings, coverage, lost = runner._evidence_misses(
            _row(["1,000", "2023년 06월 13일"], ["d1:b"]),
            _Execution([_Result("d1:a", "d1")]))

        self.assertEqual((siblings, coverage, lost), ((), (), ()))

    def test_a_fact_only_the_unserved_sibling_carries_is_a_sibling_miss(self) -> None:
        siblings, coverage, lost = runner._evidence_misses(
            _row(["1,000", "7.90"], ["d1:c"]),
            _Execution([_Result("d1:a", "d1")]))

        self.assertEqual(coverage, ())
        self.assertIn("7.90", lost)
        self.assertEqual(siblings[0]["sibling_chunk_id"], "d1:c")
        self.assertEqual(siblings[0]["table_id"], "t1")
        self.assertEqual(siblings[0]["anchor_chunk_id"], "d1:a")

    def test_a_fact_in_an_unserved_other_table_is_coverage_not_p1c(self) -> None:
        siblings, coverage, lost = runner._evidence_misses(
            _row(["1,000", "전혀 다른 표"], ["d1:z"]),
            _Execution([_Result("d1:a", "d1")]))

        self.assertEqual(siblings, ())
        self.assertEqual(coverage[0]["chunk_id"], "d1:z")
        self.assertEqual(coverage[0]["table_id"], "t9")

    def test_whitespace_differences_do_not_invent_a_loss(self) -> None:
        siblings, coverage, lost = runner._evidence_misses(
            _row(["보유주식수  1,000"], ["d1:b"]),
            _Execution([_Result("d1:a", "d1")]))

        self.assertEqual(lost, ())
        self.assertEqual((siblings, coverage), ((), ()))

    def test_a_question_without_gold_chunks_reports_nothing(self) -> None:
        self.assertEqual(
            runner._evidence_misses(_row([], []), _Execution([])),
            ((), (), ()))


class GoldReadingTests(unittest.TestCase):
    def test_gold_is_read_only_after_the_run(self) -> None:
        """No gold field may reach retrieval: observe() reads it post hoc."""

        import inspect

        source = inspect.getsource(runner.observe)
        head = source[:source.index("execution = ")]
        for forbidden in ("gold_terms(", "gold_chunks(", "_evidence_misses("):
            self.assertNotIn(forbidden, head,
                             f"{forbidden} must not run before retrieval")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
