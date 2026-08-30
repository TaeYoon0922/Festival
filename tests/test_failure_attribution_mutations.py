"""Each mutation the brief names must be caught, not merely unlikely.

A classifier whose guards are untested is a classifier that will quietly drift
back into blaming the wrong phase.  Every test here applies the mutation for
real and asserts the behaviour it would produce is not what ships.
"""

import unittest

import scripts.retrieval_failure_evaluator as ev
import scripts.run_failure_attribution as runner
from tests.test_failure_attribution_runner import _Execution, _Result, _row


def _obs(**overrides):
    base = dict(question_id="Q", retrieval_allowed=True,
                required_docs=("d1",), eligible_docs=("d1", "d2"),
                ranking_available=True, served_docs=("d1",),
                first_required_rank=1, embedding_provider="bge_m3")
    base.update(overrides)
    return ev.Observation(**base)


class MutationTests(unittest.TestCase):
    def test_dropping_filter_precedence_would_blame_ranking(self) -> None:
        """Mutation 1: a filtered-out document must never read as ranked-low."""

        excluded = _obs(eligible_docs=("d2",), served_docs=("d2",),
                        first_required_rank=None)
        self.assertEqual(ev.classify(excluded).primary, ev.FILTER_EXCLUSION)

        order = list(ev.PRECEDENCE)
        self.assertLess(order.index(ev.FILTER_EXCLUSION),
                        order.index(ev.RANKING_LOW),
                        "filter must be decided before ranking")

    def test_treating_a_low_rank_as_exclusion_would_blame_the_filter(self) -> None:
        """Mutation 2: eligible but below the cut is P1-R, never P1-B."""

        a = ev.classify(_obs(served_docs=("d2",), first_required_rank=11))

        self.assertEqual(a.primary, ev.RANKING_LOW)
        self.assertEqual(a.owner, "P1-R")
        self.assertIn("d1", a.details["eligible_docs"])

    def test_gold_cannot_reach_the_selection_path(self) -> None:
        """Mutation 3: gold steering retrieval is the one unrecoverable error."""

        source = ev.__loader__.get_source(ev.__name__)
        body = "".join(source.split('"""')[::2])
        self.assertNotIn("import app", body)
        self.assertNotIn("from app", body)

        import inspect

        observe = inspect.getsource(runner.observe)
        before_retrieval = observe[:observe.index("execution = ")]
        for leak in ("gold_terms(", "gold_chunks(", "_evidence_misses("):
            self.assertNotIn(leak, before_retrieval)

    def test_a_sibling_in_another_document_is_not_a_sibling(self) -> None:
        """Mutation 4: same table_id across documents is coincidence, not a table."""

        corpus = {"d1": {"d1:a": ("t1", "서울 지점 매출")},
                  "d2": {"d2:b": ("t1", "부산 지점 매출 999")}}
        real = runner.doc_chunks
        runner.doc_chunks = lambda doc: corpus.get(doc, {})
        try:
            siblings, coverage, lost = runner._evidence_misses(
                _row(["999"], ["d2:b"]), _Execution([_Result("d1:a", "d1")]))
        finally:
            runner.doc_chunks = real

        self.assertEqual(siblings, (), "table_id must not match across documents")
        self.assertEqual(coverage[0]["chunk_id"], "d2:b")

    def test_dropping_the_confidence_label_would_hide_an_unverified_rank(self) -> None:
        """Mutation 5: a hash-derived ranking claim must announce itself."""

        hashed = ev.classify(_obs(served_docs=("d2",), first_required_rank=11,
                                  embedding_provider="hash"))
        live = ev.classify(_obs(served_docs=("d2",), first_required_rank=11,
                                embedding_provider="bge_m3"))

        self.assertEqual(hashed.primary, live.primary)
        self.assertIn(ev.UNVERIFIED_RANKING, hashed.caveats)
        self.assertNotIn(ev.UNVERIFIED_RANKING, live.caveats)
        for attribution in (hashed, live):
            self.assertTrue(attribution.embedding_confidence)


class AdversarialCaseTests(unittest.TestCase):
    """The nine cases of section 27, each landing in exactly one category."""

    def test_each_case_maps_to_one_category(self) -> None:
        cases = {
            "1 excluded by company filter":
                (_obs(eligible_docs=("d2",),
                      excluded_by={"corp_code": "x"}), ev.FILTER_EXCLUSION),
            "2 eligible but rank 11":
                (_obs(served_docs=("d2",), first_required_rank=11), ev.RANKING_LOW),
            "3 served but fact absent":
                (_obs(evidence_available=True,
                      coverage_misses=({"chunk_id": "c"},)),
                 ev.EVIDENCE_COVERAGE_MISS),
            "4 fact in same-table sibling":
                (_obs(evidence_available=True,
                      sibling_misses=({"table_id": "t1"},)), ev.TABLE_SIBLING_MISS),
            "5 fact in another table":
                (_obs(evidence_available=True,
                      coverage_misses=({"table_id": "t9"},)),
                 ev.EVIDENCE_COVERAGE_MISS),
            "6 one of two required docs":
                (_obs(required_docs=("d1", "d3"), eligible_docs=("d1", "d3"),
                      served_docs=("d1",)), ev.MULTI_DOCUMENT_INCOMPLETE),
            "7 resolver drops evidence":
                (_obs(downstream_available=True,
                      downstream_lost_stage="resolver"), ev.DOWNSTREAM_LOSS),
            "8 P0-D decline":
                (_obs(retrieval_allowed=False,
                      understanding_state="AMBIGUOUS"),
                 ev.QUERY_UNDERSTANDING_DECLINE),
            "9 complete success": (_obs(), ev.COMPLETE),
        }
        for name, (observation, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(ev.classify(observation).primary, expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
