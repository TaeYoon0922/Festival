"""The evaluator must name the stage that lost the question, and only that one.

Each fixture is a run whose loss point is known by construction, so a
misclassification here is the evaluator's fault rather than the pipeline's.
That matters more than usual: this classifier exists to stop a filter phase
being reopened for a ranking problem, so it is judged on whether it separates
those two cases under pressure.
"""

import unittest

from scripts.retrieval_failure_evaluator import (
    COMPLETE,
    DOWNSTREAM_LOSS,
    EVIDENCE_COVERAGE_MISS,
    FILTER_EXCLUSION,
    HASH_DIAGNOSTIC,
    LIVE_BGE_M3,
    MULTI_DOCUMENT_INCOMPLETE,
    OWNER,
    PRECEDENCE,
    QUERY_UNDERSTANDING_DECLINE,
    RANKING_LOW,
    TABLE_SIBLING_MISS,
    UNVERIFIED_RANKING,
    Observation,
    aggregate,
    classify,
    confidence_from_provider,
)


def _run(**overrides):
    """A run that succeeded at every stage, unless a field says otherwise."""

    base = dict(
        question_id="Q", question="테스트 질문",
        retrieval_allowed=True,
        required_docs=("d1",), eligible_docs=("d1", "d2"),
        ranking_available=True, served_docs=("d1", "d2"),
        first_required_rank=1,
        evidence_available=True, downstream_available=True,
        embedding_provider="bge_m3",
    )
    base.update(overrides)
    return Observation(**base)


class CategoryTests(unittest.TestCase):
    """27: one fixture per loss point, each mapping to exactly one category."""

    def test_a_declined_question_is_attributed_to_understanding(self) -> None:
        a = classify(_run(retrieval_allowed=False,
                          understanding_state="AMBIGUOUS"))

        self.assertEqual(a.primary, QUERY_UNDERSTANDING_DECLINE)
        self.assertEqual(a.owner, "P0-D")
        self.assertEqual(a.details["understanding_state"], "AMBIGUOUS")

    def test_a_document_excluded_before_scoring_is_a_filter_loss(self) -> None:
        a = classify(_run(eligible_docs=("d2",),
                          excluded_by={"company": "다른회사"}))

        self.assertEqual(a.primary, FILTER_EXCLUSION)
        self.assertEqual(a.owner, "P1-B")
        self.assertEqual(a.details["ineligible_docs"], ["d1"])

    def test_an_eligible_document_below_the_cut_is_a_ranking_loss(self) -> None:
        a = classify(_run(served_docs=("d2",), first_required_rank=11))

        self.assertEqual(a.primary, RANKING_LOW)
        self.assertEqual(a.owner, "P1-R")
        self.assertEqual(a.details["missing_required_docs"], ["d1"])

    def test_a_fact_in_an_unserved_same_table_sibling_is_a_sibling_miss(self) -> None:
        a = classify(_run(sibling_misses=({"anchor_chunk_id": "c1",
                                           "sibling_chunk_id": "c2",
                                           "table_id": "t1",
                                           "row_distance": 1,
                                           "matched_term": "부채총계"},)))

        self.assertEqual(a.primary, TABLE_SIBLING_MISS)
        self.assertEqual(a.owner, "P1-C")
        self.assertEqual(a.details["sibling_misses"][0]["table_id"], "t1")

    def test_a_fact_in_another_table_is_a_coverage_miss_not_a_sibling_miss(self) -> None:
        """5: a different table is explicitly not P1-C's problem."""

        a = classify(_run(coverage_misses=({"chunk_id": "c9",
                                            "table_id": "t9"},)))

        self.assertEqual(a.primary, EVIDENCE_COVERAGE_MISS)
        self.assertNotEqual(a.owner, "P1-C")

    def test_one_of_two_required_documents_is_a_multi_document_gap(self) -> None:
        a = classify(_run(required_docs=("d1", "d3"),
                          eligible_docs=("d1", "d3"),
                          served_docs=("d1",)))

        self.assertEqual(a.primary, MULTI_DOCUMENT_INCOMPLETE)
        self.assertEqual(a.details["missing_required_docs"], ["d3"])

    def test_evidence_present_but_dropped_later_is_a_downstream_loss(self) -> None:
        a = classify(_run(downstream_lost_stage="holding_event_resolver"))

        self.assertEqual(a.primary, DOWNSTREAM_LOSS)
        self.assertEqual(a.details["downstream_lost_stage"],
                         "holding_event_resolver")

    def test_a_run_that_lost_nothing_is_complete(self) -> None:
        a = classify(_run())

        self.assertEqual(a.primary, COMPLETE)
        self.assertEqual(a.owner, "none")
        self.assertEqual(a.caveats, ())

    def test_a_graph_added_document_counts_as_served(self) -> None:
        a = classify(_run(served_docs=("d2",), graph_added_docs=("d1",)))

        self.assertEqual(a.primary, COMPLETE)


class PrecedenceTests(unittest.TestCase):
    """28: an earlier loss hides everything downstream of it."""

    def test_a_decline_outranks_every_later_symptom(self) -> None:
        a = classify(_run(retrieval_allowed=False, eligible_docs=(),
                          served_docs=(), sibling_misses=({"x": 1},),
                          downstream_lost_stage="composer"))

        self.assertEqual(a.primary, QUERY_UNDERSTANDING_DECLINE)

    def test_a_filter_loss_is_never_reported_as_a_ranking_loss(self) -> None:
        """The misattribution this evaluator exists to prevent."""

        a = classify(_run(eligible_docs=("d2",), served_docs=("d2",),
                          first_required_rank=None))

        self.assertEqual(a.primary, FILTER_EXCLUSION)
        self.assertNotEqual(a.primary, RANKING_LOW)

    def test_a_ranking_loss_outranks_evidence_and_downstream(self) -> None:
        a = classify(_run(served_docs=("d2",), sibling_misses=({"x": 1},),
                          downstream_lost_stage="composer"))

        self.assertEqual(a.primary, RANKING_LOW)

    def test_a_sibling_miss_outranks_a_generic_coverage_miss(self) -> None:
        a = classify(_run(sibling_misses=({"x": 1},),
                          coverage_misses=({"y": 2},)))

        self.assertEqual(a.primary, TABLE_SIBLING_MISS)

    def test_the_declared_order_matches_the_classifier(self) -> None:
        self.assertEqual(PRECEDENCE[0], QUERY_UNDERSTANDING_DECLINE)
        self.assertEqual(PRECEDENCE[1], FILTER_EXCLUSION)
        self.assertEqual(PRECEDENCE[2], RANKING_LOW)
        self.assertEqual(PRECEDENCE[-1], COMPLETE)
        self.assertEqual(set(PRECEDENCE), set(OWNER))


class EmbeddingConfidenceTests(unittest.TestCase):
    """16: a ranking claim must carry the provider that produced it."""

    def test_hash_ranking_carries_the_verification_caveat(self) -> None:
        a = classify(_run(served_docs=("d2",), embedding_provider="hash"))

        self.assertEqual(a.embedding_confidence, HASH_DIAGNOSTIC)
        self.assertEqual(a.primary, RANKING_LOW)
        self.assertIn(UNVERIFIED_RANKING, a.caveats)

    def test_live_ranking_carries_no_caveat(self) -> None:
        a = classify(_run(served_docs=("d2",), embedding_provider="bge_m3"))

        self.assertEqual(a.embedding_confidence, LIVE_BGE_M3)
        self.assertNotIn(UNVERIFIED_RANKING, a.caveats)

    def test_embedding_independent_findings_carry_no_caveat(self) -> None:
        """Filter and table structure read metadata, not vectors."""

        for observation in (_run(eligible_docs=("d2",), embedding_provider="hash"),
                            _run(sibling_misses=({"x": 1},),
                                 embedding_provider="hash")):
            with self.subTest(case=observation.question_id):
                a = classify(observation)
                self.assertIn(a.primary, {FILTER_EXCLUSION, TABLE_SIBLING_MISS})
                self.assertNotIn(UNVERIFIED_RANKING, a.caveats)

    def test_provider_names_map_to_confidence_labels(self) -> None:
        self.assertEqual(confidence_from_provider("hash"), HASH_DIAGNOSTIC)
        self.assertEqual(confidence_from_provider("bge-m3"), LIVE_BGE_M3)
        self.assertEqual(confidence_from_provider(None), "LEXICAL_ONLY")
        self.assertEqual(confidence_from_provider("clova"), "OTHER")


class ModeTests(unittest.TestCase):
    """19: a structural pass must not claim what it did not measure."""

    def test_a_structural_pass_says_so_instead_of_claiming_success(self) -> None:
        a = classify(_run(ranking_available=False, evidence_available=False,
                          downstream_available=False,
                          embedding_provider=None))

        self.assertEqual(a.mode, "A")
        self.assertEqual(a.primary, COMPLETE)
        self.assertIn("MODE A: ranking and evidence not evaluated", a.caveats)

    def test_a_structural_pass_still_detects_filter_exclusion(self) -> None:
        a = classify(_run(ranking_available=False, eligible_docs=("d2",)))

        self.assertEqual(a.primary, FILTER_EXCLUSION)
        self.assertEqual(a.mode, "A")

    def test_unevaluated_stages_are_named_in_the_caveats(self) -> None:
        a = classify(_run(evidence_available=False, downstream_available=False))

        self.assertEqual(a.primary, COMPLETE)
        self.assertIn("evidence coverage not evaluated", a.caveats)
        self.assertIn("downstream retention not evaluated", a.caveats)


class AggregationTests(unittest.TestCase):
    def test_counts_cover_every_category_including_zeros(self) -> None:
        rows = [classify(_run(question_id="A")),
                classify(_run(question_id="B", retrieval_allowed=False)),
                classify(_run(question_id="C", eligible_docs=("d2",)))]
        summary = aggregate(rows, groups={"A": "holding", "B": "holding",
                                          "C": "periodic"},
                            tasks={"A": "holding_change"},
                            metadata={"commit": "abc123"})

        self.assertEqual(summary["total"], 3)
        self.assertEqual(set(summary["counts"]), set(PRECEDENCE))
        self.assertEqual(summary["counts"][FILTER_EXCLUSION], 1)
        self.assertEqual(summary["counts"][RANKING_LOW], 0)
        self.assertEqual(summary["by_doc_group"]["periodic"][FILTER_EXCLUSION], 1)
        self.assertEqual(summary["metadata"]["commit"], "abc123")

    def test_aggregation_is_order_independent(self) -> None:
        rows = [classify(_run(question_id=q)) for q in ("A", "B", "C")]
        self.assertEqual(aggregate(rows)["counts"],
                         aggregate(list(reversed(rows)))["counts"])


class LeakageTests(unittest.TestCase):
    """15: gold may be read after the fact, never fed forward."""

    def test_the_evaluator_never_imports_the_application(self) -> None:
        from pathlib import Path

        source = Path("scripts/retrieval_failure_evaluator.py").read_text(
            encoding="utf-8")
        body = "".join(source.split('"""')[::2])
        self.assertNotIn("import app", body)
        self.assertNotIn("from app", body)

    def test_no_question_company_or_document_literal_appears(self) -> None:
        from pathlib import Path

        source = Path("scripts/retrieval_failure_evaluator.py").read_text(
            encoding="utf-8")
        body = "".join(source.split('"""')[::2])
        for literal in ("HX0", "HX1", "H0", "국민연금", "이마트", "삼성전자",
                        "holding_20", "periodic_20", "2023-", "2024-"):
            self.assertNotIn(literal, body, f"must not name {literal!r}")

    def test_classification_reads_only_the_observation(self) -> None:
        """Identical observations classify identically, whatever the ids say."""

        one = classify(_run(question_id="HX13", served_docs=("d2",)))
        two = classify(_run(question_id="ZZ99", served_docs=("d2",)))

        self.assertEqual(one.primary, two.primary)
        self.assertEqual(one.caveats, two.caveats)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
