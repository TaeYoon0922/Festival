"""Vector availability: production stays up and says so, evaluation refuses.

The failure this pins down was measured, not hypothesised.  With 53.6% of
candidates carrying vectors under the exact configured identity, retrieval still
reported ``vector_status == "ok"`` while 91.2% of served chunks came from the
embedded subset -- because only an embedded chunk can be scored by both lanes.
One question fell from rank 2 to 5 and another lost its gold document entirely.
Partial coverage is therefore the dangerous state: zero coverage at least
announces itself.
"""

from __future__ import annotations

import unittest

from app.api.pipeline import AnswerPipeline
from app.api.schemas import AnswerResponse
from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer
from app.reasoning.hybrid_evaluation import QueryPlanHybridEvaluator
from app.reasoning.vector_coverage_policy import (
    HEALTHY,
    PARTIAL_COVERAGE,
    ZERO_COVERAGE,
    VectorCoverageError,
    classify,
)
from app.retrieval.embeddings import DeterministicHashEmbedder, EmbeddingConfig
from app.retrieval.hybrid import HybridQueryExecutor, HybridRetrievalConfig
from scripts.bge_eval_preflight import strict_vector_executor
from tests.test_agent_end_to_end_smoke import _StaticExecutor, _StaticUnderstanding
from tests.test_answer_api import QUESTION, QUESTION_ID, _plan_and_execution
from tests.test_postgres_hybrid_evaluation import (
    EvaluationHybridBackend,
    StubUnderstanding,
)

TOP_LEVEL_KEYS = {
    "question_id",
    "question",
    "retrieved_context",
    "think_trace",
    "answer",
}
BGE = "bge_m3_local"


def _coverage(candidates, embedded, **extra):
    return {
        "available": True,
        "candidate_count": candidates,
        "embedded_count": embedded,
        "ratio": round(embedded / candidates, 6) if candidates else 0.0,
        **extra,
    }


def _answer(*, vector_status="ok", coverage=None, provider=BGE, vector_error=None):
    """A served answer whose retrieval reported the given availability."""

    plan, execution = _plan_and_execution()
    execution.vector_status = vector_status
    execution.vector_coverage = coverage if coverage is not None else _coverage(4, 4)
    execution.vector_error = vector_error
    execution.routing = {
        "hybrid": {
            "vector_status": vector_status,
            "vector_error": vector_error,
            "coverage": execution.vector_coverage,
            "embedding": {
                "provider": provider,
                "model": "BAAI/bge-m3",
                "version": "6892b95f",
                "dimensions": 1024,
            },
        }
    }
    pipeline = AnswerPipeline(
        understanding=_StaticUnderstanding(QUESTION, plan),
        executor=_StaticExecutor(plan, execution),
        verbalizer=HcxVerbalizer(HcxSettings(enabled=False)),
    )
    return pipeline.answer(QUESTION_ID, QUESTION)


def _degradations(payload):
    return [
        warning
        for warning in payload["think_trace"]["warnings"]
        if warning.startswith("vector_")
    ]


class ProductionObservabilityTests(unittest.TestCase):
    """Production must keep answering; what changes is that it says how."""

    def test_full_coverage_stays_silent(self) -> None:
        payload = _answer(coverage=_coverage(4, 4))
        self.assertEqual(_degradations(payload), [])
        self.assertTrue(payload["answer"])

    def test_partial_coverage_is_named_with_its_counts(self) -> None:
        payload = _answer(coverage=_coverage(123, 61))
        self.assertEqual(
            _degradations(payload),
            [
                "vector_coverage_partial:provider=bge_m3_local,"
                "candidates=123,embedded=61,ratio=0.495935"
            ],
        )
        # Serving is unchanged: a degraded request still answers.
        self.assertTrue(payload["answer"])
        self.assertTrue(payload["retrieved_context"])

    def test_partial_coverage_is_not_reported_as_healthy(self) -> None:
        """The core contract: a healthy status plus incomplete coverage is not healthy."""

        self.assertEqual(classify("ok", _coverage(123, 61)), PARTIAL_COVERAGE)
        self.assertNotEqual(classify("ok", _coverage(123, 61)), HEALTHY)

    def test_zero_coverage_is_distinct_from_partial(self) -> None:
        payload = _answer(vector_status="no_coverage", coverage=_coverage(123, 0))
        self.assertEqual(
            _degradations(payload),
            [
                "vector_coverage_absent:provider=bge_m3_local,"
                "candidates=123,embedded=0,ratio=0"
            ],
        )
        self.assertTrue(payload["answer"])

    def test_empty_vector_result_is_not_called_zero_coverage(self) -> None:
        """Vectors exist but matched nothing -- a different fact entirely."""

        payload = _answer(vector_status="empty", coverage=_coverage(123, 123))
        self.assertEqual(
            _degradations(payload),
            ["vector_results_empty:provider=bge_m3_local"],
        )

    def test_vector_error_reports_its_type_but_never_its_message(self) -> None:
        """An exception message can carry a DSN; the type cannot."""

        payload = _answer(
            vector_status="unavailable",
            coverage=_coverage(123, 123),
            vector_error=(
                "OperationalError: connection to host=10.0.0.4 "
                "user=festival password=hunter2 failed"
            ),
        )
        self.assertEqual(
            _degradations(payload),
            ["vector_unavailable:provider=bge_m3_local,error=OperationalError"],
        )
        serialized = str(payload["think_trace"])
        for secret in ("hunter2", "10.0.0.4", "password"):
            self.assertNotIn(secret, serialized)

    def test_hash_provider_is_never_called_a_degraded_bge_run(self) -> None:
        healthy = _answer(provider="hash", coverage=_coverage(9, 9))
        self.assertEqual(_degradations(healthy), [])
        # A hash run with incomplete coverage is still asymmetric, so it is
        # reported -- under its own identity, never as a BGE failure.
        degraded = _answer(provider="hash", coverage=_coverage(9, 4))
        self.assertIn("provider=hash", degraded["think_trace"]["warnings"][-1])
        for payload in (healthy, degraded):
            self.assertNotIn("bge", str(payload["think_trace"]).casefold())

    def test_retriever_without_coverage_introspection_is_not_degraded(self) -> None:
        """A backend that cannot report coverage has not failed at anything."""

        payload = _answer(
            coverage={
                "available": False,
                "candidate_count": 4,
                "embedded_count": None,
                "ratio": None,
            }
        )
        self.assertEqual(_degradations(payload), [])

    def test_failed_coverage_lookup_is_reported_as_unknown(self) -> None:
        payload = _answer(
            coverage={
                "available": False,
                "candidate_count": 4,
                "embedded_count": None,
                "ratio": None,
                "error": "OperationalError: host=10.0.0.4 password=hunter2",
            },
        )
        self.assertEqual(
            _degradations(payload),
            ["vector_coverage_unknown:provider=bge_m3_local,error=OperationalError"],
        )
        self.assertNotIn("hunter2", str(payload["think_trace"]))


class ApiContractTests(unittest.TestCase):
    def test_degradation_adds_no_top_level_or_trace_key(self) -> None:
        healthy = _answer(coverage=_coverage(4, 4))
        degraded = _answer(coverage=_coverage(123, 61))

        self.assertEqual(set(degraded), TOP_LEVEL_KEYS)
        self.assertEqual(set(healthy), set(degraded))
        # The degradation lives inside the existing warnings list, so the trace
        # gains no field and stays identical everywhere else.
        self.assertEqual(set(healthy["think_trace"]), set(degraded["think_trace"]))
        self.assertEqual(
            {k: v for k, v in degraded["think_trace"].items() if k != "warnings"},
            {k: v for k, v in healthy["think_trace"].items() if k != "warnings"},
        )
        AnswerResponse.model_validate(degraded)

    def test_ranking_is_untouched_by_observability(self) -> None:
        healthy = _answer(coverage=_coverage(4, 4))
        for coverage, status in (
            (_coverage(123, 61), "ok"),
            (_coverage(123, 0), "no_coverage"),
            (_coverage(123, 123), "empty"),
        ):
            degraded = _answer(vector_status=status, coverage=coverage)
            self.assertEqual(
                [item["chunk_id"] for item in degraded["retrieved_context"]],
                [item["chunk_id"] for item in healthy["retrieved_context"]],
            )
            self.assertEqual(degraded["retrieved_context"], healthy["retrieved_context"])
            self.assertEqual(degraded["answer"], healthy["answer"])


def _evaluator(*, provider, embedded):
    """An evaluator whose backend covers only ``embedded`` of its candidates."""

    config = EmbeddingConfig(provider=provider, model="mock", version="v1", dimensions=8)
    backend = EvaluationHybridBackend()
    backend.existing_embedding_chunk_ids = (
        lambda chunk_ids, **_identity: set(list(chunk_ids)[:embedded])
    )
    executor = HybridQueryExecutor(backend, DeterministicHashEmbedder(config), config)
    return QueryPlanHybridEvaluator(StubUnderstanding(), executor, top_k=10)


QUESTION_SETS = {
    "gold_40": (
        {
            "question_id": "M01",
            "doc_group": "major",
            "query": "rights offering",
            "doc_id": "d1",
            "target_type": "text",
            "target_id": "s1",
            "evidence_terms": ["gold", "evidence"],
        },
    )
}


class EvaluationStrictnessTests(unittest.TestCase):
    """A run claiming real vectors may not publish numbers it did not earn."""

    def test_full_coverage_produces_metrics(self) -> None:
        report = _evaluator(provider=BGE, embedded=2).evaluate(QUESTION_SETS)
        self.assertEqual(report["question_count"], 1)
        self.assertIn("hybrid", report)

    def test_partial_coverage_fails_before_any_metric(self) -> None:
        with self.assertRaises(VectorCoverageError) as caught:
            _evaluator(provider=BGE, embedded=1).evaluate(QUESTION_SETS)
        message = str(caught.exception)
        self.assertIn("METRICS WERE NOT PRODUCED", message)
        self.assertIn("expected candidate vectors: 2", message)
        self.assertIn("stored matching vectors:    1", message)
        self.assertIn("missing:                    1", message)
        self.assertIn(BGE, message)

    def test_zero_coverage_fails_closed(self) -> None:
        with self.assertRaises(VectorCoverageError) as caught:
            _evaluator(provider=BGE, embedded=0).evaluate(QUESTION_SETS)
        message = str(caught.exception)
        self.assertIn("lexical-only", message)
        self.assertIn("M01", message)

    def test_hash_diagnostics_are_not_blocked(self) -> None:
        """An intentional hash run is a diagnostic, not a failed BGE run."""

        for embedded in (0, 1, 2):
            report = _evaluator(provider="hash", embedded=embedded).evaluate(
                QUESTION_SETS
            )
            self.assertEqual(report["question_count"], 1)

    def test_strict_executor_stops_a_bypassing_evaluator(self) -> None:
        """Scripts that compute metrics without the shared evaluator."""

        config = EmbeddingConfig(provider=BGE, model="mock", version="v1", dimensions=8)
        backend = EvaluationHybridBackend()
        backend.existing_embedding_chunk_ids = (
            lambda chunk_ids, **_identity: set(list(chunk_ids)[:1])
        )
        executor = strict_vector_executor(
            HybridQueryExecutor(backend, DeterministicHashEmbedder(config), config)
        )
        plan = StubUnderstanding().understand("rights offering")
        with self.assertRaises(VectorCoverageError):
            executor.execute(plan)
        # Delegation is transparent for everything else.
        self.assertIs(executor.embedding_config, config)

    def test_strict_executor_passes_hash_and_complete_coverage(self) -> None:
        for provider, embedded in ((BGE, 2), ("hash", 2), ("hash", 1)):
            config = EmbeddingConfig(
                provider=provider, model="mock", version="v1", dimensions=8
            )
            backend = EvaluationHybridBackend()
            backend.existing_embedding_chunk_ids = (
                lambda chunk_ids, _n=embedded, **_i: set(list(chunk_ids)[:_n])
            )
            executor = strict_vector_executor(
                HybridQueryExecutor(
                    backend, DeterministicHashEmbedder(config), config
                )
            )
            plan = StubUnderstanding().understand("rights offering")
            self.assertTrue(executor.execute(plan).results)


class VectorErrorPolicyTests(unittest.TestCase):
    """The existing fallback switch keeps its exact behaviour."""

    def _executor(self, *, fallback):
        config = EmbeddingConfig(provider=BGE, model="mock", version="v1", dimensions=8)

        class Failing(DeterministicHashEmbedder):
            def embed_query(self, text):
                raise RuntimeError("embedding service down")

        return HybridQueryExecutor(
            EvaluationHybridBackend(),
            Failing(config),
            config,
            config=HybridRetrievalConfig(fallback_on_vector_error=fallback),
        )

    def test_fallback_true_serves_and_is_reported(self) -> None:
        plan = StubUnderstanding().understand("rights offering")
        execution = self._executor(fallback=True).execute(plan)
        self.assertEqual(execution.vector_status, "unavailable")
        self.assertTrue(execution.results)
        self.assertEqual(
            classify(execution.vector_status, execution.vector_coverage),
            "vector_unavailable",
        )

    def test_fallback_false_still_raises(self) -> None:
        plan = StubUnderstanding().understand("rights offering")
        with self.assertRaisesRegex(RuntimeError, "embedding service down"):
            self._executor(fallback=False).execute(plan)


class ClassificationTests(unittest.TestCase):
    def test_each_condition_maps_to_its_own_class(self) -> None:
        self.assertEqual(classify("ok", _coverage(10, 10)), HEALTHY)
        self.assertEqual(classify("ok", _coverage(10, 3)), PARTIAL_COVERAGE)
        self.assertEqual(classify("no_coverage", _coverage(10, 0)), ZERO_COVERAGE)
        self.assertEqual(classify("empty", _coverage(10, 10)), "empty_vector_result")
        self.assertEqual(classify("unavailable", _coverage(10, 10)), "vector_unavailable")
        # The filtered-candidate fallback replaces "ok", so coverage still rules.
        self.assertEqual(
            classify("filtered_candidates", _coverage(10, 3)), PARTIAL_COVERAGE
        )


if __name__ == "__main__":
    unittest.main()
