import unittest

from app.reasoning.query_plan import QueryPlan
from app.reasoning.router import QueryRouter
from app.retrieval.embeddings import DeterministicHashEmbedder, EmbeddingConfig
from app.retrieval.hybrid import (
    HybridQueryExecutor,
    HybridRetrievalConfig,
    RRFConfig,
    reciprocal_rank_fusion,
)
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)
from app.retrieval.vector import VectorRetrievalResult


def lexical(chunk_id: str, rank: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(chunk_id, "d1", score, rank, {})


def vector(chunk_id: str, rank: int, score: float = 0.8) -> VectorRetrievalResult:
    return VectorRetrievalResult(chunk_id, "d1", score, rank)


class RRFTests(unittest.TestCase):
    def test_merges_lexical_only_vector_only_common_and_duplicates(self) -> None:
        fused = reciprocal_rank_fusion(
            [lexical("common", 1), lexical("lex-only", 2), lexical("common", 4)],
            [vector("vector-only", 1), vector("common", 2), vector("common", 5)],
        )
        by_id = {item.chunk_id: item for item in fused}

        self.assertEqual(set(by_id), {"common", "lex-only", "vector-only"})
        self.assertEqual(by_id["common"].lexical_rank, 1)
        self.assertEqual(by_id["common"].vector_rank, 2)
        self.assertAlmostEqual(by_id["common"].rrf_score, 1 / 61 + 1 / 62)
        self.assertIsNone(by_id["lex-only"].vector_rank)
        self.assertIsNone(by_id["vector-only"].lexical_rank)
        self.assertEqual(fused[0].chunk_id, "common")

    def test_weights_are_configurable(self) -> None:
        fused = reciprocal_rank_fusion(
            [lexical("lex", 1)],
            [vector("vec", 1)],
            RRFConfig(k=60, lexical_weight=2.0, vector_weight=0.5),
        )
        self.assertEqual([item.chunk_id for item in fused], ["lex", "vec"])
        self.assertAlmostEqual(fused[0].rrf_score, 2.0 / 61)


class FakeHybridBackend:
    def __init__(self, *, vector_mode: str = "ok", lexical_mode: str = "ok") -> None:
        self.vector_mode = vector_mode
        self.lexical_mode = lexical_mode
        self.document_filters = None
        self.vector_candidate_ids = None
        self.documents = [
            CandidateDocument(
                "d1",
                {"base_year": 2024, "doc_group": "periodic"},
                MetadataMatch(),
            )
        ]
        self.chunks = [
            CandidateChunk(
                "lex",
                "d1",
                {"chunk_id": "lex", "retrieval_text": "noise", "content": "noise"},
                MetadataMatch(),
            ),
            CandidateChunk(
                "vec",
                "d1",
                {
                    "chunk_id": "vec",
                    "retrieval_text": "revenue evidence",
                    "content": "revenue evidence",
                    "section_path": ["revenue"],
                    "retrieval_priority": "high",
                },
                MetadataMatch(),
            ),
        ]

    def get_candidate_documents(self, **filters):
        self.document_filters = filters
        return self.documents

    def get_candidate_chunks(self, documents):
        self.asserted_document_ids = [document.doc_id for document in documents]
        return self.chunks

    def retrieve(self, _query, candidates, *, top_k=None):
        if self.lexical_mode == "empty":
            return []
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        return [
            RetrievalResult(
                "lex", "d1", 1.0, 1, by_id["lex"].metadata_match.to_dict()
            )
        ]

    def vector_search(
        self,
        _query_embedding,
        candidates,
        *,
        embedding_model,
        embedding_version,
        top_k=50,
    ):
        self.vector_candidate_ids = [candidate.chunk_id for candidate in candidates]
        self.embedding_identity = (embedding_model, embedding_version)
        if self.vector_mode == "error":
            raise RuntimeError("pgvector unavailable")
        if self.vector_mode == "empty":
            return []
        return [VectorRetrievalResult("vec", "d1", 0.9, 1)]

    def existing_embedding_chunk_ids(self, chunk_ids, **_identity):
        return {"vec"}.intersection(chunk_ids)


class ControlledDeterministicRouter(QueryRouter):
    def deterministic_components(
        self,
        _route,
        *,
        chunk,
        metadata_match,
        document_metadata=None,
    ):
        score = float(chunk.get("fixture_deterministic_score", 0.0))
        return {
            "exact_term": score,
            "section": score,
            "period_relevance": score,
            "basis_relevance": score,
            "metadata": score,
            "retrieval_priority": score,
            "date_relevance": score,
        }


class SingleSourcePreservationBackend:
    def __init__(self, lexical_ids, vector_ids, gold_id):
        all_ids = tuple(dict.fromkeys((*lexical_ids, *vector_ids)))
        self.documents = [CandidateDocument("d1", {}, MetadataMatch())]
        self.chunks = [
            CandidateChunk(
                chunk_id,
                "d1",
                {
                    "chunk_id": chunk_id,
                    "content": chunk_id,
                    "retrieval_text": chunk_id,
                    "fixture_deterministic_score": (
                        0.0 if chunk_id == gold_id else 1.0
                    ),
                },
                MetadataMatch(),
            )
            for chunk_id in all_ids
        ]
        self.lexical_ids = lexical_ids
        self.vector_ids = vector_ids
        self.lexical_top_k = None
        self.vector_top_k = None

    def get_candidate_documents(self, **_filters):
        return self.documents

    def get_candidate_chunks(self, _documents):
        return self.chunks

    def retrieve(self, _query, _candidates, *, top_k=None):
        self.lexical_top_k = top_k
        results = [
            RetrievalResult(chunk_id, "d1", 1.0 / rank, rank, {})
            for rank, chunk_id in enumerate(self.lexical_ids, start=1)
        ]
        return results[:top_k] if top_k is not None else results

    def vector_search(self, _embedding, _candidates, **_kwargs):
        self.vector_top_k = _kwargs.get("top_k")
        results = [
            VectorRetrievalResult(chunk_id, "d1", 1.0 / rank, rank)
            for rank, chunk_id in enumerate(self.vector_ids, start=1)
        ]
        return results[: self.vector_top_k] if self.vector_top_k is not None else results

    def existing_embedding_chunk_ids(self, chunk_ids, **_identity):
        return set(chunk_ids)


class HybridExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.embedding_config = EmbeddingConfig(
            model="mock", version="v1", dimensions=8
        )

    def executor(self, backend, **config_overrides):
        return HybridQueryExecutor(
            backend,
            DeterministicHashEmbedder(self.embedding_config),
            self.embedding_config,
            config=HybridRetrievalConfig(**config_overrides),
        )

    def test_reuses_metadata_scope_and_combines_deterministic_reranking(self) -> None:
        backend = FakeHybridBackend()
        plan = QueryPlan(
            query="revenue",
            company="Test Corp",
            years=(2024,),
            task_type="financial_metric",
            metric="revenue",
            section_boosts={"revenue": 1.0},
            top_k=10,
        )

        execution = self.executor(backend).execute(plan)

        self.assertEqual(backend.document_filters["company"], ["Test Corp"])
        self.assertEqual(backend.document_filters["year"], [2024])
        self.assertEqual(backend.vector_candidate_ids, ["lex", "vec"])
        self.assertEqual(execution.vector_coverage["embedded_count"], 1)
        self.assertEqual(execution.vector_coverage["ratio"], 0.5)
        self.assertEqual(execution.results[0].chunk_id, "vec")
        debug = execution.results[0].metadata_match["hybrid"]
        for key in (
            "lexical_rank",
            "lexical_score",
            "vector_rank",
            "vector_score",
            "rrf_score",
            "deterministic_rerank_score",
            "final_score",
            "final_rank",
        ):
            self.assertIn(key, debug)

    def test_empty_vector_result_preserves_lexical_rerank(self) -> None:
        backend = FakeHybridBackend(vector_mode="empty")
        execution = self.executor(backend).execute(QueryPlan(query="revenue", top_k=10))

        self.assertEqual(execution.vector_status, "empty")
        self.assertEqual([item.chunk_id for item in execution.results], ["lex"])
        self.assertEqual(
            execution.results[0].metadata_match["hybrid"]["fallback"],
            "lexical_only",
        )

    def test_vector_unavailable_preserves_lexical_rerank(self) -> None:
        backend = FakeHybridBackend(vector_mode="error")
        execution = self.executor(backend).execute(QueryPlan(query="revenue", top_k=10))

        self.assertEqual(execution.vector_status, "unavailable")
        self.assertIn("pgvector unavailable", execution.vector_error)
        self.assertEqual([item.chunk_id for item in execution.results], ["lex"])

    def test_zero_embedding_coverage_skips_vector_call(self) -> None:
        backend = FakeHybridBackend()
        backend.existing_embedding_chunk_ids = lambda *_args, **_kwargs: set()
        execution = self.executor(backend).execute(QueryPlan(query="revenue", top_k=10))
        self.assertEqual(execution.vector_status, "no_coverage")
        self.assertEqual(execution.vector_coverage["embedded_count"], 0)
        self.assertIsNone(backend.vector_candidate_ids)
        self.assertEqual([item.chunk_id for item in execution.results], ["lex"])

    def test_empty_lexical_and_vector_keep_filtered_candidate_chunks(self) -> None:
        backend = FakeHybridBackend(vector_mode="empty", lexical_mode="empty")
        backend.existing_embedding_chunk_ids = lambda *_args, **_kwargs: set()
        execution = self.executor(backend).execute(
            QueryPlan(
                query="시설투자 금액 자기자본 대비 현금성 자산 조달",
                top_k=10,
            )
        )

        self.assertTrue(execution.results)
        self.assertEqual(
            {item.chunk_id for item in execution.results},
            {"lex", "vec"},
        )
        self.assertEqual(
            execution.results[0].metadata_match["hybrid"]["fallback"],
            "filtered_candidates",
        )

    def test_empty_lexical_still_uses_vector_hits(self) -> None:
        backend = FakeHybridBackend(lexical_mode="empty")
        execution = self.executor(backend).execute(QueryPlan(query="revenue", top_k=10))
        self.assertEqual([item.chunk_id for item in execution.results], ["vec"])
        self.assertNotEqual(
            execution.results[0].metadata_match.get("hybrid", {}).get("fallback"),
            "filtered_candidates",
        )

    def test_balance_sheet_metric_rescue_adds_statement_chunk(self) -> None:
        backend = FakeHybridBackend(vector_mode="empty")
        backend.chunks.append(
            CandidateChunk(
                "balance-sheet",
                "d1",
                {
                    "chunk_id": "balance-sheet",
                    "doc_group": "periodic",
                    "section_path": ["(첨부)연 결 재 무 제 표"],
                    "statement_scope": "consolidated",
                    "content": (
                        "| 열 1 | 제 56 (당) 기 | 제 55 (전) 기 |\n"
                        "| --- | --- | --- |\n"
                        "| 자 산 총 계 | 514,531,948 | 455,905,980 |"
                    ),
                    "retrieval_text": (
                        "| 열 1 | 제 56 (당) 기 | 제 55 (전) 기 |\n"
                        "| --- | --- | --- |\n"
                        "| 자 산 총 계 | 514,531,948 | 455,905,980 |"
                    ),
                },
                MetadataMatch(),
            )
        )
        execution = self.executor(backend).execute(
            QueryPlan(
                query="자산총계 자산 총계 자산 총 계 자 산 총 계",
                task_type="financial_metric",
                metric="자산총계",
                disclosure_route=("periodic",),
                basis="consolidated",
                years=(2024,),
                section_boosts={"첨부연결재무제표": 1.0, "재무상태표": 1.0},
                top_k=10,
            )
        )

        self.assertEqual(execution.results[0].chunk_id, "balance-sheet")
        self.assertEqual(
            execution.results[0].metadata_match["hybrid"]["fallback"],
            "balance_sheet_metric_rescue",
        )

    def test_vector_error_can_be_strict(self) -> None:
        backend = FakeHybridBackend(vector_mode="error")
        executor = self.executor(backend, fallback_on_vector_error=False)
        with self.assertRaises(RuntimeError):
            executor.execute(QueryPlan(query="revenue", top_k=10))

    def test_embedder_and_index_configs_must_match(self) -> None:
        with self.assertRaises(ValueError):
            HybridQueryExecutor(
                FakeHybridBackend(),
                DeterministicHashEmbedder(self.embedding_config),
                EmbeddingConfig(model="other", version="v1", dimensions=8),
            )

    def test_vector_rank_nine_survives_bounded_deterministic_rerank(self) -> None:
        common = [f"common-{rank:02d}" for rank in range(1, 9)]
        lexical_ids = [*common, *[f"lex-{rank:02d}" for rank in range(10, 22)]]
        vector_ids = [
            *common,
            "gold-vector",
            *[f"vec-{rank:02d}" for rank in range(10, 22)],
        ]
        backend = SingleSourcePreservationBackend(
            lexical_ids, vector_ids, "gold-vector"
        )
        execution = HybridQueryExecutor(
            backend,
            DeterministicHashEmbedder(self.embedding_config),
            self.embedding_config,
            router=ControlledDeterministicRouter(),
            config=HybridRetrievalConfig(rerank_mode="bounded"),
        ).execute(QueryPlan(query="listing date", top_k=10))

        diagnostic = next(
            item
            for item in execution.rerank_diagnostics
            if item["chunk_id"] == "gold-vector"
        )
        self.assertEqual(diagnostic["vector_rank"], 9)
        self.assertIsNone(diagnostic["lexical_rank"])
        self.assertGreater(diagnostic["legacy_final_rank"], 10)
        self.assertEqual(diagnostic["preservation_rank"], 9)
        self.assertLessEqual(diagnostic["final_rank"], 10)
        self.assertEqual(diagnostic["final_rank"], diagnostic["bounded_final_rank"])
        self.assertEqual(diagnostic["rerank_mode"], "bounded")
        self.assertEqual(len(diagnostic["weight_grid"]), 4)
        self.assertIn("gold-vector", [result.chunk_id for result in execution.results])

    def test_lexical_rank_six_survives_bounded_deterministic_rerank(self) -> None:
        common = [f"common-{rank:02d}" for rank in range(1, 6)]
        lexical_ids = [
            *common,
            "gold-lexical",
            *[f"lex-{rank:02d}" for rank in range(7, 20)],
        ]
        vector_ids = [*common, *[f"vec-{rank:02d}" for rank in range(7, 20)]]
        backend = SingleSourcePreservationBackend(
            lexical_ids, vector_ids, "gold-lexical"
        )
        execution = HybridQueryExecutor(
            backend,
            DeterministicHashEmbedder(self.embedding_config),
            self.embedding_config,
            router=ControlledDeterministicRouter(),
            config=HybridRetrievalConfig(rerank_mode="bounded"),
        ).execute(QueryPlan(query="merger effective date", top_k=10))

        diagnostic = next(
            item
            for item in execution.rerank_diagnostics
            if item["chunk_id"] == "gold-lexical"
        )
        self.assertEqual(diagnostic["lexical_rank"], 6)
        self.assertIsNone(diagnostic["vector_rank"])
        self.assertGreater(diagnostic["legacy_final_rank"], 10)
        self.assertEqual(diagnostic["preservation_rank"], 6)
        self.assertLessEqual(diagnostic["final_rank"], 10)
        self.assertIn("gold-lexical", [result.chunk_id for result in execution.results])

    def test_legacy_is_default_and_bounded_rank_remains_diagnostic(self) -> None:
        common = [f"common-{rank:02d}" for rank in range(1, 9)]
        lexical_ids = [*common, *[f"lex-{rank:02d}" for rank in range(10, 22)]]
        vector_ids = [
            *common,
            "gold-vector",
            *[f"vec-{rank:02d}" for rank in range(10, 22)],
        ]
        backend = SingleSourcePreservationBackend(
            lexical_ids, vector_ids, "gold-vector"
        )
        execution = HybridQueryExecutor(
            backend,
            DeterministicHashEmbedder(self.embedding_config),
            self.embedding_config,
            router=ControlledDeterministicRouter(),
        ).execute(QueryPlan(query="listing date", top_k=10))

        diagnostic = next(
            item
            for item in execution.rerank_diagnostics
            if item["chunk_id"] == "gold-vector"
        )
        self.assertEqual(execution.routing["hybrid"]["config"]["rerank_mode"], "legacy")
        self.assertEqual(diagnostic["rerank_mode"], "legacy")
        self.assertEqual(diagnostic["final_rank"], diagnostic["legacy_final_rank"])
        self.assertGreater(diagnostic["final_rank"], 10)
        self.assertLessEqual(diagnostic["bounded_final_rank"], 10)
        self.assertNotIn(
            "gold-vector", [result.chunk_id for result in execution.results]
        )

    def test_rejects_unknown_rerank_mode(self) -> None:
        with self.assertRaises(ValueError):
            HybridRetrievalConfig(rerank_mode="experimental")

    def test_diagnostic_top_n_does_not_expand_production_fusion_lists(self) -> None:
        ids = [f"chunk-{rank}" for rank in range(1, 7)]
        backend = SingleSourcePreservationBackend(ids, ids, ids[-1])
        execution = HybridQueryExecutor(
            backend,
            DeterministicHashEmbedder(self.embedding_config),
            self.embedding_config,
            router=ControlledDeterministicRouter(),
            config=HybridRetrievalConfig(
                lexical_top_n=2,
                vector_top_n=2,
                diagnostic_top_n=5,
            ),
        ).execute(QueryPlan(query="diagnostic", top_k=10))

        self.assertEqual(backend.lexical_top_k, 5)
        self.assertEqual(backend.vector_top_k, 5)
        self.assertEqual(len(execution.diagnostic_lexical_results), 5)
        self.assertEqual(len(execution.diagnostic_vector_results), 5)
        self.assertEqual(len(execution.lexical_results), 2)
        self.assertEqual(len(execution.vector_results), 2)
        self.assertEqual(len(execution.fused_candidates), 2)

    def test_rejects_non_positive_diagnostic_top_n(self) -> None:
        with self.assertRaises(ValueError):
            HybridRetrievalConfig(diagnostic_top_n=0)


if __name__ == "__main__":
    unittest.main()
