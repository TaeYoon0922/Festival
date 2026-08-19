import unittest

from app.reasoning.query_plan import QueryPlan
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
    def __init__(self, *, vector_mode: str = "ok") -> None:
        self.vector_mode = vector_mode
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


if __name__ == "__main__":
    unittest.main()
