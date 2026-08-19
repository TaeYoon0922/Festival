import json
import tempfile
import unittest
from pathlib import Path

from app.reasoning.hybrid_evaluation import (
    QueryPlanHybridEvaluator,
    write_hybrid_evaluation_report,
)
from app.reasoning.query_plan import QueryPlan
from app.retrieval.embeddings import DeterministicHashEmbedder, EmbeddingConfig
from app.retrieval.hybrid import HybridQueryExecutor
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)
from app.retrieval.vector import VectorRetrievalResult


class StubUnderstanding:
    def understand(self, query, *, top_k=10):
        return QueryPlan(
            query=query,
            task_type="major_event",
            event_type="rights_offering",
            top_k=top_k,
        )


class EvaluationHybridBackend:
    def __init__(self):
        self.documents = [
            CandidateDocument("d1", {"doc_group": "major"}, MetadataMatch())
        ]
        self.chunks = [
            CandidateChunk(
                "noise",
                "d1",
                {"chunk_id": "noise", "section_id": "other", "content": "noise"},
                MetadataMatch(),
            ),
            CandidateChunk(
                "gold",
                "d1",
                {
                    "chunk_id": "gold",
                    "section_id": "s1",
                    "content": "gold evidence rights offering",
                    "retrieval_text": "gold evidence rights offering",
                },
                MetadataMatch(),
            ),
        ]

    def get_candidate_documents(self, **_filters):
        return self.documents

    def get_candidate_chunks(self, _documents):
        return self.chunks

    def retrieve(self, _query, candidates, *, top_k=None):
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        return [
            RetrievalResult(
                "noise", "d1", 1.0, 1, by_id["noise"].metadata_match.to_dict()
            )
        ]

    def vector_search(self, _embedding, candidates, **_kwargs):
        candidate_ids = {candidate.chunk_id for candidate in candidates}
        assert "gold" in candidate_ids
        return [VectorRetrievalResult("gold", "d1", 0.9, 1)]


class QueryPlanHybridEvaluatorTests(unittest.TestCase):
    def setUp(self):
        config = EmbeddingConfig(model="mock", version="v1", dimensions=8)
        backend = EvaluationHybridBackend()
        executor = HybridQueryExecutor(
            backend,
            DeterministicHashEmbedder(config),
            config,
        )
        self.evaluator = QueryPlanHybridEvaluator(
            StubUnderstanding(), executor, top_k=10
        )
        self.question_sets = {
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

    def test_compares_same_run_lexical_and_hybrid_and_records_debug(self):
        report = self.evaluator.evaluate(self.question_sets)

        self.assertEqual(report["hybrid"]["overall"]["recall_at_1"], 1.0)
        self.assertEqual(report["lexical_only"]["overall"]["recall_at_10"], 0.0)
        self.assertEqual(report["improvement"]["overall"]["recall_at_1"], 1.0)
        self.assertEqual(report["failure_counts"], {"success": 1})
        row = report["questions"][0]
        self.assertIsNone(row["lexical_gold_rank"])
        self.assertEqual(row["vector_gold_rank"], 1)
        self.assertEqual(row["hybrid_gold_rank"], 1)
        self.assertIn("rrf_score", row["hybrid_top10"][0]["hybrid"])

    def test_writes_json_markdown_and_csv(self):
        report = self.evaluator.evaluate(self.question_sets)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            write_hybrid_evaluation_report(report, output_dir)
            json_path = output_dir / "postgres_hybrid_gold60.json"
            markdown_path = output_dir / "postgres_hybrid_gold60.md"
            csv_path = output_dir / "postgres_hybrid_questions.csv"
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(csv_path.exists())
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8"))["question_count"],
                1,
            )
            self.assertIn(
                "Overall comparison", markdown_path.read_text(encoding="utf-8")
            )

    def test_rejects_top_k_below_recall_cutoff(self):
        with self.assertRaises(ValueError):
            QueryPlanHybridEvaluator(StubUnderstanding(), object(), top_k=5)


if __name__ == "__main__":
    unittest.main()
