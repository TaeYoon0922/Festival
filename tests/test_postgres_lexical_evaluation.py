import json
import tempfile
import unittest
from pathlib import Path

from app.reasoning import QueryExecutor, QueryPlan
from app.reasoning.lexical_evaluation import (
    QueryPlanLexicalEvaluator,
    load_baseline_reports,
    write_evaluation_report,
)
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


class StubUnderstanding:
    def understand(self, query, *, top_k=10):
        return QueryPlan(
            query=query,
            raw_query=query,
            task_type="holding_change" if "holding" in query else "disclosure_lookup",
            top_k=top_k,
        )


class EvaluationBackend:
    def __init__(self):
        self.documents = [
            CandidateDocument("d1", {"doc_group": "periodic"}, MetadataMatch()),
            CandidateDocument("d2", {"doc_group": "holding"}, MetadataMatch()),
        ]
        self.chunks = [
            CandidateChunk(
                "gold-table",
                "d1",
                {
                    "doc_id": "d1",
                    "table_id": "t1",
                    "section_id": "s1",
                    "content": "gold table evidence",
                },
                MetadataMatch(),
            ),
            CandidateChunk(
                "gold-text",
                "d2",
                {
                    "doc_id": "d2",
                    "section_id": "s2",
                    "content": "holding gold evidence",
                },
                MetadataMatch(),
            ),
            CandidateChunk(
                "noise",
                "d1",
                {"doc_id": "d1", "section_id": "noise", "content": "unrelated"},
                MetadataMatch(),
            ),
        ]

    def get_candidate_documents(self, **_filters):
        return self.documents

    def get_candidate_chunks(self, _documents):
        return self.chunks

    def retrieve(self, query, candidates, *, top_k=None):
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        ordered = ["noise"] if "holding" in query else ["gold-table", "noise"]
        return [
            RetrievalResult(
                chunk_id,
                by_id[chunk_id].doc_id,
                1.0 - index * 0.1,
                index + 1,
                by_id[chunk_id].metadata_match.to_dict(),
            )
            for index, chunk_id in enumerate(ordered[:top_k])
        ]


class QueryPlanLexicalEvaluatorTests(unittest.TestCase):
    def setUp(self):
        backend = EvaluationBackend()
        self.evaluator = QueryPlanLexicalEvaluator(
            StubUnderstanding(), QueryExecutor(backend), top_k=10
        )
        self.question_sets = {
            "gold_40": (
                {
                    "question_id": "P01",
                    "doc_group": "periodic",
                    "query": "successful query",
                    "doc_id": "d1",
                    "target_type": "table",
                    "target_id": "t1",
                    "evidence_terms": ["gold", "evidence"],
                },
            ),
            "holding_20": (
                {
                    "question_id": "HX01",
                    "doc_group": "holding",
                    "query": "holding failure",
                    "doc_id": "d2",
                    "target_type": "text",
                    "target_id": "s2",
                    "evidence_terms": ["holding", "evidence"],
                },
            ),
        }

    def test_evaluates_recall_groups_and_failure_diagnostics(self):
        progress = []
        report = self.evaluator.evaluate(
            self.question_sets,
            baselines={
                "gold_40": {
                    "question_count": 1,
                    "recall_at_1": 0.5,
                    "recall_at_5": 0.5,
                    "recall_at_10": 0.5,
                },
                "holding_20": {
                    "question_count": 1,
                    "recall_at_1": 0.0,
                    "recall_at_5": 0.0,
                    "recall_at_10": 0.0,
                },
            },
            progress=lambda index, total, question_id: progress.append(
                (index, total, question_id)
            ),
        )

        self.assertEqual(report["question_count"], 2)
        self.assertEqual(report["overall"]["recall_at_1"], 0.5)
        self.assertEqual(report["overall"]["recall_at_5"], 0.5)
        self.assertEqual(report["by_evaluation_set"]["gold_40"]["recall_at_10"], 1.0)
        self.assertIn("holding_change", report["by_task_type"])
        self.assertEqual(report["by_doc_group"]["holding"]["failure_count"], 1)
        self.assertEqual(progress, [(1, 2, "P01"), (2, 2, "HX01")])

        failure = report["failures"][0]
        self.assertEqual(failure["failure_class"], "lexical_ranking_failure")
        for field in (
            "question",
            "query_plan",
            "lexical_query",
            "hard_filters",
            "soft_boosts",
            "gold",
            "retrieved_top10",
            "gold_rank",
        ):
            self.assertIn(field, failure)
        self.assertIsNone(failure["gold_rank"])
        self.assertEqual(failure["gold_relevant_candidate_count"], 1)
        self.assertEqual(report["baseline_comparison"]["gold_40"]["recall_at_1"], 0.5)

    def test_writes_json_markdown_and_csv_reports(self):
        report = self.evaluator.evaluate(self.question_sets)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            write_evaluation_report(report, output_dir)
            json_path = output_dir / "postgres_lexical_gold60.json"
            markdown_path = output_dir / "postgres_lexical_gold60.md"
            csv_path = output_dir / "postgres_lexical_questions.csv"
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertTrue(csv_path.exists())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["question_count"], 2)
            self.assertIn("HX01", markdown_path.read_text(encoding="utf-8"))

    def test_loads_structural_and_direct_baseline_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "bm25_fixed_40.json").write_text(
                json.dumps(
                    {
                        "overall": {
                            "structural": {
                                "question_count": 40,
                                "recall_at_1": 0.1,
                                "recall_at_5": 0.5,
                                "recall_at_10": 0.7,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (directory / "bm25_holding_20.json").write_text(
                json.dumps(
                    {
                        "overall": {
                            "question_count": 20,
                            "recall_at_1": 0.2,
                            "recall_at_5": 0.4,
                            "recall_at_10": 0.6,
                        }
                    }
                ),
                encoding="utf-8",
            )
            baselines = load_baseline_reports(directory)
            self.assertEqual(baselines["gold_40"]["recall_at_5"], 0.5)
            self.assertEqual(baselines["holding_20"]["question_count"], 20)

    def test_rejects_top_k_below_recall_cutoff(self):
        with self.assertRaises(ValueError):
            QueryPlanLexicalEvaluator(StubUnderstanding(), object(), top_k=5)


if __name__ == "__main__":
    unittest.main()
