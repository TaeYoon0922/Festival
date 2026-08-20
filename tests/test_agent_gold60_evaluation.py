import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.agent.gold60_evaluation import (
    AgentGold60Evaluator,
    analyze_agent_gold60_failures,
    write_agent_gold60_evaluation,
)
from app.retrieval.embeddings import DeterministicHashEmbedder, EmbeddingConfig
from app.retrieval.hybrid import HybridQueryExecutor
from scripts.evaluate_postgres_agent_gold60 import load_question_sets
from tests.test_postgres_hybrid_evaluation import (
    EvaluationHybridBackend,
    StubUnderstanding,
)


class AgentGold60EvaluationTests(unittest.TestCase):
    def setUp(self):
        config = EmbeddingConfig(model="mock", version="v1", dimensions=8)
        backend = EvaluationHybridBackend()
        backend.chunks[1].chunk["source_refs"] = [
            {"table_id": "t1", "row_start": 1, "row_end": 1}
        ]
        executor = HybridQueryExecutor(
            backend,
            DeterministicHashEmbedder(config),
            config,
        )
        self.evaluator = AgentGold60Evaluator(
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
                    "evidence_terms": ["gold", "evidence", "rights offering"],
                },
            )
        }

    def test_reuses_hybrid_execution_and_saves_answer_sources_and_chunks(self):
        report = self.evaluator.evaluate(self.question_sets)
        row = report["questions"][0]

        self.assertEqual(report["hybrid"]["overall"]["recall_at_10"], 1.0)
        self.assertEqual(report["agent"]["overall"]["answerable_rate"], 1.0)
        self.assertTrue(row["end_to_end_success"])
        self.assertEqual(row["end_to_end_failure_class"], "success")
        self.assertEqual(row["retrieved_chunks"][0]["chunk_id"], "gold")
        self.assertEqual(row["retrieved_chunks"][0]["content"], "gold evidence rights offering")
        self.assertTrue(row["retrieved_chunks"][0]["is_gold_relevant"])
        self.assertEqual(row["agent"]["status"], "ok")
        self.assertTrue(row["agent"]["answerable"])
        self.assertTrue(row["agent"]["answer_draft_preserved"])
        self.assertEqual(row["source_references"][0]["chunk_id"], "gold")
        self.assertEqual(
            row["source_references"][0]["source_refs"][0]["table_id"], "t1"
        )
        self.assertTrue(row["source_references"][0]["provenance_path"])
        self.assertTrue(row["answer_gold_comparison"]["gold_doc_cited"])
        self.assertTrue(row["answer_gold_comparison"]["gold_chunk_cited"])
        self.assertTrue(
            row["answer_gold_comparison"]["all_evidence_terms_present"]
        )

    def test_failure_analysis_separates_answer_stage_failure(self):
        report = self.evaluator.evaluate(self.question_sets)
        changed = copy.deepcopy(report)
        row = changed["questions"][0]
        row["answer_gold_comparison"]["all_evidence_terms_present"] = False
        row["answer_gold_comparison"]["missing_evidence_terms"] = ["missing"]
        row["end_to_end_failure_class"] = "gold_evidence_terms_missing"
        row["end_to_end_success"] = False

        analysis = analyze_agent_gold60_failures(changed)

        self.assertEqual(analysis["summary"]["end_to_end_failures"], 1)
        self.assertEqual(
            analysis["summary"]["category_counts"],
            {"gold_evidence_terms_missing": 1},
        )
        self.assertIn("missing", analysis["failures"][0]["reason"])
        self.assertEqual(
            analysis["summary"]["retrieval_recall_at_10_misses"], 0
        )

    def test_writer_emits_evaluation_jsonl_markdown_and_failure_reports(self):
        report = self.evaluator.evaluate(self.question_sets)
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            analysis = write_agent_gold60_evaluation(report, output_dir)

            evaluation = json.loads(
                (output_dir / "gold60_agent_evaluation.json").read_text(
                    encoding="utf-8"
                )
            )
            question_lines = (
                output_dir / "gold60_agent_questions.jsonl"
            ).read_text(encoding="utf-8").splitlines()

            self.assertEqual(evaluation["question_count"], 1)
            self.assertEqual(len(question_lines), 1)
            self.assertTrue((output_dir / "gold60_agent_evaluation.md").is_file())
            self.assertTrue((output_dir / "failure_analysis.json").is_file())
            self.assertTrue((output_dir / "failure_analysis.md").is_file())
            self.assertEqual(analysis["summary"]["end_to_end_failures"], 0)

    def test_custom_question_list_input_is_validated(self):
        question = dict(self.question_sets["gold_40"][0])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "questions.json"
            path.write_text(
                json.dumps([question], ensure_ascii=False), encoding="utf-8"
            )

            loaded = load_question_sets(path)

        self.assertEqual(tuple(loaded), ("gold_60",))
        self.assertEqual(loaded["gold_60"][0]["question_id"], "M01")


if __name__ == "__main__":
    unittest.main()
