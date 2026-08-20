import json
import tempfile
import unittest
from pathlib import Path

from app.reasoning.hybrid_evaluation import (
    QueryPlanHybridEvaluator,
    _holding_structure,
    _query_alignment,
    _query_evidence_profile,
    compare_hybrid_evaluation_reports,
    write_hybrid_evaluation_report,
)
from app.reasoning.query_plan import QueryPlan
from app.retrieval.embeddings import DeterministicHashEmbedder, EmbeddingConfig
from app.retrieval.hybrid import HybridQueryExecutor, HybridRetrievalConfig
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
                {
                    "chunk_id": "noise",
                    "section_id": "other",
                    "section_path": ["III. 재무에 관한 사항"],
                    "content": "noise",
                },
                MetadataMatch(),
            ),
            CandidateChunk(
                "gold",
                "d1",
                {
                    "chunk_id": "gold",
                    "section_id": "s1",
                    "section_path": ["II. 사업의 내용", "1. 사업의 개요"],
                    "chunk_type": "text",
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

    def existing_embedding_chunk_ids(self, chunk_ids, **_identity):
        return {"gold"}.intersection(chunk_ids)


class DeepVectorGoldBackend:
    def __init__(self, *, gold_rank: int, result_count: int) -> None:
        self.gold_rank = gold_rank
        self.documents = [
            CandidateDocument("d1", {"doc_group": "periodic"}, MetadataMatch())
        ]
        self.vector_ids = [
            "gold" if rank == gold_rank else f"noise-{rank:03d}"
            for rank in range(1, result_count + 1)
        ]
        self.chunks = [
            CandidateChunk(
                chunk_id,
                "d1",
                {
                    "chunk_id": chunk_id,
                    "section_id": "s1" if chunk_id == "gold" else "noise",
                    "section_path": (
                        ["II. 사업의 내용", "1. 사업의 개요"]
                        if chunk_id == "gold"
                        else ["III. 재무에 관한 사항"]
                    ),
                    "chunk_type": "text",
                    "content": (
                        "gold evidence rights offering"
                        if chunk_id == "gold"
                        else chunk_id
                    ),
                    "retrieval_text": chunk_id,
                },
                MetadataMatch(),
            )
            for chunk_id in self.vector_ids
        ]

    def get_candidate_documents(self, **_filters):
        return self.documents

    def get_candidate_chunks(self, _documents):
        return self.chunks

    def retrieve(self, _query, _candidates, *, top_k=None):
        return []

    def vector_search(self, _embedding, _candidates, **kwargs):
        results = [
            VectorRetrievalResult(chunk_id, "d1", 1.0 / rank, rank)
            for rank, chunk_id in enumerate(self.vector_ids, start=1)
        ]
        return results[: kwargs["top_k"]]

    def existing_embedding_chunk_ids(self, chunk_ids, **_identity):
        return set(chunk_ids)


class SameDocumentChunkIdentityBackend:
    def __init__(self) -> None:
        self.documents = [CandidateDocument("d1", {}, MetadataMatch())]
        self.chunks = [
            CandidateChunk(
                "d1:ch_alpha",
                "d1",
                {
                    "chunk_id": "payload-must-not-override-alpha",
                    "section_id": "other",
                    "section_path": ["III. 재무에 관한 사항"],
                    "content": "unrelated alpha",
                    "retrieval_text": "unrelated alpha",
                    "chunk_type": "text",
                },
                MetadataMatch(),
            ),
            CandidateChunk(
                "d1:ch_gold",
                "d1",
                {
                    "chunk_id": "payload-must-not-override-gold",
                    "section_id": "s1",
                    "section_path": ["II. 사업의 내용", "1. 사업의 개요"],
                    "content": "gold evidence rights offering",
                    "retrieval_text": "gold evidence rights offering",
                    "chunk_type": "text",
                },
                MetadataMatch(),
            ),
        ]

    def get_candidate_documents(self, **_filters):
        return self.documents

    def get_candidate_chunks(self, _documents):
        return self.chunks

    def retrieve(self, _query, _candidates, *, top_k=None):
        return [
            RetrievalResult("d1:ch_alpha", "d1", 1.0, 1, {}),
            RetrievalResult("d1:ch_gold", "d1", 0.9, 2, {}),
        ][:top_k]

    def vector_search(self, _embedding, _candidates, **kwargs):
        return [
            VectorRetrievalResult("d1:ch_gold", "d1", 0.95, 1),
            VectorRetrievalResult("d1:ch_alpha", "d1", 0.90, 2),
        ][: kwargs["top_k"]]

    def existing_embedding_chunk_ids(self, chunk_ids, **_identity):
        return set(chunk_ids)


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

        self.assertEqual(report["method"]["rerank_mode"], "legacy")
        self.assertEqual(report["hybrid"]["overall"]["recall_at_1"], 1.0)
        self.assertEqual(report["lexical_only"]["overall"]["recall_at_10"], 0.0)
        self.assertEqual(report["improvement"]["overall"]["recall_at_1"], 1.0)
        self.assertEqual(report["failure_counts"], {"success": 1})
        self.assertTrue(report["vector_coverage"]["available"])
        self.assertEqual(report["vector_coverage"]["unique_candidate_count"], 2)
        self.assertEqual(report["vector_coverage"]["embedded_unique_candidate_count"], 1)
        self.assertEqual(report["vector_coverage"]["ratio"], 0.5)
        row = report["questions"][0]
        self.assertIsNone(row["lexical_gold_rank"])
        self.assertEqual(row["vector_gold_rank"], 1)
        self.assertEqual(row["hybrid_gold_rank"], 1)
        self.assertTrue(row["has_any_vector_candidate"])
        self.assertIn("rrf_score", row["hybrid_top10"][0]["hybrid"])
        diagnostic = row["gold_fusion_diagnostic"]
        self.assertEqual(diagnostic["vector_rank"], 1)
        self.assertIn("normalized_rrf_score", diagnostic)
        self.assertIn("deterministic_rerank_score", diagnostic)
        self.assertIn("legacy_final_rank", diagnostic)
        self.assertIn("bounded_final_rank", diagnostic)
        self.assertIn("final_rank", diagnostic)
        self.assertEqual(diagnostic["rerank_mode"], "legacy")
        self.assertEqual(len(diagnostic["weight_grid"]), 4)
        self.assertEqual(row["lexical_diagnostic_gold_rank"], None)
        self.assertEqual(row["vector_diagnostic_gold_rank"], 1)
        gold_chunk = row["gold"]["relevant_chunks"][0]
        self.assertEqual(gold_chunk["chunk_id"], "gold")
        self.assertEqual(gold_chunk["chunk_type"], "text")
        self.assertEqual(
            gold_chunk["section_path"], ["II. 사업의 내용", "1. 사업의 개요"]
        )
        self.assertLessEqual(len(gold_chunk["retrieval_text_preview"]), 500)
        section_debug = row["section_path_diagnostic"]
        self.assertEqual(
            section_debug["gold_section_paths"],
            ["II. 사업의 내용 > 1. 사업의 개요"],
        )
        self.assertEqual(
            section_debug["vector_top10"][0]["section_path"],
            "II. 사업의 내용 > 1. 사업의 개요",
        )
        self.assertTrue(
            section_debug["dominant_gap"]["vector_top10"]["matches_gold_section"]
        )
        self.assertEqual(
            row["query_evidence_diagnostic"]["core_terms"],
            ["rights", "offering"],
        )
        self.assertEqual(
            diagnostic["query_alignment"]["core_term_coverage_ratio"], 1.0
        )
        self.assertIn("score_diagnostic", row["hybrid_top10"][0])
        component_comparison = row["score_component_comparison"]
        self.assertIn("exact_term", component_comparison["components"])
        self.assertIn("final_score", component_comparison["scores"])

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

    def test_compares_baseline_metrics_and_question_ranks_generically(self):
        current = self.evaluator.evaluate(self.question_sets)
        baseline = json.loads(json.dumps(current))
        baseline["hybrid"]["overall"]["recall_at_1"] = 0.0
        baseline["questions"][0]["hybrid_gold_rank"] = None

        comparison = compare_hybrid_evaluation_reports(baseline, current)

        self.assertEqual(comparison["metrics"]["overall"]["delta"]["recall_at_1"], 1.0)
        self.assertEqual(comparison["recovered"], ["M01"])
        self.assertEqual(comparison["new_failures"], [])
        rank_change = comparison["question_rank_changes"][0]
        self.assertEqual(rank_change["hybrid_rank"], {"before": None, "after": 1})

    def test_vector_rank_uses_production_top_n_not_implicit_top10(self):
        for gold_rank, expected_production_rank in ((25, 25), (107, None)):
            with self.subTest(gold_rank=gold_rank):
                config = EmbeddingConfig(model="mock", version="v1", dimensions=8)
                backend = DeepVectorGoldBackend(gold_rank=gold_rank, result_count=120)
                executor = HybridQueryExecutor(
                    backend,
                    DeterministicHashEmbedder(config),
                    config,
                    config=HybridRetrievalConfig(
                        vector_top_n=50,
                        diagnostic_top_n=120,
                    ),
                )
                evaluator = QueryPlanHybridEvaluator(
                    StubUnderstanding(), executor, top_k=10
                )

                row = evaluator.evaluate(self.question_sets)["questions"][0]

                self.assertEqual(row["vector_gold_rank"], expected_production_rank)
                self.assertIsNone(row["vector_top10_gold_rank"])
                self.assertEqual(row["vector_diagnostic_gold_rank"], gold_rank)
                production_matches = row["vector_gold_matches"]["production"]
                diagnostic_matches = row["vector_gold_matches"]["diagnostic"]
                self.assertEqual(bool(production_matches), gold_rank <= 50)
                self.assertEqual(diagnostic_matches[0]["rank"], gold_rank)
                self.assertEqual(diagnostic_matches[0]["chunk_id"], "gold")
                self.assertEqual(len(row["vector_production_top_n"]), 50)
                rank_debug = row["vector_rank_diagnostic"]
                self.assertEqual(
                    rank_debug["production_contains_gold"], gold_rank <= 50
                )
                self.assertEqual(
                    rank_debug["reason"],
                    (
                        "consistent_in_production_top_n"
                        if gold_rank <= 50
                        else "outside_production_top_n"
                    ),
                )
                self.assertTrue(rank_debug["same_query_filter_execution"])
                self.assertFalse(rank_debug["section_boost_applied_to_raw_vector"])
                self.assertEqual(rank_debug["rank_level"], "chunk")

    def test_whitespace_normalized_phrase_and_mixed_product_term_alignment(self):
        holding_profile = _query_evidence_profile(
            QueryPlan(
                query="국민연금기금 변동 후 주식수",
                task_type="holding_change",
            )
        )
        holding_alignment = _query_alignment(
            holding_profile,
            {"retrieval_text": "국민연금기금 변동후 주식수 1,000주"},
        )
        product_profile = _query_evidence_profile(
            QueryPlan(query="TC BONDER 반도체 제조 장비")
        )
        product_alignment = _query_alignment(
            product_profile,
            {"retrieval_text": "반도체 제조용 장비인 TC-BONDER 제품"},
        )
        unrelated_alignment = _query_alignment(
            product_profile,
            {"retrieval_text": "임직원 보수와 이사회 현황"},
        )

        self.assertFalse(holding_alignment["exact_phrase_match"])
        self.assertTrue(holding_alignment["normalized_phrase_match"])
        self.assertEqual(holding_alignment["core_term_coverage_ratio"], 1.0)
        self.assertGreaterEqual(product_alignment["core_term_coverage_ratio"], 0.75)
        self.assertEqual(unrelated_alignment["core_term_coverage_ratio"], 0.0)

    def test_holding_projection_field_alignment_uses_existing_metadata(self):
        profile = _query_evidence_profile(
            QueryPlan(
                query="국민연금기금 변동일 감소 후 주식수",
                task_type="holding_change",
                reporter="국민연금기금",
            )
        )
        structure = _holding_structure(
            {
                "chunk_type": "table_projection",
                "projection_type": "holding_detail_row",
                "projection_state": "resolved",
                "projection_fields": {
                    "보고자/보유자": "국민연금기금",
                    "기준일/보고일": "2023년 06월 13일",
                    "직전 보유주식수": "2,485,201",
                    "증감주식수": "-283,151",
                    "보유주식수": "2,202,050",
                },
                "projection_field_refs": {
                    "보유주식수": [{"table_id": "t0019", "row_start": 2}]
                },
                "source_table_ids": ["t0019"],
                "source_refs": [{"table_id": "t0019", "row_start": 2}],
            },
            profile,
        )

        self.assertTrue(structure["is_projection"])
        self.assertEqual(structure["projection_type"], "holding_detail_row")
        self.assertEqual(structure["change_direction"], "decrease")
        self.assertEqual(structure["after_shares"], "2,202,050")
        self.assertEqual(structure["field_alignment_ratio"], 1.0)

    def test_diagnostics_keep_unique_chunk_ids_separate_from_shared_doc_id(self):
        config = EmbeddingConfig(model="mock", version="v1", dimensions=8)
        backend = SameDocumentChunkIdentityBackend()
        executor = HybridQueryExecutor(
            backend,
            DeterministicHashEmbedder(config),
            config,
        )
        evaluator = QueryPlanHybridEvaluator(
            StubUnderstanding(), executor, top_k=10
        )

        execution = executor.execute(
            StubUnderstanding().understand("rights offering", top_k=10)
        )
        production_order = [result.chunk_id for result in execution.results]
        report = evaluator.evaluate(self.question_sets)
        row = report["questions"][0]

        self.assertEqual(production_order, ["d1:ch_gold", "d1:ch_alpha"])
        self.assertEqual(report["hybrid"]["overall"]["recall_at_1"], 1.0)
        self.assertEqual(
            row["gold"]["candidate_relevant_chunk_ids"], ["d1:ch_gold"]
        )
        self.assertEqual(
            row["gold"]["relevant_chunks"][0]["chunk_id"], "d1:ch_gold"
        )
        self.assertEqual(row["gold"]["relevant_chunks"][0]["doc_id"], "d1")
        self.assertEqual(row["gold_fusion_diagnostic"]["chunk_id"], "d1:ch_gold")
        self.assertEqual(row["gold_fusion_diagnostic"]["doc_id"], "d1")
        for field in (
            "lexical_top10",
            "vector_top10",
            "vector_production_top_n",
            "hybrid_top10",
        ):
            identities = {
                (item["chunk_id"], item["doc_id"]) for item in row[field]
            }
            self.assertEqual(
                identities,
                {("d1:ch_alpha", "d1"), ("d1:ch_gold", "d1")},
            )
        self.assertEqual(
            row["lexical_gold_matches"]["production"][0]["chunk_id"],
            "d1:ch_gold",
        )
        self.assertEqual(
            row["lexical_gold_matches"]["diagnostic"][0]["chunk_id"],
            "d1:ch_gold",
        )
        self.assertEqual(
            row["vector_gold_matches"]["production"][0]["chunk_id"],
            "d1:ch_gold",
        )
        self.assertEqual(
            row["vector_gold_matches"]["diagnostic"][0]["chunk_id"],
            "d1:ch_gold",
        )

    def test_rejects_top_k_below_recall_cutoff(self):
        with self.assertRaises(ValueError):
            QueryPlanHybridEvaluator(StubUnderstanding(), object(), top_k=5)


if __name__ == "__main__":
    unittest.main()
