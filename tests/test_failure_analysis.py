import copy
import json
import tempfile
import unittest
from pathlib import Path

from app.reasoning.failure_analysis import (
    analyze_gold60_failures,
    write_failure_analysis,
)


def _failure_row(*, source_candidate: bool) -> dict:
    fusion = (
        {
            "chunk_id": "d1:ch_gold",
            "doc_id": "d1",
            "fusion_rank": 6,
            "final_rank": 14,
            "section_path": ["II. 사업의 내용", "정답 섹션"],
        }
        if source_candidate
        else None
    )
    return {
        "question_id": "M01",
        "question": "테스트 회사 권리공모 결과",
        "doc_group": "major",
        "query_plan": {"event_type": "rights_offering"},
        "hard_filters": {"corp_code": ["00123456"]},
        "hit_at_10": False,
        "lexical_raw_gold_rank": 6 if source_candidate else None,
        "lexical_gold_rank": None,
        "vector_gold_rank": None,
        "vector_diagnostic_gold_rank": 25 if not source_candidate else None,
        "hybrid_gold_rank": None,
        "gold_fusion_diagnostic": fusion,
        "gold_relevant_candidate_count": 1 if source_candidate else 0,
        "gold": {
            "doc_id": "d1",
            "target_type": "text",
            "target_id": "s_gold",
            "candidate_relevant_chunk_ids": (
                ["d1:ch_gold"] if source_candidate else []
            ),
            "relevant_chunks": (
                [
                    {
                        "chunk_id": "d1:ch_gold",
                        "doc_id": "d1",
                        "section_path": ["II. 사업의 내용", "정답 섹션"],
                    }
                ]
                if source_candidate
                else []
            ),
        },
        "lexical_gold_matches": {
            "production": (
                [{"chunk_id": "d1:ch_gold", "doc_id": "d1", "rank": 6}]
                if source_candidate
                else []
            ),
            "diagnostic": [],
        },
        "vector_gold_matches": {
            "production": [],
            "diagnostic": (
                [{"chunk_id": "d1:ch_gold", "doc_id": "d1", "rank": 25}]
                if not source_candidate
                else []
            ),
        },
        "hybrid_top10": [
            {
                "rank": 1,
                "chunk_id": "d1:ch_alpha",
                "doc_id": "d1",
                "section_path": ["II. 사업의 내용", "경쟁 섹션"],
                "hybrid": {"final_score": 0.99, "retrieval_score": 0.95},
            },
            {
                "rank": 2,
                "chunk_id": "d2:ch_beta",
                "doc_id": "d2",
                "section_path": ["III. 재무에 관한 사항"],
                "hybrid": {"final_score": 0.98, "retrieval_score": 0.94},
            },
        ],
    }


class Gold60FailureAnalysisTests(unittest.TestCase):
    def test_same_document_chunks_keep_distinct_identity_and_order(self):
        report = {"question_count": 1, "questions": [_failure_row(source_candidate=True)]}
        original = copy.deepcopy(report)

        analysis = analyze_gold60_failures(report)
        failure = analysis["failures"][0]
        competitors = failure["candidate_analysis"]["top_competitors"]

        self.assertEqual(failure["gold"]["chunk_id"], "d1:ch_gold")
        self.assertEqual(failure["gold"]["doc_id"], "d1")
        self.assertEqual(competitors[0]["chunk_id"], "d1:ch_alpha")
        self.assertEqual(competitors[0]["doc_id"], "d1")
        self.assertNotEqual(
            failure["gold"]["chunk_id"], competitors[0]["chunk_id"]
        )
        self.assertEqual(
            [item["chunk_id"] for item in competitors],
            ["d1:ch_alpha", "d2:ch_beta"],
        )
        self.assertIn("section_confusion", failure["contributing_categories"])
        self.assertEqual(report, original)

    def test_gold_in_source_pool_but_outside_top10_is_reranking_failure(self):
        report = {"question_count": 1, "questions": [_failure_row(source_candidate=True)]}

        failure = analyze_gold60_failures(report)["failures"][0]

        self.assertEqual(failure["failure_category"], "reranking_failure")
        self.assertTrue(
            failure["candidate_analysis"]["gold_in_candidate_pool"]
        )
        self.assertEqual(failure["retrieval"]["lexical_rank"], 6)
        self.assertEqual(failure["retrieval"]["fusion_rank"], 6)
        self.assertEqual(failure["retrieval"]["hybrid_rank"], 14)

    def test_gold_absent_from_production_pools_is_candidate_missing(self):
        report = {"question_count": 1, "questions": [_failure_row(source_candidate=False)]}

        failure = analyze_gold60_failures(report)["failures"][0]

        self.assertEqual(failure["failure_category"], "candidate_missing")
        self.assertFalse(
            failure["candidate_analysis"]["gold_in_candidate_pool"]
        )
        self.assertEqual(failure["retrieval"]["vector_diagnostic_rank"], 25)
        self.assertEqual(
            failure["candidate_analysis"]["source_presence"],
            {"lexical": False, "vector": False, "fusion": False},
        )

    def test_writes_json_and_markdown_without_retrieval(self):
        report = {
            "question_count": 2,
            "method": {"rerank_mode": "legacy", "top_k": 10},
            "questions": [
                _failure_row(source_candidate=True),
                {"question_id": "M02", "hit_at_10": True, "hybrid_gold_rank": 1},
            ],
        }
        analysis = analyze_gold60_failures(report)

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            write_failure_analysis(analysis, output_dir)

            json_output = json.loads(
                (output_dir / "failure_analysis.json").read_text(encoding="utf-8")
            )
            markdown = (output_dir / "failure_analysis.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(json_output["summary"]["recall_at_10_misses"], 1)
        self.assertEqual(
            json_output["summary"]["category_counts"], {"reranking_failure": 1}
        )
        self.assertIn("# Gold60 Failure Analysis", markdown)
        self.assertIn("d1:ch_alpha", markdown)


if __name__ == "__main__":
    unittest.main()
