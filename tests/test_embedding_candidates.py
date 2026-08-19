import json
import tempfile
import unittest
from pathlib import Path

from app.reasoning.query_plan import QueryPlan
from app.retrieval.embedding_candidates import (
    Gold60EmbeddingCandidateCollector,
    write_candidate_collection,
)
from app.retrieval.interfaces import CandidateChunk, CandidateDocument, MetadataMatch


class CandidateUnderstanding:
    def __init__(self):
        self.queries = []

    def understand(self, query, *, top_k=10):
        self.queries.append(query)
        return QueryPlan(query=query, company=query, top_k=top_k)


class CandidateBackend:
    def __init__(self):
        self.filters = []

    def get_candidate_documents(self, **filters):
        self.filters.append(filters)
        company = filters["company"][0]
        return [CandidateDocument(f"doc-{company}", {}, MetadataMatch())]

    def get_candidate_chunks(self, documents):
        doc_id = list(documents)[0].doc_id
        company = doc_id.removeprefix("doc-")
        return [
            CandidateChunk(
                "shared",
                doc_id,
                {
                    "retrieval_text": "shared retrieval text",
                    "content": "must not replace retrieval text",
                    "corp_code": "001",
                    "corp_name": company,
                    "doc_group": "major",
                    "chunk_type": "text",
                },
                MetadataMatch(),
            ),
            CandidateChunk(
                f"only-{company}",
                doc_id,
                {
                    "retrieval_text": f"{company} retrieval text",
                    "corp_code": "001",
                    "corp_name": company,
                    "doc_group": "major",
                    "chunk_type": "table",
                },
                MetadataMatch(),
            ),
        ]


class EmbeddingCandidateCollectorTests(unittest.TestCase):
    def test_unions_candidates_without_reading_gold_answer_fields(self) -> None:
        understanding = CandidateUnderstanding()
        backend = CandidateBackend()
        collector = Gold60EmbeddingCandidateCollector(understanding, backend)
        question_sets = {
            "gold": [
                {
                    "question_id": "Q1",
                    "query": "alpha",
                    "doc_id": "gold-doc-must-not-be-used",
                    "target_id": "gold-chunk-must-not-be-used",
                },
                {
                    "question_id": "Q2",
                    "query": "beta",
                    "doc_id": "another-gold-doc",
                    "target_id": "another-gold-chunk",
                },
            ]
        }

        collection = collector.collect(question_sets)

        self.assertEqual(understanding.queries, ["alpha", "beta"])
        self.assertEqual(
            {chunk["chunk_id"] for chunk in collection.chunks},
            {"shared", "only-alpha", "only-beta"},
        )
        self.assertEqual(collection.summary["total_candidate_occurrences"], 4)
        self.assertEqual(collection.summary["unique_candidate_chunks"], 3)
        shared = next(row for row in collection.chunks if row["chunk_id"] == "shared")
        self.assertEqual(shared["candidate_occurrences"], 2)
        self.assertEqual(shared["retrieval_text"], "shared retrieval text")
        self.assertNotIn("gold-doc", json.dumps(collection.summary))
        self.assertNotIn("gold-chunk", json.dumps(collection.summary))

    def test_writes_required_candidate_files(self) -> None:
        collection = Gold60EmbeddingCandidateCollector(
            CandidateUnderstanding(), CandidateBackend()
        ).collect({"gold": [{"question_id": "Q1", "query": "alpha"}]})
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            write_candidate_collection(collection, output_dir)
            for filename in (
                "candidate_chunk_ids.txt",
                "candidate_chunks.jsonl",
                "candidate_summary.json",
                "candidate_summary.md",
            ):
                self.assertTrue((output_dir / filename).exists())
            row = json.loads(
                (output_dir / "candidate_chunks.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()[0]
            )
            for field in (
                "chunk_id",
                "doc_id",
                "corp_code",
                "corp_name",
                "doc_group",
                "chunk_type",
                "retrieval_text",
            ):
                self.assertIn(field, row)

    def test_empty_question_set_is_supported(self) -> None:
        collection = Gold60EmbeddingCandidateCollector(
            CandidateUnderstanding(), CandidateBackend()
        ).collect({"gold": []})
        self.assertEqual(collection.chunks, ())
        self.assertEqual(collection.summary["candidate_count_stats"]["p95"], 0)


if __name__ == "__main__":
    unittest.main()
