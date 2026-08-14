import unittest

from app.retrieval.interfaces import CandidateChunk, MetadataMatch
from app.retrieval.local_backend import LocalBM25Retriever, LocalManifestBackend


class LocalManifestBackendTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = LocalManifestBackend(
            [
                {
                    "doc_id": "d1", "corp_code": "001", "corp_name": "테스트",
                    "listed_name": "TEST", "base_year": 2024, "base_month": 3,
                    "doc_group": "periodic", "doc_subtype": "quarter",
                    "is_correction": False,
                },
                {
                    "doc_id": "d2", "corp_code": "001", "corp_name": "테스트",
                    "listed_name": "TEST", "base_year": 2024, "base_month": 6,
                    "doc_group": "holding", "doc_subtype": "holding",
                    "is_correction": True,
                },
            ]
        )

    def test_company_year_period_are_hard_filters(self) -> None:
        rows = self.backend.get_candidate_documents(
            company="TEST", year=2024, period=3
        )
        self.assertEqual([row.doc_id for row in rows], ["d1"])

    def test_group_and_correction_are_soft_boosts(self) -> None:
        rows = self.backend.get_candidate_documents(
            corp_code="001", doc_group="holding", is_correction=True
        )
        self.assertEqual({row.doc_id for row in rows}, {"d1", "d2"})
        matches = {row.doc_id: row.metadata_match for row in rows}
        self.assertEqual(matches["d1"].soft_score, 0.0)
        self.assertEqual(matches["d2"].soft_score, 2.0)

    def test_bm25_result_schema_is_backend_independent(self) -> None:
        candidates = [
            CandidateChunk(
                chunk_id="c1",
                doc_id="d1",
                chunk={"chunk_id": "c1", "doc_id": "d1", "retrieval_text": "알파 베타"},
                metadata_match=MetadataMatch(soft_boosts={"doc_group": True}, soft_score=1.0),
            ),
            CandidateChunk(
                chunk_id="c2",
                doc_id="d2",
                chunk={"chunk_id": "c2", "doc_id": "d2", "retrieval_text": "감마 델타"},
                metadata_match=MetadataMatch(),
            ),
        ]
        result = LocalBM25Retriever().retrieve("알파", candidates, top_k=1)[0].to_dict()
        self.assertEqual(
            set(result), {"chunk_id", "doc_id", "bm25_score", "rank", "metadata_match"}
        )
        self.assertEqual(result["chunk_id"], "c1")
        self.assertEqual(result["rank"], 1)


if __name__ == "__main__":
    unittest.main()
