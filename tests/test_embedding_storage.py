import unittest

from app.retrieval.embedding_storage import estimate_embedding_storage
from app.retrieval.vector_index import generate_hnsw_index_sql


class EmbeddingStorageTests(unittest.TestCase):
    def test_estimator_scales_subset_and_full_corpus(self) -> None:
        estimate = estimate_embedding_storage(
            100,
            768,
            full_corpus_count=1000,
            storage_capacity_gb=100,
        )
        subset = estimate["candidate_subset"]
        full = estimate["full_corpus"]
        self.assertEqual(subset["raw_vector_bytes"], 100 * 768 * 4)
        self.assertEqual(full["raw_vector_bytes"], 1000 * 768 * 4)
        self.assertGreater(full["hnsw_index_bytes"], full["raw_vector_bytes"])
        self.assertIn("planning estimate", estimate["disclaimer"])
        self.assertTrue(estimate["capacity"]["requires_existing_db_and_free_space_check"])

    def test_empty_candidate_subset_is_supported(self) -> None:
        estimate = estimate_embedding_storage(0, 1536, full_corpus_count=10)
        self.assertEqual(estimate["candidate_subset"]["embedding_layer_total_bytes"], 0)

    def test_dimension_specific_hnsw_sql_is_generated_but_not_executed(self) -> None:
        sql = generate_hnsw_index_sql(768)
        self.assertIn("CREATE INDEX CONCURRENTLY", sql)
        self.assertIn("vector(768)", sql)
        self.assertIn("embedding_dimensions = 768", sql)
        self.assertIn("vector_cosine_ops", sql)
        with self.assertRaises(ValueError):
            generate_hnsw_index_sql(0)


if __name__ == "__main__":
    unittest.main()
