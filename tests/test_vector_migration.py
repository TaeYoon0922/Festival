import unittest
from pathlib import Path


class VectorMigrationTests(unittest.TestCase):
    def test_embeddings_are_separate_and_hnsw_uses_cosine(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        migration = (project_root / "db" / "004_vector_search.sql").read_text(
            encoding="utf-8"
        )
        self.assertIn("CREATE TABLE IF NOT EXISTS chunk_embeddings", migration)
        self.assertIn("REFERENCES chunks(chunk_id)", migration)
        self.assertIn("embedding_model text NOT NULL", migration)
        self.assertIn("embedding_version text NOT NULL", migration)
        self.assertIn("USING hnsw", migration)
        self.assertIn("vector_cosine_ops", migration)
        self.assertNotIn("ALTER TABLE chunks", migration)


if __name__ == "__main__":
    unittest.main()
