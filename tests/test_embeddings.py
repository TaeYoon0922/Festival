import math
import unittest

from app.retrieval.embeddings import (
    DeterministicHashEmbedder,
    EmbeddingConfig,
    chunk_embedding_text,
)


class EmbeddingTests(unittest.TestCase):
    def test_hash_embedder_is_deterministic_and_normalized(self) -> None:
        config = EmbeddingConfig(model="mock", version="test", dimensions=16)
        embedder = DeterministicHashEmbedder(config)

        first = embedder.embed_query("고려아연 revenue 2024")
        second = embedder.embed_documents(["고려아연 revenue 2024"])[0]

        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in first)), 1.0)

    def test_embedding_config_reads_environment_mapping(self) -> None:
        config = EmbeddingConfig.from_env(
            {
                "FESTIVAL_EMBEDDING_PROVIDER": "test-provider",
                "FESTIVAL_EMBEDDING_MODEL": "test-model",
                "FESTIVAL_EMBEDDING_VERSION": "2026-08",
                "FESTIVAL_EMBEDDING_DIMENSIONS": "8",
            }
        )
        self.assertEqual(config.provider, "test-provider")
        self.assertEqual(config.dimensions, 8)

    def test_document_embedding_uses_only_frozen_retrieval_text(self) -> None:
        chunk = {"retrieval_text": "indexed text", "content": "different content"}
        self.assertEqual(chunk_embedding_text(chunk), "indexed text")
        with self.assertRaises(ValueError):
            chunk_embedding_text({"content": "must not be used as fallback"})


if __name__ == "__main__":
    unittest.main()
