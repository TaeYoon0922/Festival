import math
import unittest

from app.retrieval.embeddings import (
    DeterministicHashEmbedder,
    EmbeddingConfig,
    HttpEmbeddingSettings,
    OpenAICompatibleEmbeddingProvider,
    chunk_embedding_text,
    create_embedding_provider,
)


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post_json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


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

    def test_openai_compatible_adapter_uses_injected_transport(self) -> None:
        config = EmbeddingConfig(
            provider="openai_compatible",
            model="semantic-model",
            version="2026-08",
            dimensions=2,
        )
        transport = FakeTransport(
            {
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            }
        )
        provider = OpenAICompatibleEmbeddingProvider(
            config,
            HttpEmbeddingSettings("https://embedding.invalid/v1", "secret"),
            transport=transport,
        )
        vectors = provider.embed_documents(["first", "second"])

        self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
        url, call = transport.calls[0]
        self.assertEqual(url, "https://embedding.invalid/v1")
        self.assertEqual(call["payload"], {"model": "semantic-model", "input": ["first", "second"]})
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret")
        self.assertNotIn("secret", str(call["payload"]))

    def test_provider_factory_reads_secret_only_from_environment(self) -> None:
        config = EmbeddingConfig(
            provider="http", model="model", version="v1", dimensions=2
        )
        provider = create_embedding_provider(
            config,
            environment={
                "FESTIVAL_EMBEDDING_API_URL": "https://embedding.invalid/v1",
                "FESTIVAL_EMBEDDING_API_KEY": "runtime-secret",
                "FESTIVAL_EMBEDDING_API_KEY_HEADER": "api-key",
                "FESTIVAL_EMBEDDING_API_KEY_PREFIX": "",
            },
            transport=FakeTransport({"data": []}),
        )
        self.assertEqual(provider.settings.request_headers(), {"api-key": "runtime-secret"})
        self.assertNotIn("runtime-secret", repr(provider.settings))

    def test_http_adapter_rejects_dimension_mismatch(self) -> None:
        config = EmbeddingConfig(
            provider="http", model="model", version="v1", dimensions=3
        )
        provider = OpenAICompatibleEmbeddingProvider(
            config,
            HttpEmbeddingSettings("https://embedding.invalid/v1", "secret"),
            transport=FakeTransport({"data": [{"embedding": [1.0, 0.0]}]}),
        )
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            provider.embed_query("query")


if __name__ == "__main__":
    unittest.main()
