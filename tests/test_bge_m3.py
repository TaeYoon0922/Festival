import math
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.retrieval.bge_m3 import (
    BGE_M3_DIMENSIONS,
    BgeM3HttpEmbeddingProvider,
    BgeM3LocalEmbeddingProvider,
    _load_flag_embedding_encoder,
)
from app.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingDimensionMismatch,
    HttpEmbeddingSettings,
    create_embedding_provider,
)


class FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, sentences, **kwargs):
        self.calls.append((list(sentences), kwargs))
        vector = [0.0] * BGE_M3_DIMENSIONS
        vector[0] = 3.0
        vector[1] = 4.0
        return {"dense_vecs": [list(vector) for _ in sentences]}


class FakeTransport:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post_json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def bge_config(**overrides):
    values = {
        "provider": "bge_m3_local",
        "model": "BAAI/bge-m3",
        "version": "pinned-revision",
        "dimensions": 1024,
        "batch_size": 4,
        "max_length": 8192,
        "device": "cpu",
    }
    values.update(overrides)
    return EmbeddingConfig(**values)


class BgeM3ProviderTests(unittest.TestCase):
    def test_config_reads_max_length_and_device(self) -> None:
        config = EmbeddingConfig.from_env(
            {
                "FESTIVAL_EMBEDDING_PROVIDER": "bge_m3_local",
                "FESTIVAL_EMBEDDING_MODEL": "BAAI/bge-m3",
                "FESTIVAL_EMBEDDING_VERSION": "revision",
                "FESTIVAL_EMBEDDING_DIMENSIONS": "1024",
                "FESTIVAL_EMBEDDING_BATCH_SIZE": "4",
                "FESTIVAL_EMBEDDING_MAX_LENGTH": "8192",
                "FESTIVAL_EMBEDDING_DEVICE": "cuda",
            }
        )
        self.assertEqual(config.max_length, 8192)
        self.assertEqual(config.device, "cuda")

    def test_local_query_and_documents_use_same_dense_normalized_path(self) -> None:
        encoder = FakeEncoder()
        provider = BgeM3LocalEmbeddingProvider(bge_config(), encoder=encoder)

        query = provider.embed_query("query")
        documents = provider.embed_documents(["doc one", "doc two"])

        self.assertEqual(len(query), 1024)
        self.assertAlmostEqual(query[0], 0.6)
        self.assertAlmostEqual(query[1], 0.8)
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in query)), 1.0)
        self.assertEqual(documents[0], query)
        for _texts, kwargs in encoder.calls:
            self.assertEqual(kwargs["batch_size"], 4)
            self.assertEqual(kwargs["max_length"], 8192)
            self.assertTrue(kwargs["return_dense"])
            self.assertFalse(kwargs["return_sparse"])
            self.assertFalse(kwargs["return_colbert_vecs"])

    def test_local_provider_uses_injected_factory_without_model_download(self) -> None:
        created = []

        def factory(config):
            created.append((config.model, config.device))
            return FakeEncoder()

        provider = create_embedding_provider(
            bge_config(device="cuda"), bge_encoder_factory=factory
        )
        self.assertEqual(created, [])
        provider.load()
        self.assertEqual(created, [("BAAI/bge-m3", "cuda")])

    def test_bge_provider_rejects_non_1024_dimension(self) -> None:
        with self.assertRaises(EmbeddingDimensionMismatch):
            BgeM3LocalEmbeddingProvider(
                bge_config(dimensions=768), encoder=FakeEncoder()
            )
        with self.assertRaisesRegex(ValueError, "max_length"):
            BgeM3LocalEmbeddingProvider(
                bge_config(max_length=8193), encoder=FakeEncoder()
            )

    def test_local_loader_propagates_device_normalization_and_revision(self) -> None:
        calls = []

        def model_factory(model, **kwargs):
            calls.append((model, kwargs))
            return FakeEncoder()

        fake_module = SimpleNamespace(BGEM3FlagModel=model_factory)
        with patch.dict(sys.modules, {"FlagEmbedding": fake_module}):
            _load_flag_embedding_encoder(bge_config(device="cuda"))
        model, kwargs = calls[0]
        self.assertEqual(model, "BAAI/bge-m3")
        self.assertEqual(kwargs["devices"], "cuda")
        self.assertTrue(kwargs["use_fp16"])
        self.assertTrue(kwargs["normalize_embeddings"])
        self.assertEqual(kwargs["revision"], "pinned-revision")

    def test_http_provider_uses_dense_only_contract_and_normalizes(self) -> None:
        vector = [0.0] * 1024
        vector[2] = 2.0
        transport = FakeTransport({"dense_vecs": [vector]})
        config = bge_config(provider="bge_m3_http", device="cpu")
        provider = BgeM3HttpEmbeddingProvider(
            config,
            HttpEmbeddingSettings("https://embedding.invalid/bge-m3", "secret"),
            transport=transport,
        )

        result = provider.embed_query("query")

        self.assertEqual(result[2], 1.0)
        _url, call = transport.calls[0]
        payload = call["payload"]
        self.assertEqual(payload["texts"], ["query"])
        self.assertEqual(payload["version"], "pinned-revision")
        self.assertEqual(payload["max_length"], 8192)
        self.assertTrue(payload["normalize_embeddings"])
        self.assertTrue(payload["return_dense"])
        self.assertFalse(payload["return_sparse"])
        self.assertFalse(payload["return_colbert_vecs"])


if __name__ == "__main__":
    unittest.main()
