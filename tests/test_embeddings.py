import json
import math
import unittest
from io import BytesIO
from urllib.error import HTTPError

from app.retrieval.embeddings import (
    DeterministicHashEmbedder,
    EmbeddingConfig,
    EmbeddingHttpError,
    HttpEmbeddingSettings,
    OpenAICompatibleEmbeddingProvider,
    UrllibJsonTransport,
    _split_embedding_text,
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

    def test_embedding_config_reads_cuda_oom_controls(self) -> None:
        config = EmbeddingConfig.from_env(
            {
                "FESTIVAL_EMBEDDING_BATCH_SIZE": "32",
                "FESTIVAL_EMBEDDING_CUDA_OOM_RETRY": "false",
                "FESTIVAL_EMBEDDING_MIN_BATCH_SIZE": "8",
            }
        )
        self.assertFalse(config.cuda_oom_retry)
        self.assertEqual(config.min_batch_size, 8)

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

    def test_clova_single_input_payload_and_1024_dimension_response(self) -> None:
        vector = [0.0] * 1024
        vector[0] = 1.0
        transport = FakeTransport(
            {"data": [{"index": 0, "embedding": vector}]}
        )
        config = EmbeddingConfig(
            provider="clova_studio",
            model="bge-m3",
            version="clova-bge-m3-v1",
            dimensions=1024,
            batch_size=1,
        )
        provider = create_embedding_provider(
            config,
            environment={
                "FESTIVAL_EMBEDDING_API_URL": (
                    "https://clovastudio.stream.ntruss.com/v1/openai/embeddings"
                ),
                "FESTIVAL_EMBEDDING_API_KEY": "secret",
            },
            transport=transport,
        )

        result = provider.embed_query("단일 입력")

        self.assertEqual(len(result), 1024)
        _url, call = transport.calls[0]
        self.assertEqual(
            call["payload"],
            {
                "model": "bge-m3",
                "input": "단일 입력",
                "encoding_format": "float",
                "dimensions": 1024,
            },
        )

    def test_clova_rejects_wrong_dimension(self) -> None:
        config = EmbeddingConfig(
            provider="clova_studio",
            model="bge-m3",
            version="clova-bge-m3-v1",
            dimensions=1024,
            batch_size=1,
        )
        provider = create_embedding_provider(
            config,
            environment={
                "FESTIVAL_EMBEDDING_API_URL": "https://embedding.invalid/v1",
                "FESTIVAL_EMBEDDING_API_KEY": "secret",
            },
            transport=FakeTransport(
                {"data": [{"index": 0, "embedding": [0.0] * 1023}]}
            ),
        )
        with self.assertRaisesRegex(ValueError, "expected 1024, got 1023"):
            provider.embed_query("query")

    def test_clova_batch_uses_explicit_sequential_single_input_fallback(self) -> None:
        vector = [0.0] * 1024
        vector[0] = 1.0
        transport = FakeTransport(
            {"data": [{"index": 0, "embedding": vector}]}
        )
        config = EmbeddingConfig(
            provider="clova_studio",
            model="bge-m3",
            version="clova-bge-m3-v1",
            dimensions=1024,
            batch_size=2,
        )
        provider = create_embedding_provider(
            config,
            environment={
                "FESTIVAL_EMBEDDING_API_URL": "https://embedding.invalid/v1",
                "FESTIVAL_EMBEDDING_API_KEY": "secret",
            },
            transport=transport,
        )

        result = provider.embed_documents(["first", "second"])

        self.assertEqual(len(result), 2)
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(
            [call[1]["payload"]["input"] for call in transport.calls],
            ["first", "second"],
        )

    def test_clova_40003_embeds_all_segments_and_mean_pools(self) -> None:
        class LongTextTransport:
            def __init__(self):
                self.calls = []

            def post_json(self, url, **kwargs):
                self.calls.append((url, kwargs))
                if len(self.calls) == 1:
                    raise EmbeddingHttpError(
                        "embedding endpoint returned HTTP 400 (error code 40003)",
                        status_code=400,
                        transient=False,
                        response_error_code="40003",
                        response_error_message="Context length exceeded",
                    )
                vector = [0.0] * 1024
                vector[len(self.calls) - 2] = 1.0
                return {"data": [{"index": 0, "embedding": vector}]}

        text = "aaaa\n\nbbbb"
        transport = LongTextTransport()
        config = EmbeddingConfig(
            provider="clova_studio",
            model="bge-m3",
            version="clova-bge-m3-2026-08-20",
            dimensions=1024,
            batch_size=1,
        )
        provider = create_embedding_provider(
            config,
            environment={
                "FESTIVAL_EMBEDDING_API_URL": "https://embedding.invalid/v1",
                "FESTIVAL_EMBEDDING_API_KEY": "secret",
                "FESTIVAL_EMBEDDING_LONG_TEXT_SEGMENT_CHARS": "8",
            },
            transport=transport,
        )

        result = provider.embed_query(text)

        segment_inputs = [call[1]["payload"]["input"] for call in transport.calls[1:]]
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual("".join(segment_inputs), text)
        self.assertEqual(segment_inputs, ["aaaa\n\n", "bbbb"])
        self.assertTrue(all(isinstance(value, str) for value in segment_inputs))
        self.assertAlmostEqual(result[0], 1 / math.sqrt(2))
        self.assertAlmostEqual(result[1], 1 / math.sqrt(2))
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in result)), 1.0)
        self.assertEqual(len(result), 1024)
        self.assertEqual(
            provider.embedding_statistics(),
            {"long_text_fallbacks": 1, "long_text_segments": 2},
        )

    def test_long_single_paragraph_is_split_without_truncation(self) -> None:
        text = "가" * 31_173
        segments = _split_embedding_text(text, max_chars=1800)

        self.assertGreater(len(segments), 1)
        self.assertTrue(all(0 < len(segment) <= 1800 for segment in segments))
        self.assertEqual("".join(segments), text)

    def test_clova_non_40003_http_400_does_not_fallback(self) -> None:
        class InvalidRequestTransport:
            def __init__(self):
                self.calls = []

            def post_json(self, url, **kwargs):
                self.calls.append((url, kwargs))
                raise EmbeddingHttpError(
                    "embedding endpoint returned HTTP 400 (error code 40001)",
                    status_code=400,
                    transient=False,
                    response_error_code="40001",
                )

        transport = InvalidRequestTransport()
        provider = create_embedding_provider(
            EmbeddingConfig(
                provider="clova_studio",
                model="bge-m3",
                version="clova-bge-m3-2026-08-20",
                dimensions=1024,
                batch_size=1,
            ),
            environment={
                "FESTIVAL_EMBEDDING_API_URL": "https://embedding.invalid/v1",
                "FESTIVAL_EMBEDDING_API_KEY": "secret",
                "FESTIVAL_EMBEDDING_LONG_TEXT_SEGMENT_CHARS": "8",
            },
            transport=transport,
        )

        with self.assertRaises(EmbeddingHttpError):
            provider.embed_query("aaaa\n\nbbbb")
        self.assertEqual(len(transport.calls), 1)
        self.assertEqual(
            provider.embedding_statistics(),
            {"long_text_fallbacks": 0, "long_text_segments": 0},
        )

    def test_clova_hostname_does_not_change_generic_batch_contract(self) -> None:
        config = EmbeddingConfig(
            provider="openai_compatible",
            model="bge-m3",
            version="v1",
            dimensions=2,
            batch_size=2,
        )
        transport = FakeTransport(
            {
                "data": [
                    {"index": 0, "embedding": [1.0, 0.0]},
                    {"index": 1, "embedding": [0.0, 1.0]},
                ]
            }
        )
        provider = OpenAICompatibleEmbeddingProvider(
            config,
            HttpEmbeddingSettings(
                "https://clovastudio.stream.ntruss.com/v1/openai/embeddings",
                "secret",
            ),
            transport=transport,
        )

        provider.embed_documents(["first", "second"])

        self.assertEqual(
            transport.calls[0][1]["payload"],
            {"model": "bge-m3", "input": ["first", "second"]},
        )

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

    def test_transport_marks_429_transient_and_sanitizes_error(self) -> None:
        secret = "must-not-leak"

        def opener(request, **_kwargs):
            raise HTTPError(
                request.full_url,
                429,
                "rate limited",
                {"Retry-After": "7"},
                BytesIO(b'{"error":"body must not leak"}'),
            )

        transport = UrllibJsonTransport(opener=opener)
        with self.assertRaises(EmbeddingHttpError) as raised:
            transport.post_json(
                "https://embedding.invalid/v1",
                headers={"Authorization": f"Bearer {secret}"},
                payload={"input": ["private input"]},
                timeout_seconds=1,
            )
        error = raised.exception
        self.assertTrue(error.transient)
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.retry_after_seconds, 7.0)
        self.assertNotIn(secret, str(error))
        self.assertNotIn("private input", str(error))
        self.assertNotIn("body must not leak", str(error))

    def test_transport_marks_non_retryable_4xx_permanent(self) -> None:
        def opener(request, **_kwargs):
            raise HTTPError(request.full_url, 400, "bad request", {}, None)

        with self.assertRaises(EmbeddingHttpError) as raised:
            UrllibJsonTransport(opener=opener).post_json(
                "https://embedding.invalid/v1",
                headers={"Authorization": "Bearer secret"},
                payload={"input": ["text"]},
                timeout_seconds=1,
            )
        self.assertFalse(raised.exception.transient)
        self.assertEqual(raised.exception.status_code, 400)

    def test_transport_preserves_sanitized_clova_error_details(self) -> None:
        body = json.dumps(
            {
                "error": {
                    "code": "40003",
                    "message": "Context length exceeded.\nAdjust the input length.",
                },
                "request": "raw retrieval text must not be retained",
            }
        ).encode("utf-8")

        def opener(request, **_kwargs):
            raise HTTPError(request.full_url, 400, "bad request", {}, BytesIO(body))

        with self.assertRaises(EmbeddingHttpError) as raised:
            UrllibJsonTransport(opener=opener).post_json(
                "https://embedding.invalid/v1",
                headers={"Authorization": "Bearer secret"},
                payload={"input": "private retrieval text"},
                timeout_seconds=1,
            )
        error = raised.exception
        self.assertEqual(error.response_error_code, "40003")
        self.assertEqual(
            error.response_error_message,
            "Context length exceeded. Adjust the input length.",
        )
        self.assertNotIn("private retrieval text", str(error))
        self.assertNotIn("raw retrieval text", str(error))

    def test_transport_marks_5xx_and_timeout_transient(self) -> None:
        def unavailable(request, **_kwargs):
            raise HTTPError(request.full_url, 503, "unavailable", {}, None)

        with self.assertRaises(EmbeddingHttpError) as unavailable_error:
            UrllibJsonTransport(opener=unavailable).post_json(
                "https://embedding.invalid/v1",
                headers={"Authorization": "Bearer secret"},
                payload={"input": ["text"]},
                timeout_seconds=1,
            )
        self.assertTrue(unavailable_error.exception.transient)
        self.assertEqual(unavailable_error.exception.status_code, 503)

        def timeout(_request, **_kwargs):
            raise TimeoutError("timed out")

        with self.assertRaises(EmbeddingHttpError) as timeout_error:
            UrllibJsonTransport(opener=timeout).post_json(
                "https://embedding.invalid/v1",
                headers={"Authorization": "Bearer secret"},
                payload={"input": ["text"]},
                timeout_seconds=1,
            )
        self.assertTrue(timeout_error.exception.transient)
        self.assertIsNone(timeout_error.exception.status_code)


if __name__ == "__main__":
    unittest.main()
