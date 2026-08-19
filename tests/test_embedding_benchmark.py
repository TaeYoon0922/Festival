import unittest

from app.retrieval.embedding_benchmark import (
    EmbeddingBenchmarkConfig,
    EmbeddingSubsetBenchmark,
)
from app.retrieval.embedding_pipeline import RetryConfig
from app.retrieval.embeddings import EmbeddingConfig
from app.retrieval.embeddings import EmbeddingHttpError


class FakeBenchmarkProvider:
    def __init__(self, *, failures=0, provider_name="fake", device="cpu"):
        self.config = EmbeddingConfig(
            provider=provider_name,
            model="fake-model",
            version="v1",
            dimensions=3,
            batch_size=2,
            max_length=128,
            device=device,
        )
        self.failures = failures
        self.load_calls = 0
        self.embedding_calls = []

    def load(self):
        self.load_calls += 1

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_documents(self, texts):
        self.embedding_calls.append(list(texts))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transient")
        return [[1.0, 0.0, 0.0] for _ in texts]

    def peak_device_memory_bytes(self):
        return 20 * 1024 * 1024 if self.config.device == "cuda" else None


class ScriptedClock:
    def __init__(self, values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


def candidates(count):
    return [
        {"chunk_id": f"c{index}", "retrieval_text": f"text {index}"}
        for index in range(count)
    ]


class EmbeddingBenchmarkTests(unittest.TestCase):
    def test_calculates_load_throughput_latency_and_duration_estimates(self) -> None:
        provider = FakeBenchmarkProvider()
        benchmark = EmbeddingSubsetBenchmark(
            provider,
            config=EmbeddingBenchmarkConfig(
                limit=3,
                batch_size=2,
                gold60_candidate_count=30,
                full_corpus_count=300,
            ),
            clock=ScriptedClock([0.0, 2.0, 2.0, 3.0, 3.0, 5.0]),
            peak_memory=lambda: 10 * 1024 * 1024,
        )

        report = benchmark.run(candidates(5))

        self.assertEqual(provider.load_calls, 1)
        self.assertEqual(report["processed"], 3)
        self.assertEqual(report["embedded"], 3)
        self.assertEqual(report["model_load_seconds"], 2.0)
        self.assertEqual(report["embedding_seconds"], 3.0)
        self.assertEqual(report["documents_per_second"], 1.0)
        self.assertEqual(report["speedup_vs_cpu"], 0.672)
        self.assertEqual(report["latency_seconds"], {"mean": 1.5, "p50": 1.5, "p95": 2.0})
        self.assertEqual(report["estimated_gold60_subset_seconds"], 30.0)
        self.assertEqual(report["estimated_full_corpus_seconds"], 300.0)
        self.assertEqual(report["peak_memory_mib"], 10.0)
        self.assertIsNone(report["peak_device_memory_mib"])
        self.assertEqual(report["effective_batch_size"], 2)
        self.assertEqual(report["cuda_oom_retries"], 0)
        self.assertEqual(report["database_writes"], 0)

    def test_retries_provider_failure_without_database(self) -> None:
        provider = FakeBenchmarkProvider(failures=1)
        benchmark = EmbeddingSubsetBenchmark(
            provider,
            config=EmbeddingBenchmarkConfig(
                limit=1,
                batch_size=1,
                retry=RetryConfig(max_attempts=2, initial_delay_seconds=0),
            ),
            clock=ScriptedClock([0.0, 0.0, 0.0, 1.0]),
            peak_memory=lambda: None,
            sleep=lambda _seconds: None,
        )
        report = benchmark.run(candidates(1))
        self.assertEqual(len(provider.embedding_calls), 2)
        self.assertEqual(report["embedded"], 1)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["database_writes"], 0)

    def test_permanent_failure_is_counted(self) -> None:
        provider = FakeBenchmarkProvider(failures=5)
        benchmark = EmbeddingSubsetBenchmark(
            provider,
            config=EmbeddingBenchmarkConfig(
                limit=2,
                batch_size=2,
                retry=RetryConfig(max_attempts=2, initial_delay_seconds=0),
            ),
            clock=ScriptedClock([0.0, 0.0, 0.0, 1.0]),
            peak_memory=lambda: None,
            sleep=lambda _seconds: None,
        )
        report = benchmark.run(candidates(2))
        self.assertEqual(report["embedded"], 0)
        self.assertEqual(report["failed"], 2)
        self.assertIsNone(report["estimated_gold60_subset_seconds"])

    def test_http_retry_after_is_honored_for_transient_failure(self) -> None:
        provider = FakeBenchmarkProvider()
        original = provider.embed_documents
        calls = []

        def embed(texts):
            calls.append(list(texts))
            if len(calls) == 1:
                raise EmbeddingHttpError(
                    "HTTP 429",
                    status_code=429,
                    transient=True,
                    retry_after_seconds=5,
                )
            return original(texts)

        provider.embed_documents = embed
        sleeps = []
        benchmark = EmbeddingSubsetBenchmark(
            provider,
            config=EmbeddingBenchmarkConfig(
                limit=1,
                batch_size=1,
                retry=RetryConfig(max_attempts=2, initial_delay_seconds=1),
            ),
            clock=ScriptedClock([0.0, 0.0, 0.0, 1.0]),
            peak_memory=lambda: None,
            sleep=sleeps.append,
        )
        report = benchmark.run(candidates(1))
        self.assertEqual(report["embedded"], 1)
        self.assertEqual(sleeps, [5.0])

    def test_permanent_http_4xx_is_not_retried(self) -> None:
        provider = FakeBenchmarkProvider()

        def embed(_texts):
            provider.embedding_calls.append(["attempt"])
            raise EmbeddingHttpError(
                "HTTP 400", status_code=400, transient=False
            )

        provider.embed_documents = embed
        benchmark = EmbeddingSubsetBenchmark(
            provider,
            config=EmbeddingBenchmarkConfig(
                limit=1,
                batch_size=1,
                retry=RetryConfig(max_attempts=3, initial_delay_seconds=0),
            ),
            clock=ScriptedClock([0.0, 0.0, 0.0, 1.0]),
            peak_memory=lambda: None,
        )
        report = benchmark.run(candidates(1))
        self.assertEqual(report["failed"], 1)
        self.assertEqual(len(provider.embedding_calls), 1)

    def test_gpu_and_http_reports_share_the_same_schema(self) -> None:
        reports = []
        for provider in (
            FakeBenchmarkProvider(provider_name="bge_m3_local", device="cuda"),
            FakeBenchmarkProvider(provider_name="bge_m3_http"),
        ):
            reports.append(
                EmbeddingSubsetBenchmark(
                    provider,
                    config=EmbeddingBenchmarkConfig(limit=1, batch_size=1),
                    clock=ScriptedClock([0.0, 0.0, 0.0, 1.0]),
                    peak_memory=lambda: None,
                ).run(candidates(1))
            )
        self.assertEqual(set(reports[0]), set(reports[1]))
        self.assertEqual(reports[0]["peak_device_memory_mib"], 20.0)
        self.assertIsNone(reports[1]["peak_device_memory_mib"])


if __name__ == "__main__":
    unittest.main()
