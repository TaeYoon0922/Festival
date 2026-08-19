import unittest

from app.retrieval.embedding_benchmark import (
    EmbeddingBenchmarkConfig,
    EmbeddingSubsetBenchmark,
)
from app.retrieval.embedding_pipeline import RetryConfig
from app.retrieval.embeddings import EmbeddingConfig


class FakeBenchmarkProvider:
    def __init__(self, *, failures=0):
        self.config = EmbeddingConfig(
            provider="fake",
            model="fake-model",
            version="v1",
            dimensions=3,
            batch_size=2,
            max_length=128,
            device="cpu",
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
        self.assertEqual(report["latency_seconds"], {"mean": 1.5, "p50": 1.5, "p95": 2.0})
        self.assertEqual(report["estimated_gold60_subset_seconds"], 30.0)
        self.assertEqual(report["estimated_full_corpus_seconds"], 300.0)
        self.assertEqual(report["peak_memory_mib"], 10.0)
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


if __name__ == "__main__":
    unittest.main()
