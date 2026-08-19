import json
import tempfile
import unittest
from pathlib import Path

from app.retrieval.embedding_pipeline import (
    EmbeddingDimensionMismatch,
    EmbeddingPipelineConfig,
    RetryConfig,
    SubsetEmbeddingPipeline,
    iter_candidate_file,
)
from app.retrieval.embeddings import EmbeddingConfig


class FakeProvider:
    def __init__(self, config, *, failures=0, dimensions=None):
        self.config = config
        self.failures = failures
        self.dimensions = dimensions or config.dimensions
        self.calls = []

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        if self.failures:
            self.failures -= 1
            raise RuntimeError("transient provider error")
        return [[float(index + 1)] * self.dimensions for index, _ in enumerate(texts)]


class FakeStore:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.lookups = []
        self.upserts = []

    def existing_embedding_chunk_ids(self, chunk_ids, **identity):
        self.lookups.append((list(chunk_ids), identity))
        return self.existing.intersection(chunk_ids)

    def upsert_embeddings(self, records, **identity):
        copied = [dict(record) for record in records]
        self.upserts.append((copied, identity))
        self.existing.update(record["chunk_id"] for record in records)
        return len(records)


def candidates(*chunk_ids):
    return [
        {
            "chunk_id": chunk_id,
            "retrieval_text": f"retrieval-{chunk_id}",
            "content": f"content-{chunk_id}",
        }
        for chunk_id in chunk_ids
    ]


class EmbeddingPipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = EmbeddingConfig(
            provider="test", model="model", version="v1", dimensions=3, batch_size=2
        )

    def pipeline(self, provider, store, **overrides):
        values = {
            "batch_size": 2,
            "retry": RetryConfig(max_attempts=3, initial_delay_seconds=0),
            **overrides,
        }
        return SubsetEmbeddingPipeline(
            provider,
            store,
            self.config,
            pipeline_config=EmbeddingPipelineConfig(**values),
            sleep=lambda _seconds: None,
        )

    def test_skips_existing_deduplicates_and_batches_retrieval_text_only(self):
        provider = FakeProvider(self.config)
        store = FakeStore(existing={"c1"})
        summary = self.pipeline(provider, store).run(candidates("c1", "c2", "c2", "c3"))

        self.assertEqual(summary["already_embedded"], 1)
        self.assertEqual(summary["duplicate_chunk_ids"], 1)
        self.assertEqual(summary["embedded"], 2)
        self.assertEqual(provider.calls, [["retrieval-c2"], ["retrieval-c3"]])
        self.assertEqual(len(store.upserts), 2)

    def test_resume_uses_database_as_checkpoint(self):
        store = FakeStore()
        first_provider = FakeProvider(self.config)
        first = self.pipeline(first_provider, store, batch_size=1).run(candidates("c1"))
        second_provider = FakeProvider(self.config)
        second = self.pipeline(second_provider, store, batch_size=1).run(
            candidates("c1", "c2")
        )
        self.assertEqual(first["embedded"], 1)
        self.assertEqual(second["already_embedded"], 1)
        self.assertEqual(second["embedded"], 1)
        self.assertEqual(second_provider.calls, [["retrieval-c2"]])

    def test_force_overwrites_existing_embedding(self):
        provider = FakeProvider(self.config)
        store = FakeStore(existing={"c1"})
        summary = self.pipeline(provider, store, force=True).run(candidates("c1"))
        self.assertEqual(summary["embedded"], 1)
        self.assertEqual(store.lookups, [])
        self.assertTrue(store.upserts[0][1]["force"])

    def test_dry_run_detects_missing_without_provider_or_upsert(self):
        provider = FakeProvider(self.config)
        store = FakeStore(existing={"c1"})
        summary = self.pipeline(provider, store, dry_run=True).run(
            candidates("c1", "c2")
        )
        self.assertEqual(summary["would_embed"], 1)
        self.assertEqual(summary["embedded"], 0)
        self.assertEqual(provider.calls, [])
        self.assertEqual(store.upserts, [])

    def test_dimension_mismatch_fails_before_database_insert(self):
        provider = FakeProvider(self.config, dimensions=2)
        store = FakeStore()
        with self.assertRaises(EmbeddingDimensionMismatch):
            self.pipeline(provider, store).run(candidates("c1"))
        self.assertEqual(store.upserts, [])

    def test_transient_provider_exception_is_retried(self):
        provider = FakeProvider(self.config, failures=2)
        store = FakeStore()
        summary = self.pipeline(provider, store).run(candidates("c1"))
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(summary["embedded"], 1)

    def test_permanent_failure_is_logged_per_chunk(self):
        provider = FakeProvider(self.config, failures=10)
        store = FakeStore()
        with tempfile.TemporaryDirectory() as temporary:
            error_path = Path(temporary) / "errors.jsonl"
            pipeline = SubsetEmbeddingPipeline(
                provider,
                store,
                self.config,
                pipeline_config=EmbeddingPipelineConfig(
                    batch_size=2,
                    resume=False,
                    retry=RetryConfig(max_attempts=2, initial_delay_seconds=0),
                ),
                error_path=error_path,
                sleep=lambda _seconds: None,
            )
            summary = pipeline.run(candidates("c1", "c2"))
            errors = [
                json.loads(line)
                for line in error_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(summary["failed"], 2)
        self.assertEqual({row["chunk_id"] for row in errors}, {"c1", "c2"})
        self.assertTrue(all(row["stage"] == "embedding" for row in errors))

    def test_candidate_file_streaming_and_id_hydration(self):
        class HydrationBackend:
            def fetch_embedding_source_chunks(self, chunk_ids):
                return [
                    {"chunk_id": chunk_id, "retrieval_text": f"text-{chunk_id}"}
                    for chunk_id in chunk_ids
                    if chunk_id != "missing"
                ]

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.txt"
            path.write_text("c1\nmissing\nc2\n", encoding="utf-8")
            rows = list(
                iter_candidate_file(path, backend=HydrationBackend(), fetch_batch_size=2)
            )
        self.assertEqual([row["chunk_id"] for row in rows], ["c1", "missing", "c2"])
        self.assertNotIn("retrieval_text", rows[1])

    def test_empty_candidate_stream_does_nothing(self):
        provider = FakeProvider(self.config)
        store = FakeStore()
        summary = self.pipeline(provider, store).run([])
        self.assertEqual(summary["unique_chunk_ids"], 0)
        self.assertEqual(summary["batches"], 0)
        self.assertEqual(provider.calls, [])
        self.assertEqual(store.upserts, [])

    def test_keyboard_interrupt_returns_resumable_status(self):
        class InterruptingProvider(FakeProvider):
            def embed_documents(self, texts):
                raise KeyboardInterrupt()

        summary = self.pipeline(InterruptingProvider(self.config), FakeStore()).run(
            candidates("c1")
        )
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["embedded"], 0)


if __name__ == "__main__":
    unittest.main()
