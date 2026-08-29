"""The preflight exists to stop a benchmark from misdescribing itself.

Hash embeddings are 1024-dimensional, exactly like BGE-M3, so nothing about the
vector shape reveals a fallback.  And when stored vectors carry a different model
or revision, the vector SQL matches no rows and the pipeline quietly returns a
lexical-only answer that still looks like a result.  Each test here is one of
those silent failures made loud.
"""

import unittest

from scripts.bge_eval_preflight import (
    PINNED_DIMENSIONS,
    PINNED_MODEL,
    PINNED_REVISION,
    PreflightError,
    assert_embedding_identity,
    assert_manifest_current,
    assert_stored_identity,
    assert_vector_coverage,
    describe,
)


class _Config:
    def __init__(self, **overrides) -> None:
        self.provider = "bge_m3_local"
        self.model = PINNED_MODEL
        self.version = PINNED_REVISION
        self.dimensions = PINNED_DIMENSIONS
        self.device = "cuda"
        for key, value in overrides.items():
            setattr(self, key, value)


class IdentityTests(unittest.TestCase):
    def test_the_pinned_configuration_is_accepted(self) -> None:
        assert_embedding_identity(_Config())

    def test_hash_is_refused_even_though_its_dimension_matches(self) -> None:
        with self.assertRaises(PreflightError) as caught:
            assert_embedding_identity(_Config(provider="hash"))

        self.assertIn("hash", str(caught.exception))

    def test_a_different_model_is_refused(self) -> None:
        with self.assertRaises(PreflightError):
            assert_embedding_identity(_Config(model="intfloat/multilingual-e5-large"))

    def test_a_different_revision_is_refused(self) -> None:
        """Same repository, different commit, is a different experiment."""

        with self.assertRaises(PreflightError):
            assert_embedding_identity(_Config(version="5617a9f61b028005a4858fdac845db406aefb181"))

    def test_a_dimension_mismatch_is_refused(self) -> None:
        with self.assertRaises(PreflightError):
            assert_embedding_identity(_Config(dimensions=768))

    def test_cpu_is_refused_for_a_gpu_scoped_baseline(self) -> None:
        with self.assertRaises(PreflightError):
            assert_embedding_identity(_Config(device="cpu"))

    def test_cpu_is_allowed_when_the_run_declares_itself_cpu_scoped(self) -> None:
        assert_embedding_identity(_Config(device="cpu"), require_device=None)

    def test_an_unknown_provider_is_refused(self) -> None:
        with self.assertRaises(PreflightError):
            assert_embedding_identity(_Config(provider="clova_studio"))


class StoredIdentityTests(unittest.TestCase):
    def test_matching_rows_pass(self) -> None:
        assert_stored_identity([
            {"embedding_model": PINNED_MODEL, "embedding_version": PINNED_REVISION,
             "embedding_dimensions": PINNED_DIMENSIONS}])

    def test_a_foreign_row_is_refused(self) -> None:
        with self.assertRaises(PreflightError):
            assert_stored_identity([
                {"embedding_model": "festival-hash-embedding",
                 "embedding_version": "v1", "embedding_dimensions": 1024}])


class CoverageTests(unittest.TestCase):
    def test_full_coverage_passes(self) -> None:
        assert_vector_coverage({"Q1": {"eligible": 10, "embedded": 10}})

    def test_a_question_with_no_bge_rows_is_refused(self) -> None:
        """This is the silent lexical-only degradation."""

        with self.assertRaises(PreflightError) as caught:
            assert_vector_coverage({"Q1": {"eligible": 10, "embedded": 0}})

        self.assertIn("lexical-only", str(caught.exception))

    def test_partial_coverage_is_refused(self) -> None:
        with self.assertRaises(PreflightError):
            assert_vector_coverage({"Q1": {"eligible": 10, "embedded": 9}})

    def test_a_question_with_no_eligible_candidates_is_not_a_failure(self) -> None:
        assert_vector_coverage({"Q1": {"eligible": 0, "embedded": 0}})


class ManifestTests(unittest.TestCase):
    EXPECTED = {
        "corpus_snapshot": "structural_v2_1_full_4204",
        "question_set": "gold60_agent_questions",
        "candidate_union_hash": "abc123",
        "embedding_model": PINNED_MODEL,
        "embedding_revision": PINNED_REVISION,
        "retrieval_config_hash": "cfg1",
    }

    def test_a_current_manifest_passes(self) -> None:
        assert_manifest_current(dict(self.EXPECTED), self.EXPECTED)

    def test_a_changed_corpus_snapshot_is_refused(self) -> None:
        stale = dict(self.EXPECTED, corpus_snapshot="structural_v2_0")
        with self.assertRaises(PreflightError):
            assert_manifest_current(stale, self.EXPECTED)

    def test_a_changed_candidate_union_is_refused(self) -> None:
        stale = dict(self.EXPECTED, candidate_union_hash="deadbeef")
        with self.assertRaises(PreflightError):
            assert_manifest_current(stale, self.EXPECTED)

    def test_a_changed_retrieval_config_is_refused(self) -> None:
        stale = dict(self.EXPECTED, retrieval_config_hash="cfg2")
        with self.assertRaises(PreflightError):
            assert_manifest_current(stale, self.EXPECTED)


class MetadataTests(unittest.TestCase):
    def test_describe_carries_every_identity_field(self) -> None:
        described = describe(_Config())

        self.assertEqual(described, {
            "embedding_provider": "bge_m3_local",
            "embedding_model": PINNED_MODEL,
            "embedding_revision": PINNED_REVISION,
            "embedding_dimensions": PINNED_DIMENSIONS,
            "embedding_device": "cuda",
        })


class IsolationTests(unittest.TestCase):
    def test_the_seeder_refuses_a_non_evaluation_database(self) -> None:
        """Evaluation writes may only reach the isolated container on 55433."""

        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "scripts/seed_eval_corpus.py", "00126380",
             "--dsn", "postgresql://user:pw@127.0.0.1:5432/festival"],
            capture_output=True, text=True, encoding="utf-8", errors="replace")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to write outside", (result.stdout or "") + (result.stderr or ""))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
