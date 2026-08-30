"""The model that loads must be provably the model that was configured.

``FlagEmbedding`` accepts a ``revision`` argument and does not honour it: asked
for a pinned commit it resolved ``main`` instead, and nothing downstream noticed
because every stored row is stamped with the *configured* revision. These tests
hold the line on the only thing that makes that impossible -- resolving the
commit here, checking what came back, and handing onward a directory rather than
a name something else could re-resolve.

No test downloads a model. Snapshots are directories of empty files, which is
enough: the resolver's job is identity and completeness, not weights.
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import app.retrieval.bge_m3 as bge_m3
from app.retrieval.bge_m3 import (
    BgeM3IdentityError,
    BgeM3LocalEmbeddingProvider,
    ResolvedSnapshot,
    _load_flag_embedding_encoder,
    _resolve_verified_bge_snapshot,
)
from app.retrieval.embeddings import EmbeddingConfig

PIN = "6892b95fed65c899a30896eb40d619ae284d0455"
MAIN = "5617a9f61b028005a4858fdac845db406aefb181"

#: Exactly the INFRA-E1 cache shape: what dense inference needs, nothing else.
DENSE_FILES = ("config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json",
               "sentencepiece.bpe.model")


def config(**overrides):
    values = {"provider": "bge_m3_local", "model": "BAAI/bge-m3", "version": PIN,
              "dimensions": 1024, "batch_size": 4, "max_length": 8192,
              "device": "cpu"}
    values.update(overrides)
    return EmbeddingConfig(**values)


def snapshot_dir(root, revision, files=DENSE_FILES):
    path = os.path.join(root, revision)
    os.makedirs(path, exist_ok=True)
    for name in files:
        with open(os.path.join(path, name), "w", encoding="utf-8") as handle:
            handle.write("{}")
    return path


def downloader_for(mapping, *, offline_only=()):
    """Stand in for ``snapshot_download`` over a fake cache.

    ``offline_only`` names revisions the network would serve but the cache does
    not, so the offline-first attempt fails and the networked retry succeeds.
    """

    def download(*, repo_id, revision, allow_patterns=None, local_files_only=False):
        if local_files_only and revision in offline_only:
            raise OSError("not cached")
        if revision not in mapping:
            raise OSError(f"no snapshot for {revision}")
        return mapping[revision]

    return download


class RevisionImmutabilityTests(unittest.TestCase):
    """A mutable name would let one stored version describe two models."""

    def test_a_branch_name_is_refused(self) -> None:
        with self.assertRaises(BgeM3IdentityError) as caught:
            _resolve_verified_bge_snapshot(config(version="main"),
                                           downloader=downloader_for({}))

        self.assertIn("commit SHA", str(caught.exception))

    def test_a_tag_name_is_refused(self) -> None:
        with self.assertRaises(BgeM3IdentityError):
            _resolve_verified_bge_snapshot(config(version="v1.5"),
                                           downloader=downloader_for({}))

    def test_a_truncated_sha_is_refused(self) -> None:
        with self.assertRaises(BgeM3IdentityError):
            _resolve_verified_bge_snapshot(config(version=PIN[:12]),
                                           downloader=downloader_for({}))

    def test_an_empty_revision_is_refused_by_the_config_layer(self) -> None:
        with self.assertRaises(ValueError):
            config(version="")


class CommitVerificationTests(unittest.TestCase):
    def test_the_pinned_commit_resolves_and_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN)
            resolved = _resolve_verified_bge_snapshot(
                config(), downloader=downloader_for({PIN: path}))

        self.assertIsInstance(resolved, ResolvedSnapshot)
        self.assertEqual(resolved.revision, PIN)
        self.assertEqual(resolved.model, "BAAI/bge-m3")
        self.assertEqual(resolved.local_path, path)

    def test_a_resolver_returning_main_for_a_pinned_config_is_refused(self) -> None:
        """The exact defect: the pin was requested and ``main`` came back."""

        with tempfile.TemporaryDirectory() as root:
            main = snapshot_dir(root, MAIN)
            with self.assertRaises(BgeM3IdentityError) as caught:
                _resolve_verified_bge_snapshot(
                    config(), downloader=lambda **_kw: main)

        message = str(caught.exception)
        self.assertIn(PIN, message)
        self.assertIn(MAIN, message)

    def test_main_cached_but_pin_absent_never_substitutes(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            main = snapshot_dir(root, MAIN)
            with self.assertRaises(OSError):
                _resolve_verified_bge_snapshot(
                    config(), downloader=downloader_for({MAIN: main}))

    def test_both_cached_selects_the_pin_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            pinned = snapshot_dir(root, PIN)
            main = snapshot_dir(root, MAIN)
            resolved = _resolve_verified_bge_snapshot(
                config(), downloader=downloader_for({PIN: pinned, MAIN: main}))

        self.assertEqual(resolved.revision, PIN)
        self.assertEqual(resolved.local_path, pinned)

    def test_a_trailing_separator_does_not_defeat_verification(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN)
            resolved = _resolve_verified_bge_snapshot(
                config(), downloader=lambda **_kw: path + os.sep)

        self.assertEqual(resolved.revision, PIN)


class RequiredFileTests(unittest.TestCase):
    """Completeness means what dense inference needs, not a repository mirror."""

    def test_the_infra_cache_shape_is_accepted(self) -> None:
        """Pattern-partial but usable: no README, no sparse or ColBERT heads."""

        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN, files=DENSE_FILES)
            resolved = _resolve_verified_bge_snapshot(
                config(), downloader=downloader_for({PIN: path}))

        self.assertEqual(resolved.revision, PIN)
        for absent in ("README.md", ".gitattributes", "colbert_linear.pt",
                       "sparse_linear.pt"):
            self.assertFalse(os.path.exists(os.path.join(resolved.local_path, absent)))

    def test_a_missing_config_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN,
                                files=("model.safetensors", "tokenizer.json"))
            with self.assertRaises(BgeM3IdentityError) as caught:
                _resolve_verified_bge_snapshot(
                    config(), downloader=downloader_for({PIN: path}))

        self.assertIn("config.json", str(caught.exception))

    def test_a_missing_safetensors_checkpoint_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN, files=("config.json", "tokenizer.json"))
            with self.assertRaises(BgeM3IdentityError) as caught:
                _resolve_verified_bge_snapshot(
                    config(), downloader=downloader_for({PIN: path}))

        self.assertIn("model.safetensors", str(caught.exception))

    def test_a_bin_checkpoint_does_not_satisfy_the_requirement(self) -> None:
        """No ``.bin`` fallback: it is the unsafe path the CVE restriction targets."""

        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN, files=("config.json", "tokenizer.json",
                                                  "pytorch_model.bin"))
            with self.assertRaises(BgeM3IdentityError) as caught:
                _resolve_verified_bge_snapshot(
                    config(), downloader=downloader_for({PIN: path}))

        self.assertIn("model.safetensors", str(caught.exception))

    def test_a_snapshot_with_no_tokenizer_asset_is_refused(self) -> None:
        """Loading would succeed by sourcing a tokenizer elsewhere -- that is drift."""

        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN,
                                files=("config.json", "model.safetensors"))
            with self.assertRaises(BgeM3IdentityError) as caught:
                _resolve_verified_bge_snapshot(
                    config(), downloader=downloader_for({PIN: path}))

        self.assertIn("tokenizer", str(caught.exception))

    def test_either_tokenizer_payload_suffices(self) -> None:
        for payload in ("tokenizer.json", "sentencepiece.bpe.model"):
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as root:
                    path = snapshot_dir(root, PIN, files=("config.json",
                                                          "model.safetensors",
                                                          payload))
                    resolved = _resolve_verified_bge_snapshot(
                        config(), downloader=downloader_for({PIN: path}))
                self.assertEqual(resolved.revision, PIN)


class CacheAndOfflineTests(unittest.TestCase):
    def test_a_warm_pinned_cache_loads_without_the_network(self) -> None:
        seen = []

        def download(*, repo_id, revision, allow_patterns=None, local_files_only=False):
            seen.append(local_files_only)
            if not local_files_only:
                raise AssertionError("network must not be consulted for a warm cache")
            return path

        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN)
            resolved = _resolve_verified_bge_snapshot(config(), downloader=download)

        self.assertEqual(resolved.revision, PIN)
        self.assertEqual(seen, [True])

    def test_a_cold_cache_falls_back_to_the_network_for_the_same_pin(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN)
            download = downloader_for({PIN: path}, offline_only=(PIN,))
            resolved = _resolve_verified_bge_snapshot(config(), downloader=download)

        self.assertEqual(resolved.revision, PIN)

    def test_absent_pin_without_network_fails_clearly(self) -> None:
        def download(**_kwargs):
            raise OSError("offline and not cached")

        with self.assertRaises(OSError):
            _resolve_verified_bge_snapshot(config(), downloader=download)


class LoaderContractTests(unittest.TestCase):
    def test_flag_embedding_receives_the_path_and_no_revision(self) -> None:
        calls = []

        def model_factory(model, **kwargs):
            calls.append((model, kwargs))
            return SimpleNamespace(encode=lambda *a, **k: None)

        with tempfile.TemporaryDirectory() as root:
            path = snapshot_dir(root, PIN)
            with patch.dict(sys.modules,
                            {"FlagEmbedding": SimpleNamespace(BGEM3FlagModel=model_factory)}), \
                    patch.object(bge_m3, "_snapshot_downloader",
                                 lambda: downloader_for({PIN: path})):
                _load_flag_embedding_encoder(config())

        model, kwargs = calls[0]
        self.assertEqual(model, path)
        self.assertNotEqual(model, "BAAI/bge-m3")
        self.assertNotIn("revision", kwargs)
        self.assertTrue(kwargs["normalize_embeddings"])

    def test_no_hash_or_alternate_model_fallback_on_failure(self) -> None:
        def download(**_kwargs):
            raise OSError("unavailable")

        with patch.dict(sys.modules,
                        {"FlagEmbedding": SimpleNamespace(BGEM3FlagModel=object)}), \
                patch.object(bge_m3, "_snapshot_downloader", lambda: download):
            with self.assertRaises(OSError):
                _load_flag_embedding_encoder(config())


class IdentityDiagnosticsTests(unittest.TestCase):
    def test_the_provider_reports_the_resolved_commit_not_the_requested_one(self) -> None:
        encoder = SimpleNamespace(encode=lambda *a, **k: None)
        encoder.festival_resolved_snapshot = ResolvedSnapshot(
            model="BAAI/bge-m3", revision=PIN, local_path="/models/x/" + PIN)
        provider = BgeM3LocalEmbeddingProvider(config(), encoder=encoder)

        identity = provider.identity()

        self.assertEqual(identity["configured_revision"], PIN)
        self.assertEqual(identity["resolved_revision"], PIN)
        self.assertEqual(identity["resolved_snapshot_path"], "/models/x/" + PIN)
        self.assertTrue(identity["verified"])

    def test_an_unverified_encoder_is_not_reported_as_verified(self) -> None:
        """An injected encoder has no provenance; saying otherwise would be a lie."""

        provider = BgeM3LocalEmbeddingProvider(
            config(), encoder=SimpleNamespace(encode=lambda *a, **k: None))

        identity = provider.identity()

        self.assertIsNone(identity["resolved_revision"])
        self.assertFalse(identity["verified"])

    def test_a_drifted_snapshot_is_reported_as_unverified(self) -> None:
        encoder = SimpleNamespace(encode=lambda *a, **k: None)
        encoder.festival_resolved_snapshot = ResolvedSnapshot(
            model="BAAI/bge-m3", revision=MAIN, local_path="/models/x/" + MAIN)
        provider = BgeM3LocalEmbeddingProvider(config(), encoder=encoder)

        identity = provider.identity()

        self.assertEqual(identity["resolved_revision"], MAIN)
        self.assertNotEqual(identity["resolved_revision"],
                            identity["configured_revision"])
        self.assertFalse(identity["verified"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
