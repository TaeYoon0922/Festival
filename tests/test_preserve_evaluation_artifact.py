from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.preserve_evaluation_artifact import (
    ARTIFACT_FILES,
    CHECKSUM_NAME,
    DEFAULT_ARTIFACT_DIR,
    KNOWN_EVALUATION_MISMATCH,
    MANIFEST_NAME,
    PreservationError,
    preserve_artifact,
    verify_artifact,
    verify_metrics,
)


_HYBRID_CONFIG = {
    "config": {
        "final_top_k": 10,
        "lexical_top_n": 50,
        "vector_top_n": 50,
        "rerank_mode": "legacy",
        "rrf": {"k": 60, "lexical_weight": 1.0, "vector_weight": 1.0},
    },
    "embedding": {
        "provider": "clova_studio",
        "model": "bge-m3",
        "version": "clova-bge-m3-2026-08-20",
        "dimensions": 1024,
    },
}


def _report() -> dict:
    """A minimal report carrying exactly the frozen Gold60 numbers."""

    return {
        "question_count": 60,
        "evaluation_version": "1",
        "method": {"top_k": 10},
        "vector_status_counts": {"ok": 60},
        "hybrid": {
            "overall": {
                "question_count": 60,
                "recall_at_1": 0.483333,
                "recall_at_5": 0.783333,
                "recall_at_10": 0.9,
                "miss_count": 6,
            }
        },
        "agent": {
            "overall": {
                "question_count": 60,
                "answerable_rate": 1.0,
                "gold_doc_citation_rate": 0.916667,
                "gold_chunk_citation_rate": 0.883333,
                "all_evidence_terms_rate": 0.95,
                "end_to_end_success_rate": 0.9,
            },
            "failure_counts": {"retrieval_miss": 6, "success": 54},
        },
        "questions": [
            {"question_id": f"Q{index:02d}", "hybrid_config": copy.deepcopy(_HYBRID_CONFIG)}
            for index in range(60)
        ],
        "end_to_end_failures": [
            {"question_id": question_id} for question_id in KNOWN_EVALUATION_MISMATCH
        ],
    }


def _identity() -> dict:
    return {
        "evaluation_commit": "d8a5f743c6784994ed6f0b625e0faf589ccf79bd",
        "evaluation_tag": "agent-gold60-90pct-2026-08-21",
        "branch": "taeyoon",
        "preservation_worktree_clean": False,
    }


def _artifact(directory: Path, report: dict | None = None) -> Path:
    payload = _report() if report is None else report
    (directory / "gold60_agent_evaluation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "gold60_agent_evaluation.md").write_text(
        "# Gold60 End-to-End Agent Evaluation\n", encoding="utf-8"
    )
    (directory / "gold60_agent_questions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in payload["questions"]
        ),
        encoding="utf-8",
    )
    (directory / "failure_analysis.json").write_text(
        json.dumps({"summary": {"total_questions": 60}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / "failure_analysis.md").write_text(
        "# Gold60 End-to-End Failure Analysis\n", encoding="utf-8"
    )
    return directory


def _digests(directory: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ARTIFACT_FILES
    }


class PreserveEvaluationArtifactTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.directory = Path(self._temp.name)

    def test_preserve_writes_manifest_and_checksums(self) -> None:
        _artifact(self.directory)

        result = preserve_artifact(self.directory, git_identity=_identity)

        self.assertTrue(result["metrics_verified"])
        self.assertEqual(result["secret_scan"], "clean")
        self.assertEqual(result["warnings"], [])
        manifest = json.loads(
            (self.directory / MANIFEST_NAME).read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["artifact_status"], "preserved_verbatim")
        self.assertEqual(
            manifest["code"]["evaluation_commit"],
            "d8a5f743c6784994ed6f0b625e0faf589ccf79bd",
        )
        self.assertEqual(
            manifest["known_evaluation_mismatch"], list(KNOWN_EVALUATION_MISMATCH)
        )
        self.assertEqual(
            [row["name"] for row in manifest["files"]], list(ARTIFACT_FILES)
        )
        self.assertEqual(
            manifest["execution_parameters"], _HYBRID_CONFIG["config"]
        )
        self.assertEqual(manifest["embedding"], _HYBRID_CONFIG["embedding"])

    def test_preserve_leaves_original_files_byte_for_byte(self) -> None:
        _artifact(self.directory)
        before = _digests(self.directory)

        preserve_artifact(self.directory, git_identity=_identity)

        self.assertEqual(_digests(self.directory), before)

    def test_checksums_match_sha256sum_format(self) -> None:
        _artifact(self.directory)
        expected = _digests(self.directory)

        preserve_artifact(self.directory, git_identity=_identity)

        lines = (
            (self.directory / CHECKSUM_NAME).read_text(encoding="utf-8").splitlines()
        )
        self.assertEqual(len(lines), len(ARTIFACT_FILES))
        for line in lines:
            digest, _, name = line.partition("  ")
            self.assertEqual(digest, expected[name])

    def test_verify_accepts_a_preserved_artifact(self) -> None:
        _artifact(self.directory)
        preserve_artifact(self.directory, git_identity=_identity)

        result = verify_artifact(self.directory)

        self.assertEqual(result["checksums_verified"], len(ARTIFACT_FILES))
        self.assertTrue(result["metrics_verified"])

    def test_verify_rejects_a_modified_artifact(self) -> None:
        _artifact(self.directory)
        preserve_artifact(self.directory, git_identity=_identity)
        (self.directory / "failure_analysis.md").write_text(
            "# tampered\n", encoding="utf-8"
        )

        with self.assertRaises(PreservationError) as error:
            verify_artifact(self.directory)

        self.assertIn("checksum mismatch", str(error.exception))

    def test_verify_requires_a_previous_preservation(self) -> None:
        _artifact(self.directory)

        with self.assertRaises(PreservationError):
            verify_artifact(self.directory)

    def test_missing_artifact_file_fails(self) -> None:
        _artifact(self.directory)
        (self.directory / "failure_analysis.json").unlink()

        with self.assertRaises(PreservationError) as error:
            preserve_artifact(self.directory, git_identity=_identity)

        self.assertIn("failure_analysis.json", str(error.exception))

    def test_drifted_metric_fails_without_touching_the_artifact(self) -> None:
        report = _report()
        report["agent"]["overall"]["end_to_end_success_rate"] = 0.916667
        _artifact(self.directory, report)
        before = _digests(self.directory)

        with self.assertRaises(PreservationError) as error:
            preserve_artifact(self.directory, git_identity=_identity)

        self.assertIn("end_to_end_success_rate", str(error.exception))
        self.assertEqual(_digests(self.directory), before)
        self.assertFalse((self.directory / MANIFEST_NAME).exists())

    def test_unexpected_failure_class_fails(self) -> None:
        report = _report()
        report["agent"]["failure_counts"] = {
            "retrieval_miss": 5,
            "citation_missing": 1,
            "success": 54,
        }
        _artifact(self.directory, report)

        with self.assertRaises(PreservationError) as error:
            preserve_artifact(self.directory, git_identity=_identity)

        self.assertIn("failure_counts", str(error.exception))

    def test_unexpected_failure_ids_fail(self) -> None:
        report = _report()
        report["end_to_end_failures"] = [{"question_id": "P01"}]
        _artifact(self.directory, report)

        with self.assertRaises(PreservationError) as error:
            preserve_artifact(self.directory, git_identity=_identity)

        self.assertIn("end_to_end_failures", str(error.exception))

    def test_wrong_question_count_fails(self) -> None:
        report = _report()
        report["question_count"] = 59
        _artifact(self.directory, report)

        with self.assertRaises(PreservationError) as error:
            preserve_artifact(self.directory, git_identity=_identity)

        self.assertIn("question_count", str(error.exception))

    def test_mixed_retrieval_configuration_fails(self) -> None:
        report = _report()
        report["questions"][7]["hybrid_config"]["config"]["final_top_k"] = 20
        _artifact(self.directory, report)

        with self.assertRaises(PreservationError) as error:
            preserve_artifact(self.directory, git_identity=_identity)

        self.assertIn("configuration", str(error.exception))

    def test_credential_like_string_stops_preservation(self) -> None:
        _artifact(self.directory)
        (self.directory / "failure_analysis.md").write_text(
            "# analysis\nDATABASE_URL=postgresql://user:pw@host:5432/db\n",
            encoding="utf-8",
        )
        before = _digests(self.directory)

        with self.assertRaises(PreservationError) as error:
            preserve_artifact(self.directory, git_identity=_identity)

        message = str(error.exception)
        self.assertIn("credential-like", message)
        self.assertIn("failure_analysis.md", message)
        self.assertEqual(_digests(self.directory), before)
        self.assertFalse((self.directory / MANIFEST_NAME).exists())

    def test_verify_metrics_returns_the_frozen_numbers(self) -> None:
        metrics = verify_metrics(_report())

        self.assertEqual(metrics["question_count"], 60)
        self.assertEqual(metrics["retrieval"]["recall_at_5"], 0.783333)
        self.assertEqual(metrics["agent"]["end_to_end_success_rate"], 0.9)
        self.assertEqual(metrics["failure_counts"], {"retrieval_miss": 6, "success": 54})
        self.assertEqual(
            metrics["end_to_end_failures"], list(KNOWN_EVALUATION_MISMATCH)
        )


class PreservedGold60ArtifactTests(unittest.TestCase):
    """Guard the real preserved artifact when it is present in the working tree."""

    def test_committed_artifact_still_verifies(self) -> None:
        if not (DEFAULT_ARTIFACT_DIR / MANIFEST_NAME).is_file():
            self.skipTest("preserved Gold60 artifact is not available")

        result = verify_artifact(DEFAULT_ARTIFACT_DIR)

        self.assertEqual(result["checksums_verified"], len(ARTIFACT_FILES))
        self.assertEqual(result["secret_scan"], "clean")
        self.assertEqual(
            result["metrics"]["failure_counts"], {"retrieval_miss": 6, "success": 54}
        )


if __name__ == "__main__":
    unittest.main()
