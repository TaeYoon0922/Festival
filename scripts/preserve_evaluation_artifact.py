"""Preserve a frozen Gold60 evaluation artifact without re-running the evaluation.

The five evaluation files produced by ``evaluate_postgres_agent_gold60.py`` are
treated as immutable inputs.  This script only reads them; it never rewrites,
reformats, or re-serializes them.  It verifies the frozen metrics, scans for
credential-like strings, collects git identity, and writes ``run_manifest.json``
and ``SHA256SUMS`` next to the originals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT / "reports" / "evaluation" / "gold60" / "2026-08-21-agent-90pct"
)

MANIFEST_NAME = "run_manifest.json"
CHECKSUM_NAME = "SHA256SUMS"
MANIFEST_SCHEMA_VERSION = "1.0"

#: Original evaluation outputs, preserved byte-for-byte.
ARTIFACT_FILES = (
    "gold60_agent_evaluation.json",
    "gold60_agent_evaluation.md",
    "gold60_agent_questions.jsonl",
    "failure_analysis.json",
    "failure_analysis.md",
)

#: Files larger than this only produce a warning; nothing is truncated or staged.
LARGE_FILE_WARNING_BYTES = 50 * 1024 * 1024

METRIC_DECIMALS = 6

EXPECTED_QUESTION_COUNT = 60

EXPECTED_RETRIEVAL_METRICS = {
    "recall_at_1": 0.483333,
    "recall_at_5": 0.783333,
    "recall_at_10": 0.900000,
    "miss_count": 6,
}

EXPECTED_AGENT_METRICS = {
    "answerable_rate": 1.000000,
    "gold_doc_citation_rate": 0.916667,
    "gold_chunk_citation_rate": 0.883333,
    "all_evidence_terms_rate": 0.950000,
    "end_to_end_success_rate": 0.900000,
}

EXPECTED_FAILURE_COUNTS = {"retrieval_miss": 6, "success": 54}

#: Development-only diagnostic.  Production retrieval and gold labels stay frozen.
KNOWN_EVALUATION_MISMATCH = ("P01", "P06", "P13", "HX08", "HX16", "HX20")

CORPUS_FINGERPRINT = {
    "docs": 4204,
    "source_files": 4619,
    "chunks": 1363336,
    "text_chunks": 229725,
    "table_chunks": 1071368,
    "projection_chunks": 62243,
}

SOURCE_SERVER_PATH = (
    "/srv/festival/app/data/processed/"
    "postgres_agent_gold60_holding_answerability_fix"
)

#: Frozen retrieval/reasoning code that produced this artifact on the test server.
EXPECTED_CODE_COMMIT = "d8a5f743c6784994ed6f0b625e0faf589ccf79bd"
EXPECTED_CODE_TAG = "agent-gold60-90pct-2026-08-21"

#: Credential-like patterns.  A hit stops preservation; nothing is redacted.
SECRET_PATTERNS = (
    ("database_url", r"DATABASE_URL"),
    ("postgres_dsn", r"postgres(?:ql)?://"),
    ("pg_password", r"PGPASSWORD"),
    ("pg_user", r"PGUSER"),
    ("pg_host", r"PGHOST"),
    ("password", r"password"),
    ("passwd", r"passwd"),
    ("api_key", r"api[_-]?key"),
    ("authorization_header", r"authorization"),
    ("bearer_token", r"bearer\s"),
    ("ncp_header", r"x-ncp-"),
    ("embedding_api_key", r"FESTIVAL_EMBEDDING_API_KEY"),
    ("hcx_api_key", r"FESTIVAL_HCX_API_KEY"),
    ("secret", r"secret"),
    ("token", r"token"),
    ("credential", r"credential"),
    ("private_key", r"private[_-]?key"),
    ("pem_block", r"BEGIN [A-Z ]*PRIVATE KEY"),
    ("clova_host", r"clovastudio"),
    ("ncp_host", r"ntruss"),
    ("ncp_key_literal", r"nv-[A-Za-z0-9]{16,}"),
)


class PreservationError(RuntimeError):
    """Raised when the artifact must not be preserved as-is."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and preserve a frozen Gold60 evaluation artifact.  The "
            "original evaluation files are never modified."
        )
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing checksums, metrics, and secret scan without writing.",
    )
    args = parser.parse_args(argv)

    try:
        report = (
            verify_artifact(args.artifact_dir)
            if args.verify
            else preserve_artifact(args.artifact_dir)
        )
    except PreservationError as error:
        print(f"PRESERVATION FAILED: {error}", file=sys.stderr)
        return 1

    for warning in report.get("warnings", []):
        print(f"WARNING: {warning}", file=sys.stderr)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def preserve_artifact(
    artifact_dir: Path,
    *,
    git_identity: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    files = _require_artifact_files(artifact_dir)
    digests = {name: _sha256(path) for name, path in files.items()}
    sizes = {name: path.stat().st_size for name, path in files.items()}

    report = json.loads(
        files["gold60_agent_evaluation.json"].read_text(encoding="utf-8")
    )
    metrics = verify_metrics(report)
    execution = _execution_identity(report)
    _scan_for_secrets(files)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_status": "preserved_verbatim",
        "preserved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "server_path": SOURCE_SERVER_PATH,
            "transfer": "scp byte-for-byte copy; not regenerated locally",
        },
        "code": dict((git_identity or _git_identity)()),
        "evaluation_version": report.get("evaluation_version"),
        "execution_parameters": execution["config"],
        "embedding": execution["embedding"],
        "vector_status_counts": report.get("vector_status_counts"),
        "method": report.get("method"),
        "corpus_fingerprint": dict(CORPUS_FINGERPRINT),
        "metrics": metrics,
        "known_evaluation_mismatch": list(KNOWN_EVALUATION_MISMATCH),
        "secret_scan": {"status": "clean", "pattern_count": len(SECRET_PATTERNS)},
        "files": [
            {"name": name, "sha256": digests[name], "bytes": sizes[name]}
            for name in ARTIFACT_FILES
        ],
    }

    _write_text(artifact_dir / CHECKSUM_NAME, _render_checksums(digests))
    _write_text(
        artifact_dir / MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

    warnings = [
        f"{name} is {sizes[name]} bytes (over 50MB); review before committing"
        for name in ARTIFACT_FILES
        if sizes[name] > LARGE_FILE_WARNING_BYTES
    ]
    return {
        "action": "preserve",
        "artifact_dir": str(artifact_dir),
        "metrics_verified": True,
        "secret_scan": "clean",
        "manifest_written": str(artifact_dir / MANIFEST_NAME),
        "checksums_written": str(artifact_dir / CHECKSUM_NAME),
        "files": manifest["files"],
        "warnings": warnings,
    }


def verify_artifact(artifact_dir: Path) -> dict[str, Any]:
    files = _require_artifact_files(artifact_dir)

    checksum_path = artifact_dir / CHECKSUM_NAME
    manifest_path = artifact_dir / MANIFEST_NAME
    for path in (checksum_path, manifest_path):
        if not path.is_file():
            raise PreservationError(f"missing {path.name}; run without --verify first")

    expected = _parse_checksums(checksum_path.read_text(encoding="utf-8"))
    if set(expected) != set(ARTIFACT_FILES):
        raise PreservationError(f"{CHECKSUM_NAME} does not cover the artifact files")

    mismatched = sorted(
        name for name, digest in expected.items() if _sha256(files[name]) != digest
    )
    if mismatched:
        raise PreservationError(
            "checksum mismatch (artifact was modified): " + ", ".join(mismatched)
        )

    report = json.loads(
        files["gold60_agent_evaluation.json"].read_text(encoding="utf-8")
    )
    metrics = verify_metrics(report)
    _scan_for_secrets(files)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = {row["name"]: row["sha256"] for row in manifest.get("files", ())}
    if recorded != expected:
        raise PreservationError(
            f"{MANIFEST_NAME} checksums disagree with {CHECKSUM_NAME}"
        )

    return {
        "action": "verify",
        "artifact_dir": str(artifact_dir),
        "checksums_verified": len(expected),
        "metrics_verified": True,
        "secret_scan": "clean",
        "metrics": metrics,
        "warnings": [],
    }


def verify_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    """Read metrics out of the artifact and fail if they drift from the frozen run."""

    question_count = report.get("question_count")
    if question_count != EXPECTED_QUESTION_COUNT:
        raise PreservationError(
            f"question_count is {question_count!r}, expected {EXPECTED_QUESTION_COUNT}"
        )

    retrieval = _mapping(report, "hybrid", "overall")
    agent = _mapping(report, "agent", "overall")
    _compare_metrics("hybrid.overall", retrieval, EXPECTED_RETRIEVAL_METRICS)
    _compare_metrics("agent.overall", agent, EXPECTED_AGENT_METRICS)

    failure_counts = dict(_mapping(report, "agent", "failure_counts"))
    if failure_counts != EXPECTED_FAILURE_COUNTS:
        raise PreservationError(
            f"agent.failure_counts is {failure_counts!r}, "
            f"expected {EXPECTED_FAILURE_COUNTS!r}"
        )

    failures = tuple(
        str(row.get("question_id")) for row in report.get("end_to_end_failures") or ()
    )
    if failures != KNOWN_EVALUATION_MISMATCH:
        raise PreservationError(
            f"end_to_end_failures is {list(failures)!r}, "
            f"expected {list(KNOWN_EVALUATION_MISMATCH)!r}"
        )

    return {
        "question_count": question_count,
        "retrieval": {key: retrieval[key] for key in EXPECTED_RETRIEVAL_METRICS},
        "agent": {key: agent[key] for key in EXPECTED_AGENT_METRICS},
        "failure_counts": failure_counts,
        "end_to_end_failures": list(failures),
    }


def _compare_metrics(
    label: str, actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    for key, want in expected.items():
        if key not in actual:
            raise PreservationError(f"{label}.{key} is missing from the artifact")
        got = actual[key]
        if isinstance(want, int) and not isinstance(want, bool):
            if got != want:
                raise PreservationError(f"{label}.{key} is {got!r}, expected {want!r}")
        elif round(float(got), METRIC_DECIMALS) != round(float(want), METRIC_DECIMALS):
            raise PreservationError(f"{label}.{key} is {got!r}, expected {want!r}")


def _execution_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    """Take retrieval and embedding parameters from the artifact, not from defaults."""

    configs: set[str] = set()
    embeddings: set[str] = set()
    for row in report.get("questions") or ():
        hybrid_config = row.get("hybrid_config") or {}
        configs.add(json.dumps(hybrid_config.get("config"), sort_keys=True))
        embeddings.add(json.dumps(hybrid_config.get("embedding"), sort_keys=True))
    if len(configs) != 1 or len(embeddings) != 1:
        raise PreservationError(
            "questions do not share one retrieval/embedding configuration"
        )
    return {
        "config": json.loads(configs.pop()),
        "embedding": json.loads(embeddings.pop()),
    }


def _scan_for_secrets(files: Mapping[str, Path]) -> None:
    hits: list[str] = []
    for name in ARTIFACT_FILES:
        text = files[name].read_text(encoding="utf-8", errors="replace")
        for label, pattern in SECRET_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match is None:
                continue
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{name}:{line} matched {label}")
    if hits:
        raise PreservationError(
            "credential-like strings found; artifact left unmodified: " + "; ".join(hits)
        )


def _git_identity() -> dict[str, Any]:
    """Record the frozen evaluation code, distinct from the preservation-time tree.

    Preservation adds new files, so the worktree is expected to be dirty here.
    What must hold is that HEAD is still the commit that produced the artifact.
    """

    commit = _git("rev-parse", "HEAD")
    tags = _git("tag", "--points-at", "HEAD").split()
    if commit != EXPECTED_CODE_COMMIT:
        raise PreservationError(
            f"HEAD is {commit}, but the artifact was produced by "
            f"{EXPECTED_CODE_COMMIT}; check out the frozen commit before preserving"
        )
    if EXPECTED_CODE_TAG not in tags:
        raise PreservationError(
            f"tag {EXPECTED_CODE_TAG} does not point at HEAD; refusing to record "
            "an unverifiable code identity"
        )
    return {
        "evaluation_commit": commit,
        "evaluation_tag": EXPECTED_CODE_TAG,
        "tags_at_head": tags,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "preservation_worktree_clean": _git("status", "--porcelain") == "",
    }


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise PreservationError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _require_artifact_files(artifact_dir: Path) -> dict[str, Path]:
    if not artifact_dir.is_dir():
        raise PreservationError(f"artifact directory not found: {artifact_dir}")
    files = {name: artifact_dir / name for name in ARTIFACT_FILES}
    missing = sorted(name for name, path in files.items() if not path.is_file())
    if missing:
        raise PreservationError("missing artifact files: " + ", ".join(missing))
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_checksums(digests: Mapping[str, str]) -> str:
    return "".join(f"{digests[name]}  {name}\n" for name in ARTIFACT_FILES)


def _parse_checksums(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        if not name:
            raise PreservationError(f"malformed {CHECKSUM_NAME} line: {line!r}")
        entries[name.strip()] = digest.strip()
    return entries


def _mapping(report: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    value: Any = report
    for key in keys:
        if not isinstance(value, Mapping):
            raise PreservationError(f"{'.'.join(keys)} is missing from the artifact")
        value = value.get(key)
    if not isinstance(value, Mapping):
        raise PreservationError(f"{'.'.join(keys)} is missing from the artifact")
    return value


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
