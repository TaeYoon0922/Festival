"""Repair provenance on serialized table chunks without parsing or rechunking."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


REPORT_DIR_NAME = "table_provenance_repair"


def _payload_invariants(payload: dict[str, Any]) -> dict[str, Any]:
    chunk_ids = [
        str(chunk.get("chunk_id") or "") for chunk in payload.get("chunks", [])
    ]
    protected = hashlib.sha256()
    for chunk in payload.get("chunks", []):
        embedding_fields = {
            key: value
            for key, value in chunk.items()
            if "embedding" in str(key).casefold()
        }
        encoded = json.dumps(
            [
                chunk.get("chunk_id"),
                chunk.get("content"),
                chunk.get("retrieval_text"),
                embedding_fields,
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        protected.update(len(encoded).to_bytes(8, "big"))
        protected.update(encoded)
    return {
        "chunk_count": len(chunk_ids),
        "chunk_id_set": set(chunk_ids),
        "protected_digest": protected.hexdigest(),
    }


def _invariants_match(
    before: dict[str, Any], after: dict[str, Any]
) -> tuple[bool, bool]:
    chunk_ids_unchanged = bool(
        before["chunk_count"] == after["chunk_count"]
        and before["chunk_id_set"] == after["chunk_id_set"]
    )
    protected_fields_unchanged = bool(
        before["protected_digest"] == after["protected_digest"]
    )
    return chunk_ids_unchanged, protected_fields_unchanged


def _failure(
    path: Path,
    chunk: dict[str, Any] | None,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "source_path": str(path),
        "chunk_id": str((chunk or {}).get("chunk_id") or ""),
        "reason": reason,
        "detail": detail,
    }


def _repair_payload(
    payload: dict[str, Any],
    *,
    source_path: Path,
    apply: bool,
) -> tuple[Counter[str], list[dict[str, Any]]]:
    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    table_row_counts = {
        str(table.get("table_id") or ""): len(table.get("rows") or [])
        for table in payload.get("tables", [])
        if table.get("table_id")
    }

    for chunk in payload.get("chunks", []):
        counts["scanned_chunk_count"] += 1
        if chunk.get("chunk_type") != "table":
            continue
        counts["table_chunk_count"] += 1
        if chunk.get("source_refs"):
            counts["already_provenanced_chunk_count"] += 1
            continue
        counts["repair_candidate_count"] += 1

        table_id = str(chunk.get("table_id") or "").strip()
        row_start = chunk.get("row_start")
        row_end = chunk.get("row_end")
        if not table_id:
            failures.append(
                _failure(source_path, chunk, "missing_table_id", "table_id is empty")
            )
            continue
        if table_id not in table_row_counts:
            failures.append(
                _failure(
                    source_path,
                    chunk,
                    "orphan_table_id",
                    f"table_id {table_id!r} is not present in payload.tables",
                )
            )
            continue
        if (
            not isinstance(row_start, int)
            or isinstance(row_start, bool)
            or not isinstance(row_end, int)
            or isinstance(row_end, bool)
        ):
            failures.append(
                _failure(
                    source_path,
                    chunk,
                    "invalid_row_range_type",
                    f"row_start={row_start!r}, row_end={row_end!r}",
                )
            )
            continue
        if (
            row_start < 0
            or row_end < row_start
            or row_end >= table_row_counts[table_id]
        ):
            failures.append(
                _failure(
                    source_path,
                    chunk,
                    "invalid_row_range",
                    (
                        f"table_id={table_id}, row_start={row_start}, "
                        f"row_end={row_end}, row_count={table_row_counts[table_id]}"
                    ),
                )
            )
            continue

        if apply:
            source_ref = {
                "table_id": table_id,
                "row_start": row_start,
                "row_end": row_end,
            }
            chunk["source_table_id"] = table_id
            chunk["source_table_ids"] = [table_id]
            chunk["source_refs"] = [source_ref]
        counts["repairable_chunk_count"] += 1

    return counts, failures


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        payload = json.load(source)
    if not isinstance(payload, dict):
        raise ValueError("serialized payload must be a JSON object")
    return payload


def _write_gzip_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    expected_invariants: dict[str, Any],
) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as target:
            json.dump(payload, target, ensure_ascii=False, separators=(",", ":"))
        written = _read_gzip_json(temporary)
        chunk_ids_unchanged, protected_fields_unchanged = _invariants_match(
            expected_invariants, _payload_invariants(written)
        )
        if not chunk_ids_unchanged:
            raise ValueError("atomic write validation changed the chunk_id set")
        if not protected_fields_unchanged:
            raise ValueError("atomic write validation changed protected chunk fields")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def repair_table_provenance(
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Backfill ordinary table provenance in existing serialized payloads only."""

    output_dir = Path(output_dir)
    documents_dir = output_dir / "documents"
    if not documents_dir.is_dir():
        raise FileNotFoundError(
            f"missing processed documents directory: {documents_dir}"
        )

    files = sorted(documents_dir.rglob("*.json.gz"))
    totals: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    candidate_sources = 0
    modified_sources = 0
    modified_chunks = 0
    all_chunk_id_sets_unchanged = True
    all_protected_fields_unchanged = True

    for index, path in enumerate(files, start=1):
        try:
            payload = _read_gzip_json(path)
            before = _payload_invariants(payload)
            counts, chunk_failures = _repair_payload(
                payload,
                source_path=path.relative_to(output_dir),
                apply=not dry_run,
            )
            failures.extend(chunk_failures)
            totals.update(counts)
            if counts["repairable_chunk_count"]:
                candidate_sources += 1

            after = _payload_invariants(payload)
            ids_unchanged, protected_unchanged = _invariants_match(before, after)
            all_chunk_id_sets_unchanged &= ids_unchanged
            all_protected_fields_unchanged &= protected_unchanged
            if not ids_unchanged or not protected_unchanged:
                failures.append(
                    _failure(
                        path.relative_to(output_dir),
                        None,
                        "payload_invariant_failure",
                        (
                            f"chunk_id_set_unchanged={ids_unchanged}, "
                            f"protected_fields_unchanged={protected_unchanged}"
                        ),
                    )
                )
                continue

            if counts["repairable_chunk_count"] and not dry_run:
                _write_gzip_atomic(path, payload, expected_invariants=before)
                modified_sources += 1
                modified_chunks += counts["repairable_chunk_count"]
        except (
            OSError,
            EOFError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            failures.append(
                _failure(
                    path.relative_to(output_dir),
                    None,
                    "source_processing_failure",
                    f"{type(error).__name__}: {error}",
                )
            )

        if index % 250 == 0 or index == len(files):
            print(
                f"table provenance repair [{index}/{len(files)}] "
                f"candidates={totals['repairable_chunk_count']} "
                f"failures={len(failures)} dry_run={dry_run}",
                flush=True,
            )

    report_dir = output_dir / REPORT_DIR_NAME
    failure_path = report_dir / "failed_chunks.json"
    report_path = report_dir / "repair_report.json"
    report = {
        "provenance_only": True,
        "full_parse_performed": False,
        "chunking_performed": False,
        "embedding_generation_performed": False,
        "dry_run": dry_run,
        "source_count": len(files),
        "candidate_source_count": candidate_sources,
        "modified_source_count": 0 if dry_run else modified_sources,
        "scanned_chunk_count": totals["scanned_chunk_count"],
        "table_chunk_count": totals["table_chunk_count"],
        "already_provenanced_chunk_count": totals[
            "already_provenanced_chunk_count"
        ],
        "repair_candidate_count": totals["repair_candidate_count"],
        "repairable_chunk_count": totals["repairable_chunk_count"],
        "modified_chunk_count": modified_chunks,
        "failed_chunk_count": len(failures),
        "chunk_id_set_unchanged": all_chunk_id_sets_unchanged,
        "protected_fields_unchanged": all_protected_fields_unchanged,
        "valid": bool(
            not failures
            and all_chunk_id_sets_unchanged
            and all_protected_fields_unchanged
        ),
        "failure_report": str(failure_path.relative_to(output_dir)),
    }
    _write_json_atomic(failure_path, failures)
    _write_json_atomic(report_path, report)
    return report
