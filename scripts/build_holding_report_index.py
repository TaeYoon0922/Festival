"""Build the holding report index from the processed corpus.

The source is the processed corpus artifact, not the evaluation database.  The
database holds whatever has been ingested so far -- at the time of writing, 380
of 1,083 holding filings -- and an index built from part of a corpus cannot
answer "latest": the filing it never saw is exactly the one that would change
the answer.  The processed corpus carries one artifact per active holding
disclosure, so completeness can be *checked* rather than assumed, and this
script refuses to write an index that claims completeness it did not verify.

Read-only with respect to the corpus and the database; the only thing written
is the artifact itself.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.reasoning.holding_report_index import ARTIFACT_SCHEMA_VERSION, _clean
from app.reasoning.holding_correction_state import (
    document_correction_states,
    is_canonical_holding_body,
)
from app.reasoning.holding_reporter import canonical_reporter_key

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed/structural_v2_1_full_4204"
HOLDING_DOCS = PROCESSED / "documents/holding"
MANIFEST = ROOT / "data/corpus/manifest.jsonl"
DEFAULT_OUT = ROOT / "data/corpus/holding_report_index.json"

PROJECTION = "holding_report"
#: The projection's own field labels.  Read, never re-derived: the projection
#: layer already decided what each cell means.
F_REPORTER = "보고자/보유자"
F_REFERENCE = "기준일/보고일"
F_PREVIOUS_DATE = "직전 보고일"
F_BEFORE_SHARES = "직전 보유주식수"
F_BEFORE_RATIO = "직전 보유비율"
F_CHANGE_SHARES = "증감주식수"
F_CHANGE_RATIO = "증감비율"
F_AFTER_SHARES = "보유주식수"
F_AFTER_RATIO = "보유비율"


def _digits8(value) -> str | None:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def _direction(change_shares: str | None) -> str | None:
    """Direction is read from the sign the filing wrote, not inferred."""

    if not change_shares:
        return None
    compact = change_shares.replace(" ", "")
    if compact.startswith("-") or compact.startswith("△") or compact.startswith("▲"):
        return "decrease"
    stripped = compact.lstrip("+")
    if any(ch.isdigit() and ch != "0" for ch in stripped):
        return "increase"
    return None


def _projection_authority(chunks) -> dict[str, dict]:
    """Build-time structural authority metadata for each report projection."""

    projections = [
        chunk for chunk in chunks
        if chunk.get("projection_type") == PROJECTION
    ]
    states = document_correction_states(chunks)
    event_counts = collections.Counter(
        (
            str(chunk.get("corp_code") or ""),
            _digits8((chunk.get("projection_fields") or {}).get(F_REFERENCE)),
        )
        for chunk in projections
    )
    return {
        str(chunk.get("chunk_id") or ""): {
            "correction_state": states.get(str(chunk.get("chunk_id") or "")),
            "is_canonical_body": is_canonical_holding_body(chunk),
            "document_event_projection_count": event_counts[
                (
                    str(chunk.get("corp_code") or ""),
                    _digits8(
                        (chunk.get("projection_fields") or {}).get(F_REFERENCE)
                    ),
                )
            ],
        }
        for chunk in projections
        if str(chunk.get("chunk_id") or "")
    }


def manifest_holding_ids() -> set[str]:
    out = set()
    for line in MANIFEST.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if str(row.get("doc_group") or "") == "holding":
            out.add(str(row.get("doc_id") or ""))
    return out


def manifest_hash() -> str:
    digest = hashlib.sha256()
    with MANIFEST.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build() -> tuple[list[dict], dict]:
    expected = manifest_holding_ids()
    present = {p.name for p in HOLDING_DOCS.iterdir() if p.is_dir()}

    records: list[dict] = []
    report = collections.Counter()
    missing_projection: list[str] = []
    no_reporter: list[str] = []
    no_reference: list[str] = []
    chunking_versions: set[str] = set()
    provenance: set[str] = set()
    schema_versions: set[str] = set()

    for doc_id in sorted(present):
        files = list((HOLDING_DOCS / doc_id).glob("*.json.gz"))
        if not files:
            report["documents_missing_artifact"] += 1
            missing_projection.append(doc_id)
            continue
        doc = json.load(gzip.open(files[0], "rt", encoding="utf-8"))
        schema_versions.add(str(doc.get("schema_version") or ""))
        chunking_versions.add(str((doc.get("chunking") or {}).get("version") or ""))
        provenance.add(str(doc.get("projection_provenance_revision") or ""))

        projections = [c for c in (doc.get("chunks") or [])
                       if c.get("projection_type") == PROJECTION]
        if not projections:
            report["documents_without_report_projection"] += 1
            missing_projection.append(doc_id)
            continue
        report["documents_with_report_projection"] += 1
        # Read the filing-wide correction grid once.  Some projection captions
        # carry only a region reference (for example ``<내용 1-6>``), so their
        # state cannot be computed from the projection in isolation.
        projection_authority = _projection_authority(doc.get("chunks") or [])

        for chunk in projections:
            report["projections_total"] += 1
            fields = chunk.get("projection_fields") or {}
            raw_reporter = str(fields.get(F_REPORTER) or "").strip()
            reporter_key = canonical_reporter_key(raw_reporter)
            reference = _digits8(fields.get(F_REFERENCE))
            if not reporter_key:
                report["projections_without_reporter"] += 1
                no_reporter.append(str(chunk.get("chunk_id")))
                continue
            if not reference:
                report["projections_without_reference_date"] += 1
                no_reference.append(str(chunk.get("chunk_id")))
                continue
            change_shares = _clean(fields.get(F_CHANGE_SHARES))
            chunk_id = str(chunk.get("chunk_id") or "")
            authority = projection_authority.get(chunk_id) or {}
            correction_state = authority.get("correction_state")
            canonical_body = bool(authority.get("is_canonical_body"))
            event_projection_count = int(
                authority.get("document_event_projection_count") or 0
            )
            records.append({
                "issuer_corp_code": str(chunk.get("corp_code") or ""),
                "reporter_key": reporter_key,
                "raw_reporter": raw_reporter,
                "doc_id": str(chunk.get("doc_id") or ""),
                "projection_chunk_id": chunk_id,
                "reference_date": reference,
                "receipt_date": _digits8(chunk.get("rcept_dt")),
                "previous_date": _digits8(fields.get(F_PREVIOUS_DATE)),
                "before_shares": _clean(fields.get(F_BEFORE_SHARES)),
                "before_ratio": _clean(fields.get(F_BEFORE_RATIO)),
                "change_shares": change_shares,
                "change_ratio": _clean(fields.get(F_CHANGE_RATIO)),
                "change_direction": _direction(change_shares),
                "after_shares": _clean(fields.get(F_AFTER_SHARES)),
                "after_ratio": _clean(fields.get(F_AFTER_RATIO)),
                "is_correction": bool(chunk.get("is_correction")),
                "correction_state": correction_state,
                "is_canonical_body": canonical_body,
                "document_event_projection_count": event_projection_count,
                "report_nm": _clean(chunk.get("report_nm")),
                "source_table_id": _clean(chunk.get("source_table_id")),
                "source_refs": list(chunk.get("source_refs") or ()),
            })
            report["records_indexed"] += 1
            if correction_state:
                report[f"projection_state_{correction_state}"] += 1
            if canonical_body:
                report["canonical_body_projections"] += 1

    records.sort(key=lambda r: (r["issuer_corp_code"], r["reporter_key"],
                                r["reference_date"], r["doc_id"],
                                r["projection_chunk_id"]))

    covered = {r["doc_id"] for r in records}
    complete = (
        expected == present
        and not missing_projection
        and report["documents_missing_artifact"] == 0
    )
    stats = {
        "manifest_holding_documents": len(expected),
        "processed_holding_documents": len(present),
        "manifest_matches_processed": expected == present,
        "documents_missing_from_processed": sorted(expected - present)[:20],
        "documents_missing_from_manifest": sorted(present - expected)[:20],
        "documents_without_report_projection": missing_projection,
        "documents_represented_in_index": len(covered),
        "projections_without_reporter": no_reporter,
        "projections_without_reference_date": no_reference,
        "counters": dict(report),
        "complete": complete,
        "schema_versions": sorted(schema_versions),
        "chunking_versions": sorted(chunking_versions),
        "projection_provenance_revisions": sorted(provenance),
    }
    return records, stats


def _serialize(payload: dict) -> str:
    """One record per line, so a corpus change reads as a line diff.

    Indenting every field instead costs ~300 KiB over this corpus and makes a
    single changed report span twenty diff lines; one compact line per record
    is both smaller and easier to review.
    """

    head = {k: v for k, v in payload.items() if k != "records"}
    lines = [json.dumps(head, ensure_ascii=False, indent=1)[:-2].rstrip() + ",",
             ' "records": [']
    rows = [json.dumps(r, ensure_ascii=False, separators=(",", ":"))
            for r in payload["records"]]
    lines.append(",\n".join("  " + row for row in rows))
    lines.append(" ]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--dry-run", action="store_true",
                        help="report coverage without writing the artifact")
    args = parser.parse_args()

    records, stats = build()

    print("=== COVERAGE ===")
    for key in ("manifest_holding_documents", "processed_holding_documents",
                "manifest_matches_processed", "documents_represented_in_index",
                "complete"):
        print(f"  {key:38s} {stats[key]}")
    for key, value in stats["counters"].items():
        print(f"  {key:38s} {value}")
    for key in ("documents_without_report_projection",
                "projections_without_reporter",
                "projections_without_reference_date"):
        rows = stats[key]
        print(f"  {key:38s} {len(rows)}"
              + (f"  e.g. {rows[:3]}" if rows else ""))

    if not stats["complete"]:
        print("\nREFUSED: the source does not cover the active holding corpus.")
        return 2

    # Correction finality needs the frozen P0-A graph.  The generator does not
    # derive it, so an index built here declines correction-bearing timelines
    # until a proven finality source is wired in.
    correction_records = sum(1 for r in records if r["is_correction"])
    identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "corpus_snapshot_id": PROCESSED.name,
        "corpus_manifest_sha256": manifest_hash(),
        "processed_schema_versions": stats["schema_versions"],
        "chunking_versions": stats["chunking_versions"],
        "projection_provenance_revisions": stats["projection_provenance_revisions"],
        "source_holding_disclosure_count": stats["manifest_holding_documents"],
        "source_holding_report_projection_count":
            stats["counters"].get("projections_total", 0),
        "generated_commit": git_commit(),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    payload = {
        "identity": identity,
        "complete": True,
        "correction_finality_available": False,
        "correction_flagged_records": correction_records,
        "records": records,
    }
    blob = _serialize(payload)
    if args.dry_run:
        print(f"\nDRY RUN: {len(records)} records, "
              f"{len(blob.encode('utf-8'))/1024:.1f} KiB")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(blob, encoding="utf-8")
    print(f"\nwrote {out} -- {len(records)} records, "
          f"{out.stat().st_size/1024:.1f} KiB, "
          f"correction-flagged {correction_records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
