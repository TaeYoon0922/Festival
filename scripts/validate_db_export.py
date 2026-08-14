"""Full logical-record, key, FK, metadata, and provenance validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NULL = r"\N"
EXPECTED = {
    "companies": 70,
    "disclosures": 4_204,
    "sections": 147_399,
    "disclosure_tables": 1_216_982,
    "chunks": 1_363_336,
}


def _rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"missing CSV header: {path}")
        for number, row in enumerate(reader, start=2):
            yield number, row


def validate(export_dir: Path) -> dict[str, Any]:
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    optional_field_counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)

    def issue(category: str, detail: str) -> None:
        issues[category] += 1
        if len(samples[category]) < 20:
            samples[category].append(detail)

    def check_shape(entity: str, number: int, row: dict[str, Any]) -> None:
        if None in row:
            issue("csv_escape_or_column_error", f"{entity}:{number}: extra fields")
        expected_columns = set(manifest["columns"][entity])
        if set(row) != expected_columns:
            issue("csv_header_mismatch", entity)

    company_ids: set[str] = set()
    for number, row in _rows(export_dir / "companies.csv"):
        check_shape("companies", number, row)
        corp_code = row["corp_code"]
        if corp_code in company_ids:
            issue("duplicate_company_pk", corp_code)
        company_ids.add(corp_code)
        if not corp_code or not row["stock_code"] or row["market"] != NULL:
            if not corp_code or not row["stock_code"]:
                issue("metadata_required_field_missing", f"companies:{number}")
            if row["market"] != NULL:
                issue("inferred_market_value", f"companies:{number}")
        if row["market"] != NULL:
            optional_field_counts["market_non_null"] += 1
        counts["companies"] += 1
    print(f"validate companies={counts['companies']}", flush=True)

    doc_ids: set[str] = set()
    for number, row in _rows(export_dir / "disclosures.csv"):
        check_shape("disclosures", number, row)
        doc_id = row["doc_id"]
        if doc_id in doc_ids:
            issue("duplicate_disclosure_pk", doc_id)
        doc_ids.add(doc_id)
        if row["corp_code"] not in company_ids:
            issue("orphan_disclosure_company", doc_id)
        for field in (
            "doc_id", "corp_code", "rcept_no", "report_nm", "doc_group",
            "file_path", "file_format", "metadata",
        ):
            if row[field] in {"", NULL}:
                issue("metadata_required_field_missing", f"disclosures:{number}:{field}")
        try:
            json.loads(row["metadata"])
        except json.JSONDecodeError:
            issue("invalid_json", f"disclosures:{number}:metadata")
        counts["disclosures"] += 1
    print(f"validate disclosures={counts['disclosures']}", flush=True)

    section_ids: set[tuple[str, str]] = set()
    parents: list[tuple[str, str, str]] = []
    for number, row in _rows(export_dir / "sections.csv"):
        check_shape("sections", number, row)
        key = (row["source_part_id"], row["section_id"])
        if key in section_ids:
            issue("duplicate_section_pk", repr(key))
        section_ids.add(key)
        if row["doc_id"] not in doc_ids:
            issue("orphan_section_disclosure", repr(key))
        if row["parent_section_id"] != NULL:
            parents.append((row["source_part_id"], row["parent_section_id"], repr(key)))
        for field in (
            "source_part_id", "section_id", "doc_id", "section_title",
            "section_path", "section_order", "section_depth", "content", "metadata",
        ):
            if row[field] == NULL:
                issue("metadata_required_field_missing", f"sections:{number}:{field}")
        try:
            json.loads(row["section_path"])
            json.loads(row["metadata"])
        except json.JSONDecodeError:
            issue("invalid_json", f"sections:{number}")
        counts["sections"] += 1
    for source_part_id, parent_id, child in parents:
        if (source_part_id, parent_id) not in section_ids:
            issue("orphan_section_parent", child)
    del parents
    print(f"validate sections={counts['sections']}", flush=True)

    table_rows: dict[tuple[str, str], int] = {}
    for number, row in _rows(export_dir / "disclosure_tables.csv"):
        check_shape("disclosure_tables", number, row)
        key = (row["source_part_id"], row["table_id"])
        if key in table_rows:
            issue("duplicate_table_pk", repr(key))
        try:
            row_count = int(row["row_count"])
        except ValueError:
            row_count = -1
            issue("metadata_required_field_missing", f"tables:{number}:row_count")
        table_rows[key] = row_count
        if row["doc_id"] not in doc_ids:
            issue("orphan_table_disclosure", repr(key))
        if (row["source_part_id"], row["section_id"]) not in section_ids:
            issue("orphan_table_section", repr(key))
        for field in (
            "source_part_id", "table_id", "doc_id", "section_id", "table_order",
            "source_path", "row_count", "attributes", "table_rows", "metadata",
        ):
            if row[field] in {"", NULL}:
                issue("metadata_required_field_missing", f"tables:{number}:{field}")
        try:
            json.loads(row["attributes"])
            json.loads(row["metadata"])
        except json.JSONDecodeError:
            issue("invalid_json", f"tables:{number}")
        counts["disclosure_tables"] += 1
    print(f"validate tables={counts['disclosure_tables']}", flush=True)

    chunk_ids: set[str] = set()
    requiring_refs: set[str] = set()
    required_field_refs: set[tuple[str, str]] = set()
    for number, row in _rows(export_dir / "chunks.csv"):
        check_shape("chunks", number, row)
        chunk_id = row["chunk_id"]
        if chunk_id in chunk_ids:
            issue("duplicate_chunk_pk", chunk_id)
        chunk_ids.add(chunk_id)
        if row["doc_id"] not in doc_ids:
            issue("orphan_chunk_disclosure", chunk_id)
        if (row["source_part_id"], row["section_id"]) not in section_ids:
            issue("orphan_chunk_section", chunk_id)
        if row["table_id"] != NULL and (
            row["source_part_id"], row["table_id"]
        ) not in table_rows:
            issue("orphan_chunk_table", chunk_id)
        for field in (
            "chunk_id", "doc_id", "source_part_id", "section_id", "chunk_type",
            "chunk_order", "content", "retrieval_text", "char_count", "metadata",
        ):
            if row[field] == NULL:
                issue("metadata_required_field_missing", f"chunks:{number}:{field}")
        try:
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError:
            metadata = {}
            issue("invalid_json", f"chunks:{number}:metadata")
        if "parser_mode" in metadata:
            optional_field_counts["parser_mode"] += 1
        if "structure_type" in metadata:
            optional_field_counts["structure_type"] += 1
        if row["retrieval_priority"] != NULL:
            optional_field_counts["retrieval_priority_non_null"] += 1
        if row["chunk_type"] in {"table", "table_projection"}:
            requiring_refs.add(chunk_id)
        if row["chunk_type"] == "table_projection":
            counts["projections"] += 1
            for field_name in (metadata.get("projection_fields") or {}):
                required_field_refs.add((chunk_id, str(field_name)))
        counts["chunks"] += 1
    print(
        f"validate chunks={counts['chunks']} projections={counts['projections']}",
        flush=True,
    )

    ref_ids: set[str] = set()
    for number, row in _rows(export_dir / "chunk_source_refs.csv"):
        check_shape("chunk_source_refs", number, row)
        ref_id = row["source_ref_id"]
        if ref_id in ref_ids:
            issue("duplicate_source_ref_pk", ref_id)
        ref_ids.add(ref_id)
        chunk_id = row["chunk_id"]
        if chunk_id not in chunk_ids:
            issue("orphan_source_ref_chunk", ref_id)
        requiring_refs.discard(chunk_id)
        table_key = (row["source_part_id"], row["table_id"])
        if table_key not in table_rows:
            issue("orphan_source_ref_table", ref_id)
        else:
            try:
                start = int(row["row_start"])
                end = int(row["row_end"])
                if start < 0 or end < start or end >= table_rows[table_key]:
                    issue("invalid_source_ref_range", ref_id)
            except ValueError:
                issue("invalid_source_ref_range", ref_id)
        if row["field_name"] != NULL:
            required_field_refs.discard((chunk_id, row["field_name"]))
        for field in (
            "source_ref_id", "chunk_id", "source_type", "source_part_id", "table_id",
            "row_start", "row_end", "source_ref", "metadata",
        ):
            if row[field] in {"", NULL}:
                issue("metadata_required_field_missing", f"refs:{number}:{field}")
        try:
            json.loads(row["source_ref"])
            json.loads(row["metadata"])
        except json.JSONDecodeError:
            issue("invalid_json", f"refs:{number}")
        counts["chunk_source_refs"] += 1
    if requiring_refs:
        issue("missing_source_ref", str(len(requiring_refs)))
    if required_field_refs:
        issue("missing_projection_field_ref", str(len(required_field_refs)))
    print(f"validate refs={counts['chunk_source_refs']}", flush=True)

    for entity, expected in EXPECTED.items():
        if counts[entity] != expected:
            issue("count_mismatch", f"{entity}: {counts[entity]} != {expected}")
    if counts["projections"] != 62_243:
        issue("projection_count_mismatch", str(counts["projections"]))
    file_sizes = {
        name: (export_dir / manifest["files"][name]).stat().st_size
        for name in manifest["load_order"]
    }
    report = {
        "valid": not issues,
        "counts": dict(sorted(counts.items())),
        "expected_counts": EXPECTED,
        "actual_optional_field_counts": {
            "market_non_null": optional_field_counts["market_non_null"],
            "parser_mode": optional_field_counts["parser_mode"],
            "structure_type": optional_field_counts["structure_type"],
            "retrieval_priority_non_null": optional_field_counts[
                "retrieval_priority_non_null"
            ],
        },
        "duplicate_pk_count": sum(
            value for key, value in issues.items() if "duplicate" in key
        ),
        "orphan_fk_count": sum(value for key, value in issues.items() if "orphan" in key),
        "missing_source_ref_count": issues["missing_source_ref"],
        "missing_projection_field_ref_count": issues["missing_projection_field_ref"],
        "metadata_required_field_missing_count": issues["metadata_required_field_missing"],
        "csv_escape_newline_quote_error_count": issues["csv_escape_or_column_error"],
        "issue_count": sum(issues.values()),
        "issue_counts_by_reason": dict(sorted(issues.items())),
        "issue_samples_by_reason": dict(sorted(samples.items())),
        "file_sizes_bytes": file_sizes,
        "total_file_size_bytes": sum(file_sizes.values()),
    }
    (export_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir", type=Path, default=PROJECT_ROOT / "data" / "db_export"
    )
    args = parser.parse_args()
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    report = validate(args.export_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
