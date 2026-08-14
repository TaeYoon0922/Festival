"""Independently count physical COPY CSV records after export."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "exports" / "structural_v2_1_postgres",
    )
    args = parser.parse_args()
    manifest = json.loads((args.export_dir / "manifest.json").read_text(encoding="utf-8"))
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    counts: dict[str, int] = {}
    errors: list[str] = []
    traceable_chunks: set[str] = set()
    projection_fields: set[tuple[str, str]] = set()
    for name in manifest["load_order"]:
        path = args.export_dir / manifest["files"][name]
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            count = 0
            for count, row in enumerate(reader, start=1):
                if None in row:
                    errors.append(f"{name} row {count}: extra CSV fields")
                    break
                if name == "chunks" and row["chunk_type"] in {"table", "table_projection"}:
                    traceable_chunks.add(row["chunk_id"])
                    if row["chunk_type"] == "table_projection":
                        metadata = json.loads(row["metadata"])
                        for field_name in (metadata.get("projection_fields") or {}):
                            projection_fields.add((row["chunk_id"], str(field_name)))
                elif name == "chunk_source_refs":
                    traceable_chunks.discard(row["chunk_id"])
                    if row.get("field_name") not in {None, "", r"\N"}:
                        projection_fields.discard((row["chunk_id"], row["field_name"]))
            counts[name] = count
        expected = int(manifest["counts"][name])
        if count != expected:
            errors.append(f"{name}: expected {expected}, read {count}")
        print(f"CSV validate {name}: {count}", flush=True)
    if traceable_chunks:
        errors.append(
            f"table/projection chunks without source refs: {len(traceable_chunks)}"
        )
    if projection_fields:
        errors.append(
            f"projection fields without field-level refs: {len(projection_fields)}"
        )
    report = {
        "valid": not errors,
        "counts": counts,
        "expected_counts": manifest["counts"],
        "error_count": len(errors),
        "errors": errors[:100],
        "missing_table_or_projection_source_ref_count": len(traceable_chunks),
        "missing_projection_field_ref_count": len(projection_fields),
    }
    report_path = args.export_dir / "reports" / "csv_file_validation.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
