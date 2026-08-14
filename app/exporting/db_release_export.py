"""Exact-schema PostgreSQL COPY export from frozen Structural v2.1 payloads."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

from app.exporting.postgres_export import NULL, _compact_json, _iso_date, classify_source
from app.parsing.chunking import REQUIRED_CHUNK_FIELDS
from app.parsing.sampling import load_manifest


EXPECTED_COUNTS = {
    "companies": 70,
    "disclosures": 4_204,
    "sections": 147_399,
    "disclosure_tables": 1_216_982,
    "chunks": 1_363_336,
}

COLUMNS: dict[str, tuple[str, ...]] = {
    "companies": (
        "corp_code", "stock_code", "corp_name", "listed_name", "market",
        "industry", "sector", "metadata",
    ),
    "disclosures": (
        "doc_id", "corp_code", "rcept_no", "report_nm", "rcept_dt", "doc_group",
        "doc_subtype", "is_correction", "base_year", "base_month", "file_path",
        "file_format", "metadata",
    ),
    "sections": (
        "source_part_id", "section_id", "doc_id", "parent_section_id",
        "section_title", "section_path", "section_order", "section_depth", "content",
        "metadata",
    ),
    "disclosure_tables": (
        "source_part_id", "table_id", "doc_id", "section_id", "table_title",
        "table_order", "source_path", "row_count", "attributes", "table_rows",
        "metadata",
    ),
    "chunks": (
        "chunk_id", "doc_id", "source_part_id", "section_id", "table_id",
        "chunk_type", "chunk_order", "content", "retrieval_text", "char_count",
        "retrieval_priority", "metadata",
    ),
    "chunk_source_refs": (
        "source_ref_id", "chunk_id", "source_type", "source_part_id", "table_id",
        "row_start", "row_end", "field_name", "source_ref", "metadata",
    ),
}


def _csv_value(value: Any) -> Any:
    if value is None:
        return NULL
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write(writer: csv.DictWriter, columns: Iterable[str], row: dict[str, Any]) -> None:
    writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _ref_id(
    chunk_id: str,
    source_type: str,
    source_part_id: str,
    table_id: str,
    row_start: int,
    row_end: int,
    field_name: str | None,
) -> str:
    identity = _compact_json(
        [
            chunk_id, source_type, source_part_id, table_id, row_start, row_end,
            field_name,
        ]
    ).encode("utf-8")
    return f"ref_{hashlib.sha256(identity).hexdigest()[:24]}"


def _chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    explicit = {
        "chunk_id", "doc_id", "section_id", "table_id", "chunk_type", "kind",
        "chunk_order", "content", "retrieval_text", "char_count",
        "retrieval_priority",
    }
    return {key: value for key, value in chunk.items() if key not in explicit}


def _source_ref_rows(
    chunk: dict[str, Any],
    source_part_id: str,
    table_row_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Flatten only source references explicitly present on the saved chunk."""
    chunk_id = str(chunk["chunk_id"])
    entries: list[tuple[str, str, int, int, str | None, dict[str, Any]]] = []
    field_refs = chunk.get("projection_field_refs") or {}
    for field_name, refs in field_refs.items():
        for raw_ref in refs or []:
            ref = dict(raw_ref)
            entries.append(
                (
                    "projection_field_ref",
                    str(ref.get("table_id") or ""),
                    int(ref.get("row_start")),
                    int(ref.get("row_end")),
                    str(field_name),
                    ref,
                )
            )
    covered_ranges = {
        (table_id, row_start, row_end)
        for _, table_id, row_start, row_end, _, _ in entries
    }
    for raw_ref in chunk.get("source_refs") or []:
        ref = dict(raw_ref)
        key = (
            str(ref.get("table_id") or ""),
            int(ref.get("row_start")),
            int(ref.get("row_end")),
        )
        if key in covered_ranges:
            continue
        entries.append(("projection_source_ref", *key, None, ref))
        covered_ranges.add(key)
    if chunk.get("chunk_type") == "table" and not entries:
        raw_ref = {
            "table_id": str(chunk.get("table_id") or ""),
            "row_start": int(chunk.get("row_start") or 0),
            "row_end": int(chunk.get("row_end") or 0),
        }
        entries.append(
            (
                "table_chunk_range",
                raw_ref["table_id"],
                raw_ref["row_start"],
                raw_ref["row_end"],
                None,
                raw_ref,
            )
        )
    rows: list[dict[str, Any]] = []
    errors: list[tuple[str, str]] = []
    seen: set[tuple[str, str, int, int, str | None]] = set()
    for source_type, table_id, row_start, row_end, field_name, raw_ref in entries:
        key = (source_type, table_id, row_start, row_end, field_name)
        if key in seen:
            continue
        seen.add(key)
        if table_id not in table_row_counts:
            errors.append(("orphan_source_ref", f"{chunk_id}: {table_id}"))
            continue
        if row_start < 0 or row_end < row_start or row_end >= table_row_counts[table_id]:
            errors.append(
                (
                    "invalid_source_ref_range",
                    f"{chunk_id}: {table_id}[{row_start}:{row_end}]",
                )
            )
            continue
        rows.append(
            {
                "source_ref_id": _ref_id(
                    chunk_id, source_type, source_part_id, table_id, row_start,
                    row_end, field_name,
                ),
                "chunk_id": chunk_id,
                "source_type": source_type,
                "source_part_id": source_part_id,
                "table_id": table_id,
                "row_start": row_start,
                "row_end": row_end,
                "field_name": field_name,
                "source_ref": _compact_json(raw_ref),
                "metadata": _compact_json({}),
            }
        )
    if chunk.get("chunk_type") in {"table", "table_projection"} and not rows:
        errors.append(("missing_source_ref", chunk_id))
    projection_fields = set((chunk.get("projection_fields") or {}).keys())
    for field_name in sorted(projection_fields - set(field_refs.keys())):
        errors.append(("missing_projection_field_ref", f"{chunk_id}: {field_name}"))
    return rows, errors


def export_db_release(
    processed_dir: Path,
    manifest_path: Path,
    export_dir: Path,
) -> dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    manifest_by_doc = {str(row["doc_id"]): row for row in manifest}
    index_records = [
        json.loads(line)
        for line in (processed_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    companies: dict[str, dict[str, Any]] = {}
    for row in manifest:
        corp_code = str(row.get("corp_code") or "")
        companies.setdefault(
            corp_code,
            {
                "corp_code": corp_code,
                "stock_code": str(row.get("stock_code") or ""),
                "corp_name": row.get("corp_name"),
                "listed_name": row.get("listed_name"),
                "market": None,
                "industry": row.get("industry"),
                "sector": row.get("sector"),
                "metadata": _compact_json(
                    {
                        "market_source": "unavailable_in_manifest_and_processed_output"
                    }
                ),
            },
        )
    counts: Counter[str] = Counter()
    chunk_type_counts: Counter[str] = Counter()
    source_type_counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    samples: dict[str, list[str]] = defaultdict(list)
    source_parts_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_chunks: set[str] = set()
    seen_refs: set[str] = set()

    def issue(category: str, detail: str) -> None:
        issues[category] += 1
        if len(samples[category]) < 20:
            samples[category].append(detail)

    temporary = {name: export_dir / f".{name}.csv.tmp" for name in COLUMNS}
    final = {name: export_dir / f"{name}.csv" for name in COLUMNS}
    try:
        with ExitStack() as stack:
            writers: dict[str, csv.DictWriter] = {}
            for name, columns in COLUMNS.items():
                handle = stack.enter_context(
                    temporary[name].open("w", encoding="utf-8", newline="")
                )
                writer = csv.DictWriter(
                    handle,
                    fieldnames=columns,
                    lineterminator="\n",
                    quoting=csv.QUOTE_MINIMAL,
                    extrasaction="raise",
                )
                writer.writeheader()
                writers[name] = writer
            for row in sorted(companies.values(), key=lambda item: item["corp_code"]):
                _write(writers["companies"], COLUMNS["companies"], row)
                counts["companies"] += 1

            for record_index, record in enumerate(index_records, start=1):
                with gzip.open(
                    processed_dir / record["output_path"], "rt", encoding="utf-8"
                ) as source:
                    payload = json.load(source)
                document = payload["document"]
                part = payload["part"]
                doc_id = str(document["doc_id"])
                part_id = str(part["part_id"])
                structure_state, evidence = classify_source(payload)
                source_parts_by_doc[doc_id].append(
                    {
                        **part,
                        "structure_state": structure_state,
                        "classification_evidence": evidence,
                        "parser_warnings": payload.get("parser_warnings") or [],
                    }
                )
                sections = payload.get("sections") or []
                tables = payload.get("tables") or []
                chunks = payload.get("chunks") or []
                section_ids = {str(row["section_id"]) for row in sections}
                table_row_counts = {
                    str(row["table_id"]): len(row.get("rows") or []) for row in tables
                }
                table_titles: dict[str, str] = {}
                for chunk in chunks:
                    table_id = str(chunk.get("table_id") or "")
                    title = str(chunk.get("table_title") or "")
                    if table_id and title:
                        existing = table_titles.get(table_id)
                        if existing and existing != title:
                            issue("conflicting_table_title", f"{part_id}:{table_id}")
                        else:
                            table_titles[table_id] = title
                if len(section_ids) != len(sections):
                    issue("duplicate_section_pk", part_id)
                if len(table_row_counts) != len(tables):
                    issue("duplicate_table_pk", part_id)
                for section_order, section in enumerate(sections):
                    section_id = str(section["section_id"])
                    parent = section.get("parent_id")
                    if parent and str(parent) not in section_ids:
                        issue("orphan_section_parent", f"{part_id}:{section_id}")
                    _write(
                        writers["sections"], COLUMNS["sections"],
                        {
                            "source_part_id": part_id,
                            "section_id": section_id,
                            "doc_id": doc_id,
                            "parent_section_id": str(parent) if parent else None,
                            "section_title": section.get("title"),
                            "section_path": _compact_json(section.get("path") or []),
                            "section_order": section_order,
                            "section_depth": section.get("level"),
                            "content": section.get("text") or "",
                            "metadata": _compact_json(
                                {
                                    "table_ids": section.get("table_ids") or [],
                                    "content_order": section.get("content_order") or [],
                                }
                            ),
                        },
                    )
                    counts["sections"] += 1
                for table_order, table in enumerate(tables):
                    table_id = str(table["table_id"])
                    section_id = str(table.get("section_id") or "")
                    if section_id not in section_ids:
                        issue("orphan_table_section", f"{part_id}:{table_id}")
                    _write(
                        writers["disclosure_tables"], COLUMNS["disclosure_tables"],
                        {
                            "source_part_id": part_id,
                            "table_id": table_id,
                            "doc_id": doc_id,
                            "section_id": section_id,
                            "table_title": table_titles.get(table_id),
                            "table_order": table_order,
                            "source_path": part.get("source_path"),
                            "row_count": len(table.get("rows") or []),
                            "attributes": _compact_json(table.get("attributes") or {}),
                            "table_rows": _compact_json(table.get("rows") or []),
                            "metadata": _compact_json(
                                {"section_path": table.get("section_path") or []}
                            ),
                        },
                    )
                    counts["disclosure_tables"] += 1
                for chunk in chunks:
                    chunk_id = str(chunk.get("chunk_id") or "")
                    chunk_type = str(chunk.get("chunk_type") or chunk.get("kind") or "")
                    if chunk_id in seen_chunks:
                        issue("duplicate_chunk_pk", chunk_id)
                    seen_chunks.add(chunk_id)
                    missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in chunk]
                    if missing or "content" not in chunk or "retrieval_text" not in chunk:
                        issue("metadata_required_field_missing", f"{chunk_id}: {missing}")
                    section_id = str(chunk.get("section_id") or "")
                    table_id = str(chunk.get("table_id") or "")
                    if section_id not in section_ids:
                        issue("orphan_chunk_section", chunk_id)
                    if table_id and table_id not in table_row_counts:
                        issue("orphan_chunk_table", chunk_id)
                    _write(
                        writers["chunks"], COLUMNS["chunks"],
                        {
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "source_part_id": part_id,
                            "section_id": section_id,
                            "table_id": table_id or None,
                            "chunk_type": chunk_type,
                            "chunk_order": chunk.get("chunk_order"),
                            "content": chunk.get("content"),
                            "retrieval_text": chunk.get("retrieval_text"),
                            "char_count": chunk.get("char_count"),
                            "retrieval_priority": chunk.get("retrieval_priority"),
                            "metadata": _compact_json(_chunk_metadata(chunk)),
                        },
                    )
                    counts["chunks"] += 1
                    chunk_type_counts[chunk_type] += 1
                    ref_rows, ref_errors = _source_ref_rows(
                        chunk, part_id, table_row_counts
                    )
                    for category, detail in ref_errors:
                        issue(category, detail)
                    for ref_row in ref_rows:
                        ref_id = str(ref_row["source_ref_id"])
                        if ref_id in seen_refs:
                            issue("duplicate_source_ref_pk", ref_id)
                        seen_refs.add(ref_id)
                        _write(
                            writers["chunk_source_refs"],
                            COLUMNS["chunk_source_refs"],
                            ref_row,
                        )
                        counts["chunk_source_refs"] += 1
                        source_type_counts[str(ref_row["source_type"])] += 1
                if record_index % 100 == 0 or record_index == len(index_records):
                    print(
                        f"db export [{record_index}/{len(index_records)}] "
                        f"chunks={counts['chunks']} refs={counts['chunk_source_refs']}",
                        flush=True,
                    )

            for doc_id, manifest_row in sorted(manifest_by_doc.items()):
                parts = source_parts_by_doc.get(doc_id) or []
                if not parts:
                    issue("orphan_disclosure_source", doc_id)
                metadata = {
                    "listed_name": manifest_row.get("listed_name"),
                    "stock_code": str(manifest_row.get("stock_code") or ""),
                    "industry": manifest_row.get("industry"),
                    "sector": manifest_row.get("sector"),
                    "flr_nm": manifest_row.get("flr_nm"),
                    "n_files": manifest_row.get("n_files"),
                    "source_parts": parts,
                    "parser_mode_present_in_output": False,
                    "structure_type_present_in_output": False,
                }
                _write(
                    writers["disclosures"], COLUMNS["disclosures"],
                    {
                        "doc_id": doc_id,
                        "corp_code": str(manifest_row.get("corp_code") or ""),
                        "rcept_no": str(manifest_row.get("rcept_no") or ""),
                        "report_nm": manifest_row.get("report_nm"),
                        "rcept_dt": _iso_date(manifest_row.get("rcept_dt")),
                        "doc_group": manifest_row.get("doc_group"),
                        "doc_subtype": manifest_row.get("doc_subtype"),
                        "is_correction": bool(manifest_row.get("is_correction")),
                        "base_year": manifest_row.get("base_year"),
                        "base_month": manifest_row.get("base_month"),
                        "file_path": manifest_row.get("file_path"),
                        "file_format": manifest_row.get("file_format"),
                        "metadata": _compact_json(metadata),
                    },
                )
                counts["disclosures"] += 1
        for name in COLUMNS:
            temporary[name].replace(final[name])
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink()

    for entity, expected in EXPECTED_COUNTS.items():
        if counts[entity] != expected:
            issue("count_mismatch", f"{entity}: {counts[entity]} != {expected}")
    if chunk_type_counts["table_projection"] != 62_243:
        issue(
            "projection_count_mismatch",
            f"{chunk_type_counts['table_projection']} != 62243",
        )
    file_sizes = {name: final[name].stat().st_size for name in COLUMNS}
    report = {
        "valid": not issues,
        "processed_output_modified": False,
        "actual_field_findings": {
            "market_present": False,
            "market_export": "NULL; no inference",
            "parser_mode_present": False,
            "structure_type_present": False,
            "retrieval_priority_present": True,
        },
        "expected_counts": EXPECTED_COUNTS,
        "counts": dict(sorted(counts.items())),
        "chunk_type_counts": dict(sorted(chunk_type_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "duplicate_pk_count": sum(
            count for category, count in issues.items() if "duplicate" in category
        ),
        "orphan_fk_count": sum(
            count for category, count in issues.items() if "orphan" in category
        ),
        "missing_source_ref_count": issues["missing_source_ref"],
        "missing_projection_field_ref_count": issues["missing_projection_field_ref"],
        "metadata_required_field_missing_count": issues["metadata_required_field_missing"],
        "issue_count": sum(issues.values()),
        "issue_counts_by_reason": dict(sorted(issues.items())),
        "issue_samples_by_reason": dict(sorted(samples.items())),
        "file_sizes_bytes": file_sizes,
        "total_file_size_bytes": sum(file_sizes.values()),
    }
    (export_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "PostgreSQL COPY CSV",
                "encoding": "UTF8",
                "header": True,
                "null": NULL,
                "load_order": list(COLUMNS),
                "columns": {key: list(value) for key, value in COLUMNS.items()},
                "counts": dict(sorted(counts.items())),
                "files": {name: f"{name}.csv" for name in COLUMNS},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
