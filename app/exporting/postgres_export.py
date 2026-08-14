"""Read-only Structural v2.1 to PostgreSQL COPY export pipeline."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Iterable

from app.parsing.chunking import REQUIRED_CHUNK_FIELDS
from app.parsing.sampling import load_manifest


EXPECTED_CHUNK_COUNT = 1_363_336
NULL = r"\N"
STATE_ORDER = (
    "structured_xml",
    "structured_html",
    "semi_structured",
    "fallback_unstructured",
)
STATE_PRIORITY = {name: index for index, name in enumerate(reversed(STATE_ORDER))}

CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    "companies": (
        "corp_code", "corp_name", "listed_name", "stock_code", "industry",
        "sector", "metadata",
    ),
    "disclosures": (
        "disclosure_id", "corp_code", "doc_group", "doc_subtype", "report_nm",
        "rcept_no", "rcept_dt", "is_correction", "base_year", "base_month",
        "file_format", "source_count", "structure_state", "metadata",
    ),
    "sections": (
        "section_id", "disclosure_id", "source_part_id", "source_section_id",
        "parent_section_id", "title", "section_level", "section_path", "content",
        "metadata",
    ),
    "disclosure_tables": (
        "table_id", "disclosure_id", "source_part_id", "source_table_id",
        "section_id", "row_count", "attributes", "table_rows", "metadata",
    ),
    "chunks": (
        "chunk_id", "disclosure_id", "source_part_id", "section_id", "table_id",
        "chunk_type", "chunk_order", "content", "retrieval_text", "char_count",
        "is_indexable", "projection_type", "metadata",
    ),
    "chunk_source_refs": (
        "source_ref_id", "chunk_id", "table_id", "source_table_id", "row_start",
        "row_end", "field_name", "ref_ordinal", "metadata",
    ),
}


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _csv_value(value: Any) -> Any:
    if value is None:
        return NULL
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _iso_date(value: Any) -> str | None:
    text = str(value or "").strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text or None


def _write_row(writer: csv.DictWriter, columns: Iterable[str], row: dict[str, Any]) -> None:
    writer.writerow({column: _csv_value(row.get(column)) for column in columns})


def _export_section_id(part_id: str, section_id: str) -> str:
    return f"{part_id}::section::{section_id}"


def _export_table_id(part_id: str, table_id: str) -> str:
    return f"{part_id}::table::{table_id}"


def _source_ref_id(
    chunk_id: str,
    table_id: str,
    row_start: int,
    row_end: int,
    field_name: str | None,
    ordinal: int,
) -> str:
    raw = _compact_json(
        [chunk_id, table_id, row_start, row_end, field_name, ordinal]
    ).encode("utf-8")
    return f"src_{hashlib.sha256(raw).hexdigest()[:24]}"


def classify_source(payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Classify only from serialized parser-output structure and source format."""
    source_format = str(payload.get("part", {}).get("source_format") or "").lower()
    sections = payload.get("sections") or []
    tables = payload.get("tables") or []
    warnings = payload.get("parser_warnings") or []
    reasons = [
        f"source_format={source_format or 'missing'}",
        f"section_count={len(sections)}",
        f"table_count={len(tables)}",
        f"parser_warning_count={len(warnings)}",
    ]
    if len(sections) == 1 and not tables:
        return "fallback_unstructured", [*reasons, "single parser fallback section and no table"]
    if warnings:
        return "semi_structured", [*reasons, "parser recovery warning present"]
    has_structure = bool(tables) or len(sections) > 1
    if has_structure and source_format == "xml":
        return "structured_xml", reasons
    if has_structure and source_format in {"html", "htm"}:
        return "structured_html", reasons
    if has_structure:
        return "semi_structured", [*reasons, "structure exists without an XML/HTML format label"]
    return "fallback_unstructured", [*reasons, "no serialized hierarchy or table structure"]


def _chunk_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "chunk_id", "doc_id", "section_id", "table_id", "chunk_type", "kind",
        "chunk_order", "content", "retrieval_text", "char_count", "is_indexable",
        "projection_type",
    }
    return {key: value for key, value in chunk.items() if key not in excluded}


def _reference_rows(
    part_id: str,
    chunk: dict[str, Any],
    table_row_counts: dict[str, int],
) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    collected: list[tuple[str, int, int, str | None, str]] = []
    field_refs = chunk.get("projection_field_refs") or {}
    for field_name, refs in field_refs.items():
        for ref in refs or []:
            collected.append(
                (
                    str(ref.get("table_id") or ""),
                    int(ref.get("row_start")),
                    int(ref.get("row_end")),
                    str(field_name),
                    "projection_field_ref",
                )
            )
    covered = {(table_id, start, end) for table_id, start, end, _, _ in collected}
    for ref in chunk.get("source_refs") or []:
        item = (
            str(ref.get("table_id") or ""),
            int(ref.get("row_start")),
            int(ref.get("row_end")),
        )
        if item not in covered:
            collected.append((*item, None, "projection_source_ref"))
            covered.add(item)
    if chunk.get("chunk_type") == "table" and not collected:
        local_table_id = str(chunk.get("table_id") or "")
        if local_table_id:
            collected.append(
                (
                    local_table_id,
                    int(chunk.get("row_start") or 0),
                    int(chunk.get("row_end") or 0),
                    None,
                    "table_chunk_range",
                )
            )
    deduplicated: list[tuple[str, int, int, str | None, str]] = []
    seen: set[tuple[str, int, int, str | None]] = set()
    for item in collected:
        key = item[:4]
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(item)
    chunk_id = str(chunk.get("chunk_id") or "")
    rows: list[dict[str, Any]] = []
    for ordinal, (source_table_id, row_start, row_end, field_name, ref_type) in enumerate(
        deduplicated, start=1
    ):
        if source_table_id not in table_row_counts:
            errors.append(f"{chunk_id}: orphan source table {source_table_id}")
            continue
        if row_start < 0 or row_end < row_start or row_end >= table_row_counts[source_table_id]:
            errors.append(
                f"{chunk_id}: invalid source row range "
                f"{source_table_id}[{row_start}:{row_end}]"
            )
            continue
        table_id = _export_table_id(part_id, source_table_id)
        rows.append(
            {
                "source_ref_id": _source_ref_id(
                    chunk_id, table_id, row_start, row_end, field_name, ordinal
                ),
                "chunk_id": chunk_id,
                "table_id": table_id,
                "source_table_id": source_table_id,
                "row_start": row_start,
                "row_end": row_end,
                "field_name": field_name,
                "ref_ordinal": ordinal,
                "metadata": _compact_json({"ref_type": ref_type}),
            }
        )
    if chunk.get("chunk_type") in {"table", "table_projection"} and not rows:
        errors.append(f"{chunk_id}: table/projection chunk has no source reference")
    projection_fields = chunk.get("projection_fields") or {}
    if projection_fields:
        missing_fields = sorted(set(projection_fields) - set(field_refs))
        if missing_fields:
            errors.append(f"{chunk_id}: projection fields without refs {missing_fields}")
    return rows, errors


def _empty_state_metrics() -> dict[str, Any]:
    return {
        "document_ids": set(),
        "source_count": 0,
        "section_count": 0,
        "table_count": 0,
        "text_chunk_count": 0,
        "table_chunk_count": 0,
        "projection_count": 0,
        "parser_warning_count": 0,
    }


def _public_metrics(metrics: dict[str, Any]) -> dict[str, int]:
    return {
        "document_count": len(metrics["document_ids"]),
        "source_count": metrics["source_count"],
        "section_count": metrics["section_count"],
        "table_count": metrics["table_count"],
        "text_chunk_count": metrics["text_chunk_count"],
        "table_chunk_count": metrics["table_chunk_count"],
        "projection_count": metrics["projection_count"],
        "parser_warning_count": metrics["parser_warning_count"],
    }


def export_structural_corpus(
    processed_dir: Path,
    manifest_path: Path,
    export_dir: Path,
    *,
    expected_chunk_count: int = EXPECTED_CHUNK_COUNT,
) -> dict[str, Any]:
    """Generate six CSVs atomically without changing processed outputs."""
    export_dir.mkdir(parents=True, exist_ok=True)
    report_dir = export_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(manifest_path)
    manifest_by_id = {str(row["doc_id"]): row for row in manifest}
    companies: dict[str, dict[str, Any]] = {}
    company_conflicts: list[str] = []
    for row in manifest:
        corp_code = str(row.get("corp_code") or "")
        candidate = {
            "corp_code": corp_code,
            "corp_name": row.get("corp_name"),
            "listed_name": row.get("listed_name"),
            "stock_code": str(row.get("stock_code") or ""),
            "industry": row.get("industry"),
            "sector": row.get("sector"),
        }
        existing = companies.get(corp_code)
        if existing and existing != candidate:
            company_conflicts.append(corp_code)
        else:
            companies[corp_code] = candidate

    index_records = [
        json.loads(line)
        for line in (processed_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    part_ids = [str(record["part_id"]) for record in index_records]
    duplicate_part_ids = len(part_ids) - len(set(part_ids))
    source_summaries: list[dict[str, Any]] = []
    sources_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_state_metrics = {state: _empty_state_metrics() for state in STATE_ORDER}
    document_state_metrics = {state: _empty_state_metrics() for state in STATE_ORDER}
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    issue_samples: dict[str, list[str]] = defaultdict(list)
    chunk_ids: set[str] = set()

    def record_issue(category: str, detail: str) -> None:
        issues[category] += 1
        if len(issue_samples[category]) < 20:
            issue_samples[category].append(detail)

    temporary_paths = {
        name: export_dir / f".{name}.csv.tmp" for name in CSV_COLUMNS
    }
    final_paths = {name: export_dir / f"{name}.csv" for name in CSV_COLUMNS}
    try:
        with ExitStack() as stack:
            writers: dict[str, csv.DictWriter] = {}
            for name, columns in CSV_COLUMNS.items():
                handle = stack.enter_context(
                    temporary_paths[name].open("w", encoding="utf-8", newline="")
                )
                writer = csv.DictWriter(
                    handle,
                    fieldnames=columns,
                    extrasaction="raise",
                    lineterminator="\n",
                    quoting=csv.QUOTE_MINIMAL,
                )
                writer.writeheader()
                writers[name] = writer

            for company in sorted(companies.values(), key=lambda item: item["corp_code"]):
                metadata = {"export_version": "structural-v2.1-pg-1"}
                _write_row(
                    writers["companies"],
                    CSV_COLUMNS["companies"],
                    {**company, "metadata": _compact_json(metadata)},
                )
                counts["companies"] += 1

            for record_index, record in enumerate(index_records, start=1):
                path = processed_dir / str(record["output_path"])
                with gzip.open(path, "rt", encoding="utf-8") as source:
                    payload = json.load(source)
                document = payload["document"]
                part = payload["part"]
                doc_id = str(document["doc_id"])
                part_id = str(part["part_id"])
                state, state_reasons = classify_source(payload)
                chunk_counts = Counter(
                    str(chunk.get("chunk_type") or chunk.get("kind") or "")
                    for chunk in payload.get("chunks") or []
                )
                source_summary = {
                    "doc_id": doc_id,
                    "part_id": part_id,
                    "source_path": part.get("source_path"),
                    "source_format": part.get("source_format"),
                    "is_primary": bool(part.get("is_primary")),
                    "structure_state": state,
                    "classification_evidence": state_reasons,
                    "section_count": len(payload.get("sections") or []),
                    "table_count": len(payload.get("tables") or []),
                    "text_chunk_count": chunk_counts["text"],
                    "table_chunk_count": chunk_counts["table"],
                    "projection_count": chunk_counts["table_projection"],
                    "parser_warning_count": len(payload.get("parser_warnings") or []),
                    "parser_warnings": payload.get("parser_warnings") or [],
                }
                source_summaries.append(source_summary)
                sources_by_doc[doc_id].append(source_summary)
                metrics = source_state_metrics[state]
                metrics["document_ids"].add(doc_id)
                for key in (
                    "section_count", "table_count", "text_chunk_count",
                    "table_chunk_count", "projection_count", "parser_warning_count",
                ):
                    metrics[key] += int(source_summary[key])
                metrics["source_count"] += 1
                if state in {"semi_structured", "fallback_unstructured"}:
                    examples[state].append(
                        {
                            **source_summary,
                            "corp_name": document.get("corp_name"),
                            "report_nm": document.get("report_nm"),
                        }
                    )

                section_ids = {
                    str(section["section_id"]) for section in payload.get("sections") or []
                }
                table_ids = {
                    str(table["table_id"]) for table in payload.get("tables") or []
                }
                table_row_counts = {
                    str(table["table_id"]): len(table.get("rows") or [])
                    for table in payload.get("tables") or []
                }
                if len(section_ids) != len(payload.get("sections") or []):
                    record_issue("duplicate_section_key", part_id)
                if len(table_ids) != len(payload.get("tables") or []):
                    record_issue("duplicate_table_key", part_id)
                for section in payload.get("sections") or []:
                    local_id = str(section["section_id"])
                    parent = section.get("parent_id")
                    if parent and str(parent) not in section_ids:
                        record_issue("orphan_section_parent", f"{part_id}:{local_id}")
                    metadata = {
                        "table_ids": section.get("table_ids") or [],
                        "content_order": section.get("content_order") or [],
                    }
                    _write_row(
                        writers["sections"],
                        CSV_COLUMNS["sections"],
                        {
                            "section_id": _export_section_id(part_id, local_id),
                            "disclosure_id": doc_id,
                            "source_part_id": part_id,
                            "source_section_id": local_id,
                            "parent_section_id": _export_section_id(part_id, str(parent)) if parent else None,
                            "title": section.get("title"),
                            "section_level": section.get("level"),
                            "section_path": _compact_json(section.get("path") or []),
                            "content": section.get("text") or "",
                            "metadata": _compact_json(metadata),
                        },
                    )
                    counts["sections"] += 1
                for table in payload.get("tables") or []:
                    local_table_id = str(table["table_id"])
                    local_section_id = str(table.get("section_id") or "")
                    if local_section_id not in section_ids:
                        record_issue("orphan_table_section", f"{part_id}:{local_table_id}")
                    table_rows = table.get("rows") or []
                    _write_row(
                        writers["disclosure_tables"],
                        CSV_COLUMNS["disclosure_tables"],
                        {
                            "table_id": _export_table_id(part_id, local_table_id),
                            "disclosure_id": doc_id,
                            "source_part_id": part_id,
                            "source_table_id": local_table_id,
                            "section_id": _export_section_id(part_id, local_section_id),
                            "row_count": len(table_rows),
                            "attributes": _compact_json(table.get("attributes") or {}),
                            "table_rows": _compact_json(table_rows),
                            "metadata": _compact_json(
                                {"section_path": table.get("section_path") or []}
                            ),
                        },
                    )
                    counts["disclosure_tables"] += 1
                for chunk in payload.get("chunks") or []:
                    chunk_id = str(chunk.get("chunk_id") or "")
                    if chunk_id in chunk_ids:
                        record_issue("duplicate_chunk_key", chunk_id)
                    chunk_ids.add(chunk_id)
                    missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in chunk]
                    if missing:
                        record_issue("missing_chunk_metadata", f"{chunk_id}: {missing}")
                    local_section_id = str(chunk.get("section_id") or "")
                    local_table_id = str(chunk.get("table_id") or "")
                    if local_section_id not in section_ids:
                        record_issue("orphan_chunk_section", chunk_id)
                    if local_table_id and local_table_id not in table_ids:
                        record_issue("orphan_chunk_table", chunk_id)
                    _write_row(
                        writers["chunks"],
                        CSV_COLUMNS["chunks"],
                        {
                            "chunk_id": chunk_id,
                            "disclosure_id": doc_id,
                            "source_part_id": part_id,
                            "section_id": _export_section_id(part_id, local_section_id),
                            "table_id": _export_table_id(part_id, local_table_id) if local_table_id else None,
                            "chunk_type": chunk.get("chunk_type") or chunk.get("kind"),
                            "chunk_order": chunk.get("chunk_order"),
                            "content": chunk.get("content") or "",
                            "retrieval_text": chunk.get("retrieval_text") or "",
                            "char_count": chunk.get("char_count"),
                            "is_indexable": bool(chunk.get("is_indexable", True)),
                            "projection_type": chunk.get("projection_type"),
                            "metadata": _compact_json(_chunk_metadata(chunk)),
                        },
                    )
                    counts["chunks"] += 1
                    ref_rows, ref_errors = _reference_rows(
                        part_id, chunk, table_row_counts
                    )
                    for error in ref_errors:
                        if "without refs" in error or "no source reference" in error:
                            category = "missing_source_reference"
                        elif "invalid source row range" in error:
                            category = "invalid_source_reference_range"
                        else:
                            category = "orphan_source_reference"
                        record_issue(category, error)
                    for ref_row in ref_rows:
                        _write_row(
                            writers["chunk_source_refs"],
                            CSV_COLUMNS["chunk_source_refs"],
                            ref_row,
                        )
                        counts["chunk_source_refs"] += 1
                if record_index % 100 == 0 or record_index == len(index_records):
                    print(
                        f"postgres export [{record_index}/{len(index_records)}] "
                        f"chunks={counts['chunks']} refs={counts['chunk_source_refs']}",
                        flush=True,
                    )

            for doc_id, row in sorted(manifest_by_id.items()):
                sources = sources_by_doc.get(doc_id, [])
                if not sources:
                    record_issue("missing_disclosure_source", doc_id)
                    continue
                primary = next((item for item in sources if item["is_primary"]), None)
                if primary is None:
                    primary = max(
                        sources,
                        key=lambda item: STATE_PRIORITY[item["structure_state"]],
                    )
                document_state = str(primary["structure_state"])
                for source in sources:
                    metrics = document_state_metrics[document_state]
                    metrics["document_ids"].add(doc_id)
                    metrics["source_count"] += 1
                    for key in (
                        "section_count", "table_count", "text_chunk_count",
                        "table_chunk_count", "projection_count", "parser_warning_count",
                    ):
                        metrics[key] += int(source[key])
                metadata = {
                    key: value
                    for key, value in row.items()
                    if key not in {
                        "doc_id", "corp_code", "doc_group", "doc_subtype", "report_nm",
                        "rcept_no", "rcept_dt", "is_correction", "base_year", "base_month",
                        "file_format",
                    }
                }
                metadata["source_parts"] = sources
                _write_row(
                    writers["disclosures"],
                    CSV_COLUMNS["disclosures"],
                    {
                        "disclosure_id": doc_id,
                        "corp_code": str(row.get("corp_code") or ""),
                        "doc_group": row.get("doc_group"),
                        "doc_subtype": row.get("doc_subtype"),
                        "report_nm": row.get("report_nm"),
                        "rcept_no": str(row.get("rcept_no") or ""),
                        "rcept_dt": _iso_date(row.get("rcept_dt")),
                        "is_correction": bool(row.get("is_correction")),
                        "base_year": row.get("base_year"),
                        "base_month": row.get("base_month"),
                        "file_format": row.get("file_format"),
                        "source_count": len(sources),
                        "structure_state": document_state,
                        "metadata": _compact_json(metadata),
                    },
                )
                counts["disclosures"] += 1
        for name, temporary in temporary_paths.items():
            temporary.replace(final_paths[name])
    finally:
        for temporary in temporary_paths.values():
            if temporary.exists():
                temporary.unlink()

    expected = {
        "companies": len(companies),
        "disclosures": len(manifest),
        "sections": sum(int(record["section_count"]) for record in index_records),
        "disclosure_tables": sum(int(record["table_count"]) for record in index_records),
        "chunks": expected_chunk_count,
    }
    for name, expected_count in expected.items():
        if counts[name] != expected_count:
            record_issue(
                "count_mismatch", f"{name}: expected {expected_count}, got {counts[name]}"
            )
    if company_conflicts:
        for corp_code in sorted(set(company_conflicts)):
            record_issue("company_metadata_conflict", corp_code)
    if duplicate_part_ids:
        record_issue("duplicate_part_key", str(duplicate_part_ids))

    structure_report = {
        "classification_basis": {
            "structured_xml": "source_format=xml and parser output has tables or multiple sections, with no parser warning",
            "structured_html": "source_format=html/htm and parser output has tables or multiple sections, with no parser warning",
            "semi_structured": "some hierarchy/table structure exists but parser recovery warnings exist or format label is unavailable",
            "fallback_unstructured": "single parser fallback section with no table, or no serialized hierarchy/table structure",
            "document_rule": "the primary source classification; strongest actual source state only when no primary source is marked",
        },
        "by_document_state": {
            state: _public_metrics(document_state_metrics[state]) for state in STATE_ORDER
        },
        "by_source_state": {
            state: _public_metrics(source_state_metrics[state]) for state in STATE_ORDER
        },
        "representative_examples": {
            state: sorted(
                examples[state],
                key=lambda item: (
                    -int(item["parser_warning_count"]),
                    int(item["table_count"]),
                    int(item["section_count"]),
                    str(item["part_id"]),
                ),
            )[:10]
            for state in ("semi_structured", "fallback_unstructured")
        },
        "source_details": source_summaries,
    }
    (report_dir / "structure_status.json").write_text(
        json.dumps(structure_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = [
        "# Parser-output structure status",
        "",
        "Classification uses only serialized source format, section/table counts, and parser warnings.",
        "",
        "| state | documents | sources | sections | tables | text chunks | table chunks | projections | parser warnings |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for state in STATE_ORDER:
        item = structure_report["by_document_state"][state]
        markdown.append(
            f"| {state} | {item['document_count']} | {item['source_count']} | "
            f"{item['section_count']} | {item['table_count']} | "
            f"{item['text_chunk_count']} | {item['table_chunk_count']} | "
            f"{item['projection_count']} | {item['parser_warning_count']} |"
        )
    markdown.extend(["", "## Semi/fallback representative sources", ""])
    for state in ("semi_structured", "fallback_unstructured"):
        state_examples = structure_report["representative_examples"][state]
        if not state_examples:
            markdown.append(f"- `{state}`: no source matched the serialized-output rule.")
            continue
        for example in state_examples:
            markdown.append(
                f"- `{state}`: {example['doc_id']} / {example['corp_name']} / "
                f"{example['report_nm']} — format={example['source_format']}, "
                f"sections={example['section_count']}, tables={example['table_count']}, "
                f"warnings={example['parser_warning_count']}"
            )
    (report_dir / "structure_status.md").write_text("\n".join(markdown), encoding="utf-8")
    validation = {
        "valid": not issues,
        "processed_output_modified": False,
        "expected_counts": expected,
        "export_counts": dict(sorted(counts.items())),
        "source_ref_rows": counts["chunk_source_refs"],
        "duplicate_key_count": sum(
            count for name, count in issues.items() if "duplicate" in name
        ),
        "orphan_fk_count": sum(count for name, count in issues.items() if "orphan" in name),
        "missing_source_reference_count": issues["missing_source_reference"],
        "missing_metadata_count": issues["missing_chunk_metadata"],
        "issue_count": sum(issues.values()),
        "issue_counts_by_reason": dict(sorted(issues.items())),
        "issue_samples_by_reason": dict(sorted(issue_samples.items())),
        "files": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "columns": list(CSV_COLUMNS[name]),
            }
            for name, path in final_paths.items()
        },
    }
    (report_dir / "export_validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (export_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "PostgreSQL COPY CSV",
                "encoding": "UTF8",
                "header": True,
                "null": NULL,
                "load_order": list(CSV_COLUMNS),
                "counts": dict(sorted(counts.items())),
                "files": {name: f"{name}.csv" for name in CSV_COLUMNS},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"structure": structure_report, "validation": validation}
