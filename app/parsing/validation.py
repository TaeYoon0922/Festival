"""Validation checks for the generated 20-document pilot."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.parsing.chunking import REQUIRED_CHUNK_FIELDS
from app.parsing.sampling import resolve_unicode_path


EXPECTED_GROUP_COUNTS = {
    "periodic": 5,
    "exchange": 5,
    "major": 5,
    "holding": 5,
}


def validate_sample_output(
    output_dir: Path,
    corpus_dir: Path,
    max_chars: int = 1_200,
) -> dict[str, Any]:
    errors: list[str] = []
    summary_path = output_dir / "summary.json"
    selection_path = output_dir / "selection.json"
    if not summary_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("summary.json and selection.json are required")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if len(selection) != 20:
        errors.append(f"selection count is {len(selection)}, expected 20")
    if any(item.get("is_correction") for item in selection):
        errors.append("selection contains a correction disclosure")
    if any(item.get("file_format") != "xml" for item in selection):
        errors.append("selection contains a non-XML disclosure")

    group_counts: Counter[str] = Counter()
    chunk_ids: set[str] = set()
    section_total = 0
    table_total = 0
    chunk_total = 0
    max_chunk_size = 0
    warning_total = 0

    for item in selection:
        doc_id = str(item["doc_id"])
        document_path = output_dir / "documents" / f"{doc_id}.json"
        if not document_path.is_file():
            errors.append(f"missing output: {document_path}")
            continue
        source_path = resolve_unicode_path(corpus_dir, str(item["source_path"]))
        if not source_path.is_file():
            errors.append(f"missing source XML: {source_path}")

        payload = json.loads(document_path.read_text(encoding="utf-8"))
        structural = payload.get("schema_version") == "2.0"
        document = payload["document"]
        group_counts[str(document["doc_group"])] += 1
        sections = payload["sections"]
        tables = payload["tables"]
        chunks = payload["chunks"]
        warnings = payload.get("parser_warnings", [])

        section_ids = {section["section_id"] for section in sections}
        table_ids = {table["table_id"] for table in tables}
        if len(section_ids) != len(sections):
            errors.append(f"{doc_id}: duplicate section IDs")
        if len(table_ids) != len(tables):
            errors.append(f"{doc_id}: duplicate table IDs")
        for section in sections:
            parent_id = section.get("parent_id")
            if parent_id and parent_id not in section_ids:
                errors.append(f"{doc_id}: invalid section parent {parent_id}")
            if not section.get("path"):
                errors.append(f"{doc_id}: empty section path")
            for event in section.get("content_order", []):
                if event.get("kind") == "table" and event.get("table_id") not in table_ids:
                    errors.append(f"{doc_id}: invalid table order reference")
        for table in tables:
            if table["section_id"] not in section_ids or not table["rows"]:
                errors.append(f"{doc_id}: invalid or empty table {table['table_id']}")
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = chunk["chunk_id"]
            content = chunk["content"]
            if chunk_id in chunk_ids:
                errors.append(f"duplicate chunk ID: {chunk_id}")
            chunk_ids.add(chunk_id)
            if not content.strip():
                errors.append(f"{chunk_id}: empty chunk")
            if chunk["char_count"] != len(content):
                errors.append(f"{chunk_id}: incorrect char_count")
            if not structural and len(content) > max_chars:
                errors.append(f"{chunk_id}: exceeds {max_chars} characters")
            if chunk["section_id"] not in section_ids or not chunk["section_path"]:
                errors.append(f"{chunk_id}: invalid section reference")
            chunk_type = chunk.get("chunk_type", chunk.get("kind"))
            if chunk_type == "table" and chunk.get("table_id") not in table_ids:
                errors.append(f"{chunk_id}: invalid table reference")
            if structural:
                missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in chunk]
                if missing:
                    errors.append(f"{chunk_id}: missing metadata {missing}")
                for field in (
                    "doc_id",
                    "corp_code",
                    "corp_name",
                    "stock_code",
                    "doc_group",
                    "doc_subtype",
                    "report_nm",
                    "rcept_no",
                    "rcept_dt",
                    "is_correction",
                    "base_year",
                    "base_month",
                ):
                    if chunk.get(field) != document.get(field):
                        errors.append(f"{chunk_id}: document metadata mismatch for {field}")
                expected_previous = chunks[chunk_index - 1]["chunk_id"] if chunk_index else None
                expected_next = (
                    chunks[chunk_index + 1]["chunk_id"]
                    if chunk_index + 1 < len(chunks)
                    else None
                )
                if chunk.get("prev_chunk_id") != expected_previous:
                    errors.append(f"{chunk_id}: invalid prev_chunk_id")
                if chunk.get("next_chunk_id") != expected_next:
                    errors.append(f"{chunk_id}: invalid next_chunk_id")
                if chunk_type == "table":
                    for field in (
                        "table_title",
                        "row_start",
                        "row_end",
                        "column_headers",
                    ):
                        if field not in chunk:
                            errors.append(f"{chunk_id}: missing table metadata {field}")
            max_chunk_size = max(max_chunk_size, len(content))

        section_total += len(sections)
        table_total += len(tables)
        chunk_total += len(chunks)
        warning_total += len(warnings)

    if dict(group_counts) != EXPECTED_GROUP_COUNTS:
        errors.append(
            f"group counts are {dict(group_counts)}, expected {EXPECTED_GROUP_COUNTS}"
        )
    if warning_total:
        errors.append(f"parser warnings found: {warning_total}")
    if chunk_total != summary.get("chunk_count"):
        errors.append("chunk total does not match summary.json")

    report = {
        "valid": not errors,
        "document_count": sum(group_counts.values()),
        "group_counts": dict(group_counts),
        "section_count": section_total,
        "table_count": table_total,
        "chunk_count": chunk_total,
        "unique_chunk_count": len(chunk_ids),
        "max_chunk_chars": max_chunk_size,
        "warning_count": warning_total,
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report
