"""Validate every compressed output from the full-corpus parse."""

from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from app.parsing.sampling import resolve_unicode_path


def validate_full_output(
    output_dir: Path,
    corpus_dir: Path,
    max_chars: int = 1_200,
) -> dict[str, Any]:
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    with (output_dir / "index.jsonl").open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]

    errors: list[str] = []
    error_count = 0

    def add_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 100:
            errors.append(message)

    part_ids: set[str] = set()
    documents: set[str] = set()
    group_source_counts: Counter[str] = Counter()
    format_counts: Counter[str] = Counter()
    section_total = 0
    table_total = 0
    chunk_total = 0
    unique_chunk_total = 0
    max_chunk_size = 0
    warning_total = 0
    started = time.monotonic()

    for index, record in enumerate(records, start=1):
        part_id = str(record["part_id"])
        if part_id in part_ids:
            add_error(f"duplicate part ID: {part_id}")
        part_ids.add(part_id)
        documents.add(str(record["doc_id"]))
        group_source_counts[str(record["doc_group"])] += 1
        format_counts[str(record["source_format"])] += 1

        source_path = resolve_unicode_path(corpus_dir, str(record["source_path"]))
        if not source_path.is_file():
            add_error(f"{part_id}: source missing")
        output_path = output_dir / str(record["output_path"])
        if not output_path.is_file():
            add_error(f"{part_id}: output missing")
            continue

        try:
            with gzip.open(output_path, "rt", encoding="utf-8") as source:
                payload = json.load(source)
        except (OSError, EOFError, json.JSONDecodeError) as error:
            add_error(f"{part_id}: unreadable output: {error}")
            continue

        sections = payload.get("sections", [])
        tables = payload.get("tables", [])
        chunks = payload.get("chunks", [])
        warnings = payload.get("parser_warnings", [])
        section_ids = {section["section_id"] for section in sections}
        table_ids = {table["table_id"] for table in tables}
        local_chunk_ids: set[str] = set()

        if not sections:
            add_error(f"{part_id}: no sections")
        if not chunks:
            add_error(f"{part_id}: no chunks")
        if len(section_ids) != len(sections):
            add_error(f"{part_id}: duplicate section IDs")
        if len(table_ids) != len(tables):
            add_error(f"{part_id}: duplicate table IDs")

        for section in sections:
            parent_id = section.get("parent_id")
            if parent_id and parent_id not in section_ids:
                add_error(f"{part_id}: invalid parent {parent_id}")
            if not section.get("path"):
                add_error(f"{part_id}: empty section path")
        for table in tables:
            if table.get("section_id") not in section_ids or not table.get("rows"):
                add_error(f"{part_id}: invalid table {table.get('table_id')}")
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            content = str(chunk["content"])
            if chunk_id in local_chunk_ids:
                add_error(f"{part_id}: duplicate chunk ID {chunk_id}")
            local_chunk_ids.add(chunk_id)
            if not content.strip():
                add_error(f"{chunk_id}: empty content")
            if chunk.get("char_count") != len(content):
                add_error(f"{chunk_id}: invalid char_count")
            if len(content) > max_chars:
                add_error(f"{chunk_id}: exceeds {max_chars} characters")
            if chunk.get("section_id") not in section_ids or not chunk.get("section_path"):
                add_error(f"{chunk_id}: invalid section reference")
            if chunk.get("kind") == "table" and chunk.get("table_id") not in table_ids:
                add_error(f"{chunk_id}: invalid table reference")
            max_chunk_size = max(max_chunk_size, len(content))

        if warnings:
            add_error(f"{part_id}: parser warnings={len(warnings)}")
        section_total += len(sections)
        table_total += len(tables)
        chunk_total += len(chunks)
        unique_chunk_total += len(local_chunk_ids)
        warning_total += len(warnings)

        if index % 250 == 0 or index == len(records):
            elapsed = max(time.monotonic() - started, 0.001)
            print(
                f"validate [{index}/{len(records)}] "
                f"rate={index / elapsed:.1f}/s errors={error_count}",
                flush=True,
            )

    expected_pairs = {
        "source_count": len(records),
        "processed_document_count": len(documents),
        "section_count": section_total,
        "table_count": table_total,
        "chunk_count": chunk_total,
        "warning_count": warning_total,
    }
    for key, actual in expected_pairs.items():
        if summary.get(key) != actual:
            add_error(f"summary mismatch for {key}: {summary.get(key)} != {actual}")
    if dict(group_source_counts) != summary.get("group_source_counts"):
        add_error("summary mismatch for group_source_counts")
    if format_counts["xml"] != summary.get("xml_source_count"):
        add_error("summary mismatch for xml_source_count")
    if format_counts["html"] != summary.get("html_source_count"):
        add_error("summary mismatch for html_source_count")

    report = {
        "valid": error_count == 0,
        "document_count": len(documents),
        "source_count": len(records),
        "format_counts": dict(format_counts),
        "group_source_counts": dict(group_source_counts),
        "section_count": section_total,
        "table_count": table_total,
        "chunk_count": chunk_total,
        "unique_chunk_count": unique_chunk_total,
        "max_chunk_chars": max_chunk_size,
        "warning_count": warning_total,
        "error_count": error_count,
        "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    if error_count:
        raise ValueError(json.dumps(report, ensure_ascii=False, indent=2))
    return report
