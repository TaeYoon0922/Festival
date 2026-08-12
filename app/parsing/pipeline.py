"""Run the XML-to-section/table/chunk pilot pipeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.parsing.chunking import CHUNKING_VERSION, build_chunks
from app.parsing.dart_xml import parse_dart_document
from app.parsing.sampling import resolve_unicode_path, select_sample_documents


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_sample_pipeline(
    corpus_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    per_group: int = 5,
    max_chars: int = 1_500,
    overlap: int = 120,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    document_dir = output_dir / "documents"
    document_dir.mkdir(parents=True, exist_ok=True)

    selected = select_sample_documents(
        manifest_path=manifest_path,
        corpus_dir=corpus_dir,
        per_group=per_group,
    )
    selection_payload = [
        {
            key: row.get(key)
            for key in (
                "doc_id",
                "doc_group",
                "doc_subtype",
                "corp_code",
                "corp_name",
                "listed_name",
                "stock_code",
                "report_nm",
                "rcept_no",
                "rcept_dt",
                "is_correction",
                "base_year",
                "base_month",
                "file_format",
                "source_path",
                "source_size",
            )
        }
        for row in selected
    ]
    _write_json(output_dir / "selection.json", selection_payload)

    summaries: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        source_path = resolve_unicode_path(corpus_dir, str(row["source_path"]))
        print(
            f"[{index:02d}/{len(selected)}] {row['doc_group']} "
            f"{row['corp_name']} {row['report_nm']}"
        )
        parsed = parse_dart_document(source_path, fallback_title=str(row["report_nm"]))
        chunks = build_chunks(
            doc_id=str(row["doc_id"]),
            parsed=parsed,
            max_chars=max_chars,
            overlap=overlap,
            document_metadata=row,
            source_file=str(row["source_path"]),
        )
        section_map = parsed.section_map()
        document_metadata = {
            key: row.get(key)
            for key in (
                "doc_id",
                "doc_group",
                "doc_subtype",
                "corp_code",
                "corp_name",
                "listed_name",
                "stock_code",
                "industry",
                "sector",
                "report_nm",
                "rcept_no",
                "rcept_dt",
                "is_correction",
                "base_year",
                "base_month",
                "file_format",
                "file_path",
                "source_path",
                "source_size",
            )
        }
        payload = {
            "schema_version": "2.0",
            "chunking": {
                "version": CHUNKING_VERSION,
                "strategy": str(row.get("doc_group") or "default"),
                "target_chars": 1_200,
                "min_chars": 700,
                "max_chars": max_chars,
                "sentence_overlap_chars": overlap,
            },
            "document": {
                **document_metadata,
                "parsed_title": parsed.document_title,
            },
            "sections": [section.to_dict() for section in parsed.sections],
            "tables": [
                table.to_dict(section_map[table.section_id].path)
                for table in parsed.tables
            ],
            "chunks": chunks,
            "parser_warnings": parsed.parser_warnings,
        }
        output_path = document_dir / f"{row['doc_id']}.json"
        _write_json(output_path, payload)

        chunk_kinds = Counter(chunk["kind"] for chunk in chunks)
        summaries.append(
            {
                **document_metadata,
                "section_count": len(parsed.sections),
                "table_count": len(parsed.tables),
                "chunk_count": len(chunks),
                "text_chunk_count": chunk_kinds["text"],
                "table_chunk_count": chunk_kinds["table"],
                "warning_count": len(parsed.parser_warnings),
                "output_path": str(output_path.relative_to(output_dir)),
            }
        )

    group_counts = Counter(summary["doc_group"] for summary in summaries)
    report = {
        "document_count": len(summaries),
        "group_counts": dict(group_counts),
        "section_count": sum(item["section_count"] for item in summaries),
        "table_count": sum(item["table_count"] for item in summaries),
        "chunk_count": sum(item["chunk_count"] for item in summaries),
        "warning_count": sum(item["warning_count"] for item in summaries),
        "documents": summaries,
    }
    _write_json(output_dir / "summary.json", report)
    return report
