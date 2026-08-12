"""Run and report the legacy-vs-structural 20-document chunking pilot."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.parsing.chunking import (
    CHUNKING_VERSION,
    DOCUMENT_METADATA_FIELDS,
    REQUIRED_CHUNK_FIELDS,
    build_chunks,
    build_legacy_chunks,
)
from app.parsing.dart_xml import parse_dart_document
from app.parsing.sampling import resolve_unicode_path, select_sample_documents


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _chunk_type(chunk: dict[str, Any]) -> str:
    return str(chunk.get("chunk_type", chunk.get("kind", "unknown")))


def _missing_metadata(chunk: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    nullable = {"parent_section_id", "prev_chunk_id", "next_chunk_id"}
    for field in REQUIRED_CHUNK_FIELDS:
        if field not in chunk:
            missing.append(field)
        elif (
            field not in nullable
            and field not in DOCUMENT_METADATA_FIELDS
            and chunk[field] in (None, "")
        ):
            missing.append(field)
    return missing


def _aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = [chunk for record in records for chunk in record["chunks"]]
    lengths = [len(str(chunk.get("content", ""))) for chunk in chunks]
    type_counts = Counter(_chunk_type(chunk) for chunk in chunks)
    duplicate_count = 0
    seen: set[tuple[str, str, str]] = set()
    missing_field_count = 0
    chunks_with_missing = 0
    orphan_sections = 0
    orphan_tables = 0
    unindexed_sections = 0
    unindexed_tables = 0

    for record in records:
        doc_id = str(record["doc_id"])
        section_ids = {str(section["section_id"]) for section in record["sections"]}
        table_ids = {str(table["table_id"]) for table in record["tables"]}
        chunk_section_ids = {
            str(chunk.get("section_id")) for chunk in record["chunks"]
        }
        chunk_table_ids = {
            str(chunk.get("table_id"))
            for chunk in record["chunks"]
            if chunk.get("table_id")
        }
        orphan_sections += sum(
            str(chunk.get("section_id")) not in section_ids
            for chunk in record["chunks"]
        )
        orphan_tables += sum(
            _chunk_type(chunk) == "table"
            and str(chunk.get("table_id")) not in table_ids
            for chunk in record["chunks"]
        )
        for section in record["sections"]:
            has_source_content = bool(
                str(section.get("text", "")).strip() or section.get("table_ids")
            )
            if has_source_content and str(section["section_id"]) not in chunk_section_ids:
                unindexed_sections += 1
        for table in record["tables"]:
            if str(table["table_id"]) not in chunk_table_ids:
                unindexed_tables += 1

        for chunk in record["chunks"]:
            fingerprint = (
                doc_id,
                _chunk_type(chunk),
                tuple(chunk.get("section_path") or []),
                str(chunk.get("content", "")).strip(),
            )
            if fingerprint in seen:
                duplicate_count += 1
            else:
                seen.add(fingerprint)
            missing = _missing_metadata(chunk)
            missing_field_count += len(missing)
            chunks_with_missing += bool(missing)

    total = len(chunks)
    return {
        "document_count": len(records),
        "chunk_count": total,
        "average_chunks_per_document": round(total / len(records), 3)
        if records
        else 0,
        "average_chunk_chars": round(statistics.mean(lengths), 3) if lengths else 0,
        "median_chunk_chars": round(statistics.median(lengths), 3)
        if lengths
        else 0,
        "chunks_le_200": sum(length <= 200 for length in lengths),
        "chunks_le_200_ratio": round(
            sum(length <= 200 for length in lengths) / total, 6
        )
        if total
        else 0,
        "chunks_ge_1500": sum(length >= 1_500 for length in lengths),
        "chunks_ge_1500_ratio": round(
            sum(length >= 1_500 for length in lengths) / total, 6
        )
        if total
        else 0,
        "text_chunk_count": type_counts["text"],
        "text_chunk_ratio": round(type_counts["text"] / total, 6) if total else 0,
        "table_chunk_count": type_counts["table"],
        "table_chunk_ratio": round(type_counts["table"] / total, 6) if total else 0,
        "table_projection_count": type_counts["table_projection"],
        "table_projection_ratio": round(
            type_counts["table_projection"] / total, 6
        ) if total else 0,
        "duplicate_chunk_count": duplicate_count,
        "duplicate_ratio": round(duplicate_count / total, 6) if total else 0,
        "chunks_with_missing_metadata": chunks_with_missing,
        "metadata_missing_field_count": missing_field_count,
        "orphan_section_count": orphan_sections,
        "orphan_table_count": orphan_tables,
        "unindexed_source_section_count": unindexed_sections,
        "unindexed_source_table_count": unindexed_tables,
    }


def _metrics_by_group(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["doc_group"])].append(record)
    return {
        group: _aggregate_metrics(grouped[group])
        for group in ("periodic", "major", "exchange", "holding")
    }


def _representative_samples(
    records: list[dict[str, Any]], per_kind: int = 3
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    samples: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group in ("periodic", "major", "exchange", "holding"):
        group_chunks: list[dict[str, Any]] = []
        for record in records:
            if record["doc_group"] != group:
                continue
            for chunk in record["chunks"]:
                if chunk.get("is_indexable", True):
                    group_chunks.append(chunk)
        samples[group] = {}
        for kind in ("text", "table"):
            candidates = [
                chunk for chunk in group_chunks if _chunk_type(chunk) == kind
            ]
            candidates.sort(
                key=lambda chunk: (
                    abs(len(str(chunk.get("content", ""))) - 900),
                    str(chunk.get("chunk_id", "")),
                )
            )
            samples[group][kind] = [
                {
                    key: chunk.get(key)
                    for key in (
                        "chunk_id",
                        "doc_id",
                        "corp_name",
                        "report_nm",
                        "section_path",
                        "chunk_type",
                        "table_id",
                        "table_title",
                        "column_headers",
                        "unit",
                        "statement_scope",
                        "basis_period",
                        "row_start",
                        "row_end",
                        "char_count",
                        "retrieval_text",
                        "content",
                    )
                    if key in chunk
                }
                for chunk in candidates[:per_kind]
            ]
    return samples


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown_report(report: dict[str, Any]) -> str:
    legacy = report["legacy"]["overall"]
    structural = report["structural"]["overall"]
    rows = [
        ("전체 chunk 수", legacy["chunk_count"], structural["chunk_count"]),
        (
            "문서당 평균 chunk 수",
            legacy["average_chunks_per_document"],
            structural["average_chunks_per_document"],
        ),
        ("평균 chunk 길이", legacy["average_chunk_chars"], structural["average_chunk_chars"]),
        ("중앙 chunk 길이", legacy["median_chunk_chars"], structural["median_chunk_chars"]),
        (
            "200자 이하 비율",
            _pct(legacy["chunks_le_200_ratio"]),
            _pct(structural["chunks_le_200_ratio"]),
        ),
        (
            "1500자 이상 비율",
            _pct(legacy["chunks_ge_1500_ratio"]),
            _pct(structural["chunks_ge_1500_ratio"]),
        ),
        (
            "text / table",
            f"{legacy['text_chunk_count']} / {legacy['table_chunk_count']}",
            f"{structural['text_chunk_count']} / {structural['table_chunk_count']}",
        ),
        (
            "완전 중복 비율",
            _pct(legacy["duplicate_ratio"]),
            _pct(structural["duplicate_ratio"]),
        ),
        (
            "metadata 누락 필드 수",
            legacy["metadata_missing_field_count"],
            structural["metadata_missing_field_count"],
        ),
        (
            "orphan section / table",
            f"{legacy['orphan_section_count']} / {legacy['orphan_table_count']}",
            f"{structural['orphan_section_count']} / {structural['orphan_table_count']}",
        ),
    ]
    lines = [
        "# Structural Chunking 20-document Pilot",
        "",
        "동일한 periodic/major/exchange/holding 각 5건을 legacy 청커와 구조 청커로 비교했습니다.",
        "전체 4,204건 코퍼스는 재처리하지 않았습니다.",
        "",
        "## Overall comparison",
        "",
        "| 항목 | Legacy | Structural |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(f"| {name} | {before} | {after} |" for name, before, after in rows)
    lines.extend(["", "## By disclosure type", ""])
    for group in ("periodic", "major", "exchange", "holding"):
        old = report["legacy"]["by_group"][group]
        new = report["structural"]["by_group"][group]
        lines.extend(
            [
                f"### {group}",
                "",
                "| 항목 | Legacy | Structural |",
                "| --- | ---: | ---: |",
                f"| chunk 수 | {old['chunk_count']} | {new['chunk_count']} |",
                f"| 평균 길이 | {old['average_chunk_chars']} | {new['average_chunk_chars']} |",
                f"| 200자 이하 | {_pct(old['chunks_le_200_ratio'])} | {_pct(new['chunks_le_200_ratio'])} |",
                f"| 1500자 이상 | {_pct(old['chunks_ge_1500_ratio'])} | {_pct(new['chunks_ge_1500_ratio'])} |",
                f"| text / table | {old['text_chunk_count']} / {old['table_chunk_count']} | {new['text_chunk_count']} / {new['table_chunk_count']} |",
                "",
            ]
        )

    lines.extend(["## Structural representative samples", ""])
    for group, kinds in report["representative_samples"].items():
        lines.extend([f"### {group}", ""])
        for kind in ("text", "table"):
            chosen = kinds[kind]
            lines.append(f"#### {kind} ({len(chosen)} samples)")
            lines.append("")
            if not chosen:
                lines.extend(["생성된 검색 대상 chunk가 없습니다.", ""])
                continue
            for sample in chosen:
                lines.append(
                    f"- `{sample['chunk_id']}` · {sample.get('corp_name')} · "
                    f"{' > '.join(sample.get('section_path') or [])}"
                )
                lines.append("")
                content = str(sample.get("content", ""))
                if len(content) > 2_000:
                    content = content[:2_000] + "\n[... review JSON에서 전체 내용 확인]"
                lines.extend(["```text", content, "```", ""])
    return "\n".join(lines)


def run_chunking_pilot(
    corpus_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    per_group: int = 5,
) -> dict[str, Any]:
    """Parse only the validated 20 documents and compare both chunkers."""

    output_dir.mkdir(parents=True, exist_ok=True)
    document_dir = output_dir / "documents"
    document_dir.mkdir(parents=True, exist_ok=True)
    selected = select_sample_documents(
        manifest_path=manifest_path,
        corpus_dir=corpus_dir,
        per_group=per_group,
    )
    _write_json(output_dir / "selection.json", selected)

    legacy_records: list[dict[str, Any]] = []
    structural_records: list[dict[str, Any]] = []
    document_summaries: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        source_path = resolve_unicode_path(corpus_dir, str(row["source_path"]))
        print(
            f"[{index:02d}/{len(selected)}] {row['doc_group']} "
            f"{row['corp_name']} {row['report_nm']}",
            flush=True,
        )
        parsed = parse_dart_document(
            source_path, fallback_title=str(row["report_nm"])
        )
        legacy_chunks = build_legacy_chunks(str(row["doc_id"]), parsed)
        structural_chunks = build_chunks(
            str(row["doc_id"]),
            parsed,
            document_metadata=row,
            source_file=str(row["source_path"]),
        )
        section_map = parsed.section_map()
        sections = [section.to_dict() for section in parsed.sections]
        tables = [
            table.to_dict(section_map[table.section_id].path)
            for table in parsed.tables
        ]
        record_base = {
            "doc_id": row["doc_id"],
            "doc_group": row["doc_group"],
            "sections": sections,
            "tables": tables,
        }
        legacy_records.append({**record_base, "chunks": legacy_chunks})
        structural_records.append({**record_base, "chunks": structural_chunks})

        document_metadata = {
            key: row.get(key)
            for key in (
                *DOCUMENT_METADATA_FIELDS,
                "listed_name",
                "industry",
                "sector",
                "file_path",
                "file_format",
                "source_path",
                "source_size",
            )
        }
        payload = {
            "schema_version": "2.0",
            "chunking": {
                "version": CHUNKING_VERSION,
                "strategy": str(row["doc_group"]),
                "target_chars": 1_200,
                "min_chars": 700,
                "max_chars": 1_500,
                "sentence_overlap_chars": 120,
            },
            "document": {
                **document_metadata,
                "parsed_title": parsed.document_title,
            },
            "sections": sections,
            "tables": tables,
            "chunks": structural_chunks,
            "parser_warnings": parsed.parser_warnings,
        }
        _write_json(document_dir / f"{row['doc_id']}.json", payload)
        document_summaries.append(
            {
                "doc_id": row["doc_id"],
                "doc_group": row["doc_group"],
                "corp_name": row["corp_name"],
                "report_nm": row["report_nm"],
                "legacy_chunk_count": len(legacy_chunks),
                "structural_chunk_count": len(structural_chunks),
            }
        )

    report = {
        "pilot_document_count": len(selected),
        "group_counts": dict(Counter(row["doc_group"] for row in selected)),
        "chunking_version": CHUNKING_VERSION,
        "full_corpus_reprocessed": False,
        "legacy": {
            "overall": _aggregate_metrics(legacy_records),
            "by_group": _metrics_by_group(legacy_records),
        },
        "structural": {
            "overall": _aggregate_metrics(structural_records),
            "by_group": _metrics_by_group(structural_records),
        },
        "documents": document_summaries,
        "representative_samples": _representative_samples(structural_records),
    }
    _write_json(output_dir / "comparison.json", report)
    (output_dir / "comparison.md").write_text(
        _markdown_report(report), encoding="utf-8"
    )
    return report
