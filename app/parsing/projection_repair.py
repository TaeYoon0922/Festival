"""In-place projection-only repair for frozen Structural Chunking v2.1 outputs."""

from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from app.parsing.chunking import (
    DOCUMENT_METADATA_FIELDS,
    REQUIRED_CHUNK_FIELDS,
    _chunk_identity,
    _holding_document_context,
    _logical_table,
    _normalized_key,
)
from app.parsing.models import Table, TableCell
from app.parsing.release_validation import _projection_error_category, _projection_errors


def _load_table(raw: dict[str, Any]) -> Table:
    return Table(
        table_id=str(raw["table_id"]),
        section_id=str(raw["section_id"]),
        attributes=dict(raw.get("attributes") or {}),
        rows=[
            [
                TableCell(
                    text=str(cell.get("text") or ""),
                    is_header=bool(cell.get("is_header", False)),
                    rowspan=int(cell.get("rowspan", 1)),
                    colspan=int(cell.get("colspan", 1)),
                    source_tag=cell.get("source_tag"),
                )
                for cell in row
            ]
            for row in raw.get("rows", [])
        ],
    )


def _row_ref(table_id: str, row_index: int) -> dict[str, Any]:
    return {"table_id": table_id, "row_start": row_index, "row_end": row_index}


def _unique_refs(field_refs: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return list(
        {
            (str(ref["table_id"]), int(ref["row_start"]), int(ref["row_end"])): ref
            for refs in field_refs.values()
            for ref in refs
        }.values()
    )


def _holding_report_field_refs(
    chunk: dict[str, Any], table_map: dict[str, Table], context: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    table_id = str(chunk["source_table_id"])
    logical = _logical_table(table_map[table_id])
    rows: dict[str, int] = {}
    start = int(chunk["source_row_start"])
    end = int(chunk["source_row_end"])
    for row_index in range(start, end + 1):
        row = logical.rows[row_index]
        label = _normalized_key(row[0]) if row else ""
        if "직전보고서" in label:
            rows["previous"] = row_index
        elif "이번보고서" in label:
            rows["current"] = row_index
        elif label in {"증감", "증감합계"}:
            rows["change"] = row_index
    mapping = {
        "보고자/보유자": "current",
        "기준일/보고일": "current",
        "보유주식수": "current",
        "보유비율": "current",
        "직전 보고일": "previous",
        "직전 보유주식수": "previous",
        "직전 보유비율": "previous",
        "증감주식수": "change",
        "증감비율": "change",
    }
    refs: dict[str, list[dict[str, Any]]] = {}
    fields = chunk.get("projection_fields") or {}
    for label, row_kind in mapping.items():
        if fields.get(label) and row_kind in rows:
            refs[label] = [_row_ref(table_id, rows[row_kind])]
    context_refs = context.get("source_refs_by_field", {})
    if fields.get("보유 목적") and context_refs.get("holding_purpose"):
        refs["보유 목적"] = [context_refs["holding_purpose"]]
    if fields.get("변동 사유") and context_refs.get("change_reason"):
        refs["변동 사유"] = [context_refs["change_reason"]]
    return refs


def _holding_detail_field_refs(
    chunk: dict[str, Any], context: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    native_ref = _row_ref(str(chunk["source_table_id"]), int(chunk["source_row_start"]))
    fields = chunk.get("projection_fields") or {}
    refs = {
        label: [native_ref]
        for label, value in fields.items()
        if value and label != "보유 목적"
    }
    purpose_ref = context.get("source_refs_by_field", {}).get("holding_purpose")
    if fields.get("보유 목적") and purpose_ref:
        refs["보유 목적"] = [purpose_ref]
    return refs


def _repair_payload(payload: dict[str, Any]) -> Counter[str]:
    changes: Counter[str] = Counter()
    tables = [_load_table(raw) for raw in payload.get("tables", [])]
    table_map = {table.table_id: table for table in tables}
    holding_context = (
        _holding_document_context(table_map)
        if payload.get("document", {}).get("doc_group") == "holding"
        else {}
    )
    old_to_new: dict[str, str] = {}
    for chunk in payload.get("chunks", []):
        projection_type = chunk.get("projection_type")
        if projection_type == "holding_report":
            field_refs = _holding_report_field_refs(chunk, table_map, holding_context)
            if chunk.get("projection_field_refs") != field_refs:
                chunk["projection_field_refs"] = field_refs
                changes["holding_report_field_provenance"] += 1
            source_refs = _unique_refs(field_refs)
            if chunk.get("source_refs") != source_refs:
                chunk["source_refs"] = source_refs
                chunk["source_table_ids"] = list(
                    dict.fromkeys(str(ref["table_id"]) for ref in source_refs)
                )
                changes["holding_report_source_refs"] += 1
        elif projection_type == "holding_detail_row":
            field_refs = _holding_detail_field_refs(chunk, holding_context)
            if chunk.get("projection_field_refs") != field_refs:
                chunk["projection_field_refs"] = field_refs
                changes["holding_detail_field_provenance"] += 1
            source_refs = _unique_refs(field_refs)
            if chunk.get("source_refs") != source_refs:
                chunk["source_refs"] = source_refs
                chunk["source_table_ids"] = list(
                    dict.fromkeys(str(ref["table_id"]) for ref in source_refs)
                )
                changes["holding_detail_source_refs"] += 1
            fields = chunk.get("projection_fields") or {}
            explicit = bool(fields) and all(
                str(value).strip() == "정정 전과 동일" for value in fields.values()
            )
            expected_state = "explicit_placeholder" if explicit else "resolved"
            if chunk.get("projection_state") != expected_state:
                chunk["projection_state"] = expected_state
                changes[f"holding_state_{expected_state}"] += 1
            flags = list(chunk.get("quality_flags") or [])
            if explicit and "explicit_placeholder" not in flags:
                flags.append("explicit_placeholder")
                chunk["quality_flags"] = flags
        elif projection_type == "extreme_table_row":
            old_content = str(chunk.get("content") or "")
            lines = old_content.splitlines()
            new_content = "\n".join(dict.fromkeys(lines))
            if new_content != old_content:
                old_id = str(chunk["chunk_id"])
                retrieval_text = str(chunk.get("retrieval_text") or "")
                if not retrieval_text.endswith(old_content):
                    raise ValueError(f"projection content is not retrieval suffix: {old_id}")
                chunk["content"] = new_content
                chunk["retrieval_text"] = retrieval_text[: -len(old_content)] + new_content
                chunk["char_count"] = len(new_content)
                new_id = f"{chunk['doc_id']}:ch_{_chunk_identity(chunk)}"
                chunk["chunk_id"] = new_id
                old_to_new[old_id] = new_id
                changes["duplicate_projection_content_repaired"] += 1
    if old_to_new:
        for chunk in payload["chunks"]:
            if chunk.get("prev_chunk_id") in old_to_new:
                chunk["prev_chunk_id"] = old_to_new[chunk["prev_chunk_id"]]
            if chunk.get("next_chunk_id") in old_to_new:
                chunk["next_chunk_id"] = old_to_new[chunk["next_chunk_id"]]
    if changes:
        payload["projection_provenance_revision"] = "v2.1-release-1"
    return changes


def _write_gzip_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as target:
            json.dump(payload, target, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def repair_projection_provenance(output_dir: Path) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    changed_sources = 0
    files = sorted((output_dir / "documents").rglob("*.json.gz"))
    for index, path in enumerate(files, start=1):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            payload = json.load(source)
        changes = _repair_payload(payload)
        if changes:
            _write_gzip_atomic(path, payload)
            totals.update(changes)
            changed_sources += 1
        if index % 250 == 0 or index == len(files):
            print(
                f"projection repair [{index}/{len(files)}] changed={changed_sources}",
                flush=True,
            )
    report = {
        "projection_only": True,
        "full_rechunk_performed": False,
        "source_count": len(files),
        "changed_source_count": changed_sources,
        "changes": dict(sorted(totals.items())),
    }
    report_dir = output_dir / "projection_repair"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "repair_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def validate_projection_provenance(output_dir: Path) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    files = sorted((output_dir / "documents").rglob("*.json.gz"))
    for index, path in enumerate(files, start=1):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            payload = json.load(source)
        raw_tables = {
            str(table["table_id"]): table for table in payload.get("tables", [])
        }
        for chunk in payload.get("chunks", []):
            if chunk.get("chunk_type") != "table_projection":
                continue
            counts[str(chunk.get("projection_type") or "unknown")] += 1
            for error in _projection_errors(chunk, raw_tables):
                category = _projection_error_category(error)
                errors[category] += 1
                samples.setdefault(category, [])
                if len(samples[category]) < 20:
                    samples[category].append(error)
        if index % 250 == 0 or index == len(files):
            print(
                f"projection validate [{index}/{len(files)}] errors={sum(errors.values())}",
                flush=True,
            )
    report = {
        "valid": not errors,
        "projection_count": sum(counts.values()),
        "counts_by_type": dict(sorted(counts.items())),
        "error_count": sum(errors.values()),
        "error_counts_by_reason": dict(sorted(errors.items())),
        "error_samples_by_reason": dict(sorted(samples.items())),
    }
    report_dir = output_dir / "projection_repair"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def validate_repaired_output_integrity(output_dir: Path) -> dict[str, Any]:
    """Validate serialized outputs after the projection-only migration."""
    issue_counts: Counter[str] = Counter()
    issue_samples: dict[str, list[str]] = {}
    global_chunk_ids: set[str] = set()
    total_chunks = 0
    files = sorted((output_dir / "documents").rglob("*.json.gz"))

    def record(category: str, detail: str) -> None:
        issue_counts[category] += 1
        issue_samples.setdefault(category, [])
        if len(issue_samples[category]) < 20:
            issue_samples[category].append(detail)

    for file_index, path in enumerate(files, start=1):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            payload = json.load(source)
        document = payload.get("document") or {}
        chunks = payload.get("chunks") or []
        section_ids = {
            str(section.get("section_id") or "")
            for section in payload.get("sections") or []
        }
        table_ids = {
            str(table.get("table_id") or "")
            for table in payload.get("tables") or []
        }
        total_chunks += len(chunks)
        for chunk_index, chunk in enumerate(chunks):
            chunk_id = str(chunk.get("chunk_id") or "")
            missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in chunk]
            if missing:
                record("missing_metadata", f"{chunk_id}: {missing}")
            for field in DOCUMENT_METADATA_FIELDS:
                if chunk.get(field) != document.get(field):
                    record("metadata_mismatch", f"{chunk_id}: {field}")
            if str(chunk.get("section_id") or "") not in section_ids:
                record("orphan_section", chunk_id)
            if chunk.get("chunk_type") in {"table", "table_projection"}:
                if str(chunk.get("table_id") or "") not in table_ids:
                    record("orphan_table", chunk_id)
            expected_id = f"{chunk.get('doc_id')}:ch_{_chunk_identity(chunk)}"
            if chunk_id != expected_id:
                record("deterministic_id_mismatch", chunk_id)
            if chunk_id in global_chunk_ids:
                record("duplicate_chunk_id", chunk_id)
            global_chunk_ids.add(chunk_id)
            expected_previous = chunks[chunk_index - 1].get("chunk_id") if chunk_index else None
            expected_next = (
                chunks[chunk_index + 1].get("chunk_id")
                if chunk_index + 1 < len(chunks)
                else None
            )
            if chunk.get("prev_chunk_id") != expected_previous:
                record("invalid_previous_link", chunk_id)
            if chunk.get("next_chunk_id") != expected_next:
                record("invalid_next_link", chunk_id)
            content = str(chunk.get("content") or "")
            if int(chunk.get("char_count") or 0) != len(content):
                record("char_count_mismatch", chunk_id)
            if content and content not in str(chunk.get("retrieval_text") or ""):
                record("incomplete_retrieval_text", chunk_id)
        if file_index % 250 == 0 or file_index == len(files):
            print(
                f"output integrity [{file_index}/{len(files)}] "
                f"issues={sum(issue_counts.values())}",
                flush=True,
            )
    report = {
        "valid": not issue_counts,
        "source_count": len(files),
        "chunk_count": total_chunks,
        "unique_chunk_id_count": len(global_chunk_ids),
        "issue_count": sum(issue_counts.values()),
        "issue_counts_by_reason": dict(sorted(issue_counts.items())),
        "issue_samples_by_reason": dict(sorted(issue_samples.items())),
    }
    report_dir = output_dir / "projection_repair"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def audit_holding_placeholders(output_dir: Path) -> dict[str, Any]:
    """Preserve source-backed placeholder evidence without inferring inheritance."""
    rows: list[dict[str, Any]] = []
    files = sorted((output_dir / "documents").rglob("*.json.gz"))
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            payload = json.load(source)
        tables = {
            str(table.get("table_id") or ""): table
            for table in payload.get("tables") or []
        }
        for chunk in payload.get("chunks") or []:
            if chunk.get("projection_state") != "explicit_placeholder":
                continue
            source_rows: list[dict[str, Any]] = []
            for ref in chunk.get("source_refs") or []:
                table_id = str(ref.get("table_id") or "")
                table = tables.get(table_id) or {}
                start = int(ref.get("row_start") or 0)
                end = int(ref.get("row_end") or start)
                for row_index in range(start, end + 1):
                    raw_row = (table.get("rows") or [])[row_index]
                    source_rows.append(
                        {
                            "table_id": table_id,
                            "row_index": row_index,
                            "cell_texts": [str(cell.get("text") or "") for cell in raw_row],
                        }
                    )
            rows.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "document_metadata": payload.get("document") or {},
                    "content": chunk.get("content"),
                    "projection_fields": chunk.get("projection_fields") or {},
                    "projection_field_refs": chunk.get("projection_field_refs") or {},
                    "source_rows": source_rows,
                    "inheritance_supported": False,
                    "decision": "explicit_placeholder_preserved",
                    "reason": (
                        "The referenced row contains the placeholder, but the document "
                        "does not provide an explicit provenance link to a unique prior value."
                    ),
                }
            )
    report = {
        "placeholder_count": len(rows),
        "inherited_count": 0,
        "explicit_placeholder_count": len(rows),
        "placeholders": rows,
    }
    report_dir = output_dir / "projection_repair"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "placeholder_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
