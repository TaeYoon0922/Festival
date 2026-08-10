"""Parallel, resumable full-corpus parsing pipeline."""

from __future__ import annotations

import gzip
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from app.parsing.chunking import build_chunks
from app.parsing.dart_xml import parse_dart_document
from app.parsing.sampling import load_manifest, resolve_unicode_path


DOCUMENT_FIELDS = (
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
    "is_correction",
    "rcept_no",
    "rcept_dt",
    "flr_nm",
    "base_year",
    "base_month",
    "file_path",
    "file_format",
    "n_files",
)


def _document_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in DOCUMENT_FIELDS}


def _write_gzip_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=3) as target:
            json.dump(
                payload,
                target,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def _summarize_payload(
    payload: dict[str, Any], output_relative: str, status: str
) -> dict[str, Any]:
    document = payload["document"]
    part = payload["part"]
    chunks = payload["chunks"]
    chunk_kinds = Counter(chunk["kind"] for chunk in chunks)
    return {
        "part_id": part["part_id"],
        "doc_id": document["doc_id"],
        "doc_group": document["doc_group"],
        "corp_name": document["corp_name"],
        "report_nm": document["report_nm"],
        "is_correction": document["is_correction"],
        "source_path": part["source_path"],
        "source_format": part["source_format"],
        "source_size": part["source_size"],
        "is_primary": part["is_primary"],
        "section_count": len(payload["sections"]),
        "table_count": len(payload["tables"]),
        "chunk_count": len(chunks),
        "text_chunk_count": chunk_kinds["text"],
        "table_chunk_count": chunk_kinds["table"],
        "warning_count": len(payload.get("parser_warnings", [])),
        "output_path": output_relative,
        "status": status,
    }


def _process_source(task: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(task["output_path"])
    if task["resume"] and output_path.is_file():
        try:
            payload = _read_gzip_json(output_path)
            return _summarize_payload(
                payload, output_relative=task["output_relative"], status="resumed"
            )
        except (OSError, EOFError, json.JSONDecodeError, KeyError):
            pass

    row = task["row"]
    source_path = Path(task["source_path"])
    parsed = parse_dart_document(
        source_path,
        fallback_title=str(row.get("report_nm") or "공시 문서"),
    )
    chunks = build_chunks(
        doc_id=task["part_id"],
        parsed=parsed,
        max_chars=int(task["max_chars"]),
        overlap=int(task["overlap"]),
    )
    section_map = parsed.section_map()
    payload = {
        "schema_version": "1.0",
        "document": {
            **_document_metadata(row),
            "parsed_title": parsed.document_title,
        },
        "part": {
            "part_id": task["part_id"],
            "source_path": task["source_relative"],
            "source_format": task["source_format"],
            "source_size": source_path.stat().st_size,
            "is_primary": task["is_primary"],
        },
        "sections": [section.to_dict() for section in parsed.sections],
        "tables": [
            table.to_dict(section_map[table.section_id].path)
            for table in parsed.tables
        ],
        "chunks": chunks,
        "parser_warnings": parsed.parser_warnings,
    }
    _write_gzip_json(output_path, payload)
    return _summarize_payload(
        payload, output_relative=task["output_relative"], status="parsed"
    )


def build_full_tasks(
    manifest_path: Path,
    corpus_dir: Path,
    output_dir: Path,
    max_chars: int,
    overlap: int,
    resume: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for row in load_manifest(manifest_path):
        folder = resolve_unicode_path(corpus_dir, str(row["file_path"]))
        suffixes = {".xml"} if row["file_format"] == "xml" else {".html"}
        sources = (
            sorted(
                [
                    path
                    for path in folder.iterdir()
                    if path.is_file() and path.suffix.lower() in suffixes
                ],
                key=lambda path: (
                    path.stem != str(row["rcept_no"]),
                    path.name,
                ),
            )
            if folder.is_dir()
            else []
        )
        if not sources:
            missing.append(
                {
                    "doc_id": row["doc_id"],
                    "file_path": row["file_path"],
                    "file_format": row["file_format"],
                    "reason": "no parseable XML or HTML source",
                }
            )
            continue

        for source in sources:
            source_relative = str(source.relative_to(corpus_dir))
            source_format = source.suffix.lower().lstrip(".")
            part_id = f"{row['doc_id']}:{source.stem}"
            output_relative = (
                Path("documents")
                / str(row["doc_group"])
                / str(row["doc_id"])
                / f"{source.stem}.json.gz"
            )
            tasks.append(
                {
                    "part_id": part_id,
                    "row": row,
                    "source_path": str(source),
                    "source_relative": source_relative,
                    "source_format": source_format,
                    "is_primary": source.stem == str(row["rcept_no"])
                    or source.stem == f"{row['rcept_no']}_viewer",
                    "output_path": str(output_dir / output_relative),
                    "output_relative": str(output_relative),
                    "max_chars": max_chars,
                    "overlap": overlap,
                    "resume": resume,
                }
            )
    return tasks, missing


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            target.write("\n")


def run_full_pipeline(
    corpus_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    workers: int = 4,
    max_chars: int = 1_200,
    overlap: int = 150,
    resume: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks, missing = build_full_tasks(
        manifest_path=manifest_path,
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        max_chars=max_chars,
        overlap=overlap,
        resume=resume,
    )
    plan_rows = [
        {
            "part_id": task["part_id"],
            "doc_id": task["row"]["doc_id"],
            "doc_group": task["row"]["doc_group"],
            "source_path": task["source_relative"],
            "source_format": task["source_format"],
            "output_path": task["output_relative"],
        }
        for task in tasks
    ]
    _write_jsonl(output_dir / "source_plan.jsonl", plan_rows)
    if missing:
        _write_json(output_dir / "missing_sources.json", missing)

    started = time.monotonic()
    last_progress = started
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total = len(tasks)
    print(
        f"Full parse start: {total} sources, "
        f"{len({task['row']['doc_id'] for task in tasks})} documents, "
        f"workers={workers}",
        flush=True,
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_tasks = {
            executor.submit(_process_source, task): task for task in tasks
        }
        for completed, future in enumerate(as_completed(future_tasks), start=1):
            task = future_tasks[future]
            try:
                records.append(future.result())
            except Exception as error:  # Keep the full run alive and report all failures.
                failures.append(
                    {
                        "part_id": task["part_id"],
                        "source_path": task["source_relative"],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
                print(
                    f"FAILED {task['part_id']}: {type(error).__name__}: {error}",
                    flush=True,
                )

            now = time.monotonic()
            if completed == total or completed % 25 == 0 or now - last_progress >= 30:
                elapsed = max(now - started, 0.001)
                rate = completed / elapsed
                remaining = (total - completed) / rate if rate else 0
                print(
                    f"[{completed}/{total}] {completed / total:.1%} "
                    f"rate={rate:.1f}/s eta={remaining / 60:.1f}m "
                    f"failures={len(failures)}",
                    flush=True,
                )
                last_progress = now

    records.sort(key=lambda row: row["part_id"])
    _write_jsonl(output_dir / "index.jsonl", records)
    if failures:
        _write_json(output_dir / "failures.json", failures)

    documents = {record["doc_id"] for record in records}
    group_document_counts = Counter()
    for row in load_manifest(manifest_path):
        if row["doc_id"] in documents:
            group_document_counts[row["doc_group"]] += 1
    elapsed_seconds = time.monotonic() - started
    summary = {
        "complete": not failures and not missing and len(records) == len(tasks),
        "manifest_document_count": len(load_manifest(manifest_path)),
        "processed_document_count": len(documents),
        "source_count": len(records),
        "xml_source_count": sum(record["source_format"] == "xml" for record in records),
        "html_source_count": sum(record["source_format"] == "html" for record in records),
        "group_document_counts": dict(group_document_counts),
        "group_source_counts": dict(Counter(record["doc_group"] for record in records)),
        "section_count": sum(record["section_count"] for record in records),
        "table_count": sum(record["table_count"] for record in records),
        "chunk_count": sum(record["chunk_count"] for record in records),
        "warning_count": sum(record["warning_count"] for record in records),
        "parsed_source_count": sum(record["status"] == "parsed" for record in records),
        "resumed_source_count": sum(record["status"] == "resumed" for record in records),
        "missing_source_count": len(missing),
        "failure_count": len(failures),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }
    _write_json(output_dir / "summary.json", summary)
    if failures or missing:
        raise RuntimeError(
            f"Full parse incomplete: failures={len(failures)}, missing={len(missing)}"
        )
    return summary
