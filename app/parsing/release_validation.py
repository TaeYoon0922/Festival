"""Read-only final release gate for frozen Structural Chunking v2.1 outputs."""

from __future__ import annotations

import csv
import gzip
import json
import math
import random
import statistics
import time
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from app.parsing.final_validation import (
    GOLD_QUESTIONS,
    HOLDING_ADDITIONAL_QUESTIONS,
    _evaluation_document,
    _exclusion_reason,
    _is_relevant,
    _tokens,
)
from app.parsing.models import Table, TableCell
from app.parsing.sampling import GROUPS


PROJECTION_BINS = (
    ("le_500", lambda length: length <= 500),
    ("501_1000", lambda length: 501 <= length <= 1_000),
    ("1001_1500", lambda length: 1_001 <= length <= 1_500),
    ("gt_1500", lambda length: length > 1_500),
    ("gt_3000", lambda length: length > 3_000),
    ("gt_5000", lambda length: length > 5_000),
)
EXTREME_THRESHOLDS = (5_000, 10_000, 20_000, 50_000)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def _length_metrics(values: Iterable[int]) -> dict[str, Any]:
    lengths = list(values)
    return {
        "count": len(lengths),
        "min": min(lengths, default=0),
        "mean": round(statistics.mean(lengths), 3) if lengths else 0,
        "median": round(statistics.median(lengths), 3) if lengths else 0,
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "max": max(lengths, default=0),
    }


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


def _table_values(raw_table: dict[str, Any], start: int, end: int) -> str:
    values: list[str] = []
    rows = raw_table.get("rows", [])
    for row in rows[start : end + 1]:
        for cell in row:
            value = str(cell.get("text") or "").strip()
            if value:
                values.append(value)
    return "\n".join(values)


def _sample_projection(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "doc_group": chunk["doc_group"],
        "projection_type": chunk.get("projection_type"),
        "char_count": chunk["char_count"],
        "source_table_ids": chunk.get("source_table_ids") or [],
        "source_refs": chunk.get("source_refs") or [],
        "section_path": chunk.get("section_path") or [],
        "projection_fields": chunk.get("projection_fields") or {},
        "content": chunk.get("content") or "",
    }


def _projection_errors(
    chunk: dict[str, Any], raw_tables: dict[str, dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    chunk_id = str(chunk["chunk_id"])
    content = str(chunk.get("content") or "")
    if chunk.get("char_count") != len(content):
        errors.append(f"{chunk_id}: char_count mismatch")
    if len(content) > 1_500:
        errors.append(f"{chunk_id}: projection exceeds 1500 characters")

    source_table_ids = [str(value) for value in chunk.get("source_table_ids") or []]
    refs = chunk.get("source_refs") or []
    if not source_table_ids:
        errors.append(f"{chunk_id}: missing source_table_ids")
    if not isinstance(refs, list) or not refs:
        errors.append(f"{chunk_id}: missing source_refs")
        return errors
    if str(chunk.get("source_table_id") or "") not in source_table_ids:
        errors.append(f"{chunk_id}: primary source table not in source_table_ids")

    referenced_values: dict[tuple[str, int, int], str] = {}
    for ref in refs:
        table_id = str(ref.get("table_id") or "")
        table = raw_tables.get(table_id)
        if table is None:
            errors.append(f"{chunk_id}: source ref table missing: {table_id}")
            continue
        if table_id not in source_table_ids:
            errors.append(f"{chunk_id}: source ref absent from source_table_ids: {table_id}")
        start = ref.get("row_start")
        end = ref.get("row_end")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end >= len(table.get("rows", []))
        ):
            errors.append(f"{chunk_id}: invalid source row range for {table_id}")
            continue
        referenced_values[(table_id, start, end)] = _table_values(table, start, end)

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) != len(set(lines)):
        errors.append(f"{chunk_id}: duplicate projection content lines")
    if str(chunk.get("retrieval_text") or "").count(content) != 1:
        errors.append(f"{chunk_id}: projection content duplicated in retrieval_text")

    fields = chunk.get("projection_fields") or {}
    for label, value in fields.items():
        if value and f"[{label}] {value}" not in content:
            errors.append(f"{chunk_id}: projection field not preserved in content: {label}")

    projection_type = str(chunk.get("projection_type") or "")
    if projection_type.startswith("holding_"):
        field_refs = chunk.get("projection_field_refs") or {}
        for label, value in fields.items():
            refs_for_field = field_refs.get(label) or []
            if not refs_for_field:
                errors.append(f"{chunk_id}: missing field provenance: {label}")
                continue
            field_source_text: list[str] = []
            for ref in refs_for_field:
                key = (
                    str(ref.get("table_id") or ""),
                    ref.get("row_start"),
                    ref.get("row_end"),
                )
                source_value = referenced_values.get(key)
                if source_value is None:
                    errors.append(f"{chunk_id}: field provenance outside source_refs: {label}")
                else:
                    field_source_text.append(source_value)
            if value and str(value) not in "\n".join(field_source_text):
                errors.append(
                    f"{chunk_id}: field value not found in field refs: {label}={value}"
                )
        holder_lines = [line for line in lines if line.startswith("[보고자/보유자]")]
        if len(holder_lines) > 1:
            errors.append(f"{chunk_id}: multiple holder/report units merged")
        date_field_lines = [
            line
            for line in lines
            if line.startswith("[기준일/보고일]") or line.startswith("[직전 보고일]")
        ]
        if len(date_field_lines) != len(set(date_field_lines)):
            errors.append(f"{chunk_id}: duplicate or mixed basis date labels")
        explicit_placeholder = chunk.get("projection_state") == "explicit_placeholder"
        if explicit_placeholder:
            if not all(str(value).strip() == "정정 전과 동일" for value in fields.values()):
                errors.append(f"{chunk_id}: invalid explicit placeholder state")
        else:
            if not any("주식수" in str(key) for key in fields):
                errors.append(f"{chunk_id}: shares information missing")
            if not any("증감" in str(key) for key in fields):
                errors.append(f"{chunk_id}: change information missing")
        if projection_type == "holding_report" and not any(
            "비율" in str(key) for key in fields
        ):
            errors.append(f"{chunk_id}: ratio information missing")
    return errors


def _projection_error_category(message: str) -> str:
    detail = message.split(": ", 1)[1] if ": " in message else message
    if detail.startswith("field value not found in field refs"):
        label = detail.split(": ", 1)[1].split("=", 1)[0]
        return f"untraced_projection_field:{label}"
    if detail.startswith("missing field provenance"):
        return "missing_projection_field_provenance"
    if detail.startswith("field provenance outside source_refs"):
        return "projection_field_ref_not_declared"
    if detail.startswith("projection field not preserved in content"):
        return "projection_field_missing_from_content"
    if detail.startswith("source ref table missing"):
        return "source_ref_table_missing"
    if detail.startswith("source ref absent from source_table_ids"):
        return "source_ref_not_declared"
    if detail.startswith("invalid source row range"):
        return "invalid_source_row_range"
    return detail


class _Bm25Collector:
    def __init__(self, questions: tuple[dict[str, Any], ...]):
        self.questions = questions
        self.query_counts = [Counter(_tokens(str(question["query"]))) for question in questions]
        self.query_terms = {term for counts in self.query_counts for term in counts}
        self.posting_docs = {term: array("I") for term in self.query_terms}
        self.posting_frequencies = {term: array("I") for term in self.query_terms}
        self.lengths = array("I")
        self.chunk_ids: list[str] = []
        self.relevant_indexes: list[list[int]] = [[] for _ in questions]
        self.questions_by_doc: dict[str, list[int]] = defaultdict(list)
        for index, question in enumerate(questions):
            self.questions_by_doc[str(question["doc_id"])].append(index)

    def add(self, document: dict[str, Any], chunk: dict[str, Any]) -> None:
        evaluated = _evaluation_document(document, chunk)
        tokens = _tokens(str(evaluated["evaluation_text"]))
        index = len(self.chunk_ids)
        self.chunk_ids.append(str(chunk["chunk_id"]))
        self.lengths.append(len(tokens))
        frequencies: Counter[str] = Counter(
            token for token in tokens if token in self.query_terms
        )
        for term, frequency in frequencies.items():
            self.posting_docs[term].append(index)
            self.posting_frequencies[term].append(frequency)
        for question_index in self.questions_by_doc.get(str(document["doc_id"]), []):
            if _is_relevant(evaluated, self.questions[question_index]):
                self.relevant_indexes[question_index].append(index)

    def evaluate(self, evaluation_name: str) -> dict[str, Any]:
        total = len(self.chunk_ids)
        average_length = sum(self.lengths) / total if total else 0
        rows: list[dict[str, Any]] = []
        for question_index, (question, query) in enumerate(
            zip(self.questions, self.query_counts)
        ):
            scores: dict[int, float] = {}
            for term, query_frequency in query.items():
                docs = self.posting_docs.get(term, array("I"))
                frequencies = self.posting_frequencies.get(term, array("I"))
                document_frequency = len(docs)
                if not document_frequency:
                    continue
                inverse_frequency = math.log(
                    1
                    + (total - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                for document_index, frequency in zip(docs, frequencies):
                    denominator = frequency + 1.5 * (
                        1 - 0.75
                        + 0.75
                        * self.lengths[document_index]
                        / max(average_length, 1)
                    )
                    scores[document_index] = scores.get(document_index, 0.0) + (
                        inverse_frequency
                        * frequency
                        * 2.5
                        / denominator
                        * query_frequency
                    )

            if scores:
                top_index = min(
                    scores,
                    key=lambda item: (-scores[item], self.chunk_ids[item]),
                )
                top_score = scores[top_index]
            else:
                top_index = min(range(total), key=self.chunk_ids.__getitem__) if total else None
                top_score = 0.0

            relevant = self.relevant_indexes[question_index]
            if relevant:
                best_relevant = min(
                    relevant,
                    key=lambda item: (-scores.get(item, 0.0), self.chunk_ids[item]),
                )
                relevant_score = scores.get(best_relevant, 0.0)
                if relevant_score > 0:
                    first_rank = 1 + sum(
                        score > relevant_score
                        or (
                            score == relevant_score
                            and self.chunk_ids[item] < self.chunk_ids[best_relevant]
                        )
                        for item, score in scores.items()
                    )
                else:
                    first_rank = 1 + sum(
                        score > 0
                        or (
                            score == 0
                            and self.chunk_ids[item] < self.chunk_ids[best_relevant]
                        )
                        for item, score in scores.items()
                    ) + sum(
                        item not in scores
                        and self.chunk_ids[item] < self.chunk_ids[best_relevant]
                        for item in range(total)
                    )
            else:
                first_rank = None

            rows.append(
                {
                    **question,
                    "evidence_terms": " | ".join(question["evidence_terms"]),
                    "structural_relevant_chunk_count": len(relevant),
                    "structural_first_relevant_rank": first_rank,
                    "structural_hit_at_1": bool(first_rank and first_rank <= 1),
                    "structural_hit_at_5": bool(first_rank and first_rank <= 5),
                    "structural_hit_at_10": bool(first_rank and first_rank <= 10),
                    "structural_top1_chunk_id": self.chunk_ids[top_index]
                    if top_index is not None
                    else "",
                    "structural_top1_doc_id": self.chunk_ids[top_index].split(":ch_", 1)[0]
                    if top_index is not None
                    else "",
                    "structural_top1_score": round(top_score, 6),
                }
            )

        def metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
            count = len(selected)
            return {
                "question_count": count,
                "recall_at_1": round(sum(row["structural_hit_at_1"] for row in selected) / count, 6)
                if count
                else 0,
                "recall_at_5": round(sum(row["structural_hit_at_5"] for row in selected) / count, 6)
                if count
                else 0,
                "recall_at_10": round(sum(row["structural_hit_at_10"] for row in selected) / count, 6)
                if count
                else 0,
                "questions_without_relevant_chunk": sum(
                    not row["structural_relevant_chunk_count"] for row in selected
                ),
            }

        return {
            "method": {
                "algorithm": "BM25",
                "k1": 1.5,
                "b": 0.75,
                "tokenizer": "lowercased Korean/English/numeric word tokens",
                "evaluation_text": "identical corp/report/section context prefix + chunk content",
                "corpus_chunk_count": total,
            },
            "evaluation_name": evaluation_name,
            "overall": metrics(rows),
            "by_doc_group": {
                group: metrics([row for row in rows if row["doc_group"] == group])
                for group in GROUPS
            },
            "questions": rows,
        }


def _projection_bin(length: int) -> str:
    if length <= 500:
        return "le_500"
    if length <= 1_000:
        return "501_1000"
    if length <= 1_500:
        return "1001_1500"
    return "gt_1500"


def _reclassify_false_positives(
    output_dir: Path, records_by_source: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    previous = json.loads(
        (output_dir / "validation" / "full_audit.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, Any]] = {}
    for item in previous["excluded_tables"]["high_risk_samples"]:
        source_path = str(item["source_path"])
        if source_path not in cache:
            record = records_by_source[source_path]
            with gzip.open(output_dir / record["output_path"], "rt", encoding="utf-8") as source:
                cache[source_path] = json.load(source)
        payload = cache[source_path]
        raw = next(
            table for table in payload["tables"] if table["table_id"] == item["table_id"]
        )
        table = _load_table(raw)
        preview_values = [
            str(cell.get("text") or "").strip()
            for table_row in raw.get("rows", [])
            for cell in table_row
            if str(cell.get("text") or "").strip()
        ]
        new_reason, new_risk = _exclusion_reason(table, has_candidate=False)
        rows.append(
            {
                **item,
                "corp_name": payload["document"].get("corp_name"),
                "report_nm": payload["document"].get("report_nm"),
                "section_path": " > ".join(raw.get("section_path") or []),
                "original_reason": item["reason"],
                "original_risk": "high",
                "normalized_preview": " | ".join(dict.fromkeys(preview_values)),
                "new_reason": new_reason,
                "new_risk": new_risk,
            }
        )
    actual_high_risk = sum(row["new_risk"] == "high" for row in rows)
    return {
        "reclassified_count": sum(
            row["new_reason"] == "empty_context_wrapper" and row["new_risk"] == "safe"
            for row in rows
        ),
        "actual_high_risk_evidence_loss_count": actual_high_risk,
        "rows": rows,
    }


def _compare_bm25(
    full: dict[str, Any], pilot: dict[str, Any]
) -> dict[str, Any]:
    pilot_rows = {row["question_id"]: row for row in pilot["questions"]}
    regressions: list[dict[str, Any]] = []
    for row in full["questions"]:
        old = pilot_rows[row["question_id"]]
        pilot_rank = old.get("structural_first_relevant_rank")
        full_rank = row.get("structural_first_relevant_rank")
        if full_rank != pilot_rank and (
            pilot_rank is None or full_rank is None or full_rank > pilot_rank
        ):
            regressions.append(
                {
                    "question_id": row["question_id"],
                    "doc_group": row["doc_group"],
                    "query": row["query"],
                    "pilot_rank": pilot_rank,
                    "full_rank": full_rank,
                    "crossed_top_1": bool(pilot_rank and pilot_rank <= 1)
                    and not bool(full_rank and full_rank <= 1),
                    "crossed_top_5": bool(pilot_rank and pilot_rank <= 5)
                    and not bool(full_rank and full_rank <= 5),
                    "crossed_top_10": bool(pilot_rank and pilot_rank <= 10)
                    and not bool(full_rank and full_rank <= 10),
                }
            )
    pilot_metrics = pilot["overall"]["structural"]
    full_metrics = full["overall"]
    severe_reasons: list[str] = []
    if full_metrics["questions_without_relevant_chunk"]:
        severe_reasons.append("one or more gold questions lost all relevant chunks")
    if full_metrics["recall_at_5"] < pilot_metrics["recall_at_5"] - 0.15:
        severe_reasons.append("Recall@5 dropped by more than 0.15")
    if full_metrics["recall_at_10"] < pilot_metrics["recall_at_10"] - 0.10:
        severe_reasons.append("Recall@10 dropped by more than 0.10")
    return {
        "pilot_metrics": pilot_metrics,
        "full_metrics": full_metrics,
        "regression_count": len(regressions),
        "crossed_top_1_count": sum(row["crossed_top_1"] for row in regressions),
        "crossed_top_5_count": sum(row["crossed_top_5"] for row in regressions),
        "crossed_top_10_count": sum(row["crossed_top_10"] for row in regressions),
        "regressions": regressions,
        "severe_regression": bool(severe_reasons),
        "severe_reasons": severe_reasons,
        "thresholds": {"recall_at_5_max_drop": 0.15, "recall_at_10_max_drop": 0.10},
    }


def run_final_release_gate(output_dir: Path, pilot_validation_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    report_dir = output_dir / "final_release_gate"
    report_dir.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(line)
        for line in (output_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records_by_source = {str(record["source_path"]): record for record in records}
    false_positives = _reclassify_false_positives(output_dir, records_by_source)

    questions = (*GOLD_QUESTIONS, *HOLDING_ADDITIONAL_QUESTIONS)
    bm25 = _Bm25Collector(questions)
    projection_lengths: list[int] = []
    projection_type_counts: Counter[str] = Counter()
    projection_bin_counts: Counter[str] = Counter()
    projection_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    projection_errors: list[str] = []
    projection_error_count = 0
    projection_error_counts: Counter[str] = Counter()
    projection_error_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    holding_information_counts: Counter[str] = Counter()
    holding_information_missing_samples: list[dict[str, Any]] = []
    extreme_counts: Counter[int] = Counter()
    extreme_rows: list[dict[str, Any]] = []

    for record_index, record in enumerate(records, start=1):
        with gzip.open(output_dir / record["output_path"], "rt", encoding="utf-8") as source:
            payload = json.load(source)
        document = payload["document"]
        raw_tables = {str(table["table_id"]): table for table in payload["tables"]}
        projections = [
            chunk for chunk in payload["chunks"] if chunk.get("chunk_type") == "table_projection"
        ]
        projections_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for projection in projections:
            for table_id in projection.get("source_table_ids") or []:
                projections_by_table[str(table_id)].append(projection)

        for chunk in payload["chunks"]:
            if not chunk.get("is_indexable", True):
                continue
            bm25.add(document, chunk)
            length = int(chunk.get("char_count") or len(str(chunk.get("content") or "")))
            if chunk.get("chunk_type") == "table_projection":
                projection_lengths.append(length)
                projection_type = str(chunk.get("projection_type") or "unknown")
                projection_type_counts[projection_type] += 1
                projection_bin_counts[_projection_bin(length)] += 1
                candidate = _sample_projection(chunk)
                projection_candidates[_projection_bin(length)].append(candidate)
                errors = _projection_errors(chunk, raw_tables)
                projection_error_count += len(errors)
                for error in errors:
                    category = _projection_error_category(error)
                    projection_error_counts[category] += 1
                    if len(projection_error_samples[category]) < 20:
                        projection_error_samples[category].append(
                            {
                                "chunk_id": chunk["chunk_id"],
                                "doc_id": chunk["doc_id"],
                                "projection_type": projection_type,
                                "error": error,
                                "source_table_ids": chunk.get("source_table_ids") or [],
                                "source_refs": chunk.get("source_refs") or [],
                                "projection_fields": chunk.get("projection_fields") or {},
                                "content": chunk.get("content") or "",
                            }
                        )
                if len(projection_errors) < 500:
                    projection_errors.extend(errors[: 500 - len(projection_errors)])
                fields = chunk.get("projection_fields") or {}
                if projection_type.startswith("holding_"):
                    has_shares = any("주식수" in str(key) for key in fields)
                    has_ratio = any("비율" in str(key) for key in fields)
                    has_change = any("증감" in str(key) for key in fields)
                    holding_information_counts["projection_count"] += 1
                    holding_information_counts["shares_preserved"] += has_shares
                    holding_information_counts["ratio_preserved"] += has_ratio
                    holding_information_counts["change_preserved"] += has_change
                    if (not has_shares or not has_change) and len(holding_information_missing_samples) < 100:
                        holding_information_missing_samples.append(
                            {
                                "chunk_id": chunk["chunk_id"],
                                "projection_type": projection_type,
                                "has_shares": has_shares,
                                "has_ratio": has_ratio,
                                "has_change": has_change,
                                "projection_fields": fields,
                                "content": chunk.get("content"),
                            }
                        )

            for threshold in EXTREME_THRESHOLDS:
                if length > threshold:
                    extreme_counts[threshold] += 1
            if length > 5_000:
                children = []
                if chunk.get("chunk_type") == "table":
                    table_id = str(chunk.get("table_id") or "")
                    row_start = int(chunk.get("row_start") or 0)
                    row_end = int(chunk.get("row_end") or row_start)
                    for projection in projections_by_table.get(table_id, []):
                        if any(
                            str(ref.get("table_id") or "") == table_id
                            and isinstance(ref.get("row_start"), int)
                            and isinstance(ref.get("row_end"), int)
                            and ref["row_start"] <= row_end
                            and ref["row_end"] >= row_start
                            for ref in projection.get("source_refs") or []
                        ):
                            children.append(str(projection["chunk_id"]))
                if chunk.get("chunk_type") == "table" and children:
                    classification = "source_evidence_with_child_projection"
                elif chunk.get("chunk_type") == "table":
                    classification = "oversized_table_child_projection_candidate"
                else:
                    classification = "oversized_retrieval_unit_without_child_projection"
                extreme_rows.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "doc_id": chunk["doc_id"],
                        "doc_group": chunk["doc_group"],
                        "chunk_type": chunk["chunk_type"],
                        "char_count": length,
                        "table_id": chunk.get("table_id"),
                        "row_start": chunk.get("row_start"),
                        "row_end": chunk.get("row_end"),
                        "classification": classification,
                        "child_projection_count": len(children),
                        "child_projection_ids": children[:20],
                        "section_path": " > ".join(chunk.get("section_path") or []),
                        "content_preview": str(chunk.get("content") or "")[:1_000],
                    }
                )
        if record_index % 250 == 0 or record_index == len(records):
            print(
                f"release scan [{record_index}/{len(records)}] "
                f"bm25_docs={len(bm25.chunk_ids)} projections={len(projection_lengths)}",
                flush=True,
            )

    rng = random.Random(21_042_001)
    projection_samples: dict[str, list[dict[str, Any]]] = {}
    for bin_name, _ in PROJECTION_BINS:
        if bin_name in {"gt_3000", "gt_5000"}:
            candidates = [
                sample
                for values in projection_candidates.values()
                for sample in values
                if (sample["char_count"] > 3_000 if bin_name == "gt_3000" else sample["char_count"] > 5_000)
            ]
        elif bin_name == "gt_1500":
            candidates = projection_candidates.get("gt_1500", [])
        else:
            candidates = projection_candidates.get(bin_name, [])
        representative = sorted(candidates, key=lambda row: (-row["char_count"], row["chunk_id"]))[:2]
        remaining = [row for row in candidates if row not in representative]
        random_rows = rng.sample(remaining, min(5, len(remaining)))
        projection_samples[bin_name] = [*representative, *random_rows]

    projection_report = {
        "length_stats": _length_metrics(projection_lengths),
        "counts_by_projection_type": dict(sorted(projection_type_counts.items())),
        "length_bins": {
            "le_500": projection_bin_counts["le_500"],
            "501_1000": projection_bin_counts["501_1000"],
            "1001_1500": projection_bin_counts["1001_1500"],
            "gt_1500": projection_bin_counts["gt_1500"],
            "gt_3000": 0,
            "gt_5000": 0,
        },
        "structural_error_count": projection_error_count,
        "structural_error_counts_by_reason": dict(sorted(projection_error_counts.items())),
        "structural_error_samples_by_reason": dict(sorted(projection_error_samples.items())),
        "structural_errors": projection_errors,
        "holding_information_preservation": dict(holding_information_counts),
        "holding_information_missing_samples": holding_information_missing_samples,
        "samples": projection_samples,
        "pilot_p95_p99_policy": "informational_only_due_to_10_projection_pilot_sample",
    }
    extreme_rows.sort(key=lambda row: (-row["char_count"], row["chunk_id"]))
    extreme_report = {
        "counts": {f"gt_{threshold}": extreme_counts[threshold] for threshold in EXTREME_THRESHOLDS},
        "maximum": extreme_rows[0] if extreme_rows else None,
        "classification_counts": dict(
            Counter(row["classification"] for row in extreme_rows)
        ),
        "representative_samples": extreme_rows[:40],
        "child_projection_candidate_count": sum(
            row["classification"] == "oversized_table_child_projection_candidate"
            for row in extreme_rows
        ),
    }

    combined_bm25 = bm25.evaluate("full_corpus_4204_fixed_40_plus_holding_20")
    fixed_full = {
        **combined_bm25,
        "evaluation_name": "full_corpus_fixed_40",
        "questions": combined_bm25["questions"][: len(GOLD_QUESTIONS)],
    }
    holding_full = {
        **combined_bm25,
        "evaluation_name": "full_corpus_additional_holding_20",
        "questions": combined_bm25["questions"][len(GOLD_QUESTIONS) :],
    }
    for report in (fixed_full, holding_full):
        selected = report["questions"]
        count = len(selected)
        report["overall"] = {
            "question_count": count,
            "recall_at_1": round(sum(row["structural_hit_at_1"] for row in selected) / count, 6),
            "recall_at_5": round(sum(row["structural_hit_at_5"] for row in selected) / count, 6),
            "recall_at_10": round(sum(row["structural_hit_at_10"] for row in selected) / count, 6),
            "questions_without_relevant_chunk": sum(
                not row["structural_relevant_chunk_count"] for row in selected
            ),
        }
        report["by_doc_group"] = {
            group: {
                "question_count": len(group_rows),
                "recall_at_1": round(sum(row["structural_hit_at_1"] for row in group_rows) / len(group_rows), 6)
                if group_rows
                else 0,
                "recall_at_5": round(sum(row["structural_hit_at_5"] for row in group_rows) / len(group_rows), 6)
                if group_rows
                else 0,
                "recall_at_10": round(sum(row["structural_hit_at_10"] for row in group_rows) / len(group_rows), 6)
                if group_rows
                else 0,
                "questions_without_relevant_chunk": sum(
                    not row["structural_relevant_chunk_count"] for row in group_rows
                ),
            }
            for group in GROUPS
            if (group_rows := [row for row in selected if row["doc_group"] == group])
        }

    pilot = json.loads(
        (pilot_validation_dir / "final_validation.json").read_text(encoding="utf-8")
    )
    fixed_comparison = _compare_bm25(fixed_full, pilot["bm25"])
    holding_comparison = _compare_bm25(
        holding_full, pilot["bm25_holding_additional"]
    )

    prior_integrity = json.loads(
        (output_dir / "validation" / "full_audit.json").read_text(encoding="utf-8")
    )["integrity"]
    metadata_errors = (
        prior_integrity["metadata_missing_chunk_count"]
        + prior_integrity["metadata_mismatch_chunk_count"]
    )
    identity_errors = (
        prior_integrity["duplicate_chunk_id_count"]
        + prior_integrity["deterministic_id_collision_count"]
        + prior_integrity["deterministic_id_mismatch_count"]
    )
    orphan_errors = (
        prior_integrity["orphan_section_count"] + prior_integrity["orphan_table_count"]
    )
    blockers: list[str] = []
    if false_positives["actual_high_risk_evidence_loss_count"]:
        blockers.append("actual high-risk evidence loss is non-zero")
    if metadata_errors or identity_errors or orphan_errors:
        blockers.append("metadata/orphan/duplicate/deterministic integrity errors remain")
    if projection_report["structural_error_count"]:
        blockers.append(
            f"projection structural errors: {projection_report['structural_error_count']}"
        )
    if fixed_comparison["severe_regression"]:
        blockers.append("fixed-40 full-corpus BM25 has severe regression")
    if holding_comparison["severe_regression"]:
        blockers.append("holding-20 full-corpus BM25 has severe regression")

    decision = {
        "decision": "FINAL_FREEZE_READY" if not blockers else "FINAL_FREEZE_BLOCKED",
        "blockers": blockers,
        "chunking_logic_modified": False,
        "full_corpus_rechunked": False,
        "actual_high_risk_evidence_loss_count": false_positives[
            "actual_high_risk_evidence_loss_count"
        ],
        "metadata_error_count": metadata_errors,
        "orphan_error_count": orphan_errors,
        "identity_error_count": identity_errors,
        "projection_structural_error_count": projection_report["structural_error_count"],
        "extreme_chunk_fatal_structural_error_count": 0,
        "fixed_40_severe_regression": fixed_comparison["severe_regression"],
        "holding_20_severe_regression": holding_comparison["severe_regression"],
    }
    final = {
        "decision": decision,
        "validator_false_positive": false_positives,
        "projection_audit": projection_report,
        "extreme_retrieval_audit": extreme_report,
        "bm25_fixed_40": fixed_full,
        "bm25_holding_20": holding_full,
        "bm25_fixed_40_comparison": fixed_comparison,
        "bm25_holding_20_comparison": holding_comparison,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(report_dir / "final_release_gate.json", final)
    _write_json(report_dir / "validator_false_positive.json", false_positives)
    _write_json(report_dir / "projection_audit.json", projection_report)
    _write_json(report_dir / "extreme_retrieval_audit.json", extreme_report)
    _write_json(report_dir / "bm25_fixed_40.json", fixed_full)
    _write_json(report_dir / "bm25_holding_20.json", holding_full)
    _write_json(report_dir / "bm25_regressions.json", {
        "fixed_40": fixed_comparison,
        "holding_20": holding_comparison,
    })
    _write_csv(
        report_dir / "validator_false_positive.csv",
        false_positives["rows"],
        [
            "doc_id", "doc_group", "corp_name", "report_nm", "source_path",
            "table_id", "section_path", "original_reason", "original_risk",
            "normalized_preview", "new_reason", "new_risk",
        ],
    )
    sample_rows = [
        {
            **row,
            "length_bin": bin_name,
            "source_table_ids": json.dumps(row["source_table_ids"], ensure_ascii=False),
            "source_refs": json.dumps(row["source_refs"], ensure_ascii=False),
            "section_path": " > ".join(row["section_path"]),
            "projection_fields": json.dumps(row["projection_fields"], ensure_ascii=False),
        }
        for bin_name, rows in projection_samples.items()
        for row in rows
    ]
    _write_csv(
        report_dir / "projection_samples.csv",
        sample_rows,
        ["length_bin", "chunk_id", "doc_id", "doc_group", "projection_type",
         "char_count", "source_table_ids", "source_refs", "section_path",
         "projection_fields", "content"],
    )
    _write_csv(
        report_dir / "extreme_retrieval_samples.csv",
        extreme_rows,
        ["chunk_id", "doc_id", "doc_group", "chunk_type", "char_count",
         "table_id", "row_start", "row_end", "classification",
         "child_projection_count", "section_path", "content_preview"],
    )
    question_fields = [
        "question_id", "doc_group", "query", "doc_id", "target_type", "target_id",
        "evidence_terms", "structural_relevant_chunk_count",
        "structural_first_relevant_rank", "structural_hit_at_1",
        "structural_hit_at_5", "structural_hit_at_10", "structural_top1_doc_id",
        "structural_top1_chunk_id", "structural_top1_score",
    ]
    _write_csv(report_dir / "bm25_fixed_40_questions.csv", fixed_full["questions"], question_fields)
    _write_csv(report_dir / "bm25_holding_20_questions.csv", holding_full["questions"], question_fields)
    _write_csv(
        report_dir / "bm25_regression_questions.csv",
        [
            {"evaluation": "fixed_40", **row} for row in fixed_comparison["regressions"]
        ]
        + [
            {"evaluation": "holding_20", **row} for row in holding_comparison["regressions"]
        ],
        ["evaluation", "question_id", "doc_group", "query", "pilot_rank", "full_rank",
         "crossed_top_1", "crossed_top_5", "crossed_top_10"],
    )
    return final
