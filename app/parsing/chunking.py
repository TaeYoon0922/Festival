"""Deterministic, structure-aware chunks for disclosure retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from app.parsing.models import ParsedDocument, Section, Table, TableCell


CHUNKING_VERSION = "2.1"
DOCUMENT_METADATA_FIELDS = (
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
)
REQUIRED_CHUNK_FIELDS = (
    *DOCUMENT_METADATA_FIELDS,
    "section_id",
    "parent_section_id",
    "section_title",
    "section_path",
    "section_depth",
    "chunk_type",
    "chunk_order",
    "source_file",
    "char_count",
    "prev_chunk_id",
    "next_chunk_id",
)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")
_ONLY_MARKUP = re.compile(r"^(?:\s*<[^>]+>\s*)+$")
_MEANINGFUL_CHARACTER = re.compile(r"[0-9A-Za-z가-힣]")
_UNIT_PATTERNS = (
    re.compile(r"\(\s*단위\s*[:：]?\s*([^)\n|]{1,40})\)"),
    re.compile(r"단위\s*[:：]\s*([^\n|\]]{1,40})"),
)
_BASIS_PERIOD = re.compile(
    r"(?:기준일|기준기간|작성기준일|보고의무발생일)\s*[:：]?\s*"
    r"([^\n|]{1,50})"
)
_PERIOD_LABEL = re.compile(
    r"(?:당기|전기|전전기|누적|분기|반기|제\s*\d+\s*기|20\d{2}(?:년|\.\d{1,2})?)"
)


@dataclass(frozen=True)
class ChunkingConfig:
    """Shared deterministic limits; none are allowed to split a table row."""

    target_chars: int = 1_200
    min_chars: int = 700
    max_chars: int = 1_500
    sentence_overlap_chars: int = 120

    def __post_init__(self) -> None:
        if not (0 < self.min_chars <= self.target_chars <= self.max_chars):
            raise ValueError("expected min_chars <= target_chars <= max_chars")
        if self.sentence_overlap_chars < 0:
            raise ValueError("sentence_overlap_chars must be non-negative")


class ChunkingStrategy:
    """Disclosure-type policy for keeping related table rows together."""

    name = "default"
    keep_whole_table_rows = 30
    row_group_size = 24

    def table_row_groups(self, row_indexes: list[int]) -> list[list[int]]:
        if not row_indexes:
            return [[]]
        if len(row_indexes) <= self.keep_whole_table_rows:
            return [row_indexes]
        return [
            row_indexes[index : index + self.row_group_size]
            for index in range(0, len(row_indexes), self.row_group_size)
        ]


class PeriodicChunkingStrategy(ChunkingStrategy):
    name = "periodic"
    keep_whole_table_rows = 20
    row_group_size = 20


class EventChunkingStrategy(ChunkingStrategy):
    """Keep major/exchange key-value event forms together."""

    name = "event"
    keep_whole_table_rows = 80
    row_group_size = 60


class HoldingChunkingStrategy(ChunkingStrategy):
    """Use stable row groups suitable for later holding timelines."""

    name = "holding"
    keep_whole_table_rows = 30
    row_group_size = 20


def get_chunking_strategy(doc_group: str | None) -> ChunkingStrategy:
    if doc_group == "periodic":
        return PeriodicChunkingStrategy()
    if doc_group in {"major", "exchange"}:
        return EventChunkingStrategy()
    if doc_group == "holding":
        return HoldingChunkingStrategy()
    return ChunkingStrategy()


def _normalized_key(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _is_meaningful_text(
    value: str, section_title: str, report_name: str = ""
) -> bool:
    text = value.strip()
    if not text or _ONLY_MARKUP.fullmatch(text):
        return False
    if not _MEANINGFUL_CHARACTER.search(text):
        return False
    if _normalized_key(text) == _normalized_key(section_title):
        return False
    if report_name and _normalized_key(text) == _normalized_key(report_name):
        return False
    if re.fullmatch(r"\(?\s*제\s*\d+\s*기(?:\s*(?:분기|반기))?\s*\)?", text):
        return False
    if re.search(r"\.(?:jpe?g|png|gif|bmp|tiff?)", text, re.IGNORECASE):
        return False
    compact = re.sub(r"\s+", "", text)
    if len(compact) >= 12 and len(set(compact)) <= 2:
        return False
    return True


def _split_sentences(paragraph: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(paragraph) if part.strip()]


def _trailing_sentence_overlap(
    sentences: list[str], overlap_chars: int
) -> list[str]:
    if overlap_chars <= 0:
        return []
    selected: list[str] = []
    size = 0
    for sentence in reversed(sentences):
        projected = size + len(sentence) + (1 if selected else 0)
        if projected > overlap_chars:
            break
        selected.append(sentence)
        size = projected
    return list(reversed(selected))


def _split_long_paragraph(
    paragraph: str, config: ChunkingConfig
) -> list[str]:
    """Split only at sentence boundaries; an indivisible long sentence stays whole."""

    if len(paragraph) <= config.max_chars:
        return [paragraph]
    sentences = _split_sentences(paragraph)
    if len(sentences) <= 1:
        return [paragraph]

    parts: list[str] = []
    buffer: list[str] = []
    for sentence in sentences:
        projected = len(" ".join([*buffer, sentence]))
        if buffer and projected > config.max_chars:
            parts.append(" ".join(buffer))
            overlap = _trailing_sentence_overlap(
                buffer, config.sentence_overlap_chars
            )
            buffer = [*overlap, sentence]
            if len(" ".join(buffer)) > config.max_chars and overlap:
                buffer = [sentence]
        else:
            buffer.append(sentence)
    if buffer:
        parts.append(" ".join(buffer))
    return parts


@dataclass
class _TextUnit:
    content: str
    block_start: int
    block_end: int
    paragraph_part: int
    starts_heading: bool = False
    context_only: bool = False


def _looks_like_heading(value: str) -> bool:
    text = value.strip()
    if not text or len(text) > 200:
        return False
    return bool(
        re.match(r"^(?:[가-힣]|\d+)\.\s*\S", text)
        or re.match(r"^\(\s*(?:\d+|[가-힣])\s*\)\s*\S", text)
        or re.fullmatch(r"【[^】]+】", text)
    )


def _has_independent_fact(value: str) -> bool:
    """Conservatively retain short key-value statements and numeric facts."""

    text = re.sub(r"\s+", " ", value).strip()
    if re.search(r"[:：]\s*\S", text) and not _find_unit(text):
        return True
    body = re.sub(
        r"^(?:[①-⑳]|\(?\d+(?:[-.]\d+)*\)?|[가-힣]\))\s*",
        "",
        text,
    )
    fact_label = re.search(
        r"(?:계약|금액|매출|주식|지분|비율|기간|일자|기준일|보고일|수량|"
        r"보유|증감|단가|이자율|상대방|목적|사유)",
        body,
    )
    fact_value = re.search(
        r"[-+△▲]?\(?\d[\d,]*(?:\.\d+)?\)?\s*"
        r"(?:%|원|억원|백만원|천원|주|건|명|배)?",
        body,
    )
    return bool(fact_label and fact_value)


def _is_context_only_text(value: str) -> bool:
    """Identify labels that need neighbouring evidence rather than their own chunk."""

    text = re.sub(r"\s+", " ", value).strip()
    if not text or len(text) > 100 or _has_independent_fact(text):
        return False
    if _find_unit(text):
        return True
    if re.fullmatch(r"[\[(【].{1,40}[\])】]", text):
        return True
    if re.fullmatch(
        r"\(?\s*(?:당기|전기|전기말|전분기|전반기|당분기|당반기|"
        r"국내|해외|제\s*\d+\s*기(?:\s*(?:분기|반기))?)\s*\)?",
        text,
    ):
        return True
    if re.fullmatch(r"-?\s*(?:해당사항(?:이)?\s*(?:없음|없습니다)\.?)", text):
        return True
    if _looks_like_heading(text):
        return True
    if re.match(
        r"^(?:[①-⑳]|\(?\d+(?:[-.]\d+)*\)?|[가-힣]\))\s*\S",
        text,
    ):
        return True
    return False


def _paragraph_units(
    section: Section,
    block_indexes: Iterable[int],
    config: ChunkingConfig,
    report_name: str = "",
) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    for block_index in block_indexes:
        if not (0 <= block_index < len(section.blocks)):
            continue
        paragraph = section.blocks[block_index].strip()
        if not _is_meaningful_text(paragraph, section.title, report_name):
            continue
        for part_index, content in enumerate(
            _split_long_paragraph(paragraph, config), start=1
        ):
            context_only = _is_context_only_text(content)
            units.append(
                _TextUnit(
                    content=content,
                    block_start=block_index,
                    block_end=block_index,
                    paragraph_part=part_index,
                    starts_heading=(
                        part_index == 1
                        and _looks_like_heading(paragraph)
                        and not context_only
                    ),
                    context_only=context_only,
                )
            )
    return units


def _merge_text_units(
    units: list[_TextUnit], config: ChunkingConfig
) -> list[dict[str, Any]]:
    """Merge short paragraphs without crossing the section or a table boundary."""

    merged: list[dict[str, Any]] = []
    buffer: list[_TextUnit] = []

    def flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        if all(unit.context_only for unit in buffer):
            buffer = []
            return
        merged.append(
            {
                "content": "\n\n".join(unit.content for unit in buffer),
                "block_start": buffer[0].block_start,
                "block_end": buffer[-1].block_end,
                "paragraph_part_start": buffer[0].paragraph_part,
                "paragraph_part_end": buffer[-1].paragraph_part,
                "context_labels": [
                    unit.content for unit in buffer if unit.context_only
                ],
            }
        )
        buffer = []

    for unit in units:
        if not buffer:
            buffer = [unit]
            continue
        if unit.starts_heading and any(not item.context_only for item in buffer):
            flush()
            buffer = [unit]
            continue
        current_size = len("\n\n".join(item.content for item in buffer))
        projected = len("\n\n".join(item.content for item in [*buffer, unit]))
        if current_size >= config.target_chars:
            flush()
            buffer = [unit]
        elif projected <= config.max_chars:
            buffer.append(unit)
        else:
            flush()
            buffer = [unit]
    flush()
    return merged


@dataclass
class LogicalTable:
    rows: list[list[str]]
    header_row_count: int
    header_row_indexes: list[int]
    column_headers: list[str]


def _expand_spans(table: Table, *, repeat_colspan: bool = False) -> list[list[str]]:
    """Expand rowspan/colspan into a rectangular logical grid."""

    sparse_rows: list[dict[int, str]] = []
    active: dict[int, tuple[str, int]] = {}
    width = 0
    for physical_row in table.rows:
        row: dict[int, str] = {}
        next_active: dict[int, tuple[str, int]] = {}
        for column, (text, remaining) in active.items():
            row[column] = text
            if remaining > 1:
                next_active[column] = (text, remaining - 1)
        active = next_active

        column = 0
        for cell in physical_row:
            while column in row:
                column += 1
            for offset in range(max(cell.colspan, 1)):
                target_column = column + offset
                value = cell.text if offset == 0 or repeat_colspan else ""
                row[target_column] = value
                if cell.rowspan > 1:
                    active[target_column] = (value, cell.rowspan - 1)
            column += max(cell.colspan, 1)
        width = max(width, max(row, default=-1) + 1)
        sparse_rows.append(row)
    return [
        [row.get(column, "") for column in range(width)] for row in sparse_rows
    ]


def _numeric_ratio(values: list[str]) -> float:
    populated = [value for value in values if value.strip()]
    if not populated:
        return 0.0
    numeric = sum(
        bool(
            re.fullmatch(
                r"[-+△▲]?\(?\s*[\d,]+(?:\.\d+)?\s*\)?(?:%|원|주|백만원|천원)?",
                value.strip(),
            )
        )
        for value in populated
    )
    return numeric / len(populated)


def _leading_context_row_count(rows: list[list[str]]) -> int:
    count = 0
    for row in rows[:3]:
        values = list(dict.fromkeys(value.strip() for value in row if value.strip()))
        joined = " ".join(values)
        if (
            _find_unit(joined)
            or _BASIS_PERIOD.search(joined)
            or (len(values) == 1 and re.fullmatch(r"【[^】]+】", joined))
        ):
            count += 1
        else:
            break
    return count


def _infer_header_rows(
    table: Table, rows: list[list[str]]
) -> tuple[int, list[int]]:
    context_count = _leading_context_row_count(rows)
    explicit = [
        index
        for index, row in enumerate(table.rows[:6])
        if any(cell.is_header for cell in row)
    ]
    if explicit:
        last_header = max(explicit)
        return last_header + 1, list(range(context_count, last_header + 1))
    if len(rows) - context_count < 2:
        return context_count, []

    inferred: list[int] = []
    for index in range(context_count, min(context_count + 2, len(rows) - 1)):
        row = rows[index]
        later = rows[index + 1 : min(len(rows), index + 5)]
        if (
            sum(bool(value.strip()) for value in row) >= 2
            and _numeric_ratio(row) < 0.5
            and any(_numeric_ratio(candidate) >= 0.5 for candidate in later)
        ):
            inferred.append(index)
        else:
            break
    data_start = inferred[-1] + 1 if inferred else context_count
    return data_start, inferred


def _column_headers(
    rows: list[list[str]], header_row_indexes: list[int]
) -> list[str]:
    if not rows or not header_row_indexes:
        return []
    headers: list[str] = []
    for column in range(len(rows[0])):
        values: list[str] = []
        for row_index in header_row_indexes:
            row = rows[row_index]
            value = row[column].strip()
            if value and value not in values:
                values.append(value)
        headers.append(" / ".join(values) or f"열 {column + 1}")
    return headers


def _logical_table(table: Table) -> LogicalTable:
    rows = _expand_spans(table)
    header_rows = _expand_spans(table, repeat_colspan=True)
    header_row_count, header_row_indexes = _infer_header_rows(table, rows)
    return LogicalTable(
        rows=rows,
        header_row_count=header_row_count,
        header_row_indexes=header_row_indexes,
        column_headers=_column_headers(header_rows, header_row_indexes),
    )


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def _markdown_row(row: list[str]) -> str:
    return "| " + " | ".join(_escape_markdown_cell(value) for value in row) + " |"


def _render_table_rows(logical: LogicalTable, row_indexes: list[int]) -> str:
    lines: list[str] = []
    if logical.column_headers:
        lines.append(_markdown_row(logical.column_headers))
        lines.append(_markdown_row(["---"] * len(logical.column_headers)))
    lines.extend(_markdown_row(logical.rows[index]) for index in row_indexes)
    return "\n".join(lines)


def _find_unit(context: str) -> str | None:
    for pattern in _UNIT_PATTERNS:
        match = pattern.search(context)
        if match:
            return match.group(1).strip(" .")
    return None


def _table_semantic_context(
    section: Section,
    table: Table,
    logical: LogicalTable,
    nearby_blocks: list[str],
) -> dict[str, Any]:
    row_preview = "\n".join(
        " | ".join(row) for row in logical.rows[: min(len(logical.rows), 5)]
    )
    context = "\n".join([*section.path, *nearby_blocks, row_preview])
    short_blocks = [
        block.strip()
        for block in nearby_blocks
        if 0 < len(block.strip()) <= 200 and _MEANINGFUL_CHARACTER.search(block)
    ]
    table_title = short_blocks[-1] if short_blocks else section.title
    if _find_unit(table_title):
        table_title = section.title

    has_consolidated = bool(re.search(r"연결(?:재무제표|기준|실적)?", context))
    has_separate = bool(re.search(r"별도(?:재무제표|기준|실적)?", context))
    statement_scope: str | None
    if has_consolidated and has_separate:
        statement_scope = "연결/별도"
    elif has_consolidated:
        statement_scope = "연결"
    elif has_separate:
        statement_scope = "별도"
    else:
        statement_scope = None

    basis_match = _BASIS_PERIOD.search(context)
    period_labels: list[str] = []
    for match in _PERIOD_LABEL.finditer(context):
        value = re.sub(r"\s+", " ", match.group(0)).strip()
        if value not in period_labels:
            period_labels.append(value)
        if len(period_labels) == 10:
            break
    return {
        "table_title": table_title,
        "unit": _find_unit(context),
        "statement_scope": statement_scope,
        "basis_period": basis_match.group(1).strip(" .") if basis_match else None,
        "period_labels": period_labels,
    }


def _is_context_only_table(
    logical: LogicalTable, context: dict[str, Any]
) -> bool:
    """Suppress only unambiguous unit/title wrappers; retain all evidence notes."""

    values = [
        value.strip()
        for row in logical.rows
        for value in row
        if value.strip()
    ]
    unique_values = list(dict.fromkeys(values))
    if len(logical.rows) > 2 or len(unique_values) > 2:
        return False
    joined = " ".join(unique_values)
    return bool(
        re.fullmatch(
            r"\(?\s*단위\s*[:：]?\s*[^)\n|]{1,40}\s*\)?",
            joined,
        )
        or re.fullmatch(r"【[^】]+】", joined)
    )


def _table_retrieval_annotation(logical: LogicalTable) -> tuple[str, str | None]:
    values = [
        value.strip()
        for row in logical.rows
        for value in row
        if value.strip()
    ]
    joined = " ".join(dict.fromkeys(values))
    if re.match(r"^[※*]\s*.+(?:참조|기재)\s*$", joined):
        return "low", "reference_note"
    if _BASIS_PERIOD.search(joined):
        return "normal", "basis_period_note"
    if len(logical.rows) <= 2 and len(values) <= 2 and not logical.column_headers:
        return "normal", "context_rule_without_intrinsic_marker"
    return "normal", None


def _physical_rows(table: Table, row_indexes: list[int]) -> list[list[dict[str, Any]]]:
    return [
        [cell.to_dict() for cell in table.rows[index]]
        for index in row_indexes
        if 0 <= index < len(table.rows)
    ]


def _explicit_column_headers(table: Table) -> tuple[list[str], list[int]]:
    indexes = [
        index
        for index, row in enumerate(table.rows)
        if any(cell.is_header for cell in row)
    ]
    if not indexes:
        return [], []
    expanded = _expand_spans(table, repeat_colspan=True)
    return _column_headers(expanded, indexes), indexes


def _first_nonempty_after(row: list[str], index: int) -> str | None:
    for value in row[index + 1 :]:
        if value.strip():
            return value.strip()
    return None


def _holding_document_context(table_map: dict[str, Table]) -> dict[str, Any]:
    context: dict[str, Any] = {"source_refs_by_field": {}}
    labels = {
        "보유목적": "holding_purpose",
        "보고사유": "change_reason",
        "변동사유": "change_reason",
    }
    for table in table_map.values():
        logical = _logical_table(table)
        for row_index, row in enumerate(logical.rows):
            for column, value in enumerate(row):
                key = labels.get(_normalized_key(value))
                if not key or context.get(key):
                    continue
                found = _first_nonempty_after(row, column)
                if found:
                    context[key] = found
                    context["source_refs_by_field"][key] = {
                        "table_id": table.table_id,
                        "row_start": row_index,
                        "row_end": row_index,
                    }
    return context


def _header_column(
    headers: list[str], *required_terms: str, excluded_terms: tuple[str, ...] = ()
) -> int | None:
    for index, header in enumerate(headers):
        compact = _normalized_key(header)
        if all(_normalized_key(term) in compact for term in required_terms) and not any(
            _normalized_key(term) in compact for term in excluded_terms
        ):
            return index
    return None


def _row_cell(row: list[str] | None, index: int | None) -> str | None:
    if row is None or index is None or not (0 <= index < len(row)):
        return None
    return row[index].strip() or None


def _holding_search_aliases(fields: list[tuple[str, Any]]) -> str:
    values = {label: str(value) for label, value in fields if value}
    aliases = [
        "이번보고서 이번보고 이번 보고 현재보고 현재 보고",
        "직전보고서 직전보고 직전 보고 대비",
        "보유주식 보유 주식 보유주식수 보유 주식수 주식수 주식수와 "
        "보유 수와 수량 수량과 비율",
        "증감주식 증감 주식 증감주식수 증감 주식수 증감 수량 증감 수량과",
    ]
    holder = values.get("보고자/보유자", "")
    for suffix in ("공단", "기금"):
        if holder.endswith(suffix) and len(holder) > len(suffix):
            aliases.append(holder[: -len(suffix)])
    change = values.get("증감주식수", "").replace(",", "")
    if change.startswith("-"):
        aliases.append("감소 감소 후 직전보고 대비 감소")
    elif change and change not in {"0", "0.0"}:
        aliases.append("증가 증가 후 직전보고 대비 증가")
    return " ".join(dict.fromkeys(" ".join(aliases).split()))


def _holding_report_projection(
    document: dict[str, Any],
    source_file: str,
    section: Section,
    table: Table,
    logical: LogicalTable,
    table_context: dict[str, Any],
    holding_context: dict[str, Any],
) -> dict[str, Any] | None:
    headers, header_indexes = _explicit_column_headers(table)
    if not headers or not any("보고서작성기준일" in _normalized_key(h) for h in headers):
        return None

    labelled_rows: dict[str, tuple[int, list[str]]] = {}
    for row_index in range(max(header_indexes, default=-1) + 1, len(logical.rows)):
        row = logical.rows[row_index]
        label = _normalized_key(row[0]) if row else ""
        if "직전보고서" in label:
            labelled_rows["previous"] = (row_index, row)
        elif "이번보고서" in label:
            labelled_rows["current"] = (row_index, row)
        elif label in {"증감", "증감합계"}:
            labelled_rows["change"] = (row_index, row)
    if "current" not in labelled_rows:
        return None

    previous = labelled_rows.get("previous", (-1, None))[1]
    current = labelled_rows["current"][1]
    change = labelled_rows.get("change", (-1, None))[1]
    date_column = _header_column(headers, "보고서작성기준일")
    holder_column = _header_column(headers, "보고자", "본인성명")
    shares_column = _header_column(headers, "주식등", "수", excluded_terms=("비율",))
    ratio_column = _header_column(headers, "주식등", "비율")
    fields = [
        ("보고자/보유자", _row_cell(current, holder_column)),
        ("기준일/보고일", _row_cell(current, date_column)),
        ("보유주식수", _row_cell(current, shares_column)),
        ("보유비율", _row_cell(current, ratio_column)),
        ("직전 보고일", _row_cell(previous, date_column)),
        ("직전 보유주식수", _row_cell(previous, shares_column)),
        ("직전 보유비율", _row_cell(previous, ratio_column)),
        ("증감주식수", _row_cell(change, shares_column)),
        ("증감비율", _row_cell(change, ratio_column)),
        ("보유 목적", holding_context.get("holding_purpose")),
        ("변동 사유", holding_context.get("change_reason")),
    ]
    content_lines = [f"[{label}] {value}" for label, value in fields if value]
    content_lines.append(f"[검색 표현] {_holding_search_aliases(fields)}")
    content = "\n".join(content_lines)
    if not content:
        return None
    row_indexes = sorted(item[0] for item in labelled_rows.values())
    current_ref = {
        "table_id": table.table_id,
        "row_start": labelled_rows["current"][0],
        "row_end": labelled_rows["current"][0],
    }
    previous_ref = (
        {
            "table_id": table.table_id,
            "row_start": labelled_rows["previous"][0],
            "row_end": labelled_rows["previous"][0],
        }
        if "previous" in labelled_rows
        else None
    )
    change_ref = (
        {
            "table_id": table.table_id,
            "row_start": labelled_rows["change"][0],
            "row_end": labelled_rows["change"][0],
        }
        if "change" in labelled_rows
        else None
    )
    context_refs = holding_context.get("source_refs_by_field", {})
    projection_field_refs = {
        "보고자/보유자": [current_ref],
        "기준일/보고일": [current_ref],
        "보유주식수": [current_ref],
        "보유비율": [current_ref],
        "직전 보고일": [previous_ref] if previous_ref else [],
        "직전 보유주식수": [previous_ref] if previous_ref else [],
        "직전 보유비율": [previous_ref] if previous_ref else [],
        "증감주식수": [change_ref] if change_ref else [],
        "증감비율": [change_ref] if change_ref else [],
        "보유 목적": [context_refs["holding_purpose"]]
        if context_refs.get("holding_purpose")
        else [],
        "변동 사유": [context_refs["change_reason"]]
        if context_refs.get("change_reason")
        else [],
    }
    projected_labels = {label for label, value in fields if value}
    projection_field_refs = {
        label: refs
        for label, refs in projection_field_refs.items()
        if label in projected_labels and refs
    }
    source_refs = list(
        {
            (str(ref["table_id"]), int(ref["row_start"]), int(ref["row_end"])): ref
            for refs in projection_field_refs.values()
            for ref in refs
        }.values()
    )
    return _base_chunk(
        document,
        source_file,
        section,
        "table_projection",
        content,
        retrieval_text=f"{_retrieval_prefix(document, section, table_context)}\n\n{content}",
        projection_type="holding_report",
        projection_fields={label: value for label, value in fields if value},
        projection_field_refs=projection_field_refs,
        table_id=table.table_id,
        source_table_id=table.table_id,
        source_table_ids=list(dict.fromkeys(ref["table_id"] for ref in source_refs)),
        source_refs=source_refs,
        row_start=row_indexes[0],
        row_end=row_indexes[-1],
        source_row_start=row_indexes[0],
        source_row_end=row_indexes[-1],
        column_headers=headers,
        header_rows=_physical_rows(table, header_indexes),
        table_rows=_physical_rows(table, row_indexes),
        table_title=table_context["table_title"],
        unit=table_context["unit"],
        statement_scope=table_context["statement_scope"],
        basis_period=table_context["basis_period"],
        period_labels=table_context["period_labels"],
        retrieval_priority="high",
        quality_flags=["retrieval_projection", "holding_report"],
        is_indexable=True,
    )


def _holding_detail_projections(
    document: dict[str, Any],
    source_file: str,
    section: Section,
    table: Table,
    logical: LogicalTable,
    table_context: dict[str, Any],
    holding_context: dict[str, Any],
) -> list[dict[str, Any]]:
    headers, header_indexes = _explicit_column_headers(table)
    compact_headers = [_normalized_key(header) for header in headers]
    if not headers or not (
        any("성명" in header or "명칭" in header for header in compact_headers)
        and any("변동일" in header for header in compact_headers)
        and any("변동내역" in header and "증감" in header for header in compact_headers)
    ):
        return []

    holder_column = next(
        (i for i, header in enumerate(compact_headers) if "성명" in header or "명칭" in header),
        None,
    )
    date_column = next((i for i, header in enumerate(compact_headers) if "변동일" in header), None)
    reason_column = next(
        (i for i, header in enumerate(compact_headers) if "취득처분방법" in header),
        None,
    )
    before_column = next(
        (i for i, header in enumerate(compact_headers) if "변동내역" in header and "변동전" in header),
        None,
    )
    change_column = next(
        (i for i, header in enumerate(compact_headers) if "변동내역" in header and "증감" in header),
        None,
    )
    after_column = next(
        (i for i, header in enumerate(compact_headers) if "변동내역" in header and "변동후" in header),
        None,
    )
    data_start = max(header_indexes, default=-1) + 1
    projections: list[dict[str, Any]] = []
    for row_index in range(data_start, len(logical.rows)):
        row = logical.rows[row_index]
        fields = [
            ("보고자/보유자", _row_cell(row, holder_column)),
            ("기준일/보고일", _row_cell(row, date_column)),
            ("변동 사유", _row_cell(row, reason_column)),
            ("직전 보유주식수", _row_cell(row, before_column)),
            ("증감주식수", _row_cell(row, change_column)),
            ("보유주식수", _row_cell(row, after_column)),
            ("보유 목적", holding_context.get("holding_purpose")),
        ]
        content_lines = [f"[{label}] {value}" for label, value in fields if value]
        content_lines.append(f"[검색 표현] {_holding_search_aliases(fields)}")
        content = "\n".join(content_lines)
        if not content:
            continue
        source_ref = {
            "table_id": table.table_id,
            "row_start": row_index,
            "row_end": row_index,
        }
        context_ref = holding_context.get("source_refs_by_field", {}).get(
            "holding_purpose"
        )
        projection_field_refs = {
            label: [source_ref]
            for label, value in fields
            if value and label != "보유 목적"
        }
        if holding_context.get("holding_purpose") and context_ref:
            projection_field_refs["보유 목적"] = [context_ref]
        source_refs = list(
            {
                (str(ref["table_id"]), int(ref["row_start"]), int(ref["row_end"])): ref
                for refs in projection_field_refs.values()
                for ref in refs
            }.values()
        )
        explicit_placeholder = all(
            str(value).strip() == "정정 전과 동일"
            for _, value in fields
            if value
        )
        projections.append(
            _base_chunk(
                document,
                source_file,
                section,
                "table_projection",
                content,
                retrieval_text=f"{_retrieval_prefix(document, section, table_context)}\n\n{content}",
                projection_type="holding_detail_row",
                projection_fields={label: value for label, value in fields if value},
                projection_field_refs=projection_field_refs,
                projection_state=(
                    "explicit_placeholder" if explicit_placeholder else "resolved"
                ),
                table_id=table.table_id,
                source_table_id=table.table_id,
                source_table_ids=list(
                    dict.fromkeys(str(ref["table_id"]) for ref in source_refs)
                ),
                source_refs=source_refs,
                row_start=row_index,
                row_end=row_index,
                source_row_start=row_index,
                source_row_end=row_index,
                column_headers=headers,
                header_rows=_physical_rows(table, header_indexes),
                table_rows=_physical_rows(table, [row_index]),
                table_title=table_context["table_title"],
                unit=table_context["unit"],
                statement_scope=table_context["statement_scope"],
                basis_period=table_context["basis_period"],
                period_labels=table_context["period_labels"],
                retrieval_priority="high",
                quality_flags=[
                    "retrieval_projection",
                    "holding_detail_row",
                    *(["explicit_placeholder"] if explicit_placeholder else []),
                ],
                is_indexable=True,
            )
        )
    return projections


def _projection_entry_parts(header: str, value: str) -> list[str]:
    prefix = f"[{header}] "
    limit = max(200, 1_500 - len(prefix))
    if len(value) <= limit:
        return [prefix + value]
    config = ChunkingConfig(
        target_chars=1_200,
        min_chars=700,
        max_chars=1_500,
        sentence_overlap_chars=0,
    )
    sentence_parts = _split_long_paragraph(value, config)
    parts: list[str] = []
    for sentence_part in sentence_parts:
        if len(sentence_part) <= limit:
            parts.append(sentence_part)
            continue
        words = sentence_part.split()
        buffer: list[str] = []
        for word in words or [sentence_part]:
            if len(word) > limit:
                if buffer:
                    parts.append(" ".join(buffer))
                    buffer = []
                parts.extend(
                    word[index : index + limit]
                    for index in range(0, len(word), limit)
                )
                continue
            if buffer and len(" ".join([*buffer, word])) > limit:
                parts.append(" ".join(buffer))
                buffer = [word]
            else:
                buffer.append(word)
        if buffer:
            parts.append(" ".join(buffer))
    return [prefix + part for part in parts]


def _extreme_table_projections(
    document: dict[str, Any],
    source_file: str,
    section: Section,
    table: Table,
    logical: LogicalTable,
    table_context: dict[str, Any],
    row_indexes: list[int],
    source_content_length: int,
) -> list[dict[str, Any]]:
    del source_content_length  # The source chunk is audited separately and remains intact.
    extreme_row_indexes = [
        index
        for index in row_indexes
        if len(_markdown_row(logical.rows[index])) > 5_000
    ]
    if not extreme_row_indexes:
        return []
    headers = logical.column_headers or [
        f"열 {index + 1}" for index in range(len(logical.rows[0]))
    ]
    projections: list[dict[str, Any]] = []
    for row_index in extreme_row_indexes:
        entries: list[tuple[str, str]] = []
        for column, value in enumerate(logical.rows[row_index]):
            value = value.strip()
            if not value:
                continue
            header = headers[column] if column < len(headers) else f"열 {column + 1}"
            entries.extend((header, part) for part in _projection_entry_parts(header, value))
        entries = list(dict.fromkeys(entries))
        groups: list[list[tuple[str, str]]] = []
        buffer: list[tuple[str, str]] = []
        size = 0
        for header, entry in entries:
            projected = size + len(entry) + (1 if buffer else 0)
            if buffer and projected > 1_500:
                groups.append(buffer)
                buffer = []
                size = 0
            buffer.append((header, entry))
            size += len(entry) + (1 if size else 0)
        if buffer:
            groups.append(buffer)
        for part_index, group in enumerate(groups, start=1):
            content = "\n".join(entry for _, entry in group)
            source_ref = {
                "table_id": table.table_id,
                "row_start": row_index,
                "row_end": row_index,
            }
            projections.append(
                _base_chunk(
                    document,
                    source_file,
                    section,
                    "table_projection",
                    content,
                    retrieval_text=f"{_retrieval_prefix(document, section, table_context)}\n\n{content}",
                    projection_type="extreme_table_row",
                    projection_part=part_index,
                    table_id=table.table_id,
                    source_table_id=table.table_id,
                    source_table_ids=[table.table_id],
                    source_refs=[source_ref],
                    row_start=row_index,
                    row_end=row_index,
                    source_row_start=row_index,
                    source_row_end=row_index,
                    column_headers=list(dict.fromkeys(header for header, _ in group)),
                    header_rows=_physical_rows(table, logical.header_row_indexes),
                    table_rows=_physical_rows(table, [row_index]),
                    table_title=table_context["table_title"],
                    unit=table_context["unit"],
                    statement_scope=table_context["statement_scope"],
                    basis_period=table_context["basis_period"],
                    period_labels=table_context["period_labels"],
                    retrieval_priority="normal",
                    quality_flags=["retrieval_projection", "extreme_table_row"],
                    is_indexable=True,
                )
            )
    return projections


def _retrieval_prefix(
    document: dict[str, Any], section: Section, table_context: dict[str, Any] | None = None
) -> str:
    lines = [
        f"[기업명] {document.get('corp_name') or ''}",
        f"[공시명] {document.get('report_nm') or ''}",
        f"[Section Path] {' > '.join(section.path)}",
    ]
    if table_context is not None:
        lines.append(f"[Table] {table_context['table_title']}")
        if table_context.get("statement_scope"):
            lines.append(f"[재무제표 범위] {table_context['statement_scope']}")
        if table_context.get("unit"):
            lines.append(f"[단위] {table_context['unit']}")
        if table_context.get("basis_period"):
            lines.append(f"[기준기간] {table_context['basis_period']}")
        if table_context.get("period_labels"):
            lines.append(
                f"[기간표현] {', '.join(table_context['period_labels'])}"
            )
    return "\n".join(lines)


def _base_chunk(
    document: dict[str, Any],
    source_file: str,
    section: Section,
    chunk_type: str,
    content: str,
    **extra: Any,
) -> dict[str, Any]:
    chunk = {field: document.get(field) for field in DOCUMENT_METADATA_FIELDS}
    chunk.update(
        {
            "kind": chunk_type,  # Backward-compatible alias.
            "chunk_type": chunk_type,
            "section_id": section.section_id,
            "parent_section_id": section.parent_id,
            "section_title": section.title,
            "section_path": section.path,
            "section_depth": len(section.path),
            "source_file": source_file,
            "content": content,
            "char_count": len(content),
            **extra,
        }
    )
    return chunk


def _table_nearby_blocks(section: Section) -> dict[str, list[str]]:
    contexts: dict[str, list[str]] = {}
    recent: list[str] = []
    events = section.content_order or [
        {"kind": "text", "block_index": index}
        for index in range(len(section.blocks))
    ]
    for event in events:
        if event.get("kind") == "text":
            block_index = int(event["block_index"])
            if 0 <= block_index < len(section.blocks):
                recent.append(section.blocks[block_index])
                recent = recent[-3:]
        elif event.get("kind") == "table":
            contexts[str(event["table_id"])] = list(recent)
    return contexts


def _text_chunks_for_section(
    document: dict[str, Any],
    source_file: str,
    section: Section,
    config: ChunkingConfig,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    pending: list[int] = []

    def flush() -> None:
        nonlocal pending
        units = _paragraph_units(
            section,
            pending,
            config,
            report_name=str(document.get("report_nm") or ""),
        )
        for item in _merge_text_units(units, config):
            content = item.pop("content")
            chunks.append(
                _base_chunk(
                    document,
                    source_file,
                    section,
                    "text",
                    content,
                    retrieval_text=(
                        f"{_retrieval_prefix(document, section)}\n\n{content}"
                    ),
                    **item,
                )
            )
        pending = []

    events = section.content_order or [
        {"kind": "text", "block_index": index}
        for index in range(len(section.blocks))
    ]
    for event_index, event in enumerate(events):
        if event.get("kind") == "text":
            block_index = int(event["block_index"])
            next_is_table = (
                event_index + 1 < len(events)
                and events[event_index + 1].get("kind") == "table"
            )
            block = (
                section.blocks[block_index]
                if 0 <= block_index < len(section.blocks)
                else ""
            )
            table_label = bool(
                len(block) <= 200
                and (
                    _looks_like_heading(block)
                    or re.fullmatch(r"【[^】]+】", block)
                    or _find_unit(block)
                )
            )
            if not (next_is_table and table_label):
                pending.append(block_index)
        elif event.get("kind") == "table":
            flush()
    flush()
    return chunks


def _table_chunks_for_section(
    document: dict[str, Any],
    source_file: str,
    section: Section,
    table_map: dict[str, Table],
    strategy: ChunkingStrategy,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    nearby = _table_nearby_blocks(section)
    holding_context = (
        _holding_document_context(table_map)
        if document.get("doc_group") == "holding"
        else {}
    )
    events = section.content_order or []
    table_ids = [
        str(event["table_id"])
        for event in events
        if event.get("kind") == "table"
    ] or list(section.table_ids)
    carried_context: list[str] = []
    for table_id in table_ids:
        table = table_map.get(table_id)
        if table is None:
            continue
        logical = _logical_table(table)
        if not logical.rows:
            continue
        context_blocks = [*nearby.get(table_id, []), *carried_context]
        context = _table_semantic_context(
            section, table, logical, context_blocks
        )
        if _is_context_only_table(logical, context):
            carried_context.extend(
                value
                for row in logical.rows
                for value in row
                if value.strip()
            )
            carried_context = carried_context[-4:]
            continue
        carried_context = []
        retrieval_priority, former_exclusion_reason = _table_retrieval_annotation(
            logical
        )
        data_indexes = list(range(logical.header_row_count, len(logical.rows)))
        header_only = not data_indexes and bool(logical.column_headers)
        groups = strategy.table_row_groups(data_indexes)
        if len(groups) > 1 and not logical.column_headers:
            logical.column_headers = [
                f"열 {index + 1}" for index in range(len(logical.rows[0]))
            ]
        for row_indexes in groups:
            rendered_indexes = row_indexes
            if not rendered_indexes and not logical.column_headers:
                rendered_indexes = list(range(len(logical.rows)))
            content = _render_table_rows(logical, rendered_indexes)
            if not content.strip():
                continue
            row_start = rendered_indexes[0] if rendered_indexes else 0
            row_end = rendered_indexes[-1] if rendered_indexes else max(
                logical.header_row_count - 1, 0
            )
            quality_flags = ["header_only"] if header_only else []
            source_ref = {
                "table_id": table.table_id,
                "row_start": row_start,
                "row_end": row_end,
            }
            chunk = _base_chunk(
                document,
                source_file,
                section,
                "table",
                content,
                retrieval_text=(
                    f"{_retrieval_prefix(document, section, context)}\n\n{content}"
                ),
                table_id=table.table_id,
                source_table_id=table.table_id,
                source_table_ids=[table.table_id],
                source_refs=[source_ref],
                table_title=context["table_title"],
                row_start=row_start,
                row_end=row_end,
                column_headers=logical.column_headers,
                unit=context["unit"],
                statement_scope=context["statement_scope"],
                basis_period=context["basis_period"],
                period_labels=context["period_labels"],
                retrieval_priority=retrieval_priority,
                former_exclusion_reason=former_exclusion_reason,
                header_rows=_physical_rows(table, logical.header_row_indexes),
                table_rows=_physical_rows(table, rendered_indexes),
                quality_flags=quality_flags,
                is_indexable=not header_only,
            )
            chunks.append(chunk)
            chunks.extend(
                _extreme_table_projections(
                    document,
                    source_file,
                    section,
                    table,
                    logical,
                    context,
                    rendered_indexes,
                    len(content),
                )
            )
        if document.get("doc_group") == "holding":
            report_projection = _holding_report_projection(
                document,
                source_file,
                section,
                table,
                logical,
                context,
                holding_context,
            )
            if report_projection is not None:
                chunks.append(report_projection)
            chunks.extend(
                _holding_detail_projections(
                    document,
                    source_file,
                    section,
                    table,
                    logical,
                    context,
                    holding_context,
                )
            )
    return chunks


def _chunk_identity(chunk: dict[str, Any]) -> str:
    identity = {
        "doc_id": chunk.get("doc_id"),
        "source_file": chunk.get("source_file"),
        "section_id": chunk.get("section_id"),
        "chunk_type": chunk.get("chunk_type"),
        "table_id": chunk.get("table_id"),
        "row_start": chunk.get("row_start"),
        "row_end": chunk.get("row_end"),
        "block_start": chunk.get("block_start"),
        "block_end": chunk.get("block_end"),
        "paragraph_part_start": chunk.get("paragraph_part_start"),
        "paragraph_part_end": chunk.get("paragraph_part_end"),
        "projection_type": chunk.get("projection_type"),
        "projection_part": chunk.get("projection_part"),
        "content": chunk.get("content"),
    }
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def _finalize_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique_chunks: list[dict[str, Any]] = []
    seen_content: set[tuple[Any, ...]] = set()
    for chunk in chunks:
        is_projection = chunk.get("chunk_type") == "table_projection"
        duplicate_key = (
            chunk.get("doc_id"),
            chunk.get("source_file"),
            chunk.get("chunk_type"),
            tuple(chunk.get("section_path") or []),
            chunk.get("table_id") if is_projection else None,
            chunk.get("row_start") if is_projection else None,
            chunk.get("row_end") if is_projection else None,
            chunk.get("projection_type") if is_projection else None,
            chunk.get("projection_part") if is_projection else None,
            str(chunk.get("content", "")).strip(),
        )
        if duplicate_key in seen_content:
            continue
        seen_content.add(duplicate_key)
        unique_chunks.append(chunk)

    for order, chunk in enumerate(unique_chunks, start=1):
        chunk["chunk_order"] = order
        chunk["chunk_id"] = f"{chunk['doc_id']}:ch_{_chunk_identity(chunk)}"
        chunk.setdefault("quality_flags", [])
        chunk.setdefault("is_indexable", True)
        chunk["duplicate_of_chunk_id"] = None

    for index, chunk in enumerate(unique_chunks):
        chunk["prev_chunk_id"] = (
            unique_chunks[index - 1]["chunk_id"] if index else None
        )
        chunk["next_chunk_id"] = (
            unique_chunks[index + 1]["chunk_id"]
            if index + 1 < len(unique_chunks)
            else None
        )
    return unique_chunks


def build_chunks(
    doc_id: str,
    parsed: ParsedDocument,
    max_chars: int = 1_500,
    overlap: int = 120,
    *,
    document_metadata: dict[str, Any] | None = None,
    source_file: str = "",
    target_chars: int = 1_200,
    min_chars: int = 700,
) -> list[dict[str, Any]]:
    """Build hierarchical text/table chunks without crossing structural boundaries."""

    document = dict(document_metadata or {})
    document["doc_id"] = doc_id
    effective_target = min(target_chars, max_chars)
    effective_min = min(min_chars, effective_target)
    config = ChunkingConfig(
        target_chars=effective_target,
        min_chars=effective_min,
        max_chars=max_chars,
        sentence_overlap_chars=overlap,
    )
    strategy = get_chunking_strategy(document.get("doc_group"))
    table_map = {table.table_id: table for table in parsed.tables}
    chunks: list[dict[str, Any]] = []

    for section in parsed.sections:
        # Reconstruct the parser's original text/table order at chunk granularity.
        text_chunks = _text_chunks_for_section(
            document, source_file, section, config
        )
        table_chunks = _table_chunks_for_section(
            document, source_file, section, table_map, strategy
        )
        text_by_start: dict[int, list[dict[str, Any]]] = {}
        for chunk in text_chunks:
            text_by_start.setdefault(int(chunk["block_start"]), []).append(chunk)
        tables_by_id: dict[str, list[dict[str, Any]]] = {}
        for chunk in table_chunks:
            tables_by_id.setdefault(str(chunk["table_id"]), []).append(chunk)

        emitted_text: set[int] = set()
        events = section.content_order or []
        for event in events:
            if event.get("kind") == "text":
                block_index = int(event["block_index"])
                for start, grouped_chunks in text_by_start.items():
                    if start <= block_index <= int(grouped_chunks[-1]["block_end"]):
                        if start not in emitted_text:
                            chunks.extend(grouped_chunks)
                            emitted_text.add(start)
                        break
            elif event.get("kind") == "table":
                chunks.extend(tables_by_id.pop(str(event["table_id"]), []))
        for start, grouped_chunks in text_by_start.items():
            if start not in emitted_text:
                chunks.extend(grouped_chunks)
        for remaining in tables_by_id.values():
            chunks.extend(remaining)

    return _finalize_chunks(chunks)


# The original fixed-character implementation is retained only for the pilot comparison.
def _legacy_split_text(
    text: str, max_chars: int = 1_200, overlap: int = 150
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            floor = start + max_chars // 2
            candidates = [
                text.rfind("\n\n", floor, hard_end),
                text.rfind("다. ", floor, hard_end),
                text.rfind(". ", floor, hard_end),
                text.rfind(" ", floor, hard_end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text[boundary : boundary + 2] == "다." else 1)
        content = text[start:end].strip()
        if content:
            chunks.append(content)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1
        start = next_start if next_start < end else end
    return chunks


def build_legacy_chunks(
    doc_id: str,
    parsed: ParsedDocument,
    max_chars: int = 1_200,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    """Exact legacy behavior used to quantify the 20-document pilot change."""

    chunks: list[dict[str, Any]] = []
    section_map = parsed.section_map()
    table_map = {table.table_id: table for table in parsed.tables}

    def add(kind: str, content: str, section_id: str, **extra: Any) -> None:
        chunks.append(
            {
                "chunk_id": f"{doc_id}:c{len(chunks) + 1:05d}",
                "kind": kind,
                "section_id": section_id,
                "section_path": section_map[section_id].path,
                "content": content,
                "char_count": len(content),
                **extra,
            }
        )

    def add_table(table: Table) -> None:
        lines = [
            " | ".join(
                (
                    f"{cell.text} [rowspan={cell.rowspan}, colspan={cell.colspan}]"
                    if cell.rowspan > 1 or cell.colspan > 1
                    else cell.text
                ).strip()
                for cell in row
            )
            for row in table.rows
        ]
        row_start = 0
        buffer: list[str] = []
        size = 0

        def flush(row_end: int) -> None:
            nonlocal buffer, size
            if buffer:
                add(
                    "table",
                    "\n".join(buffer),
                    table.section_id,
                    table_id=table.table_id,
                    row_start=row_start,
                    row_end=row_end,
                )
            buffer = []
            size = 0

        for row_index, line in enumerate(lines):
            if len(line) > max_chars:
                flush(row_index - 1)
                for part in _legacy_split_text(line, max_chars, 0):
                    row_start = row_index
                    buffer = [part]
                    size = len(part)
                    flush(row_index)
                row_start = row_index + 1
                continue
            projected = size + len(line) + (1 if buffer else 0)
            if buffer and projected > max_chars:
                flush(row_index - 1)
                row_start = row_index
            if not buffer:
                row_start = row_index
            buffer.append(line)
            size += len(line) + (1 if size else 0)
        flush(len(lines) - 1)

    for section in parsed.sections:
        pending: list[str] = []
        part_index = 0

        def flush_text() -> None:
            nonlocal pending, part_index
            for content in _legacy_split_text(
                "\n\n".join(pending), max_chars, overlap
            ):
                part_index += 1
                add(
                    "text",
                    content,
                    section.section_id,
                    part_index=part_index,
                )
            pending = []

        events = section.content_order or [
            {"kind": "text", "block_index": index}
            for index in range(len(section.blocks))
        ]
        for event in events:
            if event["kind"] == "text":
                block_index = int(event["block_index"])
                if 0 <= block_index < len(section.blocks):
                    pending.append(section.blocks[block_index])
            elif event["kind"] == "table":
                flush_text()
                table = table_map.get(str(event["table_id"]))
                if table is not None:
                    add_table(table)
        flush_text()
    return chunks
