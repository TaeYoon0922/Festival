"""Create text and table chunks from parsed disclosures."""

from __future__ import annotations

from typing import Any

from app.parsing.models import ParsedDocument, Table, TableCell


def split_text(text: str, max_chars: int = 1_200, overlap: int = 150) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be between 0 and max_chars - 1")

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        hard_end = min(start + max_chars, length)
        end = hard_end
        if hard_end < length:
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

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break

        next_start = max(end - overlap, start + 1)
        while next_start < end and not text[next_start].isspace():
            next_start += 1
        while next_start < length and text[next_start].isspace():
            next_start += 1
        start = next_start if next_start < end else end

    return chunks


def _cell_text(cell: TableCell) -> str:
    span = ""
    if cell.rowspan > 1 or cell.colspan > 1:
        span = f" [rowspan={cell.rowspan}, colspan={cell.colspan}]"
    return f"{cell.text}{span}".strip()


def _table_lines(table: Table) -> list[str]:
    return [" | ".join(_cell_text(cell) for cell in row) for row in table.rows]


def build_chunks(
    doc_id: str,
    parsed: ParsedDocument,
    max_chars: int = 1_200,
    overlap: int = 150,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    section_map = parsed.section_map()
    table_map = {table.table_id: table for table in parsed.tables}

    def add_chunk(
        kind: str,
        content: str,
        section_id: str,
        **extra: Any,
    ) -> None:
        section = section_map[section_id]
        chunks.append(
            {
                "chunk_id": f"{doc_id}:c{len(chunks) + 1:05d}",
                "kind": kind,
                "section_id": section_id,
                "section_path": section.path,
                "content": content,
                "char_count": len(content),
                **extra,
            }
        )

    def add_table_chunks(table: Table) -> None:
        lines = _table_lines(table)
        row_start = 0
        buffer: list[str] = []
        buffer_size = 0

        def flush(row_end: int) -> None:
            nonlocal row_start, buffer, buffer_size
            if not buffer:
                return
            add_chunk(
                "table",
                "\n".join(buffer),
                table.section_id,
                table_id=table.table_id,
                row_start=row_start,
                row_end=row_end,
            )
            buffer = []
            buffer_size = 0

        for row_index, line in enumerate(lines):
            if len(line) > max_chars:
                flush(row_index - 1)
                for part in split_text(line, max_chars=max_chars, overlap=0):
                    row_start = row_index
                    buffer = [part]
                    buffer_size = len(part)
                    flush(row_index)
                row_start = row_index + 1
                continue
            projected = buffer_size + len(line) + (1 if buffer else 0)
            if buffer and projected > max_chars:
                flush(row_index - 1)
                row_start = row_index
            if not buffer:
                row_start = row_index
            buffer.append(line)
            buffer_size += len(line) + (1 if buffer_size else 0)
        flush(len(lines) - 1)

    for section in parsed.sections:
        pending_blocks: list[str] = []
        text_part_index = 0

        def flush_text_blocks() -> None:
            nonlocal pending_blocks, text_part_index
            if not pending_blocks:
                return
            text = "\n\n".join(pending_blocks)
            for content in split_text(
                text, max_chars=max_chars, overlap=overlap
            ):
                text_part_index += 1
                add_chunk(
                    "text",
                    content,
                    section.section_id,
                    part_index=text_part_index,
                )
            pending_blocks = []

        events = section.content_order or [
            {"kind": "text", "block_index": index}
            for index in range(len(section.blocks))
        ]
        for event in events:
            if event["kind"] == "text":
                block_index = int(event["block_index"])
                if 0 <= block_index < len(section.blocks):
                    pending_blocks.append(section.blocks[block_index])
            elif event["kind"] == "table":
                flush_text_blocks()
                table = table_map.get(str(event["table_id"]))
                if table is not None:
                    add_table_chunks(table)
        flush_text_blocks()

    return chunks
