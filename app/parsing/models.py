"""Data models produced by the disclosure parser."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Section:
    section_id: str
    title: str
    level: int
    parent_id: str | None
    path: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    table_ids: list[str] = field(default_factory=list)
    content_order: list[dict[str, Any]] = field(default_factory=list)
    _fragments: list[str] = field(default_factory=list, repr=False)

    def append(self, value: str) -> None:
        self._fragments.append(value)

    def break_block(self, normalizer: Any) -> None:
        if not self._fragments:
            return
        text = normalizer("".join(self._fragments))
        self._fragments.clear()
        if text and (not self.blocks or self.blocks[-1] != text):
            self.blocks.append(text)
            self.content_order.append(
                {"kind": "text", "block_index": len(self.blocks) - 1}
            )

    @property
    def text(self) -> str:
        return "\n\n".join(self.blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "level": self.level,
            "parent_id": self.parent_id,
            "path": self.path,
            "text": self.text,
            "table_ids": self.table_ids,
            "content_order": self.content_order,
        }


@dataclass
class TableCell:
    text: str
    is_header: bool = False
    rowspan: int = 1
    colspan: int = 1
    source_tag: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Table:
    table_id: str
    section_id: str
    rows: list[list[TableCell]]
    attributes: dict[str, str] = field(default_factory=dict)

    def to_dict(self, section_path: list[str]) -> dict[str, Any]:
        return {
            "table_id": self.table_id,
            "section_id": self.section_id,
            "section_path": section_path,
            "attributes": self.attributes,
            "rows": [[cell.to_dict() for cell in row] for row in self.rows],
        }


@dataclass
class ParsedDocument:
    document_title: str
    sections: list[Section]
    tables: list[Table]
    parser_warnings: list[str] = field(default_factory=list)

    def section_map(self) -> dict[str, Section]:
        return {section.section_id: section for section in self.sections}
