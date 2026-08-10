"""Tolerant parser for DART XML and exchange-disclosure HTML files."""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

from app.parsing.models import ParsedDocument, Section, Table, TableCell


SECTION_TAG = re.compile(r"section-(\d+)$")
BLOCK_TAGS = {"p", "div", "li", "br", "pgbrk", "blockquote"}
CELL_TAGS = {"td", "th", "tu", "te", "ti", "tq", "tx"}
IGNORED_TAGS = {"script", "style"}


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value.replace("\xa0", " "))
    return re.sub(r"\s+", " ", value).strip()


def _positive_int(value: str | None, default: int = 1) -> int:
    try:
        number = int(value or default)
    except (TypeError, ValueError):
        return default
    return max(number, 1)


class _TableBuilder:
    def __init__(
        self,
        table_id: str,
        section_id: str,
        attributes: dict[str, str],
    ) -> None:
        self.table_id = table_id
        self.section_id = section_id
        self.attributes = attributes
        self.rows: list[list[TableCell]] = []
        self.current_row: list[TableCell] | None = None
        self.current_cell: dict[str, object] | None = None

    def start_row(self) -> None:
        self.end_row()
        self.current_row = []

    def end_row(self) -> None:
        self.end_cell()
        if self.current_row is not None:
            if any(cell.text for cell in self.current_row):
                self.rows.append(self.current_row)
            self.current_row = None

    def start_cell(self, tag: str, attributes: dict[str, str]) -> None:
        if self.current_row is None:
            self.current_row = []
        self.end_cell()
        self.current_cell = {
            "tag": tag,
            "fragments": [],
            "rowspan": _positive_int(attributes.get("rowspan")),
            "colspan": _positive_int(attributes.get("colspan")),
        }

    def append(self, value: str) -> None:
        if self.current_cell is not None:
            fragments = self.current_cell["fragments"]
            assert isinstance(fragments, list)
            fragments.append(value)

    def end_cell(self) -> None:
        if self.current_cell is None:
            return
        fragments = self.current_cell["fragments"]
        assert isinstance(fragments, list)
        cell = TableCell(
            text=normalize_text("".join(fragments)),
            is_header=self.current_cell["tag"] == "th",
            rowspan=int(self.current_cell["rowspan"]),
            colspan=int(self.current_cell["colspan"]),
        )
        assert self.current_row is not None
        self.current_row.append(cell)
        self.current_cell = None

    def finish(self) -> Table | None:
        self.end_row()
        if not self.rows:
            return None
        return Table(
            table_id=self.table_id,
            section_id=self.section_id,
            rows=self.rows,
            attributes=self.attributes,
        )


class DartDocumentParser(HTMLParser):
    """Parse both well-formed DART XML and tag-soup exchange HTML."""

    def __init__(self, fallback_title: str = "공시 문서") -> None:
        super().__init__(convert_charrefs=True)
        self.fallback_title = normalize_text(fallback_title) or "공시 문서"
        self.document_title = self.fallback_title
        self.sections: list[Section] = []
        self.tables: list[Table] = []
        self.warnings: list[str] = []

        self._section_by_id: dict[str, Section] = {}
        self._structural_stack: list[dict[str, object]] = []
        self._standalone_section_id: str | None = None
        self._table_stack: list[_TableBuilder] = []
        self._title_kind: str | None = None
        self._title_fragments: list[str] = []
        self._title_attributes: dict[str, str] = {}
        self._in_head = False
        self._in_body = False
        self._ignored_depth = 0
        self._section_counter = 0
        self._table_counter = 0

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        tag = tag.lower()
        attributes = {key.lower(): value or "" for key, value in attrs}

        if tag == "head":
            self._in_head = True
        elif tag == "body":
            self._in_body = True

        if tag in IGNORED_TAGS:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return

        if tag == "title" and self._in_head and not self._in_body:
            self._begin_title("document", attributes)
            return
        if not self._in_body:
            return

        match = SECTION_TAG.fullmatch(tag)
        if match:
            parent_id = self._nearest_structural_section_id()
            self._structural_stack.append(
                {
                    "tag": tag,
                    "level": int(match.group(1)),
                    "section_id": None,
                    "inline_section_id": None,
                    "parent_id": parent_id,
                }
            )
            return

        if tag in {"title", "cover-title"} and not self._table_stack:
            self._break_current_section()
            self._begin_title("cover" if tag == "cover-title" else "section", attributes)
            return

        if tag == "table":
            self._break_current_section()
            section_id = self._ensure_current_section()
            self._table_counter += 1
            self._table_stack.append(
                _TableBuilder(
                    table_id=f"t{self._table_counter:04d}",
                    section_id=section_id,
                    attributes=attributes,
                )
            )
            return

        if self._table_stack:
            table = self._table_stack[-1]
            if tag == "tr":
                table.start_row()
            elif tag in CELL_TAGS and table.current_cell is None:
                table.start_cell(tag, attributes)
            elif tag == "br":
                for active_table in self._table_stack:
                    active_table.append(" ")
            return

        if tag in BLOCK_TAGS:
            self._break_current_section()

    def handle_startendtag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()

        if tag in IGNORED_TAGS:
            self._ignored_depth = max(0, self._ignored_depth - 1)
            return
        if self._ignored_depth:
            return

        if self._title_kind and tag in {"title", "cover-title"}:
            self._finish_title()
            return

        if self._table_stack:
            table = self._table_stack[-1]
            if (
                tag in CELL_TAGS
                and table.current_cell is not None
                and table.current_cell["tag"] == tag
            ):
                table.end_cell()
                return
            if tag == "tr":
                table.end_row()
                return
            if tag == "table":
                finished = self._table_stack.pop().finish()
                if finished is not None:
                    self.tables.append(finished)
                    section = self._section_by_id[finished.section_id]
                    section.table_ids.append(finished.table_id)
                    section.content_order.append(
                        {"kind": "table", "table_id": finished.table_id}
                    )
                return

        match = SECTION_TAG.fullmatch(tag)
        if match and self._structural_stack:
            self._break_current_section()
            for index in range(len(self._structural_stack) - 1, -1, -1):
                if self._structural_stack[index]["tag"] == tag:
                    del self._structural_stack[index:]
                    break
            return

        if tag in BLOCK_TAGS:
            self._break_current_section()
        elif tag == "body":
            self._break_current_section()
            self._in_body = False
        elif tag == "head":
            self._in_head = False

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._title_kind:
            self._title_fragments.append(data)
            return
        if not self._in_body:
            return
        if self._table_stack:
            for table in self._table_stack:
                table.append(data)
            return
        if data.strip():
            self._section_by_id[self._ensure_current_section()].append(data)

    def _begin_title(self, kind: str, attributes: dict[str, str]) -> None:
        self._title_kind = kind
        self._title_fragments = []
        self._title_attributes = attributes

    def _finish_title(self) -> None:
        kind = self._title_kind
        title = normalize_text("".join(self._title_fragments))
        self._title_kind = None
        self._title_fragments = []
        self._title_attributes = {}
        if not title:
            return

        if kind == "document":
            self.document_title = title
            return
        if kind == "cover":
            self._standalone_section_id = self._new_section(
                title=title, level=0, parent_id=None
            )
            return

        if self._structural_stack:
            context = self._structural_stack[-1]
            if context["section_id"] is None:
                context["section_id"] = self._new_section(
                    title=title,
                    level=int(context["level"]),
                    parent_id=context["parent_id"],
                )
                context["inline_section_id"] = None
            else:
                context["inline_section_id"] = self._new_section(
                    title=title,
                    level=int(context["level"]) + 1,
                    parent_id=str(context["section_id"]),
                )
            return

        self._standalone_section_id = self._new_section(
            title=title, level=1, parent_id=None
        )

    def _new_section(self, title: str, level: int, parent_id: str | None) -> str:
        self._section_counter += 1
        section_id = f"s{self._section_counter:04d}"
        section = Section(
            section_id=section_id,
            title=normalize_text(title) or "제목 없음",
            level=level,
            parent_id=parent_id,
        )
        self.sections.append(section)
        self._section_by_id[section_id] = section
        return section_id

    def _nearest_structural_section_id(self) -> str | None:
        for context in reversed(self._structural_stack):
            if context["section_id"] is not None:
                return str(context["section_id"])
        return None

    def _ensure_current_section(self) -> str:
        if self._structural_stack:
            context = self._structural_stack[-1]
            inline_id = context["inline_section_id"]
            if inline_id is not None:
                return str(inline_id)
            section_id = context["section_id"]
            if section_id is None:
                section_id = self._new_section(
                    title="제목 없음",
                    level=int(context["level"]),
                    parent_id=context["parent_id"],
                )
                context["section_id"] = section_id
            return str(section_id)
        if self._standalone_section_id is None:
            self._standalone_section_id = self._new_section(
                title=self.document_title or self.fallback_title,
                level=1,
                parent_id=None,
            )
        return self._standalone_section_id

    def _break_current_section(self) -> None:
        section_id: str | None = None
        if self._structural_stack:
            context = self._structural_stack[-1]
            value = context["inline_section_id"] or context["section_id"]
            if value is not None:
                section_id = str(value)
        elif self._standalone_section_id is not None:
            section_id = self._standalone_section_id
        if section_id is not None:
            self._section_by_id[section_id].break_block(normalize_text)

    def build(self) -> ParsedDocument:
        self._break_current_section()
        while self._table_stack:
            finished = self._table_stack.pop().finish()
            if finished is not None:
                self.tables.append(finished)
                section = self._section_by_id[finished.section_id]
                section.table_ids.append(finished.table_id)
                section.content_order.append(
                    {"kind": "table", "table_id": finished.table_id}
                )
                self.warnings.append(f"Unclosed table recovered: {finished.table_id}")

        if not self.sections:
            self._ensure_current_section()

        for section in self.sections:
            if section.parent_id and section.parent_id in self._section_by_id:
                parent_path = self._section_by_id[section.parent_id].path
                section.path = [*parent_path, section.title]
            else:
                section.path = [section.title]

        self.tables.sort(key=lambda table: table.table_id)
        return ParsedDocument(
            document_title=self.document_title,
            sections=self.sections,
            tables=self.tables,
            parser_warnings=self.warnings,
        )


def decode_dart_file(path: Path) -> str:
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError:
            continue
    return payload.decode("utf-8", errors="replace")


def parse_dart_text(text: str, fallback_title: str = "공시 문서") -> ParsedDocument:
    parser = DartDocumentParser(fallback_title=fallback_title)
    parser.feed(text)
    parser.close()
    return parser.build()


def parse_dart_document(
    path: Path, fallback_title: str = "공시 문서"
) -> ParsedDocument:
    return parse_dart_text(decode_dart_file(path), fallback_title=fallback_title)
