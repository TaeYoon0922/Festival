"""Render the answer payload as the five string fields the evaluator expects.

The organiser's response contract states that every value in the response is a
string -- "모든 필드의 값은 문자열(string)타입입니다" -- and leaves the delimiters
inside ``retrieved_context`` to the participant ("문자열 안에서 구분 태그 등으로
자유롭게 연결"). The pipeline builds ``retrieved_context`` as a list and
``think_trace`` as a mapping because that is what the rest of this project
reasons about, and every internal consumer -- the Gold60 evaluator, the
diagnostics, the tests -- keeps reading those structures unchanged.

This module is the single place that turns them into text, and it runs at the
API boundary only. Nothing upstream has to know the wire format, and nothing
downstream of it can reintroduce a non-string value.

Readability is the point of the format. A grader reads these two fields, so a
served chunk is rendered as a labelled block rather than a JSON blob, and the
execution summary as one ``key: value`` line per component. The only field not
reproduced is ``retrieval_text``, whose distinct content -- 기업명, 공시명 and the
section path -- is already the block heading; the chunk's own text is carried
verbatim in ``내용``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


#: Between two served chunks. Long enough not to occur inside filing text.
CHUNK_SEPARATOR = "-" * 60

#: A served chunk can be a whole 손익계산서, and ten of them ran to 47,933
#: characters on one comparison question. The organiser's notice says a field
#: past what the evaluation system takes in one pass has its excess dropped, and
#: an excess dropped at an arbitrary point takes the last chunks with it. Each
#: chunk is bounded instead, so all ten still appear and the cut is visible.
MAX_CHUNK_CONTENT_CHARS = 1500

#: What one line of a filing's table costs, roughly, so the cut lands on a row
#: boundary rather than mid-figure where a reader could misread it.
TRUNCATION_NOTICE = "…(이하 생략)"

#: Section path joiner, matching how the presented answer writes a path.
SECTION_JOINER = " > "


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        # ``true``/``false`` rather than Python's capitalised spelling: the
        # reader of this field is reading JSON everywhere else in the response.
        return "true" if value else "false"
    return str(value)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def _period_text(period: Any) -> str:
    if not isinstance(period, Mapping) or not period:
        return ""
    year = period.get("base_year")
    month = period.get("base_month")
    if year and month:
        return f"{year}년 {month}월"
    if year:
        return f"{year}년"
    return _compact_json(dict(period))


def _heading(index: int, row: Mapping[str, Any]) -> str:
    """``[1] 삼성전자 · 사업보고서 (2025.12) · 접수일 2026-03-10``."""

    rank = row.get("rank")
    parts = [
        part
        for part in (
            _scalar(row.get("corp_name")),
            _scalar(row.get("report_nm")),
        )
        if part
    ]
    receipt = _scalar(row.get("rcept_dt"))
    if receipt:
        parts.append(f"접수일 {receipt}")
    label = " · ".join(parts) if parts else "공시 정보 없음"
    return f"[{rank if rank is not None else index}] {label}"


def _facts(row: Mapping[str, Any]) -> str:
    """One line of the identifiers and scores that are not the heading."""

    pairs = (
        ("chunk_type", _scalar(row.get("chunk_type"))),
        ("bm25_score", _scalar(row.get("bm25_score"))),
        ("corp_code", _scalar(row.get("corp_code"))),
        ("기준기간", _period_text(row.get("period"))),
    )
    return " | ".join(f"{key}: {value}" for key, value in pairs if value)


def _bounded(content: str) -> str:
    """The chunk's text, cut at a line boundary when it runs long."""

    if len(content) <= MAX_CHUNK_CONTENT_CHARS:
        return content
    head = content[:MAX_CHUNK_CONTENT_CHARS]
    cut = head.rfind("\n")
    if cut > MAX_CHUNK_CONTENT_CHARS // 2:
        head = head[:cut]
    return f"{head.rstrip()}{"\n"}{TRUNCATION_NOTICE}"


def render_retrieved_context(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the served Top-K chunks as one delimited string.

    An empty ranking renders as an explicit sentence rather than an empty
    string, so a grader can tell "nothing was retrieved" apart from "the field
    was not populated".
    """

    if not rows:
        return "검색된 공시 근거가 없습니다."

    blocks: list[str] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            continue
        lines = [_heading(index, row)]
        lines.append(f"doc_id: {_scalar(row.get('doc_id'))}")
        lines.append(f"chunk_id: {_scalar(row.get('chunk_id'))}")
        facts = _facts(row)
        if facts:
            lines.append(facts)
        section = row.get("section_path")
        if isinstance(section, Sequence) and not isinstance(section, (str, bytes)):
            path = SECTION_JOINER.join(_scalar(part) for part in section if part)
            if path:
                lines.append(f"섹션: {path}")
        content = _scalar(row.get("content"))
        if content:
            lines.append("내용:")
            lines.append(_bounded(content))
        for key in ("source_refs", "provenance"):
            value = row.get(key)
            if value:
                lines.append(f"{key}: {_compact_json(value)}")
        blocks.append("\n".join(lines))
    return f"\n{CHUNK_SEPARATOR}\n".join(blocks)


def render_think_trace(trace: Mapping[str, Any]) -> str:
    """Render the execution summary as ``key: value`` lines.

    Scalars are written plainly, a list of strings is joined, and a nested
    component summary is written as compact JSON on its own line. Key order is
    the order the pipeline built them, so the reading order stays the execution
    order.
    """

    if not isinstance(trace, Mapping) or not trace:
        return "실행 요약이 없습니다."

    lines: list[str] = []
    for key, value in trace.items():
        if value is None:
            lines.append(f"{key}: null")
        elif key == "stages" and isinstance(value, Sequence) and not isinstance(
            value, (str, bytes)
        ):
            lines.append(f"{key}: {' > '.join(_scalar(part) for part in value)}")
        elif isinstance(value, Mapping):
            lines.append(f"{key}: {_compact_json(value)}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            items = list(value)
            if all(isinstance(item, (str, int, float, bool)) for item in items):
                lines.append(f"{key}: {'; '.join(_scalar(item) for item in items)}")
            else:
                lines.append(f"{key}: {_compact_json(items)}")
        else:
            lines.append(f"{key}: {_scalar(value)}")
    return "\n".join(lines)
