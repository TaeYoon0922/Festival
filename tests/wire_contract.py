"""Read the string wire contract back into the structures the tests assert on.

``GET /answer`` returns every field as a string, because the organiser's
response contract says it does. ``app.api.serialization`` renders the served
ranking and the execution summary into that text; this reads them back, so the
HTTP contract tests keep asserting on meaning rather than on layout.

It doubles as a round-trip check of the renderer: a field the renderer stopped
emitting, or emitted in a shape this cannot read, fails the test that depends
on it rather than passing silently.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.api.serialization import CHUNK_SEPARATOR, SECTION_JOINER


#: The only two top-level trace values the renderer writes as a joined list.
_STAGES = "stages"
_WARNINGS = "warnings"

_HEADING = re.compile(r"^\[(\d+)\]\s*(.*)$")
_RECEIPT = re.compile(r"^접수일\s+(.*)$")


def _value(text: str) -> Any:
    if text == "null":
        return None
    if text in ("true", "false"):
        return text == "true"
    if text[:1] in ("{", "["):
        return json.loads(text)
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def parse_think_trace(text: str) -> dict[str, Any]:
    """Invert ``render_think_trace``."""

    trace: dict[str, Any] = {}
    for line in str(text).split("\n"):
        if not line or ": " not in line and not line.endswith(":"):
            continue
        key, _, raw = line.partition(":")
        key, raw = key.strip(), raw.strip()
        if key == _STAGES:
            trace[key] = raw.split(SECTION_JOINER) if raw else []
        elif key == _WARNINGS:
            trace[key] = raw.split("; ") if raw else []
        else:
            trace[key] = _value(raw)
    return trace


def _parse_facts(line: str, row: dict[str, Any]) -> None:
    for pair in line.split(" | "):
        key, _, raw = pair.partition(": ")
        if key in ("chunk_type", "corp_code"):
            row[key] = raw
        elif key == "bm25_score":
            row[key] = float(raw)


def parse_retrieved_context(text: str) -> list[dict[str, Any]]:
    """Invert ``render_retrieved_context``.

    Only the fields the renderer writes come back, which is the point: a field
    that was never rendered is absent here too.
    """

    text = str(text)
    if not text or text == "검색된 공시 근거가 없습니다.":
        return []

    rows: list[dict[str, Any]] = []
    for block in text.split(f"\n{CHUNK_SEPARATOR}\n"):
        lines = block.split("\n")
        head = _HEADING.match(lines[0]) if lines else None
        if head is None:
            continue
        row: dict[str, Any] = {"rank": int(head.group(1))}
        labels = [part.strip() for part in head.group(2).split(" · ")]
        for label in labels:
            receipt = _RECEIPT.match(label)
            if receipt:
                row["rcept_dt"] = receipt.group(1)
            elif "corp_name" not in row:
                row["corp_name"] = label
            else:
                row["report_nm"] = label
        content: list[str] | None = None
        for line in lines[1:]:
            if content is not None and not line.startswith(
                ("source_refs: ", "provenance: ")
            ):
                content.append(line)
                continue
            if line.startswith("doc_id: "):
                row["doc_id"] = line[len("doc_id: "):]
            elif line.startswith("chunk_id: "):
                row["chunk_id"] = line[len("chunk_id: "):]
            elif line.startswith("섹션: "):
                row["section_path"] = line[len("섹션: "):].split(SECTION_JOINER)
            elif line.startswith("source_refs: "):
                content = None
                row["source_refs"] = json.loads(line[len("source_refs: "):])
            elif line.startswith("provenance: "):
                content = None
                row["provenance"] = json.loads(line[len("provenance: "):])
            elif line == "내용:":
                content = []
            elif " | " in line and "bm25_score: " in line:
                _parse_facts(line, row)
        if content:
            row["content"] = "\n".join(content)
        rows.append(row)
    return rows


def structured(response: Any) -> dict[str, Any]:
    """The response payload with both rendered fields read back.

    An error body and the health payload carry neither field, so they pass
    through untouched: this is the answer contract's reader, not a general one.
    """

    payload = dict(response.json())
    if isinstance(payload.get("think_trace"), str):
        payload["think_trace"] = parse_think_trace(payload["think_trace"])
    if isinstance(payload.get("retrieved_context"), str):
        payload["retrieved_context"] = parse_retrieved_context(
            payload["retrieved_context"]
        )
    return payload

def without_filing_labels(text: str) -> str:
    """Drop the ``공시:`` lines ``annotate_citations`` adds to a citation block.

    A test comparing a served answer against the deterministic text it was
    built from is asserting that the fallback served that text, not that the
    presentation layer stayed away from it. Removing the labels keeps that
    assertion about the thing it is testing.
    """

    return "\n".join(
        line
        for line in str(text).split("\n")
        if not line.strip().startswith("공시: ")
    )

