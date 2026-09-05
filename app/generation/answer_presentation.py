"""Present a finished answer without changing what it says.

Every evidence chunk carries a retrieval prefix -- ``[기업명]``, ``[공시명]``,
``[Section Path]`` and the table tags -- built in ``chunking._retrieval_prefix``
so the embedder sees a chunk in its filing context. That prefix is search
scaffolding, and it was reaching the reader verbatim: an answer opened with five
bracketed tag lines before the first fact.

This rewrites those tag lines and nothing else. Body text, markdown tables and
citation markers pass through byte-identically, so the reformatted answer cites
exactly what the original cited. It runs at the very end of the pipeline, after
the answerability guard has already decided what the answer is, which is why it
can be a pure function on a string and why its failure mode is the answer that
was already going to be served.

Nothing here decides content. If the tag shape is absent the input is returned
unchanged, so an answer this does not recognise is an answer this cannot damage.
"""

from __future__ import annotations

import re


#: Tags emitted by ``chunking._retrieval_prefix``, in the order it writes them.
#: ``[기업명]`` and ``[공시명]`` identify the filing and are joined onto one
#: heading; the rest are context the reader benefits from seeing unbracketed.
_CORP = "[기업명]"
_REPORT = "[공시명]"
_SECTION = "[Section Path]"
_CONTEXT_TAGS = {
    "[Table]": "표",
    "[재무제표 범위]": "재무제표 범위",
    "[단위]": "단위",
    "[기준기간]": "기준기간",
    "[기간표현]": "기간",
}
_ALL_TAGS = (_CORP, _REPORT, _SECTION, *_CONTEXT_TAGS)

#: An evidence item opens with its ordinal, then the retrieval prefix.  The
#: ordinal is kept so citation-to-item correspondence stays readable.
_ITEM_HEAD = re.compile(rf"^(\s*)(\d+)\.\s*{re.escape(_CORP)}\s*(.*)$")


def _tag_value(line: str, tag: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith(tag):
        return None
    return stripped[len(tag):].strip()


def readable_answer(answer: str) -> str:
    """Rewrite retrieval tag lines into a heading, leaving everything else alone.

    Returns the input unchanged when no evidence item opens with ``[기업명]``,
    which covers every answer shape this does not recognise -- refusals,
    clarifications, holding timelines and the periodic fact sections.
    """

    text = str(answer or "")
    if _CORP not in text:
        return text

    lines = text.split("\n")
    output: list[str] = []
    index = 0
    changed = False
    while index < len(lines):
        head = _ITEM_HEAD.match(lines[index])
        if head is None:
            output.append(lines[index])
            index += 1
            continue

        indent, ordinal, corp_name = head.groups()
        index += 1
        report = None
        section = None
        context: list[str] = []
        # Consume only the contiguous tag block this item opened with.  The
        # first line that is not one of these tags ends the prefix, and the
        # body from there on is untouched.
        while index < len(lines):
            line = lines[index]
            if (value := _tag_value(line, _REPORT)) is not None and report is None:
                report = value
            elif (value := _tag_value(line, _SECTION)) is not None and section is None:
                section = value
            else:
                for tag, label in _CONTEXT_TAGS.items():
                    if (value := _tag_value(line, tag)) is not None:
                        context.append(f"{label} {value}" if value else label)
                        break
                else:
                    break
            index += 1

        heading = f"{indent}{ordinal}. {corp_name}"
        if report:
            heading = f"{heading} · {report}" if corp_name else f"{heading}{report}"
        output.append(heading.rstrip())
        pad = f"{indent}   "
        if section:
            output.append(f"{pad}{section}")
        output.extend(f"{pad}{entry}" for entry in context)
        changed = True

    return "\n".join(output) if changed else text


def citation_markers(text: str) -> list[str]:
    """The ``[n]`` markers in order, for asserting the rewrite kept every one."""

    return re.findall(r"\[\d+\]", str(text or ""))
