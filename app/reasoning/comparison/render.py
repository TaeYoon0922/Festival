"""Combine the legs into one answer without inferring anything new.

Level 1 presents each company's own evidence under its own heading, with every
citation renumbered into a single sequence.  It states no ranking: two filings
can report the same subject over different periods, and declaring a winner from
that would be an inference the evidence does not carry.  A company whose leg
found nothing is named as unconfirmed rather than omitted, so the answer never
reads as though only the reported companies were asked about.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from app.generation.answer_generator import (
    GeneratedAnswer,
    GeneratedCitation,
    GeneratedSection,
    _render_answer_text,
)


_MARKER = re.compile(r"\[(\d+)\]")

#: Said once, at the end, so a reader is never left to assume the layer ranked
#: the companies and chose not to say so.
_NO_RANKING_NOTE = (
    "기업별로 확인된 근거를 그대로 제시했습니다. "
    "보고 기간과 기준이 서로 달라 우열은 판단하지 않았습니다."
)


def _renumber(text: str, mapping: Mapping[str, str]) -> str:
    return _MARKER.sub(lambda match: mapping.get(match.group(0), match.group(0)), text)


def _leg_sections(
    leg: Any, mapping: Mapping[str, str]
) -> tuple[GeneratedSection, ...]:
    """One company's sections, each labelled with the company it came from.

    The label goes on every section rather than into a heading of its own: the
    renderer drops a section whose body is empty, so a standalone heading would
    silently disappear and leave the facts unattributed.
    """

    sections: list[GeneratedSection] = []
    for section in getattr(leg.generated, "sections", ()) or ():
        sections.append(
            GeneratedSection(
                title=f"[{leg.company}] {section.title}",
                content=_renumber(section.content, mapping),
                citations=tuple(
                    mapping.get(value, value) for value in section.citations
                ),
                metadata=tuple(
                    _renumber(value, mapping) for value in section.metadata
                ),
            )
        )
    return tuple(sections)


def render_comparison(
    question: str, execution: Any
) -> GeneratedAnswer | None:
    """Render the collected legs, or ``None`` when there is nothing to render."""

    if execution is None or not getattr(execution, "applied", False):
        return None
    answerable = execution.answerable_legs
    if not answerable:
        return None

    sections: list[GeneratedSection] = []
    citations: list[GeneratedCitation] = []
    warnings: list[str] = []

    for leg in answerable:
        offset = len(citations)
        mapping: dict[str, str] = {}
        for index, citation in enumerate(leg.generated.citations, start=1):
            new_id = f"[{offset + index}]"
            mapping[citation.citation_id] = new_id
            citations.append(
                GeneratedCitation(
                    citation_id=new_id,
                    chunk_id=citation.chunk_id,
                    doc_id=citation.doc_id,
                    source_refs=citation.source_refs,
                    section=citation.section,
                    evidence_type=citation.evidence_type,
                )
            )
        sections.extend(_leg_sections(leg, mapping))
        warnings.extend(getattr(leg.generated, "warnings", ()) or ())

    unconfirmed = [
        leg.company for leg in execution.legs if not leg.answerable
    ]
    if unconfirmed:
        sections.append(
            GeneratedSection(
                title="확인되지 않은 기업",
                content=(
                    ", ".join(unconfirmed)
                    + "에 대해서는 요청하신 내용을 뒷받침할 공시 근거를 확인하지 못했습니다."
                ),
                citations=(),
            )
        )
    sections.append(
        GeneratedSection(title="주의", content=_NO_RANKING_NOTE, citations=())
    )

    return GeneratedAnswer(
        question=question,
        answer_text=_render_answer_text(sections, citations),
        citations=tuple(citations),
        sections=tuple(sections),
        warnings=tuple(dict.fromkeys(warnings)),
        confidence={
            "level": "중간",
            "display_text": "답변 신뢰도: 중간",
            "answerable": True,
        },
        answerable=True,
    )
