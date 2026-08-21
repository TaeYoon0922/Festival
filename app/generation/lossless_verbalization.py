"""Citation-detached lossless verbalization of a single verified event.

Live experiments settled two things.  A model handed one verified event will
restate it faithfully — fourteen live calls, fourteen clean results.  A model
handed two or more events will not: it reorders fields across events, drops the
company name, and, given a single ratio, invents a prior value and a formula to
derive it.  So this module exists to serve exactly the case that works, and the
caller refuses the rest.

Two ideas do the work.  Citations never reach the model: they are stripped
before the call and reattached afterwards from the event that owns them, so a
fabricated ``[2]`` is impossible rather than merely discouraged.  And every
verified value is replaced by a digit-free placeholder, so a number can survive
the round trip or fail loudly, but never come back quietly altered.

Nothing here repairs a bad response.  Every check is fail-closed and the caller
serves the deterministic answer instead.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.generation.answer_validator import (
    CITATION_MARKER_PATTERN,
    FORBIDDEN_INVESTMENT_TERMS,
    extract_numeric_tokens,
    validate_verbalized_answer,
)
from app.generation.compact_claim import ClaimField, CompactClaim, _render
from app.generation.protected_literals import (
    PLACEHOLDER_PATTERN,
    ProtectedLiteral,
    ProtectedText,
    check_placeholder_integrity,
    protect_literals,
    restore_literals,
)


#: Phrases that mark the model concluding rather than restating.
INFERENCE_MARKERS = (
    "이를 통해",
    "알 수 있습니다",
    "따라서",
    "결론적으로",
    "요약하면",
    "종합하면",
    "판단됩니다",
    "보입니다",
)

#: The model is given a transcription task, not an analyst's.  An earlier,
#: gentler prompt let a one-field claim be read as a question to answer: the
#: model supplied a prior value, a change ratio, a formula, and a placeholder
#: nobody gave it.  Removing the role, rather than adding more rules, is what
#: made single-event verbalization reliable.  Examples are schema-only.
LOSSLESS_VERBALIZER_SYSTEM_PROMPT = """당신은 이미 검증된 사실을 자연스러운 한국어 문장으로 옮기는 변환기입니다.

당신의 역할:
- 당신은 사용자의 질문에 독자적으로 답하지 않는다.
- 당신은 공시를 분석하지 않는다.
- 당신은 빠진 정보를 채우지 않는다.
- 당신은 VERIFIED CLAIM에 이미 있는 사실만 자연스러운 한국어로 바꾼다.

절대 규칙:
1. VERIFIED CLAIM에 명시된 사실만 사용한다.
2. 새로운 사실 항목을 추가하지 않는다.
3. 다음을 추론하거나 복원하지 않는다: 변동 전 값, 변동 후 값, 증감 수량,
   증감 비율, 기준일, 거래 행위, 사유, 계산, 비교.
4. 어떤 계산도 하지 않는다.
5. 수식, 예시, 가정된 숫자, 설명용 숫자, 번호 매긴 개요를 쓰지 않는다.
6. 새로운 placeholder를 만들지 않는다. 입력에 없는 placeholder는
   어떤 이름으로도 출력하지 않는다.
7. 입력의 모든 placeholder는 출력에 정확히 한 번, 원형 그대로,
   입력과 같은 순서로 나타나야 한다.
8. placeholder의 의미 유형을 다시 해석하지 않는다. 비율에 쓰인 NUMBER
   placeholder도 그대로 NUMBER placeholder로 유지한다. 이름을 바꾸지 않는다.
9. 거래 의미를 유추하지 않는다. 매수, 매도, 매입, 처분, 취득, 사들이다 등의
   표현은 그 의미가 검증된 값으로 직접 주어진 경우가 아니면 쓰지 않는다.
10. "변동 현황은 다음과 같습니다" 같은 머리말을 쓰지 않는다. 그런 표현은
    주어지지 않은 항목을 채우도록 유도한다. 주어진 사실만 담은 가장 짧은
    문장을 쓴다.
11. 결론 문장이나 설명 문장을 덧붙이지 않는다.
12. citation을 쓰지 않는다. citation은 이 단계 밖에서 결정적으로 붙는다.

형식 예시(값이 아니라 형태만 참고):

VERIFIED CLAIM:
보유 비율: __FESTIVAL_NUMBER_A__%

GOOD:
보유 비율은 __FESTIVAL_NUMBER_A__%입니다.

BAD:
이전 보유 비율은 5%였으며 현재는 __FESTIVAL_NUMBER_A__%입니다.

BAD:
변동률은 (__FESTIVAL_NUMBER_A__ - 5) / 5 * 100 입니다.

BAD:
보유 비율은 __FESTIVAL_PERCENTAGE_A__입니다.

출력은 변환된 본문만 쓴다. 머리말, 설명, 목록, 결론을 붙이지 않는다."""


#: Rejection reasons.  Each names one fail-closed check.
GENERATED_CITATION = "generated_citation"
UNPROTECTED_NUMERIC = "unprotected_numeric_generation"
STRUCTURED_TEXT_LEAKAGE = "unprotected_structured_text_leakage"
FORBIDDEN_LANGUAGE = "forbidden_investment_language"
INFERENCE_MARKER = "inference_marker_added"
CITATION_ATTACHMENT_FAILED = "citation_attachment_failed"


@dataclass(frozen=True)
class EventCitationAttachment:
    """One event's protected fields and the citations kept outside the model."""

    field_placeholders: tuple[str, ...]
    trailing_suffix: str
    markers: tuple[str, ...]


@dataclass(frozen=True)
class CitationAttachmentResult:
    """Outcome of reattaching citations by placeholder ownership."""

    final_answer: str | None
    valid: bool
    attached_citation_count: int
    reason: str | None


@dataclass(frozen=True)
class DetachedClaimInput:
    """Citation-free model input plus the citation metadata held back."""

    text: str
    protection: ProtectedText
    attachments: tuple[EventCitationAttachment, ...]

    @property
    def event_count(self) -> int:
        return len(self.attachments)

    @property
    def expected_citation_sequence(self) -> tuple[str, ...]:
        return tuple(
            marker for attachment in self.attachments for marker in attachment.markers
        )


@dataclass(frozen=True)
class LosslessResult:
    """Whether a model reply may be served, and why not when it may not."""

    valid: bool
    final_answer: str | None = None
    reason: str | None = None
    attached_citation_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "attached_citation_count": self.attached_citation_count,
        }


def group_event_fields(
    fields: Sequence[ClaimField],
) -> tuple[tuple[ClaimField, ...], ...]:
    """Split a claim's repeated field schema back into ordered events."""

    groups: list[tuple[ClaimField, ...]] = []
    current: list[ClaimField] = []
    names: set[str] = set()
    for field in fields:
        if current and field.name in names:
            groups.append(tuple(current))
            current = []
            names = set()
        current.append(field)
        names.add(field.name)
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def claim_event_count(claim: CompactClaim) -> int:
    return len(group_event_fields(claim.fields))


def detach_claim_citations(claim: CompactClaim) -> DetachedClaimInput:
    """Protect every structured field value and hold the citations back.

    Numeric and date classification is delegated to the production literal
    protector.  A field with no numeric or date literal is protected as one
    opaque TEXT value.  Values come from ``ClaimField``; rendered prose is
    never parsed to recover them.
    """

    event_fields = group_event_fields(claim.fields)
    citation_order = {
        citation.marker: index for index, citation in enumerate(claim.citations)
    }
    kind_offsets: Counter[str] = Counter()
    literals: list[ProtectedLiteral] = []
    masked_fields: list[ClaimField] = []
    field_suffixes: list[str] = []

    for field in claim.fields:
        value = field.value
        if not value:
            raise ValueError("claim field value must be non-empty")
        value_protection = protect_literals(value)
        value_literals = tuple(
            literal
            for literal in value_protection.literals
            if literal.kind in {"date", "number"}
        )
        if len(value_literals) > 1:
            raise ValueError("each claim field must resolve to one protected literal")

        if value_literals:
            source_literal = value_literals[0]
            kind = source_literal.kind
            placeholder = _field_placeholder(kind, kind_offsets[kind])
            kind_offsets[kind] += 1
            masked_value = value_protection.masked.replace(
                source_literal.placeholder, placeholder, 1
            )
            literal_text = source_literal.text
            literal_offset = value.rfind(literal_text)
            if literal_offset < 0:
                raise ValueError("field literal is not present in its structured value")
            suffix = value[literal_offset + len(literal_text) :]
        else:
            kind = "text"
            placeholder = _field_placeholder(kind, kind_offsets[kind])
            kind_offsets[kind] += 1
            masked_value = placeholder
            literal_text = value
            suffix = ""

        literals.append(
            ProtectedLiteral(placeholder=placeholder, text=literal_text, kind=kind)
        )
        masked_fields.append(
            ClaimField(
                name=field.name,
                label=field.label,
                value=masked_value,
                marker="",
                chunk_id=field.chunk_id,
            )
        )
        field_suffixes.append(suffix)

    detached_fields = tuple(
        ClaimField(
            name=field.name,
            label=field.label,
            value=field.value,
            marker="",
            chunk_id=field.chunk_id,
        )
        for field in claim.fields
    )
    text = _render(claim.company, claim.reporter, detached_fields)
    masked = _render(claim.company, claim.reporter, masked_fields)
    if CITATION_MARKER_PATTERN.search(text) or CITATION_MARKER_PATTERN.search(masked):
        raise ValueError("detached model input still contains a citation marker")

    protection = ProtectedText(
        original=text, masked=masked, literals=tuple(literals)
    )
    attachments: list[EventCitationAttachment] = []
    field_offset = 0
    for fields in event_fields:
        event_start = field_offset
        field_offset += len(fields)
        event_markers = {field.marker for field in fields}
        unknown = event_markers.difference(citation_order)
        if unknown:
            raise ValueError("claim field references unknown citation metadata")
        markers = tuple(sorted(event_markers, key=citation_order.__getitem__))
        attachments.append(
            EventCitationAttachment(
                field_placeholders=tuple(
                    literal.placeholder
                    for literal in literals[event_start:field_offset]
                ),
                trailing_suffix=field_suffixes[field_offset - 1],
                markers=markers,
            )
        )

    return DetachedClaimInput(
        text=text, protection=protection, attachments=tuple(attachments)
    )


def attach_detached_citations(
    masked_candidate: str, detached: DetachedClaimInput
) -> CitationAttachmentResult:
    """Reattach each event's citations after its own last protected field."""

    integrity = check_placeholder_integrity(masked_candidate, detached.protection)
    if not integrity.valid:
        return CitationAttachmentResult(None, False, 0, integrity.reason)

    owned_placeholders = tuple(
        placeholder
        for attachment in detached.attachments
        for placeholder in attachment.field_placeholders
    )
    if owned_placeholders != detached.protection.placeholders:
        return CitationAttachmentResult(None, False, 0, "event_count_mismatch")

    insertions: list[tuple[int, str]] = []
    for index, attachment in enumerate(detached.attachments):
        if not attachment.field_placeholders:
            return CitationAttachmentResult(None, False, 0, "event_span_not_found")
        if not attachment.markers:
            return CitationAttachmentResult(None, False, 0, "citation_mapping_missing")

        last_placeholder = attachment.field_placeholders[-1]
        placeholder_start = masked_candidate.find(last_placeholder)
        if placeholder_start < 0:
            return CitationAttachmentResult(None, False, 0, "event_span_not_found")
        insertion_offset = placeholder_start + len(last_placeholder)
        if attachment.trailing_suffix:
            suffix_match = re.match(
                rf"[ \t]*{re.escape(attachment.trailing_suffix)}",
                masked_candidate[insertion_offset:],
            )
            if suffix_match is None:
                return CitationAttachmentResult(
                    None, False, 0, "event_field_text_mismatch"
                )
            insertion_offset += suffix_match.end()

        if index + 1 < len(detached.attachments):
            next_placeholder = detached.attachments[index + 1].field_placeholders[0]
            next_event_offset = masked_candidate.find(next_placeholder)
            if next_event_offset < 0 or insertion_offset >= next_event_offset:
                return CitationAttachmentResult(None, False, 0, "event_span_not_found")
        insertions.append((insertion_offset, " " + "".join(attachment.markers)))

    candidate = masked_candidate
    for insertion_offset, citation_text in reversed(insertions):
        candidate = (
            candidate[:insertion_offset] + citation_text + candidate[insertion_offset:]
        )

    final_answer = restore_literals(candidate, detached.protection)
    found_citations = tuple(
        match.group(0) for match in CITATION_MARKER_PATTERN.finditer(final_answer)
    )
    if found_citations != detached.expected_citation_sequence:
        return CitationAttachmentResult(
            final_answer, False, len(found_citations), "citation_sequence_mismatch"
        )
    return CitationAttachmentResult(final_answer, True, len(found_citations), None)


def expected_attached_answer(detached: DetachedClaimInput) -> str:
    """The answer a perfectly faithful reply would produce, used as reference."""

    attachment = attach_detached_citations(detached.protection.masked, detached)
    if not attachment.valid or attachment.final_answer is None:
        raise ValueError("deterministic citation plan cannot attach to its source")
    return attachment.final_answer


def verify_lossless_candidate(
    raw: str, *, claim: CompactClaim, detached: DetachedClaimInput
) -> LosslessResult:
    """Run every fail-closed check, in the order the live experiment used.

    The checks run against the masked reply first, because that is where a
    fabricated number or citation is unambiguous — a value that survives here
    can only have come from a placeholder.
    """

    protection = detached.protection

    integrity = check_placeholder_integrity(raw, protection)
    if not integrity.valid:
        return LosslessResult(False, reason=integrity.reason)

    if CITATION_MARKER_PATTERN.search(raw) is not None:
        return LosslessResult(False, reason=GENERATED_CITATION)

    if unprotected_numeric_tokens(raw, protection):
        return LosslessResult(False, reason=UNPROTECTED_NUMERIC)

    if unprotected_text_literals(raw, protection):
        return LosslessResult(False, reason=STRUCTURED_TEXT_LEAKAGE)

    introduced = tuple(
        term
        for term in FORBIDDEN_INVESTMENT_TERMS
        if term in raw and term not in protection.masked
    )
    if introduced:
        return LosslessResult(False, reason=FORBIDDEN_LANGUAGE)

    restored = restore_literals(raw, protection)
    if any(marker in restored for marker in INFERENCE_MARKERS):
        return LosslessResult(False, reason=INFERENCE_MARKER)

    validation = validate_verbalized_answer(
        restored, reference=detached.text, required_terms=claim.required_terms
    )
    if not validation.valid:
        return LosslessResult(False, reason=validation.reason)

    attachment = attach_detached_citations(raw, detached)
    if not attachment.valid or attachment.final_answer is None:
        return LosslessResult(
            False, reason=attachment.reason or CITATION_ATTACHMENT_FAILED
        )

    final_validation = validate_verbalized_answer(
        attachment.final_answer,
        reference=expected_attached_answer(detached),
        required_terms=claim.required_terms,
    )
    if not final_validation.valid:
        return LosslessResult(False, reason=final_validation.reason)

    return LosslessResult(
        True,
        final_answer=attachment.final_answer.strip(),
        attached_citation_count=attachment.attached_citation_count,
    )


def unprotected_numeric_tokens(raw: str, protection: ProtectedText) -> list[str]:
    """Numbers the model wrote itself, outside any placeholder."""

    remainder = raw
    for placeholder in protection.placeholders:
        remainder = remainder.replace(placeholder, " ")
    return sorted(extract_numeric_tokens(remainder).elements())


def unprotected_text_literals(raw: str, protection: ProtectedText) -> list[str]:
    """Structured TEXT values the model reproduced outside their placeholder."""

    return sorted(
        {
            literal.text
            for literal in protection.literals
            if literal.kind == "text" and literal.text and literal.text in raw
        }
    )


def found_placeholders(raw: str) -> list[str]:
    return PLACEHOLDER_PATTERN.findall(raw)


def _field_placeholder(kind: str, index: int) -> str:
    return f"__FESTIVAL_{kind.upper()}_{_alpha_label(index)}__"


def _alpha_label(index: int) -> str:
    label = ""
    position = index + 1
    while position > 0:
        position, remainder = divmod(position - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label
