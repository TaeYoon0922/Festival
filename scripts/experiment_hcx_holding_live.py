"""Run citation-detached HCX validation on actual Gold60 holding outputs.

This is a read-only experiment.  It runs the production query-understanding,
retrieval, orchestration, resolver, and composer path, then builds the same
verified holding claim with only the production size caps omitted.  No event is
selected, removed, reordered, or recomputed.

    python scripts/experiment_hcx_holding_live.py
    python scripts/experiment_hcx_holding_live.py --repeat 2

The script prints JSON only.  Raw HCX content is omitted unless
``--show-output`` is supplied, and credentials and headers are never emitted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.pipeline import AnswerPipeline
from app.generation.answer_validator import (
    CITATION_MARKER_PATTERN,
    FORBIDDEN_INVESTMENT_TERMS,
    extract_numeric_tokens,
    validate_verbalized_answer,
)
from app.generation.answer_generator import _text
from app.generation.compact_claim import (
    MAX_CLAIM_EVENTS,
    MAX_CLAIM_LITERALS,
    SUPPORTED_TASK_TYPE,
    ClaimField,
    CompactClaim,
    _citation_index,
    _ClaimBuilder,
    _field_value,
    _matching_events,
    _render,
    _resolution_mapping,
)
from app.generation.hcx_verbalizer import HcxSettings, _response_content
from app.generation.protected_literals import (
    PLACEHOLDER_PATTERN,
    ProtectedLiteral,
    ProtectedText,
    check_placeholder_integrity,
    protect_literals,
    restore_literals,
)
from app.parsing.final_validation import HOLDING_ADDITIONAL_QUESTIONS
from app.retrieval.embeddings import EmbeddingHttpError, UrllibJsonTransport
from scripts.experiment_hcx_multi_event import (
    EXPERIMENT_SYSTEM_PROMPT,
    INFERENCE_MARKERS,
    CitationAttachmentResult,
    DetachedClaimInput,
    EventCitationAttachment,
    _completion_tokens,
    _finish_reason,
    _group_event_fields,
)


TARGET_QUESTION_IDS = (
    "HX02",
    "HX06",
    "HX07",
    "HX10",
    "HX11",
    "HX14",
    "HX15",
    "HX16",
    "HX18",
    "HX19",
    "HX20",
)

PRIOR_AUDIT_CANDIDATE_EVENT_COUNTS = {
    "HX07": 4,
    "HX11": 4,
    "HX15": 6,
    "HX16": 6,
    "HX19": 5,
    "HX20": 4,
}

_FIELD_PLACEHOLDER_KINDS = ("date", "number", "text")

# Experimental only.  The production prompt in ``hcx_verbalizer.py`` and the
# default live-experiment prompt imported above remain byte-for-byte unchanged.
STRICT_EVENT_ORDER_SYSTEM_PROMPT = EXPERIMENT_SYSTEM_PROMPT + """

추가 불변 규칙:
- 입력의 모든 placeholder는 출력에 정확히 한 번씩 나타나야 한다.
- placeholder는 입력과 정확히 동일한 순서를 유지해야 한다.
- placeholder를 누락하거나 중복하지 않는다.
- 어떤 placeholder도 다른 placeholder의 앞이나 뒤로 이동하지 않는다.
- 입력의 각 이벤트는 나눌 수 없는 하나의 record다. 한 이벤트의 모든 필드를 함께 유지한다.
- 서로 다른 이벤트의 필드를 재그룹화하지 않는다.

금지 예시:
입력이 event1의 before A, after B 다음에 event2의 before C, after D 순서라면,
before A, C 다음에 after B, D로 재배열하는 것은 금지한다.
출력도 반드시 A → B → C → D 순서를 유지한다.

- 이벤트를 병합, 요약, 정렬, 중복 제거, 선택, 비교, 집계 또는 재구성하지 않는다.
- 서론, 설명, 요약 또는 결론에서 factual placeholder를 다시 반복하지 않는다.
- citation을 생성하지 않는다.
- 자연스러운 한국어 연결 표현만 바꿀 수 있다. placeholder의 identity, 개수와 순서는 불변이다.
- 이벤트가 많아도 모든 이벤트를 그대로 보존한다. 답변을 줄이기 위해 사실을 생략하지 않는다.
"""


class ExperimentPreparationError(ValueError):
    """A safe all-candidate claim could not be built."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class PreparedHoldingQuestion:
    question_id: str
    question: str
    task_type: str
    requested_fields: tuple[str, ...]
    candidate_event_count: int
    claim: CompactClaim
    detached: DetachedClaimInput
    structure: Mapping[str, Any]
    prepare_diagnostics: Mapping[str, Any]


def detach_live_claim_citations(claim: CompactClaim) -> DetachedClaimInput:
    """Protect every structured requested field, including plain text values.

    Numeric/date classification is delegated to the production literal
    protector.  A field with no numeric/date literal is protected as one
    opaque TEXT value.  The value comes directly from ``ClaimField``; rendered
    prose is never parsed to recover it.
    """

    event_fields = _group_event_fields(claim.fields)
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
            raise ValueError("experimental claim field value must be non-empty")
        value_protection = protect_literals(value)
        value_literals = tuple(
            literal
            for literal in value_protection.literals
            if literal.kind in {"date", "number"}
        )
        if len(value_literals) > 1:
            raise ValueError(
                "each experimental claim field must resolve to one protected literal"
            )

        if value_literals:
            source_literal = value_literals[0]
            kind = source_literal.kind
            placeholder = _field_placeholder(kind, kind_offsets[kind])
            kind_offsets[kind] += 1
            masked_value = value_protection.masked.replace(
                source_literal.placeholder,
                placeholder,
                1,
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
            ProtectedLiteral(
                placeholder=placeholder,
                text=literal_text,
                kind=kind,
            )
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
        raise ValueError("detached HCX input still contains a citation marker")

    protection = ProtectedText(
        original=text,
        masked=masked,
        literals=tuple(literals),
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
        text=text,
        protection=protection,
        attachments=tuple(attachments),
    )


def _field_placeholder(kind: str, index: int) -> str:
    return f"__FESTIVAL_{kind.upper()}_{_alpha_label(index)}__"


def _alpha_label(index: int) -> str:
    label = ""
    position = index + 1
    while position > 0:
        position, remainder = divmod(position - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def placeholder_diagnostics(
    protection: ProtectedText,
    found: Sequence[str],
) -> dict[str, Any]:
    """Report exact expected/found/missing identities for all field kinds."""

    kind_by_placeholder = {
        literal.placeholder: literal.kind for literal in protection.literals
    }
    expected_by_type = {
        kind: [
            literal.placeholder
            for literal in protection.literals
            if literal.kind == kind
        ]
        for kind in _FIELD_PLACEHOLDER_KINDS
    }
    found_by_type = {
        kind: [token for token in found if kind_by_placeholder.get(token) == kind]
        for kind in _FIELD_PLACEHOLDER_KINDS
    }
    missing_by_type = {
        kind: list(
            (
                Counter(expected_by_type[kind])
                - Counter(found_by_type[kind])
            ).elements()
        )
        for kind in _FIELD_PLACEHOLDER_KINDS
    }
    return {
        "expected_placeholders_by_type": expected_by_type,
        "found_placeholders_by_type": found_by_type,
        "missing_placeholders_by_type": missing_by_type,
        "text_placeholders_all_preserved": (
            found_by_type["text"] == expected_by_type["text"]
        ),
    }


def attach_live_detached_citations(
    masked_candidate: str,
    detached: DetachedClaimInput,
) -> CitationAttachmentResult:
    """Attach event citations by mixed DATE/NUMBER/TEXT placeholder ownership."""

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
            candidate[:insertion_offset]
            + citation_text
            + candidate[insertion_offset:]
        )

    final_answer = restore_literals(candidate, detached.protection)
    found_citations = tuple(
        match.group(0) for match in CITATION_MARKER_PATTERN.finditer(final_answer)
    )
    if found_citations != detached.expected_citation_sequence:
        return CitationAttachmentResult(
            final_answer,
            False,
            len(found_citations),
            "citation_sequence_mismatch",
        )
    return CitationAttachmentResult(final_answer, True, len(found_citations), None)


def evaluate_live_detached_candidate(
    record: dict[str, Any],
    raw: str,
    *,
    claim: CompactClaim,
    detached: DetachedClaimInput,
    show_output: bool,
    check_structured_text_leakage: bool = False,
) -> dict[str, Any]:
    """Validate the TEXT-aware live candidate before and after citations."""

    protection = detached.protection
    found = PLACEHOLDER_PATTERN.findall(raw)
    integrity = check_placeholder_integrity(raw, protection)
    unexpected_citation = CITATION_MARKER_PATTERN.search(raw) is not None
    unprotected_numeric_tokens = _unprotected_numeric_tokens(raw, protection)
    unprotected_text_literals = (
        _unprotected_text_literals(raw, protection)
        if check_structured_text_leakage
        else []
    )
    introduced_investment_terms = tuple(
        term
        for term in FORBIDDEN_INVESTMENT_TERMS
        if term in raw and term not in protection.masked
    )
    record.update(
        {
            "output_chars": len(raw),
            "expected_placeholders": len(protection.placeholders),
            "found_placeholders": len(found),
            "field_placeholder_integrity_valid": integrity.valid,
            "field_placeholder_integrity_reason": integrity.reason,
            "placeholder_integrity_valid": integrity.valid,
            "placeholder_integrity_reason": integrity.reason,
            "all_events_kept": integrity.valid,
            "unexpected_citation_generation": unexpected_citation,
            "unprotected_numeric_tokens_in_masked_candidate": (
                unprotected_numeric_tokens
            ),
            "unprotected_text_literals_in_masked_candidate": (
                unprotected_text_literals
            ),
            "forbidden_investment_language_detected": bool(
                introduced_investment_terms
            ),
            "forbidden_investment_terms": list(introduced_investment_terms),
            "pre_validation_reference_kind": "citation_detached_compact_claim",
            "final_validation_reference_kind": (
                "deterministic_event_citation_attachment"
            ),
            "candidate_valid_before_citation_attachment": False,
            "deterministic_citations_attached": False,
            "attached_citation_count": 0,
            "citation_attachment_reason": (
                None if integrity.valid else integrity.reason
            ),
            "final_answer_valid": False,
            "inference_markers": [],
            **placeholder_diagnostics(protection, found),
        }
    )
    if not integrity.valid:
        return record

    if unprotected_text_literals:
        record["validator_reason"] = "unprotected_structured_text_leakage"
        return record

    restored = restore_literals(raw, protection)
    reference_numeric_tokens = extract_numeric_tokens(detached.text)
    candidate_numeric_tokens = extract_numeric_tokens(restored)
    inference_markers = [
        marker for marker in INFERENCE_MARKERS if marker in restored
    ]
    record.update(
        {
            "reference_numeric_tokens": sorted(reference_numeric_tokens.elements()),
            "candidate_numeric_tokens": sorted(candidate_numeric_tokens.elements()),
            "numbers_only_in_reference": sorted(
                (reference_numeric_tokens - candidate_numeric_tokens).elements()
            ),
            "numbers_only_in_candidate": sorted(
                (candidate_numeric_tokens - reference_numeric_tokens).elements()
            ),
            "reference_citations": _citation_markers(detached.text),
            "candidate_citations": _citation_markers(restored),
            "inference_markers": inference_markers,
        }
    )
    if show_output:
        record["restored_output"] = restored
        record["validation_reference_text"] = detached.text
        record["restored_candidate_text"] = restored
    if unexpected_citation:
        record["validator_reason"] = "unexpected_citation_generation"
        return record

    validation = validate_verbalized_answer(
        restored,
        reference=detached.text,
        required_terms=claim.required_terms,
    )
    candidate_valid = validation.valid and not inference_markers
    record.update(
        {
            "restored_chars": len(restored),
            "length_ratio": round(len(restored) / len(detached.text), 2),
            "candidate_valid_before_citation_attachment": candidate_valid,
            "validator_reason": (
                "inference_marker_added" if inference_markers else validation.reason
            ),
        }
    )
    if not candidate_valid:
        return record

    attachment = attach_live_detached_citations(raw, detached)
    final_answer = attachment.final_answer
    record["deterministic_citations_attached"] = attachment.valid
    record["attached_citation_count"] = attachment.attached_citation_count
    record["citation_attachment_reason"] = attachment.reason
    if not attachment.valid or final_answer is None:
        record["final_validator_reason"] = (
            attachment.reason or "citation_attachment_failed"
        )
        return record

    final_validation = validate_verbalized_answer(
        final_answer,
        reference=_expected_attached_answer(detached),
        required_terms=claim.required_terms,
    )
    record["final_answer_valid"] = final_validation.valid
    record["final_validator_reason"] = final_validation.reason
    if show_output:
        record["final_output"] = final_answer
    return record


def _unprotected_numeric_tokens(raw: str, protection: ProtectedText) -> list[str]:
    remainder = raw
    for placeholder in protection.placeholders:
        remainder = remainder.replace(placeholder, " ")
    return sorted(extract_numeric_tokens(remainder).elements())


def _unprotected_text_literals(
    raw: str,
    protection: ProtectedText,
) -> list[str]:
    """Find exact structured TEXT values independently regenerated by HCX."""

    return sorted(
        {
            literal.text
            for literal in protection.literals
            if literal.kind == "text" and literal.text and literal.text in raw
        }
    )


def _citation_markers(text: str) -> list[str]:
    return list(
        dict.fromkeys(
            match.group(0) for match in CITATION_MARKER_PATTERN.finditer(text)
        )
    )


def _expected_attached_answer(detached: DetachedClaimInput) -> str:
    attachment = attach_live_detached_citations(
        detached.protection.masked,
        detached,
    )
    if not attachment.valid or attachment.final_answer is None:
        raise ValueError("deterministic citation plan cannot attach to its source")
    return attachment.final_answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the citation-detached HCX experiment on the 11 actual "
            "Holding too_many_events questions. Production remains unchanged."
        )
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="HCX calls per actual question (default: 1).",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Include raw, restored, and final HCX text in the JSON report.",
    )
    parser.add_argument(
        "--diagnose-prepare",
        action="store_true",
        help="Run through claim preparation only; make no HCX calls.",
    )
    parser.add_argument(
        "--question-id",
        action="append",
        choices=TARGET_QUESTION_IDS,
        help="Limit the run to one or more target IDs (repeatable).",
    )
    parser.add_argument(
        "--strict-event-order",
        action="store_true",
        help=(
            "Use the experiment-only exact-once/event-order prompt. "
            "The default and production prompts remain unchanged."
        ),
    )
    parser.add_argument(
        "--event-wise",
        action="store_true",
        help=(
            "Call HCX once per verified event and fall back only the event "
            "that fails. Experimental; production remains unchanged."
        ),
    )
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    settings = HcxSettings.from_env()
    if not args.diagnose_prepare and not settings.configured:
        print(
            "FESTIVAL_HCX_API_KEY is not set; this experiment needs a live key.",
            file=sys.stderr,
        )
        return 1

    pipeline = AnswerPipeline.from_env()
    transport = UrllibJsonTransport()
    rows = target_question_rows(args.question_id)
    runs: list[dict[str, Any]] = []
    event_runs: list[dict[str, Any]] = []
    event_wise_question_runs: list[dict[str, Any]] = []
    prepare_diagnostics: list[dict[str, Any]] = []
    for row in rows:
        question_id = str(row["question_id"])
        print(f"[prepare] {question_id}", file=sys.stderr, flush=True)
        prepared, diagnostic = diagnose_prepare(pipeline, row)
        prepare_diagnostics.append(diagnostic)
        if args.diagnose_prepare:
            continue
        if prepared is None:
            for repeat_index in range(1, args.repeat + 1):
                if args.event_wise:
                    event_wise_question_runs.append(
                        event_wise_preparation_failure(
                            row,
                            repeat_index,
                            diagnostic,
                        )
                    )
                else:
                    runs.append(
                        preparation_failure_record(
                            row,
                            repeat_index,
                            diagnostic,
                            strict_event_order=args.strict_event_order,
                        )
                    )
            continue

        for repeat_index in range(1, args.repeat + 1):
            print(
                f"[hcx] {question_id} run {repeat_index}/{args.repeat}",
                file=sys.stderr,
                flush=True,
            )
            if args.event_wise:
                question_run, question_event_runs = run_event_wise_question(
                    prepared,
                    transport=transport,
                    settings=settings,
                    repeat_index=repeat_index,
                    show_output=args.show_output,
                )
                event_wise_question_runs.append(question_run)
                event_runs.extend(question_event_runs)
            else:
                runs.append(
                    run_prepared_question(
                        prepared,
                        transport=transport,
                        settings=settings,
                        repeat_index=repeat_index,
                        show_output=args.show_output,
                        strict_event_order=args.strict_event_order,
                    )
                )

    report: dict[str, Any] = {
        "mode": (
            "prepare_diagnosis"
            if args.diagnose_prepare
            else "event_wise_hcx" if args.event_wise else "live_hcx"
        ),
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "repeat": args.repeat,
        "strict_event_order": args.strict_event_order or args.event_wise,
        "event_wise": args.event_wise,
        "target_question_ids": list(TARGET_QUESTION_IDS),
        "selected_question_ids": [str(row["question_id"]) for row in rows],
        "production_caps_observed_not_applied": {
            "MAX_CLAIM_EVENTS": MAX_CLAIM_EVENTS,
            "MAX_CLAIM_LITERALS": MAX_CLAIM_LITERALS,
        },
        "prepare_diagnostics": prepare_diagnostics,
    }
    if args.diagnose_prepare:
        report["summary"] = summarize_prepare_diagnostics(prepare_diagnostics)
    elif args.event_wise:
        report["event_runs"] = event_runs
        report["question_runs"] = event_wise_question_runs
        report["summary"] = summarize_event_wise_runs(
            event_wise_question_runs,
            event_runs,
            repeat=args.repeat,
        )
    else:
        report["runs"] = runs
        report["summary"] = summarize_live_runs(runs, repeat=args.repeat)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def target_question_rows(
    question_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Return the requested frozen Holding questions in explicit target order."""

    by_id = {
        str(row["question_id"]): dict(row)
        for row in HOLDING_ADDITIONAL_QUESTIONS
    }
    missing = [question_id for question_id in TARGET_QUESTION_IDS if question_id not in by_id]
    if missing:
        raise RuntimeError("frozen Holding questions missing: " + ", ".join(missing))
    selected = tuple(question_ids or TARGET_QUESTION_IDS)
    return tuple(by_id[question_id] for question_id in selected)


def prepare_actual_question(
    pipeline: Any,
    row: Mapping[str, Any],
) -> PreparedHoldingQuestion:
    """Run the existing read-only stack and prepare its complete event claim."""

    prepared, diagnostic = diagnose_prepare(pipeline, row)
    if prepared is None:
        stage = diagnostic.get("exception_stage") or "claim_builder"
        message = diagnostic.get("exception_message") or "preparation failed"
        raise ExperimentPreparationError(f"{stage}: {message}")
    return prepared


def diagnose_prepare(
    pipeline: Any,
    row: Mapping[str, Any],
) -> tuple[PreparedHoldingQuestion | None, dict[str, Any]]:
    """Trace actual preparation one boundary at a time without calling HCX."""

    question_id = str(row["question_id"])
    question = str(row["query"])
    diagnostic = empty_prepare_diagnostic(question_id, question)

    try:
        plan = pipeline.understanding.understand(
            question,
            top_k=pipeline.settings.top_k,
        )
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        return None, record_prepare_exception(
            diagnostic, error, "query_understanding"
        )
    diagnostic["plan_task_type"] = getattr(plan, "task_type", None)

    try:
        execution = pipeline.executor.execute(plan)
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        return None, record_prepare_exception(diagnostic, error, "retrieval")

    try:
        result = pipeline.orchestrator.run(question, plan, execution)
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        return None, record_prepare_exception(
            diagnostic,
            error,
            orchestrator_exception_stage(error),
        )

    task_type = str(result.task_decision.task_type)
    draft = result.answer_draft
    resolution = result.resolution
    diagnostic.update(
        {
            "routed_task_type": task_type,
            "draft_task_type": getattr(draft, "task_type", None),
            "resolution_type": (
                type(resolution).__name__ if resolution is not None else None
            ),
        }
    )
    if resolution is None:
        return None, record_prepare_exception(
            diagnostic,
            ExperimentPreparationError("missing_resolution"),
            "resolver",
        )

    resolved = _resolution_mapping(resolution)
    requested = tuple(str(field) for field in resolved.get("requested_fields") or ())
    unresolved = tuple(str(field) for field in resolved.get("unresolved_fields") or ())
    rows = draft_event_rows(draft)
    diagnostic.update(
        {
            "requested_fields": list(requested),
            "unresolved_fields": list(unresolved),
            "total_draft_events": len(rows),
            "matching_events_count": sum(
                1 for event in rows if event.get("matches_query") is True
            ),
            "complete_matching_events_count": sum(
                1
                for event in rows
                if event.get("matches_query") is True
                and not event.get("field_conflict")
            ),
            "events_with_requested_fields": sum(
                1
                for event in rows
                if event.get("matches_query") is True
                and not event.get("field_conflict")
                and requested
                and all(_field_value(event, field) is not None for field in requested)
            ),
        }
    )
    if task_type != SUPPORTED_TASK_TYPE:
        return None, record_prepare_exception(
            diagnostic,
            ExperimentPreparationError("unsupported_task_type"),
            "orchestrator",
        )
    if not getattr(draft, "answerable", False):
        return None, record_prepare_exception(
            diagnostic,
            ExperimentPreparationError("not_answerable"),
            "orchestrator",
        )
    if not requested:
        return None, record_prepare_exception(
            diagnostic,
            ExperimentPreparationError("no_requested_fields"),
            "resolver",
        )
    if unresolved:
        return None, record_prepare_exception(
            diagnostic,
            ExperimentPreparationError("unresolved_fields"),
            "resolver",
        )

    try:
        events = _matching_events(draft, requested)
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        return None, record_prepare_exception(
            diagnostic, error, "matching_events"
        )
    diagnostic["candidate_event_count"] = len(events)
    expected = PRIOR_AUDIT_CANDIDATE_EVENT_COUNTS.get(question_id)
    diagnostic["prior_audit_candidate_event_count"] = expected
    diagnostic["matches_prior_audit_candidate_event_count"] = (
        None if expected is None else expected == len(events)
    )
    if not events:
        return None, record_prepare_exception(
            diagnostic,
            ExperimentPreparationError("no_matching_event"),
            "matching_events",
        )

    try:
        citations = _citation_index(draft)
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        return None, record_prepare_exception(
            diagnostic, error, "citation_index"
        )
    diagnostic["citation_index_size"] = len(citations)
    diagnostic["events_with_citation_mapping"] = sum(
        1
        for event_index, _ in events
        if all(
            (event_index, field) in citations
            or (event_index, "event_evidence") in citations
            for field in requested
        )
    )

    try:
        builder = _ClaimBuilder(citations)
        for event_index, event in events:
            if not builder.add_event(event_index, event, requested):
                raise ExperimentPreparationError(
                    f"citation_not_linked:event_index={event_index}"
                )
        claim = claim_from_builder(draft, events, builder)
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        return None, record_prepare_exception(diagnostic, error, "claim_builder")

    diagnostic["claim_field_count"] = len(claim.fields)
    diagnostic["claim_citation_count"] = len(claim.citations)
    try:
        detached = detach_live_claim_citations(claim)
    except Exception as error:  # noqa: BLE001 - diagnostic boundary
        stage = "protect" if traceback_mentions(error, "protected_literals.py") else "detach"
        return None, record_prepare_exception(diagnostic, error, stage)

    structure = real_structure_diagnostics(
        claim,
        detached,
        requested_fields=requested,
        candidate_event_count=len(events),
    )
    diagnostic.update(
        {
            "field_literal_count": len(claim.fields),
            "protected_literal_count": len(detached.protection.literals),
            "masked_chars": len(detached.protection.masked),
            "exception_type": None,
            "exception_message": None,
            "exception_stage": None,
            "prepare_success": True,
        }
    )
    prepared = PreparedHoldingQuestion(
        question_id=question_id,
        question=question,
        task_type=task_type,
        requested_fields=requested,
        candidate_event_count=len(events),
        claim=claim,
        detached=detached,
        structure=structure,
        prepare_diagnostics=diagnostic,
    )
    return prepared, diagnostic


def empty_prepare_diagnostic(question_id: str, question: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "original_question": question,
        "plan_task_type": None,
        "routed_task_type": None,
        "draft_task_type": None,
        "resolution_type": None,
        "requested_fields": [],
        "unresolved_fields": [],
        "total_draft_events": 0,
        "matching_events_count": 0,
        "complete_matching_events_count": 0,
        "citation_index_size": 0,
        "events_with_requested_fields": 0,
        "events_with_citation_mapping": 0,
        "candidate_event_count": 0,
        "prior_audit_candidate_event_count": (
            PRIOR_AUDIT_CANDIDATE_EVENT_COUNTS.get(question_id)
        ),
        "matches_prior_audit_candidate_event_count": None,
        "field_literal_count": 0,
        "protected_literal_count": 0,
        "masked_chars": 0,
        "exception_type": None,
        "exception_message": None,
        "exception_stage": None,
        "prepare_success": False,
    }


def record_prepare_exception(
    diagnostic: dict[str, Any],
    error: Exception,
    stage: str,
) -> dict[str, Any]:
    diagnostic.update(
        {
            "exception_type": type(error).__name__,
            "exception_message": safe_exception_message(error),
            "exception_stage": stage,
            "prepare_success": False,
        }
    )
    return diagnostic


def safe_exception_message(error: Exception) -> str:
    if isinstance(error, (ExperimentPreparationError, ValueError, TypeError)):
        return str(error)
    return type(error).__name__


def orchestrator_exception_stage(error: Exception) -> str:
    return "resolver" if traceback_mentions(error, "holding_event_resolver.py") else "orchestrator"


def traceback_mentions(error: Exception, filename: str) -> bool:
    return any(
        frame.filename.replace("\\", "/").endswith(filename)
        for frame in traceback.extract_tb(error.__traceback__)
    )


def draft_event_rows(draft: Any) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for section in getattr(draft, "answer_sections", ()) or ():
        content = dict(getattr(section, "content", {}) or {})
        events = content.get("events")
        if isinstance(events, list):
            rows.extend(event for event in events if isinstance(event, Mapping))
    return rows


def claim_from_builder(
    draft: Any,
    events: Sequence[tuple[int, Mapping[str, Any]]],
    builder: _ClaimBuilder,
) -> CompactClaim:
    if not builder.fields:
        raise ExperimentPreparationError("no_claim_fields")
    company = _text(events[0][1].get("corp_name"))
    reporter = _text(events[0][1].get("reporter"))
    text = _render(company, reporter, builder.fields)
    if not text:
        raise ExperimentPreparationError("empty_claim")
    return CompactClaim(
        question=_text(getattr(draft, "question", None)) or "",
        company=company,
        reporter=reporter,
        fields=tuple(builder.fields),
        citations=tuple(builder.citations),
        deterministic_text=text,
    )


def build_all_candidate_claim(
    draft: Any,
    resolution: Any,
    *,
    task_type: str | None,
) -> tuple[CompactClaim, tuple[str, ...], int]:
    """Build the production-selected holding events without either size cap.

    Selection, value rendering, citation lookup, event ordering, and provenance
    all come from the production compact-claim helpers.  Only the two terminal
    size checks in ``build_compact_claim`` are intentionally absent here.
    """

    if task_type != SUPPORTED_TASK_TYPE:
        raise ExperimentPreparationError("unsupported_task_type")
    if draft is None or resolution is None or not getattr(draft, "answerable", False):
        raise ExperimentPreparationError("not_answerable")

    resolved = _resolution_mapping(resolution)
    requested = tuple(str(field) for field in resolved.get("requested_fields") or ())
    if not requested:
        raise ExperimentPreparationError("no_requested_fields")
    if resolved.get("unresolved_fields"):
        raise ExperimentPreparationError("unresolved_fields")

    events = _matching_events(draft, requested)
    if not events:
        raise ExperimentPreparationError("no_matching_event")

    builder = _ClaimBuilder(_citation_index(draft))
    for event_index, event in events:
        if not builder.add_event(event_index, event, requested):
            raise ExperimentPreparationError("citation_not_linked")
    if not builder.fields:
        raise ExperimentPreparationError("no_claim_fields")

    company = _text(events[0][1].get("corp_name"))
    reporter = _text(events[0][1].get("reporter"))
    text = _render(company, reporter, builder.fields)
    if not text:
        raise ExperimentPreparationError("empty_claim")

    return (
        CompactClaim(
            question=_text(getattr(draft, "question", None)) or "",
            company=company,
            reporter=reporter,
            fields=tuple(builder.fields),
            citations=tuple(builder.citations),
            deterministic_text=text,
        ),
        requested,
        len(events),
    )


def real_structure_diagnostics(
    claim: CompactClaim,
    detached: DetachedClaimInput,
    *,
    requested_fields: Sequence[str],
    candidate_event_count: int,
) -> dict[str, Any]:
    """Describe actual claim structure without changing event selection."""

    value_counts = Counter(field.value for field in claim.fields)
    duplicate_literals = [
        {"value": value, "count": count}
        for value, count in sorted(value_counts.items())
        if count > 1
    ]
    width = len(requested_fields)
    event_signatures = [
        tuple(field.value for field in claim.fields[offset : offset + width])
        for offset in range(0, len(claim.fields), width)
    ]
    signature_counts = Counter(event_signatures)
    duplicate_event_count = sum(count - 1 for count in signature_counts.values())
    event_citation_sets = [
        list(attachment.markers) for attachment in detached.attachments
    ]
    return {
        "event_citation_sets": event_citation_sets,
        "requested_fields_per_event": [
            list(requested_fields) for _ in range(candidate_event_count)
        ],
        "multi_source_event_count": sum(
            1 for markers in event_citation_sets if len(markers) > 1
        ),
        "duplicate_literal_values": duplicate_literals,
        "duplicate_event_count": duplicate_event_count,
    }


def run_prepared_question(
    prepared: PreparedHoldingQuestion,
    *,
    transport: Any,
    settings: HcxSettings,
    repeat_index: int,
    show_output: bool = False,
    strict_event_order: bool = False,
    check_structured_text_leakage: bool = False,
) -> dict[str, Any]:
    """Make one HCX call and run every detached-citation safety check."""

    record = base_run_record(prepared, repeat_index)
    record["strict_event_order"] = strict_event_order
    protection = prepared.detached.protection
    system_prompt = (
        STRICT_EVENT_ORDER_SYSTEM_PROMPT
        if strict_event_order
        else EXPERIMENT_SYSTEM_PROMPT
    )
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": protection.masked},
        ],
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    try:
        response = transport.post_json(
            settings.endpoint,
            headers=settings.request_headers(),
            payload=payload,
            timeout_seconds=settings.timeout_seconds,
        )
    except EmbeddingHttpError as error:
        record["transport_error"] = (
            "timeout" if error.status_code is None else f"HTTP {error.status_code}"
        )
        return finalize_run(record)
    except Exception as error:  # noqa: BLE001 - continue the live sweep
        record["transport_error"] = type(error).__name__
        return finalize_run(record)

    raw = _response_content(response) if isinstance(response, Mapping) else None
    record["candidate_received"] = raw is not None
    record["finish_reason"] = _finish_reason(response)
    record["completion_tokens"] = _completion_tokens(response)
    if show_output:
        record["raw_hcx_candidate"] = raw
    if raw is None:
        return finalize_run(record)

    record["expected_field_placeholders"] = list(protection.placeholders)
    record["found_field_placeholders"] = PLACEHOLDER_PATTERN.findall(raw)

    record = evaluate_live_detached_candidate(
        record,
        raw,
        claim=prepared.claim,
        detached=prepared.detached,
        show_output=show_output,
        check_structured_text_leakage=check_structured_text_leakage,
    )
    return finalize_run(record)


def split_prepared_events(
    prepared: PreparedHoldingQuestion,
) -> tuple[PreparedHoldingQuestion, ...]:
    """Create one-event claims without changing field, citation, or event order."""

    groups = _group_event_fields(prepared.claim.fields)
    if len(groups) != prepared.candidate_event_count:
        raise ValueError("prepared event count does not match claim field groups")

    event_prepared: list[PreparedHoldingQuestion] = []
    for fields in groups:
        markers = {field.marker for field in fields}
        citations = tuple(
            citation
            for citation in prepared.claim.citations
            if citation.marker in markers
        )
        if {citation.marker for citation in citations} != markers:
            raise ValueError("event citation mapping is incomplete")
        deterministic_text = _render(
            prepared.claim.company,
            prepared.claim.reporter,
            fields,
        )
        claim = CompactClaim(
            question=prepared.claim.question,
            company=prepared.claim.company,
            reporter=prepared.claim.reporter,
            fields=tuple(fields),
            citations=citations,
            deterministic_text=deterministic_text,
        )
        detached = detach_live_claim_citations(claim)
        event_requested_fields = tuple(field.name for field in fields)
        diagnostics = dict(prepared.prepare_diagnostics)
        diagnostics.update(
            {
                "candidate_event_count": 1,
                "claim_field_count": len(fields),
                "claim_citation_count": len(citations),
                "field_literal_count": len(fields),
                "protected_literal_count": len(detached.protection.literals),
                "masked_chars": len(detached.protection.masked),
            }
        )
        structure = real_structure_diagnostics(
            claim,
            detached,
            requested_fields=event_requested_fields,
            candidate_event_count=1,
        )
        event_prepared.append(
            replace(
                prepared,
                requested_fields=event_requested_fields,
                candidate_event_count=1,
                claim=claim,
                detached=detached,
                structure=structure,
                prepare_diagnostics=diagnostics,
            )
        )
    return tuple(event_prepared)


def run_event_wise_question(
    prepared: PreparedHoldingQuestion,
    *,
    transport: Any,
    settings: HcxSettings,
    repeat_index: int,
    show_output: bool = False,
    clock: Any = time.perf_counter,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Call HCX once per event and fall back only failed individual events."""

    event_records: list[dict[str, Any]] = []
    served_outputs: list[str] = []
    for event_index, event_prepared in enumerate(split_prepared_events(prepared)):
        started = clock()
        # Capture the validated output internally so the question can be
        # assembled. It is stripped from diagnostics unless --show-output was
        # explicitly requested.
        record = run_prepared_question(
            event_prepared,
            transport=transport,
            settings=settings,
            repeat_index=repeat_index,
            show_output=True,
            strict_event_order=True,
            check_structured_text_leakage=True,
        )
        elapsed_ms = round(max(clock() - started, 0.0) * 1000, 3)
        hcx_event_success = record.get("hcx_success") is True
        hcx_output = record.get("final_output")
        if hcx_event_success and not isinstance(hcx_output, str):
            hcx_event_success = False
            record["hcx_success"] = False
            record["hcx_failure_reason"] = "final_output_missing"

        fallback_used = not hcx_event_success
        served_output = (
            event_prepared.claim.deterministic_text
            if fallback_used
            else str(hcx_output)
        )
        served_outputs.append(served_output)
        record.update(
            {
                "event_index": event_index,
                "event_number": event_index + 1,
                "requested_fields": list(event_prepared.requested_fields),
                "protected_placeholders_by_type": record.get(
                    "expected_placeholders_by_type"
                ),
                "elapsed_ms": elapsed_ms,
                "pre_validator_valid": record.get(
                    "candidate_valid_before_citation_attachment"
                )
                is True,
                "citation_attachment_valid": record.get(
                    "deterministic_citations_attached"
                )
                is True,
                "final_validator_valid": record.get("final_answer_valid") is True,
                "hcx_event_success": hcx_event_success,
                "fallback_used": fallback_used,
                "fallback_reason": (
                    record.get("hcx_failure_reason") if fallback_used else None
                ),
            }
        )
        if show_output:
            record["served_event_output"] = served_output
        else:
            for key in (
                "raw_hcx_candidate",
                "restored_output",
                "validation_reference_text",
                "restored_candidate_text",
                "final_output",
            ):
                record.pop(key, None)
        event_records.append(record)

    success_count = sum(
        record.get("hcx_event_success") is True for record in event_records
    )
    failure_count = len(event_records) - success_count
    question_served_success = (
        len(served_outputs) == prepared.candidate_event_count
        and all(output.strip() for output in served_outputs)
    )
    question_record: dict[str, Any] = {
        "question_id": prepared.question_id,
        "question": prepared.question,
        "repeat_index": repeat_index,
        "event_count": prepared.candidate_event_count,
        "total_hcx_calls": len(event_records),
        "hcx_event_success_count": success_count,
        "hcx_event_failure_count": failure_count,
        "hcx_event_success_rate": (
            round(success_count / len(event_records), 6) if event_records else 0.0
        ),
        "question_all_events_hcx_success": bool(event_records)
        and failure_count == 0,
        "fallback_event_count": failure_count,
        "question_served_success": question_served_success,
        "total_completion_tokens": sum(
            int(record.get("completion_tokens") or 0) for record in event_records
        ),
        "total_elapsed_ms": round(
            sum(float(record.get("elapsed_ms") or 0.0) for record in event_records),
            3,
        ),
        "failure_reasons": dict(
            sorted(
                Counter(
                    str(record.get("fallback_reason") or "unknown_failure")
                    for record in event_records
                    if record.get("fallback_used") is True
                ).items()
            )
        ),
        "served_event_indexes": [
            int(record["event_index"]) for record in event_records
        ],
    }
    if show_output:
        question_record["served_answer"] = "\n".join(served_outputs)
    return question_record, event_records


def base_run_record(
    prepared: PreparedHoldingQuestion,
    repeat_index: int,
) -> dict[str, Any]:
    protection = prepared.detached.protection
    placeholder_breakdown = placeholder_diagnostics(protection, ())
    return {
        "question_id": prepared.question_id,
        "question": prepared.question,
        "repeat_index": repeat_index,
        "task_type": prepared.task_type,
        "requested_fields": list(prepared.requested_fields),
        "candidate_event_count": prepared.candidate_event_count,
        "event_count": prepared.candidate_event_count,
        "field_literal_count": len(prepared.claim.fields),
        "protected_literal_count": len(protection.literals),
        "masked_chars": len(protection.masked),
        "candidate_received": False,
        "finish_reason": None,
        "completion_tokens": None,
        "field_placeholder_integrity_valid": False,
        "field_placeholder_integrity_reason": None,
        "all_events_kept": False,
        "unexpected_citation_generation": False,
        "unprotected_numeric_tokens_in_masked_candidate": [],
        "unprotected_text_literals_in_masked_candidate": [],
        "forbidden_investment_language_detected": False,
        "forbidden_investment_terms": [],
        "candidate_valid_before_citation_attachment": False,
        "validator_reason": None,
        "deterministic_citations_attached": False,
        "attached_citation_count": 0,
        "citation_attachment_reason": None,
        "final_answer_valid": False,
        "final_validator_reason": None,
        "inference_markers": [],
        "hcx_success": False,
        "hcx_failure_reason": None,
        "expected_field_placeholders": list(protection.placeholders),
        "found_field_placeholders": [],
        **placeholder_breakdown,
        **dict(prepared.prepare_diagnostics),
        **dict(prepared.structure),
    }


def finalize_run(record: dict[str, Any]) -> dict[str, Any]:
    """Apply the fail-closed success predicate and name its first failure."""

    success = all(
        (
            record.get("candidate_received") is True,
            record.get("field_placeholder_integrity_valid") is True,
            record.get("all_events_kept") is True,
            record.get("unexpected_citation_generation") is False,
            not record.get("unprotected_numeric_tokens_in_masked_candidate"),
            not record.get("unprotected_text_literals_in_masked_candidate"),
            record.get("forbidden_investment_language_detected") is False,
            record.get("candidate_valid_before_citation_attachment") is True,
            record.get("deterministic_citations_attached") is True,
            record.get("final_answer_valid") is True,
            not record.get("inference_markers"),
        )
    )
    record["hcx_success"] = success
    record["hcx_failure_reason"] = None if success else first_failure_reason(record)
    return record


def first_failure_reason(record: Mapping[str, Any]) -> str:
    if record.get("preparation_error"):
        return str(record["preparation_error"])
    if record.get("transport_error"):
        return "transport_error"
    if not record.get("candidate_received"):
        return "candidate_not_received"
    if not record.get("field_placeholder_integrity_valid"):
        return str(
            record.get("field_placeholder_integrity_reason")
            or "placeholder_integrity_failed"
        )
    if record.get("unexpected_citation_generation"):
        return "unexpected_citation_generation"
    if record.get("unprotected_numeric_tokens_in_masked_candidate"):
        return "unprotected_numeric_generation"
    if record.get("unprotected_text_literals_in_masked_candidate"):
        return "unprotected_structured_text_leakage"
    if not record.get("candidate_valid_before_citation_attachment"):
        return str(record.get("validator_reason") or "pre_validation_failed")
    if record.get("forbidden_investment_language_detected"):
        return "forbidden_investment_language"
    if not record.get("deterministic_citations_attached"):
        return str(
            record.get("citation_attachment_reason")
            or "citation_attachment_failed"
        )
    if not record.get("final_answer_valid"):
        return str(record.get("final_validator_reason") or "final_validation_failed")
    if record.get("inference_markers"):
        return "inference_marker_added"
    return "unknown_failure"


def preparation_failure_record(
    row: Mapping[str, Any],
    repeat_index: int,
    diagnostic: Mapping[str, Any],
    *,
    strict_event_order: bool = False,
) -> dict[str, Any]:
    reason = str(
        diagnostic.get("exception_message")
        or diagnostic.get("exception_type")
        or "preparation_failed"
    )
    record = {
        "question_id": str(row.get("question_id") or ""),
        "question": str(row.get("query") or ""),
        "repeat_index": repeat_index,
        "strict_event_order": strict_event_order,
        "task_type": None,
        "requested_fields": [],
        "candidate_event_count": 0,
        "field_literal_count": 0,
        "protected_literal_count": 0,
        "masked_chars": 0,
        "candidate_received": False,
        "finish_reason": None,
        "completion_tokens": None,
        "field_placeholder_integrity_valid": False,
        "field_placeholder_integrity_reason": None,
        "all_events_kept": False,
        "unexpected_citation_generation": False,
        "unprotected_numeric_tokens_in_masked_candidate": [],
        "unprotected_text_literals_in_masked_candidate": [],
        "forbidden_investment_language_detected": False,
        "forbidden_investment_terms": [],
        "candidate_valid_before_citation_attachment": False,
        "validator_reason": None,
        "deterministic_citations_attached": False,
        "attached_citation_count": 0,
        "citation_attachment_reason": None,
        "final_answer_valid": False,
        "final_validator_reason": None,
        "inference_markers": [],
        "event_citation_sets": [],
        "requested_fields_per_event": [],
        "multi_source_event_count": 0,
        "duplicate_literal_values": [],
        "duplicate_event_count": 0,
        "expected_field_placeholders": [],
        "found_field_placeholders": [],
        "expected_placeholders_by_type": {
            kind: [] for kind in _FIELD_PLACEHOLDER_KINDS
        },
        "found_placeholders_by_type": {
            kind: [] for kind in _FIELD_PLACEHOLDER_KINDS
        },
        "missing_placeholders_by_type": {
            kind: [] for kind in _FIELD_PLACEHOLDER_KINDS
        },
        "text_placeholders_all_preserved": False,
        **dict(diagnostic),
        "preparation_error": reason,
        "hcx_success": False,
        "hcx_failure_reason": reason,
    }
    return record


def event_wise_preparation_failure(
    row: Mapping[str, Any],
    repeat_index: int,
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    reason = str(
        diagnostic.get("exception_message")
        or diagnostic.get("exception_type")
        or "preparation_failed"
    )
    return {
        "question_id": str(row.get("question_id") or ""),
        "question": str(row.get("query") or ""),
        "repeat_index": repeat_index,
        "event_count": 0,
        "total_hcx_calls": 0,
        "hcx_event_success_count": 0,
        "hcx_event_failure_count": 0,
        "hcx_event_success_rate": 0.0,
        "question_all_events_hcx_success": False,
        "fallback_event_count": 0,
        "question_served_success": False,
        "total_completion_tokens": 0,
        "total_elapsed_ms": 0.0,
        "failure_reasons": {reason: 1},
        "served_event_indexes": [],
        "preparation_error": reason,
    }


def summarize_prepare_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures = [row for row in diagnostics if row.get("prepare_success") is not True]
    return {
        "question_count": len(diagnostics),
        "prepare_success_count": len(diagnostics) - len(failures),
        "prepare_failure_count": len(failures),
        "failure_question_ids": [str(row["question_id"]) for row in failures],
        "failure_stages": dict(
            sorted(
                Counter(
                    str(row.get("exception_stage") or "unknown")
                    for row in failures
                ).items()
            )
        ),
        "failure_messages": dict(
            sorted(
                Counter(
                    str(row.get("exception_message") or "unknown")
                    for row in failures
                ).items()
            )
        ),
        "all_prior_audit_counts_match": all(
            row.get("matches_prior_audit_candidate_event_count") is not False
            for row in diagnostics
        ),
    }


def summarize_live_runs(
    runs: Sequence[Mapping[str, Any]],
    *,
    repeat: int,
) -> dict[str, Any]:
    """Summarize success by question and actual claim size."""

    by_question: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        by_question.setdefault(str(run.get("question_id") or ""), []).append(run)
    clean_questions = {
        question_id
        for question_id, question_runs in by_question.items()
        if len(question_runs) == repeat
        and all(run.get("hcx_success") is True for run in question_runs)
    }
    successful_runs = [run for run in runs if run.get("hcx_success") is True]
    failures = Counter(
        str(run.get("hcx_failure_reason") or "unknown_failure")
        for run in runs
        if run.get("hcx_success") is not True
    )
    question_summary = {
        question_id: _summarize_question_runs(question_runs)
        for question_id, question_runs in sorted(by_question.items())
    }
    return {
        "question_count": len(by_question),
        "run_count": len(runs),
        "hcx_success_count": len(clean_questions),
        "hcx_successful_run_count": len(successful_runs),
        "failure_reasons": dict(sorted(failures.items())),
        "by_question": question_summary,
        "by_candidate_event_count": _group_runs(runs, "candidate_event_count"),
        "by_protected_literal_count": _group_runs(runs, "protected_literal_count"),
        "max_successful_event_count": max(
            (int(run["candidate_event_count"]) for run in successful_runs),
            default=None,
        ),
        "max_successful_protected_literal_count": max(
            (int(run["protected_literal_count"]) for run in successful_runs),
            default=None,
        ),
        "all_questions_clean": (
            len(by_question) == len(TARGET_QUESTION_IDS)
            and len(clean_questions) == len(TARGET_QUESTION_IDS)
        ),
        "all_selected_questions_clean": bool(by_question)
        and len(clean_questions) == len(by_question),
    }


def summarize_event_wise_runs(
    question_runs: Sequence[Mapping[str, Any]],
    event_runs: Sequence[Mapping[str, Any]],
    *,
    repeat: int,
) -> dict[str, Any]:
    """Aggregate per-event HCX fidelity and deterministic serving fallback."""

    by_question: dict[str, list[Mapping[str, Any]]] = {}
    for run in question_runs:
        by_question.setdefault(str(run.get("question_id") or ""), []).append(run)

    question_summary: dict[str, dict[str, Any]] = {}
    for question_id, runs in sorted(by_question.items()):
        total_calls = sum(int(run.get("total_hcx_calls") or 0) for run in runs)
        success_count = sum(
            int(run.get("hcx_event_success_count") or 0) for run in runs
        )
        failure_count = sum(
            int(run.get("hcx_event_failure_count") or 0) for run in runs
        )
        failure_reasons: Counter[str] = Counter()
        for run in runs:
            failure_reasons.update(
                {
                    str(reason): int(count)
                    for reason, count in dict(
                        run.get("failure_reasons") or {}
                    ).items()
                }
            )
        event_counts = sorted({int(run.get("event_count") or 0) for run in runs})
        question_summary[question_id] = {
            "runs": len(runs),
            "event_count": (
                event_counts[0] if len(event_counts) == 1 else event_counts
            ),
            "total_hcx_calls": total_calls,
            "hcx_event_success_count": success_count,
            "hcx_event_failure_count": failure_count,
            "hcx_event_success_rate": (
                round(success_count / total_calls, 6) if total_calls else 0.0
            ),
            "question_all_events_hcx_success": len(runs) == repeat
            and all(
                run.get("question_all_events_hcx_success") is True for run in runs
            ),
            "fallback_event_count": sum(
                int(run.get("fallback_event_count") or 0) for run in runs
            ),
            "question_served_success": len(runs) == repeat
            and all(run.get("question_served_success") is True for run in runs),
            "total_completion_tokens": sum(
                int(run.get("total_completion_tokens") or 0) for run in runs
            ),
            "total_elapsed_ms": round(
                sum(float(run.get("total_elapsed_ms") or 0.0) for run in runs),
                3,
            ),
            "failure_reasons": dict(sorted(failure_reasons.items())),
        }

    event_success_count = sum(
        run.get("hcx_event_success") is True for run in event_runs
    )
    return {
        "question_count": len(by_question),
        "repeat": repeat,
        "total_hcx_calls": len(event_runs),
        "hcx_event_success_count": event_success_count,
        "hcx_event_failure_count": len(event_runs) - event_success_count,
        "hcx_event_success_rate": (
            round(event_success_count / len(event_runs), 6)
            if event_runs
            else 0.0
        ),
        "fallback_event_count": sum(
            run.get("fallback_used") is True for run in event_runs
        ),
        "all_questions_served_successfully": bool(question_runs)
        and all(run.get("question_served_success") is True for run in question_runs),
        "all_events_hcx_success": bool(event_runs)
        and all(run.get("hcx_event_success") is True for run in event_runs),
        "total_completion_tokens": sum(
            int(run.get("completion_tokens") or 0) for run in event_runs
        ),
        "total_elapsed_ms": round(
            sum(float(run.get("elapsed_ms") or 0.0) for run in event_runs),
            3,
        ),
        "failure_reasons": dict(
            sorted(
                Counter(
                    str(run.get("fallback_reason") or "unknown_failure")
                    for run in event_runs
                    if run.get("fallback_used") is True
                ).items()
            )
        ),
        "by_question": question_summary,
    }


def _summarize_question_runs(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failures = [run for run in runs if run.get("hcx_success") is not True]
    expected_counts = [
        len(tuple(run.get("expected_field_placeholders") or ())) for run in runs
    ]
    found_counts = [
        len(tuple(run.get("found_field_placeholders") or ())) for run in runs
    ]
    unique_expected = sorted(set(expected_counts))
    return {
        "runs": len(runs),
        "successes": sum(run.get("hcx_success") is True for run in runs),
        "failure_reasons": dict(
            sorted(
                Counter(
                    str(run.get("hcx_failure_reason") or "unknown_failure")
                    for run in failures
                ).items()
            )
        ),
        "failure_categories": dict(
            sorted(Counter(_failure_category(run) for run in failures).items())
        ),
        "expected_placeholder_count": (
            unique_expected[0] if len(unique_expected) == 1 else unique_expected
        ),
        "found_placeholder_counts": found_counts,
        "run_results": [
            {
                "repeat_index": run.get("repeat_index"),
                "success": run.get("hcx_success") is True,
                "expected_placeholder_count": expected,
                "found_placeholder_count": found,
                "failure_reason": run.get("hcx_failure_reason"),
            }
            for run, expected, found in zip(
                runs,
                expected_counts,
                found_counts,
                strict=True,
            )
        ],
    }


def _failure_category(run: Mapping[str, Any]) -> str:
    reason = str(run.get("hcx_failure_reason") or "unknown_failure")
    if reason in {
        "placeholder_missing",
        "placeholder_reordered",
        "placeholder_duplicated",
    }:
        return reason
    if run.get("candidate_valid_before_citation_attachment") is not True:
        return "validator_failure"
    if run.get("deterministic_citations_attached") is not True:
        return "citation_attachment_failure"
    if run.get("final_answer_valid") is not True:
        return "validator_failure"
    return reason


def _group_runs(
    runs: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, int]]:
    grouped: dict[int, dict[str, Any]] = {}
    for run in runs:
        size = int(run.get(key) or 0)
        bucket = grouped.setdefault(
            size,
            {"runs": 0, "successful_runs": 0, "question_ids": set()},
        )
        bucket["runs"] += 1
        bucket["successful_runs"] += int(run.get("hcx_success") is True)
        bucket["question_ids"].add(str(run.get("question_id") or ""))
    return {
        str(size): {
            "question_count": len(bucket["question_ids"]),
            "runs": bucket["runs"],
            "successful_runs": bucket["successful_runs"],
        }
        for size, bucket in sorted(grouped.items())
    }


if __name__ == "__main__":
    raise SystemExit(main())
