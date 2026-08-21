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
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.pipeline import AnswerPipeline
from app.generation.answer_generator import _text
from app.generation.compact_claim import (
    MAX_CLAIM_EVENTS,
    MAX_CLAIM_LITERALS,
    SUPPORTED_TASK_TYPE,
    CompactClaim,
    _citation_index,
    _ClaimBuilder,
    _matching_events,
    _render,
    _resolution_mapping,
)
from app.generation.hcx_verbalizer import HcxSettings, _response_content
from app.parsing.final_validation import HOLDING_ADDITIONAL_QUESTIONS
from app.retrieval.embeddings import EmbeddingHttpError, UrllibJsonTransport
from scripts.experiment_hcx_multi_event import (
    EXPERIMENT_SYSTEM_PROMPT,
    DetachedClaimInput,
    _completion_tokens,
    _evaluate_detached_candidate,
    _finish_reason,
    detach_claim_citations,
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
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    settings = HcxSettings.from_env()
    if not settings.configured:
        print(
            "FESTIVAL_HCX_API_KEY is not set; this experiment needs a live key.",
            file=sys.stderr,
        )
        return 1

    pipeline = AnswerPipeline.from_env()
    transport = UrllibJsonTransport()
    rows = target_question_rows()
    runs: list[dict[str, Any]] = []
    for row in rows:
        question_id = str(row["question_id"])
        print(f"[prepare] {question_id}", file=sys.stderr, flush=True)
        try:
            prepared = prepare_actual_question(pipeline, row)
        except Exception as error:  # noqa: BLE001 - continue the diagnostic sweep
            for repeat_index in range(1, args.repeat + 1):
                runs.append(
                    preparation_failure_record(row, repeat_index, error)
                )
            continue

        for repeat_index in range(1, args.repeat + 1):
            print(
                f"[hcx] {question_id} run {repeat_index}/{args.repeat}",
                file=sys.stderr,
                flush=True,
            )
            runs.append(
                run_prepared_question(
                    prepared,
                    transport=transport,
                    settings=settings,
                    repeat_index=repeat_index,
                    show_output=args.show_output,
                )
            )

    report = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "repeat": args.repeat,
        "target_question_ids": list(TARGET_QUESTION_IDS),
        "production_caps_observed_not_applied": {
            "MAX_CLAIM_EVENTS": MAX_CLAIM_EVENTS,
            "MAX_CLAIM_LITERALS": MAX_CLAIM_LITERALS,
        },
        "runs": runs,
        "summary": summarize_live_runs(runs, repeat=args.repeat),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def target_question_rows() -> tuple[dict[str, Any], ...]:
    """Return the requested frozen Holding questions in explicit target order."""

    by_id = {
        str(row["question_id"]): dict(row)
        for row in HOLDING_ADDITIONAL_QUESTIONS
    }
    missing = [question_id for question_id in TARGET_QUESTION_IDS if question_id not in by_id]
    if missing:
        raise RuntimeError("frozen Holding questions missing: " + ", ".join(missing))
    return tuple(by_id[question_id] for question_id in TARGET_QUESTION_IDS)


def prepare_actual_question(
    pipeline: Any,
    row: Mapping[str, Any],
) -> PreparedHoldingQuestion:
    """Run the existing read-only stack and prepare its complete event claim."""

    question_id = str(row["question_id"])
    question = str(row["query"])
    plan = pipeline.understanding.understand(
        question,
        top_k=pipeline.settings.top_k,
    )
    execution = pipeline.executor.execute(plan)
    result = pipeline.orchestrator.run(question, plan, execution)
    task_type = str(result.task_decision.task_type)
    claim, requested, event_count = build_all_candidate_claim(
        result.answer_draft,
        result.resolution,
        task_type=task_type,
    )
    detached = detach_claim_citations(claim)
    structure = real_structure_diagnostics(
        claim,
        detached,
        requested_fields=requested,
        candidate_event_count=event_count,
    )
    return PreparedHoldingQuestion(
        question_id=question_id,
        question=question,
        task_type=task_type,
        requested_fields=requested,
        candidate_event_count=event_count,
        claim=claim,
        detached=detached,
        structure=structure,
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
) -> dict[str, Any]:
    """Make one HCX call and run every detached-citation safety check."""

    record = base_run_record(prepared, repeat_index)
    protection = prepared.detached.protection
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": EXPERIMENT_SYSTEM_PROMPT},
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

    record = _evaluate_detached_candidate(
        record,
        raw,
        claim=prepared.claim,
        detached=prepared.detached,
        show_output=show_output,
    )
    return finalize_run(record)


def base_run_record(
    prepared: PreparedHoldingQuestion,
    repeat_index: int,
) -> dict[str, Any]:
    protection = prepared.detached.protection
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
    if not record.get("candidate_valid_before_citation_attachment"):
        return str(record.get("validator_reason") or "pre_validation_failed")
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
    error: Exception,
) -> dict[str, Any]:
    reason = (
        error.reason
        if isinstance(error, ExperimentPreparationError)
        else type(error).__name__
    )
    record = {
        "question_id": str(row.get("question_id") or ""),
        "question": str(row.get("query") or ""),
        "repeat_index": repeat_index,
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
        "preparation_error": reason,
        "hcx_success": False,
        "hcx_failure_reason": reason,
    }
    return record


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
    return {
        "question_count": len(by_question),
        "run_count": len(runs),
        "hcx_success_count": len(clean_questions),
        "hcx_successful_run_count": len(successful_runs),
        "failure_reasons": dict(sorted(failures.items())),
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
    }


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
