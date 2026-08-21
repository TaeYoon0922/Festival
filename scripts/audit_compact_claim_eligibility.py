"""Explain which holding questions can produce a compact claim, and why.

Read-only.  ``build_compact_claim`` answers yes or no; this reports the counts
behind that answer so a skip can be attributed to a specific gate rather than
guessed at.  It runs the production path — query understanding, PostgreSQL
hybrid retrieval, the orchestrator, the citation-aware generator — and then
re-derives the eligibility decision from the same structured fields the adapter
reads.  HyperCLOVA X is never called.

    python scripts/audit_compact_claim_eligibility.py --question-id HX07
    python scripts/audit_compact_claim_eligibility.py --holding20

Nothing is written and no production behaviour changes.  Credentials, headers,
connection strings, and retrieved chunk text are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.pipeline import AnswerPipeline

# The private helpers are imported, not reimplemented, so this audit measures
# the gates the adapter actually applies.  A local copy would drift and then
# explain a decision that was never made.
from app.generation.compact_claim import (
    MAX_CLAIM_EVENTS,
    MAX_CLAIM_LITERALS,
    SUPPORTED_TASK_TYPE,
    _citation_index,
    _ClaimBuilder,
    _field_value,
    _matching_events,
    _render,
    _resolution_mapping,
    build_compact_claim,
)
from app.generation.protected_literals import protect_literals
from app.parsing.final_validation import HOLDING_ADDITIONAL_QUESTIONS


UNSUPPORTED_TASK_TYPE = "unsupported_task_type"
NOT_ANSWERABLE = "not_answerable"
NO_REQUESTED_FIELDS = "no_requested_fields"
UNRESOLVED_FIELDS = "unresolved_fields"
NO_MATCHING_EVENT = "no_matching_event"
MISSING_REQUESTED_FIELD = "missing_requested_field"
CITATION_NOT_LINKED = "citation_not_linked"
TOO_MANY_EVENTS = "too_many_events"
TOO_MANY_LITERALS = "too_many_literals"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report compact-claim eligibility for one question or for the whole "
            "frozen Holding20 set.  Read-only; HCX is never called."
        )
    )
    parser.add_argument("--question-id", default=None)
    parser.add_argument("--question", default=None)
    parser.add_argument(
        "--holding20",
        action="store_true",
        help="Audit every question in the frozen Holding20 set.",
    )
    parser.add_argument(
        "--expand-events",
        action="store_true",
        help=(
            "Also measure the claim that would be built if every candidate "
            "event were included, ignoring the caps.  Measurement only; the "
            "caps are not changed and no event is selected over another."
        ),
    )
    parser.add_argument(
        "--routing",
        action="store_true",
        help="Also report how the question was routed and on what signals.",
    )
    args = parser.parse_args(argv)

    if not args.holding20 and not args.question:
        parser.error("pass --question (with --question-id) or --holding20")

    pipeline = AnswerPipeline.from_env()

    options = {"expand_events": args.expand_events, "routing": args.routing}

    if args.holding20:
        rows = [
            audit_question(
                pipeline, str(row["question_id"]), str(row["query"]), **options
            )
            for row in HOLDING_ADDITIONAL_QUESTIONS
        ]
        print(json.dumps(_summarize(rows), ensure_ascii=False, indent=2))
        return 0

    row = audit_question(
        pipeline, args.question_id or "ad-hoc", args.question, **options
    )
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return 0


def audit_question(
    pipeline: AnswerPipeline,
    question_id: str,
    question: str,
    *,
    expand_events: bool = False,
    routing: bool = False,
) -> dict[str, Any]:
    """Run the production path for one question and report eligibility."""

    plan = pipeline.understanding.understand(
        question, top_k=pipeline.settings.top_k
    )
    execution = pipeline.executor.execute(plan)
    result = pipeline.orchestrator.run(question, plan, execution)
    generated = pipeline.generator.generate(result.answer_draft)

    report = analyze_eligibility(
        result.answer_draft,
        result.resolution,
        task_type=result.task_decision.task_type,
    )
    report["question_id"] = question_id
    report["answerable"] = generated.answerable
    report["deterministic_answer_chars"] = len(generated.answer_text)
    if expand_events:
        report["expanded"] = expand_all_events(result.answer_draft, result.resolution)
    if routing:
        report["routing"] = describe_routing(
            question, plan, result, generated
        )
    return report


def analyze_eligibility(
    draft: Any, resolution: Any, *, task_type: str | None
) -> dict[str, Any]:
    """Re-derive the adapter's decision, naming the gate that stopped it."""

    report: dict[str, Any] = {
        "task_type": task_type,
        "requested_fields": [],
        "unresolved_fields": [],
        "total_events": 0,
        "matches_query_events": 0,
        "complete_matching_events": 0,
        "fully_requested_field_events": 0,
        "citation_linked_events": 0,
        "candidate_event_count": 0,
        "candidate_literal_count": 0,
        "MAX_CLAIM_EVENTS": MAX_CLAIM_EVENTS,
        "MAX_CLAIM_LITERALS": MAX_CLAIM_LITERALS,
        "compact_eligible": False,
        "skip_reason": None,
    }

    # Fill in what is knowable before any gate fires, so a skipped question
    # still shows what was asked of it.
    resolved = _resolution_mapping(resolution) if resolution is not None else {}
    requested = tuple(resolved.get("requested_fields") or ())
    report["requested_fields"] = list(requested)
    report["unresolved_fields"] = list(resolved.get("unresolved_fields") or ())

    if task_type != SUPPORTED_TASK_TYPE:
        report["skip_reason"] = UNSUPPORTED_TASK_TYPE
        return _confirm(report, draft, resolution, task_type)
    if draft is None or resolution is None or not getattr(draft, "answerable", False):
        # The composer already declined; the compact gates were never reached.
        report["skip_reason"] = NOT_ANSWERABLE
        return _confirm(report, draft, resolution, task_type)

    unresolved = report["unresolved_fields"]
    rows = _event_rows(draft)
    report["total_events"] = len(rows)
    report["matches_query_events"] = sum(
        1 for row in rows if row.get("matches_query") is True
    )
    report["complete_matching_events"] = sum(
        1
        for row in rows
        if row.get("matches_query") is True and not row.get("field_conflict")
    )

    if not requested:
        report["skip_reason"] = NO_REQUESTED_FIELDS
        return _confirm(report, draft, resolution, task_type)
    if unresolved:
        report["skip_reason"] = UNRESOLVED_FIELDS
        return _confirm(report, draft, resolution, task_type)

    report["fully_requested_field_events"] = sum(
        1
        for row in rows
        if row.get("matches_query") is True
        and not row.get("field_conflict")
        and all(_field_value(row, field) is not None for field in requested)
    )

    candidates = _matching_events(draft, requested)
    report["candidate_event_count"] = len(candidates)
    report["candidate_literal_count"] = len(candidates) * len(requested)

    citations = _citation_index(draft)
    report["citation_linked_events"] = sum(
        1
        for index, _ in candidates
        if all(
            (index, field) in citations or (index, "event_evidence") in citations
            for field in requested
        )
    )

    if not candidates:
        # Every event was filtered out.  Say whether that was for a missing
        # value or because nothing matched the question at all.
        report["skip_reason"] = (
            MISSING_REQUESTED_FIELD
            if report["complete_matching_events"] > 0
            else NO_MATCHING_EVENT
        )
    elif len(candidates) > MAX_CLAIM_EVENTS:
        report["skip_reason"] = TOO_MANY_EVENTS
    elif report["citation_linked_events"] < len(candidates):
        report["skip_reason"] = CITATION_NOT_LINKED
    elif report["candidate_literal_count"] > MAX_CLAIM_LITERALS:
        report["skip_reason"] = TOO_MANY_LITERALS
    else:
        report["compact_eligible"] = True

    return _confirm(report, draft, resolution, task_type)


def expand_all_events(draft: Any, resolution: Any) -> dict[str, Any]:
    """Measure the claim that would exist if every candidate event were kept.

    This builds the claim the adapter would build with the caps lifted.  It uses
    the adapter's own builder, takes the events in composer order, and drops
    none: the point is to size the honest all-events claim, not to choose among
    events.  Nothing here changes the caps or the served answer.
    """

    resolved = _resolution_mapping(resolution) if resolution is not None else {}
    requested = tuple(resolved.get("requested_fields") or ())
    empty = {
        "requested_fields": list(requested),
        "candidate_event_count": 0,
        "candidate_literal_count": 0,
        "compact_claim_chars_if_all_events_included": 0,
        "protected_literal_count_if_all_events_included": 0,
        "estimated_masked_chars": 0,
        "citations_count": 0,
        "duplicated_factual_values": [],
        "duplicate_events": [],
        "events_satisfying_all_query_constraints": 0,
        "event_constraints": [],
        "buildable": False,
    }
    if draft is None or not requested:
        return empty

    candidates = _matching_events(draft, requested)
    if not candidates:
        return empty

    builder = _ClaimBuilder(_citation_index(draft))
    for event_index, event in candidates:
        if not builder.add_event(event_index, event, requested):
            empty["candidate_event_count"] = len(candidates)
            empty["note"] = "a field could not be linked to a citation"
            return empty

    company = _text_field(candidates[0][1], "corp_name")
    reporter = _text_field(candidates[0][1], "reporter")
    text = _render(company, reporter, builder.fields)
    protection = protect_literals(text)

    values = [f"{field.label}={field.value}" for field in builder.fields]
    signatures = [
        tuple(_field_value(event, field) for field in requested)
        for _, event in candidates
    ]
    constraints = [_event_constraints(event) for _, event in candidates]

    return {
        "requested_fields": list(requested),
        "candidate_event_count": len(candidates),
        "candidate_literal_count": len(builder.literal_count()),
        "compact_claim_chars_if_all_events_included": len(text),
        "protected_literal_count_if_all_events_included": len(protection.literals),
        "estimated_masked_chars": len(protection.masked),
        "citations_count": len(builder.citations),
        "duplicated_factual_values": _duplicates(values),
        "duplicate_events": _duplicates([str(item) for item in signatures]),
        "events_satisfying_all_query_constraints": sum(
            1 for row in constraints if row["satisfies_all"]
        ),
        "event_constraints": constraints,
        "buildable": True,
    }


def describe_routing(
    question: str, plan: Any, result: Any, generated: Any
) -> dict[str, Any]:
    """Show how a question was routed and whether holding structure exists anyway."""

    decision = result.task_decision
    evidence = result.evidence_set
    group_types = [group.group_type for group in evidence.evidence_groups]
    sections = [
        {
            "title": section.title,
            "content_keys": sorted(dict(section.content or {}).keys()),
        }
        for section in getattr(result.answer_draft, "answer_sections", ()) or ()
    ]
    return {
        "question": question,
        "plan_task_type": getattr(plan, "task_type", None),
        "plan_disclosure_route": list(_as_tuple(getattr(plan, "disclosure_route", ()))),
        "plan_metric": getattr(plan, "metric", None),
        "plan_reporter": getattr(plan, "reporter", None),
        "plan_event_type": getattr(plan, "event_type", None),
        "routed_task_type": decision.task_type,
        "resolver_type": decision.resolver_type,
        "route_confidence": decision.confidence,
        "matched_signals": list(decision.matched_signals),
        "router_warnings": list(decision.warnings),
        "draft_task_type": getattr(result.answer_draft, "task_type", None),
        "answer_sections": sections,
        "evidence_group_types": group_types,
        "holding_evidence_groups": sum(
            1 for value in group_types if value == "holding_event"
        ),
        "resolution_type": type(result.resolution).__name__
        if result.resolution is not None
        else None,
        "answerable": generated.answerable,
        "answer_basis": dict(generated.confidence or {}).get("basis"),
        "answer_warnings": list(generated.warnings),
    }


def _event_constraints(event: Mapping[str, Any]) -> dict[str, Any]:
    matches = event.get("matches_query")
    temporal = event.get("temporal_match")
    direction = event.get("direction_match")
    return {
        "reference_date": event.get("reference_date"),
        "matches_query": matches,
        "temporal_match": temporal,
        "direction_match": direction,
        "field_conflict": bool(event.get("field_conflict")),
        # ``None`` means the question placed no such constraint, which is not a
        # failure to satisfy one.
        "satisfies_all": matches is True
        and temporal is not False
        and direction is not False
        and not event.get("field_conflict"),
    }


def _duplicates(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _text_field(event: Mapping[str, Any], name: str) -> str | None:
    value = event.get(name)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return ()


def _confirm(
    report: dict[str, Any], draft: Any, resolution: Any, task_type: str | None
) -> dict[str, Any]:
    """Cross-check the re-derived verdict against the adapter itself.

    If these ever disagree the audit is wrong, not the adapter, and saying so is
    better than reporting a confident but fabricated reason.
    """

    claim = build_compact_claim(draft, resolution, task_type=task_type)
    report["builder_returned_claim"] = claim is not None
    report["audit_agrees_with_builder"] = report["compact_eligible"] == (
        claim is not None
    )
    if claim is not None:
        report["claim_chars"] = len(claim.deterministic_text)
        report["claim_field_count"] = len(claim.fields)
        report["claim_citation_count"] = len(claim.citations)
    return report


def _event_rows(draft: Any) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for section in getattr(draft, "answer_sections", ()) or ():
        content = dict(getattr(section, "content", {}) or {})
        events = content.get("events")
        if isinstance(events, list):
            rows.extend(row for row in events if isinstance(row, Mapping))
    return rows


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    reasons = Counter(
        row["skip_reason"] for row in rows if row["skip_reason"] is not None
    )
    eligible = sum(1 for row in rows if row["compact_eligible"])
    return {
        "questions": [
            {
                "question_id": row["question_id"],
                "answerable": row["answerable"],
                "requested_fields": row["requested_fields"],
                "complete_matching_events": row["complete_matching_events"],
                "candidate_event_count": row["candidate_event_count"],
                "candidate_literal_count": row["candidate_literal_count"],
                "compact_eligible": row["compact_eligible"],
                "skip_reason": row["skip_reason"],
            }
            for row in rows
        ],
        "summary": {
            "holding_questions": len(rows),
            "compact_eligible_count": eligible,
            "too_many_events_count": reasons.get(TOO_MANY_EVENTS, 0),
            "too_many_literals_count": reasons.get(TOO_MANY_LITERALS, 0),
            "other_skip_count": sum(reasons.values())
            - reasons.get(TOO_MANY_EVENTS, 0)
            - reasons.get(TOO_MANY_LITERALS, 0),
            "skip_reasons": dict(sorted(reasons.items())),
            "audit_disagreements": [
                row["question_id"]
                for row in rows
                if not row["audit_agrees_with_builder"]
            ],
        },
        "limits": {
            "MAX_CLAIM_EVENTS": MAX_CLAIM_EVENTS,
            "MAX_CLAIM_LITERALS": MAX_CLAIM_LITERALS,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
