"""Deterministic candidate providers for the clarification layer."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping

from app.reasoning.clarification_request import (
    MAX_CANDIDATES,
    ClarificationCandidate,
    ClarificationDecision,
    ClarificationRequest,
    ClarificationState,
)
from app.reasoning.holding_company_role_resolution import has_role_provenance
from app.reasoning.holding_evidence_coverage import has_holding_acquisition_semantics
from app.reasoning.query_validation import QuerySlotStatus, QueryState


_BOUNDED_OWNERSHIP_METRIC_OPTIONS = (
    ("holding_shares", "보유주식수"),
    ("holding_ratio", "보유비율"),
)


def validation_clarification_request(
    question: str,
    validation: Any,
) -> ClarificationRequest | None:
    """Candidates already proven by query understanding or validation."""

    if validation.state not in {QueryState.AMBIGUOUS, QueryState.INCOMPLETE}:
        return None
    plan = validation.plan
    if _protected_semantics(question, plan):
        return None

    company = validation.slots.get("company")
    if (
        plan.task_type == "holding_change"
        and company is not None
        and company.status is not QuerySlotStatus.RESOLVED
    ):
        # A multi-company holding question may still be waiting for the
        # corpus-backed issuer/reporter resolver.  Metric clarification must
        # never reinterpret or bypass that authoritative role decision.
        return None

    metric = validation.slots.get("metric")
    ownership_intent = str(
        dict(getattr(plan, "evidence", {}) or {}).get("holding_ownership_intent")
        or ""
    ).strip()
    if (
        plan.task_type == "holding_change"
        and metric is not None
        and metric.status is not QuerySlotStatus.RESOLVED
        and ownership_intent
        and has_role_provenance(plan)
    ):
        candidates = tuple(
            ClarificationCandidate(
                id=f"M{index}",
                label=label,
                semantic_type="holding_metric",
                provenance="query_understanding.holding_ownership_intent",
                value=value,
            )
            for index, (value, label) in enumerate(
                _BOUNDED_OWNERSHIP_METRIC_OPTIONS, start=1
            )
        )
        return ClarificationRequest(
            question=question,
            candidates=candidates,
            reason="holding_metric_ambiguity",
            target_slot="metric",
            classifier_resolution_safe=True,
        )

    option_labels = {
        option.id: option.label
        for option in getattr(validation.clarification, "options", ()) or ()
    }
    for name in validation.ambiguous_slots:
        slot = validation.slots[name]
        values = tuple(dict.fromkeys(str(value) for value in slot.candidates if value))
        if len(values) < 2:
            continue
        labels = tuple(_public_option_label(option_labels, value) for value in values)
        if any(label is None for label in labels):
            # A slot candidate is a machine value -- a corp code, a task enum --
            # unless validation published an explicit label for it.  This layer is
            # additive, so it drops the menu and leaves the validator's own
            # question standing rather than reading the raw value out loud.  It
            # stops here instead of trying a later slot: advancing would change
            # which ambiguity the asker is questioned about.
            return None
        bounded = values[:MAX_CANDIDATES]
        candidates = tuple(
            ClarificationCandidate(
                id=f"C{index}",
                label=label,
                semantic_type=name,
                provenance=f"query_validation.{name}",
                value=value,
            )
            for index, (value, label) in enumerate(zip(bounded, labels), start=1)
        )
        return ClarificationRequest(
            question=question,
            candidates=candidates,
            reason=f"ambiguous_query_slot:{name}",
            target_slot=name,
            truncated=len(values) > len(bounded),
        )
    return None


def execution_clarification_request(
    question: str,
    plan: Any,
    result: Any,
    execution: Any,
    *,
    multi_document: Any = None,
) -> ClarificationRequest | None:
    """Candidates proven by structured corpus or resolver output."""

    if multi_document is not None or _protected_semantics(question, plan):
        return None
    return _event_instance_request(question, plan, result, execution)


def apply_resolved_candidate(
    plan: Any,
    request: ClarificationRequest,
    decision: ClarificationDecision,
) -> Any:
    """Apply only provider-declared safe semantic updates."""

    candidate = decision.selected_candidate
    if (
        decision.state is not ClarificationState.RESOLVED
        or candidate is None
        or request.target_slot != "metric"
        or candidate.semantic_type != "holding_metric"
        or candidate.value
        not in {value for value, _label in _BOUNDED_OWNERSHIP_METRIC_OPTIONS}
    ):
        return plan
    evidence = dict(getattr(plan, "evidence", {}) or {})
    evidence["clarification_resolution"] = {
        "candidate_id": candidate.id,
        "semantic_type": candidate.semantic_type,
        "source": "bounded_clarification_classifier",
    }
    return replace(plan, metric=candidate.value, evidence=evidence)


def _event_instance_request(
    question: str,
    plan: Any,
    result: Any,
    execution: Any,
) -> ClarificationRequest | None:
    if not getattr(plan, "event_type", None):
        return None
    if getattr(result, "resolution", None) is not None:
        return None
    evidence = dict(getattr(plan, "evidence", {}) or {})
    if evidence.get("operation") in {"enumerate", "lifecycle_status"}:
        return None
    if evidence.get("set_intent") is True:
        return None

    trace = getattr(execution, "event_expansion", None)
    trace = dict(trace) if isinstance(trace, Mapping) else {}
    block = trace.get("corporate_event_expansion")
    block = dict(block) if isinstance(block, Mapping) else {}
    raw_events = [event for event in block.get("events") or () if isinstance(event, Mapping)]
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_events:
        event = dict(raw)
        event_id = str(event.get("event_id") or "").strip()
        if not event_id or event_id in seen:
            continue
        seen.add(event_id)
        events.append(event)
    if len(events) < 2:
        return None

    document_metadata = {
        str(document.doc_id): dict(document.metadata)
        for document in (getattr(execution, "documents", ()) or ())
    }
    bounded = events[:MAX_CANDIDATES]
    labels = [_event_label(event, document_metadata) for event in bounded]
    if any(label is None for label in labels):
        return None
    public_labels = tuple(str(label) for label in labels)
    if len({_event_label_key(label) for label in public_labels}) != len(public_labels):
        return None
    candidates = tuple(
        ClarificationCandidate(
            id=f"E{index}",
            label=label,
            semantic_type="event_instance",
            provenance="corporate_event_graph",
            value=str(event["event_id"]),
        )
        for index, (event, label) in enumerate(
            zip(bounded, public_labels), start=1
        )
    )
    return ClarificationRequest(
        question=question,
        candidates=candidates,
        reason="multiple_event_instances",
        target_slot="event_instance",
        classifier_resolution_safe=False,
        truncated=len(events) > len(bounded),
    )


def _event_label(
    event: Mapping[str, Any],
    metadata_by_doc: Mapping[str, Mapping[str, Any]],
) -> str | None:
    metadata: Mapping[str, Any] = {}
    for key in ("seed_member_doc_id", "seed_doc_id"):
        doc_id = str(event.get(key) or "").strip()
        if doc_id and doc_id in metadata_by_doc:
            candidate = metadata_by_doc[doc_id]
            if candidate.get("report_nm") or candidate.get("rcept_dt"):
                metadata = candidate
                break
    report = str(metadata.get("report_nm") or "").strip()
    receipt = _date_text(metadata.get("rcept_dt"))
    if report and receipt:
        return f"{report} ({receipt})"
    if receipt:
        return f"{receipt} 공시"
    if report:
        return report
    return None


def _public_option_label(
    option_labels: Mapping[str, str], value: str
) -> str | None:
    """The explicit public label validation published for a slot candidate.

    A missing or blank label is not permission to show the raw candidate: slot
    candidates carry machine values, so an unlabelled one has no safe rendering.
    """

    label = str(option_labels.get(value) or "").strip()
    return label or None


def _event_label_key(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip().casefold()


def _protected_semantics(question: str, plan: Any) -> bool:
    comparison = getattr(plan, "comparison", None)
    evidence = dict(getattr(plan, "evidence", {}) or {})
    if isinstance(comparison, Mapping) and comparison.get("type") == "company_comparison":
        return True
    if evidence.get("comparison_frame") in {"cross_company", "uncertain"}:
        return True
    report_relative = evidence.get("holding_report_relative")
    if isinstance(report_relative, Mapping) and report_relative.get("selector") == "selected_context":
        return True
    try:
        # Acquisition wording -- including the unit price the resolver cannot
        # answer -- is never re-read as bare shares-vs-ratio ambiguity.
        return has_holding_acquisition_semantics(question, plan)
    except (TypeError, ValueError):
        return False


def _date_text(value: Any) -> str | None:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) < 8:
        return None
    return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"


__all__ = [
    "apply_resolved_candidate",
    "execution_clarification_request",
    "validation_clarification_request",
]
