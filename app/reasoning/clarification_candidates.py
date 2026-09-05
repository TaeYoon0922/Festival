"""Deterministic candidate providers for the clarification layer."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping

from app.reasoning.clarification_request import (
    EVENT_INSTANCE,
    MAX_CANDIDATES,
    ClarificationCandidate,
    ClarificationDecision,
    ClarificationRequest,
    ClarificationState,
)
from app.reasoning.corporate_event import LIFECYCLE_OPEN, RESOLVED
from app.reasoning.holding_company_role_resolution import has_role_provenance
from app.reasoning.holding_evidence_coverage import has_holding_acquisition_semantics
from app.reasoning.holding_report_clarification import (
    holding_report_clarification_request,
)
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
    report_index: Any = None,
    answerable: bool = True,
) -> ClarificationRequest | None:
    """Candidates proven by structured corpus or resolver output."""

    if multi_document is not None or _protected_semantics(question, plan):
        return None
    request = _event_instance_request(question, plan, result, execution)
    if request is not None:
        return request
    # Asked second so nothing that already had an answer is turned into a
    # question: this provider only speaks for a holding question that named a
    # holder, did not name one of that holder's filings, and got nothing back.
    return holding_report_clarification_request(
        question, plan, report_index=report_index, answerable=answerable
    )


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
    if _explicit_event_selector(plan):
        return None

    trace = getattr(execution, "event_expansion", None)
    trace = dict(trace) if isinstance(trace, Mapping) else {}
    block = trace.get("corporate_event_expansion")
    block = dict(block) if isinstance(block, Mapping) else {}
    raw_events = [
        event for event in block.get("events") or () if isinstance(event, Mapping)
    ]
    raw_events.extend(_settled_singleton_events(block))
    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_events:
        event = dict(raw)
        event_id = str(event.get("event_id") or "").strip()
        if not event_id or event_id in seen:
            continue
        seen.add(event_id)
        events.append(event)
    events = _matching_contract_instances(question, plan, events)
    if len(events) < 2:
        return None

    document_metadata = {
        str(document.doc_id): dict(document.metadata)
        for document in (getattr(execution, "documents", ()) or ())
    }
    bounded = events[:MAX_CANDIDATES]
    seeds = [_event_seed_doc_id(event, document_metadata) for event in bounded]
    labels = [
        _event_label(document_metadata.get(seed) or {}) if seed else None
        for seed in seeds
    ]
    if any(label is None for label in labels):
        return None
    public_labels = tuple(str(label) for label in labels)
    if len({_event_label_key(label) for label in public_labels}) != len(public_labels):
        return None
    sources = [_seed_source(seed, execution) for seed in seeds]
    candidates = tuple(
        ClarificationCandidate(
            id=f"E{index}",
            label=label,
            semantic_type=EVENT_INSTANCE,
            provenance="corporate_event_graph",
            value=str(event["event_id"]),
            source_doc_id=source[1] if source else None,
            source_chunk_id=source[0] if source else None,
        )
        for index, (event, label, source) in enumerate(
            zip(bounded, public_labels, sources), start=1
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


def _event_seed_doc_id(
    event: Mapping[str, Any],
    metadata_by_doc: Mapping[str, Mapping[str, Any]],
) -> str | None:
    """The one filing this graph root is named and cited by.

    A correction chain has already become a single identity by the time it gets
    here, so no lifecycle is ever offered twice.  Which of its filings *names*
    that identity is the question this answers, and the answer is the root:
    ``seed_root_doc_id`` is the filing the contract began from, which the graph
    proves and which no later filing can move.  ``seed_member_doc_id`` is the
    member the chain collapsed to -- the right filing to quote for what the
    contract now says, and the wrong one to tell two contracts apart, because a
    correction is received long after the contract it corrects and can land on
    the day some unrelated contract was concluded.

    A root that differs from that collapsed member is the whole of this
    candidate's identity, and nothing may stand in for it: an unserved root
    leaves the candidate unnamed, because showing the correction as the
    contract's own filing would misdate the contract and miscite it in one
    move.  ``seed_doc_id`` is not consulted for that judgement -- it is
    whichever member retrieval happened to seed the lookup with, so it says
    nothing about which filing opened the contract -- and stays what it has
    always been here, the last fallback for a trace carrying no root at all.

    Label and provenance both read whatever this returns, which is what keeps
    the marker beside a label pointing at the filing that label describes.
    """

    root = str(event.get("seed_root_doc_id") or "").strip()
    member = str(event.get("seed_member_doc_id") or "").strip()
    if root and root != member:
        return root if _names_a_filing(metadata_by_doc, root) else None
    for doc_id in (root, member, str(event.get("seed_doc_id") or "").strip()):
        if _names_a_filing(metadata_by_doc, doc_id):
            return doc_id
    return None


def _names_a_filing(
    metadata_by_doc: Mapping[str, Mapping[str, Any]], doc_id: str
) -> bool:
    """Whether retrieval carries enough of this filing to name it publicly."""

    metadata = metadata_by_doc.get(doc_id) if doc_id else None
    if not metadata:
        return False
    return bool(metadata.get("report_nm") or metadata.get("rcept_dt"))


def _seed_source(doc_id: str | None, execution: Any) -> tuple[str, str] | None:
    """The served chunk that evidences one seed filing, if there is one.

    The filing is chosen by the graph, never by rank; rank only orders the
    chunks *inside* that already-chosen filing, so the best-ranked one is the
    row the public context would have shown for it anyway.  A filing retrieval
    never served has no source, and the caller must not invent one.
    """

    if not doc_id:
        return None
    served = {
        str(candidate.chunk_id)
        for candidate in (getattr(execution, "chunks", ()) or ())
        if str(candidate.doc_id) == doc_id
    }
    ranked = sorted(
        (
            (int(result.rank), str(result.chunk_id))
            for result in (getattr(execution, "results", ()) or ())
            if str(result.doc_id) == doc_id and str(result.chunk_id) in served
        ),
    )
    if not ranked:
        return None
    return (ranked[0][1], doc_id)


def _event_label(metadata: Mapping[str, Any]) -> str | None:
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


def _settled_singleton_events(block: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Graph-proven single-filing contracts expansion correctly had nothing to add.

    Expansion records a one-member lifecycle as ``lifecycle_not_resolved``
    because it found no counterpart filing to pull in.  For a contract that was
    concluded and never followed up there is no counterpart to find, and
    :attr:`app.reasoning.corporate_event.CorporateEventState.has_dangling_reference`
    already states the rule this reads by: such a filing is stored as
    ``unresolved`` "because there was nothing to link it to -- that is complete
    evidence, not a gap", and only ``resolution_source`` marks the real gap,
    where a lifecycle names a filing the corpus does not hold.

    So a still-open singleton is as settled as a resolved one, and two of them
    that name the same counterparty are two contracts, not one unresolved
    lifecycle.  A terminated or ambiguous singleton is neither, and a dangling
    reference is the gap itself; none of those is offered as something to
    choose between.
    """

    return [
        item
        for item in (block.get("skipped") or ())
        if isinstance(item, Mapping)
        and item.get("reason") == "lifecycle_not_resolved"
        and _event_count(item.get("member_count")) == 1
        and str(item.get("event_id") or "").strip()
        and _is_settled_lifecycle(item)
    ]


def _is_settled_lifecycle(item: Mapping[str, Any]) -> bool:
    """Whether one skipped singleton states its lifecycle completely."""

    source = _event_text(item.get("resolution_source"))
    if source.endswith("_not_in_corpus"):
        return False
    if _event_text(item.get("resolution_status")) == RESOLVED:
        return True
    return _event_text(item.get("lifecycle_status")) == LIFECYCLE_OPEN


def _explicit_event_selector(plan: Any) -> bool:
    """Whether the question already identifies one contract instance."""

    evidence = dict(getattr(plan, "evidence", {}) or {})
    if any(evidence.get(key) for key in ("event_id", "contract_id")):
        return True
    period = getattr(plan, "period", None)
    if hasattr(period, "to_dict"):
        period = period.to_dict()
    values = dict(period) if isinstance(period, Mapping) else {}
    start = values.get("from") or values.get("from_date")
    end = values.get("to") or values.get("to_date")
    return bool(start and start == end)


def _matching_contract_instances(
    question: str,
    plan: Any,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the graph roots matching the question's proven event identity.

    Event ids have already collapsed original/correction members.  This scope
    then applies the remaining semantic rule: same issuer, same broad contract
    family, and -- when the graph carried it -- the same normalized
    counterparty named by the question.  Missing legacy trace fields preserve
    the previous behavior; conflicting structured identity fails closed.
    """

    corp_code = str(getattr(plan, "corp_code", None) or "").strip()
    family = str(getattr(plan, "event_type", None) or "").strip()
    scoped = [
        event
        for event in events
        if (
            not corp_code
            or not str(event.get("seed_corp_code") or "").strip()
            or str(event.get("seed_corp_code") or "").strip() == corp_code
        )
        and (
            not family
            or not _event_text(event.get("event_family"))
            or _event_text(event.get("event_family")) == family
        )
    ]
    if len(scoped) < 2:
        return scoped

    identities = {
        str(event.get("event_id")): dict(event.get("seed_identity") or {})
        for event in scoped
        if isinstance(event.get("seed_identity"), Mapping)
    }
    counterparties = {
        event_id: _identity_key(identity.get("counterparty"))
        for event_id, identity in identities.items()
        if _identity_key(identity.get("counterparty"))
    }
    if counterparties:
        query_key = _identity_key(question)
        named = {
            value for value in counterparties.values() if value in query_key
        }
        if len(named) == 1:
            wanted = next(iter(named))
            scoped = [
                event
                for event in scoped
                if counterparties.get(str(event.get("event_id"))) == wanted
            ]
        elif len(set(counterparties.values())) > 1:
            # Several counterparties were proved and the question selected
            # none of them.  This is not the same-counterparty ambiguity this
            # provider owns.
            return []

    if len(scoped) < 2:
        return scoped

    subjects = {
        event_id: _identity_key(identity.get("subject"))
        for event_id, identity in identities.items()
        if _identity_key(identity.get("subject"))
    }
    named_subjects = {
        value
        for value in subjects.values()
        if value and value in _identity_key(question)
    }
    if len(named_subjects) == 1:
        wanted = next(iter(named_subjects))
        scoped = [
            event
            for event in scoped
            if subjects.get(str(event.get("event_id"))) == wanted
        ]
    return scoped


def _identity_key(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _event_text(value: Any) -> str:
    return str(getattr(value, "value", value) or "").strip()


def _event_count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
