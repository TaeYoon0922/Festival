"""Bind a correction chain's two filings to the roles a pair question asks for.

A question like "정정 전과 정정 후 보유주식수가 각각 얼마야?" is answered by one
holding event that two filings state differently: the report as originally filed
and the correction that superseded it.  Retrieval and the resolver already serve
both -- ``_same_holding_event`` groups by ``doc_id``, so one filing becomes one
:class:`HoldingEvent` -- but nothing downstream says which of the two is the
*before* and which is the *after*.  Both carry the same reference date and the
same reporter, so they read as two indistinguishable events and the answer lists
the same field twice under the same label.

Two axes meet here and must not be confused:

``event field``
    Where a value sits inside one filing's own change.  ``before_shares`` is the
    holding before the transaction that filing reports.

``document version``
    Which filing states the value: the original, or the correction.

"정정 전" is the *document-version* before, not the event-field before.  For a
question asking the holding after the change, both roles read the same field --
``after_shares`` -- from two different filings.  Reading ``before_shares`` for
"정정 전" would answer with the holding before the transaction, which is neither
what was corrected nor what was asked.

Roles are never inferred from receipt order, ranking, or which filing arrived
first.  They come from the correction graph, through the expansion trace that
already names the chain's ``root_doc_id`` and ``latest_doc_id``.  When the graph
cannot name them, or when the two events fail any structural check, no pair is
bound and the existing multi-event behaviour is left exactly as it was.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.reasoning.holding_event_resolver import HoldingEvent, HoldingResolution
from app.reasoning.holding_reporter import reporter_matches


#: The document-version roles.  ``before`` is the report as first filed and
#: ``after`` is the final valid version of the same report.
ROLE_BEFORE = "correction_before"
ROLE_AFTER = "correction_after"

#: Only a history question asks for both versions at once.  ``latest`` wants the
#: final version alone and ``original`` wants the first, so neither binds a pair.
PAIR_INTENT = "history"

#: How a document-version role is named in an answer.  Defined here, beside the
#: roles themselves, because two renderers state them: the compact claim the
#: verbalizer is given, and the deterministic multi-event rows that are served
#: whenever the verbalizer skips.  One table, so the two cannot drift apart.
CORRECTION_ROLE_LABELS: dict[str, str] = {
    ROLE_BEFORE: "정정 전",
    ROLE_AFTER: "정정 후",
}

#: What a field is called once a correction role owns the before/after axis.
#: The frozen labels describe a position inside one filing's own change ("변동
#: 후 주식수"); a correction role describes which *version of the filing* states
#: the value.  Naming both on one line would read as one axis, so a field that
#: states the holding as of the filing is named by what it measures instead.
#: Any other field keeps its frozen label and only gains the role prefix.
ROLE_FIELD_LABELS: dict[str, str] = {
    "after_shares": "보유주식수",
    "after_ratio": "보유비율",
}

#: Why a pair was not bound.  Deterministic, and diagnostic only: nothing reads
#: these to decide anything, they are written to the execution trace so a
#: decline is visible instead of silent.
REASON_BOUND = "bound"
REASON_INTENT_NOT_HISTORY = "intent_not_history"
REASON_NO_REQUESTED_FIELDS = "no_requested_fields"
REASON_NO_MATCHING_GROUP = "no_matching_group"
REASON_MULTIPLE_MATCHING_GROUPS = "multiple_matching_groups"
REASON_EVENT_COUNT_MISMATCH = "event_count_mismatch"
REASON_ENDPOINT_EVENT_MISSING = "endpoint_event_missing"
REASON_SUBJECT_MISMATCH = "subject_mismatch"
REASON_FIELD_MISSING = "field_missing"
REASON_FIELD_CONFLICT = "field_conflict"


@dataclass(frozen=True)
class CorrectionPairClaim:
    """One holding event as two filings state it, with each role attributed.

    ``fields`` holds the requested metric names unchanged from the resolution:
    the same field is read from both versions, because the roles differ by
    document, not by position inside the event.
    """

    fields: tuple[str, ...]
    before_doc_id: str
    after_doc_id: str
    before_event: HoldingEvent
    after_event: HoldingEvent
    correction_group_id: str
    reference_date: str | None
    reporter: str | None
    corp_name: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": list(self.fields),
            "before_doc_id": self.before_doc_id,
            "after_doc_id": self.after_doc_id,
            "correction_group_id": self.correction_group_id,
            "reference_date": self.reference_date,
            "reporter": self.reporter,
            "corp_name": self.corp_name,
        }


def correction_intent(query_plan: Any) -> str | None:
    """The intent T2-A wrote onto the plan, read without assuming a plan type."""

    evidence = getattr(query_plan, "evidence", None)
    if not isinstance(evidence, Mapping) and isinstance(query_plan, Mapping):
        evidence = query_plan.get("evidence")
    if not isinstance(evidence, Mapping):
        return None
    value = evidence.get("correction_intent")
    return str(value) if value else None


def _chain_endpoints(trace: Any) -> list[tuple[str, str, str]]:
    """Every chain this execution expanded, as ``(root, latest, group_id)``.

    One execution can expand several chains: a reporter's filings for two
    different reference events are corrected independently, and a question about
    one of them retrieves both.  The trace carries each chain's own endpoints in
    ``correction_groups``; the flat ``correction_root_doc_id`` keys describe only
    the first group and are read as a single-group fallback, so a trace written
    before those per-group entries existed still resolves.

    Which of several chains the question is about is not decided here -- see
    ``_qualifying_pairs``, which asks which chain's endpoints are the served
    events, rather than picking one by order, recency, or rank.
    """

    if not isinstance(trace, Mapping):
        return []
    groups = trace.get("correction_groups")
    entries: list[Mapping[str, Any]] = []
    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes)):
        entries = [group for group in groups if isinstance(group, Mapping)]
    if not entries:
        if int(trace.get("correction_group_count") or 0) != 1:
            # Several chains, and the trace names only one chain's endpoints:
            # the rest are unattributable, so nothing here can be trusted.
            return []
        entries = [
            {
                "correction_group_id": trace.get("correction_group_id"),
                "root_doc_id": trace.get("correction_root_doc_id"),
                "latest_doc_id": trace.get("correction_latest_doc_id"),
            }
        ]
    endpoints: list[tuple[str, str, str]] = []
    for entry in entries:
        group_id = str(entry.get("correction_group_id") or "")
        root = str(entry.get("root_doc_id") or "")
        latest = str(entry.get("latest_doc_id") or "")
        if not group_id or not root or not latest or root == latest:
            # A chain whose root is its own final version has no two versions to
            # compare, and an unnamed endpoint cannot be attributed to a filing.
            continue
        if (root, latest, group_id) not in endpoints:
            endpoints.append((root, latest, group_id))
    return endpoints


def _event_for_document(
    events: Sequence[HoldingEvent], doc_id: str
) -> HoldingEvent | None:
    """The one matching event this filing produced, or nothing.

    Two events from one filing would leave the role ambiguous, so the match has
    to be unique as well as present.
    """

    found = [event for event in events if doc_id in event.doc_ids]
    return found[0] if len(found) == 1 else None


def _same_subject(before: HoldingEvent, after: HoldingEvent) -> bool:
    """Whether the two filings describe the same holding of the same issuer.

    Each check compares only what both sides state.  A field neither filing
    carries cannot separate them, but a field they disagree on always does.
    """

    for left, right in (
        (before.corp_code, after.corp_code),
        (before.company_id, after.company_id),
        (before.reference_date, after.reference_date),
    ):
        if left and right and str(left) != str(right):
            return False
    if before.reporter and after.reporter:
        # The resolver's own reporter contract, so the two cannot disagree.
        if not reporter_matches(before.reporter, after.reporter):
            return False
    return True


def _states_every_field(event: HoldingEvent, fields: Sequence[str]) -> bool:
    """Whether this filing states each requested metric, without conflict."""

    for field in fields:
        if getattr(event, field, None) is None:
            return False
        provenance = event.field_provenance.get(field)
        if provenance is None or provenance.field_conflict:
            return False
    return True


@dataclass(frozen=True)
class CorrectionPairDecision:
    """Whether a pair was bound, and the deterministic reason when it was not."""

    claim: CorrectionPairClaim | None
    reason: str

    @property
    def bound(self) -> bool:
        return self.claim is not None


def _qualifying_pairs(
    resolution: HoldingResolution,
    endpoints: Sequence[tuple[str, str, str]],
    fields: Sequence[str],
) -> tuple[list[CorrectionPairClaim], str]:
    """Every chain whose two endpoints are the served events of this question.

    A chain qualifies only when the events it names are the ones the resolver
    matched, so the question own constraints -- which is what ``matches_query``
    records -- decide which chain is in scope.  A chain for another reference
    event fails because its filings are not what the question matched, not
    because anything here reads its date or its rank.

    Several qualifying chains means the question singled none of them out.  The
    caller declines then: choosing between them would need a preference this
    module refuses to invent.
    """

    matching = [event for event in resolution.events if event.matches_query is True]
    if len(matching) < 2:
        # A pair needs two matched filings; one version answers alone.
        return [], REASON_EVENT_COUNT_MISMATCH

    claims: list[CorrectionPairClaim] = []
    reason = REASON_NO_MATCHING_GROUP
    for root_doc_id, latest_doc_id, group_id in endpoints:
        before = _event_for_document(matching, root_doc_id)
        after = _event_for_document(matching, latest_doc_id)
        if before is None or after is None or before is after:
            reason = _narrow(reason, REASON_ENDPOINT_EVENT_MISSING)
            continue
        if not _same_subject(before, after):
            reason = _narrow(reason, REASON_SUBJECT_MISMATCH)
            continue
        missing = _missing_field_reason(before, fields) or _missing_field_reason(
            after, fields
        )
        if missing is not None:
            # Half a pair is not a pair, and the missing half is never guessed.
            reason = _narrow(reason, missing)
            continue
        claims.append(
            CorrectionPairClaim(
                fields=tuple(fields),
                before_doc_id=root_doc_id,
                after_doc_id=latest_doc_id,
                before_event=before,
                after_event=after,
                correction_group_id=group_id,
                reference_date=before.reference_date or after.reference_date,
                reporter=before.reporter or after.reporter,
                corp_name=before.corp_name or after.corp_name,
            )
        )
    return claims, reason


#: Least to most specific, so the chain that got furthest explains the decline.
_REASON_ORDER = (
    REASON_NO_MATCHING_GROUP,
    REASON_ENDPOINT_EVENT_MISSING,
    REASON_SUBJECT_MISMATCH,
    REASON_FIELD_MISSING,
    REASON_FIELD_CONFLICT,
)


def _narrow(current: str, candidate: str) -> str:
    order = {reason: index for index, reason in enumerate(_REASON_ORDER)}
    return candidate if order.get(candidate, 0) > order.get(current, 0) else current


def _missing_field_reason(event: HoldingEvent, fields: Sequence[str]) -> str | None:
    for field in fields:
        if getattr(event, field, None) is None:
            return REASON_FIELD_MISSING
        provenance = event.field_provenance.get(field)
        if provenance is None:
            return REASON_FIELD_MISSING
        if provenance.field_conflict:
            return REASON_FIELD_CONFLICT
    return None


def decide_correction_pair(
    resolution: HoldingResolution,
    *,
    correction_trace: Any,
    query_plan: Any,
) -> CorrectionPairDecision:
    """Read the two versions a pair question asks about, or decline with a reason.

    Declining is the frozen behaviour: the caller keeps the resolution it
    already had, warnings and all.  Nothing here guesses a pair into existence,
    and the reason is diagnostic only -- no caller branches on it.
    """

    if correction_intent(query_plan) != PAIR_INTENT:
        return CorrectionPairDecision(None, REASON_INTENT_NOT_HISTORY)
    fields = tuple(resolution.requested_fields)
    if not fields:
        # A history question naming no metric wants the timeline, which the
        # composer already reports in full.
        return CorrectionPairDecision(None, REASON_NO_REQUESTED_FIELDS)
    endpoints = _chain_endpoints(correction_trace)
    if not endpoints:
        return CorrectionPairDecision(None, REASON_NO_MATCHING_GROUP)

    claims, reason = _qualifying_pairs(resolution, endpoints, fields)
    if len(claims) == 1:
        return CorrectionPairDecision(claims[0], REASON_BOUND)
    if len(claims) > 1:
        return CorrectionPairDecision(None, REASON_MULTIPLE_MATCHING_GROUPS)
    return CorrectionPairDecision(None, reason)


def bind_correction_pair(
    resolution: HoldingResolution,
    *,
    correction_trace: Any,
    query_plan: Any,
) -> CorrectionPairClaim | None:
    """The bound pair alone, for callers that do not report the decline."""

    return decide_correction_pair(
        resolution, correction_trace=correction_trace, query_plan=query_plan
    ).claim


def apply_correction_pair(
    resolution: HoldingResolution, claim: CorrectionPairClaim
) -> HoldingResolution:
    """Tag the paired events with their roles and order them before, then after.

    Only the two paired events change, and only by gaining a role.  Every other
    event keeps its identity and its place, and no value is recomputed: the
    numbers each filing stated are the numbers each role reports.
    """

    roles = {id(claim.before_event): ROLE_BEFORE, id(claim.after_event): ROLE_AFTER}
    tagged = [
        replace(event, correction_role=roles[id(event)])
        if id(event) in roles
        else event
        for event in resolution.events
    ]
    rank = {ROLE_BEFORE: 0, ROLE_AFTER: 1}
    # A stable sort on the role alone: the pair reads before-then-after, and
    # every unpaired event keeps the order the resolver produced.
    ordered = sorted(
        tagged,
        key=lambda event: rank.get(event.correction_role or "", 0),
    )
    return replace(resolution, events=tuple(ordered))


def pair_trace(
    decision: CorrectionPairDecision | CorrectionPairClaim | None,
) -> dict[str, Any]:
    """What the execution record says about role binding, bound or not.

    Diagnostic only.  It is written beside the existing trace fields and never
    replaces one, so a reader that does not know about it sees what it always
    saw.
    """

    if isinstance(decision, CorrectionPairDecision):
        claim, reason = decision.claim, decision.reason
    else:
        claim = decision
        reason = REASON_BOUND if claim is not None else REASON_NO_MATCHING_GROUP
    if claim is None:
        return {"bound": False, "reason": reason}
    return {"bound": True, "reason": reason, **copy.deepcopy(claim.to_dict())}
