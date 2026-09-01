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


def _chain_endpoints(trace: Any) -> tuple[str, str, str] | None:
    """The root and final document of the one chain this execution expanded.

    More than one chain leaves every filing unattributable: the trace reports
    its documents as a single list without saying which chain each came from.
    So the discipline ``corporate_event_authority`` already applies is applied
    here too -- exactly one group, or nothing.
    """

    if not isinstance(trace, Mapping):
        return None
    if int(trace.get("correction_group_count") or 0) != 1:
        return None
    group_id = str(trace.get("correction_group_id") or "")
    root = str(trace.get("correction_root_doc_id") or "")
    latest = str(trace.get("correction_latest_doc_id") or "")
    if not group_id or not root or not latest or root == latest:
        # A chain whose root is its own final version has no two versions to
        # compare, and an unnamed endpoint cannot be attributed to a filing.
        return None
    return root, latest, group_id


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


def bind_correction_pair(
    resolution: HoldingResolution,
    *,
    correction_trace: Any,
    query_plan: Any,
) -> CorrectionPairClaim | None:
    """Read the two versions a pair question asks about, or decline.

    Declining is the frozen behaviour: the caller keeps the resolution it
    already had, warnings and all.  Nothing here guesses a pair into existence.
    """

    if correction_intent(query_plan) != PAIR_INTENT:
        return None
    fields = tuple(resolution.requested_fields)
    if not fields:
        # A history question naming no metric wants the timeline, which the
        # composer already reports in full.
        return None
    endpoints = _chain_endpoints(correction_trace)
    if endpoints is None:
        return None
    root_doc_id, latest_doc_id, group_id = endpoints

    matching = [event for event in resolution.events if event.matches_query is True]
    if len(matching) != 2:
        # One version answers alone; three or more is a timeline, not a pair.
        return None
    before = _event_for_document(matching, root_doc_id)
    after = _event_for_document(matching, latest_doc_id)
    if before is None or after is None or before is after:
        return None
    if not _same_subject(before, after):
        return None
    if not _states_every_field(before, fields) or not _states_every_field(after, fields):
        # Half a pair is not a pair, and the missing half must not be guessed.
        return None
    return CorrectionPairClaim(
        fields=fields,
        before_doc_id=root_doc_id,
        after_doc_id=latest_doc_id,
        before_event=before,
        after_event=after,
        correction_group_id=group_id,
        reference_date=before.reference_date or after.reference_date,
        reporter=before.reporter or after.reporter,
        corp_name=before.corp_name or after.corp_name,
    )


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


def pair_trace(claim: CorrectionPairClaim | None) -> dict[str, Any]:
    """What the execution record says about role binding, bound or not."""

    if claim is None:
        return {"correction_pair_bound": False}
    return {"correction_pair_bound": True, **copy.deepcopy(claim.to_dict())}
