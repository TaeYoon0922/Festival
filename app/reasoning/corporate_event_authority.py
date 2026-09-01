"""Which corporate-event filing a question's evidence is authorised to answer from.

STEP 11-C.2.  P0-B already answers every identity question a field producer
needs: :class:`~app.reasoning.corporate_event.CorporateEventState` names the
event a filing belongs to, the role it plays in that lifecycle, and -- through
``canonical_doc_id`` -- which filing of a correction chain is the one that
stands.  That state was computed during retrieval and then flattened away, so
the first attempt at a field producer re-derived member identity from company
plus receipt date and correction finality from a group count.  Both were weaker
than the authority that already existed, and both were wrong.

This module reads the carried-through state and answers one question: is there
exactly one filing this request may be answered from, and which is it.  It
computes no identity of its own.  Every rule below is a direct reading of a
P0-B/P0-A field:

``canonical_doc_id``
    The filing that stands.  For a resolved correction chain P0-A guarantees it
    is the chain's last member, so ``ROOT`` and every ``MIDDLE`` link report a
    canonical document that is not themselves -- which is what makes a middle
    link recognisable without reconstructing the chain here.

``correction_resolution_status``
    ``None`` means P0-A was consulted and this filing belongs to no correction
    group.  That is a positive finding, not missing metadata.  ``ambiguous`` or
    ``unresolved`` means finality was not established, and nothing here may
    stand in for it.

``member_role``
    A termination or a contract change is a different filing from the contract
    it acts on, so it can never supply the contract's own fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.reasoning.corporate_event import (
    AMBIGUOUS,
    ROLE_CONTRACT,
    UNRESOLVED,
)


#: Where the retrieval seam leaves the carried P0-B state.
_STATE_KEY = "event_member_states"


@dataclass(frozen=True)
class SelectedCorporateMember:
    """The one filing a corporate-event field request may be answered from.

    ``authoritative_doc_id`` is what to read the field from; ``served_doc_ids``
    are the filings of the same lifecycle that were served and are *not* it, so
    a producer can record why they were ruled out without ever reading a value
    from them.
    """

    event_id: str
    corp_code: str
    authoritative_doc_id: str
    member_role: str
    superseded_doc_ids: frozenset[str]


@dataclass(frozen=True)
class CorporateSelectionIntent:
    """The one query-selection distinction field authority needs.

    P0-A still owns correction finality.  This flag only says that retrieval
    selected one filing by an exact receipt date and the question did not ask
    for a corrected/latest view, so the served mapping key is the filing whose
    field must be read.
    """

    exact_historical_receipt_date: bool = False


@dataclass(frozen=True)
class CorporateAuthority:
    """What P0-B could say about this request.

    ``member`` is set only when exactly one filing is authorised.  ``conflict``
    is set when P0-B identified competing candidates and nothing it carries
    chooses between them -- which is a finding, not an absence.  Both unset
    means P0-B has nothing to say and the request is not this lane's.
    """

    member: SelectedCorporateMember | None = None
    conflict: bool = False
    reason: str = "no_event_authority"


_DECLINED = CorporateAuthority()


def selected_corporate_member(
    *,
    execution: Any,
    corp_code: str | None,
    served_doc_ids: Sequence[str],
    candidate_doc_ids: Sequence[str] = (),
    selection_intent: CorporateSelectionIntent | None = None,
) -> CorporateAuthority:
    """Read P0-B's verdict for the filings this answer was served.

    Fails closed in every direction it cannot read: no carried state, no
    contract member among the served filings, an incomplete authority set, more
    than one lifecycle, or a correction whose finality P0-A did not establish.

    ``candidate_doc_ids`` are the served filings that could themselves answer
    the requested field -- the ones this selection is actually choosing between.
    They matter because the carried state covers the retrieval seeds, not the
    served set, and those two can differ.  A filing P0-B never spoke about is
    not thereby irrelevant, so uniqueness is only asserted once every candidate
    is accounted for.
    """

    states = _carried_states(execution)
    if not states:
        # Either the lifecycle graph never ran for this question or it is not a
        # contract-event question at all.  Absence is not authority, so this
        # lane does not claim the request.
        return _DECLINED

    served = {str(doc_id) for doc_id in served_doc_ids if str(doc_id)}
    wanted_corp = str(corp_code or "").strip()
    candidates = [
        (doc_id, state)
        for doc_id, state in states.items()
        if doc_id in served
        and str(state.get("member_role") or "") == ROLE_CONTRACT
        and (not wanted_corp or str(state.get("corp_code") or "") == wanted_corp)
    ]
    if not candidates:
        return _DECLINED

    unaccounted = _unaccounted_candidates(
        execution, states, candidate_doc_ids, [state for _doc_id, state in candidates]
    )
    if unaccounted:
        # A served filing that could answer this field, which P0-B did not
        # describe and no expansion claimed for the lifecycle already in hand.
        # It may be another contract entirely; selecting among the filings that
        # happen to carry state would be answering from whichever candidate
        # retrieval's seed window kept.
        return CorporateAuthority(conflict=True, reason="incomplete_corporate_authority")

    event_ids = {str(state.get("event_id") or "") for _doc_id, state in candidates}
    if len(event_ids) != 1:
        # Two contracts, and P0-B places them on different lifecycles.  Which
        # one the question meant is not a choice this may make.
        return CorporateAuthority(conflict=True, reason="multiple_corporate_events")

    canonical = {
        str(state.get("canonical_doc_id") or "") for _doc_id, state in candidates
    }
    canonical.discard("")
    if len(canonical) != 1:
        return CorporateAuthority(conflict=True, reason="multiple_canonical_members")

    if any(
        str(state.get("correction_resolution_status") or "") in {AMBIGUOUS, UNRESOLVED}
        for _doc_id, state in candidates
    ):
        # P0-A saw a correction group and could not say which filing is final.
        # Reading either end would be inventing the finality it declined to
        # assert.
        return CorporateAuthority(conflict=True, reason="correction_finality_unresolved")

    historical = bool(
        selection_intent and selection_intent.exact_historical_receipt_date
    )
    if historical:
        field_bearing = {
            str(value) for value in candidate_doc_ids if str(value)
        }
        field_candidates = {
            doc_id
            for doc_id, _state in candidates
            if doc_id in field_bearing
        }
        if len(field_candidates) != 1:
            return CorporateAuthority(
                conflict=True, reason="historical_field_authority_ambiguous"
            )
        # The repository mapping key is the asked document identity.  Its state
        # value may deliberately describe the chain's canonical/latest member.
        authoritative = next(iter(field_candidates))
        superseded = frozenset(
            doc_id for doc_id, _state in candidates if doc_id != authoritative
        )
    else:
        authoritative = next(iter(canonical))
        superseded = frozenset(
            str(state.get("doc_id") or "")
            for _doc_id, state in candidates
            if str(state.get("doc_id") or "") != authoritative
        )
    first = candidates[0][1]
    return CorporateAuthority(
        member=SelectedCorporateMember(
            event_id=next(iter(event_ids)),
            corp_code=str(first.get("corp_code") or ""),
            authoritative_doc_id=authoritative,
            member_role=ROLE_CONTRACT,
            superseded_doc_ids=superseded,
        ),
        reason="selected_corporate_member",
    )


def _unaccounted_candidates(
    execution: Any,
    states: Mapping[str, Mapping[str, Any]],
    candidate_doc_ids: Sequence[str],
    candidate_states: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    """Candidate filings no upstream statement covers.

    A candidate is accounted for when P0-B described it, or when an expansion
    pulled it into a lifecycle or chain one of these candidates is itself on --
    both existing upstream statements about the filing, read rather than
    inferred.  Anything else is a filing that could compete and about which
    nothing is known.

    Identity is the document, so one filing served as several chunks is one
    candidate and can never look like missing authority.
    """

    wanted = {str(doc_id) for doc_id in candidate_doc_ids if str(doc_id)}
    if not wanted:
        return frozenset()
    explained = _explained_expansion_docs(execution, candidate_states)
    return frozenset(wanted - set(states) - explained)


def _explained_expansion_docs(
    execution: Any, candidate_states: Sequence[Mapping[str, Any]]
) -> set[str]:
    """Filings an expansion added to a lifecycle these candidates are on.

    Scoped to the identities the candidates actually carry.  One request can
    expand several lifecycles at once -- a contract seed and an unrelated
    termination seed each pull in their own members -- so "an expansion added
    it" says nothing on its own.  A filing added for *another* event is exactly
    the competing contract this check exists to notice, and exempting it would
    reopen the defect the check was written to close.

    Explained here means only "this is not a second unexplained event".  It
    grants no authority: whether a filing may supply the field is still decided
    by its own state, role, canonical document and served evidence.
    """

    explained: set[str] = set()
    event_ids = {
        value
        for state in candidate_states
        if (value := str(state.get("event_id") or ""))
    }
    explained |= _event_scoped_targets(execution, event_ids)
    group_ids = {
        value
        for state in candidate_states
        if (value := str(state.get("correction_group_id") or ""))
    }
    explained |= _correction_scoped_targets(execution, group_ids)
    return explained


def _event_scoped_targets(execution: Any, event_ids: set[str]) -> set[str]:
    """Documents event expansion added *for these events*, and no others.

    The expansion trace records its targets per event, so this is a lookup of
    what upstream already said rather than an inference about what a document
    is.
    """

    if not event_ids:
        return set()
    trace = getattr(execution, "event_expansion", None)
    if not isinstance(trace, Mapping):
        return set()
    block = trace.get("corporate_event_expansion")
    if not isinstance(block, Mapping):
        return set()
    targets: set[str] = set()
    for entry in block.get("events") or ():
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("event_id") or "") not in event_ids:
            continue
        for doc_id in entry.get("target_doc_ids") or ():
            if str(doc_id):
                targets.add(str(doc_id))
    return targets


def _correction_scoped_targets(execution: Any, group_ids: set[str]) -> set[str]:
    """Documents correction expansion added *for these chains*, and no others.

    The correction trace does not map its added filings back to the chain each
    came from; it reports them as one list.  That list is only attributable
    when exactly one chain expanded -- then every added filing belongs to it,
    and the trace names which chain that was.  Anything else stays unexplained,
    because a filing from some other chain is another contract, and hiding it
    is the failure this scoping exists to prevent.
    """

    if not group_ids:
        return set()
    trace = getattr(execution, "correction_expansion", None)
    if not isinstance(trace, Mapping):
        return set()
    if int(trace.get("correction_group_count") or 0) != 1:
        return set()
    if str(trace.get("correction_group_id") or "") not in group_ids:
        return set()
    return {
        str(doc_id)
        for doc_id in trace.get("correction_added_doc_ids") or ()
        if str(doc_id)
    }


def _carried_states(execution: Any) -> dict[str, Mapping[str, Any]]:
    trace = getattr(execution, "event_expansion", None)
    if not isinstance(trace, Mapping):
        return {}
    states = trace.get(_STATE_KEY)
    if not isinstance(states, Mapping):
        return {}
    return {
        str(doc_id): state
        for doc_id, state in states.items()
        if isinstance(state, Mapping)
    }


__all__ = [
    "CorporateAuthority",
    "CorporateSelectionIntent",
    "SelectedCorporateMember",
    "selected_corporate_member",
]
