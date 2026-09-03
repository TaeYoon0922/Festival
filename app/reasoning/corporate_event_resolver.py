"""Read-only traversal of the corporate event timeline.

This is the stable surface every caller above the graph uses -- retrieval
expansion today, the multi-document planner later.  It works against either
backing store without knowing which: an in-process
:class:`~app.reasoning.corporate_event_graph.CorporateEventGraph` built from the
frozen corpus, or
:class:`~app.retrieval.corporate_event_repository.PostgresCorporateEventRepository`
reading the persisted tables.  Both expose the same four methods, so the resolver
is a thin, well-defined wrapper rather than a second implementation.

Traversal runs in both directions:

forward
    a contract filing to the termination that closed it,

reverse
    a termination back to every contract filing it closed.

Both are the same lookup, because an event holds its whole membership: the
direction is a property of the question, not of the storage.  Nothing here
searches or ranks, and a missing graph degrades to "no event" instead of
failing the request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, Sequence

from app.reasoning.corporate_event import (
    LIFECYCLE_TERMINATED,
    RESOLVED,
    ROLE_CONTRACT,
    ROLE_CONTRACT_UPDATE,
    ROLE_TERMINATION,
    CorporateEvent,
    CorporateEventMember,
    CorporateEventState,
)
from app.reasoning.corporate_event_graph import CorporateEventGraphUnavailable


logger = logging.getLogger(__name__)

#: Roles that describe the contract itself rather than its end.
CONTRACT_ROLES = frozenset({ROLE_CONTRACT, ROLE_CONTRACT_UPDATE})


class CorporateEventView(Protocol):
    """What the resolver needs from a graph or a repository."""

    def get_event(self, doc_id: str) -> CorporateEvent | None: ...

    def get_event_timeline(
        self, doc_id: str
    ) -> tuple[CorporateEventMember, ...]: ...

    def get_related_documents(self, doc_id: str) -> tuple[str, ...]: ...

    def event_states(
        self, doc_ids: Iterable[str]
    ) -> dict[str, CorporateEventState]: ...


@dataclass(frozen=True)
class TimelineEntry:
    """One filing on a lifecycle, with the provenance needed to cite it.

    ``canonical_doc_id`` is the version to quote when the question wants the
    contract as it now stands; ``doc_id`` is always the filing that actually
    said it.  Keeping both is what stops a correction from erasing the filing it
    corrects.

    ``root_doc_id`` is the third of that set and the one that never moves: the
    filing this member's correction chain began from, which for an uncorrected
    filing is the filing itself.  A canonical id answers "what does this
    contract say now"; a root id answers "which contract is this", and only the
    latter can tell two contracts apart when a correction lands on the same day
    another contract was disclosed.
    """

    doc_id: str
    canonical_doc_id: str
    member_role: str
    member_order: int
    root_doc_id: str | None = None
    event_date: str | None = None
    correction_group_id: str | None = None
    correction_resolution_status: str | None = None
    is_correction: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_termination(self) -> bool:
        return self.member_role == ROLE_TERMINATION

    @property
    def has_verified_version_history(self) -> bool:
        """Whether P0-A established this filing's own correction chain."""

        return self.correction_resolution_status in (None, RESOLVED)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "canonical_doc_id": self.canonical_doc_id,
            "member_role": self.member_role,
            "member_order": self.member_order,
            "root_doc_id": self.root_doc_id,
            "event_date": self.event_date,
            "correction_group_id": self.correction_group_id,
            "correction_resolution_status": self.correction_resolution_status,
            "is_correction": self.is_correction,
            "is_termination": self.is_termination,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class EventTimeline:
    """One lifecycle rendered in filing order, with its state and evidence."""

    doc_id: str
    event_id: str | None = None
    corp_code: str | None = None
    event_family: str | None = None
    lifecycle_status: str | None = None
    resolution_status: str | None = None
    resolution_source: str | None = None
    confidence: float = 0.0
    opened_at: str | None = None
    closed_at: str | None = None
    entries: tuple[TimelineEntry, ...] = ()

    @property
    def found(self) -> bool:
        return self.event_id is not None

    @property
    def is_terminated(self) -> bool:
        return self.lifecycle_status == LIFECYCLE_TERMINATED

    @property
    def is_resolved(self) -> bool:
        return self.resolution_status == RESOLVED and len(self.entries) > 1

    @property
    def contract_entries(self) -> tuple[TimelineEntry, ...]:
        return tuple(
            entry for entry in self.entries if entry.member_role in CONTRACT_ROLES
        )

    @property
    def termination_entries(self) -> tuple[TimelineEntry, ...]:
        return tuple(entry for entry in self.entries if entry.is_termination)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "event_id": self.event_id,
            "corp_code": self.corp_code,
            "event_family": self.event_family,
            "lifecycle_status": self.lifecycle_status,
            "resolution_status": self.resolution_status,
            "resolution_source": self.resolution_source,
            "confidence": self.confidence,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "is_terminated": self.is_terminated,
            "is_resolved": self.is_resolved,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _entry(member: CorporateEventMember) -> TimelineEntry:
    document = (member.evidence or {}).get("document") or {}
    canonical = (member.evidence or {}).get("canonical") or {}
    return TimelineEntry(
        doc_id=member.doc_id,
        canonical_doc_id=member.canonical_doc_id,
        member_role=member.member_role,
        member_order=member.member_order,
        # The member's own field, not the canonical evidence blob: P0-B already
        # settles it to the filing itself when nothing corrected this one, so it
        # is always a real filing rather than sometimes None.
        root_doc_id=member.root_doc_id or member.doc_id,
        event_date=member.event_date,
        correction_group_id=member.correction_group_id,
        correction_resolution_status=member.correction_resolution_status,
        is_correction=member.is_correction,
        provenance={
            "rcept_dt": document.get("rcept_dt"),
            "kind": document.get("kind"),
            "counterparty": document.get("counterparty"),
            "subject": document.get("subject"),
            "amount": document.get("amount"),
            "period_start": document.get("period_start"),
            "period_end": document.get("period_end"),
            "termination_date": document.get("termination_date"),
            "termination_reason": document.get("termination_reason"),
            "root_doc_id": canonical.get("root_doc_id"),
            "correction_chain": canonical.get("correction_chain") or [],
        },
    )


class CorporateEventResolver:
    """Answer lifecycle questions about one document, deterministically."""

    def __init__(
        self, view: CorporateEventView | None, *, strict: bool = False
    ) -> None:
        self._view = view
        # ``strict`` re-raises ``CorporateEventGraphUnavailable`` instead of
        # degrading to "no event".  A caller that has to tell "the graph is not
        # there" apart from "this document has no lifecycle" -- retrieval
        # expansion, which reports the reason in its trace -- asks for it.
        self._strict = strict

    # ------------------------------------------------------------------ api

    def get_event(self, doc_id: str) -> CorporateEvent | None:
        """The lifecycle this filing belongs to, or None."""

        return self._call("get_event", doc_id)

    def get_event_state(self, doc_id: str) -> CorporateEventState | None:
        """What this filing is inside its lifecycle, without the full membership."""

        states = self._call("event_states", [str(doc_id)]) or {}
        return states.get(str(doc_id))

    def get_event_timeline(self, doc_id: str) -> EventTimeline:
        """The whole lifecycle in filing order, oldest first.

        Returns an empty timeline rather than raising when the document is not a
        contract filing, when db/007 has not been applied, or when the database
        is unreachable.
        """

        doc_id = str(doc_id)
        event = self.get_event(doc_id)
        if event is None:
            return EventTimeline(doc_id=doc_id)
        entries = tuple(
            _entry(member)
            for member in sorted(event.members, key=lambda item: item.member_order)
        )
        return EventTimeline(
            doc_id=doc_id,
            event_id=event.event_id,
            corp_code=event.corp_code,
            event_family=event.event_family,
            lifecycle_status=event.lifecycle_status,
            resolution_status=event.resolution_status,
            resolution_source=event.resolution_source,
            confidence=event.confidence,
            opened_at=event.opened_at,
            closed_at=event.closed_at,
            entries=entries,
        )

    def get_related_documents(self, doc_id: str) -> tuple[str, ...]:
        """Every other filing of this lifecycle, in timeline order."""

        return tuple(self._call("get_related_documents", doc_id) or ())

    # ------------------------------------------------------- directed lookups

    def get_terminations(self, doc_id: str) -> tuple[TimelineEntry, ...]:
        """Forward: contract to the filings that closed it."""

        return self.get_event_timeline(doc_id).termination_entries

    def get_contract_documents(self, doc_id: str) -> tuple[TimelineEntry, ...]:
        """Reverse: termination back to the contract filings it closed."""

        return self.get_event_timeline(doc_id).contract_entries

    def expansion_targets(self, doc_id: str) -> tuple[str, ...]:
        """Which other filings a resolved lifecycle justifies adding as evidence.

        Only a resolved multi-document lifecycle contributes anything.  An
        ambiguous or unresolved termination is a real filing whose contract was
        never identified, so nothing is pulled in beside it.
        """

        timeline = self.get_event_timeline(doc_id)
        if not timeline.is_resolved:
            return ()
        doc_id = str(doc_id)
        return tuple(
            entry.doc_id for entry in timeline.entries if entry.doc_id != doc_id
        )

    def event_states(self, doc_ids: Iterable[str]) -> dict[str, CorporateEventState]:
        return self._call("event_states", list(doc_ids)) or {}

    # -------------------------------------------------------------- internal

    def _call(self, name: str, *args: Any) -> Any:
        """Degrade to "no event" for the two conditions a caller can live with."""

        if self._view is None:
            return None
        method = getattr(self._view, name, None)
        if not callable(method):
            return None
        try:
            return method(*args)
        except CorporateEventGraphUnavailable as error:
            if self._strict:
                raise
            logger.warning("corporate event lookup skipped: %s", error)
            return None


def resolve_event_timeline(
    view: CorporateEventView | None, doc_id: str
) -> EventTimeline:
    """One-shot convenience for callers that hold no resolver."""

    return CorporateEventResolver(view).get_event_timeline(doc_id)


def _target_priority(seed_role: str | None, entry: TimelineEntry) -> tuple[int, int]:
    """Deterministic ordering of what a seed most needs beside it.

    A contract wants the filing that closed it; a termination wants the contract
    it closed.  Ties keep filing order, so the result never depends on dict or
    row ordering.  This is a fixed preference, not an intent planner.
    """

    if seed_role in CONTRACT_ROLES:
        rank = 0 if entry.is_termination else 1
    elif seed_role == ROLE_TERMINATION:
        rank = 0 if entry.member_role in CONTRACT_ROLES else 1
    else:
        rank = 1
    return (rank, entry.member_order)


def seed_expansion_targets(
    view: CorporateEventView | None,
    doc_ids: Sequence[str],
    *,
    limit: int = 32,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Documents a set of retrieval seeds justifies adding, plus the trace.

    Returns ``(target_doc_ids, expanded_events, diagnostics)``.  ``diagnostics``
    carries what was deliberately *not* expanded -- an unresolved lifecycle, a
    truncated target list -- so a caller can report a decision it did not take.

    One event is expanded once no matter how many of its members were retrieved.
    Deduplication is on *logical* identity: a seed that is a superseded filing
    already stands for its whole correction group, so that group's representative
    is never added back as if it were a separate document.
    """

    resolver = CorporateEventResolver(view, strict=True)
    seeds = [str(doc_id) for doc_id in doc_ids if str(doc_id)]

    # Every seed, plus the logical member each one stands for.  Both forms are
    # "already present" for deduplication purposes.  One batched lookup rather
    # than one per seed: a retrieval result holds up to ``seed_limit`` documents
    # and most of them are not lifecycle members at all.
    present: set[str] = set(seeds)
    seed_member: dict[str, str] = {}
    seed_event: dict[str, str] = {}
    # The same batched lookup, additionally kept whole.  A downstream consumer
    # that needs to know which filing of a lifecycle is the canonical one --
    # and whether P0-A positively established that -- would otherwise have to
    # re-derive it from the flattened trace, which cannot express it.  Nothing
    # here computes state; it is carried instead of discarded.
    member_states: dict[str, dict[str, Any]] = {}
    for doc_id, state in (resolver.event_states(seeds) or {}).items():
        seed_member[doc_id] = state.doc_id
        seed_event[doc_id] = state.event_id
        present.add(state.doc_id)
        member_states[doc_id] = state.to_dict()

    seen_events: set[str] = set()
    wanted: list[str] = []
    events: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    truncated = False
    for doc_id in seeds:
        # A seed with no state belongs to no lifecycle, and a seed whose event
        # was already expanded needs no second traversal.  Both are settled by
        # the batched lookup above, so neither costs a query.
        event_id = seed_event.get(doc_id)
        if event_id is None or event_id in seen_events:
            continue
        timeline = resolver.get_event_timeline(doc_id)
        if not timeline.found or timeline.event_id in seen_events:
            continue
        seen_events.add(str(timeline.event_id))
        own = seed_member.get(doc_id, doc_id)
        if not timeline.is_resolved:
            # An ambiguous or unresolved lifecycle is a real filing whose
            # counterpart was never identified.  Nothing lookalike is pulled in.
            skipped.append(
                {
                    "seed_doc_id": doc_id,
                    "event_id": timeline.event_id,
                    "reason": "lifecycle_not_resolved",
                    "resolution_status": timeline.resolution_status,
                    "resolution_source": timeline.resolution_source,
                    "seed_member_doc_id": own,
                    "seed_root_doc_id": _event_root_doc_id(timeline),
                    "seed_corp_code": timeline.corp_code,
                    "event_family": timeline.event_family,
                    "lifecycle_status": timeline.lifecycle_status,
                    "member_count": len(timeline.entries),
                    "seed_identity": _seed_identity(timeline, own),
                }
            )
            continue
        seed_role = next(
            (
                entry.member_role
                for entry in timeline.entries
                if entry.doc_id == own
            ),
            None,
        )
        candidates = sorted(
            (
                entry
                for entry in timeline.entries
                if entry.doc_id != own
                and entry.doc_id not in present
                and entry.doc_id not in wanted
            ),
            key=lambda entry: _target_priority(seed_role, entry),
        )
        room = max(0, limit - len(wanted))
        if not candidates:
            # Resolved, but every member was already retrieved.  Recorded so the
            # caller reports "already retrieved" rather than "no lifecycle".
            events.append(
                {
                    "event_id": timeline.event_id,
                    "seed_doc_id": doc_id,
                    "seed_member_doc_id": own,
                    "seed_root_doc_id": _event_root_doc_id(timeline),
                    "seed_member_role": seed_role,
                    "seed_corp_code": timeline.corp_code,
                    "event_family": timeline.event_family,
                    "lifecycle_status": timeline.lifecycle_status,
                    "resolution_source": timeline.resolution_source,
                    "confidence": timeline.confidence,
                    "member_count": len(timeline.entries),
                    "correction_group_id": None,
                    "seed_identity": _seed_identity(timeline, own),
                    "target_doc_ids": [],
                }
            )
            continue
        targets = [entry.doc_id for entry in candidates[:room]]
        if len(candidates) > room:
            truncated = True
        if not targets:
            truncated = True
            break
        events.append(
            {
                "event_id": timeline.event_id,
                "seed_doc_id": doc_id,
                "seed_member_doc_id": own,
                "seed_root_doc_id": _event_root_doc_id(timeline),
                "seed_member_role": seed_role,
                "seed_corp_code": timeline.corp_code,
                "event_family": timeline.event_family,
                "lifecycle_status": timeline.lifecycle_status,
                "resolution_source": timeline.resolution_source,
                "confidence": timeline.confidence,
                "member_count": len(timeline.entries),
                "correction_group_id": next(
                    (
                        entry.correction_group_id
                        for entry in timeline.entries
                        if entry.doc_id == own and entry.correction_group_id
                    ),
                    None,
                ),
                "seed_identity": _seed_identity(timeline, own),
                "target_doc_ids": list(targets),
            }
        )
        wanted.extend(targets)
        if len(wanted) >= limit:
            truncated = truncated or len(candidates) > room
            break
    return wanted[:limit], events, {
        "skipped": skipped,
        "truncated": truncated,
        "seed_member_doc_ids": dict(seed_member),
        "member_states": dict(member_states),
        "events_considered": len(seen_events),
    }


def _event_root_doc_id(timeline: EventTimeline) -> str | None:
    """The filing one lifecycle began from, as the graph proved it.

    Entries are in filing order, so the first contract-role member is the
    conclusion that opened this lifecycle, and its ``root_doc_id`` unwinds any
    correction chain that member collapsed.  A 해지 is skipped for the same
    reason it never names a contract: it ended one, it did not open one.
    """

    entries = timeline.contract_entries or timeline.entries
    for entry in entries:
        root = str(entry.root_doc_id or entry.doc_id or "").strip()
        if root:
            return root
    return None


def _seed_identity(timeline: EventTimeline, doc_id: str) -> dict[str, str]:
    """The graph-proven contract identity for one event seed.

    Clarification needs to distinguish independent roots that nevertheless name
    the same counterparty and broad contract.  The timeline already carries
    those structured fields from the canonical event graph; preserve that
    identity in the expansion trace instead of making the clarification layer
    parse answer prose or inspect arbitrary retrieved documents.
    """

    entry = next((item for item in timeline.entries if item.doc_id == doc_id), None)
    provenance = dict(entry.provenance) if entry is not None else {}
    return {
        key: value
        for key in ("counterparty", "subject")
        if (value := str(provenance.get(key) or "").strip())
    }
