"""P1-A4 D2: complementary views of one holding event, resolved as one event.

One filing can describe a single event twice.  The detail table carries the
per-holder row -- shares, no ratios -- while the summary table carries the
filer's report -- the same shares plus the ratios.  ``EvidenceBuilder`` keeps
them apart, correctly: their holder labels differ, and across the corpus two
rows of one table with different holders really are different events.

The cost is that a question naming one day matches two logical events, and
``AnswerComposer`` narrows to a single event only when exactly one matches.  So
the requested day is answered with every retrieved event, and asking for both a
share count and a ratio can only be satisfied by two events at once.

This module builds an execution-scoped ``EvidenceSet`` in which such pairs share
one group, and lets the resolver's own field machinery union them.  Nothing is
invented: identity is argued from the numbers and the source tables, never from
holder text, and a merge that would contradict anything already resolved is
declined.  The builder's output is left untouched, and no chunk is added or
removed -- only the grouping changes.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, Sequence

from app.reasoning.evidence_builder import EvidenceGroup, EvidenceItem, EvidenceSet, _make_group
from app.reasoning.holding_event_resolver import (
    _field_candidate,
    _normalize_text,
    _reporter_matches,
    _resolve_group,
)

HOLDING_GROUP_TYPE = "holding_event"

#: The transition that has to agree for two views to be the same event.
_TRANSITION_FIELDS = ("before_shares", "change_shares", "after_shares")

#: Fields that describe the event itself.  The tables behind these are what
#: distinguishes "another view of this event" from "another event"; a shared
#: metadata row such as 보유 목적 says nothing, because every projection of a
#: filing points at it.  Same principle P1-A3 anchors on.
_EVENT_FIELDS = (
    "reference_date",
    "before_shares",
    "change_shares",
    "after_shares",
    "before_ratio",
    "after_ratio",
    "change_ratio",
)

FUSION_REASON = (
    "One filing, one reference date, an identical and internally consistent "
    "share transition, and disjoint event source tables: complementary views of "
    "a single holding event, merged so the resolver can union their fields."
)


def _value(item: EvidenceItem, field_name: str) -> Any:
    candidate = _field_candidate(field_name, item)
    return None if candidate is None else candidate.normalized


def _transition(item: EvidenceItem) -> tuple[Any, ...]:
    return tuple(_value(item, name) for name in _TRANSITION_FIELDS)


def _reference_date(item: EvidenceItem) -> str | None:
    value = _value(item, "reference_date")
    return str(value) if value else None


def _event_tables(item: EvidenceItem) -> set[str]:
    """Which tables back this item's event fields."""

    tables: set[str] = set()
    for field_name in _EVENT_FIELDS:
        candidate = _field_candidate(field_name, item)
        if candidate is None:
            continue
        for ref in candidate.source.source_refs or ():
            if isinstance(ref, Mapping):
                tables.add(str(ref.get("table_id") or ""))
    return tables


def _reporter(item: EvidenceItem) -> str | None:
    value = _value(item, "reporter")
    text = _normalize_text(str(value)) if value else ""
    return text or None


def _consistent(transition: Sequence[Any]) -> bool:
    """Whether the row's own arithmetic holds, so the numbers can be trusted."""

    before, change, after = transition
    if any(not isinstance(value, (int, float)) or isinstance(value, bool)
           for value in (before, change, after)):
        return False
    return before + change == after


def same_event(left: EvidenceItem, right: EvidenceItem) -> bool:
    """Whether two evidence items are two views of one holding event.

    Deliberately decided from structure alone.  Holder text is never consulted:
    the corpus carries no identifier that could prove two holder labels are one
    party, so a rule built on names would be a maintained alias list.
    """

    if left.doc_id != right.doc_id:
        return False
    date = _reference_date(left)
    if not date or date != _reference_date(right):
        return False

    transition = _transition(left)
    if any(value is None for value in transition):
        return False
    if transition != _transition(right):
        return False
    if not _consistent(transition):
        return False
    if all(value == 0 for value in transition):
        # A no-op row is not an identity.  Several holders of one filing can
        # each report 0 -> 0, and they are not the same party.
        return False

    # Two rows of one table are an enumeration of separate events.  Two
    # different tables can be two renderings of the same one.
    return not (_event_tables(left) & _event_tables(right))


def _items(groups: Iterable[EvidenceGroup]) -> list[EvidenceItem]:
    ordered: list[EvidenceItem] = []
    seen: set[str] = set()
    for group in groups:
        for item in group.items:
            if item.chunk_id not in seen:
                seen.add(item.chunk_id)
                ordered.append(item)
    return ordered


def _reporter_safe(items: Sequence[EvidenceItem], constraint: str | None) -> bool:
    """Whether merging these items keeps holder attribution honest.

    One holder is always safe.  Several holders are safe only when the question
    named a holder and every one of them answers to it -- then the merged event
    is about the holder that was asked for, however the filing spelled it.  With
    no holder in the question there is nothing to check the merge against, so it
    is declined rather than producing an event attributed to nobody.
    """

    reporters = {value for value in (_reporter(item) for item in items) if value}
    if len(reporters) <= 1:
        return True
    if not constraint:
        return False
    return all(_reporter_matches(value, constraint) for value in reporters)


def _field_safe(group: EvidenceGroup) -> bool:
    """Whether the merge only fills gaps instead of creating contradictions.

    Resolving the candidate group is the check: the resolver reports a conflict
    for any populated field whose values disagree.  ``reporter`` is expected to
    disagree -- that is the whole reason these views were separate -- and is
    handled by holder attribution above.  Anything else means these were not one
    event after all.
    """

    probe = _resolve_group(
        group,
        requested_fields=(),
        reporter_constraint=None,
        direction_constraint=None,
        explicit_temporal=False,
        temporal_constraint={},
    )
    return not [name for name in probe.conflicting_fields if name != "reporter"]


def _components(groups: Sequence[EvidenceGroup]) -> list[list[EvidenceGroup]]:
    components: list[list[EvidenceGroup]] = []
    for group in groups:
        destination = next(
            (
                component
                for component in components
                if any(
                    same_event(item, member)
                    for item in group.items
                    for member in _items(component)
                )
            ),
            None,
        )
        if destination is None:
            components.append([group])
        else:
            destination.append(group)
    return components


def fuse(
    evidence_set: EvidenceSet,
    *,
    reference_date: str | None,
    reporter: str | None = None,
) -> EvidenceSet:
    """An execution-scoped copy whose same-event holding groups are merged.

    ``reference_date`` scopes the pass to the one day the question asked about,
    so this stays an exact-date feature and never becomes a general holding
    normalisation.  Returns the original object when nothing merges, so an
    untouched question keeps the exact evidence set it always had.
    """

    if not reference_date:
        return evidence_set

    targets = [
        group
        for group in evidence_set.evidence_groups
        if group.group_type == HOLDING_GROUP_TYPE
        and any(_reference_date(item) == reference_date for item in group.items)
    ]
    if len(targets) < 2:
        return evidence_set

    rest = [group for group in evidence_set.evidence_groups if group not in targets]
    merged: list[EvidenceGroup] = []
    changed = False
    for component in _components(targets):
        if len(component) < 2:
            merged.extend(component)
            continue
        items = _items(component)
        candidate = _make_group(
            items, group_type=HOLDING_GROUP_TYPE, reason=FUSION_REASON
        )
        if not _reporter_safe(items, reporter) or not _field_safe(candidate):
            merged.extend(component)
            continue
        merged.append(candidate)
        changed = True

    if not changed:
        return evidence_set

    groups = sorted(
        [*merged, *rest],
        key=lambda group: (
            group.primary_evidence.retrieval_rank,
            group.primary_evidence.chunk_id,
        ),
    )
    return replace(evidence_set, evidence_groups=tuple(groups))
