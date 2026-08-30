"""Whether the served evidence can answer the holding fields a question asks.

A holding disclosure stores its rows twice: as the raw table, and as a
structured projection whose labels are canonical (``기준일/보고일``,
``보유주식수``).  Questions ask in the table's vocabulary, so the projections
match no lexical query and reach the served results only when the vector lane
happens to rank them inside the cutoff.  When they do not, the resolver is
handed table rows it can read no field from.

The projection types also carry different fields: ``holding_detail_row`` has the
share counts and the reference date, ``holding_report`` adds the ratios.  Served
evidence can therefore be plentiful and still be unable to answer -- ten detail
rows cannot supply a holding ratio.

So sufficiency is not "is a projection present".  It is whether the union of the
served, citable, reporter-compatible projections covers every field the question
actually asked for.  This module answers that question and, when the answer is
no, selects the fewest candidates from the pool retrieval already fetched that
close the gap.  It issues no query of its own.

Coverage is decided by driving the resolver's own field reader rather than
restating its vocabulary, so this module and the component that consumes what it
selects cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Mapping, Sequence

from app.reasoning.evidence_builder import _evidence_item, _normalize_date
from app.reasoning.holding_event_resolver import (
    _FIELD_LABELS,
    _field_candidate,
    _normalize_text,
    _requested_fields,
)
from app.retrieval.interfaces import CandidateChunk, RetrievalResult


#: The structured holding projections a rescue may promote.
HOLDING_PROJECTION_TYPES = ("holding_detail_row", "holding_report")

#: Statuses recorded on the assessment, for diagnostics.
STATUS_NOT_HOLDING = "not_holding_execution"
STATUS_NO_REQUESTED_FIELDS = "no_requested_fields"
STATUS_COVERED = "served_evidence_covers_request"
STATUS_NO_CANDIDATE = "no_eligible_candidate"
STATUS_NO_SAFE_DISPLACEMENT = "no_safe_displacement"
STATUS_NO_ANCHOR = "no_anchored_candidate"
STATUS_RESCUED = "rescued"
STATUS_PARTIAL = "rescued_partial_coverage"

#: How firmly a candidate is tied to evidence retrieval already served.
ANCHOR_STRONG = "strong"
ANCHOR_MEDIUM = "medium"
#: Strong outranks medium at the same served anchor.
_ANCHOR_ORDER = {ANCHOR_STRONG: 0, ANCHOR_MEDIUM: 1}

#: Projection labels that describe the holding event itself. A source ref is
#: only an event anchor when it backs one of these; a ref backing document
#: metadata such as the holding purpose is shared by every projection of a
#: filing and identifies no particular event.
_EVENT_FIELDS = (
    "reporter",
    "reference_date",
    "before_shares",
    "change_shares",
    "after_shares",
    "before_ratio",
    "after_ratio",
    "change_ratio",
)
_EVENT_LABELS = frozenset(
    label for name in _EVENT_FIELDS for label in _FIELD_LABELS.get(name, ())
)


@dataclass(frozen=True)
class CoverageAssessment:
    """What the served evidence covers, and what was added to close the gap."""

    requested: tuple[str, ...] = ()
    served_coverage: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()
    selected: tuple[RetrievalResult, ...] = ()
    displaced: tuple[str, ...] = ()
    remaining_unresolved: tuple[str, ...] = ()
    status: str = STATUS_NOT_HOLDING
    results: tuple[RetrievalResult, ...] = ()
    evaluated: bool = False
    anchored_candidate_count: int = 0
    #: (chunk_id, anchor tier, served rank of the item it completes)
    anchors: tuple[tuple[str, str, int], ...] = ()

    @property
    def rescued(self) -> bool:
        return bool(self.selected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluated": self.evaluated,
            "status": self.status,
            "requested_fields": list(self.requested),
            "served_coverage": list(self.served_coverage),
            "unresolved_fields": list(self.unresolved),
            "rescued": self.rescued,
            "selected_count": len(self.selected),
            "selected_chunk_ids": [result.chunk_id for result in self.selected],
            "displaced_chunk_ids": list(self.displaced),
            "remaining_unresolved_fields": list(self.remaining_unresolved),
            "anchored_candidate_count": self.anchored_candidate_count,
            "anchors": [
                {"chunk_id": chunk_id, "tier": tier, "served_rank": rank}
                for chunk_id, tier, rank in self.anchors
            ],
        }


def requested_holding_fields(question: str, plan: Any) -> tuple[str, ...]:
    """The canonical holding fields this question asks for.

    Delegates to the resolver's own request parser so the rescue asks for
    exactly what the resolver will later try to resolve.
    """

    mapping = plan.to_dict() if hasattr(plan, "to_dict") else plan
    return _requested_fields(str(question or ""), dict(mapping or {}))


def holding_projection_type(chunk: Mapping[str, Any]) -> str | None:
    projection = str(chunk.get("projection_type") or "")
    return projection if projection in HOLDING_PROJECTION_TYPES else None


def is_citable(chunk: Mapping[str, Any]) -> bool:
    """Whether a promoted projection could still support a citation.

    The resolver cites a field from its own ``projection_field_refs`` and falls
    back to the chunk's ``source_refs``; a projection with neither cannot be
    cited, so promoting it would only add uncitable evidence.
    """

    if any((chunk.get("projection_field_refs") or {}).values()):
        return True
    return bool(chunk.get("source_refs"))


def reporter_compatible(chunk: Mapping[str, Any], reporter: str | None) -> bool:
    """Whether this projection's holder is the one the question named.

    Containment either way on the resolver's own normalisation: a question
    naming a holder family matches its members, while a question naming a
    specific member does not match a different one.  When the question named no
    holder, nothing is excluded -- and none is invented.
    """

    wanted = _normalize_text(reporter or "")
    if not wanted:
        return True
    fields = dict(chunk.get("projection_fields") or {})
    holder = _normalize_text(
        fields.get("보고자/보유자") or chunk.get("reporter") or ""
    )
    if not holder:
        return False
    return wanted in holder or holder in wanted


def covered_fields(
    candidate: CandidateChunk,
    result: RetrievalResult,
    fields: Sequence[str],
) -> set[str]:
    """Which of ``fields`` this chunk can actually supply.

    Asks the resolver's own per-field reader rather than re-deriving the label
    table, so a field the resolver computes -- ``change_direction`` from the
    sign of the share change -- counts as covered exactly when the resolver
    would compute it, and a label the resolver does not read never does.
    """

    item = _evidence_item(candidate, result)
    return {name for name in fields if _field_candidate(name, item) is not None}


def _ref_key(ref: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (str(ref.get("table_id") or ""), ref.get("row_start"), ref.get("row_end"))


def _event_refs(chunk: Mapping[str, Any]) -> set[tuple[Any, Any, Any]]:
    """Source rows that back this projection's event fields.

    Refs behind metadata-only labels are deliberately excluded: every
    projection of a filing points at the same holding-purpose row, so treating
    it as an anchor would make any two projections of one disclosure look like
    the same event.
    """

    refs: set[tuple[Any, Any, Any]] = set()
    for label, entries in (chunk.get("projection_field_refs") or {}).items():
        if label not in _EVENT_LABELS:
            continue
        for ref in entries or []:
            refs.add(_ref_key(ref))
    return refs


def _all_refs(chunk: Mapping[str, Any]) -> set[tuple[Any, Any, Any]]:
    refs = {_ref_key(ref) for ref in (chunk.get("source_refs") or [])}
    for entries in (chunk.get("projection_field_refs") or {}).values():
        for ref in entries or []:
            refs.add(_ref_key(ref))
    return refs


def _reference_date(chunk: Mapping[str, Any]) -> str | None:
    fields = dict(chunk.get("projection_fields") or {})
    for label in _FIELD_LABELS.get("reference_date", ()):
        value = _normalize_date(fields.get(label))
        if value:
            return value
    return _normalize_date(chunk.get("reference_date"))


def anchor_tier(
    candidate: Mapping[str, Any],
    served: Mapping[str, Any],
    reporter: str | None,
) -> str | None:
    """How firmly this candidate completes that served item, if at all.

    STRONG -- the two describe the same source rows: the candidate's event
    fields are backed by a row the served item also covers, so the candidate is
    literally the structured rendering of evidence already retrieved.

    MEDIUM -- the exact row link is unavailable (a served projection and a
    candidate projection of the same filing draw on different tables), but both
    carry the same reference date inside the same disclosure for a compatible
    holder, so they describe one event.

    Same document alone is never enough: a disclosure can report several
    holding events.
    """

    if str(candidate.get("doc_id") or "") != str(served.get("doc_id") or ""):
        return None
    if _event_refs(candidate) & _all_refs(served):
        return ANCHOR_STRONG
    served_date = _reference_date(served)
    if served_date and served_date == _reference_date(candidate):
        if reporter_compatible(candidate, reporter):
            return ANCHOR_MEDIUM
    return None


def _minimum_subset(
    candidates: Sequence[tuple[CandidateChunk, frozenset[str]]],
    unresolved: Sequence[str],
) -> list[CandidateChunk]:
    """Fewest candidates from one anchor/tier that close the most fields.

    Exhaustive over the unresolved subset, which is tiny -- at most the handful
    of holding fields a question can request -- so the exact answer is cheaper
    than reasoning about when a greedy approximation would be good enough.
    Ordering of equal-sized solutions follows the caller's order, which is
    already deterministic.
    """

    wanted = set(unresolved)
    useful = [
        (chunk, fields & wanted)
        for chunk, fields in candidates
        if fields & wanted
    ]
    if not useful:
        return []
    best: list[CandidateChunk] | None = None
    best_key: tuple[int, int] | None = None
    for size in range(1, len(useful) + 1):
        for combo in combinations(range(len(useful)), size):
            gained = set().union(*(useful[i][1] for i in combo))
            key = (-len(gained), size)
            if best_key is None or key < best_key:
                best_key = key
                best = [useful[i][0] for i in combo]
            if len(gained) == len(wanted):
                return best or []
        if best_key is not None and best_key[1] <= size and best_key[0] == -len(wanted):
            break
    return best or []


def _eligible(
    candidate: CandidateChunk, reporter: str | None
) -> bool:
    chunk = candidate.chunk
    return (
        holding_projection_type(chunk) is not None
        and is_citable(chunk)
        and reporter_compatible(chunk, reporter)
    )


#: Marker written onto a promoted row so coverage-derived evidence stays
#: distinguishable from what retrieval itself ranked.
PROVENANCE_KEY = "holding_evidence_coverage"


def _synthetic(candidate: CandidateChunk, rank: int) -> RetrievalResult:
    match = dict(candidate.metadata_match.to_dict())
    match[PROVENANCE_KEY] = {"selected_for": "holding_field_coverage"}
    return RetrievalResult(
        chunk_id=candidate.chunk_id,
        doc_id=candidate.doc_id,
        bm25_score=0.0,
        rank=rank,
        metadata_match=match,
    )


def assess(
    question: str,
    plan: Any,
    chunks: Sequence[CandidateChunk],
    results: Sequence[RetrievalResult],
    *,
    routed_task_type: str | None = None,
) -> CoverageAssessment:
    """Decide whether the served results can answer the requested fields.

    ``routed_task_type`` is the execution shape the router chose.  Only a
    holding execution is assessed; every other route returns an untouched
    result list.
    """

    if routed_task_type != "holding_event":
        return CoverageAssessment(results=tuple(results), status=STATUS_NOT_HOLDING)

    reporter = str(getattr(plan, "reporter", "") or "").strip()
    requested = requested_holding_fields(
        str(getattr(plan, "raw_query", "") or question or ""), plan
    )
    if not requested:
        return CoverageAssessment(
            results=tuple(results),
            status=STATUS_NO_REQUESTED_FIELDS,
            evaluated=True,
        )

    by_id = {candidate.chunk_id: candidate for candidate in chunks}
    served_ids = [result.chunk_id for result in results]

    # What each served row contributes, so coverage and safe displacement are
    # decided from the same measurement.
    contribution: dict[str, set[str]] = {}
    served_coverage: set[str] = set()
    for result in results:
        candidate = by_id.get(result.chunk_id)
        if candidate is None or not _eligible(candidate, reporter):
            contribution[result.chunk_id] = set()
            continue
        covered = covered_fields(candidate, result, requested)
        contribution[result.chunk_id] = covered
        served_coverage |= covered

    unresolved = [name for name in requested if name not in served_coverage]
    base = CoverageAssessment(
        requested=tuple(requested),
        served_coverage=tuple(sorted(served_coverage)),
        unresolved=tuple(unresolved),
        results=tuple(results),
        evaluated=True,
    )
    if not unresolved:
        return CoverageAssessment(**{**base.__dict__, "status": STATUS_COVERED})

    # Candidates are only ever drawn from evidence retrieval already ranked, so
    # a rescue completes what was retrieved rather than introducing an event of
    # its own choosing. Served rank therefore dominates: the top-ranked served
    # item is completed first, and a lower-ranked anchor is consulted only for
    # fields the higher one cannot supply -- even when a single lower-ranked
    # candidate could have closed everything at once. Candidate count is
    # secondary to retrieval-derived relevance.
    seen = set(served_ids)
    pool: list[CandidateChunk] = []
    for candidate in chunks:
        if candidate.chunk_id in seen or not _eligible(candidate, reporter):
            continue
        probe = _synthetic(candidate, 0)
        if covered_fields(candidate, probe, requested):
            pool.append(candidate)

    anchored_count = 0
    remaining = list(unresolved)
    selected: list[RetrievalResult] = []
    picked: dict[str, tuple[str, int]] = {}
    for result in results:                       # served rank order
        served_chunk = by_id.get(result.chunk_id)
        if served_chunk is None:
            continue
        tiers: dict[str, list[tuple[CandidateChunk, frozenset[str]]]] = {
            ANCHOR_STRONG: [], ANCHOR_MEDIUM: []
        }
        for candidate in pool:
            if candidate.chunk_id in seen:
                continue
            tier = anchor_tier(candidate.chunk, served_chunk.chunk, reporter)
            if tier is None:
                continue
            anchored_count += 1
            gained = covered_fields(candidate, _synthetic(candidate, 0), requested)
            tiers[tier].append((candidate, frozenset(gained)))
        if not remaining:
            continue
        for tier in (ANCHOR_STRONG, ANCHOR_MEDIUM):
            if not remaining:
                break
            for candidate in _minimum_subset(tiers[tier], remaining):
                rank = len(results) + len(selected) + 1
                promoted = _synthetic(candidate, rank)
                gained = covered_fields(candidate, promoted, remaining)
                if not gained:
                    continue
                selected.append(promoted)
                seen.add(candidate.chunk_id)
                contribution[candidate.chunk_id] = gained
                picked[candidate.chunk_id] = (tier, result.rank)
                remaining = [n for n in remaining if n not in gained]

    if not selected:
        # Policy: anchored only. A candidate that closes the gap but completes
        # nothing already retrieved would be an event of the lane's own
        # choosing, and a missing answer is safer than a confident arbitrary one.
        status = STATUS_NO_ANCHOR if pool else STATUS_NO_CANDIDATE
        return CoverageAssessment(
            **{**base.__dict__, "status": status,
               "anchored_candidate_count": anchored_count}
        )

    merged, displaced = _merge(results, selected, contribution, requested)
    if merged is None:
        return CoverageAssessment(
            **{**base.__dict__, "status": STATUS_NO_SAFE_DISPLACEMENT,
               "anchored_candidate_count": anchored_count}
        )
    return CoverageAssessment(
        **{
            **base.__dict__,
            "selected": tuple(selected),
            "displaced": tuple(displaced),
            "remaining_unresolved": tuple(remaining),
            "results": tuple(merged),
            "status": STATUS_RESCUED if not remaining else STATUS_PARTIAL,
            "anchored_candidate_count": anchored_count,
            "anchors": tuple(
                (result.chunk_id, *picked[result.chunk_id]) for result in selected
            ),
        }
    )


def _merge(
    results: Sequence[RetrievalResult],
    selected: Sequence[RetrievalResult],
    contribution: Mapping[str, set[str]],
    requested: Sequence[str],
) -> tuple[list[RetrievalResult] | None, list[str]]:
    """Insert the selected rows without growing or damaging the served list.

    The list keeps its length, so no global limit changes.  Room is made from
    the lowest-ranked rows that contribute none of the requested fields; a row
    that is itself covering something is never dropped.  If only contributors
    remain, the rescue is abandoned rather than trading one gap for another.
    """

    dropped: set[str] = set()
    displaced: list[str] = []
    needed = len(selected)
    for result in reversed(list(results)):
        if needed <= 0:
            break
        if contribution.get(result.chunk_id):
            continue
        dropped.add(result.chunk_id)
        displaced.append(result.chunk_id)
        needed -= 1
    if needed > 0:
        return None, []
    keep = [row for row in results if row.chunk_id not in dropped]

    merged: list[RetrievalResult] = []
    seen: set[str] = set()
    for rank, result in enumerate([*keep, *selected], start=1):
        if result.chunk_id in seen:
            continue
        seen.add(result.chunk_id)
        merged.append(
            RetrievalResult(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                bm25_score=result.bm25_score,
                rank=rank,
                metadata_match=result.metadata_match,
            )
        )
    return merged, displaced
