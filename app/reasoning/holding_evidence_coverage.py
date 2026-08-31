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
from app.reasoning.holding_acquisition import DETAIL_PROJECTION, acquisition_facts
from app.reasoning.holding_event_resolver import (
    CURRENT_HOLDING_STATE_FIELDS,
    _FIELD_LABELS,
    _field_candidate,
    _normalize_text,
    _requested_fields,
)
from app.reasoning.holding_date_intent import exact_reference_date
# Aliased on purpose.  This module already defines a ``requested_holding_fields``
# answering "which resolver state fields did the question ask for"; the field
# lane's function of the same name answers "which canonical answer field did it
# ask for".  They are different questions and must not read as the same call.
from app.reasoning.holding_field_evidence import (
    ACQUISITION_UNIT_PRICE,
    requested_holding_fields as requested_answer_fields,
)
from app.reasoning.holding_report_index import (
    SELECTOR_EXACT_REFERENCE_DATE,
    HoldingReportIndex,
    HoldingReportRecord,
)
from app.reasoning.holding_reporter import canonical_reporter_key, reporter_matches
from app.retrieval.interfaces import CandidateChunk, RetrievalResult


#: The structured holding projections a rescue may promote.
HOLDING_PROJECTION_TYPES = ("holding_detail_row", "holding_report")

#: An internal coverage requirement, never a public answer field.  It means one
#: thing: the frozen acquisition parser can prove an acquisition from this
#: structural detail row.  A question that asks for an acquisition unit price
#: needs such a row served, and no holding *state* field implies one -- a report
#: projection can carry every share count the resolver reads and still prove no
#: acquisition, which is how a served-and-covered request reached the resolver
#: with nothing to resolve.
ACQUISITION_PROOF = "acquisition_proof"

#: Recorded when the acquisition lane appended proof rows.
RESCUE_MODE_ACQUISITION_PROOF = "acquisition_proof"
#: The acquisition lane ran and found no bounded proving row.
STATUS_NO_ACQUISITION_PROOF = "no_acquisition_proof_candidate"

#: Statuses recorded on the assessment, for diagnostics.
STATUS_NOT_HOLDING = "not_holding_execution"
STATUS_NO_REQUESTED_FIELDS = "no_requested_fields"
STATUS_COVERED = "served_evidence_covers_request"
STATUS_NO_CANDIDATE = "no_eligible_candidate"
STATUS_NO_SAFE_DISPLACEMENT = "no_safe_displacement"
STATUS_NO_ANCHOR = "no_anchored_candidate"
STATUS_RESCUED = "rescued"
STATUS_PARTIAL = "rescued_partial_coverage"

#: Internal diagnostic identifying which existing coverage path promoted rows.
RESCUE_MODE_SERVED_ANCHOR = "served_anchor"
RESCUE_MODE_CONTRACT_D = "contract_d"

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
    rescue_mode: str | None = None
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
            "rescue_mode": self.rescue_mode,
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


def _row_union(
    refs: set[tuple[Any, Any, Any]]
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Which rows of each table these refs cover, as one canonical form.

    Row ranges are closed intervals, and the same rows can be written either
    per row or as a span.  Merging overlapping and directly adjacent intervals
    reduces both encodings to the identical answer, so two items can be
    compared on the rows they actually cover rather than on how those rows
    happened to be recorded.
    """

    by_table: dict[str, list[tuple[int, int]]] = {}
    for table_id, start, end in refs:
        if not table_id or not isinstance(start, int) or not isinstance(end, int):
            continue
        if isinstance(start, bool) or isinstance(end, bool) or end < start:
            continue
        by_table.setdefault(str(table_id), []).append((start, end))

    canonical: dict[str, tuple[tuple[int, int], ...]] = {}
    for table_id, spans in by_table.items():
        merged: list[tuple[int, int]] = []
        for start, end in sorted(spans):
            # ``start <= last_end + 1`` merges touching spans too: rows 2-3 and
            # 4-5 cover 2-5, while a gap at row 4 must stay two spans.
            if merged and start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        canonical[table_id] = tuple(merged)
    return canonical


def _same_source_rows(
    candidate: Mapping[str, Any], served: Mapping[str, Any]
) -> bool:
    """Whether some table is covered identically by both items.

    Existential per table, as one shared table covered row-for-row already
    establishes that the candidate renders evidence the served item carries.
    Anything short of equality -- containment, partial overlap, adjacency, or
    merely sharing a table -- is a different event and is not accepted here.
    """

    candidate_rows = _row_union(_event_refs(candidate))
    served_rows = _row_union(_all_refs(served))
    return any(
        candidate_rows[table_id] == served_rows[table_id]
        for table_id in candidate_rows.keys() & served_rows.keys()
    )


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
    literally the structured rendering of evidence already retrieved.  A
    projection records those rows one per field while a served table chunk
    records them as one span, so identical rows can be written two ways; both
    spellings are read here, and neither widens what STRONG means.

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
    if _same_source_rows(candidate, served):
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


def _projection_reporter_key(chunk: Mapping[str, Any]) -> str:
    fields = dict(chunk.get("projection_fields") or {})
    for label in _FIELD_LABELS.get("reporter", ()):
        value = fields.get(label)
        if str(value or "").strip():
            return canonical_reporter_key(value)
    return canonical_reporter_key(chunk.get("reporter"))


def _projection_event_matches(
    candidate: CandidateChunk,
    record: HoldingReportRecord,
) -> bool:
    """Whether this projection states the event identity selected by the index."""

    chunk = candidate.chunk
    reference_date = _reference_date(chunk)
    return (
        candidate.doc_id == record.doc_id
        and str(chunk.get("doc_id") or candidate.doc_id) == record.doc_id
        and str(chunk.get("projection_type") or "") == "holding_report"
        and str(chunk.get("corp_code") or "") == record.issuer_corp_code
        and _projection_reporter_key(chunk) == record.reporter_key
        and str(reference_date or "").replace("-", "") == record.reference_date
        and is_citable(chunk)
    )


def _record_projection_matches(
    candidate: CandidateChunk,
    record: HoldingReportRecord,
) -> bool:
    """Whether this is the frozen projection named by an index record."""

    return (
        candidate.chunk_id == record.projection_chunk_id
        and _projection_event_matches(candidate, record)
    )


def _contract_d_candidate(
    plan: Any,
    chunks: Sequence[CandidateChunk],
    results: Sequence[RetrievalResult],
    requested: Sequence[str],
    *,
    report_index: HoldingReportIndex | None,
    active_corpus_identity: Mapping[str, Any] | None,
    ordinary_lane: bool,
) -> tuple[CandidateChunk, int] | None:
    """The one projection authorised by the Contract-D evidence relation.

    Retrieval proves document relevance, the shared report index proves event
    identity, and an exact raw-row bridge in the already-fetched pool proves
    source identity.  None of the three is allowed to stand in for another.
    """

    approved_requests = frozenset(
        {
            ("after_shares",),
            ("after_ratio",),
            tuple(CURRENT_HOLDING_STATE_FIELDS),
        }
    )
    if (
        not ordinary_lane
        or report_index is None
        or not active_corpus_identity
        or not report_index.identity
        or not requested
        or tuple(requested) not in approved_requests
        or bool(getattr(plan, "comparison", None))
    ):
        return None

    corp_code = str(getattr(plan, "corp_code", "") or "").strip()
    corp_codes = tuple(getattr(plan, "corp_codes", ()) or ())
    reporter = str(getattr(plan, "reporter", "") or "").strip()
    reference_date = exact_reference_date(plan)
    if (
        not corp_code
        or (corp_codes and len(corp_codes) != 1)
        or not canonical_reporter_key(reporter)
        or not reference_date
    ):
        return None

    try:
        selection = report_index.select_report(
            corp_code,
            reporter,
            SELECTOR_EXACT_REFERENCE_DATE,
            reference_date=reference_date,
            active_corpus_identity=active_corpus_identity,
        )
    except Exception:  # noqa: BLE001 - a deterministic guard fails closed
        return None
    if not selection.resolved or selection.selected is None:
        return None
    record = selection.selected

    served = [result for result in results if result.doc_id == record.doc_id]
    if not served:
        return None

    projections = [
        candidate
        for candidate in chunks
        if candidate.chunk_id == record.projection_chunk_id
    ]
    if len(projections) != 1:
        return None
    projection = projections[0]
    if (
        projection.chunk_id in {result.chunk_id for result in results}
        or not _record_projection_matches(projection, record)
    ):
        return None
    event_matches = [
        candidate
        for candidate in chunks
        if _projection_event_matches(candidate, record)
    ]
    if len(event_matches) != 1 or event_matches[0].chunk_id != projection.chunk_id:
        return None

    raw_matches = [
        candidate
        for candidate in chunks
        if candidate.doc_id == record.doc_id
        and str(candidate.chunk.get("chunk_type") or "") == "table"
        and not candidate.chunk.get("projection_type")
        and bool(candidate.chunk.get("is_indexable", True))
        and _same_source_rows(projection.chunk, candidate.chunk)
        and anchor_tier(projection.chunk, candidate.chunk, reporter) == ANCHOR_STRONG
    ]
    if len(raw_matches) != 1:
        return None
    raw = raw_matches[0]

    # The raw rows must identify only the index-selected report projection.
    # Otherwise the bridge merely proves that several projections copied the
    # same evidence and cannot decide which event the question names.
    bridged = [
        candidate
        for candidate in chunks
        if candidate.doc_id == record.doc_id
        and str(candidate.chunk.get("projection_type") or "") == "holding_report"
        and _same_source_rows(candidate.chunk, raw.chunk)
    ]
    if len(bridged) != 1 or bridged[0].chunk_id != projection.chunk_id:
        return None
    return projection, min(result.rank for result in served)


#: Marker written onto a promoted row so coverage-derived evidence stays
#: distinguishable from what retrieval itself ranked.
PROVENANCE_KEY = "holding_evidence_coverage"


def proves_acquisition(chunk: Mapping[str, Any]) -> bool:
    """Whether the frozen parser can prove an acquisition from this row.

    A predicate and nothing more.  The parsed method, date and quantity are
    deliberately discarded here: which transaction answers the question, and
    what its unit price says, are decisions this lane must never make.
    """

    if str(chunk.get("projection_type") or "") != DETAIL_PROJECTION:
        return False
    return acquisition_facts(chunk) is not None


def strict_reporter_identity(chunk: Mapping[str, Any], reporter: str | None) -> bool:
    """Whether this row's holder *is* the bound reporter, on the frozen contract.

    ``reporter_compatible`` accepts containment either way, which is right for
    completing a holding position: a family name and its member describe one
    holder's evidence.  An acquisition row is a different claim -- it says this
    holder bought these shares on this day -- so a neighbouring holder whose
    name merely contains the bound one must not supply it.  The frozen
    ``reporter_matches`` is the identity contract, and it is read here whole.

    An unbound reporter authorizes nothing: the acquisition lane runs only
    after validation proved which holder the question named.
    """

    wanted = str(reporter or "").strip()
    if not wanted:
        return False
    fields = dict(chunk.get("projection_fields") or {})
    holder = str(fields.get("보고자/보유자") or chunk.get("reporter") or "").strip()
    if not holder:
        return False
    return reporter_matches(holder, wanted)


def _acquisition_anchor(
    candidate: CandidateChunk,
    served: Sequence[RetrievalResult],
    by_id: Mapping[str, CandidateChunk],
    reporter: str | None,
) -> tuple[str, int] | None:
    """The firmest served item this proof row completes, if any.

    Same anchoring rule the ordinary lane uses, so a proof row reaches the
    resolver only by completing a document retrieval already served.
    """

    best: tuple[str, int] | None = None
    for result in served:
        served_chunk = by_id.get(result.chunk_id)
        if served_chunk is None:
            continue
        tier = anchor_tier(candidate.chunk, served_chunk.chunk, reporter)
        if tier is None:
            continue
        if best is None or _ANCHOR_ORDER[tier] < _ANCHOR_ORDER[best[0]]:
            best = (tier, result.rank)
    return best


def _recover_acquisition_proof(
    assessment: CoverageAssessment,
    chunks: Sequence[CandidateChunk],
    reporter: str | None,
) -> CoverageAssessment:
    """Append every bounded row that proves an acquisition.  Choose none.

    Ambiguity is the resolver's to see.  ``_minimum_subset`` exists to close a
    field with the fewest rows, and one acquisition row closes this requirement
    exactly as well as five do -- so using it here would silently keep one row
    and drop the rest, turning a contested acquisition into a confident single
    event.  Every legitimately bounded proof row is therefore promoted, and
    whether they are one event, several, or an ambiguity that must fail closed
    is decided downstream where the evidence can actually be compared.

    Breadth is structural, not policed by a cutoff: a proof row must belong to
    an already-served document, to the bound reporter on the frozen identity
    contract, and must anchor to a served row of that document -- which ties
    the set to one reporter's rows for one reference date inside one filing.

    Baseline results are appended to, never reordered, rescored or displaced.
    """

    served = list(assessment.results)
    served_ids = {result.chunk_id for result in served}
    by_id = {candidate.chunk_id: candidate for candidate in chunks}

    selected: list[RetrievalResult] = []
    anchors: list[tuple[str, str, int]] = []
    anchored_count = 0
    for candidate in chunks:
        if candidate.chunk_id in served_ids:
            continue
        if not _eligible(candidate, reporter):
            continue
        if not strict_reporter_identity(candidate.chunk, reporter):
            continue
        if not proves_acquisition(candidate.chunk):
            continue
        anchor = _acquisition_anchor(candidate, served, by_id, reporter)
        if anchor is None:
            continue
        anchored_count += 1
        tier, served_rank = anchor
        selected.append(_synthetic(candidate, len(served) + len(selected) + 1))
        anchors.append((candidate.chunk_id, tier, served_rank))

    requested = tuple([*assessment.requested, ACQUISITION_PROOF])
    if not selected:
        return CoverageAssessment(
            **{
                **assessment.__dict__,
                "requested": requested,
                "unresolved": tuple([*assessment.unresolved, ACQUISITION_PROOF]),
                "remaining_unresolved": tuple(
                    [*assessment.remaining_unresolved, ACQUISITION_PROOF]
                ),
                "status": STATUS_NO_ACQUISITION_PROOF,
                "anchored_candidate_count": (
                    assessment.anchored_candidate_count + anchored_count
                ),
                "evaluated": True,
            }
        )
    return CoverageAssessment(
        **{
            **assessment.__dict__,
            "requested": requested,
            "unresolved": tuple([*assessment.unresolved, ACQUISITION_PROOF]),
            "served_coverage": tuple(
                sorted({*assessment.served_coverage, ACQUISITION_PROOF})
            ),
            # Appended, so every baseline row keeps its object, order and rank.
            "results": tuple([*served, *selected]),
            "selected": tuple([*assessment.selected, *selected]),
            "anchors": tuple([*assessment.anchors, *anchors]),
            "anchored_candidate_count": (
                assessment.anchored_candidate_count + anchored_count
            ),
            "rescue_mode": RESCUE_MODE_ACQUISITION_PROOF,
            "status": STATUS_RESCUED,
            "evaluated": True,
        }
    )


def assess(
    question: str,
    plan: Any,
    chunks: Sequence[CandidateChunk],
    results: Sequence[RetrievalResult],
    *,
    routed_task_type: str | None = None,
    report_index: HoldingReportIndex | None = None,
    active_corpus_identity: Mapping[str, Any] | None = None,
    ordinary_lane: bool = False,
) -> CoverageAssessment:
    """Field coverage, then the acquisition proof an answer field may also need.

    The two are separate passes on purpose.  Ordinary coverage asks which of
    the resolver's state fields the served rows can supply and closes any gap
    with the fewest rows.  The acquisition lane asks a different question --
    can any served row prove an acquisition at all -- and answers it without
    choosing among the rows that can.
    """

    assessment = _assess_field_coverage(
        question,
        plan,
        chunks,
        results,
        routed_task_type=routed_task_type,
        report_index=report_index,
        active_corpus_identity=active_corpus_identity,
        ordinary_lane=ordinary_lane,
    )
    if routed_task_type != "holding_event":
        return assessment
    asked = str(getattr(plan, "raw_query", "") or question or "")
    if ACQUISITION_UNIT_PRICE not in requested_answer_fields(asked):
        return assessment
    reporter = str(getattr(plan, "reporter", "") or "").strip()
    return _recover_acquisition_proof(assessment, chunks, reporter)


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


def _assess_field_coverage(
    question: str,
    plan: Any,
    chunks: Sequence[CandidateChunk],
    results: Sequence[RetrievalResult],
    *,
    routed_task_type: str | None = None,
    report_index: HoldingReportIndex | None = None,
    active_corpus_identity: Mapping[str, Any] | None = None,
    ordinary_lane: bool = False,
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
    rescue_mode: str | None = None
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
                rescue_mode = RESCUE_MODE_SERVED_ANCHOR
                remaining = [n for n in remaining if n not in gained]

    # A raw counterpart in the full pool is not relevance evidence: almost
    # every report projection has one.  Contract D opens one narrower path only
    # after ranked retrieval served the selected document and the shared index
    # independently selected one active issuer/reporter/reference-date event.
    # The exact raw bridge then proves that the selected projection renders the
    # same physical rows.  No query, hydration, reranking or rescore occurs.
    if remaining:
        guarded = _contract_d_candidate(
            plan,
            chunks,
            results,
            requested,
            report_index=report_index,
            active_corpus_identity=active_corpus_identity,
            ordinary_lane=ordinary_lane,
        )
        if guarded is not None:
            candidate, served_rank = guarded
            promoted = _synthetic(
                candidate, len(results) + len(selected) + 1
            )
            all_covered = covered_fields(candidate, promoted, requested)
            if set(requested).issubset(all_covered):
                gained = all_covered.intersection(remaining)
                if gained:
                    selected.append(promoted)
                    seen.add(candidate.chunk_id)
                    contribution[candidate.chunk_id] = gained
                    picked[candidate.chunk_id] = (ANCHOR_STRONG, served_rank)
                    anchored_count += 1
                    rescue_mode = RESCUE_MODE_CONTRACT_D
                    remaining = [name for name in remaining if name not in gained]

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
            "rescue_mode": rescue_mode,
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
