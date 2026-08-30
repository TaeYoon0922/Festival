"""Which holding report a question means, decided from the corpus rather than from search.

"The latest report" is a claim about a corpus, not about a result list.  Asking
retrieval for it returns the best-matching filing, which is a different thing:
the newest report for one holder can rank tenth for a question that names no
date, and answering from rank one would state another holder's position as
theirs.  So this module never consults ranking.  It reads a precomputed index of
every holding report projection in the active corpus and answers by enumeration.

Two properties of the corpus make the naive orderings wrong, and both are
measured rather than assumed.  **62.5%** of holding reports state a reference
date different from the day the filing arrived, so ordering by receipt date
picks the wrong report in the majority of cases; ordering is therefore always by
``reference_date``.  And a filing that restates an earlier reference date arrives
*later*, so a correction must be resolved before dates are compared, never
after.

Where the corpus cannot prove an answer this module declines.  A tied maximum
date, a correction whose finality is unproven, an index built from a different
corpus, a previous state the filing does not record -- each returns a status, not
a plausible report.  Declining is the point: a plausible filing is a different
holder-state on a different date, presented as the requested one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.reasoning.holding_correction_finality import (
    COLLAPSED,
    UNPROVEN,
    CollapseResult,
    HoldingCorrectionFinality,
    load_finality,
)
from app.reasoning.holding_report_relative import (
    ROLE_CHANGE,
    ROLE_CURRENT,
    ROLE_PREVIOUS,
    SELECTOR_EXACT_RECEIPT_DATE,
    SELECTOR_EXACT_REFERENCE_DATE,
    SELECTOR_LATEST,
    SELECTOR_SELECTED_CONTEXT,
)
from app.reasoning.holding_reporter import canonical_reporter_key

#: The artifact this module can read.  A different layout is a different
#: contract and is refused rather than best-effort parsed.
ARTIFACT_SCHEMA_VERSION = "1.0"

#: Where the generator writes, beside the manifest the identity is bound to.
DEFAULT_ARTIFACT_PATH = "data/corpus/holding_report_index.json"

# -- selection outcomes ------------------------------------------------------
#: One eligible report was proven.
RESOLVED = "resolved"
#: The pair exists but nothing satisfies the selector.
NO_MATCH = "no_match"
#: More than one eligible report satisfies it, and no evidence separates them.
AMBIGUOUS = "ambiguous"
#: The index does not describe the active corpus.
STALE_INDEX = "stale_index"
#: The index was built from a source that did not cover the whole corpus.
INCOMPLETE_CORPUS = "incomplete_corpus"
#: A correction touches this timeline and its finality is not proven.
CORRECTION_AMBIGUOUS = "correction_ambiguous"
#: The selector needs a report this module is not the one to choose.
UNSUPPORTED_SELECTOR = "unsupported_selector"
#: No index is loaded at all.
NO_INDEX = "no_index"

#: Selectors this module can execute.  The names are the frozen parser's own,
#: imported rather than restated, so a selector cannot come to mean one thing
#: where it is read and another where it is executed.  ``selected_context`` is
#: deliberately absent: "이번 보고" points at a report the question never names,
#: and inventing one here is precisely the guess the whole module exists to
#: avoid.
_EXECUTABLE_SELECTORS = frozenset(
    {SELECTOR_LATEST, SELECTOR_EXACT_REFERENCE_DATE, SELECTOR_EXACT_RECEIPT_DATE}
)

#: Which fields of the selected report are wanted.  Also the parser's own.
_ROLES = frozenset({ROLE_CURRENT, ROLE_PREVIOUS, ROLE_CHANGE})

#: Cells a filing uses to say "there is nothing here".  They are not zero: a
#: holder who reported nothing before is not a holder who reported zero.
_PLACEHOLDERS = frozenset({"", "-", "—", "–", "…", "N/A", "n/a", "해당사항없음"})


def _clean(value: Any) -> str | None:
    """A stated value, or ``None`` when the filing stated nothing."""

    text = str(value if value is not None else "").strip()
    return None if text in _PLACEHOLDERS else text


@dataclass(frozen=True)
class HoldingReportRecord:
    """One holding report projection, as the corpus states it.

    Values are carried as the filing wrote them.  Nothing is coerced to a
    number here, and an absent field stays absent, so a caller cannot mistake a
    missing previous state for a zero one.
    """

    issuer_corp_code: str
    reporter_key: str
    raw_reporter: str
    doc_id: str
    projection_chunk_id: str
    reference_date: str
    receipt_date: str | None = None
    previous_date: str | None = None
    before_shares: str | None = None
    before_ratio: str | None = None
    change_shares: str | None = None
    change_ratio: str | None = None
    change_direction: str | None = None
    after_shares: str | None = None
    after_ratio: str | None = None
    is_correction: bool = False
    report_nm: str | None = None
    source_table_id: str | None = None
    source_refs: tuple[Mapping[str, Any], ...] = ()

    @property
    def has_previous_state(self) -> bool:
        """Whether this filing records a previous report to read values from.

        All three must agree.  A filing that names no previous report but
        carries previous numbers is describing something this contract cannot
        name, and it is refused rather than reported as a previous state.
        """

        return bool(self.previous_date and self.before_shares and self.before_ratio)

    @property
    def previous_state_is_inconsistent(self) -> bool:
        return not self.previous_date and bool(self.before_shares or self.before_ratio)

    def to_dict(self) -> dict[str, Any]:
        return {
            "issuer_corp_code": self.issuer_corp_code,
            "reporter_key": self.reporter_key,
            "raw_reporter": self.raw_reporter,
            "doc_id": self.doc_id,
            "projection_chunk_id": self.projection_chunk_id,
            "reference_date": self.reference_date,
            "receipt_date": self.receipt_date,
            "previous_date": self.previous_date,
            "before_shares": self.before_shares,
            "before_ratio": self.before_ratio,
            "change_shares": self.change_shares,
            "change_ratio": self.change_ratio,
            "change_direction": self.change_direction,
            "after_shares": self.after_shares,
            "after_ratio": self.after_ratio,
            "is_correction": self.is_correction,
            "report_nm": self.report_nm,
            "source_table_id": self.source_table_id,
            "source_refs": [dict(ref) for ref in self.source_refs],
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "HoldingReportRecord":
        return cls(
            issuer_corp_code=str(row["issuer_corp_code"]),
            reporter_key=str(row["reporter_key"]),
            raw_reporter=str(row.get("raw_reporter") or ""),
            doc_id=str(row["doc_id"]),
            projection_chunk_id=str(row["projection_chunk_id"]),
            reference_date=str(row["reference_date"]),
            receipt_date=_clean(row.get("receipt_date")),
            previous_date=_clean(row.get("previous_date")),
            before_shares=_clean(row.get("before_shares")),
            before_ratio=_clean(row.get("before_ratio")),
            change_shares=_clean(row.get("change_shares")),
            change_ratio=_clean(row.get("change_ratio")),
            change_direction=_clean(row.get("change_direction")),
            after_shares=_clean(row.get("after_shares")),
            after_ratio=_clean(row.get("after_ratio")),
            is_correction=bool(row.get("is_correction")),
            report_nm=_clean(row.get("report_nm")),
            source_table_id=_clean(row.get("source_table_id")),
            source_refs=tuple(dict(ref) for ref in (row.get("source_refs") or ())),
        )


@dataclass(frozen=True)
class ReportSelection:
    """What the index concluded, and enough of why to audit it."""

    status: str
    selected: HoldingReportRecord | None = None
    candidates: tuple[HoldingReportRecord, ...] = ()
    reporter_key: str = ""
    selector: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.status == RESOLVED and self.selected is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "selector": self.selector,
            "reporter_key": self.reporter_key,
            "candidate_count": len(self.candidates),
            "selected": self.selected.to_dict() if self.selected else None,
            "detail": dict(self.detail),
        }


class HoldingReportIndex:
    """Every holding report in one corpus, keyed by issuer and holder.

    The index is inert about meaning: it enumerates and compares dates.  What a
    question meant by "latest" or "previous" is decided upstream, and whether an
    answer may be given at all is decided by the statuses returned here.
    """

    def __init__(
        self,
        records: Iterable[HoldingReportRecord],
        *,
        identity: Mapping[str, Any] | None = None,
        complete: bool = False,
        correction_finality_available: bool = False,
        correction_finality: HoldingCorrectionFinality | None = None,
    ) -> None:
        self.identity = dict(identity or {})
        self.complete = bool(complete)
        #: Whether a frozen source can prove which member of a correction chain
        #: is final.  False means correction-bearing timelines decline.  In-memory
        #: callers set it to assert that the records they built are already
        #: collapsed; the generated artifact never sets it, so production
        #: finality arrives only as ``correction_finality`` below.
        self.correction_finality_available = bool(correction_finality_available)
        #: P0-A's materialized holding correction groups.  Accepted only when it
        #: describes this same corpus: a source built from another corpus may
        #: name a final document this index does not hold, or miss a chain it
        #: does, and either way it would collapse the wrong reports.
        self.correction_finality: HoldingCorrectionFinality | None = None
        self.correction_finality_status = "absent"
        if correction_finality is not None:
            if not correction_finality.usable:
                self.correction_finality_status = "unusable"
            elif not correction_finality.matches_identity(self.identity):
                self.correction_finality_status = "identity_mismatch"
            else:
                self.correction_finality = correction_finality
                self.correction_finality_status = "attached"
        by_pair: dict[tuple[str, str], list[HoldingReportRecord]] = {}
        seen: set[tuple[str, str]] = set()
        duplicates: list[str] = []
        for record in records:
            if not record.reporter_key or not record.issuer_corp_code:
                # A projection with no holder cannot answer a question about a
                # holder.  It is dropped from the lookup, never matched to one.
                continue
            pk = (record.projection_chunk_id, record.reference_date)
            if pk in seen:
                duplicates.append(record.projection_chunk_id)
                continue
            seen.add(pk)
            by_pair.setdefault(
                (record.issuer_corp_code, record.reporter_key), []
            ).append(record)
        # Deterministic serialization order.  Technical fields break ties only
        # for reproducibility of this list; selection below never consults them.
        for items in by_pair.values():
            items.sort(key=lambda r: (r.reference_date, r.doc_id, r.projection_chunk_id))
        self._by_pair = {pair: tuple(items) for pair, items in by_pair.items()}
        self.duplicate_chunk_ids = tuple(duplicates)
        # Which timelines a document reports for.  A correction chain may only
        # collapse a timeline when every member of it stays inside that one
        # issuer and holder, and this is what proves it.
        pairs_by_doc: dict[str, set[tuple[str, str]]] = {}
        for pair, items in self._by_pair.items():
            for record in items:
                pairs_by_doc.setdefault(record.doc_id, set()).add(pair)
        self._pairs_by_doc = {
            doc: frozenset(pairs) for doc, pairs in pairs_by_doc.items()
        }

    # ------------------------------------------------------------- inspection
    @property
    def pair_count(self) -> int:
        return len(self._by_pair)

    @property
    def record_count(self) -> int:
        return sum(len(v) for v in self._by_pair.values())

    def enumerate_reports(
        self, issuer_corp_code: str, reporter: str
    ) -> tuple[HoldingReportRecord, ...]:
        """Every report this corpus holds for one issuer and one holder.

        The holder is compared on the frozen canonical key and nothing else --
        no containment, no alias table -- so ``영풍`` never answers for
        ``영풍정밀``.
        """

        key = canonical_reporter_key(reporter)
        if not key or not issuer_corp_code:
            return ()
        return self._by_pair.get((str(issuer_corp_code), key), ())

    # -------------------------------------------------------------- selection
    def select_report(
        self,
        issuer_corp_code: str,
        reporter: str,
        selector: str,
        *,
        reference_date: str | None = None,
        receipt_date: str | None = None,
        active_corpus_identity: Mapping[str, Any] | None = None,
    ) -> ReportSelection:
        """The one report a selector names, or why there isn't one."""

        key = canonical_reporter_key(reporter)
        base = {"selector": selector, "reporter_key": key}

        if active_corpus_identity is not None and not self.matches_corpus(
            active_corpus_identity
        ):
            return ReportSelection(
                STALE_INDEX, **base,
                detail={"index_identity": self.identity,
                        "active_identity": dict(active_corpus_identity)},
            )
        if not self.complete:
            # An index built from part of the corpus cannot say "latest": the
            # report it has not seen is exactly the one that would change it.
            return ReportSelection(INCOMPLETE_CORPUS, **base)
        if selector not in _EXECUTABLE_SELECTORS:
            return ReportSelection(
                UNSUPPORTED_SELECTOR, **base,
                detail={"reason": "selector names no report this index may choose"},
            )
        if not key:
            return ReportSelection(NO_MATCH, **base,
                                   detail={"reason": "reporter has no canonical key"})

        candidates = self.enumerate_reports(issuer_corp_code, reporter)
        if not candidates:
            return ReportSelection(NO_MATCH, **base)

        # Correction finality is resolved before dates are compared, because a
        # correction restating an earlier date arrives later: comparing first
        # would rank the restatement as its own report.  Five chains in this
        # corpus have exactly that shape, with unrelated reports filed in
        # between, so the order of these two steps is not a formality.
        collapse = self._collapse(candidates)
        if not collapse.proven:
            return ReportSelection(
                CORRECTION_AMBIGUOUS, **base, candidates=candidates,
                detail={
                    "correction_records": sum(
                        1 for r in candidates if r.is_correction),
                    **dict(collapse.detail),
                },
            )
        candidates = tuple(collapse.eligible)
        # Recorded so a resolved answer can be audited back to the chain that
        # removed the filings it is not based on.
        collapsed = ({"superseded_doc_ids": list(collapse.superseded_doc_ids)}
                     if collapse.superseded_doc_ids else {})

        if selector == SELECTOR_LATEST:
            newest = max(record.reference_date for record in candidates)
            matches = [r for r in candidates if r.reference_date == newest]
            return self._unique(matches, base, candidates,
                                {"reference_date": newest, **collapsed})
        if selector == SELECTOR_EXACT_REFERENCE_DATE:
            wanted = _normalize_date(reference_date)
            if not wanted:
                return ReportSelection(NO_MATCH, **base, candidates=candidates,
                                       detail={"reason": "no reference date given"})
            matches = [r for r in candidates if r.reference_date == wanted]
            return self._unique(matches, base, candidates,
                                {"reference_date": wanted, **collapsed})
        wanted = _normalize_date(receipt_date)
        if not wanted:
            return ReportSelection(NO_MATCH, **base, candidates=candidates,
                                   detail={"reason": "no receipt date given"})
        matches = [r for r in candidates if r.receipt_date == wanted]
        return self._unique(matches, base, candidates,
                            {"receipt_date": wanted, **collapsed})

    def _collapse(self, candidates: Sequence[HoldingReportRecord]) -> CollapseResult:
        """Remove the reports P0-A proved superseded, or refuse to proceed.

        A timeline with no correction passes through untouched, so a corpus
        without a finality source keeps answering every question it could
        answer before.  A timeline *with* one is answered only when P0-A proved
        which filing supersedes which -- never by ignoring the correction, which
        is the one outcome that silently returns a withdrawn holding.
        """

        if self.correction_finality is not None:
            return self.correction_finality.collapse(
                candidates, pair_lookup=self._pairs_by_doc.get
            )
        if not any(record.is_correction for record in candidates):
            return CollapseResult(COLLAPSED, tuple(candidates))
        if self.correction_finality_available:
            # The caller states these records are already the final ones.  Used
            # by fixtures that build a collapsed timeline directly; the
            # generated artifact never asserts it.
            return CollapseResult(COLLAPSED, tuple(candidates))
        return CollapseResult(UNPROVEN, detail={
            "reason": "correction finality is not provable from a frozen source "
                      "for this corpus",
            "finality_source": self.correction_finality_status,
        })

    @staticmethod
    def _unique(matches, base, candidates, detail) -> ReportSelection:
        """One match resolves; several decline.

        Nothing here breaks a tie.  Receipt date, receipt number, document id
        and file order all *could* order these records, and every one of them
        would be inventing a reason to prefer one filing's numbers over
        another's.
        """

        if not matches:
            return ReportSelection(NO_MATCH, **base, candidates=candidates,
                                   detail=detail)
        distinct = {record.doc_id for record in matches}
        if len(matches) > 1 and len(distinct) > 1:
            return ReportSelection(
                AMBIGUOUS, **base, candidates=tuple(matches),
                detail={**detail, "doc_ids": sorted(distinct)},
            )
        if len(matches) > 1:
            return ReportSelection(
                AMBIGUOUS, **base, candidates=tuple(matches),
                detail={**detail, "reason": "one document, several projections",
                        "doc_ids": sorted(distinct)},
            )
        return ReportSelection(RESOLVED, selected=matches[0], **base,
                               candidates=candidates, detail=detail)

    # --------------------------------------------------------------- identity
    def matches_corpus(self, active: Mapping[str, Any]) -> bool:
        """Whether this index describes the corpus now in use.

        Every recorded identity field must agree.  A mismatch means the index
        may be describing filings the corpus no longer has, or missing ones it
        gained -- and "latest" is exactly the answer that would change.
        """

        for key, value in self.identity.items():
            if key in _NON_IDENTITY_FIELDS:
                continue
            if str(active.get(key, "")) != str(value):
                return False
        return True


#: Recorded for provenance, but not part of the equality that gates use: they
#: describe the generation run rather than the corpus it read.
_NON_IDENTITY_FIELDS = frozenset(
    {"generated_at", "generated_commit", "artifact_schema_version"}
)


def _normalize_date(value: Any) -> str | None:
    digits = "".join(ch for ch in str(value or "") if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else None


# -- which fields of the selected report ------------------------------------
#: The requested fields are stated by the selected filing.
PROJECTION_RESOLVED = "resolved"
#: The filing names no previous report to read a previous state from.
PREVIOUS_UNAVAILABLE = "previous_unavailable"
#: The filing carries previous numbers but names no previous report.
PREVIOUS_INCONSISTENT = "previous_inconsistent"
#: The filing states no change.
CHANGE_UNAVAILABLE = "change_unavailable"
#: The filing states no current holding.
CURRENT_UNAVAILABLE = "current_unavailable"
#: The role is not one of the three a holding filing distinguishes.
UNSUPPORTED_ROLE = "unsupported_role"


@dataclass(frozen=True)
class RoleProjection:
    """The fields one role asks for, read from a single filing.

    Every value here comes from the filing that was selected.  ``previous``
    especially: it is the previous state *that filing records*, never another
    filing's current state, and never a difference computed from a neighbour.
    """

    status: str
    role: str
    values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return self.status == PROJECTION_RESOLVED


def project_role(record: HoldingReportRecord, role: str) -> RoleProjection:
    """Read one role's fields from one filing, or say why they are not there.

    A holding filing states three things about the same holder, and each is a
    different set of its own cells.  Choosing a *different filing* for the
    previous role would answer with a state the question did not ask about, so
    no lookup happens here -- only reading.
    """

    if role == ROLE_CURRENT:
        if not (record.after_shares or record.after_ratio):
            return RoleProjection(CURRENT_UNAVAILABLE, role)
        return RoleProjection(PROJECTION_RESOLVED, role, {
            "reference_date": record.reference_date,
            "shares": record.after_shares,
            "ratio": record.after_ratio,
        })

    if role == ROLE_PREVIOUS:
        if record.previous_state_is_inconsistent:
            # Observed in this corpus: no previous report named, yet previous
            # numbers present.  Reporting them would attribute a holding to a
            # filing the corpus cannot identify.
            return RoleProjection(PREVIOUS_INCONSISTENT, role)
        if not record.has_previous_state:
            # A first report has no previous state.  It has no zero either.
            return RoleProjection(PREVIOUS_UNAVAILABLE, role)
        return RoleProjection(PROJECTION_RESOLVED, role, {
            "reference_date": record.previous_date,
            "shares": record.before_shares,
            "ratio": record.before_ratio,
        })

    if role == ROLE_CHANGE:
        if not (record.change_shares or record.change_ratio):
            return RoleProjection(CHANGE_UNAVAILABLE, role)
        return RoleProjection(PROJECTION_RESOLVED, role, {
            "shares": record.change_shares,
            "ratio": record.change_ratio,
            # Absent when the filing wrote no signed change; not inferred from
            # comparing the two states, which would be a computation this
            # module is not entitled to make.
            "direction": record.change_direction,
        })

    return RoleProjection(UNSUPPORTED_ROLE, role)


@dataclass(frozen=True)
class ReportExecution:
    """Whether a parsed report-relative intent could actually be carried out.

    Readiness lives here rather than on the parsed intent.  Whether "latest" can
    be answered depends on the corpus in front of it -- whether an index is
    loaded, whether it matches, whether *this* holder's timeline carries an
    unproven correction or a tied date -- and none of that is knowable when the
    question is read.  The parse says what was asked; this says whether it can
    be answered, and no global state connects them.
    """

    status: str
    selection: ReportSelection
    projection: RoleProjection | None = None

    @property
    def executable(self) -> bool:
        """Whether one report *and* the fields the role asked for were proven."""

        return (
            self.status == RESOLVED
            and self.projection is not None
            and self.projection.resolved
        )

    @property
    def record(self) -> HoldingReportRecord | None:
        return self.selection.selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "executable": self.executable,
            "selection": self.selection.to_dict(),
            "projection": (
                {
                    "status": self.projection.status,
                    "role": self.projection.role,
                    "values": dict(self.projection.values),
                }
                if self.projection
                else None
            ),
        }


def execute_report_relative(
    intent: Mapping[str, Any] | Any,
    *,
    index: HoldingReportIndex | None,
    issuer_corp_code: str,
    reporter: str,
    reference_date: str | None = None,
    receipt_date: str | None = None,
    active_corpus_identity: Mapping[str, Any] | None = None,
) -> ReportExecution:
    """Carry out a parsed report-relative intent against one corpus.

    ``intent`` is the frozen parser's own representation -- the mapping stored
    at ``plan.evidence["holding_report_relative"]``, or the intent object
    itself.  Its meaning is read, never widened: a ``selected_context``
    selector still names no report, and this function will not promote it to
    "the latest one" on the grounds that a latest one happens to exist.
    """

    payload = intent.to_dict() if hasattr(intent, "to_dict") else dict(intent or {})
    selector = str(payload.get("selector") or "")
    role = str(payload.get("projection_role") or ROLE_CURRENT)
    base = {"selector": selector, "reporter_key": canonical_reporter_key(reporter)}

    if index is None:
        return ReportExecution(NO_INDEX, ReportSelection(NO_INDEX, **base))
    if selector == SELECTOR_SELECTED_CONTEXT:
        # "이번 보고" / a standalone "직전보고" names a report only if one is
        # already in hand.  B.2 does not have one and must not invent one.
        return ReportExecution(
            UNSUPPORTED_SELECTOR,
            ReportSelection(
                UNSUPPORTED_SELECTOR, **base,
                detail={"reason": "selector names a report only a prior "
                                  "selection could supply"},
            ),
        )
    if role not in _ROLES:
        # No report is chosen for a role naming no fields, so the selection is
        # reported as not attempted rather than as a selector failure.
        return ReportExecution(
            UNSUPPORTED_ROLE,
            ReportSelection(UNSUPPORTED_SELECTOR, **base,
                            detail={"reason": f"unknown projection role {role!r}"}),
            projection=RoleProjection(UNSUPPORTED_ROLE, role),
        )

    selection = index.select_report(
        issuer_corp_code,
        reporter,
        selector,
        reference_date=reference_date,
        receipt_date=receipt_date,
        active_corpus_identity=active_corpus_identity,
    )
    if not selection.resolved:
        return ReportExecution(selection.status, selection)

    projection = project_role(selection.selected, role)
    status = RESOLVED if projection.resolved else projection.status
    return ReportExecution(status, selection, projection)


def load_index(
    path: str | Path, *, finality_path: str | Path | None = None
) -> HoldingReportIndex | None:
    """Read a generated index, or ``None`` when there is nothing usable.

    A malformed or differently-versioned artifact yields ``None`` rather than a
    partially populated index: half an enumeration answers "latest" wrongly and
    confidently.

    ``finality_path`` attaches P0-A's materialized correction groups.  A source
    that is missing, malformed, or built from another corpus is simply not
    attached, and correction-bearing timelines go on declining -- the one thing
    that never happens is proceeding with the corrections ignored.
    """

    location = Path(path)
    if not location.is_file():
        return None
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    identity = payload.get("identity")
    rows = payload.get("records")
    if not isinstance(identity, Mapping) or not isinstance(rows, Sequence):
        return None
    if str(identity.get("artifact_schema_version") or "") != ARTIFACT_SCHEMA_VERSION:
        return None
    try:
        records = [HoldingReportRecord.from_dict(row) for row in rows]
    except (KeyError, TypeError, ValueError):
        return None
    return HoldingReportIndex(
        records,
        identity=identity,
        complete=bool(payload.get("complete")),
        correction_finality_available=bool(
            payload.get("correction_finality_available")
        ),
        correction_finality=(
            load_finality(finality_path) if finality_path is not None else None
        ),
    )
