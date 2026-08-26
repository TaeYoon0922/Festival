"""Deterministic corporate event timeline over contract disclosures.

Where :mod:`app.reasoning.correction_graph` resolves *one filing against its own
corrections*, this module resolves *different filings against each other*: the
conclusion of a contract, any later filing about that same contract, and its
termination.  No language model takes part.  Every edge records the rule that
produced it, the fields that corroborated it, the fields that conflicted, and a
resolution status of ``resolved``, ``ambiguous``, or ``unresolved``.

The corpus audit fixed the scope.  Exactly two families carry the structured
fields a deterministic rule can stand on:

``supply_contract``
    exchange ``단일판매ㆍ공급계약체결`` / ``단일판매ㆍ공급계약해지``.  A
    termination states ``관련공시`` -- a dated list of the filings that belong to
    the contract being terminated -- plus ``계약상대``, ``-해지계약명``,
    ``시작일``/``종료일`` and the terminated amount.

``treasury_trust_contract``
    major ``자기주식취득신탁계약체결결정`` / ``자기주식취득신탁계약해지결정``.  A
    termination carries no reference field at all, but it restates the
    terminated contract's ``시작일``/``종료일`` exactly, which is unique per
    company.

There is no ``계약변경`` disclosure type in the corpus: zero filings match it.
"Change" appears either as a ``[기재정정]`` correction -- which belongs to P0-A
and is never re-modelled here -- or as a further ``체결`` filing that the
termination's ``관련공시`` lists alongside the first one.  The second case is why
an event is a multi-member timeline rather than a linked list, and why a later
conclusion carries the *member role* ``contract_update`` without its disclosure
subtype ever being rewritten.

P0-A is used as the canonicalization layer underneath all of this and is never
reimplemented: a resolved correction group contributes its latest document as the
canonical representation while root, group id, and chain provenance are kept; an
ambiguous or unresolved correction is never treated as a verified latest and is
recorded as such on the member row it produces.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from app.reasoning.corporate_event import (
    AMBIGUOUS,
    EVENT_FAMILIES,
    FAMILY_SUPPLY_CONTRACT,
    FAMILY_TREASURY_TRUST,
    LIFECYCLE_OPEN,
    LIFECYCLE_STATUSES,
    LIFECYCLE_TERMINATED,
    MEMBER_ROLES,
    RELATION_BELONGS_TO_EVENT,
    RELATION_TERMINATES_EVENT,
    RELATION_TYPES,
    RESOLVED,
    ROLE_CONTRACT,
    ROLE_CONTRACT_UPDATE,
    ROLE_TERMINATION,
    UNRESOLVED,
    CorporateEvent,
    CorporateEventMember,
    CorporateEventRelation,
    CorporateEventState,
    ambiguous_event_anchor,
    deterministic_event_id as event_id,
    deterministic_relation_id as relation_id,
    unresolved_event_anchor,
)

# These helpers are P0-A's frozen table/date/label vocabulary.  P0-B reads the
# same frozen payloads, so it reuses them rather than keeping a second copy that
# could drift from the parser output.
from app.reasoning.correction_graph import (
    CorrectionGraphUnavailable,
    DisclosureRecord,
    _cells,
    _compact,
    _iso_date,
    _label_key,
    _text,
)


class CorporateEventGraphUnavailable(RuntimeError):
    """The event graph cannot be read right now.

    Raised only for conditions a caller may reasonably degrade past: db/007 has
    not been applied yet, or the database is unreachable.  Every other error
    keeps its own type so a real defect is never disguised as a missing
    migration.
    """


#: The two families the corpus supports deterministically.

#: What a document does inside its event.  ``contract_update`` is a role, not a
#: reclassification: the disclosure stays a ``체결`` filing.

#: Whether the contract is still running as far as the corpus can tell.

#: Relation types.  ``MODIFIES`` is deliberately absent: no corpus field
#: distinguishes "amends" from "further filing about", so claiming it would be
#: an invention rather than a reading.

#: Rule names recorded on every event, member, and relation.
SOURCE_RELATED_REFERENCE = "related_reference_corroborated"
SOURCE_RELATED_REFERENCE_DISCRIMINATED = "related_reference_discriminated"
SOURCE_CONTRACT_PERIOD_KEY = "contract_period_key"
SOURCE_SINGLE_DOCUMENT = "single_document"
SOURCE_CORRECTION_GROUP = "correction_group"
SOURCE_NO_RELATED_REFERENCE = "no_related_reference"
SOURCE_NO_CONTRACT_REFERENCE = "no_contract_reference"
SOURCE_REFERENCE_NOT_IN_CORPUS = "related_reference_not_in_corpus"
SOURCE_PERIOD_KEY_NOT_IN_CORPUS = "contract_period_not_in_corpus"
SOURCE_UNCORROBORATED = "reference_uncorroborated"
SOURCE_MULTIPLE_CANDIDATES = "multiple_candidates"
SOURCE_NO_ADMISSIBLE_CANDIDATE = "no_admissible_candidate"
SOURCE_MISSING_PERIOD_KEY = "termination_without_period_key"
#: One reference date still maps to more than one distinct contract after
#: corroboration.  Never collapsed into a multi-member lifecycle: several
#: candidates behind *one* reference is a tie, whereas several *references* are
#: a lifecycle.  Telling those apart is the whole point of resolving per
#: reference rather than over a flattened candidate pool.
SOURCE_REFERENCE_TIE = "reference_candidate_tie"
#: The referenced filing is in the corpus and contradicts this termination on an
#: identity field.  Recorded as ambiguous rather than unresolved: a candidate was
#: found, it just cannot be confirmed as the same contract.
SOURCE_IDENTITY_CONFLICT = "identity_conflict"

#: Per-reference outcomes, aggregated into the termination's own status.
_REF_RESOLVED = "resolved"
_REF_EXTERNAL = "outside_corpus"
_REF_TIE = "tie"
_REF_UNCORROBORATED = "uncorroborated"
_REF_CONFLICT = "identity_conflict"

_CONFIDENCE = {
    SOURCE_RELATED_REFERENCE: 0.95,
    SOURCE_RELATED_REFERENCE_DISCRIMINATED: 0.90,
    SOURCE_CONTRACT_PERIOD_KEY: 0.95,
    SOURCE_SINGLE_DOCUMENT: 0.0,
    # The grouping is P0-A's, already resolved there; P0-B only carries it.
    SOURCE_CORRECTION_GROUP: 0.90,
}

#: Confidence is reduced by this much when a member of the event sits in a
#: correction group P0-A could not resolve.  The link itself still stands on its
#: own evidence; the penalty records that one document's version history does
#: not.
AMBIGUOUS_CORRECTION_PENALTY = 0.10

# --------------------------------------------------------------------- labels

#: ``관련공시`` on a termination, ``※관련공시`` on a conclusion.  Both normalize to
#: the same label key once the leading marker is dropped.
_RELATED_LABELS = ("관련공시", "※관련공시")
_COUNTERPARTY_LABELS = ("계약상대", "계약상대방")
_SUBJECT_LABELS = ("-체결계약명", "체결계약명", "-해지계약명", "해지계약명")
_AMOUNT_LABELS = ("계약금액(원)", "해지금액(원)", "계약금액총액(원)")
_PERIOD_START_LABELS = ("시작일",)
_PERIOD_END_LABELS = ("종료일",)
_CONTRACT_DATE_LABELS = ("계약(수주)일자", "계약(수주)일")
_TERMINATION_DATE_LABELS = ("해지일자", "해지예정일자")
_TERMINATION_REASON_LABELS = ("해지주요사유", "해지목적")
_RECENT_REVENUE_LABELS = ("최근매출액(원)",)
_TRUST_INSTITUTION_LABELS = ("계약체결기관", "해지기관")
_TRUST_BEFORE_LABELS = ("해지전",)
_TRUST_AFTER_LABELS = ("해지후",)

#: One ``YYYY-MM-DD <report name>`` entry of a ``관련공시`` list.  The lookahead
#: stops each title at the next date so a multi-entry field splits exactly.
_RELATED_ENTRY = re.compile(
    r"(\d{4})\s*[-./]\s*(\d{1,2})\s*[-./]\s*(\d{1,2})\s*([^0-9]*?)(?=\s*\d{4}\s*[-./]\s*\d{1,2}\s*[-./]\s*\d{1,2}|$)"
)

#: A ``관련공시`` entry only counts as a contract reference when its own title
#: names a supply-contract conclusion.  ``기타경영사항``, ``채무인수결정``,
#: ``조회공시``, ``전망(공정공시)`` and ``풍문 해명`` all appear in real reference
#: lists and are not this contract's filings.
_CONTRACT_REFERENCE_REQUIRED = ("단일판매", "공급계약")
_CONTRACT_REFERENCE_FORBIDDEN = ("해지",)

#: Fields whose disagreement means the two filings describe different contracts.
IDENTITY_FIELDS = ("counterparty", "subject", "period_start")
#: Fields that legitimately move over a contract's life.  The audit found a
#: termination amount of 362,307,038,000 against a conclusion amount of
#: 351,882,914,000 for one and the same 대우건설 contract, so a difference here
#: is recorded as a change and never as a conflict.
MUTABLE_FIELDS = ("amount", "period_end", "recent_revenue")


def _norm_value(value: Any) -> str:
    """Whitespace-insensitive, case-insensitive comparison form.

    Deliberately shallow.  Legal-form tokens are *not* stripped: real
    counterparty values include lists such as
    ``한국전력기술㈜, 금호건설㈜, LS일렉트릭㈜`` where aggressive normalization would
    start merging distinct parties.
    """

    text = _compact(value)
    if text in {"", "-", "--", "해당사항없음"}:
        return ""
    return text.casefold()


def _digits(value: Any) -> str:
    """Amounts compare on their digits only, ignoring separators and units."""

    text = re.sub(r"[^0-9]", "", _text(value))
    return text


# ---------------------------------------------------------------- extraction


def classify_contract_document(record: DisclosureRecord) -> tuple[str, str] | None:
    """Return ``(family, kind)`` for a v1 contract filing, else None.

    ``kind`` is ``conclusion`` or ``termination``.  Classification reads the
    frozen metadata only.  ``doc_subtype`` carries it for exchange filings; major
    filings have no subtype at all in this corpus, so their ``report_nm`` is the
    only available discriminator.
    """

    subtype = _compact(record.doc_subtype)
    if record.doc_group == "exchange":
        if subtype == "단일판매공급계약체결":
            return FAMILY_SUPPLY_CONTRACT, "conclusion"
        if subtype == "단일판매공급계약해지":
            return FAMILY_SUPPLY_CONTRACT, "termination"
        return None
    if record.doc_group == "major":
        report = _compact(record.report_nm)
        if "자기주식취득신탁계약체결" in report:
            return FAMILY_TREASURY_TRUST, "conclusion"
        if "자기주식취득신탁계약해지" in report:
            return FAMILY_TREASURY_TRUST, "termination"
    return None


@dataclass(frozen=True)
class RelatedReference:
    """One ``YYYY-MM-DD <title>`` entry of a ``관련공시`` list."""

    reference_date: str
    reference_title: str
    is_contract_reference: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_date": self.reference_date,
            "reference_title": self.reference_title,
            "is_contract_reference": self.is_contract_reference,
        }


def parse_related_disclosures(value: Any) -> tuple[RelatedReference, ...]:
    """Split a ``관련공시`` field and mark which entries are contract filings.

    Unrelated entries are kept rather than dropped so the evidence shows what was
    read and rejected, but only entries whose own title names a supply-contract
    conclusion are ever followed.
    """

    text = _text(value).strip()
    if not text or text == "-":
        return ()
    references: list[RelatedReference] = []
    for year, month, day, title in _RELATED_ENTRY.findall(text):
        date = "%04d-%02d-%02d" % (int(year), int(month), int(day))
        clean = re.sub(r"\s+", " ", title).strip(" .,·ㆍ")
        compact = _compact(clean)
        is_contract = all(
            token in compact for token in _CONTRACT_REFERENCE_REQUIRED
        ) and not any(token in compact for token in _CONTRACT_REFERENCE_FORBIDDEN)
        references.append(RelatedReference(date, clean, is_contract))
    return tuple(references)


@dataclass(frozen=True)
class ContractDocument:
    """The structured fields one contract filing states about itself.

    Only labels the corpus audit found are read.  Everything is optional: a real
    conclusion filing can carry no counterparty and no amount at all, so a
    missing value is always neutral and never evidence of a different contract.
    """

    doc_id: str
    corp_code: str
    event_family: str
    kind: str
    rcept_dt: str | None = None
    counterparty: str | None = None
    subject: str | None = None
    amount: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    contract_date: str | None = None
    termination_date: str | None = None
    termination_reason: str | None = None
    recent_revenue: str | None = None
    institution: str | None = None
    balance_before: str | None = None
    balance_after: str | None = None
    related: tuple[RelatedReference, ...] = ()
    source_table_id: str | None = None

    @property
    def is_termination(self) -> bool:
        return self.kind == "termination"

    @property
    def event_date(self) -> str | None:
        """The date this filing puts on the contract, not on its own receipt."""

        if self.is_termination:
            return self.termination_date or self.rcept_dt
        return self.contract_date or self.period_start or self.rcept_dt

    @property
    def contract_references(self) -> tuple[RelatedReference, ...]:
        return tuple(item for item in self.related if item.is_contract_reference)

    @property
    def period_key(self) -> tuple[str, str, str] | None:
        """``(corp_code, 시작일, 종료일)`` -- the treasury trust matching key."""

        if not self.period_start or not self.period_end:
            return None
        return (self.corp_code, self.period_start, self.period_end)

    def identity(self) -> dict[str, str]:
        return {
            "counterparty": _norm_value(self.counterparty),
            "subject": _norm_value(self.subject),
            "period_start": _norm_value(self.period_start),
        }

    def mutable(self) -> dict[str, str]:
        return {
            "amount": _digits(self.amount),
            "period_end": _norm_value(self.period_end),
            "recent_revenue": _digits(self.recent_revenue),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "corp_code": self.corp_code,
            "event_family": self.event_family,
            "kind": self.kind,
            "rcept_dt": self.rcept_dt,
            "counterparty": self.counterparty,
            "subject": self.subject,
            "amount": self.amount,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "contract_date": self.contract_date,
            "termination_date": self.termination_date,
            "termination_reason": self.termination_reason,
            "recent_revenue": self.recent_revenue,
            "institution": self.institution,
            "balance_before": self.balance_before,
            "balance_after": self.balance_after,
            "related": [item.to_dict() for item in self.related],
            "source_table_id": self.source_table_id,
        }


def _first(values: Mapping[str, list[str]], labels: Sequence[str]) -> str | None:
    for label in labels:
        for value in values.get(label, ()):
            text = value.strip()
            if text and text not in {"-", "--"}:
                return text
    return None


def extract_contract_document(
    record: DisclosureRecord, tables: Iterable[Mapping[str, Any]]
) -> ContractDocument | None:
    """Read one contract filing's own structured fields.

    ``tables`` accepts the frozen table payloads unchanged: each mapping needs a
    ``table_id`` plus ``rows`` or ``table_rows``.  Only labelled cells are read;
    no free text is interpreted.
    """

    classified = classify_contract_document(record)
    if classified is None:
        return None
    family, kind = classified

    values: dict[str, list[str]] = {}
    table_id: str | None = None
    for table in tables or ():
        rows = table.get("rows")
        if rows is None:
            rows = table.get("table_rows")
        for row in rows or ():
            cells = [cell.strip() for cell in _cells(row)]
            if len(cells) < 2:
                continue
            label = _label_key(cells[-2])
            if not label:
                continue
            value = cells[-1].strip()
            values.setdefault(label, []).append(value)
            if table_id is None and label in _RELATED_LABELS:
                table_id = _text(table.get("table_id")) or None

    if not values:
        return None
    return ContractDocument(
        doc_id=record.doc_id,
        corp_code=record.corp_code,
        event_family=family,
        kind=kind,
        rcept_dt=record.rcept_dt,
        counterparty=_first(values, _COUNTERPARTY_LABELS),
        subject=_first(values, _SUBJECT_LABELS),
        amount=_first(values, _AMOUNT_LABELS),
        period_start=_iso_date(_first(values, _PERIOD_START_LABELS)),
        period_end=_iso_date(_first(values, _PERIOD_END_LABELS)),
        contract_date=_iso_date(_first(values, _CONTRACT_DATE_LABELS)),
        termination_date=_iso_date(_first(values, _TERMINATION_DATE_LABELS)),
        termination_reason=_first(values, _TERMINATION_REASON_LABELS),
        recent_revenue=_first(values, _RECENT_REVENUE_LABELS),
        institution=_first(values, _TRUST_INSTITUTION_LABELS),
        balance_before=_first(values, _TRUST_BEFORE_LABELS),
        balance_after=_first(values, _TRUST_AFTER_LABELS),
        related=parse_related_disclosures(_first(values, _RELATED_LABELS)),
        source_table_id=table_id,
    )


# ----------------------------------------------------------- canonicalization


@dataclass(frozen=True)
class CanonicalDocument:
    """A document seen through P0-A.

    ``canonical_doc_id`` is the document to compare and present.  For a resolved
    correction group that is the group's latest valid filing; for an ambiguous or
    unresolved correction it is the document itself, because nothing has verified
    it as anybody's latest.
    """

    doc_id: str
    canonical_doc_id: str
    correction_group_id: str | None = None
    root_doc_id: str | None = None
    correction_resolution_status: str | None = None
    correction_chain: tuple[str, ...] = ()
    is_correction: bool = False

    @property
    def has_unverified_correction(self) -> bool:
        return self.correction_resolution_status in {AMBIGUOUS, UNRESOLVED}

    @property
    def logical_key(self) -> str:
        """The one contract this filing is a version of.

        Mirrors :attr:`CorporateEventMember.logical_key` so candidate counting
        and event identity agree on what "one contract" means.  An ambiguous or
        unresolved correction is its own logical contract, because nothing has
        established which filing it is a version of.
        """

        if self.correction_group_id and self.correction_resolution_status == RESOLVED:
            return self.correction_group_id
        return self.doc_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "canonical_doc_id": self.canonical_doc_id,
            "logical_key": self.logical_key,
            "correction_group_id": self.correction_group_id,
            "root_doc_id": self.root_doc_id,
            "correction_resolution_status": self.correction_resolution_status,
            "correction_chain": list(self.correction_chain),
            "is_correction": self.is_correction,
        }


class CorrectionCanonicalizer:
    """P0-A, used as the canonicalization layer under event matching.

    Nothing here re-derives a correction edge.  It reads the groups P0-A already
    resolved and answers two questions: which document represents this filing,
    and how trustworthy is that answer.
    """

    def __init__(self, correction_graph: Any | None = None) -> None:
        self._graph = correction_graph
        self._cache: dict[str, CanonicalDocument] = {}

    def canonical(self, doc_id: str) -> CanonicalDocument:
        doc_id = str(doc_id)
        cached = self._cache.get(doc_id)
        if cached is not None:
            return cached
        result = self._compute(doc_id)
        self._cache[doc_id] = result
        return result

    def _compute(self, doc_id: str) -> CanonicalDocument:
        if self._graph is None:
            return CanonicalDocument(doc_id=doc_id, canonical_doc_id=doc_id)
        try:
            group = self._graph.get_correction_group(doc_id)
        except CorrectionGraphUnavailable:
            # db/006 not applied, or the database is unreachable: matching
            # continues without canonicalization.  Every other error -- a SQL
            # typo, a schema mismatch, a programming bug -- keeps its own type
            # and propagates, so a real defect is never silently downgraded into
            # "this document has no corrections".
            group = None
        if group is None:
            return CanonicalDocument(doc_id=doc_id, canonical_doc_id=doc_id)
        member = next(
            (item for item in group.members if str(item.doc_id) == doc_id), None
        )
        status = (
            str(member.resolution_status) if member is not None else group.resolution_status
        )
        chain = tuple(str(item.doc_id) for item in group.members)
        # ``resolved_latest_doc_id`` is None unless P0-A established a real
        # original-plus-corrections chain, which is exactly the condition for
        # letting another document stand in for this one.
        latest = group.resolved_latest_doc_id
        canonical = str(latest) if latest else doc_id
        return CanonicalDocument(
            doc_id=doc_id,
            canonical_doc_id=canonical,
            correction_group_id=str(group.correction_group_id),
            root_doc_id=str(group.root_doc_id),
            correction_resolution_status=status,
            correction_chain=chain,
            is_correction=bool(getattr(member, "is_correction", False)),
        )


# --------------------------------------------------------------- comparison


@dataclass(frozen=True)
class FieldComparison:
    """How a termination and one contract candidate line up field by field."""

    matched_fields: tuple[str, ...] = ()
    changed_fields: tuple[str, ...] = ()
    conflicting_identity_fields: tuple[str, ...] = ()
    compared_fields: tuple[str, ...] = ()

    @property
    def admissible(self) -> bool:
        """No identity field says these are different contracts."""

        return not self.conflicting_identity_fields

    @property
    def corroborated(self) -> bool:
        """At least one identity field positively agrees."""

        return any(name in IDENTITY_FIELDS for name in self.matched_fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_fields": list(self.matched_fields),
            "changed_fields": list(self.changed_fields),
            "conflicting_identity_fields": list(self.conflicting_identity_fields),
            "compared_fields": list(self.compared_fields),
        }


def compare_contract_documents(
    left: ContractDocument, right: ContractDocument
) -> FieldComparison:
    """Corroborate two filings against each other on the audited fields.

    A field only participates when *both* sides state it.  Identity fields that
    disagree veto the link; mutable fields that disagree are recorded as changes,
    because an amount or an end date moving is what a contract's life looks like.
    """

    matched: list[str] = []
    changed: list[str] = []
    conflicting: list[str] = []
    compared: list[str] = []

    left_identity, right_identity = left.identity(), right.identity()
    for name in IDENTITY_FIELDS:
        first, second = left_identity[name], right_identity[name]
        if not first or not second:
            continue
        compared.append(name)
        if first == second:
            matched.append(name)
        else:
            conflicting.append(name)

    left_mutable, right_mutable = left.mutable(), right.mutable()
    for name in MUTABLE_FIELDS:
        first, second = left_mutable[name], right_mutable[name]
        if not first or not second:
            continue
        compared.append(name)
        if first == second:
            matched.append(name)
        else:
            changed.append(name)

    return FieldComparison(
        matched_fields=tuple(matched),
        changed_fields=tuple(changed),
        conflicting_identity_fields=tuple(conflicting),
        compared_fields=tuple(compared),
    )


def _best_comparison(
    termination: ContractDocument,
    candidate: ContractDocument,
    canonical: ContractDocument | None,
) -> tuple[FieldComparison, str]:
    """Compare against the referenced filing and against its canonical version.

    A ``관련공시`` reference names the filing as it was submitted, which is often
    the group root.  In the corpus the root can state almost nothing while its
    resolved latest correction states the counterparty, the amount and the whole
    contract period.  Reading both and keeping the better-corroborated result is
    what makes P0-A worth reusing here rather than a formality.
    """

    direct = compare_contract_documents(termination, candidate)
    if canonical is None or canonical.doc_id == candidate.doc_id:
        return direct, candidate.doc_id
    through_canonical = compare_contract_documents(termination, canonical)
    # A conflict seen against either version is a conflict: the canonical filing
    # cannot clear a contradiction the referenced one states, and vice versa.
    if not direct.admissible:
        return direct, candidate.doc_id
    if not through_canonical.admissible:
        return through_canonical, canonical.doc_id
    if len(through_canonical.matched_fields) > len(direct.matched_fields):
        return through_canonical, canonical.doc_id
    return direct, candidate.doc_id


# ------------------------------------------------------------------- model


# ------------------------------------------------------------------ matching


@dataclass(frozen=True)
class TerminationMatch:
    """One termination's resolution: which contracts it closes, and on what."""

    termination_doc_id: str
    event_family: str
    resolution_status: str
    resolution_source: str
    confidence: float
    contract_doc_ids: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)


def _ordered(documents: Iterable[str], by_record: Mapping[str, DisclosureRecord]) -> list[str]:
    """Receipt order, with ``rcept_no`` breaking same-day ties deterministically."""

    return sorted(
        documents,
        key=lambda doc_id: (
            by_record[doc_id].order_key if doc_id in by_record else ("", "", doc_id)
        ),
    )


def match_supply_termination(
    termination: ContractDocument,
    *,
    conclusions_by_corp_date: Mapping[tuple[str, str], Sequence[str]],
    contract_documents: Mapping[str, ContractDocument],
    canonicalizer: CorrectionCanonicalizer,
    records: Mapping[str, DisclosureRecord],
) -> TerminationMatch:
    """Rule S1 -- explicit ``관련공시`` reference plus structured corroboration.

    The reference list decides which filings are even considered; corroboration
    on ``계약상대``/``계약명``/``시작일`` decides which of them are accepted.  A
    reference on its own never resolves anything.
    """

    references = termination.related
    contract_references = termination.contract_references
    evidence: dict[str, Any] = {
        "rule": "S1",
        "related_references": [item.to_dict() for item in references],
        "contract_reference_dates": [item.reference_date for item in contract_references],
        "rejected_reference_titles": [
            item.reference_title for item in references if not item.is_contract_reference
        ],
        "termination": termination.to_dict(),
    }

    if not references:
        return TerminationMatch(
            termination.doc_id,
            termination.event_family,
            UNRESOLVED,
            SOURCE_NO_RELATED_REFERENCE,
            0.0,
            evidence=evidence,
        )
    if not contract_references:
        return TerminationMatch(
            termination.doc_id,
            termination.event_family,
            UNRESOLVED,
            SOURCE_NO_CONTRACT_REFERENCE,
            0.0,
            evidence=evidence,
        )

    # Each reference is resolved on its own.  Flattening every reference into one
    # candidate pool would make "two contracts filed on one referenced date" and
    # "two separate referenced dates" indistinguishable, and the first of those
    # is a tie that must stay ambiguous while the second is a legitimate
    # multi-member lifecycle.
    resolutions: list[dict[str, Any]] = []
    accepted: list[str] = []
    discriminated = False
    for reference in contract_references:
        resolution = _resolve_one_reference(
            termination,
            reference,
            conclusions_by_corp_date=conclusions_by_corp_date,
            contract_documents=contract_documents,
            canonicalizer=canonicalizer,
            records=records,
        )
        resolutions.append(resolution)
        if resolution["outcome"] == _REF_RESOLVED:
            accepted.extend(resolution["accepted_doc_ids"])
        if resolution["rejected_doc_ids"]:
            discriminated = True
    evidence["reference_resolutions"] = resolutions
    evidence["candidate_doc_ids"] = [
        doc_id
        for resolution in resolutions
        for doc_id in resolution["candidate_doc_ids"]
    ]
    evidence["accepted_doc_ids"] = list(dict.fromkeys(accepted))
    outcomes = [resolution["outcome"] for resolution in resolutions]
    evidence["reference_outcomes"] = _counted(outcomes)

    unconfirmed = [
        resolution
        for resolution in resolutions
        if resolution["outcome"] in {_REF_TIE, _REF_UNCORROBORATED, _REF_CONFLICT}
    ]
    if unconfirmed:
        # One reference the corpus cannot pin down poisons the whole match: the
        # membership would otherwise silently omit whichever contract the
        # reference actually meant.  Every candidate stays in the evidence.
        return TerminationMatch(
            termination.doc_id,
            termination.event_family,
            AMBIGUOUS,
            {
                _REF_TIE: SOURCE_REFERENCE_TIE,
                _REF_UNCORROBORATED: SOURCE_UNCORROBORATED,
                _REF_CONFLICT: SOURCE_IDENTITY_CONFLICT,
            }[unconfirmed[0]["outcome"]],
            0.0,
            evidence=evidence,
        )
    if accepted:
        # A discriminated match is one where some referenced filing had to be
        # ruled out on an identity conflict, so it carries slightly less weight.
        source = (
            SOURCE_RELATED_REFERENCE_DISCRIMINATED
            if discriminated
            else SOURCE_RELATED_REFERENCE
        )
        return TerminationMatch(
            termination.doc_id,
            termination.event_family,
            RESOLVED,
            source,
            _CONFIDENCE[source],
            tuple(_ordered(dict.fromkeys(accepted), records)),
            evidence,
        )
    if all(outcome == _REF_EXTERNAL for outcome in outcomes):
        # The filer named the contract and it is simply before the corpus window.
        # The reference date and title stay in the evidence; no lookalike
        # contract inside the corpus is ever substituted for it.
        return TerminationMatch(
            termination.doc_id,
            termination.event_family,
            UNRESOLVED,
            SOURCE_REFERENCE_NOT_IN_CORPUS,
            0.0,
            evidence=evidence,
        )
    return TerminationMatch(
        termination.doc_id,
        termination.event_family,
        UNRESOLVED,
        SOURCE_NO_ADMISSIBLE_CANDIDATE,
        0.0,
        evidence=evidence,
    )


def _resolve_one_reference(
    termination: ContractDocument,
    reference: RelatedReference,
    *,
    conclusions_by_corp_date: Mapping[tuple[str, str], Sequence[str]],
    contract_documents: Mapping[str, ContractDocument],
    canonicalizer: CorrectionCanonicalizer,
    records: Mapping[str, DisclosureRecord],
) -> dict[str, Any]:
    """Resolve one ``관련공시`` entry to at most one logical contract.

    Candidates are grouped by their P0-A logical key first, so a filing and the
    corrections P0-A resolved onto it count once.  Two *distinct* logical
    contracts left standing is a tie, never a lifecycle.
    """

    candidates: list[str] = []
    for doc_id in conclusions_by_corp_date.get(
        (termination.corp_code, reference.reference_date), ()
    ):  # keyed on corp_code, so a cross-company candidate cannot be produced
        record = records.get(doc_id)
        if record is None or doc_id in candidates:
            continue
        # A contract cannot be concluded after it is terminated.
        if record.order_key >= records[termination.doc_id].order_key:
            continue
        candidates.append(doc_id)

    resolution: dict[str, Any] = {
        "reference_date": reference.reference_date,
        "reference_title": reference.reference_title,
        "candidate_doc_ids": list(candidates),
        "accepted_doc_ids": [],
        "rejected_doc_ids": [],
        "uncorroborated_doc_ids": [],
        "comparisons": {},
    }
    if not candidates:
        resolution["outcome"] = _REF_EXTERNAL
        return resolution

    # Logical key, not doc id: a resolved correction group is one contract.
    by_logical: dict[str, list[str]] = {}
    for doc_id in candidates:
        candidate = contract_documents.get(doc_id)
        if candidate is None:
            resolution["rejected_doc_ids"].append(doc_id)
            continue
        canonical = canonicalizer.canonical(doc_id)
        comparison, compared_against = _best_comparison(
            termination, candidate, contract_documents.get(canonical.canonical_doc_id)
        )
        resolution["comparisons"][doc_id] = {
            **comparison.to_dict(),
            "compared_against_doc_id": compared_against,
        }
        if not comparison.admissible:
            resolution["rejected_doc_ids"].append(doc_id)
        elif comparison.corroborated:
            by_logical.setdefault(canonical.logical_key, []).append(doc_id)
        else:
            resolution["uncorroborated_doc_ids"].append(doc_id)

    if len(by_logical) == 1:
        resolution["accepted_doc_ids"] = sorted(next(iter(by_logical.values())))
        resolution["outcome"] = _REF_RESOLVED
    elif len(by_logical) > 1:
        resolution["outcome"] = _REF_TIE
        resolution["tied_logical_keys"] = sorted(by_logical)
    elif resolution["uncorroborated_doc_ids"]:
        resolution["outcome"] = _REF_UNCORROBORATED
    else:
        resolution["outcome"] = _REF_CONFLICT
    return resolution


def match_trust_termination(
    termination: ContractDocument,
    *,
    conclusions_by_period: Mapping[tuple[str, str, str], Sequence[str]],
    canonicalizer: CorrectionCanonicalizer,
    records: Mapping[str, DisclosureRecord],
) -> TerminationMatch:
    """Rule T1 -- exact ``(corp_code, 시작일, 종료일)``.

    A treasury trust termination carries no reference field, but it restates the
    terminated contract's period exactly.  The audit found that key unique per
    company across the whole corpus, so no similarity fallback is offered: zero
    candidates stays unresolved and two stays ambiguous.
    """

    evidence: dict[str, Any] = {
        "rule": "T1",
        "period_key": list(termination.period_key or ()),
        "termination": termination.to_dict(),
    }
    key = termination.period_key
    if key is None:
        return TerminationMatch(
            termination.doc_id,
            termination.event_family,
            UNRESOLVED,
            SOURCE_MISSING_PERIOD_KEY,
            0.0,
            evidence=evidence,
        )

    candidates: list[str] = []
    for doc_id in conclusions_by_period.get(key, ()):
        record = records.get(doc_id)
        if record is None or doc_id in candidates:
            continue
        if record.order_key >= records[termination.doc_id].order_key:
            continue
        candidates.append(doc_id)
    evidence["candidate_doc_ids"] = list(candidates)

    # Two filings of one corrected contract are one candidate, not two.  This is
    # P0-A doing the work: without it the corrected conclusion would look like a
    # second, competing contract with an identical period.
    canonical_groups: dict[str, list[str]] = {}
    for doc_id in candidates:
        canonical = canonicalizer.canonical(doc_id)
        canonical_groups.setdefault(canonical.canonical_doc_id, []).append(doc_id)
    evidence["canonical_candidate_count"] = len(canonical_groups)

    if not canonical_groups:
        return TerminationMatch(
            termination.doc_id,
            termination.event_family,
            UNRESOLVED,
            SOURCE_PERIOD_KEY_NOT_IN_CORPUS,
            0.0,
            evidence=evidence,
        )
    if len(canonical_groups) > 1:
        return TerminationMatch(
            termination.doc_id,
            termination.event_family,
            AMBIGUOUS,
            SOURCE_MULTIPLE_CANDIDATES,
            0.0,
            evidence=evidence,
        )
    accepted = next(iter(canonical_groups.values()))
    return TerminationMatch(
        termination.doc_id,
        termination.event_family,
        RESOLVED,
        SOURCE_CONTRACT_PERIOD_KEY,
        _CONFIDENCE[SOURCE_CONTRACT_PERIOD_KEY],
        tuple(_ordered(accepted, records)),
        evidence,
    )


# ------------------------------------------------------------------- assembly


class _Components:
    """Union-find over document ids.

    Two terminations that accept the same contract filing describe one lifecycle,
    not two.  In the corpus this is how a ``[기재정정]`` of a termination behaves
    when P0-A could not attribute it: merging keeps it a single event instead of
    double-counting the contract's end.
    """

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self._parent.setdefault(item, item)

    def find(self, item: str) -> str:
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        first, second = self.find(left), self.find(right)
        if first == second:
            return
        # Smallest id wins so the component representative is deterministic.
        if second < first:
            first, second = second, first
        self._parent[second] = first

    def groups(self) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for item in self._parent:
            result.setdefault(self.find(item), []).append(item)
        return {key: sorted(value) for key, value in result.items()}


def assemble_events(
    contract_documents: Mapping[str, ContractDocument],
    matches: Sequence[TerminationMatch],
    *,
    records: Mapping[str, DisclosureRecord],
    canonicalizer: CorrectionCanonicalizer,
) -> tuple[list[CorporateEvent], list[CorporateEventRelation]]:
    """Turn resolved terminations into events, then give every other filing one.

    Every contract document ends up in exactly one event.  A filing nothing links
    to becomes an open event of one so ``get_event_state`` answers for it, and so
    a lifecycle question about it can be told there is no termination on record.
    """

    match_by_doc = {match.termination_doc_id: match for match in matches}
    components = _Components()
    for doc_id in contract_documents:
        components.add(doc_id)

    # A filing and its corrections are one contract, so they are one event.
    # Without this a termination that references the correction would leave the
    # original filing in an event of its own, and a question anchored on the
    # original could never reach the termination.  Only groups P0-A resolved are
    # merged: an ambiguous correction has not been shown to be anybody's version
    # and stays independent.
    correction_families: dict[tuple[str, str], list[str]] = {}
    for doc_id, document in contract_documents.items():
        canonical = canonicalizer.canonical(doc_id)
        if (
            canonical.correction_group_id is None
            or canonical.correction_resolution_status != RESOLVED
        ):
            continue
        correction_families.setdefault(
            (canonical.correction_group_id, document.event_family), []
        ).append(doc_id)
    for siblings in correction_families.values():
        for doc_id in siblings[1:]:
            components.union(siblings[0], doc_id)

    for match in matches:
        if match.resolution_status != RESOLVED:
            continue
        for contract_doc_id in match.contract_doc_ids:
            components.union(match.termination_doc_id, contract_doc_id)

    events: list[CorporateEvent] = []
    relations: list[CorporateEventRelation] = []

    for _, doc_ids in sorted(components.groups().items()):
        ordered = _ordered(doc_ids, records)
        family = contract_documents[ordered[0]].event_family
        corp_codes = {contract_documents[doc_id].corp_code for doc_id in ordered}
        if len(corp_codes) != 1:
            # Structurally impossible: every rule keys on corp_code.  Kept as a
            # hard stop so a damaged input can never produce a cross-company
            # lifecycle instead of an error.
            raise ValueError(
                f"cross-company event component: {sorted(corp_codes)} in {ordered}"
            )
        corp_code = next(iter(corp_codes))

        terminations = [
            doc_id for doc_id in ordered if contract_documents[doc_id].is_termination
        ]
        contracts = [
            doc_id for doc_id in ordered if not contract_documents[doc_id].is_termination
        ]

        if contracts:
            opening = canonicalizer.canonical(contracts[0])
            root_logical_key = (
                opening.correction_group_id
                if opening.correction_resolution_status == RESOLVED
                and opening.correction_group_id
                else contracts[0]
            )
        else:
            termination_doc_id = terminations[0]
            termination_match = match_by_doc[termination_doc_id]
            source_reference = {
                key: termination_match.evidence[key]
                for key in ("contract_reference_dates", "period_key")
                if termination_match.evidence.get(key)
            } or None
            if termination_match.resolution_status == AMBIGUOUS:
                root_logical_key = ambiguous_event_anchor(
                    termination_doc_id, source_reference
                )
            else:
                root_logical_key = unresolved_event_anchor(
                    termination_doc_id, source_reference
                )
        identifier = event_id(corp_code, family, root_logical_key)

        # Versions of one filing share a role: a correction is the same contract
        # filing seen again, while a separate later ``체결`` filing is an update
        # to the lifecycle.  Grouping on the correction group keeps the two apart.
        version_of: dict[str, str] = {}
        for doc_id in contracts:
            canonical = canonicalizer.canonical(doc_id)
            version_of[doc_id] = (
                canonical.correction_group_id
                if canonical.correction_resolution_status == RESOLVED
                and canonical.correction_group_id
                else doc_id
            )
        opening_version = version_of.get(contracts[0]) if contracts else None

        # One member row per *logical* filing.  A filing and the corrections
        # P0-A resolved onto it are one contract state, not several, so the
        # group contributes a single row whose ``doc_id`` is the verified latest
        # and whose provenance carries the root and the whole chain.  Emitting a
        # row per raw document would restate the correction graph inside the
        # event graph and inflate every membership count that depends on it.
        logical_docs: dict[str, list[str]] = {}
        for doc_id in ordered:
            logical_docs.setdefault(
                canonicalizer.canonical(doc_id).logical_key, []
            ).append(doc_id)

        members: list[CorporateEventMember] = []
        unverified_correction = any(
            canonicalizer.canonical(doc_id).has_unverified_correction
            for doc_id in ordered
        )
        # Chronological by the earliest filing of each logical contract, so the
        # timeline still reads in the order the company disclosed things.
        def _earliest(entry: tuple[str, list[str]]) -> tuple[str, str, str]:
            first = _ordered(entry[1], records)[0]
            return records[first].order_key

        for index, (logical_key, raw_docs) in enumerate(
            sorted(logical_docs.items(), key=_earliest)
        ):
            group_docs = _ordered(raw_docs, records)
            # The representative is the version P0-A verified as current; for an
            # uncorrected filing that is the filing itself.
            representative = canonicalizer.canonical(group_docs[0]).canonical_doc_id
            if representative not in group_docs:
                representative = group_docs[-1]
            canonical = canonicalizer.canonical(representative)
            document = contract_documents[representative]
            if document.is_termination:
                role = ROLE_TERMINATION
            elif logical_key == opening_version:
                role = ROLE_CONTRACT
            else:
                # A further ``체결`` filing about the same contract.  A role, not
                # a reclassification: the disclosure is still a conclusion.
                role = ROLE_CONTRACT_UPDATE
            member_match = next(
                (match_by_doc[doc] for doc in group_docs if doc in match_by_doc), None
            )
            members.append(
                CorporateEventMember(
                    event_id=identifier,
                    corp_code=corp_code,
                    doc_id=representative,
                    canonical_doc_id=canonical.canonical_doc_id,
                    member_role=role,
                    member_order=index,
                    event_date=document.event_date,
                    root_doc_id=canonical.root_doc_id or representative,
                    correction_group_id=canonical.correction_group_id,
                    correction_resolution_status=canonical.correction_resolution_status,
                    correction_chain=canonical.correction_chain,
                    is_correction=canonical.is_correction,
                    confidence=member_match.confidence if member_match else 0.0,
                    provenance={
                        "logical_key": logical_key,
                        "root_doc_id": canonical.root_doc_id or representative,
                        "correction_group_id": canonical.correction_group_id,
                        "correction_chain": list(canonical.correction_chain),
                        # Every raw filing this one row stands for, so nothing
                        # the corpus contained is lost by collapsing.
                        "collapsed_doc_ids": list(group_docs),
                    },
                    evidence={
                        "canonical": canonical.to_dict(),
                        "document": document.to_dict(),
                        **(
                            {"match": dict(member_match.evidence)}
                            if member_match is not None
                            else {}
                        ),
                    },
                )
            )

        resolved_matches = [
            match_by_doc[doc_id]
            for doc_id in terminations
            if doc_id in match_by_doc
            and match_by_doc[doc_id].resolution_status == RESOLVED
        ]
        if resolved_matches:
            lifecycle = LIFECYCLE_TERMINATED
            status = RESOLVED
            source = resolved_matches[0].resolution_source
            confidence = min(match.confidence for match in resolved_matches)
            if unverified_correction:
                confidence = max(0.0, confidence - AMBIGUOUS_CORRECTION_PENALTY)
        elif terminations:
            # An unresolved or ambiguous termination stands alone: it is a real
            # termination filing whose contract could not be identified, and it
            # is never attached to the nearest-looking contract.
            match = match_by_doc[terminations[0]]
            lifecycle = LIFECYCLE_TERMINATED
            status = match.resolution_status
            source = match.resolution_source
            confidence = match.confidence
        else:
            # No termination on record.  A filing plus the corrections P0-A
            # resolved onto it is an established group; a lone filing is not a
            # lifecycle at all and says so.
            lifecycle = LIFECYCLE_OPEN
            grouped = len(ordered) > 1
            status = RESOLVED if grouped else UNRESOLVED
            source = SOURCE_CORRECTION_GROUP if grouped else SOURCE_SINGLE_DOCUMENT
            confidence = _CONFIDENCE[source]
            if grouped and unverified_correction:
                confidence = max(0.0, confidence - AMBIGUOUS_CORRECTION_PENALTY)

        contract_dates = [
            contract_documents[doc_id].event_date
            for doc_id in contracts
            if contract_documents[doc_id].event_date
        ]
        closing_dates = [
            contract_documents[doc_id].event_date
            for doc_id in terminations
            if contract_documents[doc_id].event_date
        ]
        events.append(
            CorporateEvent(
                event_id=identifier,
                corp_code=corp_code,
                event_family=family,
                root_logical_key=root_logical_key,
                lifecycle_status=lifecycle,
                resolution_status=status,
                resolution_source=source,
                members=tuple(members),
                opened_at=min(contract_dates) if contract_dates else None,
                closed_at=max(closing_dates) if closing_dates else None,
                confidence=round(float(confidence), 4),
                evidence={
                    "termination_doc_ids": list(terminations),
                    "contract_doc_ids": list(contracts),
                    "has_unverified_correction_member": unverified_correction,
                    "matches": [
                        {
                            "termination_doc_id": match_by_doc[doc_id].termination_doc_id,
                            "resolution_status": match_by_doc[doc_id].resolution_status,
                            "resolution_source": match_by_doc[doc_id].resolution_source,
                            "confidence": match_by_doc[doc_id].confidence,
                            **dict(match_by_doc[doc_id].evidence),
                        }
                        for doc_id in terminations
                        if doc_id in match_by_doc
                    ],
                },
            )
        )

        # Edges join logical members, not raw filings.  A termination that
        # referenced three corrections of one contract closes *one* contract, so
        # it emits one edge to that contract's representative rather than three
        # to its versions.
        member_of_doc = {
            raw: item.doc_id
            for item in members
            for raw in item.provenance.get("collapsed_doc_ids", (item.doc_id,))
        }
        for doc_id in terminations:
            match = match_by_doc.get(doc_id)
            if match is None:
                continue
            source_member = member_of_doc.get(doc_id, doc_id)
            if match.resolution_status == RESOLVED:
                targets = dict.fromkeys(
                    member_of_doc.get(target, target)
                    for target in match.contract_doc_ids
                )
                for target in targets:
                    if target == source_member:
                        continue
                    relations.append(
                        CorporateEventRelation(
                            relation_id=relation_id(
                                source_member, RELATION_TERMINATES_EVENT, target
                            ),
                            source_doc_id=source_member,
                            target_doc_id=target,
                            relation_type=RELATION_TERMINATES_EVENT,
                            event_id=identifier,
                            resolution_status=RESOLVED,
                            resolution_source=match.resolution_source,
                            confidence=match.confidence,
                            evidence=dict(match.evidence),
                        )
                    )
            else:
                relations.append(
                    CorporateEventRelation(
                        relation_id=relation_id(
                            source_member, RELATION_TERMINATES_EVENT, None
                        ),
                        source_doc_id=source_member,
                        target_doc_id=None,
                        relation_type=RELATION_TERMINATES_EVENT,
                        event_id=identifier,
                        resolution_status=match.resolution_status,
                        resolution_source=match.resolution_source,
                        confidence=match.confidence,
                        evidence=dict(match.evidence),
                    )
                )

        # Distinct logical contracts only.  Versions of one filing are already
        # linked by P0-A, and restating that here would duplicate the correction
        # graph inside the event graph.
        contract_members = [
            item for item in members if item.member_role != ROLE_TERMINATION
        ]
        if len(contract_members) > 1:
            anchor = contract_members[0].doc_id
            for item in contract_members[1:]:
                relations.append(
                    CorporateEventRelation(
                        relation_id=relation_id(
                            item.doc_id, RELATION_BELONGS_TO_EVENT, anchor
                        ),
                        source_doc_id=item.doc_id,
                        target_doc_id=anchor,
                        relation_type=RELATION_BELONGS_TO_EVENT,
                        event_id=identifier,
                        resolution_status=RESOLVED,
                        resolution_source=source,
                        confidence=confidence,
                        evidence={
                            "reason": "listed together by the termination's 관련공시",
                            "termination_doc_ids": list(terminations),
                        },
                    )
                )

    events.sort(key=lambda item: item.event_id)
    relations.sort(key=lambda item: item.relation_id)
    return events, relations


class CorporateEventGraph:
    """The assembled events, with the read API P0-C and retrieval both use."""

    def __init__(
        self,
        events: Sequence[CorporateEvent],
        relations: Sequence[CorporateEventRelation] = (),
    ) -> None:
        self._events = tuple(events)
        self._relations = tuple(relations)
        self._by_event = {event.event_id: event for event in self._events}
        # Membership is stored per *logical* contract, but lookup has to answer
        # for every raw filing: a question can be anchored on the original of a
        # correction chain, and that filing must still reach its lifecycle.  The
        # collapsed versions are indexed here without becoming member rows.
        self._by_doc: dict[str, CorporateEvent] = {}
        self._member_of_doc: dict[str, CorporateEventMember] = {}
        for event in self._events:
            for member in event.members:
                for doc_id in {
                    member.doc_id,
                    *member.correction_chain,
                    *(member.provenance.get("collapsed_doc_ids") or ()),
                }:
                    self._by_doc.setdefault(str(doc_id), event)
                    self._member_of_doc.setdefault(str(doc_id), member)
                # The representative always wins over an inherited entry.
                self._by_doc[member.doc_id] = event
                self._member_of_doc[member.doc_id] = member

    @property
    def events(self) -> tuple[CorporateEvent, ...]:
        return self._events

    @property
    def relations(self) -> tuple[CorporateEventRelation, ...]:
        return self._relations

    @property
    def members(self) -> tuple[CorporateEventMember, ...]:
        return tuple(member for event in self._events for member in event.members)

    # ------------------------------------------------------------------ read

    def get_event(self, doc_id: str) -> CorporateEvent | None:
        return self._by_doc.get(str(doc_id))

    def get_event_by_id(self, event_id: str) -> CorporateEvent | None:
        return self._by_event.get(str(event_id))

    def get_event_timeline(self, doc_id: str) -> tuple[CorporateEventMember, ...]:
        """Every member of this document's event, oldest filing first."""

        event = self.get_event(doc_id)
        if event is None:
            return ()
        return tuple(sorted(event.members, key=lambda member: member.member_order))

    def get_member(self, doc_id: str) -> CorporateEventMember | None:
        """The logical member this filing belongs to, collapsed versions included."""

        return self._member_of_doc.get(str(doc_id))

    def get_related_documents(self, doc_id: str) -> tuple[str, ...]:
        """The other logical members of this document's event, in timeline order."""

        own = self.get_member(doc_id)
        own_doc_id = own.doc_id if own is not None else str(doc_id)
        return tuple(
            member.doc_id
            for member in self.get_event_timeline(doc_id)
            if member.doc_id != own_doc_id
        )

    def get_event_state(self, doc_id: str) -> CorporateEventState | None:
        doc_id = str(doc_id)
        event = self.get_event(doc_id)
        if event is None:
            return None
        member = self.get_member(doc_id)
        if member is None:
            return None
        # ``doc_id`` names the *logical member* this filing belongs to, which for
        # a superseded filing is its representative rather than the id asked
        # about.  ``event_states`` still keys the result by what the caller
        # asked, exactly as the PostgreSQL repository does.
        return CorporateEventState(
            doc_id=member.doc_id,
            event_id=event.event_id,
            corp_code=event.corp_code,
            event_family=event.event_family,
            member_role=member.member_role,
            lifecycle_status=event.lifecycle_status,
            resolution_status=event.resolution_status,
            canonical_doc_id=member.canonical_doc_id,
            member_count=event.member_count,
            correction_group_id=member.correction_group_id,
            correction_resolution_status=member.correction_resolution_status,
        )

    def event_states(
        self, doc_ids: Iterable[str]
    ) -> dict[str, CorporateEventState]:
        states: dict[str, CorporateEventState] = {}
        for doc_id in doc_ids:
            state = self.get_event_state(doc_id)
            if state is not None:
                states[str(doc_id)] = state
        return states

    def diagnostics(self) -> dict[str, Any]:
        return corporate_event_diagnostics(self._events, self._relations)


def build_corporate_event_graph(
    records: Sequence[DisclosureRecord],
    contract_documents: Mapping[str, ContractDocument],
    *,
    correction_graph: Any | None = None,
) -> CorporateEventGraph:
    """Resolve every termination, then assemble the events they imply."""

    by_doc = {record.doc_id: record for record in records}
    documents = {
        doc_id: document
        for doc_id, document in contract_documents.items()
        if doc_id in by_doc
    }
    canonicalizer = CorrectionCanonicalizer(correction_graph)

    conclusions_by_corp_date: dict[tuple[str, str], list[str]] = {}
    conclusions_by_period: dict[tuple[str, str, str], list[str]] = {}
    for doc_id, document in documents.items():
        if document.is_termination:
            continue
        record = by_doc[doc_id]
        if document.event_family == FAMILY_SUPPLY_CONTRACT and record.rcept_dt:
            conclusions_by_corp_date.setdefault(
                (document.corp_code, record.rcept_dt), []
            ).append(doc_id)
        if document.event_family == FAMILY_TREASURY_TRUST:
            key = document.period_key
            if key is not None:
                conclusions_by_period.setdefault(key, []).append(doc_id)
    for bucket in (conclusions_by_corp_date, conclusions_by_period):
        for key in bucket:
            bucket[key] = _ordered(bucket[key], by_doc)

    matches: list[TerminationMatch] = []
    for doc_id in _ordered(
        [doc_id for doc_id, item in documents.items() if item.is_termination], by_doc
    ):
        document = documents[doc_id]
        if document.event_family == FAMILY_SUPPLY_CONTRACT:
            matches.append(
                match_supply_termination(
                    document,
                    conclusions_by_corp_date=conclusions_by_corp_date,
                    contract_documents=documents,
                    canonicalizer=canonicalizer,
                    records=by_doc,
                )
            )
        else:
            matches.append(
                match_trust_termination(
                    document,
                    conclusions_by_period=conclusions_by_period,
                    canonicalizer=canonicalizer,
                    records=by_doc,
                )
            )

    events, relations = assemble_events(
        documents, matches, records=by_doc, canonicalizer=canonicalizer
    )
    return CorporateEventGraph(events, relations)


# ---------------------------------------------------------------- diagnostics


def _counted(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _correction_status(member: CorporateEventMember) -> str | None:
    """The member's P0-A status as a plain string, enum or not."""

    status = member.correction_resolution_status
    return getattr(status, "value", status)


def _collapsed_docs(member: CorporateEventMember) -> tuple[str, ...]:
    """Every raw filing one member row stands for."""

    collapsed = (member.provenance or {}).get("collapsed_doc_ids")
    return tuple(str(item) for item in collapsed) if collapsed else (member.doc_id,)


def _membership_diagnostics(events: Sequence[CorporateEvent]) -> dict[str, Any]:
    """Separate raw disclosures from the logical members they collapse into.

    ``membership_count`` used to conflate the two, which made a fifteen-filing
    correction chain look like a fifteen-member lifecycle.  A member row is one
    logical contract state; the raw filings behind it stay in its provenance.
    """

    raw = 0
    logical = 0
    collapsed_documents = 0
    collapsed_groups: set[str] = set()
    ambiguous_preserved = 0
    for event in events:
        for member in event.members:
            docs = _collapsed_docs(member)
            raw += len(docs)
            logical += 1
            if len(docs) > 1:
                collapsed_documents += len(docs) - 1
                if member.correction_group_id:
                    collapsed_groups.add(str(member.correction_group_id))
            if _correction_status(member) in {AMBIGUOUS, UNRESOLVED}:
                ambiguous_preserved += 1
    return {
        "raw_contract_document_count": raw,
        "logical_event_member_count": logical,
        "membership_count": logical,
        "resolved_correction_documents_collapsed": collapsed_documents,
        "resolved_correction_groups_reused": len(collapsed_groups),
        "ambiguous_correction_documents_preserved": ambiguous_preserved,
    }


def _reference_diagnostics(events: Sequence[CorporateEvent]) -> dict[str, Any]:
    """What the ``관련공시`` parser read, kept and discarded across the corpus.

    Counted from the evidence each termination member already carries, so these
    numbers describe the same rows the manual audit reviews.
    """

    valid = 0
    ignored = 0
    multi_reference = 0
    outcomes: dict[str, int] = {}
    for event in events:
        for member in event.members:
            match = (member.evidence or {}).get("match") or {}
            if not match:
                continue
            dates = match.get("contract_reference_dates") or []
            valid += len(dates)
            ignored += len(match.get("rejected_reference_titles") or [])
            if len(dates) > 1:
                multi_reference += 1
            for outcome, count in (match.get("reference_outcomes") or {}).items():
                outcomes[str(outcome)] = outcomes.get(str(outcome), 0) + int(count)
    return {
        "valid_explicit_reference_count": valid,
        "ignored_unrelated_reference_count": ignored,
        "multi_reference_termination_count": multi_reference,
        "reference_outcome_counts": dict(sorted(outcomes.items())),
    }


def corporate_event_diagnostics(
    events: Sequence[CorporateEvent], relations: Sequence[CorporateEventRelation]
) -> dict[str, Any]:
    """Read-only invariants.  A healthy graph reports zero for every guard."""

    membership: dict[str, list[str]] = {}
    for event in events:
        for member in event.members:
            membership.setdefault(member.doc_id, []).append(event.event_id)

    # Compared against the member's own company rather than anything inside
    # ``evidence``, so a row assembled without that evidence block is still checked.
    cross_company = 0
    for event in events:
        corp_codes = {
            member.corp_code
            or (member.evidence.get("document") or {}).get("corp_code")
            for member in event.members
        }
        corp_codes.discard(None)
        if len(corp_codes) > 1 or (
            corp_codes and event.corp_code not in corp_codes
        ):
            cross_company += 1

    relation_keys = [
        (relation.source_doc_id, relation.relation_type, relation.target_doc_id or "")
        for relation in relations
    ]
    duplicate_relations = len(relation_keys) - len(set(relation_keys))

    edges: dict[str, set[str]] = {}
    for relation in relations:
        if relation.target_doc_id:
            edges.setdefault(relation.source_doc_id, set()).add(relation.target_doc_id)
    cycles = _cycle_count(edges)

    terminated = [event for event in events if event.is_terminated]
    return {
        "event_count": len(events),
        **_membership_diagnostics(events),
        "relation_count": len(relations),
        "family_counts": _counted(event.event_family for event in events),
        "lifecycle_counts": _counted(event.lifecycle_status for event in events),
        "resolution_counts": _counted(event.resolution_status for event in events),
        "resolution_sources": _counted(event.resolution_source for event in events),
        "relation_types": _counted(relation.relation_type for relation in relations),
        "relation_resolution_counts": _counted(
            relation.resolution_status for relation in relations
        ),
        "terminated_event_count": len(terminated),
        "resolved_terminated_event_count": sum(
            1 for event in terminated if event.resolution_status == RESOLVED
        ),
        # Both counts are over logical members. ``multi_contract`` needs two
        # distinct *contracts*, not a contract plus its termination, so a plain
        # 체결 -> 해지 lifecycle is multi-member but not multi-contract.
        "multi_member_event_count": sum(1 for event in events if event.member_count > 1),
        "multi_contract_event_count": sum(
            1 for event in events if len(event.contract_members) > 1
        ),
        "max_event_members": max((event.member_count for event in events), default=0),
        "reused_correction_group_count": len(
            {
                member.correction_group_id
                for event in events
                for member in event.members
                if member.correction_group_id
            }
        ),
        "unverified_correction_member_count": sum(
            1
            for event in events
            for member in event.members
            if _correction_status(member) in {AMBIGUOUS, UNRESOLVED}
        ),
        "ambiguous_correction_member_count": sum(
            1
            for event in events
            for member in event.members
            if _correction_status(member) == AMBIGUOUS
        ),
        **_reference_diagnostics(events),
        # Guards: every one of these must be zero.
        "cross_company_event_count": cross_company,
        "duplicate_membership_count": sum(
            len(event_ids) - 1
            for event_ids in membership.values()
            if len(event_ids) > 1
        ),
        "duplicate_relation_count": duplicate_relations,
        "self_relation_count": sum(
            1
            for relation in relations
            if relation.target_doc_id
            and relation.target_doc_id == relation.source_doc_id
        ),
        "invalid_resolution_relation_count": sum(
            1
            for relation in relations
            if (relation.resolution_status == RESOLVED)
            != (relation.target_doc_id is not None)
        ),
        "cycle_count": cycles,
    }


def _cycle_count(edges: Mapping[str, set[str]]) -> int:
    """Nodes that take part in a directed cycle.

    Structurally there can be none: a ``terminates_event`` edge always runs from
    a termination filing to a conclusion filing, and the two sets are disjoint.
    The check exists so a damaged rebuild reports rather than hides it.
    """

    state: dict[str, int] = {}
    cyclic: set[str] = set()

    def visit(node: str, stack: list[str]) -> None:
        state[node] = 1
        stack.append(node)
        for target in sorted(edges.get(node, ())):
            if state.get(target, 0) == 0:
                visit(target, stack)
            elif state.get(target) == 1:
                cyclic.update(stack[stack.index(target):])
        stack.pop()
        state[node] = 2

    for node in sorted(edges):
        if state.get(node, 0) == 0:
            visit(node, [])
    return len(cyclic)
