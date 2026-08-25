"""Deterministic correction graph over original and correcting disclosures.

The graph is built only from frozen disclosure metadata and from the correction
notice a correcting disclosure carries in its own tables.  No language model
takes part in deciding which document corrects which: every edge records the
rule that produced it, the evidence it used, and a resolution status of
``resolved``, ``ambiguous``, or ``unresolved``.

Three rules resolve an edge, in order:

``correction_notice``
    A correcting disclosure states the submission date of the document it
    corrects (``정정대상 공시서류의 최초제출일`` on DART filings,
    ``정정관련 공시서류제출일`` on exchange filings).  That date plus the
    ``corp_code`` identifies the target.

``periodic_period_key``
    A periodic report is uniquely identified by
    ``corp_code``/``doc_subtype``/``base_year``/``base_month``.  Corrections of
    a periodic report join that report's chain in receipt order.

``event_title_key``
    An event disclosure without a usable notice falls back to an exact
    normalized ``report_nm`` match inside the same company and document group.
    It resolves only when exactly one earlier candidate exists; title
    similarity alone never confirms a relation.

Anything else stays ``ambiguous`` (several candidates, no discriminator) or
``unresolved`` (no candidate at all).  Neither is ever forced into another
group, so an unresolved correction cannot disturb a resolved group's latest
document.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence


CORRECTION_RELATION_TYPE = "correction_of"


class CorrectionGraphUnavailable(RuntimeError):
    """The correction graph cannot be read right now.

    Raised only for conditions a caller may reasonably degrade past: the
    migration has not been applied yet, or the database is unreachable.  A SQL
    typo, a schema mismatch, or any other programming error is never reported
    this way, so it surfaces instead of silently disabling the feature.
    """

RESOLVED = "resolved"
AMBIGUOUS = "ambiguous"
UNRESOLVED = "unresolved"
RESOLUTION_STATUSES = (RESOLVED, AMBIGUOUS, UNRESOLVED)

#: Rule names recorded on every relation and group member.
SOURCE_CORRECTION_NOTICE = "correction_notice"
SOURCE_CORRECTION_NOTICE_TITLE = "correction_notice_title"
SOURCE_PERIODIC_PERIOD_KEY = "periodic_period_key"
SOURCE_EVENT_TITLE_KEY = "event_title_key"
SOURCE_GROUP_ROOT = "group_root"
SOURCE_NO_CANDIDATE = "no_candidate"
SOURCE_MULTIPLE_CANDIDATES = "multiple_candidates"
SOURCE_NOTICE_TARGET_MISSING = "notice_target_not_in_corpus"
SOURCE_CYCLE_DETECTED = "cycle_detected"
SOURCE_SELF_REFERENCE = "self_reference"

_CONFIDENCE = {
    SOURCE_CORRECTION_NOTICE: 0.95,
    SOURCE_CORRECTION_NOTICE_TITLE: 0.90,
    SOURCE_PERIODIC_PERIOD_KEY: 0.90,
    SOURCE_EVENT_TITLE_KEY: 0.70,
}

#: Labels a correcting disclosure uses for the target's submission date.
_SUBMITTED_LABELS = (
    "정정관련공시서류제출일",
    "정정대상공시서류의최초제출일",
    "정정대상공시서류최초제출일",
    "정정대상공시서류의제출일",
    "정정대상공시서류제출일",
)
#: Labels a correcting disclosure uses for the target's report name.
_TARGET_NAME_LABELS = ("정정관련공시서류", "정정대상공시서류")
#: Labels carrying the date the correction itself was filed.
_CORRECTED_ON_LABELS = ("정정일자", "정정신고일자")

_DATE = re.compile(r"(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})")
_LEADING_ORDINAL = re.compile(r"^[0-9]+\s*[.)-]\s*")
_BRACKETED = re.compile(r"[\[(（【][^\])）】]*[\])）】]")
_TITLE_NOISE = re.compile(r"[\s·ㆍ・･∙‧,./\\|:;_~\-–—'\"“”‘’]+")


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", _text(value))


def _iso_date(value: Any) -> str | None:
    """Normalize a receipt/submission date to ``YYYY-MM-DD``."""

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _compact(value)
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    match = _DATE.search(text)
    if match:
        return "%04d-%02d-%02d" % tuple(int(part) for part in match.groups())
    return None


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_report_title(value: Any) -> str:
    """Drop correction markers and separators so titles compare exactly.

    ``[기재정정]단일판매ㆍ공급계약체결`` and ``단일판매·공급계약 체결`` both
    normalize to ``단일판매공급계약체결``.  Trailing period markers such as
    ``(2024.12)`` are preserved because they distinguish periodic reports.
    """

    text = _text(value)
    # Only leading annotation brackets are correction markers; a trailing
    # qualifier such as "(자율공시)" or "(2024.12)" is part of the identity.
    while True:
        stripped = _BRACKETED.sub(" ", text, count=1) if _BRACKETED.match(text.lstrip()) else text
        if stripped == text:
            break
        text = stripped
    return _TITLE_NOISE.sub("", text).casefold()


def relation_id(source_doc_id: str, relation_type: str, target_doc_id: str | None) -> str:
    """Stable identifier so a repeated backfill updates instead of duplicating."""

    raw = "\x1f".join(
        [str(source_doc_id), str(relation_type), str(target_doc_id or "")]
    ).encode("utf-8")
    return f"corr_{hashlib.sha256(raw).hexdigest()[:24]}"


@dataclass(frozen=True)
class DisclosureRecord:
    """The frozen disclosure metadata the graph is allowed to reason over."""

    doc_id: str
    corp_code: str
    doc_group: str
    report_nm: str
    rcept_no: str
    rcept_dt: str | None = None
    doc_subtype: str | None = None
    base_year: int | None = None
    base_month: int | None = None
    is_correction: bool = False

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "DisclosureRecord":
        return cls(
            doc_id=str(row.get("doc_id") or ""),
            corp_code=str(row.get("corp_code") or ""),
            doc_group=str(row.get("doc_group") or ""),
            report_nm=_text(row.get("report_nm")),
            rcept_no=str(row.get("rcept_no") or ""),
            rcept_dt=_iso_date(row.get("rcept_dt")),
            doc_subtype=(
                str(row["doc_subtype"]) if row.get("doc_subtype") not in (None, "") else None
            ),
            base_year=_optional_int(row.get("base_year")),
            base_month=_optional_int(row.get("base_month")),
            is_correction=bool(row.get("is_correction")),
        )

    @property
    def order_key(self) -> tuple[str, str, str]:
        """Receipt order.  ``rcept_no`` breaks same-day ties deterministically."""

        return (self.rcept_dt or "", self.rcept_no, self.doc_id)

    @property
    def title_key(self) -> str:
        return normalize_report_title(self.report_nm)


@dataclass(frozen=True)
class CorrectionNotice:
    """Correction target evidence parsed from the correcting document itself."""

    doc_id: str
    target_submitted_on: str | None = None
    target_report_nm: str | None = None
    corrected_on: str | None = None
    source_table_id: str | None = None
    source_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_submitted_on": self.target_submitted_on,
            "target_report_nm": self.target_report_nm,
            "corrected_on": self.corrected_on,
            "source_table_id": self.source_table_id,
            "source_label": self.source_label,
        }


@dataclass(frozen=True)
class CorrectionRelation:
    """One ``source corrects target`` edge with the rule that produced it."""

    relation_id: str
    source_doc_id: str
    relation_type: str = CORRECTION_RELATION_TYPE
    target_doc_id: str | None = None
    resolution_status: str = UNRESOLVED
    resolution_source: str = SOURCE_NO_CANDIDATE
    confidence: float = 0.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "source_doc_id": self.source_doc_id,
            "target_doc_id": self.target_doc_id,
            "relation_type": self.relation_type,
            "resolution_status": self.resolution_status,
            "resolution_source": self.resolution_source,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CorrectionGroupMember:
    """One disclosure's place inside a correction group."""

    doc_id: str
    correction_group_id: str
    root_doc_id: str
    parent_doc_id: str | None
    correction_order: int
    is_latest: bool
    resolution_status: str
    resolution_source: str
    confidence: float
    is_correction: bool = False
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "correction_group_id": self.correction_group_id,
            "root_doc_id": self.root_doc_id,
            "parent_doc_id": self.parent_doc_id,
            "correction_order": self.correction_order,
            "is_latest": self.is_latest,
            "resolution_status": self.resolution_status,
            "resolution_source": self.resolution_source,
            "confidence": self.confidence,
            "is_correction": self.is_correction,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CorrectionGroup:
    """A root disclosure plus every correction that resolved onto it."""

    correction_group_id: str
    root_doc_id: str
    members: tuple[CorrectionGroupMember, ...]
    resolution_status: str

    @property
    def is_resolved(self) -> bool:
        """Whether this group is an established original-plus-corrections chain.

        A single ambiguous or unresolved correction is also stored as a group of
        its own so it is never attached to somebody else's chain.  Such a group
        is not resolved, and its member must not be read as a verified final
        version of anything.
        """

        return self.resolution_status == RESOLVED and len(self.members) > 1

    @property
    def latest_doc_id(self) -> str | None:
        """The last document in the chain, resolved or not.

        Use :attr:`resolved_latest_doc_id` when the answer is only meaningful
        for an established chain.
        """

        for member in self.members:
            if member.is_latest:
                return member.doc_id
        return None

    @property
    def resolved_latest_doc_id(self) -> str | None:
        """The final valid document, or None when the group is not resolved."""

        return self.latest_doc_id if self.is_resolved else None

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(member.doc_id for member in self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correction_group_id": self.correction_group_id,
            "root_doc_id": self.root_doc_id,
            "resolution_status": self.resolution_status,
            "is_resolved": self.is_resolved,
            "latest_doc_id": self.latest_doc_id,
            "resolved_latest_doc_id": self.resolved_latest_doc_id,
            "members": [member.to_dict() for member in self.members],
        }


def _cells(row: Any) -> list[str]:
    if not isinstance(row, (list, tuple)):
        return []
    values: list[str] = []
    for cell in row:
        if isinstance(cell, Mapping):
            values.append(_text(cell.get("text")))
        else:
            values.append(_text(cell))
    return values


def _label_key(value: str) -> str:
    return _LEADING_ORDINAL.sub("", _compact(value)).rstrip(":：").strip()


def extract_correction_notice(
    doc_id: str, tables: Iterable[Mapping[str, Any]]
) -> CorrectionNotice | None:
    """Read the correction notice a correcting disclosure states about itself.

    ``tables`` accepts the frozen table payloads unchanged: each mapping needs a
    ``table_id`` plus ``rows`` or ``table_rows``.  Only labelled header cells are
    read, so no free text is interpreted.
    """

    submitted_on: str | None = None
    target_name: str | None = None
    corrected_on: str | None = None
    table_id: str | None = None
    label_used: str | None = None

    for table in tables or ():
        rows = table.get("rows")
        if rows is None:
            rows = table.get("table_rows")
        for row in rows or ():
            cells = _cells(row)
            if len(cells) < 2:
                continue
            label = _label_key(cells[0])
            if not label:
                continue
            remainder = [value.strip() for value in cells[1:] if value.strip()]
            if not remainder:
                continue
            value = " ".join(remainder)
            if submitted_on is None and any(
                label.endswith(candidate) for candidate in _SUBMITTED_LABELS
            ):
                parsed = _iso_date(value)
                if parsed:
                    submitted_on = parsed
                    table_id = table_id or _text(table.get("table_id")) or None
                    label_used = label
            elif target_name is None and label in _TARGET_NAME_LABELS:
                target_name = value
                table_id = table_id or _text(table.get("table_id")) or None
            elif corrected_on is None and label in _CORRECTED_ON_LABELS:
                corrected_on = _iso_date(value)
        if submitted_on and target_name and corrected_on:
            break

    if not any((submitted_on, target_name, corrected_on)):
        return None
    return CorrectionNotice(
        doc_id=str(doc_id),
        target_submitted_on=submitted_on,
        target_report_nm=target_name,
        corrected_on=corrected_on,
        source_table_id=table_id,
        source_label=label_used,
    )


def _titles_match(left: Any, right: Any) -> bool:
    first = normalize_report_title(left)
    second = normalize_report_title(right)
    if not first or not second:
        return False
    return first == second or first in second or second in first


def _unresolved(source: DisclosureRecord, reason: str, evidence: Mapping[str, Any]) -> CorrectionRelation:
    return CorrectionRelation(
        relation_id=relation_id(source.doc_id, CORRECTION_RELATION_TYPE, None),
        source_doc_id=source.doc_id,
        target_doc_id=None,
        resolution_status=UNRESOLVED,
        resolution_source=reason,
        confidence=0.0,
        evidence=dict(evidence),
    )


def _ambiguous(
    source: DisclosureRecord, reason: str, evidence: Mapping[str, Any]
) -> CorrectionRelation:
    return CorrectionRelation(
        relation_id=relation_id(source.doc_id, CORRECTION_RELATION_TYPE, None),
        source_doc_id=source.doc_id,
        target_doc_id=None,
        resolution_status=AMBIGUOUS,
        resolution_source=reason,
        confidence=0.0,
        evidence=dict(evidence),
    )


def _resolved(
    source: DisclosureRecord,
    target: DisclosureRecord,
    rule: str,
    evidence: Mapping[str, Any],
) -> CorrectionRelation:
    return CorrectionRelation(
        relation_id=relation_id(source.doc_id, CORRECTION_RELATION_TYPE, target.doc_id),
        source_doc_id=source.doc_id,
        target_doc_id=target.doc_id,
        resolution_status=RESOLVED,
        resolution_source=rule,
        confidence=_CONFIDENCE[rule],
        evidence=dict(evidence),
    )


def _earlier(candidates: Sequence[DisclosureRecord], source: DisclosureRecord) -> list[DisclosureRecord]:
    return sorted(
        (
            candidate
            for candidate in candidates
            if candidate.doc_id != source.doc_id
            and candidate.order_key < source.order_key
        ),
        key=lambda record: record.order_key,
    )


def resolve_correction_edges(
    records: Sequence[DisclosureRecord],
    notices: Mapping[str, CorrectionNotice] | Iterable[CorrectionNotice] = (),
) -> list[CorrectionRelation]:
    """Resolve one edge per correcting disclosure, deterministically."""

    notice_by_doc: dict[str, CorrectionNotice] = (
        dict(notices)
        if isinstance(notices, Mapping)
        else {notice.doc_id: notice for notice in notices}
    )
    ordered = sorted(records, key=lambda record: record.order_key)

    by_corp: dict[str, list[DisclosureRecord]] = {}
    by_periodic_key: dict[tuple[Any, ...], list[DisclosureRecord]] = {}
    by_event_key: dict[tuple[Any, ...], list[DisclosureRecord]] = {}
    for record in ordered:
        by_corp.setdefault(record.corp_code, []).append(record)
        if record.doc_group == "periodic" and record.base_year is not None:
            key = (
                record.corp_code,
                record.doc_subtype,
                record.base_year,
                record.base_month,
            )
            by_periodic_key.setdefault(key, []).append(record)
        by_event_key.setdefault(
            (record.corp_code, record.doc_group, record.title_key), []
        ).append(record)

    relations: list[CorrectionRelation] = []
    for source in ordered:
        if not source.is_correction:
            continue
        notice = notice_by_doc.get(source.doc_id)
        relation = _resolve_one(
            source,
            notice,
            by_corp=by_corp,
            by_periodic_key=by_periodic_key,
            by_event_key=by_event_key,
        )
        relations.append(relation)
    return relations


def _resolve_one(
    source: DisclosureRecord,
    notice: CorrectionNotice | None,
    *,
    by_corp: Mapping[str, list[DisclosureRecord]],
    by_periodic_key: Mapping[tuple[Any, ...], list[DisclosureRecord]],
    by_event_key: Mapping[tuple[Any, ...], list[DisclosureRecord]],
) -> CorrectionRelation:
    evidence: dict[str, Any] = {"rules_tried": []}
    if notice is not None:
        evidence["notice"] = notice.to_dict()

    if notice is not None and notice.target_submitted_on:
        evidence["rules_tried"].append(SOURCE_CORRECTION_NOTICE)
        same_date = [
            candidate
            for candidate in _earlier(by_corp.get(source.corp_code, ()), source)
            if candidate.rcept_dt == notice.target_submitted_on
        ]
        same_group = [
            candidate for candidate in same_date if candidate.doc_group == source.doc_group
        ]
        pool = same_group or same_date
        evidence["notice_candidates"] = [candidate.doc_id for candidate in pool]
        if len(pool) == 1:
            return _resolved(
                source,
                pool[0],
                SOURCE_CORRECTION_NOTICE,
                {**evidence, "cross_group": not same_group},
            )
        if len(pool) > 1:
            # Several filings share the stated submission date, so the stated
            # target report name is used only to separate them.
            narrowed = [
                candidate
                for candidate in pool
                if _titles_match(candidate.report_nm, notice.target_report_nm)
            ] or [
                candidate
                for candidate in pool
                if candidate.title_key == source.title_key
            ]
            evidence["title_narrowed_candidates"] = [
                candidate.doc_id for candidate in narrowed
            ]
            if len(narrowed) == 1:
                return _resolved(
                    source,
                    narrowed[0],
                    SOURCE_CORRECTION_NOTICE_TITLE,
                    dict(evidence),
                )
            # The stated target exists but cannot be told apart from its
            # same-day twins.  A weaker rule must not overrule that evidence.
            return _ambiguous(source, SOURCE_MULTIPLE_CANDIDATES, evidence)
        # The correcting document names a target this corpus does not hold.
        # That is a known-missing original, not an unknown one, so no
        # title-based rule is allowed to attach it to a different document.
        evidence["notice_target_in_corpus"] = False
        return _unresolved(source, SOURCE_NOTICE_TARGET_MISSING, evidence)

    if source.doc_group == "periodic" and source.base_year is not None:
        evidence["rules_tried"].append(SOURCE_PERIODIC_PERIOD_KEY)
        key = (source.corp_code, source.doc_subtype, source.base_year, source.base_month)
        pool = _earlier(by_periodic_key.get(key, ()), source)
        evidence["periodic_candidates"] = [candidate.doc_id for candidate in pool]
        if pool:
            # One reporting period is one chain, so the filing immediately
            # before this correction is the document it supersedes.
            return _resolved(
                source,
                pool[-1],
                SOURCE_PERIODIC_PERIOD_KEY,
                {**evidence, "period_key": list(key)},
            )
        return _unresolved(source, SOURCE_NO_CANDIDATE, evidence)

    evidence["rules_tried"].append(SOURCE_EVENT_TITLE_KEY)
    key = (source.corp_code, source.doc_group, source.title_key)
    pool = _earlier(by_event_key.get(key, ()), source)
    evidence["event_candidates"] = [candidate.doc_id for candidate in pool]
    if len(pool) == 1:
        return _resolved(
            source, pool[0], SOURCE_EVENT_TITLE_KEY, {**evidence, "event_key": list(key)}
        )
    if len(pool) > 1:
        return _ambiguous(source, SOURCE_MULTIPLE_CANDIDATES, evidence)
    return _unresolved(source, SOURCE_NO_CANDIDATE, evidence)


def _demote(relation: CorrectionRelation, reason: str, note: str) -> CorrectionRelation:
    evidence = {**dict(relation.evidence), "rejected_target_doc_id": relation.target_doc_id, "reason": note}
    return replace(
        relation,
        relation_id=relation_id(relation.source_doc_id, relation.relation_type, None),
        target_doc_id=None,
        resolution_status=AMBIGUOUS,
        resolution_source=reason,
        confidence=0.0,
        evidence=evidence,
    )


def assemble_correction_groups(
    records: Sequence[DisclosureRecord],
    relations: Sequence[CorrectionRelation],
) -> tuple[list[CorrectionGroupMember], list[CorrectionRelation]]:
    """Turn resolved edges into groups, rejecting self-references and cycles.

    Returns the group members plus the relations after rejection, so a caller
    persists exactly what the graph accepted.
    """

    by_doc = {record.doc_id: record for record in records}
    accepted: dict[str, CorrectionRelation] = {}
    output: list[CorrectionRelation] = []

    for relation in relations:
        if relation.resolution_status != RESOLVED or relation.target_doc_id is None:
            output.append(relation)
            continue
        if relation.target_doc_id == relation.source_doc_id:
            output.append(_demote(relation, SOURCE_SELF_REFERENCE, "self_reference"))
            continue
        if relation.target_doc_id not in by_doc or relation.source_doc_id not in by_doc:
            output.append(_demote(relation, SOURCE_NO_CANDIDATE, "target_not_in_corpus"))
            continue
        if relation.source_doc_id in accepted:
            # One correcting document has exactly one parent; a duplicate edge
            # is dropped instead of creating a second relation row.
            output.append(_demote(relation, SOURCE_MULTIPLE_CANDIDATES, "duplicate_edge"))
            continue
        accepted[relation.source_doc_id] = relation
        output.append(relation)

    parent = {
        source: relation.target_doc_id for source, relation in accepted.items()
    }
    cyclic = _find_cyclic_nodes(parent)
    if cyclic:
        rewritten: list[CorrectionRelation] = []
        for relation in output:
            if (
                relation.resolution_status == RESOLVED
                and relation.source_doc_id in cyclic
            ):
                rewritten.append(_demote(relation, SOURCE_CYCLE_DETECTED, "cycle_detected"))
                accepted.pop(relation.source_doc_id, None)
                parent.pop(relation.source_doc_id, None)
            else:
                rewritten.append(relation)
        output = rewritten

    children: dict[str, list[str]] = {}
    for child, target in parent.items():
        children.setdefault(target, []).append(child)

    roots = {
        _walk_to_root(doc_id, parent) for doc_id in parent
    }
    members: list[CorrectionGroupMember] = []
    grouped: set[str] = set()
    for root_doc_id in sorted(roots):
        member_ids = _collect_descendants(root_doc_id, children)
        # Receipt order, with the root pinned first. Every rule already picks a
        # parent that precedes its child, so this only matters if a caller hands
        # in hand-built edges; it keeps "order 0 == root == no parent" true for
        # any input, which is the invariant the database also enforces.
        ordered = sorted(
            (by_doc[doc_id] for doc_id in member_ids if doc_id in by_doc),
            key=lambda record: (record.doc_id != root_doc_id, record.order_key),
        )
        last_index = len(ordered) - 1
        for index, record in enumerate(ordered):
            relation = accepted.get(record.doc_id)
            members.append(
                CorrectionGroupMember(
                    doc_id=record.doc_id,
                    correction_group_id=root_doc_id,
                    root_doc_id=root_doc_id,
                    parent_doc_id=parent.get(record.doc_id),
                    correction_order=index,
                    is_latest=index == last_index,
                    resolution_status=RESOLVED,
                    resolution_source=(
                        relation.resolution_source if relation else SOURCE_GROUP_ROOT
                    ),
                    confidence=relation.confidence if relation else 1.0,
                    is_correction=record.is_correction,
                    evidence=dict(relation.evidence) if relation else {},
                )
            )
            grouped.add(record.doc_id)

    for relation in output:
        if relation.resolution_status == RESOLVED or relation.source_doc_id in grouped:
            continue
        record = by_doc.get(relation.source_doc_id)
        if record is None:
            continue
        # An unresolved or ambiguous correction stands alone.  It is never
        # attached to the closest-looking group, so no resolved group's latest
        # document can be displaced by it.
        members.append(
            CorrectionGroupMember(
                doc_id=record.doc_id,
                correction_group_id=record.doc_id,
                root_doc_id=record.doc_id,
                parent_doc_id=None,
                correction_order=0,
                is_latest=True,
                resolution_status=relation.resolution_status,
                resolution_source=relation.resolution_source,
                confidence=relation.confidence,
                is_correction=record.is_correction,
                evidence=dict(relation.evidence),
            )
        )

    members.sort(key=lambda member: (member.correction_group_id, member.correction_order))
    # One row per edge identity, so a caller can persist the result behind a
    # unique key without ever inserting the same relation twice.
    deduplicated: dict[str, CorrectionRelation] = {}
    for relation in output:
        deduplicated.setdefault(relation.relation_id, relation)
    return members, list(deduplicated.values())


def _find_cyclic_nodes(parent: Mapping[str, str]) -> set[str]:
    cyclic: set[str] = set()
    state: dict[str, int] = {}
    for start in parent:
        path: list[str] = []
        node: str | None = start
        while node is not None and state.get(node, 0) == 0:
            state[node] = 1
            path.append(node)
            node = parent.get(node)
        if node is not None and state.get(node) == 1:
            cyclic.update(path[path.index(node):])
        for visited in path:
            state[visited] = 2
    return cyclic


def _walk_to_root(doc_id: str, parent: Mapping[str, str]) -> str:
    seen = {doc_id}
    current = doc_id
    while current in parent:
        current = parent[current]
        if current in seen:
            return current
        seen.add(current)
    return current


def _collect_descendants(root: str, children: Mapping[str, list[str]]) -> list[str]:
    collected = [root]
    queue = [root]
    seen = {root}
    while queue:
        node = queue.pop()
        for child in children.get(node, ()):
            if child in seen:
                continue
            seen.add(child)
            collected.append(child)
            queue.append(child)
    return collected


@dataclass(frozen=True)
class CorrectionDocumentState:
    """What retrieval needs to know about one document's correction status."""

    doc_id: str
    correction_group_id: str
    root_doc_id: str
    parent_doc_id: str | None
    correction_order: int
    is_latest: bool
    resolution_status: str
    is_correction: bool

    @property
    def is_resolved(self) -> bool:
        return self.resolution_status == RESOLVED

    @property
    def is_resolved_latest(self) -> bool:
        """The verified final document of an established chain.

        ``is_latest`` alone is not that: an ambiguous or unresolved correction
        is stored as a one-member group and carries ``is_latest`` too, because
        nothing is known to supersede it.  Ranking and policy must use this
        predicate so an unverified correction is never promoted as if its
        original had been identified.
        """

        return self.resolution_status == RESOLVED and self.is_latest

    @property
    def is_superseded(self) -> bool:
        return self.resolution_status == RESOLVED and not self.is_latest

    @property
    def is_group_root(self) -> bool:
        """True for the head of a chain -- and for a lone unresolved correction.

        Combine with :attr:`is_resolved` before reading it as "this is the
        original document".
        """

        return self.doc_id == self.root_doc_id

    @property
    def is_resolved_root(self) -> bool:
        return self.resolution_status == RESOLVED and self.is_group_root


def _state(member: CorrectionGroupMember) -> CorrectionDocumentState:
    return CorrectionDocumentState(
        doc_id=member.doc_id,
        correction_group_id=member.correction_group_id,
        root_doc_id=member.root_doc_id,
        parent_doc_id=member.parent_doc_id,
        correction_order=member.correction_order,
        is_latest=member.is_latest,
        resolution_status=member.resolution_status,
        is_correction=member.is_correction,
    )


class CorrectionGraph:
    """In-memory read API over the built graph.

    ``PostgresCorrectionRepository`` exposes the same four operations against
    the persisted tables, so callers depend on one contract regardless of where
    the graph lives.
    """

    def __init__(
        self,
        members: Sequence[CorrectionGroupMember] = (),
        relations: Sequence[CorrectionRelation] = (),
    ) -> None:
        self._members = tuple(members)
        self._relations = tuple(relations)
        self._by_doc = {member.doc_id: member for member in self._members}
        self._by_group: dict[str, list[CorrectionGroupMember]] = {}
        for member in self._members:
            self._by_group.setdefault(member.correction_group_id, []).append(member)
        for group_members in self._by_group.values():
            group_members.sort(key=lambda member: member.correction_order)

    @property
    def members(self) -> tuple[CorrectionGroupMember, ...]:
        return self._members

    @property
    def relations(self) -> tuple[CorrectionRelation, ...]:
        return self._relations

    def get_correction_group(self, doc_id: str) -> CorrectionGroup | None:
        member = self._by_doc.get(str(doc_id))
        if member is None:
            return None
        group_members = tuple(self._by_group.get(member.correction_group_id, ()))
        return CorrectionGroup(
            correction_group_id=member.correction_group_id,
            root_doc_id=member.root_doc_id,
            members=group_members,
            resolution_status=group_members[0].resolution_status
            if len(group_members) == 1
            else RESOLVED,
        )

    def get_correction_chain(self, doc_id: str) -> tuple[CorrectionGroupMember, ...]:
        """Original first, then every correction in receipt order."""

        group = self.get_correction_group(doc_id)
        return group.members if group is not None else ()

    def get_latest_report(
        self, doc_id: str | None = None, *, correction_group_id: str | None = None
    ) -> str | None:
        if correction_group_id is not None:
            group_members = self._by_group.get(str(correction_group_id), ())
            for member in group_members:
                if member.is_latest:
                    return member.doc_id
            return None
        if doc_id is None:
            raise ValueError("doc_id or correction_group_id is required")
        group = self.get_correction_group(doc_id)
        if group is None:
            # A document outside any group is its own latest valid version.
            return str(doc_id)
        return group.latest_doc_id

    def document_states(
        self, doc_ids: Iterable[str]
    ) -> dict[str, CorrectionDocumentState]:
        states: dict[str, CorrectionDocumentState] = {}
        for doc_id in doc_ids:
            member = self._by_doc.get(str(doc_id))
            if member is not None:
                states[str(doc_id)] = _state(member)
        return states

    def diagnostics(self) -> dict[str, Any]:
        return correction_diagnostics(self._members, self._relations)


def build_correction_graph(
    records: Sequence[DisclosureRecord] | Sequence[Mapping[str, Any]],
    notices: Mapping[str, CorrectionNotice] | Iterable[CorrectionNotice] = (),
) -> CorrectionGraph:
    """Resolve every correction edge and assemble the groups they imply."""

    normalized = [
        record
        if isinstance(record, DisclosureRecord)
        else DisclosureRecord.from_mapping(record)
        for record in records
    ]
    relations = resolve_correction_edges(normalized, notices)
    members, accepted = assemble_correction_groups(normalized, relations)
    return CorrectionGraph(members, accepted)


def correction_diagnostics(
    members: Sequence[CorrectionGroupMember],
    relations: Sequence[CorrectionRelation],
) -> dict[str, Any]:
    """Read-only invariants for the whole graph."""

    parent = {
        member.doc_id: member.parent_doc_id
        for member in members
        if member.parent_doc_id
    }
    by_group: dict[str, list[CorrectionGroupMember]] = {}
    for member in members:
        by_group.setdefault(member.correction_group_id, []).append(member)

    status_counts = {status: 0 for status in RESOLUTION_STATUSES}
    for relation in relations:
        status_counts[relation.resolution_status] = (
            status_counts.get(relation.resolution_status, 0) + 1
        )

    depths: dict[str, int] = {}
    for member in members:
        depth = 0
        node = member.doc_id
        seen = {node}
        while parent.get(node):
            node = parent[node]
            if node in seen:
                depth = -1
                break
            seen.add(node)
            depth += 1
        depths[member.doc_id] = depth

    resolved_groups = {
        group_id: group_members
        for group_id, group_members in by_group.items()
        if len(group_members) > 1
    }
    latest_violations = sum(
        1
        for group_members in resolved_groups.values()
        if sum(1 for member in group_members if member.is_latest) != 1
    )
    # A multi-hop chain is an original plus at least two corrections, whether
    # the second correction targets the first or the shared original.
    multi_hop = sum(
        1 for group_members in resolved_groups.values() if len(group_members) >= 3
    )
    relation_keys = [
        (relation.source_doc_id, relation.relation_type, relation.target_doc_id or "")
        for relation in relations
    ]
    duplicate_relations = len(relation_keys) - len(set(relation_keys))

    return {
        "correction_count": sum(1 for relation in relations),
        "resolved_relations": status_counts.get(RESOLVED, 0),
        "ambiguous_relations": status_counts.get(AMBIGUOUS, 0),
        "unresolved_relations": status_counts.get(UNRESOLVED, 0),
        "group_count": len(by_group),
        "resolved_group_count": len(resolved_groups),
        "singleton_group_count": len(by_group) - len(resolved_groups),
        "member_count": len(members),
        "multi_hop_chain_count": multi_hop,
        "max_chain_depth": max(depths.values(), default=0),
        "cycle_count": sum(1 for depth in depths.values() if depth < 0),
        "invalid_latest_group_count": latest_violations,
        "duplicate_relation_count": duplicate_relations,
        "self_reference_count": sum(
            1 for member in members if member.parent_doc_id == member.doc_id
        ),
        "resolution_sources": _counted(
            relation.resolution_source for relation in relations
        ),
    }


def _counted(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
