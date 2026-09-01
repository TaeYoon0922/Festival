"""Deterministic, read-only evidence grouping over frozen retrieval results."""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from app.reasoning.holding_acquisition import acquisition_facts
from app.reasoning.holding_correction_state import (
    PRIOR_STATE,
    PRIOR_STATE_SUPERSEDED,
    current_state_is_authoritative,
    declared_correction_state,
    item_correction_state,
)
from app.retrieval.interfaces import CandidateChunk, RetrievalResult


_QUARTER_BY_BASE_MONTH = {
    3: 1,
    6: 2,
    9: 3,
    12: 4,
}


@dataclass(frozen=True)
class EvidenceItem:
    chunk_id: str
    doc_id: str
    company_id: str | None
    corp_code: str | None
    corp_name: str | None
    doc_group: str | None
    chunk_type: str | None
    section_path: tuple[str, ...]
    evidence_text: str
    retrieval_rank: int
    retrieval_score: float
    rcept_dt: str | None
    report_nm: str | None
    period: Mapping[str, Any]
    source_refs: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any]
    holding: Mapping[str, Any]
    temporal_match: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "company_id": self.company_id,
            "corp_code": self.corp_code,
            "corp_name": self.corp_name,
            "doc_group": self.doc_group,
            "chunk_type": self.chunk_type,
            "section_path": list(self.section_path),
            "evidence_text": self.evidence_text,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": self.retrieval_score,
            "rcept_dt": self.rcept_dt,
            "report_nm": self.report_nm,
            "period": copy.deepcopy(dict(self.period)),
            "source_refs": copy.deepcopy(list(self.source_refs)),
            "provenance": copy.deepcopy(dict(self.provenance)),
            "holding": copy.deepcopy(dict(self.holding)),
            "temporal_match": self.temporal_match,
        }


@dataclass(frozen=True)
class EvidenceGroup:
    group_id: str
    group_type: str
    member_chunk_ids: tuple[str, ...]
    primary_evidence: EvidenceItem
    supporting_evidence: tuple[EvidenceItem, ...]
    doc_ids: tuple[str, ...]
    reason: str

    @property
    def items(self) -> tuple[EvidenceItem, ...]:
        return (self.primary_evidence, *self.supporting_evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_type": self.group_type,
            "member_chunk_ids": list(self.member_chunk_ids),
            "primary_evidence": self.primary_evidence.to_dict(),
            "supporting_evidence": [
                item.to_dict() for item in self.supporting_evidence
            ],
            "doc_ids": list(self.doc_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class EvidenceSet:
    question: str
    query_plan: Mapping[str, Any]
    task_type: str | None
    evidence_groups: tuple[EvidenceGroup, ...]
    retrieval_order: tuple[str, ...]
    raw_candidate_count: int
    selected_evidence_count: int
    warnings: tuple[str, ...]
    ambiguity: Mapping[str, Any]

    @property
    def served_items(self) -> tuple[EvidenceItem, ...]:
        """Every served item once, in retrieval order.

        Grouping can place one item in more than one group; a consumer that
        needs the served evidence itself -- a field-evidence producer reading
        the structured row behind a chunk -- wants each item exactly once, and
        wants them in the order they were served rather than the order the
        groups happened to be built in.
        """

        by_id = {
            item.chunk_id: item
            for group in self.evidence_groups
            for item in group.items
        }
        ordered = [
            by_id.pop(chunk_id)
            for chunk_id in self.retrieval_order
            if chunk_id in by_id
        ]
        return tuple([*ordered, *by_id.values()])

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "query_plan": copy.deepcopy(dict(self.query_plan)),
            "task_type": self.task_type,
            "evidence_groups": [group.to_dict() for group in self.evidence_groups],
            "retrieval_order": list(self.retrieval_order),
            "raw_candidate_count": self.raw_candidate_count,
            "selected_evidence_count": self.selected_evidence_count,
            "warnings": list(self.warnings),
            "ambiguity": copy.deepcopy(dict(self.ambiguity)),
        }


class EvidenceBuilder:
    """Build an EvidenceSet from a completed production retrieval execution."""

    def build(
        self,
        execution: Any,
        *,
        question: str | None = None,
        grouping_intent: str | None = None,
    ) -> EvidenceSet:
        """Build the EvidenceSet for one execution.

        ``grouping_intent`` lets the caller state which grouping strategy the
        resolver it has already chosen needs.  Omitting it keeps the plan's own
        task type as the grouping strategy, which is what every caller did
        before the argument existed.
        """

        plan = execution.plan
        return build_evidence_set(
            question=question or _plan_question(plan),
            query_plan=plan,
            candidates=execution.chunks,
            results=execution.results,
            grouping_intent=grouping_intent,
        )


def build_evidence_set(
    *,
    question: str,
    query_plan: Any,
    candidates: Sequence[CandidateChunk],
    results: Sequence[RetrievalResult],
    grouping_intent: str | None = None,
) -> EvidenceSet:
    """Build evidence without mutating, filtering, or reranking the inputs.

    ``task_type`` reports what query understanding decided and is what the
    EvidenceSet carries downstream.  ``grouping_intent`` is a separate question:
    which grouping the resolver chosen for this execution can consume.  They are
    usually the same, and are deliberately allowed to differ -- a question the
    plan reads as a plain disclosure lookup can still be executed as a holding
    event, and the groups have to match the resolver, not the plan.  When no
    intent is given the plan's task type is used, so an omitted argument
    reproduces the previous behaviour exactly.
    """

    plan = _plan_mapping(query_plan)
    task_type = _text(plan.get("task_type"))
    grouping_task = _text(grouping_intent) or task_type
    candidate_by_id = {candidate.chunk_id: candidate for candidate in candidates}
    warnings: list[str] = []
    seen: set[str] = set()
    items: list[EvidenceItem] = []
    for result in results:
        if result.chunk_id in seen:
            warnings.append(f"duplicate_retrieval_chunk:{result.chunk_id}")
            continue
        seen.add(result.chunk_id)
        candidate = candidate_by_id.get(result.chunk_id)
        if candidate is None:
            warnings.append(f"missing_candidate_chunk:{result.chunk_id}")
            continue
        items.append(_evidence_item(candidate, result))

    temporal_constraint = _temporal_constraint(plan)
    items = [
        replace(
            item,
            temporal_match=_temporal_match(item, temporal_constraint),
        )
        for item in items
    ]
    # A corrected filing reprints the report it corrects, so its own 정정 전
    # region states values it has itself superseded. Settle that before the
    # groups are built: the resolver reads a group's fields as equally
    # authoritative statements of one event, and the superseded half was never
    # that. Nothing is dropped -- the chunk stays served, cited and readable,
    # and only its standing as this event's evidence changes.
    superseded, superseded_warnings = _superseded_prior_states(items, plan)
    warnings.extend(superseded_warnings)
    groups = _group_items(
        items, task_type=grouping_task, superseded_holding_ids=superseded
    )
    ambiguity = _ambiguity_metadata(groups, temporal_constraint)
    if ambiguity["temporal_ambiguity"]:
        warnings.append("multiple_temporal_alternatives")
    if (
        temporal_constraint["explicit"]
        and ambiguity["matching_event_count"] == 0
        and ambiguity["temporal_alternative_count"] > 0
    ):
        warnings.append("explicit_temporal_constraint_unmatched")

    return EvidenceSet(
        question=str(question),
        query_plan=copy.deepcopy(plan),
        task_type=task_type,
        evidence_groups=tuple(groups),
        retrieval_order=tuple(result.chunk_id for result in results),
        raw_candidate_count=len(results),
        selected_evidence_count=len(items),
        warnings=tuple(dict.fromkeys(warnings)),
        ambiguity=ambiguity,
    )


def _evidence_item(
    candidate: CandidateChunk, result: RetrievalResult
) -> EvidenceItem:
    chunk = copy.deepcopy(dict(candidate.chunk))
    section_path = chunk.get("section_path") or []
    if isinstance(section_path, str):
        section_path = [section_path]
    company_id = _first_text(chunk.get("company_id"), chunk.get("corp_code"))
    corp_code = _first_text(chunk.get("corp_code"), chunk.get("company_id"))
    source_refs = tuple(copy.deepcopy(list(chunk.get("source_refs") or [])))
    provenance = {
        "source_chunk_id": candidate.chunk_id,
        "source_doc_id": candidate.doc_id,
        "source_refs": copy.deepcopy(list(source_refs)),
        "source_table_id": chunk.get("source_table_id"),
        "source_table_ids": copy.deepcopy(list(chunk.get("source_table_ids") or [])),
        "table_id": chunk.get("table_id"),
        "projection_type": chunk.get("projection_type"),
        "projection_field_refs": copy.deepcopy(
            dict(chunk.get("projection_field_refs") or {})
        ),
        "source_chunk": chunk,
    }
    return EvidenceItem(
        chunk_id=candidate.chunk_id,
        doc_id=candidate.doc_id,
        company_id=company_id,
        corp_code=corp_code,
        corp_name=_first_text(chunk.get("corp_name"), chunk.get("listed_name")),
        doc_group=_text(chunk.get("doc_group")),
        chunk_type=_text(chunk.get("chunk_type")),
        section_path=tuple(str(value) for value in section_path),
        evidence_text=str(
            chunk.get("retrieval_text") or chunk.get("content") or ""
        ),
        retrieval_rank=int(result.rank),
        retrieval_score=_retrieval_score(result),
        rcept_dt=_text(chunk.get("rcept_dt")),
        report_nm=_text(chunk.get("report_nm")),
        period=_period_metadata(chunk),
        source_refs=source_refs,
        provenance=provenance,
        holding=_holding_metadata(chunk),
    )


def _group_items(
    items: Sequence[EvidenceItem],
    *,
    task_type: str | None,
    superseded_holding_ids: frozenset[str] = frozenset(),
) -> list[EvidenceGroup]:
    remaining = list(items)
    groups: list[EvidenceGroup] = []

    if task_type == "holding_change":
        holding_groups, remaining = _holding_groups(
            remaining, superseded_holding_ids
        )
        groups.extend(holding_groups)

    periodic_groups, remaining = _periodic_groups(remaining)
    groups.extend(periodic_groups)

    document_groups, remaining = _same_document_groups(remaining)
    groups.extend(document_groups)
    groups.extend(
        _make_group(
            [item],
            group_type="standalone_evidence",
            reason="No conservative grouping relation was found.",
        )
        for item in remaining
    )
    return sorted(
        groups,
        key=lambda group: (
            group.primary_evidence.retrieval_rank,
            group.primary_evidence.chunk_id,
        ),
    )


def _holding_groups(
    items: Sequence[EvidenceItem],
    superseded_holding_ids: frozenset[str] = frozenset(),
) -> tuple[list[EvidenceGroup], list[EvidenceItem]]:
    """Holding-event groups, and everything that is not one of their members.

    ``superseded_holding_ids`` names projections a filing's own correction-state
    labels proved it no longer states.  They take no part in a holding event --
    neither as a seed nor as a member reached through one -- and fall through to
    the grouping every other chunk gets, so the evidence set keeps them.
    """

    eligible = [
        item for item in items if item.chunk_id not in superseded_holding_ids
    ]
    seeds = [item for item in eligible if _is_holding_evidence(item)]
    holding_items = [
        item
        for item in eligible
        if item in seeds
        or any(
            item.doc_id == seed.doc_id and _source_related(item, seed)
            for seed in seeds
        )
    ]
    nonholding = [item for item in items if item not in holding_items]
    components = _holding_components(holding_items)
    return (
        [
            _make_group(
                component,
                group_type="holding_event",
                reason=(
                    "Same-document holding evidence linked by event fields and/or "
                    "source table provenance; dates are never merged when they differ."
                ),
            )
            for component in components
        ],
        nonholding,
    )


def _superseded_prior_states(
    items: Sequence[EvidenceItem], plan: Mapping[str, Any]
) -> tuple[frozenset[str], list[str]]:
    """Which served projections their own filing has already superseded.

    A projection qualifies on three counts, all of them structural.  Its own
    labels declare it the filing's prior state.  The question is not one that
    asks for that state -- a correction history and an original-filing question
    both want it, and are refused here rather than repaired downstream.  And the
    same filing also serves a projection of the same event that it does *not*
    declare prior, so resolving one state away can never leave the event without
    one.

    Two projections that both declare the prior state, or a filing that serves
    the prior state alone, change nothing: the point is not to remove a
    disagreement but to stop treating a filing's own superseded text as a rival
    to what it now says.  Where the labels prove nothing, the conflict stands.
    """

    if not current_state_is_authoritative(plan):
        return frozenset(), []

    holding = [item for item in items if _is_holding_evidence(item)]
    prior = [item for item in holding if item_correction_state(item) == PRIOR_STATE]
    if not prior:
        return frozenset(), []

    superseded: list[str] = []
    for item in prior:
        if any(
            other.chunk_id != item.chunk_id
            and item_correction_state(other) != PRIOR_STATE
            # ``_same_holding_event`` is the frozen relation, so a correction
            # state can only supersede inside the one filing and the one event
            # the grouping would have merged it into anyway.
            and _same_holding_event(item, other)
            for other in holding
        ):
            superseded.append(item.chunk_id)
    return (
        frozenset(superseded),
        [f"{PRIOR_STATE_SUPERSEDED}:{chunk_id}" for chunk_id in sorted(superseded)],
    )


def _holding_components(items: Sequence[EvidenceItem]) -> list[list[EvidenceItem]]:
    """Group related items without allowing an undated provenance bridge."""

    components: list[list[EvidenceItem]] = []
    for item in sorted(items, key=lambda value: (value.retrieval_rank, value.chunk_id)):
        destination = next(
            (
                component
                for component in components
                if _holding_component_compatible(item, component)
                and any(_same_holding_event(item, member) for member in component)
            ),
            None,
        )
        if destination is None:
            components.append([item])
        else:
            destination.append(item)
    return components


def _holding_component_compatible(
    item: EvidenceItem, component: Sequence[EvidenceItem]
) -> bool:
    dates = {_holding_date(member) for member in component if _holding_date(member)}
    item_date = _holding_date(item)
    if item_date and dates and item_date not in dates:
        return False
    reporters = {
        reporter
        for reporter in (
            _normalized_value(member.holding.get("reporter"))
            for member in component
        )
        if reporter
    }
    item_reporter = _normalized_value(item.holding.get("reporter"))
    return not (item_reporter and reporters and item_reporter not in reporters)


def _periodic_groups(
    items: Sequence[EvidenceItem],
) -> tuple[list[EvidenceGroup], list[EvidenceItem]]:
    buckets: dict[tuple[str, str, str], list[EvidenceItem]] = {}
    for item in items:
        if item.doc_group != "periodic":
            continue
        company = item.corp_code or item.corp_name or ""
        fingerprint = _fact_fingerprint(item)
        section = _section_key(item.section_path)
        if company and section and len(fingerprint) >= 20:
            buckets.setdefault((company, section, fingerprint), []).append(item)

    grouped_ids: set[str] = set()
    groups: list[EvidenceGroup] = []
    for bucket in buckets.values():
        periods = {_item_period_signature(item) for item in bucket}
        periods.discard(None)
        doc_ids = {item.doc_id for item in bucket}
        if len(bucket) < 2 or len(periods) < 2 or len(doc_ids) < 2:
            continue
        groups.append(
            _make_group(
                bucket,
                group_type="periodic_repeated_fact",
                reason=(
                    "Exact normalized fact text and section match for the same company "
                    "across distinct reporting periods; every period is retained."
                ),
            )
        )
        grouped_ids.update(item.chunk_id for item in bucket)
    return groups, [item for item in items if item.chunk_id not in grouped_ids]


def _same_document_groups(
    items: Sequence[EvidenceItem],
) -> tuple[list[EvidenceGroup], list[EvidenceItem]]:
    components = _connected_components(items, _same_document_relation)
    grouped = [component for component in components if len(component) > 1]
    grouped_ids = {item.chunk_id for group in grouped for item in group}
    return (
        [
            _make_group(
                component,
                group_type="document_evidence",
                reason=(
                    "Same-document evidence linked by source references or an exact "
                    "normalized fact in the same section."
                ),
            )
            for component in grouped
        ],
        [item for item in items if item.chunk_id not in grouped_ids],
    )


def _same_holding_event(left: EvidenceItem, right: EvidenceItem) -> bool:
    if left.doc_id != right.doc_id:
        return False
    left_date = _holding_date(left)
    right_date = _holding_date(right)
    if left_date and right_date and left_date != right_date:
        return False
    left_reporter = _normalized_value(left.holding.get("reporter"))
    right_reporter = _normalized_value(right.holding.get("reporter"))
    if left_reporter and right_reporter and left_reporter != right_reporter:
        return False
    if _source_related(left, right):
        return True
    return bool(
        left_date
        and right_date
        and left_date == right_date
        and (left_reporter or right_reporter)
    )


def _same_document_relation(left: EvidenceItem, right: EvidenceItem) -> bool:
    if left.doc_id != right.doc_id:
        return False
    if _source_related(left, right):
        return True
    left_fingerprint = _fact_fingerprint(left)
    return bool(
        len(left_fingerprint) >= 20
        and left_fingerprint == _fact_fingerprint(right)
        and _section_key(left.section_path) == _section_key(right.section_path)
    )


def _source_related(left: EvidenceItem, right: EvidenceItem) -> bool:
    left_tables = _source_table_ids(left)
    right_tables = _source_table_ids(right)
    return bool(left_tables.intersection(right_tables))


def _source_table_ids(item: EvidenceItem) -> set[str]:
    provenance = item.provenance
    values = {
        provenance.get("table_id"),
        provenance.get("source_table_id"),
        *(provenance.get("source_table_ids") or []),
    }
    values.update(
        ref.get("table_id")
        for ref in item.source_refs
        if isinstance(ref, Mapping)
    )
    return {str(value) for value in values if value}


def _connected_components(
    items: Sequence[EvidenceItem], relation: Any
) -> list[list[EvidenceItem]]:
    if not items:
        return []
    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            if relation(items[left], items[right]):
                union(left, right)
    grouped: dict[int, list[EvidenceItem]] = {}
    for index, item in enumerate(items):
        grouped.setdefault(find(index), []).append(item)
    return list(grouped.values())


def _make_group(
    items: Sequence[EvidenceItem], *, group_type: str, reason: str
) -> EvidenceGroup:
    ordered = sorted(items, key=lambda item: (item.retrieval_rank, item.chunk_id))
    chunk_ids = tuple(item.chunk_id for item in ordered)
    digest = hashlib.sha256(
        f"{group_type}|{'|'.join(sorted(chunk_ids))}".encode("utf-8")
    ).hexdigest()[:16]
    return EvidenceGroup(
        group_id=f"{group_type}:{digest}",
        group_type=group_type,
        member_chunk_ids=chunk_ids,
        primary_evidence=ordered[0],
        supporting_evidence=tuple(ordered[1:]),
        doc_ids=tuple(dict.fromkeys(item.doc_id for item in ordered)),
        reason=reason,
    )


def _ambiguity_metadata(
    groups: Sequence[EvidenceGroup], constraint: Mapping[str, Any]
) -> dict[str, Any]:
    temporal_group_ids: list[str] = []
    alternative_keys: set[str] = set()
    matching_group_ids: list[str] = []
    for group in groups:
        group_keys = {
            signature
            for signature in (_temporal_signature(item) for item in group.items)
            if signature
        }
        if group.group_type == "holding_event" and group_keys:
            temporal_group_ids.append(group.group_id)
            alternative_keys.add(f"holding:{group.group_id}:{'|'.join(sorted(group_keys))}")
        elif group.group_type == "periodic_repeated_fact":
            temporal_group_ids.append(group.group_id)
            alternative_keys.update(
                f"periodic:{group.group_id}:{key}" for key in group_keys
            )
        if any(item.temporal_match is True for item in group.items):
            matching_group_ids.append(group.group_id)

    explicit = bool(constraint.get("explicit"))
    alternative_count = len(alternative_keys)
    matching_count = len(matching_group_ids) if explicit else alternative_count
    temporal_ambiguity = (
        len(matching_group_ids) > 1 if explicit else alternative_count > 1
    )
    return {
        "temporal_ambiguity": temporal_ambiguity,
        "matching_event_count": matching_count,
        "temporal_alternative_count": alternative_count,
        "temporal_constraint": copy.deepcopy(dict(constraint)),
        "temporal_group_ids": temporal_group_ids,
        "matching_group_ids": matching_group_ids,
    }


def _temporal_constraint(plan: Mapping[str, Any]) -> dict[str, Any]:
    period = plan.get("period")
    if hasattr(period, "to_dict"):
        period = period.to_dict()
    period = dict(period) if isinstance(period, Mapping) else {}
    year = _integer(period.get("year"))
    quarter = _integer(period.get("quarter"))
    from_date = _normalize_date(period.get("from") or period.get("from_date"))
    to_date = _normalize_date(period.get("to") or period.get("to_date"))
    return {
        "explicit": any(value is not None for value in (year, quarter, from_date, to_date)),
        "year": year,
        "quarter": quarter,
        "from_date": from_date,
        "to_date": to_date,
        "period_type": _text(period.get("period_type")),
    }


def _temporal_match(
    item: EvidenceItem, constraint: Mapping[str, Any]
) -> bool | None:
    if not constraint.get("explicit"):
        return None
    period_type = str(constraint.get("period_type") or "")
    if period_type.startswith("holding_reference"):
        item_date = _holding_date(item)
    elif period_type == "receipt_date":
        item_date = _normalize_date(item.rcept_dt)
    else:
        item_date = _period_date(item.period)
    item_year = _item_year(item, item_date)
    if constraint.get("year") is not None:
        if item_year is None:
            return None
        if item_year != constraint["year"]:
            return False
    item_quarter = _temporal_item_quarter(item, item_date)
    if constraint.get("quarter") is not None:
        if item_quarter is None:
            return None
        if item_quarter != constraint["quarter"]:
            return False
    if constraint.get("from_date") or constraint.get("to_date"):
        if not item_date:
            return None
        if constraint.get("from_date") and item_date < constraint["from_date"]:
            return False
        if constraint.get("to_date") and item_date > constraint["to_date"]:
            return False
    return True


def _holding_metadata(chunk: Mapping[str, Any]) -> dict[str, Any]:
    fields = dict(chunk.get("projection_fields") or {})
    change_shares = _first_text(
        fields.get("증감주식수"), chunk.get("change_shares")
    )
    direction = _first_text(chunk.get("change_direction"))
    if not direction and change_shares:
        compact = change_shares.replace(",", "").strip()
        if compact.startswith("-"):
            direction = "decrease"
        elif re.search(r"\d", compact) and compact not in {"0", "0.0"}:
            direction = "increase"
    return {
        "reporter": _first_text(
            fields.get("보고자/보유자"), chunk.get("reporter")
        ),
        "reference_date": _normalize_date(
            _first_text(
                fields.get("기준일/보고일"),
                chunk.get("reference_date"),
                chunk.get("change_date"),
            )
        ),
        "before_shares": _first_text(
            fields.get("직전 보유주식수"), chunk.get("before_shares")
        ),
        "change_shares": change_shares,
        "after_shares": _first_text(
            fields.get("보유주식수"), chunk.get("after_shares")
        ),
        "holding_ratio": _first_text(
            fields.get("보유비율"), chunk.get("holding_ratio")
        ),
        "change_ratio": _first_text(
            fields.get("증감비율"), chunk.get("change_ratio")
        ),
        "change_direction": direction,
        "projection_type": _text(chunk.get("projection_type")),
        "projection_state": _text(chunk.get("projection_state")),
        # Which state of a corrected filing this projection states, read from
        # the filing's own labels. ``None`` for the body of a report and for
        # every filing that corrects nothing, which is nearly all of them.
        "correction_state": declared_correction_state(chunk),
        "projection_fields": copy.deepcopy(fields),
        "projection_field_refs": copy.deepcopy(
            dict(chunk.get("projection_field_refs") or {})
        ),
        # Present only when this one row proves an acquisition on its own.  The
        # facts are carried together because the transaction method is what
        # makes the date and the quantity mean "acquired", and that only holds
        # for the row they were read from.
        "acquisition": acquisition_facts(chunk),
    }


def _period_metadata(chunk: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "year",
        "base_year",
        "base_month",
        "fiscal_year",
        "quarter",
        "period",
        "period_type",
        "from_date",
        "to_date",
        "period_start",
        "period_end",
        "report_period",
        "basis_period",
        "period_labels",
        "statement_scope",
    )
    period = {
        key: copy.deepcopy(chunk[key]) for key in keys if chunk.get(key) is not None
    }
    return period


def _is_holding_evidence(item: EvidenceItem) -> bool:
    return bool(
        item.holding.get("projection_type") in {"holding_detail_row", "holding_report"}
        or item.holding.get("reporter")
        or item.holding.get("reference_date")
    )


def _holding_date(item: EvidenceItem) -> str | None:
    return _normalize_date(item.holding.get("reference_date"))


def _fact_fingerprint(item: EvidenceItem) -> str:
    source_chunk = item.provenance.get("source_chunk") or {}
    text = source_chunk.get("content") or item.evidence_text
    return _normalized_value(text)


def _section_key(path: Sequence[str]) -> str:
    return _normalized_value(path[-1] if path else "")


def _item_period_signature(item: EvidenceItem) -> str | None:
    year = _item_year(item, _period_date(item.period))
    quarter = _stored_item_quarter(item)
    period_type = _first_text(
        item.period.get("period_type"), item.period.get("basis_period")
    )
    if year is None and quarter is None and not period_type:
        return None
    return f"{year or '?'}:{quarter or '?'}:{_normalized_value(period_type)}"


def _temporal_signature(item: EvidenceItem) -> str | None:
    if item.doc_group == "holding" or item.holding.get("reference_date"):
        return _holding_date(item)
    return _item_period_signature(item)


def _period_date(period: Mapping[str, Any]) -> str | None:
    for key in ("to_date", "to", "period_end", "report_period"):
        value = _normalize_date(period.get(key))
        if value:
            return value
    return None


def _item_year(item: EvidenceItem, item_date: str | None) -> int | None:
    for key in ("fiscal_year", "base_year", "year"):
        value = _integer(item.period.get(key))
        if value is not None:
            return value
    return _integer(item_date[:4]) if item_date else None


def _stored_item_quarter(item: EvidenceItem) -> int | None:
    value = _integer(item.period.get("quarter"))
    if value is not None and 1 <= value <= 4:
        return value
    return None


def _temporal_item_quarter(item: EvidenceItem, item_date: str | None) -> int | None:
    value = _integer(item.period.get("quarter"))
    if value is not None and 1 <= value <= 4:
        return value
    value = _QUARTER_BY_BASE_MONTH.get(_integer(item.period.get("base_month")))
    if value is not None:
        return value
    if item_date:
        month = _integer(item_date[5:7])
        if month:
            return (month - 1) // 3 + 1
    return None


def _retrieval_score(result: RetrievalResult) -> float:
    metadata = dict(result.metadata_match or {})
    hybrid = dict(metadata.get("hybrid") or {})
    components = dict(metadata.get("score_components") or {})
    for value in (
        hybrid.get("final_score"),
        components.get("final_score"),
        result.bm25_score,
    ):
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0


def _plan_mapping(plan: Any) -> dict[str, Any]:
    if hasattr(plan, "to_dict"):
        return copy.deepcopy(dict(plan.to_dict()))
    if isinstance(plan, Mapping):
        return copy.deepcopy(dict(plan))
    raise TypeError("query_plan must be a QueryPlan or mapping")


def _plan_question(plan: Any) -> str:
    if isinstance(plan, Mapping):
        return str(plan.get("raw_query") or plan.get("query") or "")
    return str(getattr(plan, "raw_query", None) or getattr(plan, "query", ""))


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(
        r"((?:19|20)\d{2})\s*(?:년|[.\-/])\s*(\d{1,2})\s*"
        r"(?:월|[.\-/])\s*(\d{1,2})(?:\s*일)?",
        str(value),
    )
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _normalized_value(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value).casefold())


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _text(value: Any) -> str | None:
    return str(value) if value is not None and str(value) else None


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
