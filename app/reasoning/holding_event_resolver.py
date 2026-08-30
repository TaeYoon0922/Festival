"""Task-aware, read-only reconstruction of holding events from EvidenceSet."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from app.reasoning.evidence_builder import EvidenceGroup, EvidenceItem, EvidenceSet
from app.reasoning.holding_reporter import reporter_matches


_FIELD_NAMES = (
    "reporter",
    "reference_date",
    "report_date",
    "receipt_date",
    "before_shares",
    "change_shares",
    "after_shares",
    "before_ratio",
    "after_ratio",
    "change_ratio",
    "change_direction",
)
#: Acquisition facts are kept out of ``_FIELD_NAMES`` so the frozen completeness
#: and confidence denominators keep their exact values for every other event.
_ACQUISITION_FIELD_NAMES = (
    "acquisition_date",
    "acquired_shares",
)
_ALL_FIELD_NAMES = _FIELD_NAMES + _ACQUISITION_FIELD_NAMES
_NUMERIC_FIELDS = {
    "before_shares",
    "change_shares",
    "after_shares",
    "before_ratio",
    "after_ratio",
    "change_ratio",
    "acquired_shares",
}
_RATIO_FIELDS = {"before_ratio", "after_ratio", "change_ratio"}
_FIELD_LABELS: dict[str, tuple[str, ...]] = {
    "reporter": ("보고자/보유자", "보고자", "보유자"),
    "reference_date": ("기준일/보고일", "변동일", "기준일"),
    "report_date": ("보고일",),
    "receipt_date": (),
    "before_shares": ("직전 보유주식수", "변동전 주식수"),
    "change_shares": ("증감주식수", "증감 주식수"),
    "after_shares": ("보유주식수", "변동후 주식수"),
    "before_ratio": ("직전 보유비율", "변동전 비율"),
    "after_ratio": ("보유비율", "지분율", "변동후 비율"),
    "change_ratio": ("증감비율", "증감 비율"),
    "change_direction": ("change_direction", "변동방향"),
    # Read from the detail row itself, never from a projection field label.
    "acquisition_date": (),
    "acquired_shares": (),
}
_QUERY_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "reporter": (r"보고자(?:는|가|명|\s*성명|\s*누구)", r"보유자(?:는|가|명|\s*누구)"),
    "reference_date": (r"변동\s*일", r"기준\s*일", r"보고\s*일"),
    "before_shares": (r"변동\s*전.{0,8}주식", r"직전.{0,8}주식\s*수"),
    "change_shares": (
        r"증감.{0,6}주식\s*수",
        r"(?:증가|감소)(?!\s*후).{0,5}주식\s*수",
        # "증감 수량": the unit word is dropped in ordinary phrasing.
        r"증감\s*수량",
    ),
    "after_shares": (
        r"변동\s*후.{0,8}주식\s*수",
        r"(?:증가|감소)\s*후.{0,8}주식\s*수",
        r"보유\s*(?:주식\s*)?(?:수|수량)",
    ),
    "before_ratio": (r"변동\s*전.{0,8}비율", r"직전.{0,8}(?:비율|지분율)"),
    "after_ratio": (
        r"변동\s*후.{0,8}비율",
        r"보유\s*비율",
        r"지분\s*율",
        # "보유 수와 비율", "보유 수량과 비율": the second noun of a coordinated
        # pair keeps no qualifier of its own, so "보유 비율" never appears.
        r"(?:와|과)\s*비율",
    ),
    "change_ratio": (r"증감\s*비율", r"(?:증가|감소)\s*비율"),
    "change_direction": (r"증감\s*방향", r"증가", r"감소", r"변동\s*없음"),
    # "취득일", "취득 일자" -- the acquisition's own date, which is not
    # interchangeable with a report or base date.
    "acquisition_date": (r"취득\s*일",),
    # "취득 수량", "취득수량", "취득 주식수", "취득주식수".
    "acquired_shares": (r"취득\s*수량", r"취득\s*주식\s*수"),
}
#: The unit price is acquisition language this module deliberately cannot
#: answer: ``holding_acquisition`` keeps the "취득/처분단가" column out of every
#: extracted quantity and method.  It is recognised here only so callers can
#: tell that a question is about an acquisition, never as a requested field --
#: which is why it is absent from ``_QUERY_FIELD_ALIASES`` above.
_ACQUISITION_SEMANTIC_PATTERNS = (r"취득\s*(?:/\s*처분\s*)?단가",)
_QUERY_FIELD_CANONICAL = {
    "direction": "change_direction",
    "보고자/보유자": "reporter",
    "기준일/보고일": "reference_date",
    "직전 보유주식수": "before_shares",
    "증감주식수": "change_shares",
    "보유주식수": "after_shares",
    "직전 보유비율": "before_ratio",
    "보유비율": "after_ratio",
    "증감비율": "change_ratio",
    "변동방향": "change_direction",
}
_MISSING_VALUES = {
    "",
    "-",
    "--",
    "–",
    "—",
    "n/a",
    "na",
    "none",
    "null",
    "unknown",
    "미상",
    "해당없음",
}


@dataclass(frozen=True)
class NumericValue:
    raw: str
    normalized: int | float | None

    def to_dict(self) -> dict[str, Any]:
        return {"raw": self.raw, "normalized": self.normalized}


@dataclass(frozen=True)
class FieldSource:
    chunk_id: str
    doc_id: str
    source_refs: tuple[Mapping[str, Any], ...]
    projection_type: str | None
    retrieval_rank: int
    direct_field_ref: bool
    derived: bool = False

    def to_dict(self) -> dict[str, Any]:
        refs = copy.deepcopy(list(self.source_refs))
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_ref": refs[0] if refs else None,
            "source_refs": refs,
            "projection_type": self.projection_type,
            "retrieval_rank": self.retrieval_rank,
            "direct_field_ref": self.direct_field_ref,
            "derived": self.derived,
        }


@dataclass(frozen=True)
class FieldAlternative:
    value: str | NumericValue
    sources: tuple[FieldSource, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": _serialize_value(self.value),
            "sources": [source.to_dict() for source in self.sources],
        }


@dataclass(frozen=True)
class FieldProvenance:
    field_name: str
    value: str | NumericValue | None
    sources: tuple[FieldSource, ...]
    alternatives: tuple[FieldAlternative, ...]
    field_conflict: bool

    def to_dict(self) -> dict[str, Any]:
        sources = [source.to_dict() for source in self.sources]
        return {
            "field_name": self.field_name,
            "value": _serialize_value(self.value),
            "chunk_id": (
                sources[0]["chunk_id"] if sources and not self.field_conflict else None
            ),
            "source_ref": (
                sources[0]["source_ref"] if sources and not self.field_conflict else None
            ),
            "sources": sources,
            "alternatives": [value.to_dict() for value in self.alternatives],
            "field_conflict": self.field_conflict,
        }


@dataclass(frozen=True)
class HoldingEvent:
    company_id: str | None
    corp_code: str | None
    corp_name: str | None
    reporter: str | None
    reference_date: str | None
    report_date: str | None
    receipt_date: str | None
    before_shares: NumericValue | None
    change_shares: NumericValue | None
    after_shares: NumericValue | None
    before_ratio: NumericValue | None
    after_ratio: NumericValue | None
    change_ratio: NumericValue | None
    change_direction: str | None
    event_type: str
    doc_id: str | None
    doc_ids: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]
    source_refs: tuple[Mapping[str, Any], ...]
    field_provenance: Mapping[str, FieldProvenance]
    field_conflict: bool
    conflicting_fields: tuple[str, ...]
    temporal_match: bool | None
    direction_match: bool | None
    matches_query: bool | None
    confidence: Mapping[str, Any]
    completeness: Mapping[str, Any]
    warnings: tuple[str, ...]
    #: Present only for a detail row that proved its own acquisition, so every
    #: pre-existing event keeps exactly the shape it had.
    acquisition_date: str | None = None
    acquired_shares: NumericValue | None = None
    transaction_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "company_id": self.company_id,
            "corp_code": self.corp_code,
            "corp_name": self.corp_name,
            "reporter": self.reporter,
            "reference_date": self.reference_date,
            "report_date": self.report_date,
            "receipt_date": self.receipt_date,
            "before_shares": _serialize_value(self.before_shares),
            "change_shares": _serialize_value(self.change_shares),
            "after_shares": _serialize_value(self.after_shares),
            "before_ratio": _serialize_value(self.before_ratio),
            "after_ratio": _serialize_value(self.after_ratio),
            "change_ratio": _serialize_value(self.change_ratio),
            "change_direction": self.change_direction,
            **(
                {
                    "acquisition_date": self.acquisition_date,
                    "acquired_shares": _serialize_value(self.acquired_shares),
                    "transaction_method": self.transaction_method,
                }
                if self.transaction_method is not None
                else {}
            ),
            "event_type": self.event_type,
            "doc_id": self.doc_id,
            "doc_ids": list(self.doc_ids),
            "evidence_chunk_ids": list(self.evidence_chunk_ids),
            "source_refs": copy.deepcopy(list(self.source_refs)),
            "field_provenance": {
                field: provenance.to_dict()
                for field, provenance in self.field_provenance.items()
            },
            "field_conflict": self.field_conflict,
            "conflicting_fields": list(self.conflicting_fields),
            "temporal_match": self.temporal_match,
            "direction_match": self.direction_match,
            "matches_query": self.matches_query,
            "confidence": copy.deepcopy(dict(self.confidence)),
            "completeness": copy.deepcopy(dict(self.completeness)),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class HoldingResolution:
    question: str
    requested_fields: tuple[str, ...]
    events: tuple[HoldingEvent, ...]
    matching_event_count: int
    temporal_ambiguity: bool
    unresolved_fields: tuple[str, ...]
    warnings: tuple[str, ...]
    reporter_constraint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "requested_fields": list(self.requested_fields),
            "events": [event.to_dict() for event in self.events],
            "matching_event_count": self.matching_event_count,
            "temporal_ambiguity": self.temporal_ambiguity,
            "unresolved_fields": list(self.unresolved_fields),
            "warnings": list(self.warnings),
            "reporter_constraint": self.reporter_constraint,
        }


class HoldingEventResolver:
    """Resolve structured holding facts without changing evidence order."""

    def resolve(
        self, evidence_set: EvidenceSet, *, query_plan: Any | None = None
    ) -> HoldingResolution:
        return resolve_holding_events(evidence_set, query_plan=query_plan)


def resolve_holding_events(
    evidence_set: EvidenceSet, *, query_plan: Any | None = None
) -> HoldingResolution:
    plan = _plan_mapping(query_plan or evidence_set.query_plan)
    requested_fields = _requested_fields(evidence_set.question, plan)
    direction_constraint = _requested_direction(evidence_set.question)
    reporter_constraint = _text(plan.get("reporter"))
    explicit_temporal = _explicit_temporal_constraint(evidence_set, plan)

    groups = [
        group
        for group in evidence_set.evidence_groups
        if group.group_type == "holding_event"
    ]
    warnings = list(evidence_set.warnings)
    if not groups:
        warnings.append("no_holding_event_groups")

    events = tuple(
        _resolve_group(
            group,
            requested_fields=requested_fields,
            reporter_constraint=reporter_constraint,
            direction_constraint=direction_constraint,
            explicit_temporal=explicit_temporal,
            temporal_constraint=_temporal_constraint(evidence_set, plan),
        )
        for group in groups
    )
    matching_events = [event for event in events if event.matches_query is True]
    temporal_ambiguity = len(matching_events) > 1
    unresolved = tuple(
        field
        for field in requested_fields
        if not any(
            getattr(event, field) is not None and not event.field_provenance[field].field_conflict
            for event in events
            if field in event.field_provenance
        )
    )
    if temporal_ambiguity:
        warnings.append("multiple_matching_holding_events")
    if unresolved:
        warnings.append("unresolved_requested_fields")
    warnings.extend(
        warning for event in events for warning in event.warnings
    )
    return HoldingResolution(
        question=evidence_set.question,
        requested_fields=requested_fields,
        events=events,
        matching_event_count=len(matching_events),
        temporal_ambiguity=temporal_ambiguity,
        unresolved_fields=unresolved,
        warnings=tuple(dict.fromkeys(warnings)),
        reporter_constraint=reporter_constraint,
    )


@dataclass(frozen=True)
class _CandidateValue:
    raw: str
    normalized: Any
    source: FieldSource


def _resolve_group(
    group: EvidenceGroup,
    *,
    requested_fields: Sequence[str],
    reporter_constraint: str | None,
    direction_constraint: str | None,
    explicit_temporal: bool,
    temporal_constraint: Mapping[str, Any],
) -> HoldingEvent:
    provenance: dict[str, FieldProvenance] = {}
    resolved: dict[str, Any] = {}
    for field_name in _ALL_FIELD_NAMES:
        if field_name == "change_direction":
            continue
        field = _resolve_field(field_name, group.items)
        provenance[field_name] = field
        resolved[field_name] = field.value

    direction_field = _resolve_direction(group.items, provenance["change_shares"])
    provenance["change_direction"] = direction_field
    resolved["change_direction"] = direction_field.value

    conflicting = tuple(
        field for field, value in provenance.items() if value.field_conflict
    )
    event_warnings = [f"field_conflict:{field}" for field in conflicting]
    if "change_direction" in conflicting:
        event_warnings.append("change_direction_metadata_mismatch")

    reference_date = _string_value(resolved["reference_date"])
    temporal_match = _event_temporal_match(
        group.items,
        reference_date,
        explicit=explicit_temporal,
        constraint=temporal_constraint,
    )
    direction = _string_value(resolved["change_direction"])
    direction_match = (
        direction == direction_constraint if direction_constraint and direction else None
    )
    reporter = _string_value(resolved["reporter"])
    reporter_match = (
        _reporter_matches(reporter, reporter_constraint)
        if reporter_constraint and reporter
        else _conflicting_reporter_match(provenance["reporter"], reporter_constraint)
        if reporter_constraint
        else True
    )
    checks: list[bool | None] = [reporter_match]
    if explicit_temporal:
        checks.append(temporal_match)
    if direction_constraint:
        checks.append(direction_match)
    matches_query = _combine_matches(checks)

    source_refs = _unique_refs(
        [ref for item in group.items for ref in item.source_refs]
        + [
            ref
            for field in provenance.values()
            for source in field.sources
            for ref in source.source_refs
        ]
    )
    # Restricted to the frozen field set: acquisition facts must not move the
    # completeness or confidence numbers of any pre-existing event.
    resolved_count = sum(
        provenance[field].value is not None for field in _FIELD_NAMES
    )
    requested_resolved = sum(
        provenance[field].value is not None and not provenance[field].field_conflict
        for field in requested_fields
        if field in provenance
    )
    direct_count = sum(
        any(source.direct_field_ref for source in provenance[field].sources)
        for field in _FIELD_NAMES
        if provenance[field].value is not None
    )
    completeness = {
        "resolved_field_count": resolved_count,
        "total_field_count": len(_FIELD_NAMES),
        "resolved_ratio": round(resolved_count / len(_FIELD_NAMES), 6),
        "requested_field_count": len(requested_fields),
        "resolved_requested_field_count": requested_resolved,
        "requested_ratio": (
            round(requested_resolved / len(requested_fields), 6)
            if requested_fields
            else 1.0
        ),
    }
    confidence_ratio = (
        direct_count / resolved_count if resolved_count else 0.0
    )
    confidence = {
        "level": "high" if confidence_ratio >= 0.75 else "medium" if confidence_ratio >= 0.4 else "low",
        "direct_provenance_ratio": round(confidence_ratio, 6),
        "has_conflict": bool(conflicting),
    }
    doc_ids = tuple(dict.fromkeys(item.doc_id for item in group.items))
    company_id = _single_value(item.company_id for item in group.items)
    corp_code = _single_value(item.corp_code for item in group.items)
    corp_name = _single_value(item.corp_name for item in group.items)
    return HoldingEvent(
        company_id=company_id,
        corp_code=corp_code,
        corp_name=corp_name,
        reporter=reporter,
        reference_date=reference_date,
        report_date=_string_value(resolved["report_date"]),
        receipt_date=_string_value(resolved["receipt_date"]),
        before_shares=_numeric_value(resolved["before_shares"]),
        change_shares=_numeric_value(resolved["change_shares"]),
        after_shares=_numeric_value(resolved["after_shares"]),
        before_ratio=_numeric_value(resolved["before_ratio"]),
        after_ratio=_numeric_value(resolved["after_ratio"]),
        change_ratio=_numeric_value(resolved["change_ratio"]),
        change_direction=direction,
        event_type="holding_change",
        doc_id=doc_ids[0] if len(doc_ids) == 1 else None,
        doc_ids=doc_ids,
        evidence_chunk_ids=tuple(item.chunk_id for item in group.items),
        source_refs=source_refs,
        field_provenance=provenance,
        field_conflict=bool(conflicting),
        conflicting_fields=conflicting,
        temporal_match=temporal_match,
        direction_match=direction_match,
        matches_query=matches_query,
        confidence=confidence,
        completeness=completeness,
        warnings=tuple(event_warnings),
        acquisition_date=_string_value(resolved["acquisition_date"]),
        acquired_shares=_numeric_value(resolved["acquired_shares"]),
        transaction_method=_acquisition_method(group.items),
    )


def _resolve_field(
    field_name: str, items: Sequence[EvidenceItem]
) -> FieldProvenance:
    candidates = [
        candidate
        for item in items
        if (candidate := _field_candidate(field_name, item)) is not None
    ]
    return _resolve_candidates(field_name, candidates)


def _field_candidate(
    field_name: str, item: EvidenceItem
) -> _CandidateValue | None:
    holding = dict(item.holding)
    if field_name in _ACQUISITION_FIELD_NAMES:
        return _acquisition_candidate(field_name, item, holding)
    projection_fields = dict(holding.get("projection_fields") or {})
    source_chunk = dict(item.provenance.get("source_chunk") or {})
    label = next(
        (label for label in _FIELD_LABELS[field_name] if projection_fields.get(label) is not None),
        None,
    )
    if field_name == "receipt_date":
        value = item.rcept_dt
    elif field_name == "report_date":
        value = _first_text(holding.get("report_date"), source_chunk.get("report_date"))
    else:
        value = _first_text(
            projection_fields.get(label) if label else None,
            holding.get(field_name),
            source_chunk.get(field_name),
        )
    raw = _usable_raw(value)
    if raw is None:
        return None
    field_refs = dict(holding.get("projection_field_refs") or {})
    direct_refs = tuple(copy.deepcopy(list(field_refs.get(label) or []))) if label else ()
    refs = direct_refs or tuple(copy.deepcopy(list(item.source_refs)))
    normalized = _normalize_field_value(field_name, raw)
    return _CandidateValue(
        raw=raw,
        normalized=normalized,
        source=FieldSource(
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            source_refs=refs,
            projection_type=_text(holding.get("projection_type")),
            retrieval_rank=item.retrieval_rank,
            direct_field_ref=bool(direct_refs),
        ),
    )


def _acquisition_method(items: Sequence[EvidenceItem]) -> str | None:
    """The proving method, kept only when the group agrees on exactly one."""

    methods = {
        method
        for item in items
        if (method := _text(
            dict(dict(item.holding).get("acquisition") or {}).get(
                "transaction_method")))
    }
    return next(iter(methods)) if len(methods) == 1 else None


def _acquisition_candidate(
    field_name: str, item: EvidenceItem, holding: Mapping[str, Any]
) -> _CandidateValue | None:
    """An acquisition fact, or nothing -- the row either proves it or does not.

    ``holding["acquisition"]`` is present only for a detail row whose own
    transaction method names an acquisition and whose own signed change agrees,
    so a report summary, a disposal, and an unexplained increase all yield
    ``None`` here rather than a weaker answer.
    """

    acquisition = dict(holding.get("acquisition") or {})
    if not acquisition:
        return None
    raw = _usable_raw(acquisition.get(field_name))
    if raw is None:
        return None
    ref = acquisition.get("source_ref")
    # The reference points at the row the method was read from, so a citation
    # can never be satisfied by a neighbouring row of the same table.
    refs = (copy.deepcopy(ref),) if ref else ()
    return _CandidateValue(
        raw=raw,
        normalized=_normalize_field_value(field_name, raw),
        source=FieldSource(
            chunk_id=item.chunk_id,
            doc_id=item.doc_id,
            source_refs=refs or tuple(copy.deepcopy(list(item.source_refs))),
            projection_type=_text(holding.get("projection_type")),
            retrieval_rank=item.retrieval_rank,
            direct_field_ref=bool(refs),
        ),
    )


def _resolve_direction(
    items: Sequence[EvidenceItem], change_shares: FieldProvenance
) -> FieldProvenance:
    candidates: list[_CandidateValue] = []
    for item in items:
        metadata = _field_candidate("change_direction", item)
        if metadata is not None:
            metadata = _CandidateValue(
                raw=metadata.raw,
                normalized=_canonical_direction(metadata.raw),
                source=metadata.source,
            )
            if metadata.normalized:
                candidates.append(metadata)
    for alternative in change_shares.alternatives:
        value = alternative.value
        if not isinstance(value, NumericValue) or value.normalized is None:
            continue
        direction = (
            "increase"
            if value.normalized > 0
            else "decrease"
            if value.normalized < 0
            else "unchanged"
        )
        candidates.extend(
            _CandidateValue(
                raw=direction,
                normalized=direction,
                source=FieldSource(
                    chunk_id=source.chunk_id,
                    doc_id=source.doc_id,
                    source_refs=source.source_refs,
                    projection_type=source.projection_type,
                    retrieval_rank=source.retrieval_rank,
                    direct_field_ref=source.direct_field_ref,
                    derived=True,
                ),
            )
            for source in alternative.sources
        )
    return _resolve_candidates("change_direction", candidates)


def _resolve_candidates(
    field_name: str, candidates: Sequence[_CandidateValue]
) -> FieldProvenance:
    grouped: dict[Any, list[_CandidateValue]] = {}
    for candidate in candidates:
        key = _candidate_key(field_name, candidate)
        grouped.setdefault(key, []).append(candidate)
    alternatives: list[FieldAlternative] = []
    for values in grouped.values():
        representative = values[0]
        value: str | NumericValue
        if field_name in _NUMERIC_FIELDS:
            value = NumericValue(representative.raw, representative.normalized)
        else:
            value = str(representative.normalized or representative.raw)
        alternatives.append(
            FieldAlternative(
                value=value,
                sources=_unique_sources(value.source for value in values),
            )
        )
    alternatives.sort(
        key=lambda alternative: min(
            (source.retrieval_rank for source in alternative.sources), default=10**9
        )
    )
    conflict = len(alternatives) > 1
    resolved = alternatives[0].value if len(alternatives) == 1 else None
    sources = (
        alternatives[0].sources
        if len(alternatives) == 1
        else tuple(source for alternative in alternatives for source in alternative.sources)
    )
    return FieldProvenance(
        field_name=field_name,
        value=resolved,
        sources=sources,
        alternatives=tuple(alternatives),
        field_conflict=conflict,
    )


def _requested_fields(question: str, plan: Mapping[str, Any]) -> tuple[str, ...]:
    requested: list[str] = []
    configured = (
        plan.get("requested_holding_fields")
        or _nested(plan, "evidence", "requested_holding_fields")
        or []
    )
    if isinstance(configured, str):
        configured = [configured]
    for value in configured:
        canonical = _QUERY_FIELD_CANONICAL.get(str(value), str(value))
        if canonical in _ALL_FIELD_NAMES and canonical not in requested:
            requested.append(canonical)

    for field, patterns in _QUERY_FIELD_ALIASES.items():
        if any(re.search(pattern, question) for pattern in patterns) and field not in requested:
            requested.append(field)

    metric = _text(plan.get("metric"))
    if metric == "holding_ratio" and not any(
        field in requested for field in ("before_ratio", "after_ratio", "change_ratio")
    ):
        requested.append("after_ratio")
    elif metric == "holding_shares" and not any(
        field in requested for field in ("before_shares", "change_shares", "after_shares")
    ):
        requested.append("after_shares")
    return tuple(requested)


def has_acquisition_semantics(question: str, plan: Mapping[str, Any]) -> bool:
    """Whether a question belongs to the holding acquisition family.

    Wider than the acquisition fields this resolver can answer: it also covers
    the acquisition unit price, which the row parser deliberately excludes from
    every extracted quantity.  So a true result proves what the question is
    about, never that the answer exists.  Callers use it as an intent firewall,
    keeping acquisition wording from being reinterpreted as something else.

    The answerable half is read through ``_requested_fields`` rather than a
    second copy of its patterns, so this predicate cannot drift from it.
    """

    text = str(question or "")
    if any(re.search(pattern, text) for pattern in _ACQUISITION_SEMANTIC_PATTERNS):
        return True
    requested = _requested_fields(text, dict(plan or {}))
    return any(field in _ACQUISITION_FIELD_NAMES for field in requested)


def _requested_direction(question: str) -> str | None:
    compact = re.sub(r"\s+", "", question)
    if "감소" in compact:
        return "decrease"
    if "증가" in compact:
        return "increase"
    if any(value in compact for value in ("변동없음", "증감없음", "변화없음")):
        return "unchanged"
    return None


def _event_temporal_match(
    items: Sequence[EvidenceItem],
    reference_date: str | None,
    *,
    explicit: bool,
    constraint: Mapping[str, Any],
) -> bool | None:
    values = [item.temporal_match for item in items if item.temporal_match is not None]
    if True in values:
        return True
    if values and all(value is False for value in values):
        return False
    if not explicit or not reference_date:
        return None
    year = constraint.get("year")
    if year is not None and reference_date[:4] != str(year):
        return False
    quarter = constraint.get("quarter")
    if quarter is not None:
        month = _integer(reference_date[5:7])
        if month is None or (month - 1) // 3 + 1 != quarter:
            return False
    if constraint.get("from_date") and reference_date < constraint["from_date"]:
        return False
    if constraint.get("to_date") and reference_date > constraint["to_date"]:
        return False
    return True


def _explicit_temporal_constraint(
    evidence_set: EvidenceSet, plan: Mapping[str, Any]
) -> bool:
    return bool(_temporal_constraint(evidence_set, plan).get("explicit"))


def _temporal_constraint(
    evidence_set: EvidenceSet, plan: Mapping[str, Any]
) -> dict[str, Any]:
    period = plan.get("period")
    if hasattr(period, "to_dict"):
        period = period.to_dict()
    period = dict(period) if isinstance(period, Mapping) else {}
    year = _integer(period.get("year"))
    quarter = _integer(period.get("quarter"))
    from_date = _normalize_date(period.get("from") or period.get("from_date"))
    to_date = _normalize_date(period.get("to") or period.get("to_date"))
    calculated = {
        "explicit": any(
            value is not None for value in (year, quarter, from_date, to_date)
        ),
        "year": year,
        "quarter": quarter,
        "from_date": from_date,
        "to_date": to_date,
        "period_type": _text(period.get("period_type")),
    }
    if calculated["explicit"]:
        return calculated
    existing = evidence_set.ambiguity.get("temporal_constraint")
    if isinstance(existing, Mapping):
        return copy.deepcopy(dict(existing))
    return calculated


def _normalize_field_value(field_name: str, raw: str) -> Any:
    if field_name in _NUMERIC_FIELDS:
        return _normalize_number(raw, ratio=field_name in _RATIO_FIELDS)
    if field_name in {
        "reference_date", "report_date", "receipt_date", "acquisition_date",
    }:
        return _normalize_date(raw) or raw.strip()
    if field_name == "change_direction":
        return _canonical_direction(raw)
    return re.sub(r"\s+", " ", raw).strip()


def _normalize_number(raw: str, *, ratio: bool) -> int | float | None:
    value = raw.strip()
    if ratio:
        value = re.sub(r"\s*%\s*$", "", value)
    value = value.replace(",", "").replace(" ", "")
    negative_parentheses = value.startswith("(") and value.endswith(")")
    if negative_parentheses:
        value = f"-{value[1:-1]}"
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    if number == number.to_integral_value():
        return int(number)
    return float(number)


def _candidate_key(field_name: str, candidate: _CandidateValue) -> Any:
    if candidate.normalized is not None:
        return candidate.normalized
    return _normalize_text(candidate.raw) if field_name not in _NUMERIC_FIELDS else candidate.raw.strip()


def _canonical_direction(value: Any) -> str | None:
    compact = _normalize_text(value)
    if compact in {"increase", "increased", "positive", "증가", "증"}:
        return "increase"
    if compact in {"decrease", "decreased", "negative", "감소", "감"}:
        return "decrease"
    if compact in {"unchanged", "zero", "변동없음", "증감없음", "변화없음"}:
        return "unchanged"
    return None


def _reporter_matches(value: str, constraint: str) -> bool:
    """Delegates to the shared contract so the composer cannot drift from it."""

    return reporter_matches(value, constraint)


def _conflicting_reporter_match(
    provenance: FieldProvenance, constraint: str
) -> bool | None:
    """Whether a group holding several holder labels answers to the constraint.

    A group normally carries one holder, so this is reached only when evidence
    describing one event was spelled two ways and the field could not resolve.
    Every label must answer to the constraint: if even one names someone else,
    the group is not wholly about the holder that was asked for, and attributing
    it to them would be wrong.  Unknown, not False, when nothing is resolvable --
    the same answer this returned before there was anything to look at.
    """

    values = [
        text
        for alternative in provenance.alternatives
        if (text := _string_value(alternative.value))
    ]
    if not values:
        return None
    return all(_reporter_matches(value, constraint) for value in values)


def _combine_matches(values: Sequence[bool | None]) -> bool | None:
    if False in values:
        return False
    if all(value is True for value in values):
        return True
    return None


def _unique_refs(values: Any) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        key = (
            value.get("table_id"),
            value.get("row_start"),
            value.get("row_end"),
        )
        if key not in seen:
            seen.add(key)
            output.append(copy.deepcopy(dict(value)))
    return tuple(output)


def _unique_sources(values: Any) -> tuple[FieldSource, ...]:
    output: list[FieldSource] = []
    seen: set[tuple[Any, ...]] = set()
    for value in values:
        key = (
            value.chunk_id,
            tuple(
                (
                    ref.get("table_id"),
                    ref.get("row_start"),
                    ref.get("row_end"),
                )
                for ref in value.source_refs
            ),
            value.derived,
        )
        if key not in seen:
            seen.add(key)
            output.append(value)
    return tuple(output)


def _single_value(values: Any) -> str | None:
    unique = list(dict.fromkeys(str(value) for value in values if value))
    return unique[0] if len(unique) == 1 else None


def _usable_raw(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    return None if raw.casefold() in _MISSING_VALUES else raw


def _normalize_date(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(
        r"((?:19|20)\d{2})\s*(?:년|[.\-/])\s*(\d{1,2})\s*"
        r"(?:월|[.\-/])\s*(\d{1,2})(?:\s*일)?",
        str(value),
    )
    if not match:
        compact = re.fullmatch(r"((?:19|20)\d{2})(\d{2})(\d{2})", str(value).strip())
        match = compact
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _serialize_value(value: Any) -> Any:
    return value.to_dict() if isinstance(value, NumericValue) else value


def _numeric_value(value: Any) -> NumericValue | None:
    return value if isinstance(value, NumericValue) else None


def _string_value(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _plan_mapping(plan: Any) -> dict[str, Any]:
    if hasattr(plan, "to_dict"):
        return copy.deepcopy(dict(plan.to_dict()))
    if isinstance(plan, Mapping):
        return copy.deepcopy(dict(plan))
    raise TypeError("query_plan must be a QueryPlan or mapping")


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _first_text(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _text(value: Any) -> str | None:
    return str(value) if value is not None and str(value).strip() else None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value).casefold())


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
