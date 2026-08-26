"""Single domain contract for deterministic corporate-event lifecycles.

Matching belongs in :mod:`app.reasoning.corporate_event_graph`. This module
owns only shared enums, value objects, identity functions, invariants, and the
interfaces consumed by the builder, repository, and resolver.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, TypeVar

from app.reasoning.correction_graph import CorrectionGroup


class EventFamily(str, Enum):
    SUPPLY_CONTRACT = "supply_contract"
    TREASURY_TRUST_CONTRACT = "treasury_trust_contract"


class EventResolutionStatus(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


class EventLifecycleStatus(str, Enum):
    OPEN = "open"
    TERMINATED = "terminated"


class EventMemberRole(str, Enum):
    CONTRACT = "contract"
    CONTRACT_UPDATE = "contract_update"
    TERMINATION = "termination"


class EventRelationType(str, Enum):
    """Only relations for which the audited v1 corpus has evidence."""

    BELONGS_TO_EVENT = "belongs_to_event"
    TERMINATES_EVENT = "terminates_event"


FAMILY_SUPPLY_CONTRACT = EventFamily.SUPPLY_CONTRACT.value
FAMILY_TREASURY_TRUST = EventFamily.TREASURY_TRUST_CONTRACT.value
EVENT_FAMILIES = tuple(item.value for item in EventFamily)

RESOLVED = EventResolutionStatus.RESOLVED.value
AMBIGUOUS = EventResolutionStatus.AMBIGUOUS.value
UNRESOLVED = EventResolutionStatus.UNRESOLVED.value
RESOLUTION_STATUSES = tuple(item.value for item in EventResolutionStatus)

LIFECYCLE_OPEN = EventLifecycleStatus.OPEN.value
LIFECYCLE_TERMINATED = EventLifecycleStatus.TERMINATED.value
LIFECYCLE_STATUSES = tuple(item.value for item in EventLifecycleStatus)

ROLE_CONTRACT = EventMemberRole.CONTRACT.value
ROLE_CONTRACT_UPDATE = EventMemberRole.CONTRACT_UPDATE.value
ROLE_TERMINATION = EventMemberRole.TERMINATION.value
MEMBER_ROLES = tuple(item.value for item in EventMemberRole)

RELATION_BELONGS_TO_EVENT = EventRelationType.BELONGS_TO_EVENT.value
RELATION_TERMINATES_EVENT = EventRelationType.TERMINATES_EVENT.value
RELATION_TYPES = tuple(item.value for item in EventRelationType)


class CorrectionGraphView(Protocol):
    """The P0-A public surface needed by a P0-B builder."""

    def get_correction_group(self, doc_id: str) -> CorrectionGroup | None: ...


_EnumT = TypeVar("_EnumT", bound=Enum)


def _enum_value(enum_type: type[_EnumT], value: _EnumT | str, field_name: str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(str(value))
    except ValueError as error:
        allowed = ", ".join(str(item.value) for item in enum_type)
        raise ValueError(
            f"invalid {field_name}: {value!r}; expected one of {allowed}"
        ) from error


def _required(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _confidence(value: float) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return number


def _iso_date(value: str | date | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.isoformat() if isinstance(value, date) else str(value)
    try:
        return date.fromisoformat(normalized).isoformat()
    except ValueError as error:
        raise ValueError(f"{field_name} must be an ISO date") from error


def _reference_digest(source_doc_id: str, reference: Any) -> str:
    payload = json.dumps(
        {
            "source_doc_id": _required(source_doc_id, "source_doc_id"),
            "reference": reference,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def unresolved_event_anchor(
    termination_doc_id: str, explicit_external_reference: Any | None = None
) -> str:
    """Stable source anchor when the opening contract is outside the corpus."""

    if explicit_external_reference is None:
        return f"unresolved:{_required(termination_doc_id, 'termination_doc_id')}"
    digest = _reference_digest(termination_doc_id, explicit_external_reference)
    return f"unresolved:{digest}"


def ambiguous_event_anchor(
    source_doc_id: str, source_reference: Any | None = None
) -> str:
    """Stable source anchor that never chooses one ambiguous root candidate."""

    return f"ambiguous:{_reference_digest(source_doc_id, source_reference)}"


def deterministic_event_id(
    corp_code: str,
    event_family: EventFamily | str,
    root_logical_key: str,
) -> str:
    """Hash company, family, and the lifecycle's immutable root anchor.

    Later contract updates and termination filings do not participate in the
    hash. A resolved P0-A correction group uses its ``correction_group_id`` as
    ``root_logical_key``; an ordinary opening disclosure uses its ``doc_id``.
    """

    family = _enum_value(EventFamily, event_family, "event_family")
    raw = "\x1f".join(
        [
            _required(corp_code, "corp_code"),
            str(family.value),
            _required(root_logical_key, "root_logical_key"),
        ]
    )
    return f"evt_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def deterministic_relation_id(
    source_doc_id: str,
    relation_type: EventRelationType | str,
    target_doc_id: str | None,
) -> str:
    relation = _enum_value(EventRelationType, relation_type, "relation_type")
    raw = "\x1f".join(
        [
            _required(source_doc_id, "source_doc_id"),
            str(relation.value),
            str(target_doc_id or ""),
        ]
    )
    return f"evr_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class CorporateEventMember:
    """One disclosure's role and P0-A provenance inside a lifecycle."""

    doc_id: str
    canonical_doc_id: str
    member_role: EventMemberRole | str
    member_order: int
    event_id: str = ""
    corp_code: str | None = None
    event_date: str | date | None = None
    root_doc_id: str | None = None
    correction_group_id: str | None = None
    correction_resolution_status: EventResolutionStatus | str | None = None
    correction_chain: tuple[str, ...] = ()
    is_correction: bool = False
    confidence: float = 0.0
    provenance: Mapping[str, Any] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _required(self.doc_id, "doc_id"))
        object.__setattr__(
            self,
            "canonical_doc_id",
            _required(self.canonical_doc_id, "canonical_doc_id"),
        )
        object.__setattr__(
            self,
            "member_role",
            _enum_value(EventMemberRole, self.member_role, "member_role"),
        )
        if self.event_id:
            object.__setattr__(self, "event_id", _required(self.event_id, "event_id"))
        if self.corp_code is not None:
            object.__setattr__(self, "corp_code", _required(self.corp_code, "corp_code"))
        if self.member_order < 0:
            raise ValueError("member_order must be non-negative")
        object.__setattr__(self, "event_date", _iso_date(self.event_date, "event_date"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "correction_chain",
            tuple(_required(item, "correction_chain item") for item in self.correction_chain),
        )
        object.__setattr__(self, "provenance", dict(self.provenance))
        object.__setattr__(self, "evidence", dict(self.evidence))

        status = self.correction_resolution_status
        if status is not None:
            status = _enum_value(
                EventResolutionStatus, status, "correction_resolution_status"
            )
            object.__setattr__(self, "correction_resolution_status", status)

        has_group_metadata = any(
            (
                self.correction_group_id is not None,
                status is not None,
                bool(self.correction_chain),
            )
        )
        if has_group_metadata:
            if not self.correction_group_id or status is None or not self.correction_chain:
                raise ValueError(
                    "correction_group_id, correction_resolution_status, and "
                    "correction_chain must be supplied together"
                )
            root_doc_id = _required(
                self.root_doc_id or "", "root_doc_id for a correction group"
            )
            object.__setattr__(self, "root_doc_id", root_doc_id)
            if len(set(self.correction_chain)) != len(self.correction_chain):
                raise ValueError("correction_chain must not contain duplicate documents")
            if root_doc_id not in self.correction_chain:
                raise ValueError("root_doc_id must be present in correction_chain")
            if self.doc_id not in self.correction_chain:
                raise ValueError("doc_id must be present in correction_chain")
            if status is EventResolutionStatus.RESOLVED:
                if self.canonical_doc_id != self.correction_chain[-1]:
                    raise ValueError(
                        "a resolved correction member must use the chain latest as canonical"
                    )
            elif self.canonical_doc_id != self.doc_id:
                raise ValueError(
                    "ambiguous or unresolved corrections cannot use a different "
                    "canonical document"
                )
        else:
            if self.root_doc_id not in (None, self.doc_id):
                raise ValueError("a non-correction member's root_doc_id must equal doc_id")
            object.__setattr__(self, "root_doc_id", self.doc_id)

    @property
    def logical_key(self) -> str:
        if (
            self.correction_group_id
            and self.correction_resolution_status is EventResolutionStatus.RESOLVED
        ):
            return self.correction_group_id
        return self.doc_id

    @classmethod
    def from_correction_graph(
        cls,
        *,
        corp_code: str,
        doc_id: str,
        member_role: EventMemberRole | str,
        correction_graph: CorrectionGraphView | None,
        member_order: int = 0,
        event_id: str = "",
        event_date: str | date | None = None,
        confidence: float = 0.0,
        provenance: Mapping[str, Any] | None = None,
        evidence: Mapping[str, Any] | None = None,
    ) -> "CorporateEventMember":
        """Reuse P0-A without promoting an unverified latest document."""

        doc_id = _required(doc_id, "doc_id")
        group = correction_graph.get_correction_group(doc_id) if correction_graph else None
        if group is None:
            return cls(
                event_id=event_id,
                corp_code=corp_code,
                doc_id=doc_id,
                canonical_doc_id=doc_id,
                member_role=member_role,
                member_order=member_order,
                event_date=event_date,
                confidence=confidence,
                provenance=dict(provenance or {}),
                evidence=dict(evidence or {}),
            )

        chain = tuple(str(member.doc_id) for member in group.members)
        status = _enum_value(
            EventResolutionStatus, group.resolution_status, "P0-A resolution_status"
        )
        canonical_doc_id = (
            group.resolved_latest_doc_id
            if status is EventResolutionStatus.RESOLVED
            else None
        ) or doc_id
        source_member = next(
            (item for item in group.members if str(item.doc_id) == doc_id), None
        )
        return cls(
            event_id=event_id,
            corp_code=corp_code,
            doc_id=doc_id,
            canonical_doc_id=str(canonical_doc_id),
            member_role=member_role,
            member_order=member_order,
            event_date=event_date,
            root_doc_id=str(group.root_doc_id),
            correction_group_id=str(group.correction_group_id),
            correction_resolution_status=status,
            correction_chain=chain,
            is_correction=bool(getattr(source_member, "is_correction", False)),
            confidence=confidence,
            provenance=dict(provenance or {}),
            evidence=dict(evidence or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "corp_code": self.corp_code,
            "doc_id": self.doc_id,
            "canonical_doc_id": self.canonical_doc_id,
            "member_role": str(self.member_role.value),
            "member_order": self.member_order,
            "event_date": self.event_date,
            "root_doc_id": self.root_doc_id,
            "correction_group_id": self.correction_group_id,
            "correction_resolution_status": (
                str(self.correction_resolution_status.value)
                if self.correction_resolution_status is not None
                else None
            ),
            "correction_chain": list(self.correction_chain),
            "is_correction": self.is_correction,
            "confidence": self.confidence,
            "provenance": dict(self.provenance),
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CorporateEvent:
    """A validated lifecycle anchored to its opening/root logical contract."""

    event_id: str
    corp_code: str
    event_family: EventFamily | str
    root_logical_key: str
    lifecycle_status: EventLifecycleStatus | str
    resolution_status: EventResolutionStatus | str
    resolution_source: str
    members: tuple[CorporateEventMember, ...] = ()
    opened_at: str | date | None = None
    closed_at: str | date | None = None
    confidence: float = 0.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _required(self.event_id, "event_id"))
        object.__setattr__(self, "corp_code", _required(self.corp_code, "corp_code"))
        object.__setattr__(
            self,
            "event_family",
            _enum_value(EventFamily, self.event_family, "event_family"),
        )
        object.__setattr__(
            self,
            "root_logical_key",
            _required(self.root_logical_key, "root_logical_key"),
        )
        object.__setattr__(
            self,
            "lifecycle_status",
            _enum_value(EventLifecycleStatus, self.lifecycle_status, "lifecycle_status"),
        )
        object.__setattr__(
            self,
            "resolution_status",
            _enum_value(EventResolutionStatus, self.resolution_status, "resolution_status"),
        )
        object.__setattr__(
            self, "resolution_source", _required(self.resolution_source, "resolution_source")
        )
        object.__setattr__(self, "opened_at", _iso_date(self.opened_at, "opened_at"))
        object.__setattr__(self, "closed_at", _iso_date(self.closed_at, "closed_at"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "evidence", dict(self.evidence))

        members = tuple(
            replace(member, event_id=self.event_id) if not member.event_id else member
            for member in self.members
        )
        object.__setattr__(self, "members", members)
        if not members:
            raise ValueError("a corporate event must contain at least one member")
        if any(member.event_id != self.event_id for member in members):
            raise ValueError("every member must carry its containing event_id")

        doc_ids = [member.doc_id for member in members]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError("duplicate document membership is not allowed")
        member_orders = [member.member_order for member in members]
        if len(member_orders) != len(set(member_orders)):
            raise ValueError("member_order must be unique inside an event")

        member_corps = {
            member.corp_code
            or str((member.evidence.get("document") or {}).get("corp_code") or "")
            for member in members
        }
        member_corps.discard("")
        if member_corps and member_corps != {self.corp_code}:
            raise ValueError("cross-company event membership is not allowed")

        expected_id = deterministic_event_id(
            self.corp_code, self.event_family, self.root_logical_key
        )
        if self.event_id != expected_id:
            raise ValueError(f"event_id must equal deterministic id {expected_id}")

        has_termination = any(
            member.member_role is EventMemberRole.TERMINATION for member in members
        )
        if self.lifecycle_status is EventLifecycleStatus.TERMINATED and not has_termination:
            raise ValueError("a terminated event requires a termination member")
        if self.lifecycle_status is EventLifecycleStatus.OPEN and has_termination:
            raise ValueError("an open event cannot contain a termination member")
        if self.lifecycle_status is EventLifecycleStatus.OPEN and self.closed_at is not None:
            raise ValueError("an open event cannot have closed_at")

    @classmethod
    def create(
        cls,
        *,
        corp_code: str,
        event_family: EventFamily | str,
        root_logical_key: str,
        lifecycle_status: EventLifecycleStatus | str,
        resolution_status: EventResolutionStatus | str,
        resolution_source: str,
        members: Sequence[CorporateEventMember],
        opened_at: str | date | None = None,
        closed_at: str | date | None = None,
        confidence: float = 0.0,
        evidence: Mapping[str, Any] | None = None,
    ) -> "CorporateEvent":
        identifier = deterministic_event_id(corp_code, event_family, root_logical_key)
        return cls(
            event_id=identifier,
            corp_code=corp_code,
            event_family=event_family,
            root_logical_key=root_logical_key,
            lifecycle_status=lifecycle_status,
            resolution_status=resolution_status,
            resolution_source=resolution_source,
            members=tuple(members),
            opened_at=opened_at,
            closed_at=closed_at,
            confidence=confidence,
            evidence=dict(evidence or {}),
        )

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(member.doc_id for member in self.members)

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def is_terminated(self) -> bool:
        return self.lifecycle_status is EventLifecycleStatus.TERMINATED

    @property
    def is_resolved(self) -> bool:
        return (
            self.resolution_status is EventResolutionStatus.RESOLVED
            and len(self.members) > 1
        )

    @property
    def termination_members(self) -> tuple[CorporateEventMember, ...]:
        return tuple(
            member
            for member in self.members
            if member.member_role is EventMemberRole.TERMINATION
        )

    @property
    def contract_members(self) -> tuple[CorporateEventMember, ...]:
        return tuple(
            member
            for member in self.members
            if member.member_role is not EventMemberRole.TERMINATION
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "corp_code": self.corp_code,
            "event_family": str(self.event_family.value),
            "root_logical_key": self.root_logical_key,
            "lifecycle_status": str(self.lifecycle_status.value),
            "resolution_status": str(self.resolution_status.value),
            "resolution_source": self.resolution_source,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "confidence": self.confidence,
            "member_count": self.member_count,
            "is_resolved": self.is_resolved,
            "members": [member.to_dict() for member in self.members],
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CorporateEventRelation:
    """Document-to-document evidence edge inside an event."""

    relation_id: str
    source_doc_id: str
    relation_type: EventRelationType | str
    resolution_status: EventResolutionStatus | str
    resolution_source: str
    target_doc_id: str | None = None
    event_id: str | None = None
    confidence: float = 0.0
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "relation_id", _required(self.relation_id, "relation_id"))
        object.__setattr__(self, "source_doc_id", _required(self.source_doc_id, "source_doc_id"))
        object.__setattr__(
            self,
            "relation_type",
            _enum_value(EventRelationType, self.relation_type, "relation_type"),
        )
        object.__setattr__(
            self,
            "resolution_status",
            _enum_value(EventResolutionStatus, self.resolution_status, "resolution_status"),
        )
        object.__setattr__(
            self, "resolution_source", _required(self.resolution_source, "resolution_source")
        )
        if self.event_id is not None:
            object.__setattr__(self, "event_id", _required(self.event_id, "event_id"))
        if self.target_doc_id is not None:
            object.__setattr__(
                self, "target_doc_id", _required(self.target_doc_id, "target_doc_id")
            )
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(self, "evidence", dict(self.evidence))

        if self.target_doc_id == self.source_doc_id:
            raise ValueError("a corporate event relation cannot reference itself")
        is_resolved = self.resolution_status is EventResolutionStatus.RESOLVED
        if is_resolved != (self.target_doc_id is not None):
            raise ValueError("only a resolved relation may name a target document")
        expected_id = deterministic_relation_id(
            self.source_doc_id, self.relation_type, self.target_doc_id
        )
        if self.relation_id != expected_id:
            raise ValueError(f"relation_id must equal deterministic id {expected_id}")

    @classmethod
    def create(
        cls,
        *,
        event_id: str | None,
        source_doc_id: str,
        relation_type: EventRelationType | str,
        resolution_status: EventResolutionStatus | str,
        resolution_source: str,
        target_doc_id: str | None = None,
        confidence: float = 0.0,
        evidence: Mapping[str, Any] | None = None,
    ) -> "CorporateEventRelation":
        return cls(
            relation_id=deterministic_relation_id(
                source_doc_id, relation_type, target_doc_id
            ),
            event_id=event_id,
            source_doc_id=source_doc_id,
            target_doc_id=target_doc_id,
            relation_type=relation_type,
            resolution_status=resolution_status,
            resolution_source=resolution_source,
            confidence=confidence,
            evidence=dict(evidence or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation_id": self.relation_id,
            "event_id": self.event_id,
            "source_doc_id": self.source_doc_id,
            "target_doc_id": self.target_doc_id,
            "relation_type": str(self.relation_type.value),
            "resolution_status": str(self.resolution_status.value),
            "resolution_source": self.resolution_source,
            "confidence": self.confidence,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class CorporateEventState:
    """Compact per-document view shared by graph, repository, and resolver."""

    doc_id: str
    event_id: str
    corp_code: str
    event_family: EventFamily | str
    member_role: EventMemberRole | str
    lifecycle_status: EventLifecycleStatus | str
    resolution_status: EventResolutionStatus | str
    canonical_doc_id: str
    member_count: int
    correction_group_id: str | None = None
    correction_resolution_status: EventResolutionStatus | str | None = None

    @property
    def is_resolved(self) -> bool:
        return self.resolution_status == RESOLVED and self.member_count > 1

    @property
    def is_terminated(self) -> bool:
        return self.lifecycle_status == LIFECYCLE_TERMINATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "event_id": self.event_id,
            "corp_code": self.corp_code,
            "event_family": self.event_family,
            "member_role": self.member_role,
            "lifecycle_status": self.lifecycle_status,
            "resolution_status": self.resolution_status,
            "canonical_doc_id": self.canonical_doc_id,
            "member_count": self.member_count,
            "correction_group_id": self.correction_group_id,
            "correction_resolution_status": self.correction_resolution_status,
            "is_resolved": self.is_resolved,
            "is_terminated": self.is_terminated,
        }


@dataclass(frozen=True)
class CorporateEventBuildResult:
    """Output boundary for the next-stage deterministic builder."""

    events: tuple[CorporateEvent, ...] = ()
    relations: tuple[CorporateEventRelation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "relations", tuple(self.relations))
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate event_id is not allowed")
        relation_ids = [relation.relation_id for relation in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("duplicate relation is not allowed")

        memberships: set[str] = set()
        for event in self.events:
            for member in event.members:
                if member.doc_id in memberships:
                    raise ValueError("a document cannot belong to multiple events")
                memberships.add(member.doc_id)


class CorporateEventBuilder(Protocol):
    """Interface only; matching remains in ``corporate_event_graph.py``."""

    def build(self) -> CorporateEventBuildResult: ...


__all__ = [
    "AMBIGUOUS",
    "CorporateEvent",
    "CorporateEventBuilder",
    "CorporateEventBuildResult",
    "CorporateEventMember",
    "CorporateEventRelation",
    "CorporateEventState",
    "CorrectionGraphView",
    "EVENT_FAMILIES",
    "EventFamily",
    "EventLifecycleStatus",
    "EventMemberRole",
    "EventRelationType",
    "EventResolutionStatus",
    "FAMILY_SUPPLY_CONTRACT",
    "FAMILY_TREASURY_TRUST",
    "LIFECYCLE_OPEN",
    "LIFECYCLE_STATUSES",
    "LIFECYCLE_TERMINATED",
    "MEMBER_ROLES",
    "RELATION_BELONGS_TO_EVENT",
    "RELATION_TERMINATES_EVENT",
    "RELATION_TYPES",
    "RESOLVED",
    "RESOLUTION_STATUSES",
    "ROLE_CONTRACT",
    "ROLE_CONTRACT_UPDATE",
    "ROLE_TERMINATION",
    "UNRESOLVED",
    "ambiguous_event_anchor",
    "deterministic_event_id",
    "deterministic_relation_id",
    "unresolved_event_anchor",
]
