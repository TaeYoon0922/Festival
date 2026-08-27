"""Deterministic query validation and clarification for the P0-D firewall.

This module decides whether a :class:`~app.reasoning.query_plan.QueryPlan` is
safe to execute.  It performs no retrieval and never asks a model to establish
facts about a company, a filing, a correction chain, or an event lifecycle.
Semantic fallback output may fill only unresolved semantic slots and is always
validated here before it can reach retrieval.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.reasoning.query_plan import QueryPeriod, QueryPlan


class QueryState(str, Enum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"
    OUT_OF_SCOPE = "out_of_scope"


class QuerySlotSource(str, Enum):
    DETERMINISTIC = "deterministic"
    HCX_FALLBACK = "hcx_fallback"
    USER_CLARIFICATION = "user_clarification"


class QuerySlotStatus(str, Enum):
    RESOLVED = "resolved"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


ALLOWED_TASK_TYPES = frozenset(
    {
        "financial_metric",
        "corporate_event",
        "holding_change",
        "disclosure_lookup",
        "correction_lookup",
        "enumeration",
        "enumeration_plus_event",
        "periodic_fact",
        "holding_event",
        "general_evidence",
        "exchange_event",
        "business_product",
        "listing_history",
        "merger_history",
    }
)

# QueryUnderstanding's actual event vocabulary.  HCX is not allowed to invent
# a family outside this set.
ALLOWED_EVENT_FAMILIES = frozenset(
    {
        "capital_increase",
        "convertible_bond",
        "treasury_share_disposal",
        "treasury_share_trust_termination",
        "treasury_share_trust_contract",
        "write_down_contingent_capital_security",
        "spin_off",
        "merger",
        "supply_contract",
        "contract_termination",
        "facility_investment",
    }
)

ALLOWED_OPERATIONS = frozenset(
    {
        "lookup_metric",
        "lookup_holding",
        "lookup_disclosure",
        "inspect_event",
        "find_terminated",
        "enumerate",
        "correction_lookup",
        "compare",
        "latest",
        "lifecycle_status",
    }
)

_EVENT_ROUTES = {
    "capital_increase": ("major",),
    "convertible_bond": ("major",),
    "treasury_share_disposal": ("major",),
    "treasury_share_trust_termination": ("major",),
    "treasury_share_trust_contract": ("major",),
    "write_down_contingent_capital_security": ("major",),
    "spin_off": ("major",),
    "merger": ("major", "periodic"),
    "supply_contract": ("exchange",),
    "contract_termination": ("exchange",),
    "facility_investment": ("exchange",),
}

_SET_MARKERS = (
    "몇 건", "몇건", "몇 개", "몇개", "모두", "전체", "전부", "목록", "나열",
    "중에서", "가운데", "들 중",
)
_LIFECYCLE_MARKERS = ("해지", "종료", "취소")
_EXISTENTIAL_MARKERS = ("있는가", "있나", "존재", "없는가", "있습니까")

_UNSUPPORTED_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("future_prediction", ("전망해", "예측해", "예상 주가", "미래 주가", "오를까", "내릴까")),
    ("investment_advice", ("매수해", "매도해", "투자해도", "추천 종목", "투자 판단")),
    ("personal_or_private_news", ("개인 신상", "사생활", "루머", "찌라시")),
    (
        "external_side_effect",
        (
            "메일 보내", "이메일 보내", "이메일로 보내", "문서 수정", "문서를 수정",
            "공시 삭제", "대신 주문",
        ),
    ),
    ("non_corpus_news", ("오늘 뉴스", "실시간 뉴스", "최신 뉴스 기사")),
)

_GENERIC_COMPANY_REFERENCES = (
    "그 회사", "이 회사", "저 회사", "해당 회사", "그 기업", "해당 기업", "그곳",
)

_VAGUE_OPERATION_PATTERNS = (
    "바뀐 거", "변경된 거", "어떻게 됐", "무슨 일", "피해 본 건", "취소한 건",
)
_VAGUE_TASK_PATTERNS = ("재무지표", "재무 수치")


@dataclass(frozen=True)
class ClarificationOption:
    id: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label}


@dataclass(frozen=True)
class Clarification:
    question: str
    options: tuple[ClarificationOption, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "options": [option.to_dict() for option in self.options],
        }


@dataclass(frozen=True)
class QuerySlot:
    name: str
    value: Any = None
    source: QuerySlotSource = QuerySlotSource.DETERMINISTIC
    status: QuerySlotStatus = QuerySlotStatus.MISSING
    locked: bool = False
    candidates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": _json_clean(self.value),
            "source": self.source.value,
            "status": self.status.value,
        }
        if self.candidates:
            payload["candidates"] = list(self.candidates)
        return payload


@dataclass(frozen=True)
class CorpusScope:
    """Read-only company and date bounds loaded from frozen corpus metadata."""

    companies: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    receipt_from: str | None = None
    receipt_to: str | None = None
    fiscal_years: tuple[int, ...] = ()
    event_from: str | None = None
    event_to: str | None = None

    @classmethod
    def from_files(
        cls,
        *,
        universe_path: str | Path,
        manifest_path: str | Path,
    ) -> "CorpusScope":
        aliases: dict[str, tuple[str, str]] = {}
        with Path(universe_path).open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                corp_code = str(row.get("corp_code") or "").strip()
                corp_name = str(row.get("corp_name") or "").strip()
                listed_name = str(row.get("listed_name") or "").strip()
                if not corp_code or not corp_name:
                    continue
                for name in {corp_name, listed_name}:
                    if name:
                        aliases[_company_key(name)] = (corp_name, corp_code)

        dates: list[str] = []
        fiscal_years: set[int] = set()
        with Path(manifest_path).open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                receipt = _iso_date(row.get("rcept_dt"))
                if receipt:
                    dates.append(receipt)
                if row.get("doc_group") == "periodic" and row.get("base_year"):
                    try:
                        fiscal_years.add(int(row["base_year"]))
                    except (TypeError, ValueError):
                        pass
        return cls(
            companies=aliases,
            receipt_from=min(dates, default=None),
            receipt_to=max(dates, default=None),
            fiscal_years=tuple(sorted(fiscal_years)),
        )

    @classmethod
    def repository_default(cls) -> "CorpusScope | None":
        root = Path(__file__).resolve().parents[2]
        universe = root / "data" / "corpus" / "universe.csv"
        manifest = root / "data" / "corpus" / "manifest.jsonl"
        if not universe.is_file() or not manifest.is_file():
            return None
        try:
            return cls.from_files(universe_path=universe, manifest_path=manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            # A missing local metadata bundle must not make the service fail to
            # start. Production still has the authoritative DB resolver.
            return None

    def resolve_company(self, value: str) -> tuple[str, str] | None:
        return self.companies.get(_company_key(value))

    def company_aliases(self) -> dict[str, set[str]]:
        """Aliases in the format QueryUnderstanding already consumes."""

        return {
            alias: {canonical}
            for alias, (canonical, _corp_code) in self.companies.items()
        }


class PostgresEventScopeProvider:
    """Read the P0-B graph's actual date bounds once, without changing it."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self._loaded = False
        self._value: tuple[str | None, str | None] = (None, None)

    def __call__(self) -> tuple[str | None, str | None]:
        if self._loaded:
            return self._value
        try:
            with self.backend.connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT min(opened_at), "
                        "max(coalesce(closed_at, opened_at)) "
                        "FROM corporate_events"
                    )
                    row = cursor.fetchone()
            if row:
                self._value = (_date_text(row[0]), _date_text(row[1]))
        except Exception:  # noqa: BLE001 - missing graph degrades to static scope
            self._value = (None, None)
        self._loaded = True
        return self._value


@dataclass(frozen=True)
class QueryValidationResult:
    state: QueryState
    plan: QueryPlan
    slots: Mapping[str, QuerySlot]
    required_slots: tuple[str, ...]
    issues: tuple[str, ...] = ()
    clarification: Clarification | None = None
    fallback_used: bool = False
    fallback_status: str = "not_needed"
    hcx_elapsed_ms: float | None = None
    hcx_diagnostic: Mapping[str, Any] = field(default_factory=dict)

    @property
    def retrieval_allowed(self) -> bool:
        return self.state is QueryState.RESOLVED

    @property
    def missing_slots(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.required_slots
            if self.slots[name].status is QuerySlotStatus.MISSING
        )

    @property
    def ambiguous_slots(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.required_slots
            if self.slots[name].status
            in (QuerySlotStatus.AMBIGUOUS, QuerySlotStatus.INVALID)
        )

    @property
    def fallback_recommended(self) -> bool:
        semantic = {"task_type", "event_family", "operation", "set_intent", "requested_state"}
        return self.state in {QueryState.AMBIGUOUS, QueryState.INCOMPLETE} and bool(
            semantic.intersection({*self.missing_slots, *self.ambiguous_slots})
        )

    def with_fallback_failure(
        self,
        status: str,
        *,
        elapsed_ms: float | None = None,
        used: bool = True,
        diagnostic: Mapping[str, Any] | None = None,
    ) -> "QueryValidationResult":
        return replace(
            self,
            fallback_used=used,
            fallback_status=status,
            hcx_elapsed_ms=elapsed_ms,
            hcx_diagnostic=dict(diagnostic or {}),
        )

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "status": self.state.value,
            "resolved_slots": {
                name: slot.to_dict()
                for name, slot in self.slots.items()
                if slot.status is QuerySlotStatus.RESOLVED
            },
            "missing_slots": list(self.missing_slots),
            "ambiguous_slots": list(self.ambiguous_slots),
            "hcx_fallback_used": self.fallback_used,
            "hcx_fallback_status": self.fallback_status,
            "hcx_elapsed_ms": self.hcx_elapsed_ms,
            "hcx_diagnostic": (
                dict(self.hcx_diagnostic) if self.hcx_diagnostic else None
            ),
            "clarification_required": self.clarification is not None,
            "clarification": (
                self.clarification.to_dict() if self.clarification else None
            ),
        }

    def to_validation_dict(self) -> dict[str, Any]:
        """Minimal public firewall result; contains no hidden reasoning."""

        summary: dict[str, Any] = {
            "status": (
                "valid" if self.state is QueryState.RESOLVED else self.state.value
            ),
            "retrieval_allowed": self.retrieval_allowed,
        }
        if self.issues:
            summary["reason"] = self.issues[0]
        return summary


class QueryValidator:
    """Validate required slots and corpus scope without invoking an LLM."""

    def __init__(
        self,
        *,
        corpus_scope: CorpusScope | None = None,
        multi_document_planner: Any = None,
        event_scope_provider: Callable[
            [], tuple[str | None, str | None]
        ] | None = None,
    ) -> None:
        self.corpus_scope = corpus_scope
        self.multi_document_planner = multi_document_planner
        self.event_scope_provider = event_scope_provider

    def validate(
        self,
        plan: QueryPlan,
        *,
        semantic: Mapping[str, Any] | Any | None = None,
        fallback_used: bool = False,
        fallback_status: str = "not_needed",
        hcx_elapsed_ms: float | None = None,
        hcx_diagnostic: Mapping[str, Any] | None = None,
    ) -> QueryValidationResult:
        semantic_values = _semantic_mapping(semantic)
        question = str(plan.raw_query or plan.query)

        unsupported = _unsupported_reason(question)
        if unsupported:
            return self._terminal(
                QueryState.UNSUPPORTED,
                plan,
                issue=unsupported,
                fallback_used=fallback_used,
                fallback_status=fallback_status,
                hcx_elapsed_ms=hcx_elapsed_ms,
                hcx_diagnostic=hcx_diagnostic,
            )

        company_slot, company_issue, normalized_plan = self._company_slot(plan)
        multi_plan = self._multi_plan(question, normalized_plan)
        out_of_scope = company_issue or self._time_scope_issue(
            normalized_plan,
            multi_plan=multi_plan,
        )
        if out_of_scope:
            return self._terminal(
                QueryState.OUT_OF_SCOPE,
                normalized_plan,
                issue=out_of_scope,
                slots={"company": company_slot},
                fallback_used=fallback_used,
                fallback_status=fallback_status,
                hcx_elapsed_ms=hcx_elapsed_ms,
                hcx_diagnostic=hcx_diagnostic,
            )

        semantic_gap = _semantic_plan_gap(normalized_plan, question, multi_plan)
        task_value, task_locked = _deterministic_task(normalized_plan, question, multi_plan)
        vague_semantics = _task_is_ambiguous(
            question, normalized_plan
        ) or _operation_is_ambiguous(
            question,
            event_value=normalized_plan.event_type,
            task=str(normalized_plan.task_type or ""),
        ) or semantic_gap is not None
        task_slot, task_conflict = _merge_slot(
            "task_type",
            task_value,
            semantic_values.get("task_type"),
            allowed=ALLOWED_TASK_TYPES,
            locked=task_locked,
            deterministic_ambiguous=vague_semantics and task_value is None,
        )
        issues = list(task_conflict)
        task = str(task_slot.value or "")
        if task and task != normalized_plan.task_type:
            scoped_task = (
                "corporate_event"
                if task in {"enumeration", "enumeration_plus_event"}
                else task
            )
            semantic_scope_issue = self._time_scope_issue(
                replace(normalized_plan, task_type=scoped_task),
                multi_plan=multi_plan,
            )
            if semantic_scope_issue:
                return self._terminal(
                    QueryState.OUT_OF_SCOPE,
                    normalized_plan,
                    issue=semantic_scope_issue,
                    slots={"company": company_slot, "task_type": task_slot},
                    fallback_used=fallback_used,
                    fallback_status=fallback_status,
                    hcx_elapsed_ms=hcx_elapsed_ms,
                    hcx_diagnostic=hcx_diagnostic,
                )

        event_value = _deterministic_event(normalized_plan, multi_plan)
        event_slot, conflicts = _merge_slot(
            "event_family",
            event_value,
            semantic_values.get("event_family"),
            allowed=ALLOWED_EVENT_FAMILIES,
            locked=event_value is not None,
            deterministic_ambiguous=semantic_gap is not None,
        )
        issues.extend(conflicts)

        operation_value = _deterministic_operation(normalized_plan, question, task)
        operation_ambiguous = vague_semantics or _operation_is_ambiguous(
            question, event_value=event_value, task=task
        )
        operation_slot, conflicts = _merge_slot(
            "operation",
            None if operation_ambiguous else operation_value,
            semantic_values.get("operation"),
            allowed=ALLOWED_OPERATIONS,
            locked=bool(operation_value and not operation_ambiguous),
            deterministic_ambiguous=operation_ambiguous,
        )
        issues.extend(conflicts)

        period_value = _period_value(normalized_plan)
        metric_value = normalized_plan.metric
        correction_intent = normalized_plan.evidence.get("correction_intent")
        set_intent = _set_intent(question, multi_plan)
        requested_state = _requested_state(question, multi_plan)

        slots: dict[str, QuerySlot] = {
            "company": company_slot,
            "task_type": task_slot,
            "period": _deterministic_slot("period", period_value),
            "metric": _deterministic_slot("metric", metric_value),
            "event_family": event_slot,
            "operation": operation_slot,
            "correction_intent": _deterministic_slot(
                "correction_intent", correction_intent
            ),
            "set_intent": _semantic_fill_slot(
                "set_intent", set_intent, semantic_values.get("set_intent")
            ),
            "requested_state": _semantic_fill_slot(
                "requested_state",
                requested_state,
                semantic_values.get("requested_state"),
            ),
        }

        required = _required_slots(task)
        if semantic_gap == "explicit_year_contract_without_event_family":
            required = tuple(dict.fromkeys((*required, "event_family")))
            if normalized_plan.date_basis.value == "unspecified":
                slots["date_basis"] = QuerySlot(
                    "date_basis",
                    None,
                    QuerySlotSource.DETERMINISTIC,
                    QuerySlotStatus.AMBIGUOUS,
                    False,
                    ("contract_date", "receipt_date"),
                )
                required = tuple(dict.fromkeys((*required, "date_basis")))
        if task == "correction_lookup" and _document_identity(question):
            required = tuple(name for name in required if name != "company")
            slots["document_identity"] = _deterministic_slot(
                "document_identity", _document_identity(question)
            )
            required = ("document_identity", *required)

        # Model-declared ambiguity is advisory only. It can prevent execution,
        # never force a route or erase a deterministic lock.
        if semantic_values.get("ambiguity") is True:
            target = next(
                (name for name in ("operation", "event_family", "task_type") if name in required),
                "operation",
            )
            current = slots[target]
            if not current.locked:
                slots[target] = replace(
                    current,
                    status=QuerySlotStatus.AMBIGUOUS,
                    candidates=_interpretation_ids(
                        semantic_values.get("possible_interpretations")
                    ),
                )

        missing = [name for name in required if slots[name].status is QuerySlotStatus.MISSING]
        ambiguous = [
            name
            for name in required
            if slots[name].status in {QuerySlotStatus.AMBIGUOUS, QuerySlotStatus.INVALID}
        ]
        state = (
            QueryState.AMBIGUOUS
            if ambiguous or issues
            else QueryState.INCOMPLETE
            if missing
            else QueryState.RESOLVED
        )

        final_plan = normalized_plan
        if state is QueryState.RESOLVED:
            final_plan = _apply_validated_semantics(normalized_plan, slots)
        clarification = (
            _clarification(question, slots, required, issues)
            if state in {QueryState.AMBIGUOUS, QueryState.INCOMPLETE}
            else None
        )
        return QueryValidationResult(
            state=state,
            plan=final_plan,
            slots=slots,
            required_slots=required,
            issues=tuple(dict.fromkeys(issues)),
            clarification=clarification,
            fallback_used=fallback_used,
            fallback_status=fallback_status,
            hcx_elapsed_ms=hcx_elapsed_ms,
            hcx_diagnostic=dict(hcx_diagnostic or {}),
        )

    def _terminal(
        self,
        state: QueryState,
        plan: QueryPlan,
        *,
        issue: str,
        slots: Mapping[str, QuerySlot] | None = None,
        fallback_used: bool,
        fallback_status: str,
        hcx_elapsed_ms: float | None,
        hcx_diagnostic: Mapping[str, Any] | None,
    ) -> QueryValidationResult:
        return QueryValidationResult(
            state=state,
            plan=plan,
            slots=dict(slots or {}),
            required_slots=tuple((slots or {}).keys()),
            issues=(issue,),
            fallback_used=fallback_used,
            fallback_status=fallback_status,
            hcx_elapsed_ms=hcx_elapsed_ms,
            hcx_diagnostic=dict(hcx_diagnostic or {}),
        )

    def _company_slot(
        self, plan: QueryPlan
    ) -> tuple[QuerySlot, str | None, QueryPlan]:
        question = str(plan.raw_query or "")
        if not plan.companies:
            return _deterministic_slot("company", None), None, plan
        if len(plan.companies) > 1:
            if self.corpus_scope is not None:
                resolved = [
                    self.corpus_scope.resolve_company(company)
                    for company in plan.companies
                ]
                if any(value is None for value in resolved):
                    return (
                        _deterministic_slot("company", list(plan.companies)),
                        "company_out_of_corpus",
                        plan,
                    )
                known = [value for value in resolved if value is not None]
                plan = replace(
                    plan,
                    companies=tuple(value[0] for value in known),
                    corp_codes=tuple(value[1] for value in known),
                )
            if (
                isinstance(plan.comparison, Mapping)
                and plan.comparison.get("type") == "company_comparison"
            ):
                return _deterministic_slot("company", list(plan.companies)), None, plan
            return (
                QuerySlot(
                    "company",
                    list(plan.companies),
                    status=QuerySlotStatus.AMBIGUOUS,
                    locked=True,
                    candidates=plan.companies,
                ),
                None,
                plan,
            )

        company = plan.companies[0]
        if plan.corp_codes:
            if self.corpus_scope is not None:
                resolved = self.corpus_scope.resolve_company(company)
                if resolved is None:
                    return (
                        _deterministic_slot("company", company),
                        "company_out_of_corpus",
                        plan,
                    )
                canonical, expected_code = resolved
                if len(plan.corp_codes) != 1 or plan.corp_codes[0] != expected_code:
                    return (
                        QuerySlot(
                            "company",
                            company,
                            QuerySlotSource.DETERMINISTIC,
                            QuerySlotStatus.INVALID,
                            True,
                            (expected_code, *plan.corp_codes),
                        ),
                        None,
                        plan,
                    )
                if canonical != company:
                    plan = replace(plan, companies=(canonical,))
                    company = canonical
            return _deterministic_slot("company", company), None, plan
        if self.corpus_scope is not None:
            resolved = self.corpus_scope.resolve_company(company)
            if resolved:
                canonical, corp_code = resolved
                normalized = replace(
                    plan,
                    companies=(canonical,),
                    corp_codes=(corp_code,),
                )
                return _deterministic_slot("company", canonical), None, normalized

        attempted = bool(plan.evidence.get("company_resolution_attempted"))
        if attempted or self.corpus_scope is not None:
            if any(reference in question for reference in _GENERIC_COMPANY_REFERENCES):
                return _deterministic_slot("company", None), None, replace(
                    plan, companies=(), corp_codes=()
                )
            return _deterministic_slot("company", company), "company_out_of_corpus", plan
        return _deterministic_slot("company", company), None, plan

    def _time_scope_issue(
        self,
        plan: QueryPlan,
        *,
        multi_plan: Any = None,
    ) -> str | None:
        scope = self.corpus_scope
        if scope is None:
            return None
        period = plan.period
        years = set(plan.years)
        if period.year is not None:
            years.add(period.year)
        if (
            period.is_fiscal
            or plan.task_type in {
                "financial_metric",
                "periodic_fact",
                "business_product",
                "listing_history",
                "merger_history",
            }
        ) and years and scope.fiscal_years:
            if any(year not in set(scope.fiscal_years) for year in years):
                return "period_out_of_corpus"
        if period.period_type == "receipt_date" and scope.receipt_from and scope.receipt_to:
            if period.to_date and period.to_date < scope.receipt_from:
                return "period_out_of_corpus"
            if period.from_date and period.from_date > scope.receipt_to:
                return "period_out_of_corpus"
        event_from, event_to = scope.event_from, scope.event_to
        event_scoped_query = plan.task_type == "corporate_event" or bool(
            plan.event_type
        ) or (
            plan.task_type == "disclosure_lookup"
            and any(
                marker in str(plan.raw_query or "")
                for marker in ("계약", "해지", "시설투자", "유상증자", "전환사채")
            )
        )
        explicit_event_date_semantics = plan.date_basis in {
            "contract_date",
            "period_start",
        }
        event_bounds_relevant = event_scoped_query and (
            period.period_type != "reference_year"
            or explicit_event_date_semantics
        )
        if (
            event_bounds_relevant
            and self.event_scope_provider is not None
            and not (event_from and event_to)
        ):
            event_from, event_to = self.event_scope_provider()
        event_bounds_available = bool(event_from and event_to and event_bounds_relevant)
        if event_from and event_to and event_bounds_relevant:
            if period.to_date and period.to_date < event_from:
                return "period_out_of_corpus"
            if period.from_date and period.from_date > event_to:
                return "period_out_of_corpus"
            if period.year and not period.from_date:
                if period.year < int(event_from[:4]) or period.year > int(event_to[:4]):
                    return "period_out_of_corpus"
        if (
            period.period_type == "reference_year"
            and years
            and not event_bounds_available
            and not (
                multi_plan is not None
                and plan.date_basis in {"contract_date", "period_start"}
            )
            and scope.receipt_from
            and scope.receipt_to
        ):
            # A reference year is not automatically a receipt year: P0-C may
            # explicitly bind it to contract_date/period_start.  A generic
            # disclosure lookup has no such event-date semantics, however, so
            # an event graph whose history predates the disclosure corpus must
            # not widen retrieval.  Use the frozen manifest's broad receipt-
            # year coverage for that unspecified case.
            first_year = int(scope.receipt_from[:4])
            last_year = int(scope.receipt_to[:4])
            if any(year < first_year or year > last_year for year in years):
                return "period_out_of_corpus"
        return None

    def _multi_plan(self, question: str, plan: QueryPlan) -> Any:
        if self.multi_document_planner is None:
            return None
        try:
            compiled = self.multi_document_planner.plan(question, plan)
        except (TypeError, ValueError):
            return None
        return compiled if bool(getattr(compiled, "applied", False)) else None


def _deterministic_task(
    plan: QueryPlan, question: str, multi_plan: Any
) -> tuple[str | None, bool]:
    if multi_plan is not None:
        return str(multi_plan.plan_type), True
    task = str(plan.task_type or "") or None
    if task == "disclosure_lookup" and (
        _task_is_ambiguous(question, plan)
        or _operation_is_ambiguous(
            question, event_value=None, task=task
        )
        or _semantic_plan_gap(plan, question, multi_plan) is not None
    ):
        return None, False
    if plan.evidence.get("correction_intent"):
        return "correction_lookup", True
    return task, task is not None


def _deterministic_event(plan: QueryPlan, multi_plan: Any) -> str | None:
    if plan.event_type:
        return str(plan.event_type)
    if multi_plan is None:
        return None
    for slot in getattr(multi_plan, "slots", ()):
        family = getattr(slot, "event_family", None)
        if family == "treasury_trust_contract":
            return "treasury_share_trust_contract"
        if family in ALLOWED_EVENT_FAMILIES:
            return str(family)
    return None


def _deterministic_operation(plan: QueryPlan, question: str, task: str) -> str | None:
    operation = plan.evidence.get("operation")
    if operation in ALLOWED_OPERATIONS:
        if task == "enumeration_plus_event":
            return "lifecycle_status"
        if task == "enumeration":
            return "enumerate"
        return str(operation)
    if task == "enumeration_plus_event":
        return "lifecycle_status"
    if task == "enumeration":
        return "enumerate"
    if task == "correction_lookup":
        return "correction_lookup"
    return None


def _required_slots(task: str) -> tuple[str, ...]:
    return {
        "financial_metric": ("company", "task_type", "period", "metric", "operation"),
        "corporate_event": ("company", "task_type", "event_family", "operation"),
        "holding_change": ("company", "task_type", "metric", "operation"),
        "correction_lookup": ("company", "task_type", "correction_intent", "operation"),
        "enumeration": (
            "company", "task_type", "event_family", "period", "set_intent", "operation",
        ),
        "enumeration_plus_event": (
            "company", "task_type", "event_family", "period", "set_intent",
            "requested_state", "operation",
        ),
        "disclosure_lookup": ("company", "task_type", "operation"),
    }.get(task, ("company", "task_type", "operation"))


def _merge_slot(
    name: str,
    deterministic: Any,
    semantic: Any,
    *,
    allowed: Sequence[str] | frozenset[str],
    locked: bool,
    deterministic_ambiguous: bool = False,
) -> tuple[QuerySlot, tuple[str, ...]]:
    if deterministic is not None:
        if str(deterministic) not in allowed:
            return (
                QuerySlot(
                    name,
                    deterministic,
                    QuerySlotSource.DETERMINISTIC,
                    QuerySlotStatus.INVALID,
                    True,
                ),
                (f"invalid_deterministic_value:{name}",),
            )
        if semantic not in (None, "") and semantic != deterministic and locked:
            return (
                QuerySlot(
                    name,
                    deterministic,
                    QuerySlotSource.DETERMINISTIC,
                    QuerySlotStatus.AMBIGUOUS,
                    True,
                    (str(deterministic), str(semantic)),
                ),
                (f"locked_slot_conflict:{name}",),
            )
        return _deterministic_slot(name, deterministic), ()
    if semantic not in (None, ""):
        if str(semantic) not in allowed:
            return (
                QuerySlot(
                    name,
                    semantic,
                    QuerySlotSource.HCX_FALLBACK,
                    QuerySlotStatus.INVALID,
                ),
                (f"invalid_semantic_value:{name}",),
            )
        return (
            QuerySlot(
                name,
                semantic,
                QuerySlotSource.HCX_FALLBACK,
                QuerySlotStatus.RESOLVED,
            ),
            (),
        )
    return (
        QuerySlot(
            name,
            None,
            QuerySlotSource.DETERMINISTIC,
            QuerySlotStatus.AMBIGUOUS if deterministic_ambiguous else QuerySlotStatus.MISSING,
        ),
        (),
    )


def _deterministic_slot(name: str, value: Any) -> QuerySlot:
    return QuerySlot(
        name=name,
        value=value,
        source=QuerySlotSource.DETERMINISTIC,
        status=(
            QuerySlotStatus.RESOLVED
            if value not in (None, "", (), [])
            else QuerySlotStatus.MISSING
        ),
        locked=value not in (None, "", (), []),
    )


def _semantic_fill_slot(name: str, deterministic: Any, semantic: Any) -> QuerySlot:
    if deterministic not in (None, "", False):
        return _deterministic_slot(name, deterministic)
    if semantic not in (None, "", False):
        return QuerySlot(
            name,
            semantic,
            QuerySlotSource.HCX_FALLBACK,
            QuerySlotStatus.RESOLVED,
        )
    return _deterministic_slot(name, None)


def _period_value(plan: QueryPlan) -> Any:
    period = plan.period
    if len(plan.years) > 1:
        return list(plan.years)
    if period.period_type in {"latest_event", "latest_holding"}:
        return period.to_dict()
    if period.year is not None or period.from_date or period.to_date:
        return period.to_dict()
    return None


def _set_intent(question: str, multi_plan: Any) -> bool | None:
    if multi_plan is not None:
        return True
    return True if any(marker in question for marker in _SET_MARKERS) else None


def _requested_state(question: str, multi_plan: Any) -> str | None:
    if (
        multi_plan is not None
        and str(getattr(multi_plan, "plan_type", ""))
        == "enumeration_plus_event"
    ):
        return "termination_state"
    if any(marker in question for marker in _LIFECYCLE_MARKERS) and any(
        marker in question for marker in _EXISTENTIAL_MARKERS
    ):
        return "termination_state"
    return None


def _operation_is_ambiguous(
    question: str, *, event_value: str | None, task: str
) -> bool:
    compact = re.sub(r"\s+", "", question)
    if "취소" in compact and event_value is None:
        return True
    if task == "disclosure_lookup" and any(
        pattern.replace(" ", "") in compact
        for pattern in _VAGUE_OPERATION_PATTERNS
    ):
        return True
    return False


def _task_is_ambiguous(question: str, plan: QueryPlan) -> bool:
    if plan.task_type != "disclosure_lookup":
        return False
    if any(marker in question for marker in _VAGUE_TASK_PATTERNS):
        return True
    compact = re.sub(r"\s+", "", question)
    return bool(
        re.search(
            r"(?<!생산)실적(?:은|이|을|를|알려|보여|어때|어떻게|\?|$)",
            compact,
        )
    )


def _semantic_plan_gap(
    plan: QueryPlan, question: str, multi_plan: Any
) -> str | None:
    """Find a narrow strong-cue gap without changing frozen generic lookup.

    Gold60 intentionally contains periodless general-evidence questions that
    mention contracts.  The unsafe serving case is narrower: an explicit
    reference year plus a contract cue silently lost both the event family and
    the real-world date basis.  Such a plan cannot safely authorize retrieval.
    """

    if multi_plan is not None or plan.task_type != "disclosure_lookup":
        return None
    if plan.period.period_type != "reference_year" or plan.period.year is None:
        return None
    if plan.date_basis.value != "unspecified":
        # Explicit contract/receipt semantics are frozen P0-C inputs, including
        # its intentional single-field fallback cases.
        return None
    if plan.event_type or plan.metric or plan.evidence.get("correction_intent"):
        return None
    if plan.evidence.get("periodic_intent") or "holding" in plan.disclosure_route:
        return None
    compact = re.sub(r"\s+", "", question)
    if "계약" not in compact:
        return None
    return "explicit_year_contract_without_event_family"


def _unsupported_reason(question: str) -> str | None:
    compact = re.sub(r"\s+", "", question)
    for reason, patterns in _UNSUPPORTED_PATTERNS:
        if any(re.sub(r"\s+", "", pattern) in compact for pattern in patterns):
            return reason
    return None


def _clarification(
    question: str,
    slots: Mapping[str, QuerySlot],
    required: Sequence[str],
    issues: Sequence[str],
) -> Clarification:
    unresolved = [
        name
        for name in required
        if slots[name].status in {
            QuerySlotStatus.MISSING,
            QuerySlotStatus.AMBIGUOUS,
            QuerySlotStatus.INVALID,
        }
    ]
    if "company" in unresolved:
        return Clarification("어느 회사에 대한 공시를 확인할까요?")
    if "period" in unresolved:
        return Clarification("어느 연도 또는 기간의 공시를 확인할까요?")
    if "metric" in unresolved:
        return Clarification("확인하려는 재무 지표를 구체적으로 알려주세요.")
    if "date_basis" in unresolved:
        year = re.search(r"(?<!\d)((?:19|20)\d{2})\s*년?", question)
        label = f"{year.group(1)}년" if year else "말씀하신 기간"
        return Clarification(
            f"{label}이 계약 체결일 기준인가요, 공시 접수일 기준인가요?",
            (
                ClarificationOption("contract_date", "계약 체결일 기준"),
                ClarificationOption("receipt_date", "공시 접수일 기준"),
            ),
        )
    if "operation" in unresolved and "취소" in question:
        return Clarification(
            "말씀하신 '취소'가 체결된 계약의 이후 해지를 뜻하나요, 아니면 투자 계획의 철회를 뜻하나요?",
            (
                ClarificationOption("contract_termination", "계약 해지"),
                ClarificationOption("investment_cancellation", "투자 계획 철회"),
            ),
        )
    if "event_family" in unresolved:
        return Clarification(
            "어떤 공시 사건을 확인할까요? 예: 공급계약, 시설투자, 유상증자"
        )
    if any(issue.startswith("locked_slot_conflict") for issue in issues):
        return Clarification("질문의 의미가 서로 충돌합니다. 확인하려는 대상을 한 가지로 다시 적어주세요.")
    return Clarification("확인하려는 공시 내용이나 동작을 조금 더 구체적으로 알려주세요.")


def _apply_validated_semantics(plan: QueryPlan, slots: Mapping[str, QuerySlot]) -> QueryPlan:
    task = str(slots["task_type"].value or plan.task_type or "")
    event = slots["event_family"].value
    operation = slots["operation"].value
    plan_task = plan.task_type
    if slots["task_type"].source is QuerySlotSource.HCX_FALLBACK:
        plan_task = (
            "corporate_event"
            if task in {"enumeration", "enumeration_plus_event"}
            else "disclosure_lookup"
            if task == "correction_lookup"
            else task
        )
    plan_event = plan.event_type
    if event and slots["event_family"].source is QuerySlotSource.HCX_FALLBACK:
        plan_event = str(event)
    evidence = dict(plan.evidence)
    for name, value in (
        ("operation", operation),
        ("set_intent", slots["set_intent"].value),
        ("requested_state", slots["requested_state"].value),
    ):
        if value is not None:
            evidence[name] = value
    route = plan.disclosure_route
    if event and slots["event_family"].source is QuerySlotSource.HCX_FALLBACK:
        route = _EVENT_ROUTES.get(str(event), route)
    if (
        plan_task == plan.task_type
        and plan_event == plan.event_type
        and route == plan.disclosure_route
        and evidence == dict(plan.evidence)
    ):
        return plan
    return replace(
        plan,
        task_type=plan_task,
        event_type=plan_event,
        disclosure_route=route,
        evidence=evidence,
    )


def _semantic_mapping(value: Mapping[str, Any] | Any | None) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("semantic fallback output must be a mapping or expose to_dict()")


def _interpretation_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    output = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("id") or item.get("operation") or item.get("event_family")
        text = str(item or "").strip()
        if text:
            output.append(text)
    return tuple(dict.fromkeys(output))


def _document_identity(question: str) -> str | None:
    match = re.search(r"(?<!\d)(\d{14})(?!\d)", question)
    return match.group(1) if match else None


def _company_key(value: Any) -> str:
    return re.sub(r"[\s㈜()]+", "", str(value or "")).casefold()


def _iso_date(value: Any) -> str | None:
    digits = re.sub(r"\D", "", str(value or ""))
    if len(digits) < 8:
        return None
    candidate = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    try:
        date.fromisoformat(candidate)
    except ValueError:
        return None
    return candidate


def _date_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())[:10]
    text = str(value)
    return text[:10] if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", text) else None


def _json_clean(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


__all__ = [
    "ALLOWED_EVENT_FAMILIES",
    "ALLOWED_OPERATIONS",
    "ALLOWED_TASK_TYPES",
    "Clarification",
    "ClarificationOption",
    "CorpusScope",
    "PostgresEventScopeProvider",
    "QuerySlot",
    "QuerySlotSource",
    "QuerySlotStatus",
    "QueryState",
    "QueryValidationResult",
    "QueryValidator",
]
