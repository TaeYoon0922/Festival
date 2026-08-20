"""Deterministic task routing for the read-only reasoning pipeline."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Mapping


_HOLDING_SIGNALS = (
    "국민연금",
    "대량보유",
    "변동일",
    "보유주식수",
    "변동전",
    "변동후",
    "증감",
    "지분율",
)
_PERIODIC_SIGNALS = (
    "사업내용",
    "주요제품",
    "사업개요",
    "매출",
    "생산",
    "연구개발",
    "사업보고서",
    "보고기간",
)
_GENERAL_EVIDENCE_SIGNALS = (
    "공시",
    "보고서",
    "합병",
    "상장",
    "계약",
    "유상증자",
    "교환",
)
_HOLDING_PLAN_TASKS = {"holding_change", "holding_event"}
_PERIODIC_PLAN_TASKS = {
    "periodic_fact",
    "financial_metric",
    "business_product",
    "listing_history",
    "merger_history",
}
_PASSTHROUGH_PLAN_TASKS = {"corporate_event"}
_GENERAL_PLAN_TASKS = {"general_evidence", "exchange_event"}
_RESOLVERS = {
    "holding_event": "holding_event_resolver",
    "periodic_fact": "periodic_fact_resolver",
    "corporate_event": None,
    "general_evidence": None,
    "unknown": None,
}


@dataclass(frozen=True)
class TaskDecision:
    task_type: str
    resolver_type: str | None
    confidence: float
    matched_signals: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "resolver_type": self.resolver_type,
            "confidence": self.confidence,
            "matched_signals": list(self.matched_signals),
            "warnings": list(self.warnings),
        }


class TaskRouter:
    """Route by explicit QueryPlan semantics and fixed lexical signals only."""

    def route(self, question: str, query_plan: Any | None = None) -> TaskDecision:
        return route_task(question, query_plan=query_plan)


def route_task(question: str, query_plan: Any | None = None) -> TaskDecision:
    text = _compact(question)
    plan = _plan_mapping(query_plan)
    plan_task = _text(plan.get("task_type"))
    plan_evidence = plan.get("evidence")
    plan_evidence = (
        dict(plan_evidence) if isinstance(plan_evidence, Mapping) else {}
    )
    periodic_intent = _text(plan_evidence.get("periodic_intent"))
    routes = _routes(plan.get("disclosure_route"))
    metric = _text(plan.get("metric"))

    holding_matches = _matched_query_signals(text, _HOLDING_SIGNALS)
    periodic_matches = _matched_query_signals(text, _PERIODIC_SIGNALS)
    general_matches = _matched_query_signals(text, _GENERAL_EVIDENCE_SIGNALS)
    plan_signals: list[str] = []
    warnings: list[str] = []

    plan_decision: str | None = None
    if plan_task in _HOLDING_PLAN_TASKS:
        plan_decision = "holding_event"
        plan_signals.append(f"plan:task_type={plan_task}")
    elif metric in {"holding_ratio", "holding_shares"}:
        plan_decision = "holding_event"
        plan_signals.append(f"plan:metric={metric}")
    elif periodic_intent or plan_task in _PERIODIC_PLAN_TASKS:
        plan_decision = "periodic_fact"
        value = periodic_intent or plan_task
        plan_signals.append(f"plan:periodic={value}")
    elif plan_task == "disclosure_lookup" and "periodic" in routes:
        plan_decision = "periodic_fact"
        plan_signals.append("plan:route=periodic")
    elif plan_task in _PASSTHROUGH_PLAN_TASKS:
        plan_decision = plan_task
        plan_signals.append(f"plan:task_type={plan_task}")
    elif plan_task in _GENERAL_PLAN_TASKS or (
        plan_task == "disclosure_lookup" and bool(plan)
    ):
        plan_decision = "general_evidence"
        plan_signals.append(f"plan:task_type={plan_task}")

    if holding_matches and periodic_matches:
        warnings.append("mixed_holding_periodic_signals")

    if plan_decision is not None:
        task_type = plan_decision
        confidence = 0.99
        query_matches = (
            holding_matches
            if task_type == "holding_event"
            else periodic_matches
            if task_type == "periodic_fact"
            else general_matches
        )
        if (
            task_type == "holding_event" and periodic_matches and not holding_matches
        ) or (
            task_type == "periodic_fact" and holding_matches and not periodic_matches
        ):
            warnings.append("query_signals_differ_from_query_plan")
            confidence = 0.9
    elif holding_matches:
        task_type = "holding_event"
        query_matches = holding_matches
        confidence = _signal_confidence(len(holding_matches))
    elif periodic_matches:
        task_type = "periodic_fact"
        query_matches = periodic_matches
        confidence = _signal_confidence(len(periodic_matches))
    elif general_matches or plan:
        task_type = "general_evidence"
        query_matches = general_matches
        confidence = 0.6 if general_matches else 0.5
        if not general_matches:
            warnings.append("general_evidence_from_unmapped_query_plan")
    else:
        task_type = "unknown"
        query_matches = ()
        confidence = 0.0
        warnings.append("no_task_signal")

    matched = tuple(
        dict.fromkeys(
            [*plan_signals, *(f"query:{value}" for value in query_matches)]
        )
    )
    return TaskDecision(
        task_type=task_type,
        resolver_type=_RESOLVERS[task_type],
        confidence=confidence,
        matched_signals=matched,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _matched_query_signals(text: str, signals: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(signal for signal in signals if _compact(signal) in text)


def _signal_confidence(count: int) -> float:
    return min(0.95, 0.6 + 0.1 * count)


def _routes(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if str(item))
    return ()


def _plan_mapping(plan: Any | None) -> dict[str, Any]:
    if plan is None:
        return {}
    if hasattr(plan, "to_dict"):
        return copy.deepcopy(dict(plan.to_dict()))
    if isinstance(plan, Mapping):
        return copy.deepcopy(dict(plan))
    raise TypeError("query_plan must be a QueryPlan, mapping, or None")


def _compact(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").casefold())


def _text(value: Any) -> str | None:
    return str(value).strip() if value is not None and str(value).strip() else None
