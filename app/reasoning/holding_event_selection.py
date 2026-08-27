"""P1-A5-A: did the question identify one holding event, or merely narrow them?

A holding question can constrain the events without naming one.  A question
giving a company, a holder and the fields to report says nothing about *which*
change is meant.  Answering it with a single event asserts something the
question never said -- and the corpus shows why that matters: a holder alone
leaves more than one event in 84% of cases, a year in 71%, a direction in 69%.

Only an exact reference date expresses intent to name one event.  Everything
else filters.  So this module reports what the *question* supplied, and nothing
about what retrieval happened to return: how many events were served is a fact
about the search, not about what was asked.

The date itself is never re-parsed here.  P1-A4 already decided what an exact
holding reference date is, and that decision is imported rather than repeated.
"""

from __future__ import annotations

from typing import Any, Mapping

from app.reasoning.holding_date_intent import (
    ROUTED_TASK_TYPE,
    exact_reference_date,
)

#: The question named one event: an exact holding reference date.
EXACT = "exact"

#: The question narrowed the events without naming one -- a holder, a year, a
#: quarter, a range, a receipt date, a direction.  Any of these can leave many.
FILTERED = "filtered"

#: The question named nothing that narrows events at all.
UNSPECIFIED = "unspecified"

#: Not a holding execution; this lane has no opinion.
NOT_APPLICABLE = "not_applicable"

#: Words that make a question about a rise or a fall.  Sourced from the
#: resolver's own direction reading, so the two cannot drift apart.
_DIRECTION_TERMS = ("증가", "감소")

#: Period shapes that constrain time without naming a single day.  A receipt
#: date is included deliberately: one filing can report an entire trading
#: history, so the day it arrived says nothing about which change is meant.
_FILTERING_PERIOD_TYPES = (
    "receipt_date",
    "reference_year",
    "date_range",
    "holding_reference_range",
    "holding_reference_year",
    "fiscal_year",
    "fiscal_quarter",
)


def _period(plan: Any) -> Mapping[str, Any]:
    period = getattr(plan, "period", None)
    if hasattr(period, "to_dict"):
        return dict(period.to_dict())
    return dict(period) if isinstance(period, Mapping) else {}


def _narrows_without_naming(plan: Any) -> bool:
    """Whether the question constrains which events qualify.

    Deliberately blind to the labels ``직전보고``/``현재``/``최근``/``최신``/
    ``최초``/``마지막``: the resolver does not read the period types they
    produce, so treating them as constraints here would claim a narrowing the
    system never performs.  ``최초`` and ``마지막`` are opposites that produce
    the same label, which is the clearest evidence they select nothing.
    """

    if getattr(plan, "reporter", None):
        return True
    period = _period(plan)
    if str(period.get("period_type") or "") in _FILTERING_PERIOD_TYPES:
        return True
    if period.get("year") is not None or period.get("quarter") is not None:
        return True
    if period.get("from") or period.get("to"):
        return True
    compact = str(getattr(plan, "raw_query", "") or "").replace(" ", "")
    return any(term in compact for term in _DIRECTION_TERMS)


def classify_holding_event_selection(
    plan: Any, *, routed_task_type: str | None
) -> str:
    """What the question, on its own, says about which event is wanted.

    Reads the plan and nothing else -- never the served evidence, the resolved
    events, retrieval rank, or how many projections happened to be returned.
    """

    if routed_task_type != ROUTED_TASK_TYPE:
        return NOT_APPLICABLE
    if exact_reference_date(plan):
        return EXACT
    return FILTERED if _narrows_without_naming(plan) else UNSPECIFIED


def is_semantically_unique(selection_mode: str, matching_event_count: int) -> bool:
    """Whether one event may be presented as the one that was asked for.

    Both halves are required.  The mode alone cannot know whether the evidence
    agrees, and the count alone cannot supply intent the question never gave --
    a count of one is routinely produced by which projections retrieval served.
    """

    return selection_mode == EXACT and matching_event_count == 1
