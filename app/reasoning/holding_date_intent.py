"""P1-A4 D1: the exact holding date a routed holding execution was asked about.

A question that names one calendar day -- "2023년 6월 13일 보유 수와 비율" -- is
asking about one event.  The resolver already filters events by an exact
``from``/``to`` period; it simply never receives one for these questions.

Why: ``QueryUnderstanding`` materialises ``holding_reference_date`` only when the
plan's own ``task_type`` is ``holding_change``.  A question whose wording misses
the holding-metric vocabulary is planned as ``disclosure_lookup`` and only
promoted to a holding execution later, by ``TaskRouter``.  By then the plan is
frozen, and its period has fallen back to the year.

This module re-asks the frozen parser the same question with the routed
semantics, and hands the answer forward on an execution-scoped copy of the plan.
The parser is imported, never re-implemented: there is no second date ontology
here, no regex, and no alias list.  The original plan is never mutated, so the
public ``query_understanding`` trace keeps reporting exactly what P0-D decided.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from app.reasoning.query_plan import QueryPeriod
from app.reasoning.query_understanding import _period_from_query

#: The period vocabulary the frozen parser uses for a single holding date.
EXACT_PERIOD_TYPE = "holding_reference_date"

#: P0-D's own name for "this date is when the filing was received", which is a
#: different question from when the holding changed.
RECEIPT_ROLE = "receipt"

#: The routed execution shape this lane applies to.  Anything else is untouched.
ROUTED_TASK_TYPE = "holding_event"


def _period_mapping(plan: Any) -> Mapping[str, Any]:
    period = getattr(plan, "period", None)
    if hasattr(period, "to_dict"):
        return dict(period.to_dict())
    return dict(period) if isinstance(period, Mapping) else {}


def _bounds(period: Mapping[str, Any]) -> tuple[Any, Any]:
    return (
        period.get("from") if period.get("from") is not None else period.get("from_date"),
        period.get("to") if period.get("to") is not None else period.get("to_date"),
    )


def exact_reference_date(plan: Any) -> str | None:
    """The single holding date this execution is scoped to, if there is one.

    Reads the plan as it stands, so it answers the same way for a plan that
    always carried an exact date and for an execution-scoped copy that gained
    one here.  A range, a year, a whole month, or no date at all is not a single
    date and returns ``None``.
    """

    period = _period_mapping(plan)
    if str(period.get("period_type") or "") != EXACT_PERIOD_TYPE:
        return None
    start, end = _bounds(period)
    if not start or not end or start != end:
        return None
    return str(start)


def question_reference_date(question: str) -> str | None:
    """The one holding reference date a question names, as eight digits.

    Reads the question with the routed holding semantics, exactly as
    :func:`derive_exact_period` does, and returns nothing for a range, a year, a
    receipt date or no date at all.  Offered separately because a component that
    only needs to know *which day was asked about* has no plan to gate on -- and
    re-parsing the date with a second ontology is precisely what this module
    exists to prevent.
    """

    try:
        period, _spans, _years, semantics = _period_from_query(
            str(question or ""), task_type="holding_change", routes=("holding",)
        )
    except (TypeError, ValueError):  # a question the frozen parser cannot read
        return None
    if isinstance(semantics, Mapping) and str(semantics.get("role") or "") == RECEIPT_ROLE:
        return None
    if getattr(period, "period_type", None) != EXACT_PERIOD_TYPE:
        return None
    start = getattr(period, "from_date", None)
    if not start or start != getattr(period, "to_date", None):
        return None
    digits = "".join(character for character in str(start) if character.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def _already_exact(plan: Any) -> bool:
    """Whether the frozen plan already pins a single day.

    A native holding_change question does, and its date intent must not be
    rewritten -- re-deriving it could only agree or disagree, and disagreeing
    would mean overruling P0-D.
    """

    start, end = _bounds(_period_mapping(plan))
    return bool(start and end)


def _reads_a_receipt_date(plan: Any) -> bool:
    evidence = getattr(plan, "evidence", None)
    semantics = (dict(evidence) if isinstance(evidence, Mapping) else {}).get(
        "date_semantics"
    )
    if not isinstance(semantics, Mapping):
        return False
    return str(semantics.get("role") or "") == RECEIPT_ROLE


def derive_exact_period(question: str, plan: Any) -> QueryPeriod | None:
    """Re-read the question's date with the semantics the router settled on.

    Returns the parser's own ``QueryPeriod`` when -- and only when -- it yields a
    single holding reference date that the frozen plan did not already carry.
    """

    if _already_exact(plan) or _reads_a_receipt_date(plan):
        return None

    routes = tuple(getattr(plan, "disclosure_route", ()) or ())
    if isinstance(routes, str):
        routes = (routes,)
    try:
        period, _spans, _years, _semantics = _period_from_query(
            str(question or ""), task_type="holding_change", routes=routes
        )
    except (TypeError, ValueError):  # a question the frozen parser cannot read
        return None

    if getattr(period, "period_type", None) != EXACT_PERIOD_TYPE:
        # A range yields holding_reference_range, a bare year yields
        # holding_reference_year, and receipt wording yields receipt_date.
        # None of those name one event.
        return None
    if not period.from_date or period.from_date != period.to_date:
        return None
    return period


def execution_plan(question: str, plan: Any, *, routed_task_type: str | None) -> Any:
    """The plan this holding execution should reason with.

    Returns the original object unless an exact holding date was recovered, in
    which case a copy carrying it is returned.  ``task_type`` is copied through
    untouched: the routed shape decided which resolver runs, and it is not this
    lane's business to restate that on the plan.
    """

    if routed_task_type != ROUTED_TASK_TYPE:
        return plan
    period = derive_exact_period(question, plan)
    if period is None:
        return plan
    try:
        return replace(plan, period=period)
    except TypeError:  # not a dataclass -- leave the caller's plan alone
        return plan
