"""Run one comparison as N independent single-company questions.

Each leg is the question the pipeline already answers well: one company, one
subject, one period.  Retrieval, routing, resolvers, and citation composition
are reused unchanged -- this module only scopes the plan and collects what came
back.  Nothing here ranks, merges, or reinterprets a leg's evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


#: Reasons the whole comparison could not be executed.  A leg that simply found
#: nothing is not one of these: that is a result, and it is reported.
UNAVAILABLE_NOT_APPLIED = "plan_not_applied"
UNAVAILABLE_NO_LEG_SUCCEEDED = "no_leg_completed"


@dataclass(frozen=True)
class ComparisonLeg:
    """What one company's own question produced."""

    company: str
    corp_code: str
    generated: Any = None
    result: Any = None
    execution: Any = None
    failure: str | None = None

    @property
    def completed(self) -> bool:
        return self.failure is None and self.generated is not None

    @property
    def answerable(self) -> bool:
        return bool(self.completed and getattr(self.generated, "answerable", False))

    def to_dict(self) -> dict[str, Any]:
        # Counts and statuses only.  No company identifier reaches the trace,
        # matching what the multi-document planner already exposes.
        return {
            "completed": self.completed,
            "answerable": self.answerable,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class ComparisonExecution:
    """The collected legs, and whether the comparison can be rendered."""

    plan: Any
    legs: tuple[ComparisonLeg, ...] = ()
    unavailable_reason: str | None = None

    @property
    def applied(self) -> bool:
        return self.unavailable_reason is None and bool(self.legs)

    @property
    def answerable_legs(self) -> tuple[ComparisonLeg, ...]:
        return tuple(leg for leg in self.legs if leg.answerable)

    def trace(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "leg_count": len(self.legs),
            "answerable_leg_count": len(self.answerable_legs),
            "unavailable_reason": self.unavailable_reason,
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }


def _scoped_plan(query_plan: Any, company: str, corp_code: str) -> Any:
    """The original plan asked about exactly one company.

    ``comparison`` is cleared so the leg is an ordinary single-company question
    downstream.  ``comparison_frame`` in ``evidence`` is deliberately left as it
    was: it is a firewall signal, and a leg must not become eligible for role
    reinterpretation just because it was scoped.
    """

    return replace(
        query_plan,
        company=company,
        companies=(company,),
        corp_code=corp_code,
        corp_codes=(corp_code,),
        comparison=None,
    )


class ComparisonExecutor:
    """Execute one leg per company through the existing serving components."""

    def __init__(
        self,
        *,
        retrieval_executor: Any,
        orchestrator: Any,
        generator: Any,
    ) -> None:
        self.retrieval_executor = retrieval_executor
        self.orchestrator = orchestrator
        self.generator = generator

    def execute(
        self, question: str, query_plan: Any, comparison_plan: Any
    ) -> ComparisonExecution:
        if comparison_plan is None or not getattr(comparison_plan, "applied", False):
            return ComparisonExecution(
                plan=comparison_plan, unavailable_reason=UNAVAILABLE_NOT_APPLIED
            )

        legs: list[ComparisonLeg] = []
        for company, corp_code in comparison_plan.companies:
            legs.append(self._leg(question, query_plan, company, corp_code))

        if not any(leg.completed for leg in legs):
            return ComparisonExecution(
                plan=comparison_plan,
                legs=tuple(legs),
                unavailable_reason=UNAVAILABLE_NO_LEG_SUCCEEDED,
            )
        return ComparisonExecution(plan=comparison_plan, legs=tuple(legs))

    def _leg(
        self, question: str, query_plan: Any, company: str, corp_code: str
    ) -> ComparisonLeg:
        scoped = _scoped_plan(query_plan, company, corp_code)
        try:
            execution = self.retrieval_executor.execute(scoped)
            result = self.orchestrator.run(question, scoped, execution)
            generated = self.generator.generate(result.answer_draft)
        except Exception as error:  # noqa: BLE001 - one leg must not fail the rest
            return ComparisonLeg(
                company=company,
                corp_code=corp_code,
                failure=type(error).__name__,
            )
        return ComparisonLeg(
            company=company,
            corp_code=corp_code,
            generated=generated,
            result=result,
            execution=execution,
        )
