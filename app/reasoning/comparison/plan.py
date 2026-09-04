"""Decide whether a question is an executable multi-company comparison.

This planner only *recognizes* the shape.  It never widens the comparison
firewall in ``query_understanding``: that signal says two companies must not be
re-read as issuer and reporter, and it keeps saying exactly that.  What this
adds is a separate, explicit permission to run the same single-company question
once per company -- a door beside the firewall, not a hole through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


#: A leg costs one retrieval and one reasoning pass, so the fan-out is bounded.
#: A question naming more companies than this is declined rather than served
#: slowly.
MAX_COMPARISON_COMPANIES = 5

#: Reasons the planner declined.  They are execution summary values, never
#: shown as an answer.
DECLINE_NO_PLAN = "no_query_plan"
DECLINE_TOO_FEW = "fewer_than_two_companies"
DECLINE_TOO_MANY = "more_than_max_companies"
DECLINE_UNRESOLVED = "company_not_resolved_in_corpus"
DECLINE_NO_INTENT = "no_comparison_intent"
DECLINE_NO_SUBJECT = "no_shared_subject"

_COMPANY_COMPARISON = "company_comparison"
_CROSS_COMPANY_FRAMES = ("cross_company",)


@dataclass(frozen=True)
class ComparisonPlan:
    """What the comparison layer will run, or why it declined."""

    applied: bool
    companies: tuple[tuple[str, str], ...] = ()
    subject: str | None = None
    subject_kind: str | None = None
    decline_reason: str | None = None

    @property
    def company_count(self) -> int:
        return len(self.companies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "company_count": self.company_count,
            "subject_kind": self.subject_kind,
            "decline_reason": self.decline_reason,
        }

    @classmethod
    def declined(cls, reason: str) -> "ComparisonPlan":
        return cls(applied=False, decline_reason=reason)


def _comparison_intent(query_plan: Any) -> bool:
    """Whether the question relates the named companies to each other.

    Two independent signals already exist upstream and either one is enough
    here.  ``comparison`` is the executable payload query understanding builds
    for explicit 비교/대비 wording; ``comparison_frame`` is the firewall signal
    that also catches 중/더 큰 constructions.  Reading the frame here does not
    relax it -- role reinterpretation stays blocked either way.
    """

    comparison = getattr(query_plan, "comparison", None)
    if isinstance(comparison, Mapping) and comparison.get("type") == _COMPANY_COMPARISON:
        return True
    evidence = getattr(query_plan, "evidence", None)
    if isinstance(evidence, Mapping):
        if evidence.get("comparison_frame") in _CROSS_COMPANY_FRAMES:
            return True
    return False


def _shared_subject(query_plan: Any) -> tuple[str, str] | None:
    """What is being compared, read from the plan alone.

    Without a common axis the legs would ask different questions and the
    combined answer would compare nothing, so a plan that carries none is
    declined instead of being served as two unrelated lookups.
    """

    metric = getattr(query_plan, "metric", None)
    if metric:
        return str(metric), "metric"
    event_type = getattr(query_plan, "event_type", None)
    if event_type:
        return str(event_type), "event_type"
    # ``period`` is always normalized to an object, so its presence proves
    # nothing.  Only a populated one scopes the legs to the same window.
    period = getattr(query_plan, "period", None)
    if _period_is_populated(period):
        return "period", "period"
    if getattr(query_plan, "years", ()):
        return "period", "period"
    return None


def _period_is_populated(period: Any) -> bool:
    if period is None:
        return False
    if isinstance(period, (int, tuple)):
        return True
    return any(
        getattr(period, name, None) is not None
        for name in ("year", "quarter", "from_date", "to_date", "period_type")
    )


def _paired(
    companies: Sequence[str], corp_codes: Sequence[str]
) -> tuple[tuple[str, str], ...] | None:
    if len(companies) != len(corp_codes):
        return None
    pairs = []
    for name, code in zip(companies, corp_codes):
        if not str(name).strip() or not str(code).strip():
            return None
        pairs.append((str(name), str(code)))
    return tuple(pairs)


class ComparisonPlanner:
    """Recognize an executable multi-company comparison, or decline."""

    def __init__(self, *, max_companies: int = MAX_COMPARISON_COMPANIES) -> None:
        if max_companies < 2:
            raise ValueError("max_companies must be at least 2")
        self.max_companies = max_companies

    def plan(self, question: str, query_plan: Any) -> ComparisonPlan:
        del question  # The plan already carries every signal this reads.
        if query_plan is None:
            return ComparisonPlan.declined(DECLINE_NO_PLAN)

        companies = tuple(getattr(query_plan, "companies", ()) or ())
        corp_codes = tuple(getattr(query_plan, "corp_codes", ()) or ())
        if len(companies) < 2:
            return ComparisonPlan.declined(DECLINE_TOO_FEW)
        if len(companies) > self.max_companies:
            return ComparisonPlan.declined(DECLINE_TOO_MANY)

        # Only companies the corpus resolved are compared.  A category the
        # question named without listing its members ("2차전지 기업들") never
        # reaches here, because expanding it would answer about companies the
        # asker never named.
        pairs = _paired(companies, corp_codes)
        if pairs is None:
            return ComparisonPlan.declined(DECLINE_UNRESOLVED)

        if not _comparison_intent(query_plan):
            return ComparisonPlan.declined(DECLINE_NO_INTENT)

        subject = _shared_subject(query_plan)
        if subject is None:
            return ComparisonPlan.declined(DECLINE_NO_SUBJECT)

        return ComparisonPlan(
            applied=True,
            companies=pairs,
            subject=subject[0],
            subject_kind=subject[1],
        )
