"""Deterministic multi-company comparison.

A comparison is not one question.  It is the same question asked of each named
company, answered independently by the pipeline that already answers single
company questions well, and combined afterwards by a rule that does no
inference of its own.

The layer is opt-in: :class:`ComparisonPlanner` declines every question that
does not name at least two companies the corpus can resolve, and a declined
question takes exactly the path it always did.
"""

from app.reasoning.comparison.plan import (
    MAX_COMPARISON_COMPANIES,
    ComparisonPlan,
    ComparisonPlanner,
)

__all__ = [
    "MAX_COMPARISON_COMPANIES",
    "ComparisonPlan",
    "ComparisonPlanner",
]
