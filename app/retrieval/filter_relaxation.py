"""Recover a candidate set that routing emptied entirely.

P1-B measured every relaxation ladder it could and found nothing to recover:
relaxing company or doc_group degraded recall, precision and latency together.
It deferred implementation and named the one case that would justify reopening —
a real ``STRICT_ZERO``: the candidate set empty *after* filtering, with the
answering document excluded before anything scored it.

That case is now demonstrated. A question asking for a company's funding history
by instrument type ("유상증자, CB, BW, EB") has its event type inferred from the
first term in that list, and routing then drops every filing that is not that one
type -- including the convertible-bond filing the question was about. Sixty-four
documents entered the filter and none came out.

The relaxation is deliberately the narrow one P1-B specified: it triggers only on
strict zero, it takes at most two attempts, and it relaxes only metadata the
system *inferred* from the question's wording. Company, corp_code, doc_group and
date stay hard, because those are what the asker actually said. A question whose
strict candidate set is non-empty is untouched, so nothing that retrieves today
can change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence


#: The only trigger.  Not "few candidates", not "a wrong answer" -- empty.
STRICT_ZERO = "strict_zero"

#: Inferred from question wording, never stated by the asker as a filter.  Order
#: matters: ``event_type`` is the narrowest inference and the one measured to
#: empty the set, so it is surrendered first.
RELAXABLE_ROUTES = ("event_type", "doc_subtype")

#: P1-B's own bound.  Two attempts is one per relaxable key, cumulative.
MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class FilterRelaxation:
    """What was surrendered to recover a candidate set, if anything."""

    applied: bool = False
    trigger: str | None = None
    relaxed: tuple[str, ...] = ()
    attempts: int = 0
    strict_document_count: int = 0
    recovered_document_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "filter_relaxed": self.applied,
            "trigger": self.trigger,
            "relaxed_routes": list(self.relaxed),
            "attempts": self.attempts,
            "strict_document_count": self.strict_document_count,
            "recovered_document_count": self.recovered_document_count,
        }


def _without(route: Any, keys: Sequence[str]) -> Any:
    hard_routes = {
        key: value for key, value in route.hard_routes.items() if key not in keys
    }
    return replace(route, hard_routes=hard_routes)


def relax_when_strict_zero(
    router: Any,
    documents: Sequence[Any],
    strict: Sequence[Any],
    route: Any,
) -> tuple[list[Any], FilterRelaxation]:
    """Re-filter with inferred routes surrendered, only when nothing survived.

    ``documents`` is what metadata filtering returned, ``strict`` what routing
    left of it.  Returns ``strict`` unchanged in every case except the one this
    exists for.
    """

    strict_list = list(strict)
    if strict_list:
        return strict_list, FilterRelaxation(strict_document_count=len(strict_list))
    candidates = list(documents)
    if not candidates:
        # Nothing entered the filter, so nothing was excluded by it.  The corpus
        # simply holds no document for this company and window, and saying so is
        # the correct answer rather than a failure to recover.
        return strict_list, FilterRelaxation()

    surrendered: list[str] = []
    for attempt, key in enumerate(RELAXABLE_ROUTES[:MAX_ATTEMPTS], start=1):
        if key not in route.hard_routes:
            continue
        surrendered.append(key)
        recovered = router.filter_documents(candidates, _without(route, surrendered))
        if recovered:
            return list(recovered), FilterRelaxation(
                applied=True,
                trigger=STRICT_ZERO,
                relaxed=tuple(surrendered),
                attempts=attempt,
                strict_document_count=0,
                recovered_document_count=len(recovered),
            )
    return strict_list, FilterRelaxation(
        trigger=STRICT_ZERO if surrendered else None,
        relaxed=tuple(surrendered),
        attempts=len(surrendered),
    )
