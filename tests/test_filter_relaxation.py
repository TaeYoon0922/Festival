"""Strict-zero recovery: relax only what was inferred, only when nothing survived.

P1-B's measurement stands -- relaxing company or doc_group degrades everything.
What these assert is that the exception is exactly as narrow as that measurement
allows: an empty candidate set, inferred routes only, two attempts at most, and
no effect whatsoever on a question that already retrieves.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field, replace
from typing import Any, Mapping


@dataclass(frozen=True)
class _Route:
    hard_routes: Mapping[str, Any] = field(default_factory=dict)
    hard_filters: Mapping[str, Any] = field(default_factory=dict)
    correction_policy: str = "any"


@dataclass(frozen=True)
class _Document:
    doc_id: str
    metadata: Mapping[str, Any]


class _Router:
    """Filters on the same keys the production router treats as hard routes."""

    def __init__(self) -> None:
        self.calls = 0

    def filter_documents(self, documents, route):
        self.calls += 1
        selected = []
        for document in documents:
            hard = route.hard_routes
            if "doc_group" in hard and document.metadata.get("doc_group") != hard["doc_group"]:
                continue
            if "doc_subtype" in hard and document.metadata.get("doc_subtype") != hard["doc_subtype"]:
                continue
            if "event_type" in hard and document.metadata.get("event_type") != hard["event_type"]:
                continue
            selected.append(document)
        return selected


from app.retrieval.filter_relaxation import (  # noqa: E402
    MAX_ATTEMPTS,
    RELAXABLE_ROUTES,
    STRICT_ZERO,
    relax_when_strict_zero,
)


def _kakao_documents() -> list[_Document]:
    """카카오's major filings: a convertible bond, never a rights issue."""

    return [
        _Document("cb", {"doc_group": "major", "event_type": "convertible_bond"}),
        _Document("treasury", {"doc_group": "major", "event_type": "treasury_stock"}),
        _Document("holding", {"doc_group": "holding", "event_type": None}),
    ]


class StrictZeroTests(unittest.TestCase):
    def test_a_non_empty_candidate_set_is_untouched(self) -> None:
        router = _Router()
        documents = _kakao_documents()
        strict = [documents[0]]

        recovered, relaxation = relax_when_strict_zero(
            router, documents, strict, _Route(hard_routes={"event_type": "convertible_bond"})
        )

        self.assertEqual(recovered, strict)
        self.assertFalse(relaxation.applied)
        self.assertEqual(router.calls, 0)

    def test_an_empty_input_is_a_genuine_absence_not_a_filter_problem(self) -> None:
        router = _Router()
        recovered, relaxation = relax_when_strict_zero(
            router, [], [], _Route(hard_routes={"event_type": "capital_increase"})
        )

        self.assertEqual(recovered, [])
        self.assertFalse(relaxation.applied)
        self.assertIsNone(relaxation.trigger)
        self.assertEqual(router.calls, 0)

    def test_an_inferred_event_that_emptied_the_set_is_surrendered(self) -> None:
        router = _Router()
        documents = _kakao_documents()
        route = _Route(
            hard_routes={"doc_group": "major", "event_type": "capital_increase"}
        )
        strict = router.filter_documents(documents, route)
        self.assertEqual(strict, [])

        recovered, relaxation = relax_when_strict_zero(
            router, documents, strict, route
        )

        self.assertTrue(relaxation.applied)
        self.assertEqual(relaxation.trigger, STRICT_ZERO)
        self.assertEqual(relaxation.relaxed, ("event_type",))
        self.assertEqual(relaxation.attempts, 1)
        self.assertEqual({document.doc_id for document in recovered}, {"cb", "treasury"})

    def test_doc_group_is_never_surrendered(self) -> None:
        router = _Router()
        documents = _kakao_documents()
        route = _Route(
            hard_routes={"doc_group": "exchange", "event_type": "capital_increase"}
        )

        recovered, relaxation = relax_when_strict_zero(
            router, documents, router.filter_documents(documents, route), route
        )

        # Relaxing every inferred route still cannot admit another doc group.
        self.assertEqual(recovered, [])
        self.assertFalse(relaxation.applied)
        self.assertNotIn("doc_group", relaxation.relaxed)

    def test_at_most_two_attempts_are_made(self) -> None:
        router = _Router()
        documents = [_Document("x", {"doc_group": "exchange"})]
        route = _Route(
            hard_routes={
                "doc_group": "major",
                "event_type": "capital_increase",
                "doc_subtype": "유상증자결정",
            }
        )

        _, relaxation = relax_when_strict_zero(
            router, documents, [], route
        )

        self.assertLessEqual(relaxation.attempts, MAX_ATTEMPTS)
        self.assertLessEqual(router.calls, MAX_ATTEMPTS)

    def test_only_inferred_routes_are_relaxable(self) -> None:
        self.assertEqual(RELAXABLE_ROUTES, ("event_type", "doc_subtype"))
        for hard in ("company", "corp_code", "doc_group", "rcept_dt"):
            self.assertNotIn(hard, RELAXABLE_ROUTES)

    def test_the_trace_reports_what_was_given_up(self) -> None:
        router = _Router()
        documents = _kakao_documents()
        route = _Route(
            hard_routes={"doc_group": "major", "event_type": "capital_increase"}
        )

        _, relaxation = relax_when_strict_zero(
            router, documents, [], route
        )

        self.assertEqual(
            relaxation.to_dict(),
            {
                "filter_relaxed": True,
                "trigger": STRICT_ZERO,
                "relaxed_routes": ["event_type"],
                "attempts": 1,
                "strict_document_count": 0,
                "recovered_document_count": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
