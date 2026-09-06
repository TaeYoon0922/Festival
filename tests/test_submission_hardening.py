"""Three failures found by review, each reachable from an ordinary question.

None of them is a crash. Each produces a plausible-looking response that is
wrong about something the caller cannot see: an answer about a company nobody
asked about, a citation pointing at evidence that was not returned, and a
health check that reports the service down while it is merely busy.
"""

from __future__ import annotations

import inspect
import unittest
from typing import Any

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.api.pipeline import _comparison_count, _unconfirmed_acquirer
from app.reasoning.query_understanding import (
    ACTOR_SOURCE_DIRECTED_HOLDER,
    HOLDING_ACTOR_CANDIDATE_KEY,
)


class _Plan:
    def __init__(self, evidence: dict[str, Any], reporter: str | None = None) -> None:
        self.evidence = evidence
        self.reporter = reporter


def _named(surface: str = "가상투자자") -> dict[str, Any]:
    return {
        HOLDING_ACTOR_CANDIDATE_KEY: {
            "surface": surface,
            "source": ACTOR_SOURCE_DIRECTED_HOLDER,
        }
    }


class UnconfirmedAcquirerTests(unittest.TestCase):
    """A named acquirer the corpus does not know must not be answered around.

    Validation binds the name to one of the issuer's own filers when it can,
    and otherwise leaves the plan with no holder at all -- at which point the
    question is indistinguishable from one that named nobody, and an unscoped
    search answers from whichever holder ranked first.
    """

    def test_a_named_but_unbound_acquirer_is_reported(self) -> None:
        self.assertEqual(_unconfirmed_acquirer(_Plan(_named())), "가상투자자")

    def test_a_bound_acquirer_is_not_reported(self) -> None:
        # Validation proved this issuer files under that name, so the question
        # is scoped and the ordinary path owns it.
        self.assertEqual(
            _unconfirmed_acquirer(_Plan(_named("(주)LG"), reporter="(주)LG")), ""
        )

    def test_a_question_naming_no_acquirer_is_not_reported(self) -> None:
        for evidence in ({}, {HOLDING_ACTOR_CANDIDATE_KEY: None}):
            with self.subTest(evidence=evidence):
                self.assertEqual(_unconfirmed_acquirer(_Plan(evidence)), "")

    def test_a_candidate_from_another_producer_is_not_reported(self) -> None:
        # Only the directed-acquisition parser's own output is read here.
        evidence = {
            HOLDING_ACTOR_CANDIDATE_KEY: {
                "surface": "가상투자자",
                "source": "somewhere_else",
            }
        }
        self.assertEqual(_unconfirmed_acquirer(_Plan(evidence)), "")


class ComparisonSliceTests(unittest.TestCase):
    """Every cited row has to be in the list the caller was handed.

    A comparison retrieves each company on its own plan and interleaves them,
    so the served evidence is a multiple of one company's Top-K while the
    public slice was still one Top-K -- cutting the later companies off after
    the answer had already cited them.
    """

    class _Execution:
        def __init__(self, count: int) -> None:
            self.results = tuple(range(count))

    class _Comparison:
        def __init__(self, applied: bool = True) -> None:
            self.applied = applied

    def test_the_fan_out_beyond_one_top_k_is_added(self) -> None:
        self.assertEqual(
            _comparison_count(self._Comparison(), self._Execution(20), 10), 10
        )

    def test_a_declined_comparison_adds_nothing(self) -> None:
        self.assertEqual(
            _comparison_count(self._Comparison(applied=False), self._Execution(20), 10),
            0,
        )

    def test_no_comparison_adds_nothing(self) -> None:
        self.assertEqual(_comparison_count(None, self._Execution(20), 10), 0)

    def test_a_single_company_result_adds_nothing(self) -> None:
        # Never negative: one company's slice is already the ordinary limit.
        self.assertEqual(
            _comparison_count(self._Comparison(), self._Execution(7), 10), 0
        )


class HealthCheckTests(unittest.TestCase):
    """The health check must answer while answers are in flight.

    FastAPI runs a plain ``def`` endpoint in the threadpool that ``/answer``
    occupies, so a queue of slow answers made ``/healthz`` wait behind them --
    reporting the service down at the moment it was busiest.
    """

    def _healthz(self):
        app = create_app(pipeline_factory=lambda: None)
        route = next(r for r in app.routes if getattr(r, "path", "") == "/healthz")
        return route.endpoint

    def test_the_endpoint_is_a_coroutine(self) -> None:
        self.assertTrue(inspect.iscoroutinefunction(self._healthz()))

    def test_it_still_answers_ok(self) -> None:
        client = TestClient(create_app(pipeline_factory=lambda: None))
        response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
