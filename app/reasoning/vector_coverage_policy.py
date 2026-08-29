"""One definition of what "the vector lane had what it needed".

Serving and evaluation answer that question differently -- production degrades
and keeps answering, evaluation refuses to publish -- but they must not disagree
about the facts, so both read the coverage object through this module.

The distinction that motivates it: lexical retrieval ranks every candidate while
vector search can only return the embedded ones, so under partial coverage an
embedded chunk competes in two lanes and an unembedded chunk in one.  A measured
53.6% coverage still reported ``vector_status == "ok"`` while 91.2% of served
chunks came from the embedded subset.  Partial coverage is therefore not a
milder form of zero coverage; it is the case that looks healthy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Deliberate diagnostics rather than claims about semantic retrieval.  A hash
#: run is not a degraded BGE run and must never be reported as one.
DIAGNOSTIC_PROVIDERS = frozenset({"hash"})

HEALTHY = "healthy"
ZERO_COVERAGE = "zero_coverage"
PARTIAL_COVERAGE = "partial_coverage"
EMPTY_VECTOR_RESULT = "empty_vector_result"
VECTOR_UNAVAILABLE = "vector_unavailable"
COVERAGE_UNKNOWN = "coverage_unknown"

#: Degradation -> the ``think_trace`` warning token.  Repository warnings are
#: snake_case tokens with an optional ``:key=value`` payload, not prose.
_WARNING_TOKENS = {
    ZERO_COVERAGE: "vector_coverage_absent",
    PARTIAL_COVERAGE: "vector_coverage_partial",
    EMPTY_VECTOR_RESULT: "vector_results_empty",
    VECTOR_UNAVAILABLE: "vector_unavailable",
    COVERAGE_UNKNOWN: "vector_coverage_unknown",
}


class VectorCoverageError(RuntimeError):
    """A run claiming real vector retrieval did not have the vectors."""


def normalized_provider(provider: Any) -> str:
    return str(provider or "").strip().casefold().replace("-", "_")


def claims_real_vectors(provider: Any) -> bool:
    """Whether a run under this provider claims actual semantic retrieval."""

    name = normalized_provider(provider)
    return bool(name) and name not in DIAGNOSTIC_PROVIDERS


def classify(vector_status: str | None, coverage: Mapping[str, Any] | None) -> str:
    """Derive the degradation class from the statuses retrieval already sets.

    Existing ``vector_status`` values are preserved and read, never renamed.
    """

    facts = dict(coverage or {})
    if vector_status == "unavailable":
        return VECTOR_UNAVAILABLE
    if vector_status == "no_coverage":
        return ZERO_COVERAGE
    if vector_status == "empty":
        return EMPTY_VECTOR_RESULT
    if not facts.get("available"):
        # A retriever that cannot answer the question at all is a capability
        # gap, not a degradation; one that tried and failed is a degradation.
        return COVERAGE_UNKNOWN if facts.get("error") else HEALTHY
    candidates = int(facts.get("candidate_count") or 0)
    embedded = int(facts.get("embedded_count") or 0)
    if not candidates:
        return HEALTHY
    if not embedded:
        return ZERO_COVERAGE
    if embedded < candidates:
        return PARTIAL_COVERAGE
    return HEALTHY


def _ratio(coverage: Mapping[str, Any]) -> str:
    value = coverage.get("ratio")
    if value is None:
        candidates = int(coverage.get("candidate_count") or 0)
        embedded = int(coverage.get("embedded_count") or 0)
        value = round(embedded / candidates, 6) if candidates else 0.0
    return f"{float(value):g}"


def warning_for(
    degradation: str,
    coverage: Mapping[str, Any] | None,
    *,
    provider: Any = None,
    error: str | None = None,
) -> str | None:
    """The single warning token for a degraded request, or None when healthy.

    Deterministic and free of free-form detail: an exception *type* is reported
    but never its message, which can carry a DSN or other connection detail.
    """

    token = _WARNING_TOKENS.get(degradation)
    if token is None:
        return None
    fields = [f"provider={normalized_provider(provider) or 'unknown'}"]
    if degradation in (ZERO_COVERAGE, PARTIAL_COVERAGE):
        facts = dict(coverage or {})
        fields.append(f"candidates={int(facts.get('candidate_count') or 0)}")
        fields.append(f"embedded={int(facts.get('embedded_count') or 0)}")
        fields.append(f"ratio={_ratio(facts)}")
    elif degradation in (VECTOR_UNAVAILABLE, COVERAGE_UNKNOWN):
        fields.append(f"error={_error_type(error) or 'unknown'}")
    return f"{token}:{','.join(fields)}"


def _error_type(error: str | None) -> str | None:
    """``"RuntimeError: could not connect to host=..."`` -> ``"RuntimeError"``."""

    if not error:
        return None
    head = str(error).split(":", 1)[0].strip()
    return head or None


def assert_complete_coverage(
    coverage_by_question: Mapping[str, Mapping[str, Any]],
    *,
    identity: Mapping[str, Any] | None = None,
) -> None:
    """Refuse to let a real-vector run publish metrics it did not earn.

    Complete means, for every evaluated question, that every vector-eligible
    candidate had a stored vector under the exact configured
    model/version/dimensions -- not merely that some vectors existed.
    """

    unknown = sorted(
        qid for qid, counts in coverage_by_question.items()
        if not counts.get("available", True)
    )
    starved = sorted(
        qid for qid, counts in coverage_by_question.items()
        if int(counts.get("candidate_count") or 0) > 0
        and int(counts.get("embedded_count") or 0) == 0
    )
    incomplete = {
        qid: (int(c.get("candidate_count") or 0), int(c.get("embedded_count") or 0))
        for qid, c in coverage_by_question.items()
        if int(c.get("embedded_count") or 0) < int(c.get("candidate_count") or 0)
    }
    if not unknown and not starved and not incomplete:
        return

    expected = sum(int(c.get("candidate_count") or 0)
                   for c in coverage_by_question.values())
    stored = sum(int(c.get("embedded_count") or 0)
                 for c in coverage_by_question.values())
    facts = dict(identity or {})
    lines = [
        "vector coverage is incomplete; METRICS WERE NOT PRODUCED and this run "
        "must not be labelled a real vector evaluation.",
        f"  provider:    {facts.get('provider')}",
        f"  model:       {facts.get('model')}",
        f"  version:     {facts.get('version')}",
        f"  dimensions:  {facts.get('dimensions')}",
        f"  questions:   {len(coverage_by_question)}",
        f"  expected candidate vectors: {expected}",
        f"  stored matching vectors:    {stored}",
        f"  missing:                    {max(expected - stored, 0)}",
    ]
    if starved:
        lines.append(
            "  questions with vector-eligible candidates but zero stored vectors, "
            f"which would silently degrade to lexical-only: {starved}"
        )
    if incomplete:
        lines.append(
            "  incomplete (question: eligible/embedded): "
            + str({q: f"{e}/{m}" for q, (e, m) in sorted(incomplete.items())})
        )
    if unknown:
        lines.append(f"  coverage could not be verified: {unknown}")
    raise VectorCoverageError("\n".join(lines))
