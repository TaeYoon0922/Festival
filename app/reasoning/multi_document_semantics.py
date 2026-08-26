"""Semantic guards for P0-C answers (Step 6).

The deterministic layer already knows whether a termination exists, does not
exist, or could not be determined.  This module checks that the *rendered*
answer still says the same thing, because a count being right is worthless if
the sentence built from it reads as the opposite claim.

The guard is pattern-based, not a model.  It is deliberately narrow: a bare
substring test would reject the correct wording, since "해지 여부를 단정할 수
없습니다" contains "없습니다" while asserting the exact opposite of "해지된
계약은 없습니다".  So confident negatives are matched as whole claims, and an
uncertainty marker is required separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.reasoning.multi_document_evidence import (
    LIFECYCLE_EXISTS,
    LIFECYCLE_NONE,
    LIFECYCLE_NO_MEMBERS,
    LIFECYCLE_UNDETERMINED,
)


#: A claim that no termination exists.  Each pattern is a *complete* negative
#: assertion, so hedged wording built on the same words does not match.
_CONFIDENT_NEGATIVE = (
    re.compile(r"해지(된|한)?\s*계약(은|이|도)?\s*(전혀\s*)?없(습니다|다|음|어요)"),
    re.compile(r"해지(된|되었|하였)?\s*(적|바)\s*(가|이)?\s*없"),
    re.compile(r"해지\s*되지\s*않았"),
    re.compile(r"해지\s*사실\s*(이|은)?\s*없"),
    re.compile(r"해지(된)?\s*계약(은|이)?\s*0\s*건"),
)

#: Wording that admits the set was not fully verified.
_UNCERTAINTY = (
    re.compile(r"단정할\s*수\s*없"),
    re.compile(r"확정(할\s*수\s*없|하지\s*못)"),
    re.compile(r"확인(할\s*수\s*없|하지\s*못|되지\s*않은)"),
    re.compile(r"미확인"),
    re.compile(r"불확실"),
)

#: Wording that asserts a termination was found.
_POSITIVE = (
    re.compile(r"해지(된|한)?\s*계약(은|이)?\s*\d+\s*건"),
    re.compile(r"해지(된)?\s*계약이\s*(존재|있)"),
    re.compile(r"해지되었"),
)

#: Wording that would mis-report an empty set as a retrieval failure.
_RETRIEVAL_FAILURE = (
    re.compile(r"검색(에)?\s*실패"),
    re.compile(r"문서를\s*찾지\s*못"),
    re.compile(r"자료가\s*부족"),
)


@dataclass(frozen=True)
class SemanticVerdict:
    """Whether a rendered answer preserved its deterministic meaning."""

    lifecycle_answer: str | None
    ok: bool
    reason: str | None = None

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.ok


def _any(patterns, text: str) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def has_confident_negative(text: str) -> bool:
    """Whether the text claims outright that no termination exists."""

    return _any(_CONFIDENT_NEGATIVE, text)


def has_uncertainty(text: str) -> bool:
    return _any(_UNCERTAINTY, text)


def check_answer(lifecycle_answer: str | None, text: str) -> SemanticVerdict:
    """Check a rendered answer against the state the executor determined.

    The rule that matters most: ``undetermined`` may never render as a confident
    negative, and must carry an explicit uncertainty marker.  Getting the counts
    right does not excuse a sentence that says "없습니다".
    """

    body = text or ""
    if lifecycle_answer == LIFECYCLE_EXISTS:
        if has_confident_negative(body):
            return SemanticVerdict(lifecycle_answer, False, "negative_for_exists")
        if not _any(_POSITIVE, body):
            return SemanticVerdict(lifecycle_answer, False, "no_positive_assertion")
        return SemanticVerdict(lifecycle_answer, True)

    if lifecycle_answer == LIFECYCLE_NONE:
        if has_uncertainty(body):
            return SemanticVerdict(lifecycle_answer, False, "uncertainty_for_none")
        if not (has_confident_negative(body) or "확인되지 않았" in body):
            return SemanticVerdict(lifecycle_answer, False, "no_negative_assertion")
        return SemanticVerdict(lifecycle_answer, True)

    if lifecycle_answer == LIFECYCLE_UNDETERMINED:
        if has_confident_negative(body):
            # The failure Step 6 exists to prevent.
            return SemanticVerdict(
                lifecycle_answer, False, "confident_negative_for_undetermined"
            )
        if not has_uncertainty(body):
            return SemanticVerdict(lifecycle_answer, False, "no_uncertainty_marker")
        return SemanticVerdict(lifecycle_answer, True)

    if lifecycle_answer == LIFECYCLE_NO_MEMBERS:
        if _any(_RETRIEVAL_FAILURE, body):
            return SemanticVerdict(lifecycle_answer, False, "reads_as_retrieval_failure")
        if "확인되지 않" not in body and "없" not in body:
            return SemanticVerdict(lifecycle_answer, False, "no_empty_set_assertion")
        return SemanticVerdict(lifecycle_answer, True)

    return SemanticVerdict(lifecycle_answer, True)


__all__ = [
    "SemanticVerdict",
    "check_answer",
    "has_confident_negative",
    "has_uncertainty",
]
