"""Deterministic validation of a verbalized answer against the frozen rendering.

The verbalizer may only restate what the deterministic pipeline already proved.
Every check here compares a candidate string against the deterministic reference
string; no model, corpus, or database is consulted.  A failure is never fatal —
callers fall back to the deterministic answer.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


#: Citation markers rendered by the deterministic generator, e.g. ``[1]``.
CITATION_MARKER_PATTERN = re.compile(r"\[(\d+)\]")

#: Numbers keep their rendered form: separators, leading zeros, and percent
#: signs are all part of the token, so "2,967,759" never becomes "약 297만".
NUMERIC_TOKEN_PATTERN = re.compile(r"-?\d[\d,]*(?:\.\d+)?%?")

#: Forecast and recommendation language.  Only flagged when the verbalizer
#: introduces it; wording already present in the deterministic answer is kept.
FORBIDDEN_INVESTMENT_TERMS = (
    "매수",
    "매도",
    "추천",
    "목표주가",
    "투자의견",
    "비중확대",
    "비중축소",
    "저평가",
    "고평가",
    "유망",
    "수혜",
    "전망됩니다",
    "기대됩니다",
    "예상됩니다",
    "상승할",
    "하락할",
    "성장할",
)


@dataclass(frozen=True)
class ValidationPolicy:
    """Bounds applied to a verbalized answer."""

    max_length_ratio: float = 2.0
    forbidden_terms: tuple[str, ...] = FORBIDDEN_INVESTMENT_TERMS

    def __post_init__(self) -> None:
        if self.max_length_ratio <= 0:
            raise ValueError("max_length_ratio must be positive")


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validating one candidate answer."""

    valid: bool
    reason: str | None = None
    detail: str = ""
    changed_tokens: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "detail": self.detail,
            "changed_tokens": list(self.changed_tokens),
        }


def validate_verbalized_answer(
    candidate: str,
    *,
    reference: str,
    required_terms: Iterable[str] = (),
    policy: ValidationPolicy | None = None,
) -> ValidationResult:
    """Return whether ``candidate`` restates ``reference`` without changing facts."""

    rules = policy or ValidationPolicy()

    if not candidate or not candidate.strip():
        return ValidationResult(False, "empty_output", "verbalizer returned no text")

    if len(candidate) > rules.max_length_ratio * len(reference):
        return ValidationResult(
            False,
            "length_exceeded",
            f"candidate is {len(candidate)} chars against a "
            f"{len(reference)} char reference",
        )

    candidate_markers = extract_citation_markers(candidate)
    reference_markers = extract_citation_markers(reference)
    if candidate_markers != reference_markers:
        return ValidationResult(
            False,
            "citation_marker_changed",
            "citation markers differ from the deterministic answer",
            tuple(sorted(candidate_markers ^ reference_markers)),
        )

    candidate_numbers = extract_numeric_tokens(candidate)
    reference_numbers = extract_numeric_tokens(reference)
    if candidate_numbers != reference_numbers:
        return ValidationResult(
            False,
            "numeric_token_changed",
            "numbers or dates differ from the deterministic answer",
            tuple(sorted(candidate_numbers ^ reference_numbers)),
        )

    missing = _missing_required_terms(candidate, reference, required_terms)
    if missing:
        return ValidationResult(
            False,
            "entity_missing",
            "required entities are absent from the candidate",
            missing,
        )

    introduced = _introduced_terms(candidate, reference, rules.forbidden_terms)
    if introduced:
        return ValidationResult(
            False,
            "forbidden_investment_language",
            "candidate introduces forecast or recommendation language",
            introduced,
        )

    return ValidationResult(True, None, "candidate preserves every verified fact")


def extract_citation_markers(text: str) -> set[str]:
    return set(CITATION_MARKER_PATTERN.findall(text))


def extract_numeric_tokens(text: str) -> set[str]:
    """Collect rendered numbers, ignoring digits that belong to citation markers."""

    without_markers = CITATION_MARKER_PATTERN.sub(" ", text)
    return set(NUMERIC_TOKEN_PATTERN.findall(without_markers))


def _missing_required_terms(
    candidate: str, reference: str, required_terms: Iterable[str]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                term
                for term in _clean_terms(required_terms)
                if term in reference and term not in candidate
            }
        )
    )


def _introduced_terms(
    candidate: str, reference: str, terms: Sequence[str]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {term for term in terms if term in candidate and term not in reference}
        )
    )


def _clean_terms(terms: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(term).strip() for term in terms if str(term).strip())
