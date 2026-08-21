"""Immutable placeholders for literals a verbalizer must not rewrite.

A language model asked to make text read naturally will normalize it: it turns
``2023년 03월 07일`` into ``2023년 3월 7일``, drops citation markers it reads as
noise, and adds emphasis.  Those edits are invisible to a prompt but fatal to a
cited disclosure answer.

This module removes the temptation.  Before the model sees the text, every
citation marker, date, and number is replaced by a deterministic token that
carries no digits and no meaning.  The model rewrites the prose around those
tokens.  Afterwards the tokens are checked for exact survival and swapped back
for the original characters, byte for byte.  Anything less than exact survival
is a failure, never a repair: guessing which literal a mangled token meant is
how wrong numbers reach a reader.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


#: Placeholders deliberately contain no digits, so neither the numeric-token
#: tokenizer nor the citation-marker tokenizer in
#: :mod:`app.generation.answer_validator` can mistake one for a fact.
PLACEHOLDER_TEMPLATE = "__FESTIVAL_{kind}_{label}__"
PLACEHOLDER_PATTERN = re.compile(r"__FESTIVAL_[A-Z]+_[A-Z]+__")

#: Matched left to right with citations first and dates before bare numbers, so
#: a full date span is protected as one literal instead of being shredded into
#: its year, month, and day.
LITERAL_PATTERN = re.compile(
    r"(?P<citation>\[\d+\])"
    r"|(?P<date>"
    r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일"
    r"|\d{4}[-./]\d{1,2}[-./]\d{1,2}"
    r")"
    r"|(?P<number>-?\d[\d,]*(?:\.\d+)?%?)"
)

_KIND_LABELS = {
    "citation": "CITATION",
    "date": "DATE",
    "number": "NUMBER",
}

PLACEHOLDER_MISSING = "placeholder_missing"
PLACEHOLDER_UNEXPECTED = "placeholder_unexpected"
PLACEHOLDER_DUPLICATED = "placeholder_duplicated"
PLACEHOLDER_REORDERED = "placeholder_reordered"


@dataclass(frozen=True)
class ProtectedLiteral:
    """One literal held out of the model's reach."""

    placeholder: str
    text: str
    kind: str


@dataclass(frozen=True)
class ProtectedText:
    """The masked text plus everything needed to restore it exactly."""

    original: str
    masked: str
    literals: tuple[ProtectedLiteral, ...]

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(literal.placeholder for literal in self.literals)

    def mapping(self) -> dict[str, str]:
        return {literal.placeholder: literal.text for literal in self.literals}


@dataclass(frozen=True)
class IntegrityResult:
    """Whether a candidate preserved every placeholder exactly."""

    valid: bool
    reason: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "reason": self.reason, "detail": self.detail}


def contains_placeholder_syntax(text: str) -> bool:
    """Report text that already looks like a placeholder.

    Protection would be ambiguous on such input, so callers refuse it rather
    than risk restoring a token they did not create.
    """

    return PLACEHOLDER_PATTERN.search(text) is not None


def protect_literals(text: str) -> ProtectedText:
    """Replace citations, dates, and numbers with digit-free placeholders."""

    literals: list[ProtectedLiteral] = []

    def substitute(match: re.Match[str]) -> str:
        kind = match.lastgroup or "number"
        placeholder = PLACEHOLDER_TEMPLATE.format(
            kind=_KIND_LABELS[kind], label=_label(len(literals))
        )
        literals.append(
            ProtectedLiteral(placeholder=placeholder, text=match.group(0), kind=kind)
        )
        return placeholder

    masked = LITERAL_PATTERN.sub(substitute, text)
    return ProtectedText(original=text, masked=masked, literals=tuple(literals))


def check_placeholder_integrity(
    candidate: str, protection: ProtectedText
) -> IntegrityResult:
    """Require the candidate's placeholders to match the expected run exactly.

    Comparing the two sequences in order settles presence, count, duplication,
    and ordering at once; the checks below only exist to name which one broke.
    """

    found = PLACEHOLDER_PATTERN.findall(candidate)
    expected = list(protection.placeholders)
    if found == expected:
        return IntegrityResult(True, None, "every placeholder survived unchanged")

    found_set = set(found)
    expected_set = set(expected)

    missing = sorted(expected_set - found_set)
    if missing:
        return IntegrityResult(
            False, PLACEHOLDER_MISSING, "dropped: " + ", ".join(missing)
        )

    unexpected = sorted(found_set - expected_set)
    if unexpected:
        return IntegrityResult(
            False, PLACEHOLDER_UNEXPECTED, "invented: " + ", ".join(unexpected)
        )

    duplicated = sorted({token for token in found if found.count(token) > 1})
    if duplicated:
        return IntegrityResult(
            False, PLACEHOLDER_DUPLICATED, "repeated: " + ", ".join(duplicated)
        )

    return IntegrityResult(
        False,
        PLACEHOLDER_REORDERED,
        f"expected {' '.join(expected)} but found {' '.join(found)}",
    )


def restore_literals(candidate: str, protection: ProtectedText) -> str:
    """Swap every placeholder back for its original characters.

    Call only after :func:`check_placeholder_integrity` passes.  Restored text
    never contains placeholder syntax, so a single pass cannot cascade.
    """

    mapping = protection.mapping()
    return PLACEHOLDER_PATTERN.sub(lambda match: _restore(match, mapping), candidate)


def _restore(match: re.Match[str], mapping: Mapping[str, str]) -> str:
    token = match.group(0)
    if token not in mapping:
        raise ValueError(f"unknown placeholder in candidate: {token}")
    return mapping[token]


def _label(index: int) -> str:
    """Return A, B, ... Z, AA, AB, ... so labels stay digit-free."""

    label = ""
    position = index + 1
    while position > 0:
        position, remainder = divmod(position - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label
