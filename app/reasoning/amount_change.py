"""How much a contract's amount moved between two of its filings.

A contract is filed once and then corrected, or filed once and then terminated,
and the question asks for the distance between the two amounts.  Neither filing
states that distance: the first does not know it will be revised, and the second
states only where it ended up.  So the answer is arithmetic over two documents,
and the only safe way to do arithmetic over documents is to keep each value
bound to the filing it came from -- which is what :mod:`scoped_operands` is for.

This module is the narrow part: reading which two filings a question is talking
about, and saying the result.  It recognizes one question shape and declines
everything else.  There is no expression language here and no general
calculator; the corpus asks for a difference and a percentage, and that is what
this can express.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.reasoning.corporate_event_field_evidence import CONTRACT_AMOUNT
from app.reasoning.scoped_operands import (
    ROLE_FINAL,
    ROLE_INITIAL,
    TERMINATION_AMOUNT,
    Delta,
    OperandScope,
    amount_sources,
    difference,
    resolve_operands,
)

#: ``plan.evidence`` key describing the two filings a change question compares.
AMOUNT_CHANGE_KEY = "contract_amount_change"

#: The question has to be about an amount.  Either the field by name, or the
#: bare noun when a contract already governs the sentence.
_NAMES_AMOUNT = re.compile(r"계약금액|해지금액|계약의금액|금액")
#: And it has to ask how far it moved.  Without this a question naming two
#: filings is just a question about two filings.
_ASKS_CHANGE = re.compile(
    r"차이|증감|증감액|증감률|얼마나\s*(?:늘|줄|커|작|변|증가|감소)"
    r"|늘었|줄었|증가했|감소했|변경됐|바뀌었|변동됐"
)
#: The later filing is a correction of the first, or its termination.  Which one
#: decides whether the final operand reads 계약금액 or 해지금액.
_LATER_IS_TERMINATION = re.compile(r"해지\s*공시|해지금액|해지된")

#: ``2024년 3월 15일`` and ``2024년 12월``, in the order the question writes them.
_FULL_DATE = re.compile(r"(?P<y>\d{4})\s*년\s*(?P<m>\d{1,2})\s*월\s*(?P<d>\d{1,2})\s*일")
_MONTH = re.compile(r"(?P<y>\d{4})\s*년\s*(?P<m>\d{1,2})\s*월")


@dataclass(frozen=True)
class AmountChangeRequest:
    """The two filings a change question names, as date scopes."""

    initial_on: str
    final_on: str
    final_field: str = CONTRACT_AMOUNT

    def scopes(self, corp_code: str | None) -> tuple[OperandScope, ...]:
        return (
            OperandScope(
                role=ROLE_INITIAL, corp_code=corp_code, on_date=self.initial_on
            ),
            OperandScope(
                role=ROLE_FINAL,
                corp_code=corp_code,
                on_date=self.final_on,
                field=self.final_field,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_on": self.initial_on,
            "final_on": self.final_on,
            "final_field": self.final_field,
        }


def amount_change_requested(query: str) -> AmountChangeRequest | None:
    """The two dated filings this question compares, or nothing.

    Both halves are required: an amount, and a question about how far it moved.
    Two distinct dates are required as well, because the whole point is that the
    two amounts live in two filings -- one date names one filing and cannot pose
    this question at all.
    """

    text = str(query or "")
    compact = re.sub(r"\s+", "", text)
    if not _NAMES_AMOUNT.search(compact) or not _ASKS_CHANGE.search(compact):
        return None
    dates = _dated_scopes(text)
    if len(dates) < 2 or dates[0] == dates[-1]:
        return None
    field = (
        TERMINATION_AMOUNT
        if _LATER_IS_TERMINATION.search(compact)
        else CONTRACT_AMOUNT
    )
    return AmountChangeRequest(
        initial_on=dates[0], final_on=dates[-1], final_field=field
    )


def _dated_scopes(text: str) -> list[str]:
    """Every date the question writes, in order, as a date or month prefix.

    A full date scopes to one day and a bare month to that month, which is how
    these questions are actually written: the original filing is named to the
    day and the revision only by the month it landed in.
    """

    found: list[tuple[int, str]] = []
    for match in _FULL_DATE.finditer(text):
        found.append(
            (
                match.start(),
                "%04d-%02d-%02d"
                % (int(match.group("y")), int(match.group("m")), int(match.group("d"))),
            )
        )
    covered = [match.span() for match in _FULL_DATE.finditer(text)]
    for match in _MONTH.finditer(text):
        if any(start <= match.start() < end for start, end in covered):
            continue
        found.append(
            (match.start(), "%04d-%02d" % (int(match.group("y")), int(match.group("m"))))
        )
    return [value for _position, value in sorted(found)]


def requested_amount_change(plan: Any) -> AmountChangeRequest | None:
    """What query understanding recorded, read without re-parsing the question."""

    value = dict(getattr(plan, "evidence", None) or {}).get(AMOUNT_CHANGE_KEY)
    if not isinstance(value, Mapping):
        return None
    try:
        return AmountChangeRequest(
            initial_on=str(value["initial_on"]),
            final_on=str(value["final_on"]),
            final_field=str(value.get("final_field") or CONTRACT_AMOUNT),
        )
    except KeyError:
        return None


def resolve_amount_change(
    request: AmountChangeRequest,
    chunks: Sequence[Mapping[str, Any]],
    *,
    corp_code: str | None = None,
) -> Delta | None:
    """Read both amounts from the served filings and take the difference."""

    operands = resolve_operands(request.scopes(corp_code), amount_sources(chunks))
    return difference(operands)


def compose_amount_change_text(delta: Delta) -> str:
    """State the change, its direction, and the two amounts it came from."""

    sign = "+" if delta.difference >= 0 else "-"
    lines = [f"계약금액 증감: {sign}{abs(delta.difference):,}원"]
    if delta.pct_change is not None:
        lines.append(f"증감률: {delta.pct_change:+.2f}%")
    lines.append(f"최초 계약금액: {int(delta.initial.value):,}원")
    lines.append(f"변경 후 금액: {int(delta.final.value):,}원")
    return "\n".join(lines)


__all__ = [
    "AMOUNT_CHANGE_KEY",
    "AmountChangeRequest",
    "amount_change_requested",
    "compose_amount_change_text",
    "requested_amount_change",
    "resolve_amount_change",
]
