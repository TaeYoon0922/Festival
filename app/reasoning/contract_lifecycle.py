"""What became of a contract, read from the filings that were already served.

A question that follows a contract forward -- *was it terminated*, *what is its
final state*, *what happened to it after that* -- is answered by two documents
that retrieval already returns together: the filing that concluded the contract
and, when one exists, the filing that terminated it.  The lifecycle expansion
puts both in the evidence set.  What was missing is the last step: saying which
is which and stating the outcome, instead of listing both as undifferentiated
evidence rows.

Two things follow from that, and both are handled here.

The terminal filing is selected by *role*, not by rank.  A termination is a
short document that shares almost no wording with the question, so it ranks
low -- on the server it arrived seventh, past the point where general evidence
composition stops reading -- and an answer about a contract's end that omits the
document ending it is missing its own subject.  Once the role is known the rank
does not matter.

Roles are read from the filing's own name.  ``단일판매ㆍ공급계약해지`` is the
corpus stating what the document is; nothing here infers a termination from
prose, from a date, or from the absence of something.  A correction to a
contract is a correction, never a termination, however much it changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

#: The filing that concluded the contract.
ROLE_ORIGIN = "contract_origin"
#: The filing that terminated it.
ROLE_TERMINATION = "contract_termination"

#: How the corpus names each family.  Matched against ``report_nm`` -- the
#: document's own title -- after whitespace and the separators DART writes
#: inside filing names are removed.  A ``[기재정정]`` prefix marks a corrected
#: reprint of that filing and does not change which family it belongs to.
_TERMINATION_NAME = re.compile(r"공급계약해지")
_ORIGIN_NAME = re.compile(r"공급계약체결")

#: The disclosure field naming the day a termination took effect, which is not
#: the day it was filed.  Read only from the terminal document, and only when
#: the document states it: an outcome may be reported without one.
_EFFECTIVE_DATE_FIELD = re.compile(
    r"해지\s*(?:일자|일)\s*[:：]?\s*"
    r"(?P<date>\d{4}\s*[-./년]\s*\d{1,2}\s*[-./월]\s*\d{1,2}\s*일?)"
)
_DATE_PARTS = re.compile(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})")


@dataclass(frozen=True)
class LifecycleOutcome:
    """The contract's story, as the served filings tell it."""

    origin: Any | None
    terminal: Any | None
    effective_date: str | None = None

    @property
    def terminated(self) -> bool:
        return self.terminal is not None

    @property
    def documents(self) -> tuple[Any, ...]:
        """Both ends of the lifecycle, in the order they happened."""

        return tuple(item for item in (self.origin, self.terminal) if item is not None)

    @property
    def resolved(self) -> bool:
        """Whether anything was established that plain evidence rows would not."""

        return self.origin is not None or self.terminal is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "terminated": self.terminated,
            "origin_doc_id": getattr(self.origin, "doc_id", None),
            "terminal_doc_id": getattr(self.terminal, "doc_id", None),
            "effective_date": self.effective_date,
        }


def lifecycle_role(item: Any) -> str | None:
    """Which end of a contract lifecycle a served filing is, by its own name."""

    name = re.sub(r"[\sㆍ·・‧]+", "", str(getattr(item, "report_nm", "") or ""))
    if not name:
        return None
    if _TERMINATION_NAME.search(name):
        return ROLE_TERMINATION
    if _ORIGIN_NAME.search(name):
        return ROLE_ORIGIN
    return None


def lifecycle_outcome(items: Sequence[Any]) -> LifecycleOutcome | None:
    """Bind the origin and, when served, the filing that ended it.

    The earliest conclusion filing is the origin and the latest termination is
    the outcome, both by receipt date, so a contract that was corrected several
    times still has one beginning and one end.  Nothing is inferred from an
    absence: with no termination served the outcome simply carries none, and the
    composer says so rather than asserting the contract is still live.
    """

    origins = [item for item in items if lifecycle_role(item) == ROLE_ORIGIN]
    terminations = [item for item in items if lifecycle_role(item) == ROLE_TERMINATION]
    if not origins and not terminations:
        return None
    origin = min(origins, key=_receipt_key) if origins else None
    terminal = max(terminations, key=_receipt_key) if terminations else None
    return LifecycleOutcome(
        origin=origin,
        terminal=terminal,
        effective_date=_effective_date(terminal) if terminal is not None else None,
    )


def _receipt_key(item: Any) -> str:
    return str(getattr(item, "rcept_dt", "") or "")


def _effective_date(item: Any) -> str | None:
    match = _EFFECTIVE_DATE_FIELD.search(str(getattr(item, "evidence_text", "") or ""))
    if match is None:
        return None
    return _iso(match.group("date"))


def _iso(value: str) -> str | None:
    parts = _DATE_PARTS.search(value)
    if parts is None:
        return None
    year, month, day = parts.groups()
    return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"


def _display_date(value: str | None) -> str | None:
    """``YYYYMMDD`` or ``YYYY-MM-DD`` as the answer states dates."""

    text = re.sub(r"\D", "", str(value or ""))
    if len(text) != 8:
        return None
    return f"{int(text[:4])}년 {int(text[4:6])}월 {int(text[6:])}일"


def compose_lifecycle_text(outcome: LifecycleOutcome) -> str | None:
    """State the outcome, naming the dates the served filings actually carry."""

    if not outcome.resolved:
        return None
    lines: list[str] = []
    origin_date = _display_date(getattr(outcome.origin, "rcept_dt", None))
    if outcome.terminated:
        lines.append("해당 계약은 이후 해지되었습니다.")
        filed = _display_date(getattr(outcome.terminal, "rcept_dt", None))
        if filed:
            lines.append(f"해지 공시일: {filed}")
        if outcome.effective_date:
            effective = _display_date(outcome.effective_date)
            if effective:
                lines.append(f"해지일자: {effective}")
    else:
        # No termination filing was served.  That is the absence of evidence,
        # not evidence of continuation, and the answer says only what it has.
        lines.append("서비스된 공시에서는 해지 공시가 확인되지 않습니다.")
    if origin_date:
        lines.append(f"최초 계약 공시일: {origin_date}")
    return "\n".join(lines)


def lifecycle_items(
    outcome: LifecycleOutcome, served: Sequence[Any], *, limit: int
) -> tuple[Any, ...]:
    """The evidence an outcome has to cite, ahead of whatever else was served.

    Both ends come first regardless of where retrieval ranked them -- that is
    the whole point -- and the rest of the served evidence fills the remaining
    room in its own order.
    """

    chosen: dict[str, Any] = {}
    for item in (*outcome.documents, *served):
        chunk_id = str(getattr(item, "chunk_id", "") or "")
        if chunk_id and chunk_id not in chosen:
            chosen[chunk_id] = item
        if len(chosen) >= max(limit, len(outcome.documents)):
            break
    return tuple(chosen.values())


#: ``plan.evidence`` key set when the question follows a contract forward.
LIFECYCLE_OUTCOME_KEY = "contract_lifecycle_outcome"

#: The contract noun a lifecycle question anchors on, and how it asks.
_CONTRACT = r"계약"
#: ``해지`` bound closely to the contract, as in ``계약은 해지됐나``.
_ASKS_TERMINATED = re.compile(_CONTRACT + r"[^\s]{0,6}?해지")
#: ``계약의 최종 상태``.
_ASKS_FINAL_STATE = re.compile(_CONTRACT + r".{0,4}?최종상태")
#: ``이후 어떻게 됐어`` -- the outcome asked for without naming it.
_ASKS_WHAT_HAPPENED = re.compile(r"이후.{0,6}?(?:어떻게|어찌)")
#: Change vocabulary.  A question asking *how much* something moved is asking
#: for arithmetic across filings, not for a lifecycle state, even when it names
#: a termination as one of its two operands.
_ASKS_CHANGE = re.compile(
    r"차이|증감|증가|감소|얼마나\s*(?:늘|줄|커|작|변)|변경됐|바뀌었|늘었|줄었"
)


def lifecycle_outcome_requested(query: str) -> str | None:
    """Whether this question follows a contract forward, and on what wording.

    Declines outright when the question asks how much something changed: that
    is a two-operand arithmetic question which happens to mention a
    termination, and answering it with a lifecycle state would drop the number
    it actually asked for.
    """

    compact = re.sub(r"\s+", "", str(query or ""))
    if not compact or _ASKS_CHANGE.search(compact):
        return None
    for pattern in (_ASKS_TERMINATED, _ASKS_FINAL_STATE, _ASKS_WHAT_HAPPENED):
        match = pattern.search(compact)
        if match:
            return match.group(0)
    return None


def requested_lifecycle_outcome(plan: Any) -> str | None:
    """The wording query understanding recorded, read without re-parsing."""

    evidence = getattr(plan, "evidence", None)
    value = dict(evidence or {}).get(LIFECYCLE_OUTCOME_KEY)
    return str(value) if value else None


__all__ = [
    "LIFECYCLE_OUTCOME_KEY",
    "LifecycleOutcome",
    "ROLE_ORIGIN",
    "ROLE_TERMINATION",
    "compose_lifecycle_text",
    "lifecycle_items",
    "lifecycle_outcome",
    "lifecycle_outcome_requested",
    "lifecycle_role",
    "requested_lifecycle_outcome",
]
