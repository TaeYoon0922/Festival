"""Values that never travel without saying whose they are and where they came from.

Several questions in this corpus are not answered by one number read from one
filing.  They ask what the difference is between a contract's original amount
and its corrected one, or which of four companies signed the largest contract.
Both need the same thing and neither had it: more than one value, each bound to
the scope that selected it and the document it was read from, kept together
until the arithmetic is done and the answer cites them.

The flat field lane could not express that.  It resolves *one* authoritative
cell for *one* question, so a second contract by the same company is not a
second operand -- it is a conflict, and the field comes back unresolved.  What
follows keeps the scope attached to the value instead:

    scope (role, company, filing date, requested field)
      -> exactly one source document
      -> one value
      -> one citation

The invariant is that no step may drop a link in that chain.  A value with no
source cannot be cited, and a value with no scope can be attributed to the
wrong company, which is the one failure mode a comparison must never have.  So
resolution is strict on both sides: a scope no document uniquely satisfies
stays unresolved, and one document can satisfy at most one scope.

Reading the number itself is not this module's job.  ``_amount_cells`` already
knows which labelled cells of a filing carry a formal amount and which are the
corpus's own blanks, so it is reused whole rather than reimplemented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.reasoning.corporate_event_field_evidence import (
    CONTRACT_AMOUNT,
    _amount_cells,
)

#: The amount a termination filing states, which is a different cell from the
#: amount the contract was concluded at.  Requested only when the question says
#: so; a lifecycle question that names no amount asks for neither.
TERMINATION_AMOUNT = "termination_amount"

#: What an operand is *for*.  A delta has two, ordered in time; a comparison has
#: one per company and no inherent order until the values decide it.
ROLE_INITIAL = "initial"
ROLE_FINAL = "final"
ROLE_MEMBER = "member"

#: Operations these questions actually ask for.  Deliberately three: this is not
#: an expression language, and anything it cannot name it declines.
OP_DIFFERENCE = "difference"
OP_RANKING = "ranking"

_DIGITS_ONLY = re.compile(r"[^\d-]")


@dataclass(frozen=True)
class OperandScope:
    """Which document a value must come from, and what it will be called."""

    role: str
    #: Whose filing.  Required for a comparison, optional for a delta whose two
    #: operands are two filings of the same company.
    corp_code: str | None = None
    company: str | None = None
    #: The receipt date the question named, ``YYYY-MM-DD``, or a ``YYYY-MM``
    #: prefix when it named only a month.  This is what keeps an unrelated later
    #: contract by the same company out of the operand.
    on_date: str | None = None
    #: Which labelled amount to read.
    field: str = CONTRACT_AMOUNT

    @property
    def label(self) -> str:
        return self.company or self.role

    def matches(self, source: "AmountSource") -> bool:
        if self.corp_code and str(source.corp_code or "") != self.corp_code:
            return False
        if self.on_date and not str(source.receipt_date or "").startswith(self.on_date):
            return False
        return source.value_for(self.field) is not None


@dataclass(frozen=True)
class AmountSource:
    """One served filing, and the amounts its own table rows state."""

    doc_id: str
    chunk_id: str
    corp_code: str | None
    receipt_date: str | None
    report_nm: str | None
    amounts: Mapping[str, int]

    def value_for(self, field: str) -> int | None:
        value = self.amounts.get(field)
        return int(value) if value is not None else None


@dataclass(frozen=True)
class ResolvedOperand:
    """A value that still knows its scope and its document."""

    scope: OperandScope
    value: int | None = None
    source: AmountSource | None = None

    @property
    def resolved(self) -> bool:
        return self.value is not None and self.source is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.scope.role,
            "label": self.scope.label,
            "corp_code": self.scope.corp_code,
            "field": self.scope.field,
            "value": self.value,
            "doc_id": getattr(self.source, "doc_id", None),
            "chunk_id": getattr(self.source, "chunk_id", None),
        }


@dataclass(frozen=True)
class Delta:
    """final - initial, and what that is as a proportion of where it started."""

    initial: ResolvedOperand
    final: ResolvedOperand
    difference: int
    pct_change: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": OP_DIFFERENCE,
            "difference": self.difference,
            "pct_change": self.pct_change,
            "operands": [self.initial.to_dict(), self.final.to_dict()],
        }


@dataclass(frozen=True)
class Ranking:
    """Operands ordered by value, largest first, each keeping its own source."""

    operands: tuple[ResolvedOperand, ...]

    @property
    def largest(self) -> ResolvedOperand | None:
        return self.operands[0] if self.operands else None

    @property
    def spread(self) -> int | None:
        """The distance between the extremes, which is what "차이" asks for."""

        if len(self.operands) < 2:
            return None
        return abs(int(self.operands[0].value) - int(self.operands[-1].value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": OP_RANKING,
            "order": [operand.scope.label for operand in self.operands],
            "spread": self.spread,
            "operands": [operand.to_dict() for operand in self.operands],
        }


def amount_source(chunk: Mapping[str, Any]) -> AmountSource | None:
    """Read one served chunk's formal amounts, or nothing if it states none.

    The label vocabulary and the blank-cell rules belong to the field-evidence
    reader and are used exactly as it defines them.  A filing whose amount cell
    is the corpus's own blank yields no value here, so it can never silently
    become a zero in an arithmetic answer.
    """

    doc_id = str(chunk.get("doc_id") or "")
    chunk_id = str(chunk.get("chunk_id") or "")
    if not doc_id or not chunk_id:
        return None
    amounts: dict[str, int] = {}
    for cell in _amount_cells(chunk):
        value = _amount(cell.value)
        if value is None:
            continue
        field = (
            TERMINATION_AMOUNT
            if "해지" in str(chunk.get("report_nm") or "")
            else CONTRACT_AMOUNT
        )
        amounts.setdefault(field, value)
        # A termination filing's amount also answers a question that asked for
        # the contract amount of that filing by name, so it is reachable under
        # both keys; the scope decides which one was requested.
        amounts.setdefault(CONTRACT_AMOUNT, value)
    if not amounts:
        return None
    return AmountSource(
        doc_id=doc_id,
        chunk_id=chunk_id,
        corp_code=_text(chunk.get("corp_code")),
        receipt_date=_receipt(chunk.get("rcept_dt")),
        report_nm=_text(chunk.get("report_nm")),
        amounts=amounts,
    )


def amount_sources(chunks: Sequence[Mapping[str, Any]]) -> tuple[AmountSource, ...]:
    """One source per document, keeping the first chunk that states an amount."""

    sources: dict[str, AmountSource] = {}
    for chunk in chunks:
        source = amount_source(chunk)
        if source is not None:
            sources.setdefault(source.doc_id, source)
    return tuple(sources.values())


def resolve_operands(
    scopes: Sequence[OperandScope], sources: Sequence[AmountSource]
) -> tuple[ResolvedOperand, ...]:
    """Bind each scope to the one document that satisfies it, or to nothing.

    Two rules do all the safety work.  A scope matched by more than one document
    is left unresolved rather than served by the first: that is the "unrelated
    later contract by the same company" case, and picking one would answer with
    an amount the question never referred to.  And a document already taken by
    an earlier scope cannot be taken again, so one company's filing can never
    supply another company's operand.
    """

    taken: set[str] = set()
    resolved: list[ResolvedOperand] = []
    for scope in scopes:
        candidates = [
            source
            for source in sources
            if source.doc_id not in taken and scope.matches(source)
        ]
        if len(candidates) != 1:
            resolved.append(ResolvedOperand(scope=scope))
            continue
        source = candidates[0]
        taken.add(source.doc_id)
        resolved.append(
            ResolvedOperand(
                scope=scope, value=source.value_for(scope.field), source=source
            )
        )
    return tuple(resolved)


def difference(operands: Sequence[ResolvedOperand]) -> Delta | None:
    """``final - initial``, with the proportion when the start is non-zero."""

    initial = _by_role(operands, ROLE_INITIAL)
    final = _by_role(operands, ROLE_FINAL)
    if initial is None or final is None:
        return None
    change = int(final.value) - int(initial.value)
    pct = (
        round(change / int(initial.value) * 100, 2)
        if int(initial.value) != 0
        else None
    )
    return Delta(initial=initial, final=final, difference=change, pct_change=pct)


def ranking(operands: Sequence[ResolvedOperand]) -> Ranking | None:
    """Every operand ordered by value, largest first.

    All or nothing: a ranking missing one of its members states a different
    order than the true one, and a partial ordering presented as complete is
    worse than declining.
    """

    members = [operand for operand in operands if operand.resolved]
    if not members or len(members) != len(operands):
        return None
    ordered = sorted(members, key=lambda operand: int(operand.value), reverse=True)
    return Ranking(operands=tuple(ordered))


def _by_role(operands: Sequence[ResolvedOperand], role: str) -> ResolvedOperand | None:
    for operand in operands:
        if operand.scope.role == role and operand.resolved:
            return operand
    return None


def _amount(value: Any) -> int | None:
    text = _DIGITS_ONLY.sub("", str(value or ""))
    if not text or text in {"-", ""}:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _receipt(value: Any) -> str | None:
    text = re.sub(r"\D", "", str(value or ""))
    if len(text) != 8:
        return _text(value)
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


__all__ = [
    "AmountSource",
    "Delta",
    "OP_DIFFERENCE",
    "OP_RANKING",
    "OperandScope",
    "ROLE_FINAL",
    "ROLE_INITIAL",
    "ROLE_MEMBER",
    "Ranking",
    "ResolvedOperand",
    "TERMINATION_AMOUNT",
    "amount_source",
    "amount_sources",
    "difference",
    "ranking",
    "resolve_operands",
]
