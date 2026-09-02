"""Comparing one stated field across several companies, one scope at a time.

A question naming several companies is usually a question this system must
refuse: two company names in a holding question are an issuer and a filer, not
two operands, and guessing which is which answers about the wrong party.  That
firewall is right and stays.

But some questions genuinely do name N companies and one comparable field --
*which of these contracts is the largest*, *put them in order* -- and refusing
them is only correct while nothing can execute them safely.  What makes them
safe is that each company is answered on its own: one scope per company, one
retrieval per scope, one document per operand, and a ranking only when every
company brought its own evidence.  No company's filing can stand in for
another's, and a ranking missing a member is not served at all, because a
partial order states a different order than the true one.

Recognition is deliberately narrow.  It requires an explicit comparison or
ordering, a field that is comparable across issuers, and every named company
resolved to its own code.  A holding role pair satisfies none of those: it names
no comparable contract field and asks for one party's position, so it keeps the
firewall it has today.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from app.reasoning.corporate_event_field_evidence import CONTRACT_AMOUNT
from app.reasoning.scoped_operands import (
    ROLE_MEMBER,
    OperandScope,
    Ranking,
    amount_sources,
    ranking,
    resolve_operands,
)

#: ``plan.evidence`` key describing an N-company comparison, when the question
#: is one.  Absent for every question that is not, which is nearly all of them.
COMPANY_COMPARISON_KEY = "company_comparison_request"

#: Asking which is larger, or for an order.  One of these must be present: a
#: question merely naming two companies is not comparing them.
_ASKS_COMPARISON = re.compile(
    r"더\s*(?:큰|많|높|작|적|낮)|가장\s*(?:큰|많|높|작|적|낮)"
    r"|큰\s*순서|작은\s*순서|순서대로|순으로|나열|차이가\s*얼마|차이는\s*얼마"
    r"|어느\s*(?:쪽|것|회사)|누가\s*더|비교"
)
#: Ordering rather than a single winner.  Only changes how the answer reads.
_ASKS_ORDER = re.compile(r"순서대로|순으로|나열|순위")

#: The field being compared.  One field, named explicitly: a comparison is only
#: meaningful when both sides state the same thing in the same unit, and
#: 계약금액 is the one such field this corpus exposes across issuers.
_NAMES_CONTRACT_AMOUNT = re.compile(r"계약금액|계약\s*금액")


@dataclass(frozen=True)
class CompanyComparisonRequest:
    """One operand per company, plus how the answer should read."""

    companies: tuple[tuple[str, str], ...]
    field: str = CONTRACT_AMOUNT
    ordered: bool = False

    @property
    def size(self) -> int:
        return len(self.companies)

    def scopes(self) -> tuple[OperandScope, ...]:
        return tuple(
            OperandScope(
                role=ROLE_MEMBER, corp_code=code, company=name, field=self.field
            )
            for name, code in self.companies
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "companies": [list(pair) for pair in self.companies],
            "field": self.field,
            "ordered": self.ordered,
        }


def comparison_requested(
    query: str,
    companies: Sequence[str],
    corp_codes: Sequence[str],
) -> CompanyComparisonRequest | None:
    """Whether this question compares one field across the companies it names.

    Every company must carry its own code: an unresolved company cannot own an
    operand, and a comparison one of whose sides cannot be retrieved separately
    is exactly the guess this refuses to make.
    """

    names = [str(name).strip() for name in companies if str(name).strip()]
    codes = [str(code).strip() for code in corp_codes if str(code).strip()]
    if len(names) < 2 or len(names) != len(codes) or len(set(codes)) != len(codes):
        return None
    compact = re.sub(r"\s+", "", str(query or ""))
    if not _ASKS_COMPARISON.search(compact):
        return None
    if not _NAMES_CONTRACT_AMOUNT.search(compact):
        return None
    return CompanyComparisonRequest(
        companies=tuple(zip(names, codes)),
        ordered=bool(_ASKS_ORDER.search(compact)),
    )


def requested_comparison(plan: Any) -> CompanyComparisonRequest | None:
    """What query understanding recorded, read without re-parsing the question."""

    value = dict(getattr(plan, "evidence", None) or {}).get(COMPANY_COMPARISON_KEY)
    if not isinstance(value, Mapping):
        return None
    pairs = value.get("companies")
    if not isinstance(pairs, Sequence) or len(pairs) < 2:
        return None
    try:
        companies = tuple((str(pair[0]), str(pair[1])) for pair in pairs)
    except (IndexError, TypeError):
        return None
    return CompanyComparisonRequest(
        companies=companies,
        field=str(value.get("field") or CONTRACT_AMOUNT),
        ordered=bool(value.get("ordered")),
    )


def execute_company_scopes(
    request: CompanyComparisonRequest,
    plan: Any,
    execute: Callable[[Any], Any],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Retrieve once per company, against a plan narrowed to that company alone.

    Narrowing is the point.  One retrieval over all the companies at once ranks
    them against each other and can return several documents for one issuer and
    none for another; asked separately, each company either produces its own
    evidence or produces none, and the ranking refuses rather than filling the
    gap from a neighbour.
    """

    from dataclasses import replace as _replace

    per_company: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for _name, code in request.companies:
        scoped = _replace(plan, companies=(_name,), corp_codes=(code,))
        try:
            execution = execute(scoped)
        except Exception:  # noqa: BLE001 - one company's failure is not an answer
            per_company[code] = ()
            continue
        per_company[code] = tuple(getattr(execution, "chunks", ()) or ())
    return per_company


def resolve_comparison(
    request: CompanyComparisonRequest,
    per_company: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Ranking | None:
    """Rank the companies, each from its own retrieval, or decline.

    A company's operand may only be satisfied from that company's own chunks,
    which is enforced here by resolving each scope against its own slice rather
    than against a merged pool.
    """

    operands = []
    for scope in request.scopes():
        chunks = list(per_company.get(str(scope.corp_code), ()) or ())
        resolved = resolve_operands([scope], amount_sources(chunks))
        operands.append(resolved[0])
    return ranking(operands)


def compose_comparison_text(
    request: CompanyComparisonRequest, order: Ranking
) -> str:
    """State the order or the winner, naming every company's own amount."""

    lines: list[str] = []
    if request.ordered or order and len(order.operands) > 2:
        lines.append(
            "계약금액이 큰 순서: "
            + " > ".join(operand.scope.label for operand in order.operands)
        )
    else:
        largest = order.largest
        lines.append(
            f"계약금액이 더 큰 쪽: {largest.scope.label} "
            f"({int(largest.value):,}원)"
        )
        if order.spread is not None:
            lines.append(f"차이: {order.spread:,}원")
    for operand in order.operands:
        lines.append(f"- {operand.scope.label}: {int(operand.value):,}원")
    return "\n".join(lines)


__all__ = [
    "COMPANY_COMPARISON_KEY",
    "CompanyComparisonRequest",
    "comparison_requested",
    "compose_comparison_text",
    "execute_company_scopes",
    "requested_comparison",
    "resolve_comparison",
]
