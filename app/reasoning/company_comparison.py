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
from app.reasoning.query_plan import DateBasis, QueryPeriod
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
#: The field term appended to each operand's clause when it is retrieved.
_FIELD_TERM = "계약금액"


#: The date a per-company hint carries, written either way this corpus writes it.
_ISO_DATE = re.compile(r"(?P<y>\d{4})\s*[-./년]\s*(?P<m>\d{1,2})\s*[-./월]\s*(?P<d>\d{1,2})\s*일?")
#: Particles a clause opens with once the company name in front of it is removed.
_LEADING_PARTICLE = re.compile(r"^\s*(?:의|가|이|은|는|와|과|도|에서|에|,)\s*")
#: Connectors and the comparison tail a clause must not carry into retrieval.
#: The tail is the question *about* the operands, not part of any one of them.
_TRAILING_NOISE = re.compile(
    r"\s*(?:와|과|랑|,|및|그리고)?\s*"
    r"(?:각각\s*)?(?:공시한\s*)?(?:중\s*)?"
    r"(?:계약\s*금액.*)?$"
)
#: A trailing connector left once the tail is gone.
_TRAILING_CONNECTOR = re.compile(r"[\s,]*(?:와|과|랑|및|그리고|중)?\s*$")


@dataclass(frozen=True)
class CompanyOperand:
    """One company's side of a comparison: who, what filing, and when."""

    name: str
    corp_code: str
    #: The words in the question that describe *this* company's contract. Used
    #: as the lexical query of that company's own retrieval, so one operand's
    #: wording can never rank another operand's documents.
    clause: str | None = None
    #: The receipt date this company's hint names, ``YYYY-MM-DD``. Absent when
    #: the question dates no filing for it.
    on_date: str | None = None

    @property
    def pair(self) -> tuple[str, str]:
        return (self.name, self.corp_code)

    def scope(self, field: str) -> OperandScope:
        return OperandScope(
            role=ROLE_MEMBER,
            corp_code=self.corp_code,
            company=self.name,
            on_date=self.on_date,
            field=field,
        )

    def lexical_query(self, field_term: str = "계약금액") -> str:
        """This company's retrieval text: its own clause, then the field."""

        clause = (self.clause or "").strip()
        return f"{clause} {field_term}".strip() if clause else field_term

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "corp_code": self.corp_code,
            "clause": self.clause,
            "on_date": self.on_date,
        }


@dataclass(frozen=True)
class CompanyComparisonRequest:
    """One operand per company, plus how the answer should read."""

    companies: tuple[CompanyOperand, ...]
    field: str = CONTRACT_AMOUNT
    ordered: bool = False

    @property
    def size(self) -> int:
        return len(self.companies)

    @property
    def pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(operand.pair for operand in self.companies)

    def scopes(self) -> tuple[OperandScope, ...]:
        return tuple(operand.scope(self.field) for operand in self.companies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "companies": [operand.to_dict() for operand in self.companies],
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
    operands = company_operands(str(query or ""), tuple(zip(names, codes)))
    if operands is None:
        return None
    return CompanyComparisonRequest(
        companies=operands, ordered=bool(_ASKS_ORDER.search(compact))
    )


def company_operands(
    query: str, resolved: Sequence[tuple[str, str]]
) -> tuple[CompanyOperand, ...] | None:
    """Bind each company to the words and the date that describe *its* filing.

    Order comes from the question, never from the order the companies were
    resolved in: the resolver sorts them, and a comparison whose operands are
    bound in sorted order attaches one company's contract to another's name.

    Where the question spells the operands out in a trailing list -- a
    parenthetical naming each company with its date and its contract -- that
    list is the binding, because it is the only place the question says which
    filing it means for each company.  Otherwise each company's clause is the
    text between its own mention and the next one.

    Every company must come away with a position; one that cannot be located
    cannot be bound to a clause, and a comparison with an unbound operand is
    the guess this refuses to make.
    """

    text = str(query or "")
    if not text or len(resolved) < 2:
        return None
    spelled = _spelled_out_operands(text, resolved)
    if spelled is not None:
        return spelled
    return _sentence_operands(text, resolved)


def _spelled_out_operands(
    text: str, resolved: Sequence[tuple[str, str]]
) -> tuple[CompanyOperand, ...] | None:
    """Operands read from a trailing list that names each company's filing."""

    for opening in range(len(text)):
        if text[opening] != "(":
            continue
        closing = text.find(")", opening)
        if closing < 0:
            continue
        inside = text[opening + 1 : closing]
        bound = _bind_listed_items(inside, resolved)
        if bound is not None:
            return bound
    return None


def _bind_listed_items(
    inside: str, resolved: Sequence[tuple[str, str]]
) -> tuple[CompanyOperand, ...] | None:
    """Match every company to exactly one comma-separated item of a list."""

    items = [part.strip() for part in inside.split(",") if part.strip()]
    if len(items) < len(resolved):
        return None
    taken: set[int] = set()
    placed: list[tuple[int, CompanyOperand]] = []
    for name, code in resolved:
        index = _item_for_company(items, name, taken)
        if index is None:
            return None
        taken.add(index)
        on_date, remainder = _split_date(_strip_company(items[index], name))
        placed.append(
            (
                index,
                CompanyOperand(
                    name=name, corp_code=code, clause=remainder or None, on_date=on_date
                ),
            )
        )
    # The list's own order is the order the question puts its operands in, and
    # each operand keeps the index of the item it was bound to.
    placed.sort(key=lambda entry: entry[0])
    return tuple(operand for _index, operand in placed)


def _item_for_company(
    items: Sequence[str], name: str, taken: set[int]
) -> int | None:
    """The list item naming this company, by the longest agreement in either
    direction: a list may abbreviate a long name, and a name may extend a short
    item.  Ambiguity between two untaken items declines rather than guessing.
    """

    matches = [
        index
        for index, item in enumerate(items)
        if index not in taken and _names_company(item, name)
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def _names_company(item: str, name: str) -> bool:
    head = item.split()[0] if item.split() else ""
    compact_item = re.sub(r"\s+", "", item)
    compact_name = re.sub(r"\s+", "", name)
    if compact_name and compact_name in compact_item:
        return True
    return bool(head) and len(head) >= 2 and compact_name.startswith(re.sub(r"\s+", "", head))


def _strip_company(item: str, name: str) -> str:
    compact_name = re.sub(r"\s+", "", name)
    head = item.split()[0] if item.split() else ""
    for prefix in (name, compact_name, head):
        if prefix and item.startswith(prefix):
            return item[len(prefix) :].strip()
    return item.strip()


def _sentence_operands(
    text: str, resolved: Sequence[tuple[str, str]]
) -> tuple[CompanyOperand, ...] | None:
    """Each company's clause is the text from its own mention to the next."""

    located: list[tuple[int, int, str, str]] = []
    for name, code in resolved:
        start = text.find(name)
        if start < 0:
            return None
        located.append((start, start + len(name), name, code))
    located.sort()
    operands: list[CompanyOperand] = []
    for index, (_start, end, name, code) in enumerate(located):
        stop = located[index + 1][0] if index + 1 < len(located) else len(text)
        clause = _clean_clause(text[end:stop])
        on_date, clause = _split_date(clause)
        operands.append(
            CompanyOperand(
                name=name, corp_code=code, clause=clause or None, on_date=on_date
            )
        )
    return tuple(operands)


def _clean_clause(segment: str) -> str:
    """One company's own words, without the question asked about all of them."""

    clause = _LEADING_PARTICLE.sub("", segment.strip())
    clause = _TRAILING_NOISE.sub("", clause).strip()
    clause = _TRAILING_CONNECTOR.sub("", clause).strip()
    return clause


def _split_date(segment: str) -> tuple[str | None, str]:
    """Lift a receipt date out of a clause, leaving the description behind."""

    match = _ISO_DATE.search(segment)
    if match is None:
        return None, segment.strip()
    on_date = "%04d-%02d-%02d" % (
        int(match.group("y")), int(match.group("m")), int(match.group("d"))
    )
    remainder = (segment[: match.start()] + " " + segment[match.end() :]).strip()
    return on_date, re.sub(r"\s{2,}", " ", remainder)


def requested_comparison(plan: Any) -> CompanyComparisonRequest | None:
    """What query understanding recorded, read without re-parsing the question."""

    value = dict(getattr(plan, "evidence", None) or {}).get(COMPANY_COMPARISON_KEY)
    if not isinstance(value, Mapping):
        return None
    pairs = value.get("companies")
    if not isinstance(pairs, Sequence) or len(pairs) < 2:
        return None
    try:
        companies = tuple(_operand_from_dict(entry) for entry in pairs)
    except (IndexError, KeyError, TypeError):
        return None
    if any(operand is None for operand in companies):
        return None
    return CompanyComparisonRequest(
        companies=companies,
        field=str(value.get("field") or CONTRACT_AMOUNT),
        ordered=bool(value.get("ordered")),
    )


def _operand_from_dict(entry: Any) -> CompanyOperand | None:
    if isinstance(entry, Mapping):
        name = str(entry.get("name") or "")
        code = str(entry.get("corp_code") or "")
        if not name or not code:
            return None
        clause = entry.get("clause")
        on_date = entry.get("on_date")
        return CompanyOperand(
            name=name,
            corp_code=code,
            clause=str(clause) if clause else None,
            on_date=str(on_date) if on_date else None,
        )
    # A plain pair, from a caller that carried no clause or date.
    return CompanyOperand(name=str(entry[0]), corp_code=str(entry[1]))


def operand_subplan(plan: Any, operand: CompanyOperand, *, field_term: str = "계약금액"):
    """One company's own plan: its code, its words, and only its own date.

    The parent's period is deliberately not inherited.  A question that dates
    four filings differently still resolves to one global period upstream, and
    handing that period to every company would ask three of them for a filing
    on a day they filed nothing.  An operand with no date of its own gets an
    empty period rather than somebody else's.
    """

    from dataclasses import replace as _replace

    period = QueryPeriod()
    basis = DateBasis.UNSPECIFIED
    if operand.on_date:
        year = int(operand.on_date[:4])
        period = QueryPeriod(
            year=year,
            from_date=operand.on_date,
            to_date=operand.on_date,
            period_type="receipt_date",
        )
        basis = DateBasis.RECEIPT_DATE
    return _replace(
        plan,
        query=operand.lexical_query(field_term),
        companies=(operand.name,),
        corp_codes=(operand.corp_code,),
        years=(),
        period=period,
        date_basis=basis,
        comparison=None,
    )


def executable_comparison(plan: Any) -> "CompanyComparisonRequest | None":
    """The comparison this plan may actually be executed as, or nothing.

    Completeness is the whole permission.  A question naming several companies
    is not executable because it names them; it is executable because every one
    of them resolved to its own code, carries its own operand, and asks for the
    one field this lane compares.  Anything short of that keeps the ambiguity it
    has today, which is what protects an issuer/reporter pair from being read as
    two operands.

    The company set is checked against the plan rather than trusted from the
    request, so a request that survived from an earlier, differently-resolved
    plan cannot authorise execution of this one.
    """

    request = requested_comparison(plan)
    if request is None or request.size < 2:
        return None
    if request.field != CONTRACT_AMOUNT:
        return None
    if any(not operand.name or not operand.corp_code for operand in request.companies):
        return None
    codes = [operand.corp_code for operand in request.companies]
    if len(set(codes)) != len(codes):
        return None
    if set(request.pairs) != set(
        zip(tuple(getattr(plan, "companies", ())), tuple(getattr(plan, "corp_codes", ())))
    ):
        return None
    return request


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

    per_company: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for operand in request.companies:
        try:
            execution = execute(operand_subplan(plan, operand, field_term=_FIELD_TERM))
        except Exception:  # noqa: BLE001 - one company's failure is not an answer
            per_company[operand.corp_code] = ()
            continue
        per_company[operand.corp_code] = tuple(
            getattr(execution, "chunks", ()) or ()
        )
    return per_company


def executions_for(
    request: CompanyComparisonRequest, plan: Any, execute: Callable[[Any], Any]
) -> dict[str, Any]:
    """Each company's whole execution, kept apart, for callers that need more
    than its chunks -- documents and ranked results, so one merged execution
    can carry every operand's evidence into the ordinary answer path."""

    executions: dict[str, Any] = {}
    for operand in request.companies:
        try:
            executions[operand.corp_code] = execute(
                operand_subplan(plan, operand, field_term=_FIELD_TERM)
            )
        except Exception:  # noqa: BLE001 - one company's failure is not an answer
            executions[operand.corp_code] = None
    return executions


def chunks_by_company(
    request: CompanyComparisonRequest, chunks: Sequence[Mapping[str, Any]]
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    """Split served chunks back into per-company slices by their own corp_code.

    Grouping on the document's own code rather than on where it came from means
    a merged evidence set is still resolved company by company, so isolation
    survives the merge.
    """

    wanted = {operand.corp_code for operand in request.companies}
    slices: dict[str, list[Mapping[str, Any]]] = {code: [] for code in wanted}
    for chunk in chunks:
        code = str(chunk.get("corp_code") or "") if hasattr(chunk, "get") else ""
        if code in slices:
            slices[code].append(chunk)
    return {code: tuple(items) for code, items in slices.items()}


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
    "CompanyOperand",
    "chunks_by_company",
    "executable_comparison",
    "company_operands",
    "executions_for",
    "operand_subplan",
    "comparison_requested",
    "compose_comparison_text",
    "execute_company_scopes",
    "requested_comparison",
    "resolve_comparison",
]
