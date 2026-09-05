"""Fail-closed amount ranking for side-by-side company evidence.

``comparison_evidence`` deliberately retrieves each named company without
claiming that the resulting figures are comparable.  This module adds the one
safe promotion from that evidence-only answer to a ranking: every company must
provide a total amount from the same filing kind and the same fiscal year.

The filing kind comes from ``report_nm``.  Words such as ``당기`` in disclosure
text are relative to the filing and therefore cannot distinguish a full year
from a half year.  Amounts do come from the text, but only when ``총`` or
``누적`` identifies exactly one monetary expression.  Any uncertainty returns
``None``; callers then serve the original side-by-side execution unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping, Sequence

from app.retrieval.interfaces import RetrievalResult


COMPARISON_RANKING_KEY = "conditional_comparison_ranking"

ANNUAL = "annual"
HALF_YEAR = "half_year"
QUARTERLY = "quarterly"
_REPORT_KINDS = {ANNUAL, HALF_YEAR, QUARTERLY}
_REPORT_LABELS = {
    ANNUAL: "사업보고서",
    HALF_YEAR: "반기보고서",
    QUARTERLY: "분기보고서",
}

_TOTAL_MARKER = re.compile(r"총|누적")
_CLAUSE_BOUNDARY = re.compile(r"[.!?。;\n\r]")
_AMOUNT_SUBJECT = re.compile(
    r"금액|규모|매출|매출액|수익|이익|손실|비용|연구개발비|투자|자산|부채|자본|현금"
)
_ASCENDING_COMPARISON = re.compile(
    r"더\s*(?:작|적|낮)|가장\s*(?:작|적|낮)|(?:작|적|낮)은\s*순서|오름차순"
)
_TEXT_YEAR = re.compile(r"(?<!\d)(20\d{2})\s*년")
_NUMBER_PATTERN = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_AMOUNT_EXPRESSION = re.compile(
    rf"(?<![\d,.])(?P<amount>"
    rf"{_NUMBER_PATTERN}\s*[조억만천]"
    rf"(?:(?:\s*{_NUMBER_PATTERN})?\s*[조억만천])*"
    rf"(?:\s*{_NUMBER_PATTERN})?\s*원)"
)
_AMOUNT_TOKEN = re.compile(
    rf"(?P<number>{_NUMBER_PATTERN})|(?P<unit>[조억만천])"
)
_LARGE_UNITS = {
    "조": Decimal(1_000_000_000_000),
    "억": Decimal(100_000_000),
    "만": Decimal(10_000),
}


@dataclass(frozen=True)
class ParsedAmount:
    """One unambiguous total-marked amount found in evidence text."""

    value: int
    text: str
    #: The clause the amount was read from.  The subject that gives the amount
    #: its meaning has to be in this, not merely somewhere in the chunk.
    clause: str = ""


@dataclass(frozen=True)
class RankingAmount:
    """One company's amount, still bound to the filing that stated it."""

    label: str
    corp_code: str
    value: int
    amount_text: str
    report_kind: str
    report_nm: str
    base_year: int
    chunk_id: str
    doc_id: str
    total_marker: bool = True
    #: The question's own subject words this filing states.  Ranking compares
    #: amounts, and an amount means nothing without what it counts.
    subjects: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "corp_code": self.corp_code,
            "value": self.value,
            "amount_text": self.amount_text,
            "report_kind": self.report_kind,
            "report_nm": self.report_nm,
            "base_year": self.base_year,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "total_marker": self.total_marker,
        }


@dataclass(frozen=True)
class ConditionalRanking:
    """A descending order whose four comparison gates all passed."""

    operands: tuple[RankingAmount, ...]
    report_kind: str
    base_year: int
    direction: str = "descending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": True,
            "report_kind": self.report_kind,
            "base_year": self.base_year,
            "direction": self.direction,
            "operands": [operand.to_dict() for operand in self.operands],
        }


def parse_total_amount(text: Any) -> ParsedAmount | None:
    """Parse exactly one amount carrying its own ``총``/``누적`` marker.

    A marker is associated only with the next monetary expression in the same
    clause and after the preceding monetary expression.  This is what prevents
    a leading total from blessing the later per-segment figures in a sentence
    containing several amounts.
    """

    source = str(text or "")
    if len(_TOTAL_MARKER.findall(source)) != 1:
        return None
    matches = list(_AMOUNT_EXPRESSION.finditer(source))
    if not matches:
        return None

    marked: list[ParsedAmount] = []
    previous_amount_end = 0
    for match in matches:
        preceding = source[: match.start()]
        boundaries = list(_CLAUSE_BOUNDARY.finditer(preceding))
        clause_start = boundaries[-1].end() if boundaries else 0
        marker_scope = source[max(clause_start, previous_amount_end) : match.start()]
        previous_amount_end = match.end()
        if not _TOTAL_MARKER.search(marker_scope):
            continue
        amount_text = re.sub(r"\s+", " ", match.group("amount")).strip()
        value = _parse_amount_expression(amount_text)
        if value is None:
            return None
        following = _CLAUSE_BOUNDARY.search(source, match.end())
        clause = source[clause_start : following.start() if following else len(source)]
        marked.append(
            ParsedAmount(value=value, text=amount_text, clause=clause)
        )

    # Several totals in one chunk can describe several subjects or periods.
    # Retrieval rank cannot tell which one the question meant, so fail closed.
    return marked[0] if len(marked) == 1 else None


def conditional_ranking(
    comparison: Any,
    plan: Any,
    executions: Mapping[str, Any],
) -> ConditionalRanking | None:
    """Build a ranking only when every company passes every comparison gate."""

    companies = tuple(getattr(comparison, "companies", ()) or ())
    if len(companies) < 2 or not _amount_comparison_requested(plan):
        return None

    report_kind = _requested_report_kind(plan)
    requested_year = _requested_year(plan)
    if report_kind is None or requested_year is False:
        return None

    selected: list[RankingAmount] = []
    for operand in companies:
        found = _select_company_amount(
            operand,
            executions.get(str(getattr(operand, "corp_code", "") or "")),
            report_kind=report_kind,
            subjects=_subject_terms(plan),
        )
        if found is None:
            return None
        selected.append(found)

    years = {operand.base_year for operand in selected}
    kinds = {operand.report_kind for operand in selected}
    if len(years) != 1 or kinds != {report_kind}:
        return None
    base_year = next(iter(years))
    if isinstance(requested_year, int) and base_year != requested_year:
        return None
    if len({operand.corp_code for operand in selected}) != len(selected):
        return None
    if not all(operand.total_marker for operand in selected):
        return None
    # Every company's amount must be about the same thing, and that thing must
    # be one the question named. Without this, "설비투자가 더 큰 곳" ranks a
    # 총 매출액 against a 총 부채 and calls the larger one the answer: each
    # figure passes the company, year, filing-kind and total gates on its own,
    # and nothing above asks what the figure counts.
    if not _shared_subject(selected):
        return None

    direction = _requested_direction(plan)
    # Python's sort is stable, so ties retain the question's company order.
    ordered = tuple(
        sorted(
            selected,
            key=(
                (lambda operand: operand.value)
                if direction == "ascending"
                else (lambda operand: -operand.value)
            ),
        )
    )
    return ConditionalRanking(
        operands=ordered,
        report_kind=report_kind,
        base_year=base_year,
        direction=direction,
    )


def apply_conditional_ranking(
    execution: Any, ranking: ConditionalRanking | None
) -> Any:
    """Attach and expose a proven ranking, or return ``execution`` unchanged.

    Proven operands lead the result list so every citation used by the ranking
    is inside the public Top-K.  This reorder happens only after all gates pass;
    a declined ranking returns the exact stage-one object, which is the safety
    property this layer is built around.
    """

    if ranking is None:
        return execution

    results = list(getattr(execution, "results", ()) or ())
    by_id = {str(getattr(result, "chunk_id", "") or ""): result for result in results}
    leading_ids = [operand.chunk_id for operand in ranking.operands]
    if len(set(leading_ids)) != len(leading_ids) or any(
        chunk_id not in by_id for chunk_id in leading_ids
    ):
        return execution

    ordered_results = [by_id[chunk_id] for chunk_id in leading_ids]
    ordered_results.extend(
        result
        for result in results
        if str(getattr(result, "chunk_id", "") or "") not in set(leading_ids)
    )
    reranked = tuple(
        RetrievalResult(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            bm25_score=result.bm25_score,
            rank=index,
            metadata_match=result.metadata_match,
        )
        for index, result in enumerate(ordered_results, start=1)
    )

    chunks = list(getattr(execution, "chunks", ()) or ())
    chunks_by_id = {_chunk_id(chunk): chunk for chunk in chunks if _chunk_id(chunk)}
    ordered_chunks = [
        chunks_by_id[result.chunk_id]
        for result in reranked
        if result.chunk_id in chunks_by_id
    ]
    result_ids = {result.chunk_id for result in reranked}
    ordered_chunks.extend(
        chunk for chunk in chunks if _chunk_id(chunk) not in result_ids
    )

    routing = dict(getattr(execution, "routing", None) or {})
    routing[COMPARISON_RANKING_KEY] = ranking.to_dict()
    return replace(
        execution,
        chunks=tuple(ordered_chunks),
        results=reranked,
        routing=routing,
    )


def ranking_from_outcome(outcome: Any) -> ConditionalRanking | None:
    """Rebuild and revalidate the internal routing payload downstream."""

    if not isinstance(outcome, Mapping) or outcome.get("applied") is not True:
        return None
    report_kind = str(outcome.get("report_kind") or "")
    base_year = _year(outcome.get("base_year"))
    direction = str(outcome.get("direction") or "descending")
    entries = outcome.get("operands")
    if (
        report_kind not in _REPORT_KINDS
        or base_year is None
        or direction not in {"ascending", "descending"}
        or not isinstance(entries, Sequence)
        or isinstance(entries, (str, bytes))
        or len(entries) < 2
    ):
        return None

    operands: list[RankingAmount] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            return None
        value = entry.get("value")
        amount_text = str(entry.get("amount_text") or "").strip()
        entry_year = _year(entry.get("base_year"))
        entry_kind = str(entry.get("report_kind") or "")
        report_nm = str(entry.get("report_nm") or "").strip()
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            or _parse_amount_expression(amount_text) != value
            or entry.get("total_marker") is not True
            or entry_year != base_year
            or entry_kind != report_kind
            or _report_kind(report_nm) != report_kind
        ):
            return None
        operand = RankingAmount(
            label=str(entry.get("label") or "").strip(),
            corp_code=str(entry.get("corp_code") or "").strip(),
            value=value,
            amount_text=amount_text,
            report_kind=entry_kind,
            report_nm=report_nm,
            base_year=entry_year,
            chunk_id=str(entry.get("chunk_id") or "").strip(),
            doc_id=str(entry.get("doc_id") or "").strip(),
            total_marker=True,
        )
        if not all(
            (operand.label, operand.corp_code, operand.chunk_id, operand.doc_id)
        ):
            return None
        operands.append(operand)

    if len({operand.corp_code for operand in operands}) != len(operands):
        return None
    if len({operand.chunk_id for operand in operands}) != len(operands):
        return None
    ordered = tuple(
        sorted(
            operands,
            key=(
                (lambda operand: operand.value)
                if direction == "ascending"
                else (lambda operand: -operand.value)
            ),
        )
    )
    return ConditionalRanking(
        operands=ordered,
        report_kind=report_kind,
        base_year=base_year,
        direction=direction,
    )


def compose_conditional_ranking_text(
    ranking: ConditionalRanking, citation_ids: Sequence[str]
) -> str:
    """State the winner/order and cite every operand used to establish it."""

    if len(citation_ids) != len(ranking.operands):
        raise ValueError("one citation is required for every ranked company")
    basis = f"{ranking.base_year}년 {_REPORT_LABELS[ranking.report_kind]} 기준"
    operands = ranking.operands
    if len(operands) == 2 and operands[0].value != operands[1].value:
        first, second = operands
        predicate = "작습니다" if ranking.direction == "ascending" else "큽니다"
        statement = (
            f"{basis}, {first.label}의 합계 금액은 {first.amount_text}으로 "
            f"{second.label}({second.amount_text})보다 {predicate}."
        )
    elif len(operands) == 2:
        first, second = operands
        statement = (
            f"{basis}, {first.label}와 {second.label}의 합계 금액은 "
            f"각각 {first.amount_text}으로 같습니다."
        )
    else:
        groups: list[str] = []
        for value in dict.fromkeys(operand.value for operand in operands):
            tied = [
                f"{operand.label}({operand.amount_text})"
                for operand in operands
                if operand.value == value
            ]
            groups.append(" = ".join(tied))
        order_label = "작은 순서" if ranking.direction == "ascending" else "큰 순서"
        separator = " < " if ranking.direction == "ascending" else " > "
        statement = f"{basis}, 합계 금액이 {order_label}: " + separator.join(groups) + "."
    return f"{statement} {' '.join(citation_ids)}"


#: Words that say how a comparison is phrased rather than what it is about.
_SUBJECT_NOISE = frozenset(
    """
    중 더 가장 제일 큰 작은 많은 적은 높은 낮은 어디 어느 무엇 누구 얼마 규모
    기업 회사 곳 쪽 순서 순위 비교 알려줘 알려 정리 확인 인가 인가요 입니까
    있나 있는지 해줘 그리고 대비 기준 년 년도 연도 분기 반기 상반기 하반기
    금액 총액 합계 누적 수치 실적 내역 현황 자료 공시 보고서 사업보고서
    반기보고서 분기보고서 기말 기준일 당기 전기
    """.split()
)


#: Particles that ride on the end of a Korean noun.  "설비투자가" in a question
#: and "설비투자에" in a filing are the same subject, and comparing the tokens
#: whole says they are not.
_PARTICLES = (
    "으로부터", "에서는", "에게서", "으로써", "으로서", "이라는", "라는",
    "에서", "에게", "으로", "부터", "까지", "보다", "처럼", "만큼", "이나",
    "와의", "과의", "의", "은", "는", "이", "가", "을", "를", "에", "로",
    "와", "과", "도", "만", "라", "야",
)


def _stem(token: str) -> str:
    """The token with one trailing particle removed, when one is there."""

    for particle in _PARTICLES:
        if token.endswith(particle) and len(token) - len(particle) >= 2:
            return token[: -len(particle)]
    return token


def _subject_terms(plan: Any) -> frozenset[str]:
    """The content words the question used to say what it is asking about."""

    query = str(getattr(plan, "raw_query", None) or getattr(plan, "query", None) or "")
    return frozenset(
        stem
        for token in re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9]+", query)
        if len(stem := _stem(token)) >= 2 and stem not in _SUBJECT_NOISE
    )


def _shared_subject(selected: Sequence[RankingAmount]) -> bool:
    """Whether every filing states at least one subject word all of them state.

    Deliberately strict. Structured metrics are not available here -- the
    amounts are read out of narrative sentences -- so the only evidence that
    two figures measure the same thing is that both filings say so in the
    asker's own words. Where they do not, the honest outcome is the evidence
    side by side without a winner, which is what returning nothing produces.
    """

    if not selected:
        return False
    common = frozenset.intersection(*(operand.subjects for operand in selected))
    return bool(common)


def _select_company_amount(
    operand: Any,
    execution: Any,
    *,
    report_kind: str,
    subjects: frozenset[str] = frozenset(),
) -> RankingAmount | None:
    if execution is None:
        return None
    wanted_code = str(getattr(operand, "corp_code", "") or "").strip()
    label = str(getattr(operand, "name", "") or "").strip()
    if not wanted_code or not label:
        return None

    candidates = {_chunk_id(chunk): _chunk_payload(chunk) for chunk in execution.chunks}
    for result in getattr(execution, "results", ()) or ():
        chunk_id = str(getattr(result, "chunk_id", "") or "")
        payload = candidates.get(chunk_id, {})
        report_nm = str(payload.get("report_nm") or "").strip()
        corp_code = str(
            payload.get("corp_code") or payload.get("company_id") or ""
        ).strip()
        base_year = _year(payload.get("base_year"))
        if (
            not chunk_id
            or corp_code != wanted_code
            or _report_kind(report_nm) != report_kind
        ):
            continue
        if base_year is None:
            return None
        evidence_text = str(
            payload.get("retrieval_text") or payload.get("content") or ""
        )
        mentioned_years = {int(value) for value in _TEXT_YEAR.findall(evidence_text)}
        if mentioned_years and mentioned_years != {base_year}:
            return None
        parsed = parse_total_amount(evidence_text)
        if parsed is None:
            return None
        doc_id = str(getattr(result, "doc_id", "") or payload.get("doc_id") or "")
        if not doc_id:
            continue
        return RankingAmount(
            label=label,
            corp_code=wanted_code,
            value=parsed.value,
            amount_text=parsed.text,
            report_kind=report_kind,
            report_nm=report_nm,
            base_year=base_year,
            chunk_id=chunk_id,
            doc_id=doc_id,
            # Read from the amount's own clause. "설비투자는 검토 중이다.
            # 총 매출액 10조원" names the subject and then states a different
            # quantity, and a chunk-wide search cannot tell those apart.
            subjects=frozenset(
                term for term in subjects if term in parsed.clause
            ),
        )
    return None


def _requested_report_kind(plan: Any) -> str | None:
    raw_query = re.sub(
        r"\s+", "", str(getattr(plan, "raw_query", None) or "")
    )
    explicit = {
        kind
        for term, kind in (
            ("사업보고서", ANNUAL),
            ("반기보고서", HALF_YEAR),
            ("분기보고서", QUARTERLY),
        )
        if term in raw_query
    }
    if len(explicit) > 1:
        return None
    if explicit:
        return next(iter(explicit))

    # A second-half amount cannot be read from one filing kind without doing
    # subtraction across reports, which this layer deliberately does not do.
    if "하반기" in raw_query:
        return None
    if "상반기" in raw_query:
        return HALF_YEAR

    period = getattr(plan, "period", None)
    quarter = getattr(period, "quarter", None)
    if quarter in (1, 3):
        return QUARTERLY
    if quarter == 2:
        return HALF_YEAR
    if quarter == 4:
        return ANNUAL

    # A year with no narrower reporting window means the full fiscal year.
    return ANNUAL if _requested_year(plan) is not None else None


def _amount_comparison_requested(plan: Any) -> bool:
    raw_query = str(getattr(plan, "raw_query", None) or "")
    return bool(_AMOUNT_SUBJECT.search(raw_query))


def _requested_direction(plan: Any) -> str:
    raw_query = str(getattr(plan, "raw_query", None) or "")
    return "ascending" if _ASCENDING_COMPARISON.search(raw_query) else "descending"


def _requested_year(plan: Any) -> int | bool | None:
    values: set[int] = set()
    candidates = (
        *tuple(getattr(plan, "years", ()) or ()),
        getattr(getattr(plan, "period", None), "year", None),
    )
    for candidate in candidates:
        normalized = _year(candidate)
        if normalized is not None:
            values.add(normalized)
    if len(values) > 1:
        return False
    return next(iter(values)) if values else None


def _report_kind(report_nm: Any) -> str | None:
    name = str(report_nm or "")
    if "사업보고서" in name:
        return ANNUAL
    if "반기보고서" in name:
        return HALF_YEAR
    if "분기보고서" in name:
        return QUARTERLY
    return None


def _parse_amount_expression(value: str) -> int | None:
    expression = str(value or "").strip()
    if not expression.endswith("원"):
        return None
    body = expression[:-1]
    tokens = list(_AMOUNT_TOKEN.finditer(body))
    if not tokens:
        return None
    position = 0
    for token in tokens:
        if body[position : token.start()].strip():
            return None
        position = token.end()
    if body[position:].strip():
        return None

    total = Decimal(0)
    section = Decimal(0)
    pending: Decimal | None = None
    previous_large = Decimal("Infinity")
    used_thousand = False
    saw_unit = False
    try:
        for token in tokens:
            number = token.group("number")
            if number is not None:
                if pending is not None:
                    return None
                pending = Decimal(number.replace(",", ""))
                continue

            unit = str(token.group("unit"))
            saw_unit = True
            if unit == "천":
                if used_thousand:
                    return None
                section += (pending if pending is not None else Decimal(1)) * 1_000
                pending = None
                used_thousand = True
                continue

            multiplier = _LARGE_UNITS[unit]
            if multiplier >= previous_large:
                return None
            coefficient = section + (pending if pending is not None else Decimal(0))
            if coefficient == 0:
                return None
            total += coefficient * multiplier
            section = Decimal(0)
            pending = None
            used_thousand = False
            previous_large = multiplier
    except InvalidOperation:
        return None

    total += section + (pending if pending is not None else Decimal(0))
    if not saw_unit or total < 0 or total != total.to_integral_value():
        return None
    return int(total)


def _chunk_payload(chunk: Any) -> Mapping[str, Any]:
    if isinstance(chunk, Mapping):
        return chunk
    payload = getattr(chunk, "chunk", None)
    if not isinstance(payload, Mapping):
        return {}
    hydrated = dict(payload)
    for name in ("chunk_id", "doc_id"):
        value = getattr(chunk, name, None)
        if value:
            hydrated.setdefault(name, str(value))
    return hydrated


def _chunk_id(chunk: Any) -> str:
    if isinstance(chunk, Mapping):
        return str(chunk.get("chunk_id") or "")
    return str(getattr(chunk, "chunk_id", "") or "")


def _year(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1900 <= year <= 2100 else None


__all__ = [
    "ANNUAL",
    "COMPARISON_RANKING_KEY",
    "ConditionalRanking",
    "HALF_YEAR",
    "ParsedAmount",
    "QUARTERLY",
    "RankingAmount",
    "apply_conditional_ranking",
    "compose_conditional_ranking_text",
    "conditional_ranking",
    "parse_total_amount",
    "ranking_from_outcome",
]
