"""Side-by-side evidence for a comparison no lane can rank.

The amount lane compares 계약금액 because that is the one field this corpus
states the same way, in the same unit, for every issuer. A question comparing
anything else -- 설비투자, 연구개발비, 매출 -- reaches validation with several
companies named, cannot resolve the single-company slot that retrieval requires,
and is asked back "어느 회사에 대한 공시를 확인할까요?" when the asker already
named two.

This first answers those questions without ranking them. Each company is
retrieved on its own plan, scoped to that company and the period the question
asked about, and the results are interleaved so every company's best evidence
sits inside the first few positions. What comes out is the ordinary evidence
path's input, so evidence building, citation alignment, the answerability guard
and the think trace all run exactly as they do for a single-company question.

A separate fail-closed layer may rank narrative amounts only after filing kind,
fiscal year, amount syntax and a total marker all agree for every company. Two
filings that merely state the same subject over different windows -- one
company's 당기, another's 상반기 -- still remain side by side without a winner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import zip_longest
from typing import Any, Mapping, Sequence

from app.reasoning.company_comparison import CompanyOperand
from app.reasoning.query_plan import QueryExecution
from app.retrieval.interfaces import RetrievalResult


#: A leg costs one retrieval, so the fan-out is bounded rather than served
#: slowly.
MAX_COMPARISON_COMPANIES = 5

DECLINE_NO_PLAN = "no_query_plan"
DECLINE_TOO_FEW = "fewer_than_two_companies"
DECLINE_TOO_MANY = "more_than_max_companies"
DECLINE_UNRESOLVED = "company_not_resolved_in_corpus"
DECLINE_NO_INTENT = "no_comparison_intent"
DECLINE_NO_SUBJECT = "no_shared_subject"

_COMPANY_COMPARISON = "company_comparison"
_CROSS_COMPANY = "cross_company"


@dataclass(frozen=True)
class EvidenceComparison:
    """The companies to retrieve separately, or why this was declined."""

    applied: bool
    companies: tuple[CompanyOperand, ...] = ()
    subject_kind: str | None = None
    decline_reason: str | None = None

    @property
    def company_count(self) -> int:
        return len(self.companies)

    def to_dict(self) -> dict[str, Any]:
        # Counts and statuses only, matching what the other planners expose.
        return {
            "applied": self.applied,
            "company_count": self.company_count,
            "subject_kind": self.subject_kind,
            "decline_reason": self.decline_reason,
        }

    @classmethod
    def declined(cls, reason: str) -> "EvidenceComparison":
        return cls(applied=False, decline_reason=reason)


def _comparison_intended(plan: Any) -> bool:
    """Whether the question relates the named companies to each other.

    Both upstream signals are read and either is enough. ``comparison`` is the
    payload query understanding builds for explicit 비교/대비 wording;
    ``comparison_frame`` is the firewall signal that also catches 중/더 큰
    constructions. Reading the frame here does not relax it -- it keeps blocking
    issuer/reporter reinterpretation exactly as before, and what this adds is
    permission to retrieve each company separately, never to merge their roles.
    """

    comparison = getattr(plan, "comparison", None)
    if isinstance(comparison, Mapping) and comparison.get("type") == _COMPANY_COMPARISON:
        return True
    evidence = getattr(plan, "evidence", None)
    return bool(
        isinstance(evidence, Mapping)
        and evidence.get("comparison_frame") == _CROSS_COMPANY
    )


def _period_is_populated(period: Any) -> bool:
    if period is None:
        return False
    if isinstance(period, (int, tuple)):
        return True
    return any(
        getattr(period, name, None) is not None
        for name in ("year", "quarter", "from_date", "to_date", "period_type")
    )


def _subject_kind(plan: Any) -> str | None:
    """What the companies are being compared on, read from the plan alone.

    Without a shared subject the legs would ask different questions and the
    answer would put unrelated evidence side by side, so a plan carrying none is
    declined rather than served as several unconnected lookups.
    """

    if getattr(plan, "metric", None):
        return "metric"
    if getattr(plan, "event_type", None):
        return "event_type"
    if _period_is_populated(getattr(plan, "period", None)):
        return "period"
    if getattr(plan, "years", ()):
        return "period"
    return None


def evidence_comparison(plan: Any) -> EvidenceComparison:
    """Recognize a comparison this module can serve as evidence, or decline."""

    if plan is None:
        return EvidenceComparison.declined(DECLINE_NO_PLAN)

    companies = tuple(getattr(plan, "companies", ()) or ())
    corp_codes = tuple(getattr(plan, "corp_codes", ()) or ())
    if len(companies) < 2:
        return EvidenceComparison.declined(DECLINE_TOO_FEW)
    if len(companies) > MAX_COMPARISON_COMPANIES:
        return EvidenceComparison.declined(DECLINE_TOO_MANY)
    if len(companies) != len(corp_codes) or not all(
        str(name).strip() and str(code).strip()
        for name, code in zip(companies, corp_codes)
    ):
        # Only companies the corpus resolved are retrieved. A category the
        # question named without listing its members never reaches here, because
        # expanding it would answer about companies nobody asked about.
        return EvidenceComparison.declined(DECLINE_UNRESOLVED)
    if not _comparison_intended(plan):
        return EvidenceComparison.declined(DECLINE_NO_INTENT)

    subject_kind = _subject_kind(plan)
    if subject_kind is None:
        return EvidenceComparison.declined(DECLINE_NO_SUBJECT)

    return EvidenceComparison(
        applied=True,
        companies=tuple(
            CompanyOperand(name=str(name), corp_code=str(code))
            for name, code in zip(companies, corp_codes)
        ),
        subject_kind=subject_kind,
    )


def evidence_subplan(plan: Any, operand: CompanyOperand) -> Any:
    """One company's own plan, scoped to the period the question asked about.

    This differs from the amount lane's subplan deliberately. There, each
    company carries its own filing date and inheriting a single global period
    would ask most of them about a day they filed nothing. Here the period *is*
    shared -- "2025년 설비투자" means 2025 for every company.

    Carrying the period is not enough on its own. Metadata filtering turns a
    period into a year condition only when the period is fiscal *and* ``years``
    is populated, and a question phrased "2025년에" resolves to a reference year
    with neither. The filter therefore never engages, and each company retrieves
    every year it ever filed -- which is how a question about 2025 comes back
    with 2023 and 2024 ranked above it. Naming the year here scopes the leg to
    ``base_year``, so a fiscal-2025 report filed in March 2026 is kept and the
    earlier years are not.

    A company that filed nothing for that year retrieves nothing and is simply
    absent from the answer. That is the honest outcome: showing its 2023 figure
    beside another company's 2025 one would compare different years without
    saying so.
    """

    period = getattr(plan, "period", None)
    years = tuple(getattr(plan, "years", ()) or ())
    year = getattr(period, "year", None)
    if year is not None and not years:
        years = (int(year),)
        period = replace(
            period,
            period_type=(
                "fiscal_quarter"
                if getattr(period, "quarter", None) is not None
                else "fiscal_year"
            ),
        )
    return replace(
        plan,
        company=operand.name,
        companies=(operand.name,),
        corp_code=operand.corp_code,
        corp_codes=(operand.corp_code,),
        years=years,
        period=period,
        comparison=None,
    )


def execute_per_company(
    comparison: EvidenceComparison, plan: Any, execute: Any
) -> dict[str, Any]:
    """Retrieve once per company, each against its own narrowed plan.

    One retrieval over every company at once ranks them against each other and
    returns several documents for whichever company matched the wording best.
    Asked separately, each company either produces its own evidence or produces
    none, and a gap stays a gap.
    """

    executions: dict[str, Any] = {}
    for operand in comparison.companies:
        try:
            executions[operand.corp_code] = execute(evidence_subplan(plan, operand))
        except Exception:  # noqa: BLE001 - one company's failure is not an answer
            executions[operand.corp_code] = None
    return executions


def _chunk_id(chunk: Any) -> str:
    value = getattr(chunk, "chunk_id", None)
    if value is None and hasattr(chunk, "get"):
        value = chunk.get("chunk_id")
    return str(value or "")


def merge_executions(
    plan: Any, comparison: EvidenceComparison, executions: Mapping[str, Any]
) -> QueryExecution:
    """Interleave the per-company executions into one the rest of the pipeline reads.

    Interleaving rather than concatenating is the whole point. Each retrieval
    returns its own top-N for its own company, so appending them puts the second
    company's best document past the evidence limit and produces an answer that
    cites the first company several times and the second not at all. Taking each
    company's best first keeps every company inside the first few positions, for
    any number of companies, and each company's internal order is untouched.
    """

    documents: list[Any] = []
    seen_documents: set[str] = set()
    candidates: dict[str, Any] = {}
    per_company_results: list[list[Any]] = []

    for operand in comparison.companies:
        execution = executions.get(operand.corp_code)
        if execution is None:
            per_company_results.append([])
            continue
        for document in getattr(execution, "documents", ()) or ():
            doc_id = str(getattr(document, "doc_id", "") or "")
            if doc_id and doc_id not in seen_documents:
                seen_documents.add(doc_id)
                documents.append(document)
        for chunk in getattr(execution, "chunks", ()) or ():
            chunk_id = _chunk_id(chunk)
            if chunk_id:
                # The candidate object itself, not a mapping built from it:
                # every stage after this reads CandidateChunk attributes.
                candidates.setdefault(chunk_id, chunk)
        per_company_results.append(list(getattr(execution, "results", ()) or ()))

    ordered: list[Any] = []
    seen_results: set[str] = set()
    for row in zip_longest(*per_company_results):
        for result in row:
            if result is None:
                continue
            chunk_id = str(getattr(result, "chunk_id", "") or "")
            if chunk_id and chunk_id not in seen_results:
                seen_results.add(chunk_id)
                ordered.append(result)

    # Ranks came from separate retrievals and would otherwise collide.
    ranked = tuple(
        RetrievalResult(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            bm25_score=result.bm25_score,
            rank=index + 1,
            metadata_match=result.metadata_match,
        )
        for index, result in enumerate(ordered)
    )
    chunks: list[Any] = [
        candidates[result.chunk_id]
        for result in ranked
        if result.chunk_id in candidates
    ]
    chunks.extend(
        candidate
        for chunk_id, candidate in candidates.items()
        if chunk_id not in seen_results
    )
    return QueryExecution(
        plan=plan,
        documents=tuple(documents),
        chunks=tuple(chunks),
        results=ranked,
        routing={"comparison_evidence": comparison.to_dict()},
    )


def served_company_count(
    comparison: EvidenceComparison, execution: Any
) -> int:
    """How many of the named companies the merged evidence actually speaks for."""

    wanted = {operand.corp_code for operand in comparison.companies}
    served: set[str] = set()
    for chunk in getattr(execution, "chunks", ()) or ():
        payload = getattr(chunk, "chunk", None)
        code = ""
        if isinstance(payload, Mapping):
            code = str(payload.get("corp_code") or "")
        if not code and hasattr(chunk, "get"):
            code = str(chunk.get("corp_code") or "")
        if code in wanted:
            served.add(code)
    return len(served)
