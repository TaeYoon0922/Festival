"""Evidence sufficiency states and deterministic final-answer guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from app.reasoning.exchange_field_aggregate import exchange_aggregate_requested
from app.reasoning.multi_document_evidence import (
    LIFECYCLE_NO_MEMBERS,
    LIFECYCLE_NONE,
)


class AnswerabilityStatus(str, Enum):
    ANSWERABLE = "answerable"
    PARTIALLY_ANSWERABLE = "partially_answerable"
    NOT_FOUND = "not_found"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_SCOPE = "out_of_scope"


_CATEGORICAL_NEGATIVE_PATTERNS = (
    r"없습니다",
    r"존재하지\s*않습니다",
    r"해당\s*사항(?:은|이)?\s*없",
    r"확인되지\s*않았습니다",
)


@dataclass(frozen=True)
class AnswerabilityResult:
    status: AnswerabilityStatus
    evidence_count: int = 0
    citation_count: int = 0
    complete: bool | None = None
    confirmed_fields: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    reason: str | None = None
    relevant_to_request: bool | None = None

    @property
    def citable(self) -> bool:
        return self.citation_count > 0

    @property
    def answerable(self) -> bool:
        return self.status is AnswerabilityStatus.ANSWERABLE

    @property
    def model_answer_allowed(self) -> bool:
        return self.status is AnswerabilityStatus.ANSWERABLE

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "evidence_count": self.evidence_count,
            "citation_count": self.citation_count,
            "complete": self.complete,
            "confirmed_fields": list(self.confirmed_fields),
            "missing_fields": list(self.missing_fields),
            "reason": self.reason,
            "citable": self.citable,
            "relevant_to_request": self.relevant_to_request,
            "answerable": self.answerable,
        }


class AnswerabilityGuard:
    """Classify answerability from deterministic resolver/public P0 state."""

    def evaluate(
        self,
        generated: Any,
        *,
        plan: Any = None,
        agent_result: Any = None,
        execution: Any = None,
        multi_document: Any = None,
    ) -> AnswerabilityResult:
        evidence_count = len(list(getattr(execution, "results", ()) or ()))
        citation_count = len(tuple(getattr(generated, "citations", ()) or ()))

        facts = getattr(multi_document, "facts", None)
        plan_payload = (
            plan.to_dict()
            if hasattr(plan, "to_dict")
            else (plan if isinstance(plan, Mapping) else None)
        )
        if facts is None and exchange_aggregate_requested(plan_payload):
            draft = getattr(agent_result, "answer_draft", None)
            draft_warnings = tuple(getattr(draft, "warnings", ()) or ())
            if "exchange_aggregate_from_retrieval_fallback" in draft_warnings:
                return AnswerabilityResult(
                    AnswerabilityStatus.ANSWERABLE,
                    evidence_count,
                    citation_count,
                    complete=True,
                    reason="retrieval_exchange_aggregate",
                )

        if facts is not None:
            complete = bool(getattr(facts, "complete", False))
            unresolved = int(getattr(facts, "unresolved_count", 0) or 0)
            logical_count = int(getattr(facts, "logical_count", 0) or 0)
            lifecycle = getattr(facts, "lifecycle_answer", None)
            aggregate = getattr(facts, "aggregate", None)
            if lifecycle is None and exchange_aggregate_requested(plan_payload):
                if aggregate is not None or logical_count > 0:
                    return AnswerabilityResult(
                        AnswerabilityStatus.ANSWERABLE,
                        evidence_count,
                        citation_count,
                        complete=complete,
                        reason=(
                            "complete_exchange_aggregate"
                            if aggregate is not None
                            else "complete_exchange_enumeration"
                        ),
                    )
                if complete and logical_count == 0:
                    return AnswerabilityResult(
                        AnswerabilityStatus.NOT_FOUND,
                        evidence_count,
                        citation_count,
                        complete=True,
                        reason="complete_empty_set",
                    )
            if unresolved > 0 or not complete:
                return AnswerabilityResult(
                    AnswerabilityStatus.INSUFFICIENT_EVIDENCE,
                    evidence_count,
                    citation_count,
                    complete=False,
                    reason="incomplete_multi_document_set",
                )
            if logical_count == 0 or lifecycle in {LIFECYCLE_NONE, LIFECYCLE_NO_MEMBERS}:
                return AnswerabilityResult(
                    AnswerabilityStatus.NOT_FOUND,
                    evidence_count,
                    citation_count,
                    complete=True,
                    reason=(
                        "complete_set_without_matching_lifecycle"
                        if lifecycle == LIFECYCLE_NONE
                        else "complete_empty_set"
                    ),
                )
            if citation_count == 0:
                return AnswerabilityResult(
                    AnswerabilityStatus.INSUFFICIENT_EVIDENCE,
                    evidence_count,
                    citation_count,
                    complete=True,
                    reason="citation_capable_evidence_missing",
                )
            return AnswerabilityResult(
                AnswerabilityStatus.ANSWERABLE,
                evidence_count,
                citation_count,
                complete=True,
                reason="complete_multi_document_evidence",
            )

        semantic_relevance = _semantic_evidence_relevance(
            plan, generated=generated, execution=execution
        )
        if semantic_relevance is False:
            return AnswerabilityResult(
                AnswerabilityStatus.INSUFFICIENT_EVIDENCE,
                evidence_count,
                citation_count,
                complete=False,
                reason="evidence_semantic_mismatch",
                relevant_to_request=False,
            )

        confirmed, missing = _field_coverage(getattr(agent_result, "resolution", None))
        if not confirmed and not missing:
            requested = _requested_evidence_fields(plan)
            if requested:
                answer_text = _compact(getattr(generated, "answer_text", ""))
                confirmed = tuple(
                    field
                    for field, evidence_aliases in requested
                    if any(_compact(alias) in answer_text for alias in evidence_aliases)
                )
                confirmed_set = set(confirmed)
                missing = tuple(
                    field for field, _aliases in requested if field not in confirmed_set
                )
                if missing:
                    return AnswerabilityResult(
                        (
                            AnswerabilityStatus.PARTIALLY_ANSWERABLE
                            if confirmed and citation_count > 0
                            else AnswerabilityStatus.INSUFFICIENT_EVIDENCE
                        ),
                        evidence_count,
                        citation_count,
                        complete=False,
                        confirmed_fields=confirmed,
                        missing_fields=missing,
                        reason=(
                            "some_requested_fields_are_supported"
                            if confirmed and citation_count > 0
                            else "requested_field_evidence_missing"
                        ),
                        relevant_to_request=semantic_relevance,
                    )
        if bool(getattr(generated, "answerable", False)) and citation_count > 0:
            return AnswerabilityResult(
                AnswerabilityStatus.ANSWERABLE,
                evidence_count,
                citation_count,
                complete=True,
                confirmed_fields=confirmed,
                reason="requested_evidence_is_citable",
                relevant_to_request=semantic_relevance,
            )
        if citation_count > 0 and confirmed:
            return AnswerabilityResult(
                AnswerabilityStatus.PARTIALLY_ANSWERABLE,
                evidence_count,
                citation_count,
                complete=False,
                confirmed_fields=confirmed,
                missing_fields=missing,
                reason="some_requested_fields_are_supported",
                relevant_to_request=semantic_relevance,
            )
        return AnswerabilityResult(
            AnswerabilityStatus.INSUFFICIENT_EVIDENCE,
            evidence_count,
            citation_count,
            complete=False,
            confirmed_fields=confirmed,
            missing_fields=missing,
            reason=(
                "citation_capable_evidence_missing"
                if citation_count == 0
                else "requested_field_evidence_missing"
            ),
            relevant_to_request=semantic_relevance,
        )


def guarded_answer_text(
    result: AnswerabilityResult,
    deterministic_answer: str,
    *,
    multi_document: Any = None,
) -> str:
    """Render non-answerable states without letting prose overstate evidence."""

    status = result.status
    if status is AnswerabilityStatus.ANSWERABLE:
        return deterministic_answer
    if status is AnswerabilityStatus.NOT_FOUND:
        facts = getattr(multi_document, "facts", None)
        lifecycle = getattr(facts, "lifecycle_answer", None)
        if lifecycle == LIFECYCLE_NONE:
            return "제공된 공시 기준으로 확인한 이후 해지 계약은 없습니다."
        if lifecycle == LIFECYCLE_NO_MEMBERS:
            return "제공된 공시 기준으로 확인한 해당 기간의 조건에 맞는 계약은 없습니다."
        return "제공된 공시 기준으로 확인한 해당 항목은 없습니다."
    if status is AnswerabilityStatus.OUT_OF_SCOPE:
        return "현재 제공된 공시 데이터 범위에서는 해당 내용을 확인할 수 없습니다."
    if status is AnswerabilityStatus.PARTIALLY_ANSWERABLE:
        base = deterministic_answer.strip()
        limitation = "다만 제공된 공시 근거에서 일부 요청 항목은 확인하기 어렵습니다."
        output = f"{base}\n\n{limitation}" if base else limitation
        return _negative_firewall(output, status)
    return _negative_firewall(
        "현재 확보된 공시 근거만으로는 해당 내용을 확인하기 어렵습니다.",
        status,
    )


def contains_categorical_negative(text: str) -> bool:
    return any(re.search(pattern, str(text or "")) for pattern in _CATEGORICAL_NEGATIVE_PATTERNS)


def _negative_firewall(text: str, status: AnswerabilityStatus) -> str:
    if status is AnswerabilityStatus.NOT_FOUND or not contains_categorical_negative(text):
        return text
    return "현재 확보된 공시 근거만으로는 해당 내용을 확인하기 어렵습니다."


def _field_coverage(resolution: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if resolution is None:
        return (), ()
    requested = tuple(str(value) for value in getattr(resolution, "requested_fields", ()) or ())
    missing = tuple(str(value) for value in getattr(resolution, "unresolved_fields", ()) or ())
    if requested:
        missing_set = set(missing)
        return tuple(value for value in requested if value not in missing_set), missing

    requested_fact = getattr(resolution, "requested_fact", None)
    unresolved = tuple(
        str(value) for value in getattr(resolution, "unresolved_requirements", ()) or ()
    )
    facts = tuple(getattr(resolution, "facts", ()) or ())
    if requested_fact and facts:
        confirmed = () if requested_fact in set(unresolved) else (str(requested_fact),)
        return confirmed, unresolved
    return (), unresolved


def _requested_evidence_fields(
    plan: Any,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    question = _compact(
        getattr(plan, "raw_query", None) or getattr(plan, "query", None) or ""
    )
    if not question:
        return ()
    definitions = (
        (
            "contract_amount",
            ("계약금액", "해지금액"),
            ("계약금액", "해지금액"),
        ),
        (
            "termination_reason",
            ("해지사유", "해지주요사유", "해지이유", "종료사유"),
            ("해지사유", "주요사유", "종료사유"),
        ),
        (
            "contract_counterparty",
            ("계약상대", "상대방"),
            ("계약상대", "상대방"),
        ),
        (
            "contract_period",
            ("계약기간",),
            ("계약기간", "시작일", "종료일"),
        ),
        (
            "investment_amount",
            ("투자금액", "투자규모"),
            ("투자금액", "투자규모"),
        ),
        (
            "investment_purpose",
            ("투자목적",),
            ("투자목적",),
        ),
    )
    return tuple(
        (field, evidence_aliases)
        for field, question_aliases, evidence_aliases in definitions
        if any(_compact(alias) in question for alias in question_aliases)
    )


def _semantic_evidence_relevance(
    plan: Any, *, generated: Any, execution: Any
) -> bool | None:
    """Check cited event evidence against the plan's frozen route metadata.

    This is deliberately not a lexical relevance judge and does not rerank or
    filter retrieval.  It only rejects the observable mismatch where an event
    plan cites documents from a different disclosure group.
    """

    if plan is None or not getattr(plan, "event_type", None):
        return None
    expected_routes = tuple(getattr(plan, "disclosure_route", ()) or ())
    if not expected_routes:
        return None
    cited_chunk_ids = {
        str(chunk_id)
        for citation in (getattr(generated, "citations", ()) or ())
        if (chunk_id := getattr(citation, "chunk_id", None))
    }
    if not cited_chunk_ids:
        return None
    observed_routes = []
    for candidate in (getattr(execution, "chunks", ()) or ()):
        if str(getattr(candidate, "chunk_id", "")) not in cited_chunk_ids:
            continue
        chunk = getattr(candidate, "chunk", None)
        if not isinstance(chunk, Mapping):
            continue
        route = str(chunk.get("doc_group") or "").strip()
        if route:
            observed_routes.append(route)
    if not observed_routes:
        return None
    return any(route in expected_routes for route in observed_routes)


def _compact(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").casefold())


__all__ = [
    "AnswerabilityGuard",
    "AnswerabilityResult",
    "AnswerabilityStatus",
    "contains_categorical_negative",
    "guarded_answer_text",
]
