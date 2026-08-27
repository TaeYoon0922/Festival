"""Read-only coordinator for existing Festival reasoning components."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from app.agent.task_router import TaskDecision, TaskRouter
from app.reasoning.answer_composer import (
    AnswerComposer,
    AnswerDraft,
    AnswerSection,
    EvidenceCitation,
)
from app.reasoning.evidence_builder import EvidenceBuilder, EvidenceItem, EvidenceSet
from app.reasoning.holding_event_resolver import (
    HoldingEventResolver,
    HoldingResolution,
)
from app.reasoning.periodic_fact_resolver import (
    PeriodicFactResolution,
    PeriodicFactResolver,
)
from app.reasoning.holding_date_intent import (
    exact_reference_date,
    execution_plan as holding_execution_plan,
)
from app.reasoning.holding_event_fusion import fuse as fuse_holding_events
from app.reasoning.holding_evidence_coverage import (
    CoverageAssessment,
    assess as assess_holding_coverage,
)
from app.reasoning.periodic_evidence_selector import PeriodicEvidenceSelector


#: How a routed execution shape maps onto the grouping EvidenceBuilder performs.
#:
#: TaskRouter answers "which resolver runs"; EvidenceBuilder answers "how is the
#: evidence grouped". Those are separate questions in separate vocabularies, and
#: the resolver can only read groups built for it -- ``holding_event_resolver``
#: consumes ``holding_event`` groups, which EvidenceBuilder builds under its own
#: ``holding_change`` grouping name. Routes absent from this table keep the
#: plan's own task type as the grouping, which is what they already did: the
#: periodic resolver accepts the default document/standalone groups, and the
#: remaining routes use no group-consuming resolver at all.
_EXECUTION_GROUPING_INTENT = {
    "holding_event": "holding_change",
}


MAX_GENERAL_EVIDENCE = 5
MAX_CORPORATE_EVENT_EVIDENCE = 6
MAX_HOLDING_GENERAL_EVIDENCE = 9
MAX_GENERAL_EVIDENCE_TEXT_CHARS = 600
MAX_PRIMARY_GENERAL_EVIDENCE_TEXT_CHARS = 1200


@dataclass(frozen=True)
class AgentResult:
    question: str
    task_decision: TaskDecision
    evidence_set: EvidenceSet
    resolution: HoldingResolution | PeriodicFactResolution | None
    answer_draft: AnswerDraft
    warnings: tuple[str, ...]
    execution_trace: tuple[str, ...]
    #: Internal diagnostic only; never serialised into the public response.
    holding_coverage: CoverageAssessment = field(default_factory=CoverageAssessment)
    #: The served evidence the answer was actually built from.  This equals the
    #: retrieval output unless a post-retrieval stage enriched it; because that
    #: output is immutable, an enriched list can only be carried out here.  The
    #: public response renders this list, so what the resolver used and what the
    #: caller is shown stay the same evidence.
    evidence_results: tuple[Any, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        resolution = self.resolution
        return {
            "question": self.question,
            "task_decision": self.task_decision.to_dict(),
            "evidence_set": self.evidence_set.to_dict(),
            "resolution": resolution.to_dict() if resolution is not None else None,
            "answer_draft": self.answer_draft.to_dict(),
            "warnings": list(self.warnings),
            "execution_trace": list(self.execution_trace),
        }


class AgentOrchestrator:
    """Connect frozen components without modifying retrieval or resolved evidence."""

    def __init__(
        self,
        *,
        task_router: TaskRouter | None = None,
        evidence_builder: EvidenceBuilder | None = None,
        holding_resolver: HoldingEventResolver | None = None,
        periodic_resolver: PeriodicFactResolver | None = None,
        periodic_evidence_selector: PeriodicEvidenceSelector | None = None,
        answer_composer: AnswerComposer | None = None,
    ) -> None:
        self.task_router = task_router or TaskRouter()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.holding_resolver = holding_resolver or HoldingEventResolver()
        self.periodic_resolver = periodic_resolver or PeriodicFactResolver()
        self.periodic_evidence_selector = (
            periodic_evidence_selector or PeriodicEvidenceSelector()
        )
        self.answer_composer = answer_composer or AnswerComposer()

    def run(
        self,
        question: str,
        query_plan: Any,
        retrieval_execution: Any,
        *,
        candidate_chunks: Sequence[Any] | None = None,
        multi_document: Any = None,
    ) -> AgentResult:
        execution = _execution_view(
            query_plan,
            retrieval_execution,
            candidate_chunks=candidate_chunks,
        )
        input_before = _execution_snapshot(execution)
        plan_before = _plan_snapshot(query_plan)
        trace = ["task_router"]
        decision = self.task_router.route(question, query_plan)

        # A holding execution can be served plenty of evidence and still be
        # unable to answer: the structured projections carrying the fields may
        # have missed the cutoff, or the served ones may be the wrong projection
        # type for what was asked. Top the gap up from the pool retrieval
        # already fetched -- no query is issued here -- and leave the original
        # execution untouched so the read-only invariants still hold.
        coverage = assess_holding_coverage(
            question,
            query_plan,
            execution.chunks,
            execution.results,
            routed_task_type=decision.task_type,
        )
        evidence_input = execution
        if coverage.rescued:
            trace.append("holding_evidence_coverage")
            evidence_input = SimpleNamespace(
                plan=execution.plan,
                chunks=execution.chunks,
                results=list(coverage.results),
            )

        # P1-A4 D1. A question naming one calendar day is asking about one
        # event, but the plan only carries that day when its own task type was
        # already holding_change -- a promoted execution gets the year instead.
        # Re-read the date with the routed semantics and carry it on a copy, so
        # the frozen plan (and the understanding trace) stay exactly as P0-D
        # left them.
        holding_plan = holding_execution_plan(
            question, query_plan, routed_task_type=decision.task_type
        )
        if holding_plan is not query_plan:
            trace.append("holding_date_intent")

        trace.append("evidence_builder")
        evidence = self.evidence_builder.build(
            evidence_input,
            question=question,
            grouping_intent=_EXECUTION_GROUPING_INTENT.get(decision.task_type),
        )

        # P1-A4 D2. One filing can render one event twice -- the detail row
        # carries the shares, the report the ratios -- and the two are grouped
        # apart because their holder labels differ. Asked about a single day,
        # that reads as two events, which is both one too many and each one
        # short of the requested fields. Merge only those views, only on the day
        # that was asked about, and only when the numbers say they are the same
        # event. The builder's own output is left untouched.
        if decision.task_type == "holding_event":
            fused = fuse_holding_events(
                evidence,
                reference_date=exact_reference_date(holding_plan),
                reporter=getattr(holding_plan, "reporter", None),
            )
            if fused is not evidence:
                trace.append("holding_event_fusion")
                evidence = fused

        evidence_before = copy.deepcopy(evidence.to_dict())
        resolution: HoldingResolution | PeriodicFactResolution | None

        if decision.task_type == "holding_event":
            trace.append("holding_event_resolver")
            resolution = self.holding_resolver.resolve(
                evidence, query_plan=holding_plan
            )
            resolution_before = copy.deepcopy(resolution.to_dict())
            trace.append("answer_composer")
            draft = self.answer_composer.compose(
                evidence, holding_resolution=resolution
            )
        elif decision.task_type == "periodic_fact":
            trace.append("periodic_fact_resolver")
            resolution = self.periodic_resolver.resolve(
                evidence, query_plan=query_plan
            )
            resolution_before = copy.deepcopy(resolution.to_dict())
            trace.append("periodic_evidence_selector")
            selection = self.periodic_evidence_selector.select(
                resolution, query_plan=query_plan
            )
            trace.append("answer_composer")
            draft = self.answer_composer.compose(
                evidence, periodic_resolution=selection.resolution
            )
        else:
            resolution = None
            resolution_before = None
            trace.append("answer_composer")
            draft = _compose_general_evidence(
                evidence,
                task_type=decision.task_type,
                multi_document=multi_document,
            )

        _enforce_read_only_invariants(
            execution=execution,
            input_before=input_before,
            query_plan=query_plan,
            plan_before=plan_before,
            evidence=evidence,
            evidence_before=evidence_before,
            resolution=resolution,
            resolution_before=resolution_before,
        )
        warnings = tuple(
            dict.fromkeys(
                [
                    *decision.warnings,
                    *evidence.warnings,
                    *(
                        resolution.warnings
                        if resolution is not None
                        else ()
                    ),
                    *draft.warnings,
                ]
            )
        )
        return AgentResult(
            question=question,
            task_decision=decision,
            evidence_set=evidence,
            resolution=resolution,
            answer_draft=draft,
            warnings=warnings,
            execution_trace=tuple(trace),
            holding_coverage=coverage,
            evidence_results=tuple(evidence_input.results),
        )


def orchestrate(
    question: str,
    query_plan: Any,
    retrieval_execution: Any,
    *,
    candidate_chunks: Sequence[Any] | None = None,
) -> AgentResult:
    return AgentOrchestrator().run(
        question,
        query_plan,
        retrieval_execution,
        candidate_chunks=candidate_chunks,
    )


def _compose_general_evidence(
    evidence: EvidenceSet, *, task_type: str, multi_document: Any = None
) -> AnswerDraft:
    item_by_id = {
        item.chunk_id: item
        for group in evidence.evidence_groups
        for item in group.items
    }
    ordered_items = [
        item_by_id[chunk_id]
        for chunk_id in evidence.retrieval_order
        if chunk_id in item_by_id
    ]
    max_evidence = _general_evidence_limit(evidence, task_type=task_type)
    selected_items = ordered_items[:max_evidence]
    facts = getattr(multi_document, "facts", None)
    multi_document_items = _multi_document_items(multi_document, item_by_id)
    citation_items = tuple(
        {
            item.chunk_id: item
            for item in (*selected_items, *multi_document_items)
        }.values()
    )
    evidence_rows = [
        _general_evidence_row(
            item,
            question=evidence.question,
            task_type=task_type,
            is_primary=index == 0,
        )
        for index, item in enumerate(selected_items)
    ]
    citations = tuple(_general_citation(item) for item in citation_items)
    evidence_ids = tuple(item.chunk_id for item in selected_items)
    unknown = task_type == "unknown"
    # An applied P0-C plan contributes deterministic answer material even when
    # the logical set is empty (and therefore has no filing to hydrate).  Treat
    # that set-level result as supported; otherwise ``no_members`` would be
    # followed by the generic retrieval-failure fallback.
    answerable = (bool(selected_items and citations) and not unknown) or facts is not None
    warnings = list(evidence.warnings)
    warnings.append("resolver_not_required" if not unknown else "unknown_task")
    if len(ordered_items) > len(selected_items):
        warnings.append(f"general_evidence_limited:max={max_evidence}")
    if not answerable:
        warnings.append("answer_not_supported")
    sections = (
        (
            AnswerSection(
                title="General evidence",
                content={"evidence": evidence_rows},
                supporting_evidence_ids=evidence_ids,
            ),
        )
        if evidence_rows
        else ()
    )
    if facts is not None:
        # Deterministic counts the model must state rather than derive. Placed
        # first so the set-level claim leads, and absent entirely when P0-C did
        # not engage -- which keeps every existing draft byte-identical.
        sections = (
            AnswerSection(
                title="계약 확인 결과",
                content={"summary": facts.statement(), **facts.to_dict()},
                supporting_evidence_ids=tuple(
                    item.chunk_id for item in multi_document_items
                ),
            ),
            *sections,
        )
    return AnswerDraft(
        question=evidence.question,
        task_type=task_type,
        answer_sections=sections,
        evidence_references=evidence_ids,
        citations=citations,
        ambiguity={
            **copy.deepcopy(dict(evidence.ambiguity)),
            "automatically_resolved": False,
        },
        warnings=tuple(dict.fromkeys(warnings)),
        confidence={
            "level": "medium" if answerable else "low",
            "score": 0.5 if answerable else 0.0,
            "answerable": answerable,
            "citation_count": len(citations),
            "basis": "evidence_presence_and_provenance",
        },
        answerable=answerable,
    )


def _multi_document_items(
    multi_document: Any, item_by_id: Mapping[str, EvidenceItem]
) -> tuple[EvidenceItem, ...]:
    """P0-C hydrated rows in their answer-critical evidence order.

    These rows may sit after the frozen ranked Top-K and therefore outside the
    general evidence display limit.  They still have to enter the citation
    registry: otherwise a termination summary could be rendered while its
    opening and termination filings were present in ``retrieved_context`` but
    absent from the actual answer citations.
    """

    if multi_document is None:
        return ()
    return tuple(
        item_by_id[result.chunk_id]
        for result in (getattr(multi_document, "added_results", ()) or ())
        if result.chunk_id in item_by_id
    )


def _general_evidence_row(
    item: EvidenceItem,
    *,
    question: str = "",
    task_type: str = "",
    is_primary: bool = False,
) -> dict[str, Any]:
    limit = (
        MAX_PRIMARY_GENERAL_EVIDENCE_TEXT_CHARS
        if is_primary
        else MAX_GENERAL_EVIDENCE_TEXT_CHARS
    )
    focused_text, focused = _focused_general_evidence_text(
        item.evidence_text,
        question=question,
        task_type=task_type,
    )
    evidence_text, truncated = _bounded_general_evidence_text(
        focused_text,
        limit=limit,
    )
    return {
        "chunk_id": item.chunk_id,
        "doc_id": item.doc_id,
        "section_path": list(item.section_path),
        "evidence_text": evidence_text,
        "truncated": truncated or focused,
        "retrieval_rank": item.retrieval_rank,
        "retrieval_score": item.retrieval_score,
        "source_refs": copy.deepcopy(list(item.source_refs)),
    }


def _focused_general_evidence_text(
    value: str,
    *,
    question: str,
    task_type: str,
) -> tuple[str, bool]:
    text = str(value or "")
    if task_type != "corporate_event":
        return text, False
    focus_terms = _event_focus_terms(question)
    if not focus_terms:
        return text, False

    prefix: list[str] = []
    focused_rows: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|"):
            if _is_markdown_separator(stripped):
                continue
            if _event_focus_row_matches(stripped, focus_terms):
                focused_rows.append(stripped)
        elif len(prefix) < 5 and (
            stripped.startswith("[기업명]")
            or stripped.startswith("[공시명]")
            or stripped.startswith("[Section Path]")
            or stripped.startswith("[Table]")
            or stripped.startswith("[기간표현]")
        ):
            prefix.append(stripped)

    if not focused_rows:
        return text, False
    return "\n".join([*prefix, *focused_rows]), True


def _event_focus_terms(question: str) -> tuple[str, ...]:
    compact = _compact_general_text(question)
    terms: list[str] = []
    if any(
        trigger in compact
        for trigger in ("자기주식처분", "자사주처분", "처분예정", "처분가격", "처분주식")
    ):
        terms.append("처분예정주식")
        if any(term in compact for term in ("가격", "주식가격")):
            terms.append("처분대상주식가격")
        if "금액" in compact:
            terms.append("처분예정금액")
        if any(term in compact for term in ("목적", "내용")):
            terms.append("처분목적")
        return tuple(dict.fromkeys(_compact_general_text(term) for term in terms))

    if any(trigger in compact for trigger in ("공급계약", "단일판매", "수주계약")):
        terms.append("체결계약명")
        terms.append("계약상대")
        terms.append("판매공급지역")
        if any(term in compact for term in ("금액", "계약내역", "내용")):
            terms.append("계약금액")
        if any(term in compact for term in ("매출액대비", "비율", "내용")):
            terms.extend(("최근매출액", "매출액대비"))
        if any(term in compact for term in ("기간", "내용")):
            terms.append("계약기간")
        if any(term in compact for term in ("일자", "내용")):
            terms.append("계약수주일자")
        return tuple(dict.fromkeys(_compact_general_text(term) for term in terms))

    rules = (
        (
            ("신규시설투자", "시설투자"),
            (
                "투자금액",
                "자기자본",
                "투자목적",
                "투자기간",
                "자금조달",
            ),
        ),
        (
            ("신탁계약해지", "자기주식취득신탁계약해지"),
            ("계약금액", "해지목적", "해지기관", "해지예정일", "해지전", "해지후"),
        ),
        (
            ("조건부자본증권", "상각형조건부자본증권", "사채총액"),
            ("사채의권면", "전자등록총액", "발행가액", "이자율", "만기일"),
        ),
        (
            ("유상증자",),
            (
                "신주의종류와수",
                "신주발행가액",
                "자금조달의목적",
                "시설자금",
                "운영자금",
                "채무상환자금",
                "증자방식",
            ),
        ),
    )
    terms: list[str] = []
    for triggers, row_terms in rules:
        if any(trigger in compact for trigger in triggers):
            terms.extend(_compact_general_text(term) for term in row_terms)
    return tuple(dict.fromkeys(term for term in terms if term))


def _event_focus_row_matches(row: str, focus_terms: Sequence[str]) -> bool:
    cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
    if not cells:
        return False
    leading = cells[0]
    leading_compact = _compact_general_text(leading)
    if _event_note_row(leading, leading_compact):
        return False
    label = _compact_general_text(" ".join(cells[:2]))
    return any(term in label for term in focus_terms)


def _event_note_row(label: str, compact_label: str) -> bool:
    if "체결계약명" in compact_label:
        return False
    stripped = label.strip()
    return bool(
        re.match(r"^[가-힣]\.", stripped)
        or re.match(r"^\d+\)", stripped)
        or stripped.startswith("※")
        or stripped.startswith("-")
        or stripped.startswith("상기")
        or stripped.startswith("- 상기")
    )


def _is_markdown_separator(value: str) -> bool:
    return bool(re.fullmatch(r"\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?", value))


def _compact_general_text(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", str(value or "")).casefold()


def _bounded_general_evidence_text(value: str, *, limit: int = MAX_GENERAL_EVIDENCE_TEXT_CHARS) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= limit:
        return text, False
    return (
        text[:limit].rstrip() + "\n[truncated]",
        True,
    )


def _general_evidence_limit(evidence: EvidenceSet, *, task_type: str) -> int:
    if task_type == "corporate_event":
        return MAX_CORPORATE_EVENT_EVIDENCE
    if task_type == "general_evidence" and _holding_route_general_evidence(evidence):
        return MAX_HOLDING_GENERAL_EVIDENCE
    return MAX_GENERAL_EVIDENCE


def _holding_route_general_evidence(evidence: EvidenceSet) -> bool:
    routes = evidence.query_plan.get("disclosure_route") or ()
    if isinstance(routes, str):
        routes = (routes,)
    if "holding" not in {str(route) for route in routes}:
        return False
    return any(
        item.doc_group == "holding"
        for group in evidence.evidence_groups
        for item in group.items
    )


def _general_citation(item: EvidenceItem) -> EvidenceCitation:
    source_chunk = item.provenance.get("source_chunk") or {}
    return EvidenceCitation(
        chunk_id=item.chunk_id,
        doc_id=item.doc_id,
        source_refs=tuple(copy.deepcopy(list(item.source_refs))),
        provenance_path=(
            {
                "resolver": None,
                "source_chunk_id": item.provenance.get("source_chunk_id"),
                "source_doc_id": item.provenance.get("source_doc_id"),
                "original_source_chunk_id": source_chunk.get("chunk_id"),
                "original_source_doc_id": source_chunk.get("doc_id"),
            },
        ),
    )


def _execution_view(
    query_plan: Any,
    retrieval_execution: Any,
    *,
    candidate_chunks: Sequence[Any] | None,
) -> Any:
    if hasattr(retrieval_execution, "chunks") and hasattr(
        retrieval_execution, "results"
    ):
        chunks = retrieval_execution.chunks
        results = retrieval_execution.results
    elif isinstance(retrieval_execution, Sequence) and not isinstance(
        retrieval_execution, (str, bytes)
    ):
        if candidate_chunks is None:
            raise ValueError(
                "candidate_chunks are required when retrieval_execution is a result sequence"
            )
        chunks = candidate_chunks
        results = retrieval_execution
    else:
        raise TypeError("retrieval_execution must expose chunks/results or be a sequence")
    return SimpleNamespace(plan=query_plan, chunks=chunks, results=results)


def _execution_snapshot(execution: Any) -> dict[str, Any]:
    return {
        "chunks": [
            (
                chunk.chunk_id,
                chunk.doc_id,
                copy.deepcopy(dict(chunk.chunk)),
                copy.deepcopy(chunk.metadata_match),
            )
            for chunk in execution.chunks
        ],
        "results": [
            (
                result.chunk_id,
                result.doc_id,
                result.rank,
                result.bm25_score,
                copy.deepcopy(dict(result.metadata_match or {})),
            )
            for result in execution.results
        ],
    }


def _plan_snapshot(query_plan: Any) -> Any:
    if hasattr(query_plan, "to_dict"):
        return copy.deepcopy(query_plan.to_dict())
    if isinstance(query_plan, Mapping):
        return copy.deepcopy(dict(query_plan))
    return copy.deepcopy(query_plan)


def _enforce_read_only_invariants(
    *,
    execution: Any,
    input_before: Mapping[str, Any],
    query_plan: Any,
    plan_before: Any,
    evidence: EvidenceSet,
    evidence_before: Mapping[str, Any],
    resolution: HoldingResolution | PeriodicFactResolution | None,
    resolution_before: Mapping[str, Any] | None,
) -> None:
    violations = []
    if _execution_snapshot(execution) != dict(input_before):
        violations.append("retrieval_or_candidate_mutated")
    if _plan_snapshot(query_plan) != plan_before:
        violations.append("query_plan_mutated")
    if evidence.to_dict() != dict(evidence_before):
        violations.append("evidence_set_mutated")
    if resolution is not None and resolution.to_dict() != dict(
        resolution_before or {}
    ):
        violations.append("resolver_output_mutated")
    if violations:
        raise RuntimeError("read_only_invariant_violation:" + ",".join(violations))
