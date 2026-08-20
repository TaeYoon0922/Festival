"""Read-only coordinator for existing Festival reasoning components."""

from __future__ import annotations

import copy
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AgentResult:
    question: str
    task_decision: TaskDecision
    evidence_set: EvidenceSet
    resolution: HoldingResolution | PeriodicFactResolution | None
    answer_draft: AnswerDraft
    warnings: tuple[str, ...]
    execution_trace: tuple[str, ...]

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
        answer_composer: AnswerComposer | None = None,
    ) -> None:
        self.task_router = task_router or TaskRouter()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.holding_resolver = holding_resolver or HoldingEventResolver()
        self.periodic_resolver = periodic_resolver or PeriodicFactResolver()
        self.answer_composer = answer_composer or AnswerComposer()

    def run(
        self,
        question: str,
        query_plan: Any,
        retrieval_execution: Any,
        *,
        candidate_chunks: Sequence[Any] | None = None,
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

        trace.append("evidence_builder")
        evidence = self.evidence_builder.build(execution, question=question)
        evidence_before = copy.deepcopy(evidence.to_dict())
        resolution: HoldingResolution | PeriodicFactResolution | None

        if decision.task_type == "holding_event":
            trace.append("holding_event_resolver")
            resolution = self.holding_resolver.resolve(
                evidence, query_plan=query_plan
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
            trace.append("answer_composer")
            draft = self.answer_composer.compose(
                evidence, periodic_resolution=resolution
            )
        else:
            resolution = None
            resolution_before = None
            trace.append("answer_composer")
            draft = _compose_general_evidence(
                evidence,
                task_type=decision.task_type,
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
    evidence: EvidenceSet, *, task_type: str
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
    evidence_rows = [_general_evidence_row(item) for item in ordered_items]
    citations = tuple(_general_citation(item) for item in ordered_items)
    evidence_ids = tuple(item.chunk_id for item in ordered_items)
    unknown = task_type == "unknown"
    answerable = bool(ordered_items and citations) and not unknown
    warnings = list(evidence.warnings)
    warnings.append("resolver_not_required" if not unknown else "unknown_task")
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


def _general_evidence_row(item: EvidenceItem) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk_id,
        "doc_id": item.doc_id,
        "section_path": list(item.section_path),
        "evidence_text": item.evidence_text,
        "retrieval_rank": item.retrieval_rank,
        "retrieval_score": item.retrieval_score,
        "source_refs": copy.deepcopy(list(item.source_refs)),
    }


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
