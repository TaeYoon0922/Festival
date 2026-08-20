"""Read-only Gold60 evaluation across retrieval, reasoning, and generation."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.reasoning.failure_analysis import analyze_gold60_failures
from app.reasoning.hybrid_evaluation import QueryPlanHybridEvaluator
from app.reasoning.lexical_evaluation import _candidate_is_relevant


ProgressCallback = Callable[[int, int, str], None]
AGENT_EVALUATION_VERSION = "1"


class _RecordingUnderstanding:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.last_plan: Any | None = None

    def understand(self, query: str, *, top_k: int = 10) -> Any:
        plan = self.delegate.understand(query, top_k=top_k)
        self.last_plan = plan
        return plan


class _RecordingExecutor:
    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.last_execution: Any | None = None

    def execute(self, plan: Any) -> Any:
        execution = self.delegate.execute(plan)
        self.last_execution = execution
        return execution

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


class AgentGold60Evaluator(QueryPlanHybridEvaluator):
    """Extend the frozen retrieval evaluation with deterministic Agent outputs."""

    def __init__(
        self,
        understanding: Any,
        executor: Any,
        *,
        top_k: int = 10,
        orchestrator: AgentOrchestrator | None = None,
        generator: CitationAwareAnswerGenerator | None = None,
    ) -> None:
        self._recording_understanding = _RecordingUnderstanding(understanding)
        self._recording_executor = _RecordingExecutor(executor)
        self.orchestrator = orchestrator or AgentOrchestrator()
        self.generator = generator or CitationAwareAnswerGenerator()
        super().__init__(
            self._recording_understanding,
            self._recording_executor,
            top_k=top_k,
        )

    def evaluate(
        self,
        question_sets: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        report = super().evaluate(question_sets, progress=progress)
        rows = report["questions"]
        agent_summary = _agent_summary(rows)
        report["evaluation_version"] = AGENT_EVALUATION_VERSION
        report["method"]["agent_flow"] = (
            "same HybridExecution -> AgentOrchestrator -> TaskRouter -> "
            "EvidenceBuilder -> optional resolver -> AnswerComposer -> "
            "CitationAwareAnswerGenerator"
        )
        report["method"]["answer_gold_comparison"] = {
            "gold_doc_cited": "a generated citation references the gold document",
            "gold_chunk_cited": "a cited chunk satisfies the frozen gold judgment",
            "all_evidence_terms_present": (
                "every frozen gold evidence term is present in the generated answer "
                "after deterministic normalization"
            ),
            "end_to_end_success": (
                "retrieval hit@10, answerable answer, gold document citation, and "
                "all gold evidence terms present"
            ),
        }
        report["agent"] = agent_summary
        report["end_to_end_failures"] = [
            row for row in rows if row["end_to_end_failure_class"] != "success"
        ]
        return report

    def _evaluate_question(
        self, set_name: str, question: Mapping[str, Any]
    ) -> dict[str, Any]:
        row = super()._evaluate_question(set_name, question)
        plan = self._recording_understanding.last_plan
        execution = self._recording_executor.last_execution
        if plan is None or execution is None:
            raise RuntimeError("recorded QueryPlan/HybridExecution is unavailable")
        try:
            row.update(
                _agent_evaluation_payload(
                    question,
                    plan=plan,
                    execution=execution,
                    orchestrator=self.orchestrator,
                    generator=self.generator,
                    top_k=self.top_k,
                )
            )
        finally:
            self._recording_understanding.last_plan = None
            self._recording_executor.last_execution = None
        return row


def _agent_evaluation_payload(
    question: Mapping[str, Any],
    *,
    plan: Any,
    execution: Any,
    orchestrator: AgentOrchestrator,
    generator: CitationAwareAnswerGenerator,
    top_k: int,
) -> dict[str, Any]:
    retrieved = _retrieved_chunks(execution, question, limit=top_k)
    try:
        agent_result = orchestrator.run(str(question["query"]), plan, execution)
        draft_before = copy.deepcopy(agent_result.answer_draft.to_dict())
        generated = generator.generate(agent_result.answer_draft)
        draft_preserved = agent_result.answer_draft.to_dict() == draft_before
        source_references = _source_references(
            agent_result.answer_draft, generated
        )
        comparison = _answer_gold_comparison(
            question,
            generated=generated,
            source_references=source_references,
            candidates=execution.chunks,
        )
        failure_class = _end_to_end_failure_class(
            retrieval_hit=any(
                row["is_gold_relevant"] and row["rank"] <= 10
                for row in retrieved
            ),
            agent_status="ok",
            answerable=generated.answerable,
            comparison=comparison,
        )
        return {
            "retrieved_chunks": retrieved,
            "agent": {
                "status": "ok",
                "task_decision": agent_result.task_decision.to_dict(),
                "execution_trace": [
                    *agent_result.execution_trace,
                    "answer_generator",
                ],
                "evidence": {
                    "raw_candidate_count": agent_result.evidence_set.raw_candidate_count,
                    "selected_evidence_count": (
                        agent_result.evidence_set.selected_evidence_count
                    ),
                    "evidence_group_count": len(
                        agent_result.evidence_set.evidence_groups
                    ),
                    "ambiguity": copy.deepcopy(
                        dict(agent_result.evidence_set.ambiguity)
                    ),
                },
                "resolution": (
                    agent_result.resolution.to_dict()
                    if agent_result.resolution is not None
                    else None
                ),
                "answer_draft": agent_result.answer_draft.to_dict(),
                "generated_answer": generated.to_dict(),
                "answerable": generated.answerable,
                "warnings": list(
                    dict.fromkeys([*agent_result.warnings, *generated.warnings])
                ),
                "answer_draft_preserved": draft_preserved,
            },
            "source_references": source_references,
            "answer_gold_comparison": comparison,
            "end_to_end_failure_class": failure_class,
            "end_to_end_success": failure_class == "success",
        }
    except Exception as error:
        comparison = _empty_answer_comparison(question)
        return {
            "retrieved_chunks": retrieved,
            "agent": {
                "status": "failed",
                "error_type": type(error).__name__,
                "answerable": False,
                "warnings": ["agent_pipeline_failed"],
            },
            "source_references": [],
            "answer_gold_comparison": comparison,
            "end_to_end_failure_class": "agent_pipeline_error",
            "end_to_end_success": False,
        }


def _retrieved_chunks(
    execution: Any,
    question: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates = {candidate.chunk_id: candidate for candidate in execution.chunks}
    rows: list[dict[str, Any]] = []
    for result in execution.results[:limit]:
        candidate = candidates.get(result.chunk_id)
        chunk = candidate.chunk if candidate is not None else {}
        section_path = chunk.get("section_path") or []
        if isinstance(section_path, str):
            section_path = [section_path]
        rows.append(
            {
                "rank": int(result.rank),
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "bm25_score": float(result.bm25_score),
                "score_metadata": copy.deepcopy(dict(result.metadata_match or {})),
                "is_gold_relevant": _candidate_is_relevant(
                    chunk, result.doc_id, question
                ),
                "chunk_type": chunk.get("chunk_type"),
                "section_path": list(section_path),
                "report_nm": chunk.get("report_nm"),
                "corp_code": chunk.get("corp_code"),
                "corp_name": chunk.get("corp_name"),
                "rcept_dt": chunk.get("rcept_dt"),
                "period": _chunk_period(chunk),
                "content": chunk.get("content"),
                "retrieval_text": chunk.get("retrieval_text"),
                "source_refs": copy.deepcopy(list(chunk.get("source_refs") or [])),
                "provenance": copy.deepcopy(dict(chunk.get("provenance") or {})),
            }
        )
    return rows


def _chunk_period(chunk: Mapping[str, Any]) -> dict[str, Any]:
    period = chunk.get("period")
    if isinstance(period, Mapping):
        return copy.deepcopy(dict(period))
    return {
        key: chunk.get(key)
        for key in ("base_year", "base_month", "fiscal_year", "quarter", "period_type")
        if chunk.get(key) is not None
    }


def _source_references(draft: Any, generated: Any) -> list[dict[str, Any]]:
    generated_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    for citation in generated.citations:
        generated_ids[(citation.chunk_id, citation.doc_id)].append(
            citation.citation_id
        )
    return [
        {
            "citation_ids": generated_ids.get(
                (citation.chunk_id, citation.doc_id), []
            ),
            "chunk_id": citation.chunk_id,
            "doc_id": citation.doc_id,
            "source_refs": copy.deepcopy(list(citation.source_refs)),
            "provenance_path": copy.deepcopy(list(citation.provenance_path)),
        }
        for citation in draft.citations
    ]


def _answer_gold_comparison(
    question: Mapping[str, Any],
    *,
    generated: Any,
    source_references: Sequence[Mapping[str, Any]],
    candidates: Sequence[Any],
) -> dict[str, Any]:
    candidate_by_id = {
        candidate.chunk_id: candidate.chunk for candidate in candidates
    }
    gold_doc_id = str(question.get("doc_id") or "")
    cited = [row for row in source_references if row.get("citation_ids")]
    gold_doc_cited = any(str(row.get("doc_id") or "") == gold_doc_id for row in cited)
    gold_chunk_ids = [
        str(row.get("chunk_id") or "")
        for row in cited
        if _candidate_is_relevant(
            candidate_by_id.get(str(row.get("chunk_id") or ""), {}),
            str(row.get("doc_id") or ""),
            question,
        )
    ]
    term_rows = _evidence_term_comparison(
        question.get("evidence_terms") or [], generated.answer_text
    )
    return {
        "gold_doc_id": gold_doc_id,
        "gold_doc_cited": gold_doc_cited,
        "gold_chunk_cited": bool(gold_chunk_ids),
        "gold_cited_chunk_ids": list(dict.fromkeys(gold_chunk_ids)),
        "evidence_terms": term_rows,
        "matched_evidence_terms": [
            row["term"] for row in term_rows if row["present"]
        ],
        "missing_evidence_terms": [
            row["term"] for row in term_rows if not row["present"]
        ],
        "all_evidence_terms_present": bool(term_rows)
        and all(row["present"] for row in term_rows),
    }


def _empty_answer_comparison(question: Mapping[str, Any]) -> dict[str, Any]:
    terms = [str(value) for value in question.get("evidence_terms") or []]
    return {
        "gold_doc_id": str(question.get("doc_id") or ""),
        "gold_doc_cited": False,
        "gold_chunk_cited": False,
        "gold_cited_chunk_ids": [],
        "evidence_terms": [{"term": term, "present": False} for term in terms],
        "matched_evidence_terms": [],
        "missing_evidence_terms": terms,
        "all_evidence_terms_present": False,
    }


def _evidence_term_comparison(
    terms: Sequence[Any], answer_text: str
) -> list[dict[str, Any]]:
    normalized_answer = _normalize_comparison_text(answer_text)
    answer_digits = re.sub(r"\D+", "", answer_text)
    output = []
    for value in terms:
        term = str(value)
        normalized = _normalize_comparison_text(term)
        digits = re.sub(r"\D+", "", term)
        present = bool(normalized and normalized in normalized_answer)
        if not present and len(digits) >= 4:
            present = digits in answer_digits
        output.append({"term": term, "present": present})
    return output


def _normalize_comparison_text(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").casefold())


def _end_to_end_failure_class(
    *,
    retrieval_hit: bool,
    agent_status: str,
    answerable: bool,
    comparison: Mapping[str, Any],
) -> str:
    if agent_status != "ok":
        return "agent_pipeline_error"
    if not retrieval_hit:
        return "retrieval_miss"
    if not answerable:
        return "answer_not_supported"
    if not comparison.get("gold_doc_cited"):
        return "gold_source_not_cited"
    if not comparison.get("all_evidence_terms_present"):
        return "gold_evidence_terms_missing"
    return "success"


def _agent_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("doc_group") or "unknown")].append(row)
    return {
        "overall": _agent_metrics(rows),
        "by_doc_group": {
            name: _agent_metrics(groups[name]) for name in sorted(groups)
        },
        "failure_counts": dict(
            sorted(Counter(row["end_to_end_failure_class"] for row in rows).items())
        ),
        "question_count": count,
    }


def _agent_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)

    def ratio(predicate: Callable[[Mapping[str, Any]], bool]) -> float:
        return round(sum(predicate(row) for row in rows) / count, 6) if count else 0.0

    return {
        "question_count": count,
        "answerable_rate": ratio(
            lambda row: bool((row.get("agent") or {}).get("answerable"))
        ),
        "gold_doc_citation_rate": ratio(
            lambda row: bool(
                (row.get("answer_gold_comparison") or {}).get("gold_doc_cited")
            )
        ),
        "gold_chunk_citation_rate": ratio(
            lambda row: bool(
                (row.get("answer_gold_comparison") or {}).get("gold_chunk_cited")
            )
        ),
        "all_evidence_terms_rate": ratio(
            lambda row: bool(
                (row.get("answer_gold_comparison") or {}).get(
                    "all_evidence_terms_present"
                )
            )
        ),
        "end_to_end_success_rate": ratio(
            lambda row: bool(row.get("end_to_end_success"))
        ),
    }


def analyze_agent_gold60_failures(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in report.get("questions") or []]
    failures = []
    for row in rows:
        category = str(row.get("end_to_end_failure_class") or "unknown")
        if category == "success":
            continue
        comparison = dict(row.get("answer_gold_comparison") or {})
        agent = dict(row.get("agent") or {})
        failures.append(
            {
                "question_id": row.get("question_id"),
                "evaluation_set": row.get("evaluation_set"),
                "doc_group": row.get("doc_group"),
                "question": row.get("question"),
                "failure_category": category,
                "reason": _failure_reason(category, comparison),
                "retrieval": {
                    "hybrid_gold_rank": row.get("hybrid_gold_rank"),
                    "hit_at_10": row.get("hit_at_10"),
                    "retrieved_chunks": row.get("retrieved_chunks") or [],
                },
                "answer": {
                    "status": agent.get("status"),
                    "answerable": agent.get("answerable"),
                    "warnings": agent.get("warnings") or [],
                    "task_decision": agent.get("task_decision"),
                    "resolution": agent.get("resolution"),
                    "answer_draft": agent.get("answer_draft"),
                    "generated_answer": agent.get("generated_answer"),
                },
                "gold": row.get("gold"),
                "answer_gold_comparison": comparison,
                "source_references": row.get("source_references") or [],
            }
        )
    retrieval_analysis = analyze_gold60_failures(report)
    return {
        "analysis_version": AGENT_EVALUATION_VERSION,
        "summary": {
            "total_questions": len(rows),
            "end_to_end_failures": len(failures),
            "category_counts": dict(
                sorted(Counter(row["failure_category"] for row in failures).items())
            ),
            "retrieval_recall_at_10_misses": retrieval_analysis["summary"][
                "recall_at_10_misses"
            ],
        },
        "retrieval_failure_analysis": retrieval_analysis,
        "failures": failures,
    }


def _failure_reason(category: str, comparison: Mapping[str, Any]) -> str:
    return {
        "agent_pipeline_error": "Agent reasoning or generation raised an error.",
        "retrieval_miss": "Frozen gold evidence was not retrieved in hybrid Top10.",
        "answer_not_supported": "The Agent completed but produced an unanswerable draft.",
        "gold_source_not_cited": "The generated answer did not cite the gold document.",
        "gold_evidence_terms_missing": (
            "The answer cited the gold document but omitted gold evidence terms: "
            + ", ".join(comparison.get("missing_evidence_terms") or [])
        ),
    }.get(category, "Unclassified end-to-end evaluation failure.")


def write_agent_gold60_evaluation(
    report: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = output_dir / "gold60_agent_evaluation.json"
    evaluation_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "gold60_agent_evaluation.md").write_text(
        render_agent_gold60_markdown(report), encoding="utf-8"
    )
    with (output_dir / "gold60_agent_questions.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in report.get("questions") or []:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    analysis = analyze_agent_gold60_failures(report)
    (output_dir / "failure_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    (output_dir / "failure_analysis.md").write_text(
        render_agent_failure_markdown(analysis), encoding="utf-8"
    )
    return analysis


def render_agent_gold60_markdown(report: Mapping[str, Any]) -> str:
    retrieval = dict((report.get("hybrid") or {}).get("overall") or {})
    agent = dict((report.get("agent") or {}).get("overall") or {})
    counts = dict((report.get("agent") or {}).get("failure_counts") or {})
    lines = [
        "# Gold60 End-to-End Agent Evaluation",
        "",
        "## Summary",
        "",
        f"- Questions: {report.get('question_count', 0)}",
        f"- Retrieval Recall@1: {retrieval.get('recall_at_1', 0.0)}",
        f"- Retrieval Recall@5: {retrieval.get('recall_at_5', 0.0)}",
        f"- Retrieval Recall@10: {retrieval.get('recall_at_10', 0.0)}",
        f"- Answerable rate: {agent.get('answerable_rate', 0.0)}",
        f"- Gold document citation rate: {agent.get('gold_doc_citation_rate', 0.0)}",
        f"- Gold evidence terms rate: {agent.get('all_evidence_terms_rate', 0.0)}",
        f"- End-to-end success rate: {agent.get('end_to_end_success_rate', 0.0)}",
        "",
        "| Failure category | Count |",
        "|---|---:|",
    ]
    lines.extend(
        (f"| {name} | {value} |" for name, value in counts.items()),
    )
    return "\n".join(lines) + "\n"


def render_agent_failure_markdown(analysis: Mapping[str, Any]) -> str:
    summary = dict(analysis.get("summary") or {})
    lines = [
        "# Gold60 End-to-End Failure Analysis",
        "",
        "## Summary",
        "",
        f"- Total questions: {summary.get('total_questions', 0)}",
        f"- End-to-end failures: {summary.get('end_to_end_failures', 0)}",
        f"- Retrieval Recall@10 misses: {summary.get('retrieval_recall_at_10_misses', 0)}",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    counts = dict(summary.get("category_counts") or {})
    lines.extend(
        (f"| {name} | {value} |" for name, value in counts.items()),
    )
    for value in analysis.get("failures") or []:
        failure = dict(value)
        comparison = dict(failure.get("answer_gold_comparison") or {})
        lines.extend(
            [
                "",
                f"## {failure.get('question_id')}",
                "",
                str(failure.get("question") or ""),
                "",
                f"- Category: `{failure.get('failure_category')}`",
                f"- Reason: {failure.get('reason')}",
                f"- Hybrid gold rank: `{(failure.get('retrieval') or {}).get('hybrid_gold_rank')}`",
                f"- Gold document cited: `{comparison.get('gold_doc_cited')}`",
                f"- Missing evidence terms: `{', '.join(comparison.get('missing_evidence_terms') or [])}`",
            ]
        )
    return "\n".join(lines) + "\n"
