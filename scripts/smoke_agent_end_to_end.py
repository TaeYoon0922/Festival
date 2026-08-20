"""Run the frozen Festival query-to-answer pipeline as a read-only smoke test.

The production entry point performs database reads through the existing query
understanding and hybrid retrieval components.  ``validate_completed_execution``
is deliberately dependency-injected so the same invariants can be exercised in
unit tests without a PostgreSQL connection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from app.retrieval.hybrid import HybridQueryExecutor, HybridRetrievalConfig
from app.retrieval.postgres_backend import PostgresBackend


DEFAULT_QUERIES: tuple[str, ...] = (
    "효성중공업 국민연금기금 변동일 변동후 주식수",
    "삼성전자 DX 부문의 주요 제품은 무엇인가",
    "삼성전자 유상증자 공시 내용",
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "agent_smoke" / "agent_smoke.json"
)

_FACT_PATTERNS = (
    re.compile(r"(?<!\d)(?:19|20)\d{2}[-./년]\s*\d{1,2}(?:[-./월]\s*\d{1,2}일?)?"),
    re.compile(r"(?<![\w\[])\-?\d{1,3}(?:,\d{3})+(?:\.\d+)?"),
    re.compile(r"(?<![\w\[])\-?\d+(?:\.\d+)?%"),
)
_UNSUPPORTED_ADDITIONS = (
    "투자 추천",
    "매수 추천",
    "매도 추천",
    "시장 전망",
    "주가 상승",
    "주가 하락",
)


def validate_completed_execution(
    question: str,
    query_plan: Any,
    retrieval_execution: Any,
    *,
    orchestrator: AgentOrchestrator | None = None,
    generator: CitationAwareAnswerGenerator | None = None,
) -> dict[str, Any]:
    """Run reasoning/generation over a completed retrieval and validate immutability."""

    coordinator = orchestrator or AgentOrchestrator()
    renderer = generator or CitationAwareAnswerGenerator()
    retrieval_before = _retrieval_snapshot(retrieval_execution)
    plan_before = _object_snapshot(query_plan)

    agent_result = coordinator.run(question, query_plan, retrieval_execution)
    retrieval_after_reasoning = _retrieval_snapshot(retrieval_execution)
    plan_after_reasoning = _object_snapshot(query_plan)
    resolution_before_generation = _object_snapshot(agent_result.resolution)
    draft_before_generation = _object_snapshot(agent_result.answer_draft)

    generated = renderer.generate(agent_result.answer_draft)

    resolution_after_generation = _object_snapshot(agent_result.resolution)
    draft_after_generation = _object_snapshot(agent_result.answer_draft)
    retrieval_after_generation = _retrieval_snapshot(retrieval_execution)
    plan_after_generation = _object_snapshot(query_plan)

    validations = {
        "retrieval_order_unchanged": (
            retrieval_before["result_order"]
            == retrieval_after_reasoning["result_order"]
            == retrieval_after_generation["result_order"]
        ),
        "retrieval_scores_unchanged": (
            retrieval_before["result_scores"]
            == retrieval_after_reasoning["result_scores"]
            == retrieval_after_generation["result_scores"]
        ),
        "candidate_identity_unchanged": (
            retrieval_before["candidate_identity"]
            == retrieval_after_reasoning["candidate_identity"]
            == retrieval_after_generation["candidate_identity"]
        ),
        "candidate_payload_unchanged": (
            retrieval_before["candidate_payload_digest"]
            == retrieval_after_reasoning["candidate_payload_digest"]
            == retrieval_after_generation["candidate_payload_digest"]
        ),
        "query_plan_unchanged": (
            plan_before == plan_after_reasoning == plan_after_generation
        ),
        "resolver_output_preserved": (
            resolution_before_generation == resolution_after_generation
        ),
        "ambiguity_preserved": _ambiguity_preserved(agent_result, generated),
        "no_latest_period_selection": _no_latest_selection(agent_result, generated),
        "provenance_preserved": _provenance_preserved(
            agent_result.answer_draft, generated
        ),
        "answer_draft_not_mutated": (
            draft_before_generation == draft_after_generation
        ),
        "citations_have_provenance": _citations_have_provenance(
            agent_result.answer_draft, generated
        ),
        "no_evidence_state_consistent": _no_evidence_state_consistent(
            agent_result.answer_draft, generated
        ),
        "unsupported_facts_not_generated": _unsupported_facts_absent(
            agent_result.answer_draft, generated
        ),
    }
    validations["all_invariants_preserved"] = all(validations.values())
    failed_invariants = _failed_invariants(validations)

    warnings = tuple(
        dict.fromkeys([*agent_result.warnings, *generated.warnings])
    )
    trace = (*agent_result.execution_trace, "answer_generator")
    return {
        "question": question,
        "status": "ok",
        "task_decision": {
            "task_type": agent_result.task_decision.task_type,
            "resolver_type": agent_result.task_decision.resolver_type,
            "confidence": agent_result.task_decision.confidence,
            "matched_signals": list(agent_result.task_decision.matched_signals),
            "warnings": list(agent_result.task_decision.warnings),
        },
        "execution_trace": list(trace),
        "answerable": generated.answerable,
        "citation_count": len(generated.citations),
        "warnings": list(warnings),
        "failed_invariants": failed_invariants,
        "evidence": {
            "raw_candidate_count": agent_result.evidence_set.raw_candidate_count,
            "selected_evidence_count": agent_result.evidence_set.selected_evidence_count,
            "evidence_group_count": len(agent_result.evidence_set.evidence_groups),
            "ambiguity": copy.deepcopy(dict(agent_result.evidence_set.ambiguity)),
        },
        "resolution": (
            agent_result.resolution.to_dict()
            if agent_result.resolution is not None
            else None
        ),
        "answer_draft": agent_result.answer_draft.to_dict(),
        "generated_answer": generated.to_dict(),
        "validation": validations,
    }


def run_production_smoke(
    questions: Sequence[str],
    *,
    output_path: Path = DEFAULT_OUTPUT,
    top_k: int = 10,
    lexical_top_n: int = 50,
    vector_top_n: int = 50,
) -> dict[str, Any]:
    """Read from the configured production retrieval environment and write one report."""

    embedding_config = EmbeddingConfig.from_env()
    embedder = create_embedding_provider(embedding_config)
    backend = PostgresBackend()
    understanding = QueryUnderstanding(company_resolver=backend.resolve_company)
    executor = HybridQueryExecutor(
        backend,
        embedder,
        embedding_config,
        config=HybridRetrievalConfig(
            lexical_top_n=lexical_top_n,
            vector_top_n=vector_top_n,
            final_top_k=top_k,
            rerank_mode="legacy",
        ),
    )

    return run_smoke_pipeline(
        questions,
        understanding=understanding,
        executor=executor,
        output_path=output_path,
        top_k=top_k,
    )


def run_smoke_pipeline(
    questions: Sequence[str],
    *,
    understanding: Any,
    executor: Any,
    output_path: Path | None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Execute the full pipeline with injectable read-only front-end components."""

    rows: list[dict[str, Any]] = []
    for question in questions:
        stage = "query_understanding"
        try:
            plan = understanding.understand(question, top_k=top_k)
            stage = "hybrid_retrieval"
            execution = executor.execute(plan)
            stage = "reasoning_and_generation"
            rows.append(validate_completed_execution(question, plan, execution))
        except Exception as error:  # report every query without leaking credentials/data
            rows.append(
                {
                    "question": question,
                    "status": "failed",
                    "failure_stage": stage,
                    "error_type": type(error).__name__,
                    "task_decision": None,
                    "execution_trace": [],
                    "answerable": False,
                    "citation_count": 0,
                    "warnings": [f"pipeline_failed:{stage}"],
                    "failed_invariants": [f"pipeline_failed:{stage}"],
                    "validation": {"all_invariants_preserved": False},
                }
            )

    successful = sum(row["status"] == "ok" for row in rows)
    report = {
        "query_count": len(rows),
        "successful_query_count": successful,
        "failed_query_count": len(rows) - successful,
        "all_invariants_preserved": bool(rows)
        and successful == len(rows)
        and all(
            row.get("validation", {}).get("all_invariants_preserved") is True
            for row in rows
        ),
        "invariant_failures": [
            {
                "question": row["question"],
                "failed_invariants": list(row.get("failed_invariants") or ()),
            }
            for row in rows
            if row.get("failed_invariants")
        ],
        "queries": rows,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    return report


def _retrieval_snapshot(execution: Any) -> dict[str, Any]:
    chunks = list(getattr(execution, "chunks", ()))
    results = list(getattr(execution, "results", ()))
    return {
        "candidate_identity": [
            {"chunk_id": row.chunk_id, "doc_id": row.doc_id} for row in chunks
        ],
        "candidate_payload_digest": [
            _digest(
                {
                    "chunk": copy.deepcopy(dict(row.chunk)),
                    "metadata_match": _serialize(row.metadata_match),
                }
            )
            for row in chunks
        ],
        "result_order": [
            {"rank": row.rank, "chunk_id": row.chunk_id, "doc_id": row.doc_id}
            for row in results
        ],
        "result_scores": [
            {
                "chunk_id": row.chunk_id,
                "bm25_score": row.bm25_score,
                "metadata_match": _serialize(row.metadata_match),
            }
            for row in results
        ],
    }


def _object_snapshot(value: Any) -> Any:
    if value is None:
        return None
    return copy.deepcopy(_serialize(value))


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _serialize(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ambiguity_preserved(agent_result: Any, generated: Any) -> bool:
    resolution = agent_result.resolution
    if resolution is None:
        draft_preserved = (
            agent_result.answer_draft.ambiguity.get("automatically_resolved") is False
        )
    else:
        expected = bool(getattr(resolution, "temporal_ambiguity", False))
        if getattr(resolution, "matching_event_count", 0) > 1:
            expected = True
        draft_preserved = (
            bool(agent_result.answer_draft.ambiguity.get("temporal_ambiguity"))
            == expected
        )
    ambiguity_warnings = (
        warning
        for warning in agent_result.answer_draft.warnings
        if "ambigu" in warning or warning.startswith("multiple_matching_")
    )
    return draft_preserved and all(
        warning in generated.warnings for warning in ambiguity_warnings
    )


def _no_latest_selection(agent_result: Any, generated: Any) -> bool:
    ambiguity = agent_result.answer_draft.ambiguity
    if ambiguity.get("latest_event_selected") is not False and (
        "latest_event_selected" in ambiguity
    ):
        return False
    if ambiguity.get("latest_period_selected") is not False and (
        "latest_period_selected" in ambiguity
    ):
        return False
    visible_values = _values_requiring_preservation(agent_result.answer_draft)
    return all(value in generated.answer_text for value in visible_values)


def _values_requiring_preservation(draft: Any) -> tuple[str, ...]:
    values: list[str] = []
    for section in draft.answer_sections:
        content = section.content
        for event in _mapping_sequence(content.get("events")):
            value = event.get("reference_date")
            if value:
                values.append(str(value))
        fact = content.get("fact")
        if isinstance(fact, Mapping):
            fact_text = fact.get("fact_text")
            if fact_text:
                values.append(str(fact_text))
            for period in _mapping_sequence(fact.get("reporting_periods")):
                year = period.get("fiscal_year") or period.get("year")
                if year:
                    values.append(str(year))
    return tuple(dict.fromkeys(values))


def _provenance_preserved(draft: Any, generated: Any) -> bool:
    draft_by_key = {
        (citation.chunk_id, citation.doc_id): citation for citation in draft.citations
    }
    generated_keys = [
        (citation.chunk_id, citation.doc_id) for citation in generated.citations
    ]
    expected_keys = [
        (citation.chunk_id, citation.doc_id)
        for citation in draft.citations
        if citation.provenance_path
    ]
    if generated_keys != expected_keys:
        return False
    return all(
        bool(draft_by_key[key].provenance_path)
        and _serialize(draft_by_key[key].source_refs)
        == _serialize(citation.source_refs)
        for key, citation in zip(generated_keys, generated.citations)
    )


def _citations_have_provenance(draft: Any, generated: Any) -> bool:
    draft_by_key = {
        (citation.chunk_id, citation.doc_id): citation for citation in draft.citations
    }
    return all(
        (citation.chunk_id, citation.doc_id) in draft_by_key
        and bool(draft_by_key[(citation.chunk_id, citation.doc_id)].provenance_path)
        and _serialize(
            draft_by_key[(citation.chunk_id, citation.doc_id)].source_refs
        )
        == _serialize(citation.source_refs)
        for citation in generated.citations
    )


def _no_evidence_state_consistent(draft: Any, generated: Any) -> bool:
    if generated.answerable or generated.citations:
        return True
    return bool(
        not draft.evidence_references
        and not draft.citations
        and "answer_not_supported" in generated.warnings
    )


def _unsupported_facts_absent(draft: Any, generated: Any) -> bool:
    draft_text = json.dumps(draft.to_dict(), ensure_ascii=False, default=str)
    generated_sections = "\n".join(section.content for section in generated.sections)
    for phrase in _UNSUPPORTED_ADDITIONS:
        if phrase in generated.answer_text and phrase not in draft_text:
            return False
    for pattern in _FACT_PATTERNS:
        for match in pattern.findall(generated_sections):
            if not _fact_token_supported(str(match).strip(), draft_text):
                return False
    valid_markers = {citation.citation_id for citation in generated.citations}
    for section in generated.sections:
        for line in section.content.splitlines():
            if any(pattern.search(line) for pattern in _FACT_PATTERNS):
                if not any(marker in line for marker in valid_markers):
                    return False
    return True


def _fact_token_supported(token: str, draft_text: str) -> bool:
    if token in draft_text:
        return True
    variants = {
        token.rstrip("%"),
        token.replace(",", ""),
        token.rstrip("%").replace(",", ""),
    }
    year = re.match(r"((?:19|20)\d{2})년", token)
    if year:
        variants.add(year.group(1))
    return any(value and value in draft_text for value in variants)


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _failed_invariants(validations: Mapping[str, Any]) -> list[str]:
    return sorted(
        name
        for name, passed in validations.items()
        if name != "all_invariants_preserved" and passed is not True
    )


def _concise_report(report: Mapping[str, Any], output_path: Path) -> dict[str, Any]:
    return {
        "query_count": report["query_count"],
        "successful_query_count": report["successful_query_count"],
        "failed_query_count": report["failed_query_count"],
        "all_invariants_preserved": report["all_invariants_preserved"],
        "invariant_failures": report.get("invariant_failures", []),
        "queries": [
            {
                "question": row["question"],
                "status": row["status"],
                "task_type": (row.get("task_decision") or {}).get("task_type"),
                "answerable": row.get("answerable"),
                "citation_count": row.get("citation_count"),
                "warnings": row.get("warnings", []),
                "failed_invariants": row.get("failed_invariants", []),
            }
            for row in report["queries"]
        ],
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only end-to-end Festival agent smoke validation."
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Repeat to replace the three representative default queries.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lexical-top-n", type=int, default=50)
    parser.add_argument("--vector-top-n", type=int, default=50)
    args = parser.parse_args()

    questions = tuple(args.queries or DEFAULT_QUERIES)
    report = run_production_smoke(
        questions,
        output_path=args.output,
        top_k=args.top_k,
        lexical_top_n=args.lexical_top_n,
        vector_top_n=args.vector_top_n,
    )
    print(
        json.dumps(
            _concise_report(report, args.output),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["all_invariants_preserved"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
