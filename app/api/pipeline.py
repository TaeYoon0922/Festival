"""Serving path for ``GET /answer``.

The wiring here is the one the frozen Gold60 evaluation exercises:
``QueryUnderstanding`` produces a plan, ``HybridQueryExecutor`` retrieves, the
``AgentOrchestrator`` reasons, and ``CitationAwareAnswerGenerator`` renders.  The
only addition is the constrained HyperCLOVA X verbalizer, which restates the
rendered answer and falls back to it whenever validation fails.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import psycopg

from app.agent.orchestrator import AgentOrchestrator
from app.api.settings import ApiSettings
from app.generation.answer_generator import (
    CitationAwareAnswerGenerator,
    GeneratedAnswer,
)
from app.generation.hcx_verbalizer import (
    SKIPPED_MULTI_EVENT_CLAIM,
    SKIPPED_NO_COMPACT_CLAIM,
    HcxVerbalizer,
    VerbalizationOutcome,
)
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.router import QueryRouter
from app.retrieval.correction_expansion import build_default_expander
from app.retrieval.event_expansion import build_default_event_expander
from app.retrieval.corporate_event_repository import (
    PostgresCorporateEventRepository,
)
from app.retrieval.correction_repository import PostgresCorrectionRepository
from app.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingHttpError,
    create_embedding_provider,
)
from app.retrieval.hybrid import HybridQueryExecutor
from app.retrieval.postgres_backend import PostgresBackend


#: Public failure reasons.  Messages never carry a DSN, key, or traceback.
DATABASE_UNAVAILABLE = "database_unavailable"
EMBEDDING_UNAVAILABLE = "embedding_unavailable"
INTERNAL_ERROR = "internal_error"

#: The contract promises answer text on every 200, including when nothing in
#: the corpus supports the question.
EMPTY_ANSWER_FALLBACK = "확인되지 않은 정보가 있습니다."

#: Statuses reached without ever calling the model.
_HCX_NOT_CALLED = frozenset(
    {
        "disabled",
        "not_configured",
        "skipped_not_answerable",
        SKIPPED_NO_COMPACT_CLAIM,
        SKIPPED_MULTI_EVENT_CLAIM,
    }
)

_PUBLIC_MESSAGES = {
    DATABASE_UNAVAILABLE: "The disclosure database is unavailable.",
    EMBEDDING_UNAVAILABLE: "The embedding service is unavailable.",
    INTERNAL_ERROR: "The answer pipeline failed to complete.",
}


class AnswerPipelineError(RuntimeError):
    """A serving failure reduced to a reason the client may safely see."""

    def __init__(self, reason: str) -> None:
        super().__init__(_PUBLIC_MESSAGES.get(reason, _PUBLIC_MESSAGES[INTERNAL_ERROR]))
        self.reason = reason

    @property
    def public_message(self) -> str:
        return str(self)


class AnswerPipeline:
    """Answer one question through the frozen retrieval and reasoning stack."""

    def __init__(
        self,
        *,
        understanding: Any,
        executor: Any,
        settings: ApiSettings | None = None,
        orchestrator: AgentOrchestrator | None = None,
        generator: CitationAwareAnswerGenerator | None = None,
        verbalizer: HcxVerbalizer | None = None,
    ) -> None:
        self.settings = settings or ApiSettings()
        self.understanding = understanding
        self.executor = executor
        self.orchestrator = orchestrator or AgentOrchestrator()
        self.generator = generator or CitationAwareAnswerGenerator()
        self.verbalizer = verbalizer or HcxVerbalizer()

    @classmethod
    def from_env(cls) -> "AnswerPipeline":
        """Build the serving stack.  No connection is opened until a request."""

        settings = ApiSettings.from_env()
        try:
            embedding_config = EmbeddingConfig.from_env()
            embedder = create_embedding_provider(embedding_config)
        except Exception as error:  # noqa: BLE001 - surfaced as a sanitized reason
            raise AnswerPipelineError(EMBEDDING_UNAVAILABLE) from error

        # An unreachable database must fail fast instead of hanging a request.
        backend = PostgresBackend(
            connect_kwargs={
                "connect_timeout": settings.db_connect_timeout_seconds
            }
        )
        correction_repository = PostgresCorrectionRepository(backend)
        event_repository = PostgresCorporateEventRepository(backend)
        return cls(
            settings=settings,
            understanding=QueryUnderstanding(
                company_resolver=backend.resolve_company
            ),
            executor=HybridQueryExecutor(
                backend,
                embedder,
                embedding_config,
                # Correction policy reads the persisted graph instead of the
                # is_correction flag alone.  A database without the correction
                # tables degrades to the previous behaviour rather than failing.
                router=QueryRouter(correction_graph=correction_repository),
                # A question about the final version or the history of a report
                # needs documents the question's own date window excludes, so
                # the resolved chain is added back after retrieval.
                correction_expander=build_default_expander(
                    correction_repository, backend, backend, backend
                ),
                # A contract question needs the rest of that contract's
                # lifecycle: the termination of a contract that was retrieved, or
                # the contract behind a termination that was. This executor is
                # the single owner of event expansion on the serving path; the
                # graph is followed once, never searched again. A database
                # without db/007 degrades to the previous behaviour.
                event_expander=build_default_event_expander(
                    event_repository, backend, backend, backend
                ),
                config=settings.retrieval_config(),
            ),
        )

    def answer(self, question_id: str, question: str) -> dict[str, Any]:
        plan, execution = self._retrieve(question)
        try:
            result = self.orchestrator.run(question, plan, execution)
            generated = self.generator.generate(result.answer_draft)
        except Exception as error:  # noqa: BLE001 - surfaced as a sanitized reason
            raise AnswerPipelineError(_classify(error)) from error

        outcome = self.verbalizer.verbalize(
            generated,
            draft=result.answer_draft,
            resolution=result.resolution,
            task_type=result.task_decision.task_type,
        )
        return {
            "question_id": question_id,
            "question": question,
            "retrieved_context": retrieved_context(
                execution, self.settings.top_k + _expanded_count(execution)
            ),
            "think_trace": think_trace(result, generated, outcome, execution),
            "answer": _non_empty(outcome.text, generated.answer_text),
        }

    def _retrieve(self, question: str) -> tuple[Any, Any]:
        try:
            plan = self.understanding.understand(question, top_k=self.settings.top_k)
            return plan, self.executor.execute(plan)
        except Exception as error:  # noqa: BLE001 - surfaced as a sanitized reason
            raise AnswerPipelineError(_classify(error)) from error


def _expanded_count(execution: Any) -> int:
    """How many rows the correction graph contributed to this execution.

    The served Top-K stays the retrieved Top-K; relation-derived rows are added
    on top of it so a correction the graph supplied is never truncated away by
    the very limit it was appended after.
    """

    total = 0
    for attribute, key in (
        ("correction_expansion", "correction_added_result_count"),
        ("event_expansion", "event_added_result_count"),
    ):
        trace = getattr(execution, attribute, None)
        if not isinstance(trace, Mapping):
            continue
        try:
            total += max(int(trace.get(key) or 0), 0)
        except (TypeError, ValueError):
            continue
    return total


def retrieved_context(execution: Any, limit: int) -> list[dict[str, Any]]:
    """Render the served Top-K in the shape the Gold60 report records."""

    candidates = {candidate.chunk_id: candidate for candidate in execution.chunks}
    rows: list[dict[str, Any]] = []
    for result in list(execution.results)[:limit]:
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


def think_trace(
    result: Any,
    generated: GeneratedAnswer,
    outcome: VerbalizationOutcome,
    execution: Any,
) -> dict[str, Any]:
    """Summarize what the pipeline executed.

    This is an execution summary, not reasoning.  Component names, counts, and
    statuses only: no intermediate deliberation is exposed.
    """

    stages = [*result.execution_trace, "answer_generator"]
    if outcome.status not in _HCX_NOT_CALLED:
        stages.append("hcx_verbalizer")
    trace = {
        "task_type": result.task_decision.task_type,
        "route": _route(result),
        "stages": stages,
        "retrieval_count": len(list(execution.results)),
        "selected_evidence_count": result.evidence_set.selected_evidence_count,
        "answerable": generated.answerable,
        "warnings": list(generated.warnings),
        "hcx_status": outcome.status,
    }
    correction = getattr(execution, "correction_expansion", None)
    if isinstance(correction, Mapping) and correction.get("correction_expanded"):
        # Which documents the graph added and which chain they came from. This
        # is an execution summary, not reasoning.
        # Expansion runs inside retrieval, before the orchestrator's stages.
        stages.insert(0, "correction_expansion")
        trace["correction"] = {
            key: correction.get(key)
            for key in (
                "correction_intent",
                "correction_group_id",
                "correction_root_doc_id",
                "correction_latest_doc_id",
                "correction_added_doc_ids",
            )
        }
    event = getattr(execution, "event_expansion", None)
    if isinstance(event, Mapping) and event.get("event_expanded"):
        # Which lifecycle contributed which filings. An execution summary, not
        # reasoning. Expansion runs inside retrieval, after correction expansion.
        stages.insert(0, "corporate_event_expansion")
        trace["corporate_event"] = event.get("corporate_event_expansion")
    return trace


def _route(result: Any) -> str:
    for stage in result.execution_trace:
        if stage.endswith("_resolver"):
            return stage
    return "general_evidence"


def _chunk_period(chunk: Mapping[str, Any]) -> dict[str, Any]:
    period = chunk.get("period")
    if isinstance(period, Mapping):
        return copy.deepcopy(dict(period))
    return {
        key: chunk.get(key)
        for key in ("base_year", "base_month", "fiscal_year", "quarter", "period_type")
        if chunk.get(key) is not None
    }


def _non_empty(*candidates: str) -> str:
    for value in candidates:
        if value and value.strip():
            return value
    return EMPTY_ANSWER_FALLBACK


def _classify(error: Exception) -> str:
    if isinstance(error, AnswerPipelineError):
        return error.reason
    if isinstance(error, EmbeddingHttpError):
        return EMBEDDING_UNAVAILABLE
    if isinstance(error, psycopg.Error):
        return DATABASE_UNAVAILABLE
    return INTERNAL_ERROR
