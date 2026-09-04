"""Serving path for ``GET /answer``.

The wiring here is the one the frozen Gold60 evaluation exercises:
``QueryUnderstanding`` produces a plan, ``HybridQueryExecutor`` retrieves, the
``AgentOrchestrator`` reasons, and ``CitationAwareAnswerGenerator`` renders.  The
only addition is the constrained HyperCLOVA X verbalizer, which restates the
rendered answer and falls back to it whenever validation fails.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import psycopg

from app.agent.orchestrator import AgentOrchestrator
from app.api.settings import ApiSettings
from app.generation.answer_generator import (
    CitationAwareAnswerGenerator,
    GeneratedAnswer,
    GeneratedCitation,
)
from app.generation.hcx_verbalizer import (
    SKIPPED_MULTI_EVENT_CLAIM,
    SKIPPED_NO_COMPACT_CLAIM,
    HcxSettings,
    HcxVerbalizer,
    VerbalizationOutcome,
)
from app.reasoning.answerability import (
    AnswerabilityGuard,
    AnswerabilityResult,
    AnswerabilityStatus,
    guarded_answer_text,
)
from app.reasoning.clarification_candidates import (
    apply_resolved_candidate,
    execution_clarification_request,
    validation_clarification_request,
)
from app.reasoning.clarification_request import (
    EVENT_INSTANCE,
    ClarificationDecision,
    ClarificationState,
    clarification_text,
)
from app.reasoning.clarification_resolver import ClarificationResolver
from app.reasoning.hcx_clarification_classifier import HcxClarificationClassifier
from app.reasoning.multi_document_evidence import (
    MultiDocumentEvidence,
    MultiDocumentEvidenceBuilder,
)
from app.reasoning.multi_document_executor import MultiDocumentExecutor
from app.reasoning.multi_document_planner import MultiDocumentPlanner
from app.reasoning.multi_document_semantics import check_answer
from app.reasoning.holding_report_relative_execution import (
    HoldingReportRelativeExecution,
)
from app.reasoning.holding_company_role_resolution import (
    HoldingCompanyRoleResolver,
)
from app.reasoning.comparison_evidence import (
    evidence_comparison,
    execute_per_company,
    merge_executions,
)
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import (
    CorpusScope,
    PostgresEventScopeProvider,
    QueryState,
    QueryValidationResult,
    QueryValidator,
)
from app.reasoning.router import QueryRouter
from app.reasoning.company_comparison import (
    COMPARISON_OUTCOME_KEY,
    comparison_outcome,
    executable_comparison,
    executions_for,
)
from app.reasoning.semantic_query_fallback import HcxSemanticQueryFallback
from app.retrieval.correction_expansion import (
    apply_expansion,
    build_default_expander,
)
from app.retrieval.event_expansion import build_default_event_expander
from app.retrieval.corporate_event_repository import (
    PostgresCorporateEventRepository,
)
from app.retrieval.correction_repository import PostgresCorrectionRepository
from app.reasoning.vector_coverage_policy import (
    HEALTHY as _VECTOR_HEALTHY,
    classify as _classify_vector_availability,
    warning_for as _vector_warning_for,
)
from app.retrieval.embeddings import (
    EmbeddingConfig,
    EmbeddingHttpError,
    create_embedding_provider,
)
from app.retrieval.hybrid import HybridQueryExecutor
from app.retrieval.interfaces import RetrievalResult
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
        "skipped_query_not_resolved",
        "skipped_answerability_guard",
        "skipped_clarification",
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
        multi_document_planner: MultiDocumentPlanner | None = None,
        multi_document_executor: MultiDocumentExecutor | None = None,
        multi_document_evidence: MultiDocumentEvidenceBuilder | None = None,
        query_validator: QueryValidator | None = None,
        semantic_fallback: HcxSemanticQueryFallback | None = None,
        answerability_guard: AnswerabilityGuard | None = None,
        clarification_resolver: ClarificationResolver | None = None,
    ) -> None:
        self.settings = settings or ApiSettings()
        self.understanding = understanding
        self.executor = executor
        self.orchestrator = orchestrator or AgentOrchestrator()
        self.generator = generator or CitationAwareAnswerGenerator()
        self.verbalizer = verbalizer or HcxVerbalizer()
        # P0-C is additive and opt-in. Without an executor wired the pipeline
        # behaves exactly as it did before, which is what the frozen Gold60
        # path depends on.
        self.multi_document_planner = multi_document_planner
        self.multi_document_executor = multi_document_executor
        self.multi_document_evidence = multi_document_evidence
        # Optional at construction for byte-for-byte compatibility with the
        # frozen unit fixtures. ``from_env`` always wires the complete P0-D
        # firewall used by the real serving path.
        self.query_validator = query_validator
        self.semantic_fallback = semantic_fallback
        self.answerability_guard = answerability_guard
        self.clarification_resolver = clarification_resolver
        self._query_metrics = {
            "deterministic_resolved_count": 0,
            "hcx_fallback_count": 0,
            "clarification_count": 0,
            "unsupported_count": 0,
            "out_of_scope_count": 0,
        }

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
        multi_document_planner = MultiDocumentPlanner()
        hcx_settings = HcxSettings.from_env()
        corpus_scope = CorpusScope.repository_default()
        report_relative_execution = HoldingReportRelativeExecution.from_repository(
            document_backend=backend,
            chunk_backend=backend,
        )
        holding_company_role_resolver = HoldingCompanyRoleResolver(
            report_relative_execution.index,
            active_corpus_identity=report_relative_execution.active_corpus_identity,
        )
        return cls(
            settings=settings,
            understanding=QueryUnderstanding(
                corpus_scope.company_aliases() if corpus_scope else None,
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
            orchestrator=AgentOrchestrator(
                report_relative_execution=report_relative_execution,
                holding_company_role_resolver=holding_company_role_resolver,
                holding_report_index=report_relative_execution.index,
                active_corpus_identity=(
                    report_relative_execution.active_corpus_identity
                ),
            ),
            # P0-C: a deterministic completeness layer on top of the frozen
            # ranking. It engages only for questions that name a company, an
            # enumerable family, a bounded period, and an explicit date basis;
            # everything else takes the path above unchanged.
            verbalizer=HcxVerbalizer(hcx_settings),
            multi_document_planner=multi_document_planner,
            multi_document_executor=MultiDocumentExecutor(
                event_repository=event_repository,
                correction_repository=correction_repository,
                disclosure_backend=backend,
            ),
            multi_document_evidence=MultiDocumentEvidenceBuilder(
                event_repository=event_repository,
                metadata_backend=backend,
                chunk_backend=backend,
                retriever=backend,
            ),
            query_validator=QueryValidator(
                corpus_scope=corpus_scope,
                multi_document_planner=multi_document_planner,
                event_scope_provider=PostgresEventScopeProvider(backend),
                holding_company_role_resolver=holding_company_role_resolver,
            ),
            semantic_fallback=HcxSemanticQueryFallback(hcx_settings),
            answerability_guard=AnswerabilityGuard(),
            clarification_resolver=ClarificationResolver(
                HcxClarificationClassifier(hcx_settings)
            ),
        )

    def answer(self, question_id: str, question: str) -> dict[str, Any]:
        validation: QueryValidationResult | None = None
        clarification_decision: ClarificationDecision | None = None
        if self.query_validator is None:
            plan, execution = self._retrieve(question)
        else:
            plan, validation = self._validated_understanding(question)
            # A question naming several companies cannot resolve the
            # single-company slot retrieval requires, so validation blocks it and
            # asks which company was meant -- of an asker who named two. Before
            # treating that as an unanswered question, check whether it is a
            # comparison this pipeline can retrieve company by company.
            comparison_evidence = None
            if not validation.retrieval_allowed:
                candidate = evidence_comparison(validation.plan)
                comparison_evidence = candidate if candidate.applied else None
            if not validation.retrieval_allowed and comparison_evidence is None:
                request = (
                    validation_clarification_request(question, validation)
                    if self.clarification_resolver is not None
                    else None
                )
                if request is not None:
                    clarification_decision = self.clarification_resolver.resolve(request)
                    if clarification_decision.state is ClarificationState.RESOLVED:
                        updated_plan = apply_resolved_candidate(
                            plan, request, clarification_decision
                        )
                        revalidated = self.query_validator.validate(updated_plan)
                        if revalidated.retrieval_allowed:
                            self._query_metrics["clarification_count"] = max(
                                0, self._query_metrics["clarification_count"] - 1
                            )
                            plan, validation = revalidated.plan, revalidated
                        else:
                            clarification_decision = ClarificationDecision(
                                state=ClarificationState.CLARIFY,
                                reason="resolved_candidate_failed_revalidation",
                                candidates=request.candidates,
                                classifier_status=clarification_decision.classifier_status,
                                truncated=request.truncated,
                            )
                    if not validation.retrieval_allowed:
                        self._adjust_metrics_for_clarification(clarification_decision)
                        return self._blocked_response(
                            question_id,
                            question,
                            validation,
                            clarification_decision=clarification_decision,
                        )
                if not validation.retrieval_allowed:
                    return self._blocked_response(question_id, question, validation)
            try:
                if comparison_evidence is not None:
                    # Each company retrieved on its own plan, then interleaved,
                    # so what reaches evidence building is the ordinary shape and
                    # every named company sits inside the first few positions.
                    plan = validation.plan
                    execution = merge_executions(
                        plan,
                        comparison_evidence,
                        execute_per_company(
                            comparison_evidence, plan, self.executor.execute
                        ),
                    )
                else:
                    comparison = executable_comparison(plan)
                    execution = (
                        self._comparison_execution(comparison, plan)
                        if comparison is not None
                        else self.executor.execute(plan)
                    )
            except Exception as error:  # noqa: BLE001 - sanitized at API boundary
                raise AnswerPipelineError(_classify(error)) from error
        # P0-C runs after retrieval so it can only add completeness evidence on
        # top of the frozen ranking, never replace it. A question P0-C declines
        # -- which is every Gold60 question -- takes exactly the path it always
        # did, including an unchanged think_trace.
        multi = self._multi_document(question, plan, execution)
        try:
            result = self.orchestrator.run(
                question, plan, execution, multi_document=multi
            )
            generated = self.generator.generate(result.answer_draft)
        except Exception as error:  # noqa: BLE001 - surfaced as a sanitized reason
            raise AnswerPipelineError(_classify(error)) from error

        # One evidence set from here on: whatever the orchestrator resolved,
        # composed and cited from is what the guard judges and what the caller
        # is shown.  ``execution`` stays the immutable retrieval record and is
        # still what the retrieval diagnostics describe.
        evidence = final_evidence(execution, result)

        answerability: AnswerabilityResult | None = None
        if self.answerability_guard is not None:
            answerability = self.answerability_guard.evaluate(
                generated,
                plan=plan,
                agent_result=result,
                execution=evidence,
                multi_document=multi,
            )
        if self.clarification_resolver is not None:
            request = execution_clarification_request(
                question,
                plan,
                result,
                execution,
                multi_document=multi,
            )
            if request is not None:
                post_decision = self.clarification_resolver.resolve(request)
                if post_decision.state is not ClarificationState.RESOLVED:
                    if post_decision.state is ClarificationState.CLARIFY:
                        self._query_metrics["clarification_count"] += 1
                    elif post_decision.state is ClarificationState.UNSUPPORTED:
                        self._query_metrics["unsupported_count"] += 1
                    return self._post_resolution_clarification_response(
                        question_id,
                        question,
                        result=result,
                        generated=generated,
                        execution=execution,
                        multi_document=multi,
                        validation=validation,
                        answerability=answerability,
                        decision=post_decision,
                    )
        if answerability is None or answerability.model_answer_allowed:
            outcome = self.verbalizer.verbalize(
                generated,
                draft=result.answer_draft,
                resolution=result.resolution,
                task_type=result.task_decision.task_type,
            )
            outcome = _preserve_multi_document_semantics(outcome, generated, multi)
        else:
            outcome = VerbalizationOutcome(
                guarded_answer_text(
                    answerability,
                    generated.answer_text,
                    multi_document=multi,
                ),
                "skipped_answerability_guard",
                answerability.status.value,
            )
        public_context = retrieved_context(
            evidence,
            self.settings.top_k
            + _expanded_count(execution)
            + _multi_document_count(multi),
        )
        public_answer, citation_alignment = align_public_citations(
            _non_empty(outcome.text, generated.answer_text),
            generated.citations,
            public_context,
        )
        trace = think_trace(
            result,
            generated,
            outcome,
            execution,
            multi_document=multi,
            query_validation=validation,
            answerability=answerability,
            clarification_decision=clarification_decision,
        )
        if citation_alignment["unmapped"]:
            trace["warnings"] = list(
                dict.fromkeys(
                    [*trace["warnings"], "citation_alignment_unmapped"]
                )
            )
        return {
            "question_id": question_id,
            "question": question,
            "retrieved_context": public_context,
            "think_trace": trace,
            "answer": _non_empty(public_answer),
        }

    @property
    def query_metrics(self) -> dict[str, int]:
        """A copy of the bounded, query-level P0-D counters."""

        return dict(self._query_metrics)

    def _comparison_execution(self, comparison: Any, plan: Any) -> Any:
        """One execution per company, merged into the execution shape downstream
        already understands.

        Merging rather than answering here is deliberate: evidence building,
        citation alignment, answerability and the think trace are all written
        against a single execution, and a comparison that produced its own
        response object would have to reimplement every one of them. Each
        company is still retrieved on its own narrowed plan, so what is merged
        is four separate answers to four separate questions, not one search over
        four companies.
        """

        from itertools import zip_longest

        from app.reasoning.query_plan import QueryExecution
        from app.retrieval.interfaces import RetrievalResult

        documents: list[Any] = []
        seen_documents: set[str] = set()
        per_company_results: list[list[Any]] = []
        candidates: dict[str, Any] = {}
        executions = executions_for(comparison, plan, self.executor.execute)
        # Each operand is decided inside its own company's execution, where
        # relevance still means that company's clause and date. Once the
        # executions are merged for evidence and citation, that scoping is gone,
        # and a company with several amount-bearing documents would no longer
        # resolve to one.
        outcome = comparison_outcome(comparison, executions)
        for execution in executions.values():
            if execution is None:
                continue
            for document in getattr(execution, "documents", ()) or ():
                if str(document.doc_id) not in seen_documents:
                    seen_documents.add(str(document.doc_id))
                    documents.append(document)
            for chunk in getattr(execution, "chunks", ()) or ():
                chunk_id, _doc_id = _chunk_identity(chunk)
                if chunk_id:
                    # The candidate itself, not a mapping made from it: every
                    # stage after this one reads ``CandidateChunk`` attributes,
                    # and flattening here would hand them a shape they do not
                    # expect. Only identity is read out; the object is kept.
                    candidates.setdefault(chunk_id, chunk)
            per_company_results.append(
                list(getattr(execution, "results", ()) or ())
            )

        # Interleave by rank instead of concatenating one company after
        # another. Concatenation starves every company but the first: each
        # retrieval returns ten results for its own company, so the second
        # company's best document sits at merged position eleven and never
        # survives the evidence limit -- which is how a two-company comparison
        # came back citing one company twice. Taking each company's best first
        # puts every operand's own top hit inside the first N positions, for any
        # number of companies. Each company's internal order is untouched.
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

        # Ranks came from separate retrievals and would otherwise collide;
        # renumbering states the merged order the evidence builder will read.
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
        # Chunks follow the merged result order, so evidence is built in the
        # same sequence the ranks describe.
        chunks = [
            candidates[result.chunk_id]
            for result in ranked
            if result.chunk_id in candidates
        ]
        chunks.extend(
            candidate
            for chunk_id, candidate in candidates.items()
            if chunk_id not in seen_results
        )
        # The subexecutions were a search for the operands, not the answer's
        # evidence. Once every operand is chosen, the rest of what retrieval
        # returned is noise for this question: it cannot be cited, because the
        # comparison rests on exactly one document per company, and carrying it
        # forward leaves an answer whose evidence and citations disagree about
        # how complete it is. So the served evidence becomes the operands
        # themselves, in the order the question put its companies in.
        if outcome.get("resolved"):
            selected = [
                str(entry.get("chunk_id") or "")
                for entry in outcome.get("operands") or ()
            ]
            by_result = {result.chunk_id: result for result in ranked}
            ranked = tuple(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=(
                        by_result[chunk_id].doc_id
                        if chunk_id in by_result
                        else _chunk_identity(candidates[chunk_id])[1]
                    ),
                    bm25_score=(
                        by_result[chunk_id].bm25_score
                        if chunk_id in by_result
                        else 0.0
                    ),
                    rank=index + 1,
                    metadata_match=(
                        by_result[chunk_id].metadata_match
                        if chunk_id in by_result
                        else getattr(candidates[chunk_id], "metadata_match", {})
                    ),
                )
                for index, chunk_id in enumerate(selected)
                if chunk_id in candidates
            )
            chunks = [candidates[result.chunk_id] for result in ranked]
        return QueryExecution(
            plan=plan,
            documents=tuple(documents),
            chunks=tuple(chunks),
            results=ranked,
            routing={COMPARISON_OUTCOME_KEY: outcome},
        )

    def _validated_understanding(
        self, question: str
    ) -> tuple[Any, QueryValidationResult]:
        try:
            plan = self.understanding.understand(
                question, top_k=self.settings.top_k
            )
            validation = self.query_validator.validate(plan)
        except Exception as error:  # noqa: BLE001 - sanitized at API boundary
            raise AnswerPipelineError(_classify(error)) from error

        if validation.state is QueryState.RESOLVED:
            self._query_metrics["deterministic_resolved_count"] += 1
            return validation.plan, validation

        if validation.fallback_recommended and self.semantic_fallback is not None:
            outcome = self.semantic_fallback.interpret(question, validation)
            if outcome.status not in {"not_needed", "disabled", "not_configured"}:
                self._query_metrics["hcx_fallback_count"] += 1
            if outcome.succeeded:
                validation = self.query_validator.validate(
                    plan,
                    semantic=outcome.result,
                    fallback_used=True,
                    fallback_status=outcome.status,
                    hcx_elapsed_ms=outcome.elapsed_ms,
                    hcx_diagnostic=outcome.diagnostic(),
                )
            else:
                validation = validation.with_fallback_failure(
                    outcome.status,
                    elapsed_ms=outcome.elapsed_ms,
                    used=outcome.status not in {"not_needed", "disabled", "not_configured"},
                    diagnostic=outcome.diagnostic(),
                )

        if validation.state in {QueryState.AMBIGUOUS, QueryState.INCOMPLETE}:
            self._query_metrics["clarification_count"] += 1
        elif validation.state is QueryState.UNSUPPORTED:
            self._query_metrics["unsupported_count"] += 1
        elif validation.state is QueryState.OUT_OF_SCOPE:
            self._query_metrics["out_of_scope_count"] += 1
        return validation.plan, validation

    def _blocked_response(
        self,
        question_id: str,
        question: str,
        validation: QueryValidationResult,
        *,
        clarification_decision: ClarificationDecision | None = None,
    ) -> dict[str, Any]:
        state = validation.state
        if (
            clarification_decision is not None
            and clarification_decision.state is ClarificationState.CLARIFY
        ):
            answer = clarification_text(clarification_decision)
            route = "clarification"
        elif (
            clarification_decision is not None
            and clarification_decision.state is ClarificationState.INSUFFICIENT_EVIDENCE
        ):
            answer = "요청하신 내용을 뒷받침할 공시 근거를 확인하지 못했습니다."
            route = "insufficient_evidence"
        elif (
            clarification_decision is not None
            and clarification_decision.state is ClarificationState.UNSUPPORTED
        ):
            answer = "현재 시스템이 지원하는 공시 의미로 요청을 해석할 수 없습니다."
            route = "unsupported"
        elif state in {QueryState.AMBIGUOUS, QueryState.INCOMPLETE}:
            answer = (
                validation.clarification.question
                if validation.clarification is not None
                else "질문을 조금 더 구체적으로 알려주세요."
            )
            route = "clarification"
        elif state is QueryState.UNSUPPORTED:
            answer = "현재 시스템은 해당 요청 유형을 지원하지 않습니다. 공시 사실 조회 질문으로 바꿔주세요."
            route = "unsupported"
        else:
            year = validation.plan.period.year
            answer = (
                f"현재 제공된 공시 데이터 범위에서는 {year}년 내용을 확인할 수 없습니다."
                if year is not None and "period_out_of_corpus" in validation.issues
                else "현재 제공된 공시 데이터 범위에서는 해당 기업 또는 기간을 확인할 수 없습니다."
            )
            route = "out_of_scope"
        stages = ["query_understanding", "query_validation"]
        if validation.fallback_used:
            stages.append("hcx_semantic_fallback")
            stages.append("query_revalidation")
        if clarification_decision is not None:
            stages.append("clarification_resolver")
            if _clarification_classifier_called(clarification_decision):
                stages.append("hcx_clarification_classifier")
        return {
            "question_id": question_id,
            "question": question,
            "retrieved_context": [],
            "think_trace": {
                "task_type": validation.plan.task_type,
                "route": route,
                "stages": stages,
                "retrieval_count": 0,
                "selected_evidence_count": 0,
                "answerable": False,
                "warnings": list(validation.issues),
                "hcx_status": "skipped_query_not_resolved",
                "correction": None,
                "query_understanding": validation.to_public_dict(),
                "query_validation": validation.to_validation_dict(),
                **(
                    {"clarification": clarification_decision.to_public_dict()}
                    if clarification_decision is not None
                    else {}
                ),
            },
            "answer": answer,
        }

    def _adjust_metrics_for_clarification(
        self, decision: ClarificationDecision
    ) -> None:
        if decision.state is ClarificationState.CLARIFY:
            return
        self._query_metrics["clarification_count"] = max(
            0, self._query_metrics["clarification_count"] - 1
        )
        if decision.state is ClarificationState.UNSUPPORTED:
            self._query_metrics["unsupported_count"] += 1

    def _post_resolution_clarification_response(
        self,
        question_id: str,
        question: str,
        *,
        result: Any,
        generated: GeneratedAnswer,
        execution: Any,
        multi_document: Any,
        validation: QueryValidationResult | None,
        answerability: AnswerabilityResult | None,
        decision: ClarificationDecision,
    ) -> dict[str, Any]:
        context: list[dict[str, Any]] = []
        unmapped = False
        if decision.state is ClarificationState.CLARIFY:
            answer = clarification_text(decision)
            route = "clarification"
            proven = _clarification_candidate_evidence(execution, decision)
            if proven is not None:
                rows, citations, markers = proven
                cited, alignment = align_public_citations(
                    clarification_text(decision, citation_markers=markers),
                    citations,
                    rows,
                )
                # A marker that cannot be joined to a served row would be a
                # claim about evidence nobody can check.  Drop the whole
                # attempt rather than serve half of it.
                unmapped = bool(alignment["unmapped"])
                if not unmapped:
                    answer, context = cited, rows
        elif decision.state is ClarificationState.UNSUPPORTED:
            answer = "현재 시스템이 지원하는 공시 의미로 요청을 해석할 수 없습니다."
            route = "unsupported"
        else:
            answer = "요청하신 내용을 뒷받침할 공시 근거를 확인하지 못했습니다."
            route = "insufficient_evidence"
        outcome = VerbalizationOutcome(
            answer,
            "skipped_clarification",
            decision.state.value,
        )
        trace = think_trace(
            result,
            generated,
            outcome,
            execution,
            multi_document=multi_document,
            query_validation=validation,
            answerability=answerability,
            clarification_decision=decision,
        )
        trace["route"] = route
        trace["answerable"] = False
        if unmapped:
            trace["warnings"] = list(
                dict.fromkeys([*trace["warnings"], "citation_alignment_unmapped"])
            )
        return {
            "question_id": question_id,
            "question": question,
            # The semantic choice is unresolved, so no candidate's evidence is
            # exposed as though it were the answer.  What may be exposed is the
            # opposite claim: the filings that prove the choice exists, bounded
            # to exactly those, and only when each one is cited.
            "retrieved_context": context,
            "think_trace": trace,
            "answer": answer,
        }

    def _multi_document(
        self, question: str, plan: Any, execution: Any
    ) -> MultiDocumentEvidence | None:
        """Plan, execute, and hydrate the deterministic set, if there is one.

        Returns ``None`` whenever P0-C declines, which keeps the rest of the
        pipeline on its original path byte for byte.  A repository that is
        unavailable degrades to the same ``None``: the existing retrieval
        evidence still answers the question.
        """

        if self.multi_document_planner is None or self.multi_document_executor is None:
            return None
        multi_plan = self.multi_document_planner.plan(question, plan)
        if not multi_plan.applied:
            return None
        executed = self.multi_document_executor.execute(multi_plan)
        if executed.unavailable_reason is not None:
            return None
        if self.multi_document_evidence is None:
            return None
        evidence = self.multi_document_evidence.build(
            executed, plan=plan, start_rank=len(list(execution.results)) + 1
        )
        if evidence.added_results:
            # Appended after ranking, exactly as correction and event expansion
            # already do, so the retrieved order is untouched and the added
            # rows stay identifiable.
            chunks, results = apply_expansion(
                evidence, execution.chunks, execution.results
            )
            object.__setattr__(execution, "chunks", chunks)
            object.__setattr__(execution, "results", results)
        return evidence


    def _retrieve(self, question: str) -> tuple[Any, Any]:
        try:
            plan = self.understanding.understand(question, top_k=self.settings.top_k)
            return plan, self.executor.execute(plan)
        except Exception as error:  # noqa: BLE001 - surfaced as a sanitized reason
            raise AnswerPipelineError(_classify(error)) from error


def _multi_document_count(multi: Any) -> int:
    """How many rows P0-C contributed, so the slice never truncates them."""

    if multi is None:
        return 0
    return len(getattr(multi, "added_results", ()) or ())


def _preserve_multi_document_semantics(
    outcome: VerbalizationOutcome,
    generated: GeneratedAnswer,
    multi_document: Any,
) -> VerbalizationOutcome:
    """Reject a final rewrite that changes the deterministic lifecycle state."""

    facts = getattr(multi_document, "facts", None)
    state = getattr(facts, "lifecycle_answer", None)
    if state is None:
        return outcome
    verdict = check_answer(state, outcome.text)
    if verdict.ok:
        return outcome

    # The generator renders ``facts.statement()`` verbatim.  Check it too so
    # the release gate remains fail-closed even if a later generator edit
    # accidentally changes that contract.
    deterministic = generated.answer_text
    fallback = check_answer(state, deterministic)
    if not fallback.ok:
        deterministic = facts.statement()
    return VerbalizationOutcome(
        deterministic,
        "fallback_semantic_guard",
        verdict.reason,
    )


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
    # Amount-change operand recovery is also additive: it can append an
    # explicitly scoped filing after the frozen Top-K.  Count exactly the
    # results it actually added so public ``retrieved_context`` cannot truncate
    # away evidence the deterministic arithmetic and citation registry used.
    routing = getattr(execution, "routing", None)
    if isinstance(routing, Mapping):
        hybrid = routing.get("hybrid")
        if isinstance(hybrid, Mapping):
            recovery = hybrid.get("amount_change_operand_recovery")
            if isinstance(recovery, Mapping):
                added = recovery.get("added_chunk_ids") or ()
                total += len(
                    {
                        str(chunk_id)
                        for chunk_id in added
                        if str(chunk_id)
                    }
                )

    return total


class _FinalEvidence:
    """An execution whose served results were replaced, not mutated.

    Retrieval output is immutable, so a post-retrieval stage that enriches the
    served list has to hand its own list forward.  This wraps the original so
    every retrieval attribute still reads through untouched while ``results``
    reports what the agent actually answered from.
    """

    __slots__ = ("_execution", "chunks", "results")

    def __init__(
        self,
        execution: Any,
        chunks: Sequence[Any],
        results: Sequence[Any],
    ) -> None:
        self._execution = execution
        self.chunks = chunks
        self.results = tuple(results)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._execution, name)


def final_evidence(execution: Any, result: Any) -> Any:
    """The evidence set the answer was actually built from.

    Returns the execution itself whenever nothing enriched it, so an untouched
    question keeps rendering the exact object it always did.
    """

    overridden = bool(getattr(result, "evidence_overridden", False))
    results = tuple(getattr(result, "evidence_results", ()) or ())
    chunks = tuple(getattr(result, "evidence_chunks", ()) or ())
    if overridden:
        return _FinalEvidence(execution, chunks, results)
    if not results:
        return execution
    if list(results) == list(execution.results) and not chunks:
        return execution
    if not chunks:
        chunks = getattr(execution, "chunks", ())
    if (
        list(results) == list(execution.results)
        and list(chunks) == list(execution.chunks)
    ):
        return execution
    return _FinalEvidence(execution, chunks, results)


def _clarification_candidate_evidence(
    execution: Any,
    decision: ClarificationDecision,
) -> tuple[list[dict[str, Any]], tuple[GeneratedCitation, ...], dict[str, str]] | None:
    """The filings that prove one event-instance choice, and nothing else.

    Citing a candidate's filing asserts that the filing exists and describes a
    contract distinct from the other candidate's -- never that its 계약금액 is
    the answer.  That is why the served rows here are the candidates and only
    the candidates: exposing the ranked Top-K instead would put the amounts of
    unrelated filings behind an answer that deliberately states none.

    Returns ``None`` -- leaving the caller on its original uncited path -- for
    any choice that is not between filings, and for any candidate the provider
    could not bind to a chunk this execution actually served.  A citation is a
    claim about evidence, so an unbound candidate ends the attempt rather than
    borrowing a neighbour's row.
    """

    candidates = decision.candidates
    if not candidates or any(
        candidate.semantic_type != EVENT_INSTANCE for candidate in candidates
    ):
        return None
    chunks_by_id = {
        str(candidate.chunk_id): candidate
        for candidate in (getattr(execution, "chunks", ()) or ())
    }
    scores: dict[str, float] = {}
    for result in getattr(execution, "results", ()) or ():
        scores.setdefault(str(result.chunk_id), float(result.bm25_score))

    served: list[Any] = []
    ranks: dict[tuple[str, str], int] = {}
    markers: dict[str, str] = {}
    citations: list[GeneratedCitation] = []
    for candidate in candidates:
        identity = candidate.source
        if identity is None:
            return None
        chunk_id, doc_id = identity
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None or str(chunk.doc_id) != doc_id:
            return None
        rank = ranks.get(identity)
        if rank is None:
            # Two candidates that collapsed onto one filing share its row, so
            # the same logical root is never served or numbered twice.
            rank = len(served) + 1
            ranks[identity] = rank
            served.append(chunk)
            citations.append(
                GeneratedCitation(
                    citation_id=f"[{rank}]",
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    source_refs=tuple(
                        copy.deepcopy(list(dict(chunk.chunk).get("source_refs") or []))
                    ),
                    section=_chunk_section(chunk.chunk),
                    evidence_type="clarification_candidate",
                )
            )
        markers[candidate.id] = f"[{rank}]"

    results = tuple(
        RetrievalResult(
            chunk_id=str(chunk.chunk_id),
            doc_id=str(chunk.doc_id),
            bm25_score=scores.get(str(chunk.chunk_id), 0.0),
            rank=rank,
            metadata_match={},
        )
        for rank, chunk in enumerate(served, start=1)
    )
    rows = retrieved_context(
        SimpleNamespace(chunks=tuple(served), results=results), len(results)
    )
    return rows, tuple(citations), markers


def _chunk_section(chunk: Any) -> str:
    """The section label the citation names, in the generator's own shape."""

    path = dict(chunk or {}).get("section_path") or []
    if isinstance(path, str):
        return path
    return " > ".join(str(part) for part in path if part)


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


_PUBLIC_CITATION_MARKER = re.compile(r"\[(\d{1,2})\]")


def align_public_citations(
    answer_text: str,
    citations: Sequence[GeneratedCitation],
    context: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    """Translate internal citation ordinals to the served rank namespace.

    A generated marker identifies a citation object, not a retrieval position.
    The public API contract is different: ``[n]`` names the row whose
    ``retrieved_context.rank`` is ``n``.  Join those namespaces by exact
    ``(chunk_id, doc_id)`` identity after the final context has been built.

    Any marker without one unambiguous exact match is removed in the same regex
    pass used for successful rewrites.  This fails closed without changing the
    served evidence or allowing replacements such as ``[1] -> [2] -> [4]`` to
    cascade.
    """

    markers = tuple(
        dict.fromkeys(
            match.group(0)
            for match in _PUBLIC_CITATION_MARKER.finditer(answer_text)
        )
    )
    if not markers:
        return answer_text, {
            "attempted": False,
            "mapped": [],
            "unmapped": [],
            "status": "not_needed",
        }

    ranks_by_identity: dict[tuple[str, str], set[int]] = {}
    identities_by_rank: dict[int, set[tuple[str, str]]] = {}
    for row in context:
        chunk_id = str(row.get("chunk_id") or "")
        doc_id = str(row.get("doc_id") or "")
        try:
            rank = int(row.get("rank"))
        except (TypeError, ValueError):
            continue
        if not chunk_id or not doc_id:
            continue
        identity = (chunk_id, doc_id)
        ranks_by_identity.setdefault(identity, set()).add(rank)
        identities_by_rank.setdefault(rank, set()).add(identity)

    citations_by_marker: dict[str, list[GeneratedCitation]] = {}
    for citation in citations:
        citations_by_marker.setdefault(str(citation.citation_id), []).append(citation)

    replacements: dict[str, str] = {}
    mapped: list[dict[str, Any]] = []
    unmapped: list[dict[str, Any]] = []
    for marker in markers:
        marker_number = int(marker[1:-1])
        marker_citations = citations_by_marker.get(marker, [])
        identities = {
            (str(citation.chunk_id), str(citation.doc_id))
            for citation in marker_citations
            if citation.chunk_id and citation.doc_id
        }
        if len(identities) != 1:
            replacements[marker] = ""
            unmapped.append(
                {
                    "internal_marker": marker_number,
                    "reason": (
                        "marker_not_generated"
                        if not marker_citations
                        else "ambiguous_generated_identity"
                    ),
                }
            )
            continue

        identity = next(iter(identities))
        ranks = ranks_by_identity.get(identity, set())
        if len(ranks) != 1:
            replacements[marker] = ""
            unmapped.append(
                {
                    "internal_marker": marker_number,
                    "doc_id": identity[1],
                    "chunk_id": identity[0],
                    "reason": (
                        "identity_not_served"
                        if not ranks
                        else "identity_rank_ambiguous"
                    ),
                }
            )
            continue

        public_rank = next(iter(ranks))
        if identities_by_rank.get(public_rank) != {identity}:
            replacements[marker] = ""
            unmapped.append(
                {
                    "internal_marker": marker_number,
                    "doc_id": identity[1],
                    "chunk_id": identity[0],
                    "reason": "served_rank_ambiguous",
                }
            )
            continue

        replacements[marker] = f"[{public_rank}]"
        mapped.append(
            {
                "internal_marker": marker_number,
                "public_rank": public_rank,
                "doc_id": identity[1],
                "chunk_id": identity[0],
            }
        )

    aligned = _PUBLIC_CITATION_MARKER.sub(
        lambda match: replacements.get(match.group(0), ""), answer_text
    )
    status = "aligned"
    if unmapped:
        status = "partial" if mapped else "unmapped"
    return aligned, {
        "attempted": True,
        "mapped": mapped,
        "unmapped": unmapped,
        "status": status,
    }


def _vector_degradation_warning(execution: Any) -> str | None:
    """Make a degraded retrieval lane visible without changing what is served.

    Serving policy is untouched: a degraded request still answers.  What changes
    is that partial coverage can no longer look identical to a healthy hybrid
    run -- a corpus with 53.6% of candidates embedded still reported
    ``vector_status == "ok"`` while 91.2% of the chunks it served came from the
    embedded subset, because only embedded chunks can compete in both lanes.
    """

    status = getattr(execution, "vector_status", None)
    if status is None:
        return None
    coverage = getattr(execution, "vector_coverage", None) or {}
    degradation = _classify_vector_availability(status, coverage)
    if degradation == _VECTOR_HEALTHY:
        return None
    hybrid = (getattr(execution, "routing", None) or {}).get("hybrid") or {}
    return _vector_warning_for(
        degradation,
        coverage,
        # Carried from the configured identity, so an intentional hash
        # diagnostic is never reported as a degraded BGE run.
        provider=(hybrid.get("embedding") or {}).get("provider"),
        error=getattr(execution, "vector_error", None) or coverage.get("error"),
    )


def think_trace(
    result: Any,
    generated: GeneratedAnswer,
    outcome: VerbalizationOutcome,
    execution: Any,
    *,
    multi_document: Any = None,
    query_validation: QueryValidationResult | None = None,
    answerability: AnswerabilityResult | None = None,
    clarification_decision: ClarificationDecision | None = None,
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
    coverage = getattr(result, "holding_coverage", None)
    coverage_trace = coverage.to_dict() if coverage is not None else {}
    if coverage_trace.get("rescue_mode"):
        trace["holding_evidence_coverage"] = {
            "status": coverage_trace.get("status"),
            "rescued": bool(coverage_trace.get("rescued")),
            "rescue_mode": coverage_trace["rescue_mode"],
        }
    degradation = _vector_degradation_warning(execution)
    if degradation:
        trace["warnings"] = [*trace["warnings"], degradation]
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
    pair = getattr(result, "correction_pair", None)
    if isinstance(pair, Mapping) and pair:
        # Whether a correction before/after pair was bound, and why not when it
        # was not. Written beside the existing correction summary rather than
        # into it, so every field a reader already had is unchanged, and absent
        # entirely for a question that never reached the binder.
        trace["correction_pair"] = dict(pair)
    routing = getattr(execution, "routing", None)
    if isinstance(routing, Mapping) and routing.get("comparison_evidence"):
        # Retrieval ran once per company before anything was ranked together,
        # so the stage belongs ahead of the ordinary ones.
        stages.insert(0, "comparison_evidence")
        trace["comparison_evidence"] = routing["comparison_evidence"]
    event = getattr(execution, "event_expansion", None)
    if isinstance(event, Mapping) and event.get("event_expanded"):
        # Which lifecycle contributed which filings. An execution summary, not
        # reasoning. Expansion runs inside retrieval, after correction expansion.
        stages.insert(0, "corporate_event_expansion")
        trace["corporate_event"] = event.get("corporate_event_expansion")
    facts = getattr(multi_document, "facts", None)
    if facts is not None:
        # Counts and statuses only -- no identifier ever reaches the trace.
        # Absent entirely when P0-C declined, so a non-engaged question's trace
        # is byte-identical to what it was before P0-C existed.
        stages.insert(0, "multi_document_planner")
        stages.insert(1, "multi_document_executor")
        if getattr(multi_document, "added_results", ()):
            stages.insert(2, "multi_document_evidence")
        trace["multi_document_planner"] = multi_document.trace()
    if query_validation is not None:
        stages.insert(0, "query_understanding")
        stages.insert(1, "query_validation")
        if query_validation.fallback_used:
            stages.insert(2, "hcx_semantic_fallback")
            stages.insert(3, "query_revalidation")
        trace["query_understanding"] = query_validation.to_public_dict()
        trace["query_validation"] = query_validation.to_validation_dict()
    if answerability is not None:
        stages.append("answerability_guard")
        trace["answerability"] = answerability.to_public_dict()
        trace["answerable"] = (
            answerability.status is AnswerabilityStatus.ANSWERABLE
        )
    if clarification_decision is not None:
        stages.append("clarification_resolver")
        if _clarification_classifier_called(clarification_decision):
            stages.append("hcx_clarification_classifier")
        trace["clarification"] = clarification_decision.to_public_dict()
    return trace


def _chunk_identity(chunk: Any) -> tuple[str, str]:
    """``(chunk_id, doc_id)`` from a retrieval chunk in either shape.

    ``HybridQueryExecutor`` yields ``CandidateChunk`` dataclasses whose payload
    sits under ``.chunk``; unit fixtures and some backends yield the payload
    mapping directly. The candidate's own attributes win when present, because
    they are the identity retrieval assigned, and the payload is consulted only
    to fill a gap.
    """

    if isinstance(chunk, Mapping):
        return str(chunk.get("chunk_id") or ""), str(chunk.get("doc_id") or "")
    payload = getattr(chunk, "chunk", None)
    if not isinstance(payload, Mapping):
        payload = {}
    chunk_id = str(getattr(chunk, "chunk_id", "") or payload.get("chunk_id") or "")
    doc_id = str(getattr(chunk, "doc_id", "") or payload.get("doc_id") or "")
    return chunk_id, doc_id


def _route(result: Any) -> str:
    for stage in result.execution_trace:
        if stage.endswith("_resolver"):
            return stage
    return "general_evidence"


def _clarification_classifier_called(decision: ClarificationDecision) -> bool:
    return decision.classifier_status not in {
        "not_called",
        "disabled",
        "not_configured",
        "not_needed",
    }


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
