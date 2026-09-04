"""Read-only coordinator for existing Festival reasoning components."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from app.agent.task_router import TaskDecision, TaskRouter
from app.reasoning.company_comparison import (
    COMPARISON_OUTCOME_KEY,
    compose_comparison_text,
    executable_comparison,
    ranking_from_outcome,
)
from app.reasoning.comparison_ranking import (
    COMPARISON_RANKING_KEY,
    compose_conditional_ranking_text,
    ranking_from_outcome as conditional_ranking_from_outcome,
)
from app.reasoning.amount_change import (
    compose_amount_change_text,
    requested_amount_change,
    resolve_amount_change,
)
from app.reasoning.contract_lifecycle import (
    compose_lifecycle_text,
    lifecycle_items,
    lifecycle_outcome,
    requested_lifecycle_outcome,
)
from app.reasoning.answer_composer import (
    AnswerComposer,
    AnswerDraft,
    AnswerSection,
    EvidenceCitation,
)
from app.reasoning.corporate_event_field_evidence import corporate_event_field_evidence
from app.reasoning.evidence_builder import EvidenceBuilder, EvidenceItem, EvidenceSet
from app.reasoning.field_evidence import FieldEvidence, FieldStatus
from app.reasoning.holding_field_evidence import (
    ACQUISITION_UNIT_PRICE,
    holding_field_evidence,
    requested_holding_fields as requested_holding_field_evidence,
)
from app.reasoning.holding_event_resolver import (
    HoldingEventResolver,
    HoldingResolution,
)
from app.reasoning.holding_company_role_resolution import (
    HoldingCompanyRoleResolver,
    has_role_provenance,
)
from app.reasoning.periodic_fact_resolver import (
    PeriodicFactResolution,
    PeriodicFactResolver,
)
from app.reasoning.holding_date_intent import (
    exact_reference_date,
    execution_plan as holding_execution_plan,
)
from app.reasoning.correction_pair_roles import (
    apply_correction_pair,
    decide_correction_pair,
    pair_trace,
)
from app.reasoning.holding_event_fusion import fuse as fuse_holding_events
from app.reasoning.holding_event_selection import (
    EXACT,
    classify_holding_event_selection,
)
from app.reasoning.holding_evidence_coverage import (
    CoverageAssessment,
    assess as assess_holding_coverage,
)
from app.reasoning.holding_report_relative_execution import (
    HoldingReportRelativeExecution,
    ReportRelativeEvidenceExecution,
)
from app.reasoning.holding_report_index import HoldingReportIndex
from app.reasoning.holding_report_relative import SELECTOR_EXACT_REFERENCE_DATE
from app.reasoning.periodic_evidence_selector import PeriodicEvidenceSelector
from app.retrieval.interfaces import RetrievalResult


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
    #: Whether a correction before/after pair was bound, and the deterministic
    #: reason when it was not.  Internal diagnostic, additive: it is reported
    #: beside the existing trace and never replaces any field a reader has.
    correction_pair: Mapping[str, Any] = field(default_factory=dict)
    #: The served evidence the answer was actually built from.  This equals the
    #: retrieval output unless a post-retrieval stage enriched it; because that
    #: output is immutable, an enriched list can only be carried out here.  The
    #: public response renders this list, so what the resolver used and what the
    #: caller is shown stay the same evidence.
    evidence_results: tuple[Any, ...] = ()
    #: Candidate chunks paired with ``evidence_results``.  Usually this is the
    #: frozen retrieval pool; exact deterministic hydration can instead supply
    #: a selected chunk that retrieval never fetched.
    evidence_chunks: tuple[Any, ...] = ()
    #: Distinguishes an intentional empty deterministic failure from an older
    #: caller that simply did not populate the two internal evidence fields.
    evidence_overridden: bool = False
    #: Internal Phase 3 diagnostic; deliberately absent from ``to_dict`` and
    #: therefore from the public API schema.
    report_relative_execution: ReportRelativeEvidenceExecution | None = None
    #: STEP 11-C.  What the domain producers found about the canonical fields
    #: this question requested, if any producer had the authority to speak.
    #: Empty for every question outside the two fielded lanes, which is what
    #: keeps those questions on the answerability path they always took.
    field_evidence: tuple[FieldEvidence, ...] = ()

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
        report_relative_execution: HoldingReportRelativeExecution | None = None,
        holding_company_role_resolver: HoldingCompanyRoleResolver | None = None,
        holding_report_index: HoldingReportIndex | None = None,
        active_corpus_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.task_router = task_router or TaskRouter()
        self.evidence_builder = evidence_builder or EvidenceBuilder()
        self.holding_resolver = holding_resolver or HoldingEventResolver()
        self.periodic_resolver = periodic_resolver or PeriodicFactResolver()
        self.periodic_evidence_selector = (
            periodic_evidence_selector or PeriodicEvidenceSelector()
        )
        self.answer_composer = answer_composer or AnswerComposer()
        self.report_relative_execution = report_relative_execution
        self.holding_company_role_resolver = holding_company_role_resolver
        self.holding_report_index = (
            holding_report_index
            if holding_report_index is not None
            else getattr(report_relative_execution, "index", None)
        )
        inherited_identity = getattr(
            report_relative_execution, "active_corpus_identity", None
        )
        self.active_corpus_identity = (
            dict(active_corpus_identity)
            if active_corpus_identity is not None
            else dict(inherited_identity)
            if inherited_identity is not None
            else None
        )

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
        holding_plan = holding_execution_plan(
            question, query_plan, routed_task_type=decision.task_type
        )

        # Phase 3.  The report index, never ranked retrieval, decides which
        # filing report-relative wording names.  A returned execution is
        # authoritative even when empty: empty means the deterministic source
        # declined and ordinary retrieval must not answer from another report.
        report_relative = (
            self.report_relative_execution.adapt(
                question,
                query_plan,
                execution,
                routed_task_type=decision.task_type,
            )
            if self.report_relative_execution is not None
            else None
        )

        # A holding execution can be served plenty of evidence and still be
        # unable to answer: the structured projections carrying the fields may
        # have missed the cutoff, or the served ones may be the wrong projection
        # type for what was asked. Top the gap up from the pool retrieval
        # already fetched -- no query is issued here -- and leave the original
        # execution untouched so the read-only invariants still hold.
        reporter_scoped = False
        if report_relative is not None:
            trace.append("holding_report_relative_execution")
            evidence_input = SimpleNamespace(
                plan=execution.plan,
                chunks=report_relative.chunks,
                results=report_relative.results,
            )
            coverage = CoverageAssessment(results=report_relative.results)
        else:
            scoped_execution = _holding_reporter_scope(
                execution,
                query_plan,
                routed_task_type=decision.task_type,
                resolver=self.holding_company_role_resolver,
            )
            ordinary_execution = scoped_execution or execution
            if scoped_execution is not None:
                reporter_scoped = True
                trace.append("holding_reporter_scope")
            coverage = assess_holding_coverage(
                question,
                holding_plan,
                ordinary_execution.chunks,
                ordinary_execution.results,
                routed_task_type=decision.task_type,
                report_index=self.holding_report_index,
                active_corpus_identity=self.active_corpus_identity,
                # The production stack always has the Phase-3 adapter.  Its
                # declining this question is what proves ordinary ownership;
                # without that boundary component the fallback stays closed.
                ordinary_lane=self.report_relative_execution is not None,
            )
            evidence_input = ordinary_execution
            if coverage.rescued:
                trace.append("holding_evidence_coverage")
                evidence_input = SimpleNamespace(
                    plan=ordinary_execution.plan,
                    chunks=ordinary_execution.chunks,
                    results=list(coverage.results),
                )

        # P1-A4 D1. A question naming one calendar day is asking about one
        # event, but the plan only carries that day when its own task type was
        # already holding_change -- a promoted execution gets the year instead.
        # Re-read the date with the routed semantics and carry it on a copy, so
        # the frozen plan (and the understanding trace) stay exactly as P0-D
        # left them.
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
        correction_pair_trace: dict[str, Any] = {}

        if decision.task_type == "holding_event":
            trace.append("holding_event_resolver")
            resolution = self.holding_resolver.resolve(
                evidence, query_plan=holding_plan
            )
            # A "정정 전과 정정 후" question is answered by one event that two
            # filings state differently, not by two events. The resolver groups
            # by document and so reports both without saying which version each
            # is; the correction graph already knows, and its expansion trace
            # names the chain's root and final filing. Bind those two roles
            # here, before the composer renders the rows and the citations that
            # attribute them. Declining leaves the resolution exactly as it was.
            correction_pair = decide_correction_pair(
                resolution,
                correction_trace=getattr(
                    retrieval_execution, "correction_expansion", None
                ),
                query_plan=query_plan,
            )
            # Diagnostic only: a decline used to be invisible, so a question
            # that should have paired and did not looked the same as one that
            # was never a pair. Recording it changes nothing about execution.
            correction_pair_trace = pair_trace(correction_pair)
            if correction_pair.claim is not None:
                trace.append("correction_pair_roles")
                resolution = apply_correction_pair(resolution, correction_pair.claim)
            resolution_before = copy.deepcopy(resolution.to_dict())
            trace.append("answer_composer")
            # What the question itself said about which event is wanted, read
            # from the plan alone. The composer must not re-derive it, and the
            # count of served events must not stand in for it.
            draft = self.answer_composer.compose(
                evidence,
                holding_resolution=resolution,
                selection_mode=(
                    EXACT
                    if report_relative is not None and report_relative.resolved
                    else classify_holding_event_selection(
                        holding_plan, routed_task_type=decision.task_type
                    )
                ),
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
            # A contract followed forward is answered by naming which served
            # filing began it and which ended it. Both are already in the
            # evidence set; only the roles were missing.
            lifecycle = (
                lifecycle_outcome(evidence.served_items)
                if requested_lifecycle_outcome(query_plan)
                else None
            )
            # How far an amount moved between two filings. Both values are read
            # from the served chunks' own tables, each staying bound to the
            # filing it came from, so the two can never be crossed.
            change_request = requested_amount_change(query_plan)
            delta = (
                resolve_amount_change(
                    change_request,
                    _served_chunks(retrieval_execution, evidence),
                    corp_code=getattr(query_plan, "corp_code", None),
                )
                if change_request is not None
                else None
            )
            # Several companies, one field, each answered from its own scope.
            comparison = executable_comparison(query_plan)
            evidence_ranking = conditional_ranking_from_outcome(
                (getattr(retrieval_execution, "routing", None) or {}).get(
                    COMPARISON_RANKING_KEY
                )
            )
            # Read the operands execution already chose, rather than looking
            # for them again in evidence that no longer carries each company's
            # own scoping.
            order = (
                ranking_from_outcome(
                    (getattr(retrieval_execution, "routing", None) or {}).get(
                        COMPARISON_OUTCOME_KEY
                    )
                )
                if comparison is not None
                else None
            )
            trace.append("answer_composer")
            ranking_draft = (
                _compose_conditional_comparison_ranking(
                    evidence,
                    evidence_ranking,
                    task_type=decision.task_type,
                    multi_document=multi_document,
                )
                if evidence_ranking is not None
                else None
            )
            if ranking_draft is not None:
                trace.insert(len(trace) - 1, "conditional_comparison_ranking")
                draft = ranking_draft
            elif comparison is not None:
                trace.insert(len(trace) - 1, "company_comparison")
                draft = _compose_company_comparison(
                    evidence,
                    comparison,
                    order,
                    task_type=decision.task_type,
                    multi_document=multi_document,
                )
            elif delta is not None:
                trace.insert(len(trace) - 1, "contract_amount_change")
                draft = _compose_amount_change(
                    evidence,
                    delta,
                    task_type=decision.task_type,
                    multi_document=multi_document,
                )
            elif lifecycle is not None and lifecycle.resolved:
                trace.insert(len(trace) - 1, "contract_lifecycle_resolver")
                draft = _compose_lifecycle(
                    evidence,
                    lifecycle,
                    task_type=decision.task_type,
                    multi_document=multi_document,
                )
            else:
                draft = _compose_general_evidence(
                    evidence,
                    task_type=decision.task_type,
                    multi_document=multi_document,
                )

        # STEP 11-C.  Ask the domain producers what the authoritative source
        # says about the canonical fields this question requested.  Both read
        # already-resolved identity and already-served evidence, so this adds a
        # finding, never a selection -- and both decline outright for a question
        # neither lane owns, which is every question that answered before.
        field_evidence = _field_evidence(
            question=question,
            plan=query_plan,
            # The retrieval execution itself, not the chunks/results view: the
            # upstream event and correction metadata a producer consumes lives
            # on the original object and the view deliberately carries neither.
            execution=retrieval_execution,
            evidence=evidence,
            resolution=resolution,
            multi_document=multi_document,
            authoritative_holding_report=_authoritative_holding_report(
                question,
                query_plan,
                self.holding_report_index,
                active_corpus_identity=self.active_corpus_identity,
            ),
        )

        # A resolved scoped amount-change operation has already bound two
        # different contract amounts to two explicitly named filing roles.
        # The ordinary corporate-event field lane intentionally asks for one
        # flat authoritative contract_amount, so those two legitimate values
        # appear there as a conflict.  That flat conflict must not veto the
        # more-specific scoped operation that already resolved and cited both.
        #
        # This applies only after contract_amount_change actually resolved;
        # ordinary single-field corporate-event questions retain their exact
        # previous conflict behavior.
        if "contract_amount_change" in trace and field_evidence:
            field_evidence = tuple(
                finding
                for finding in field_evidence
                if not (
                    finding.domain == "corporate_event"
                    and finding.field == "contract_amount"
                )
            )

        if field_evidence:
            trace.append("field_evidence")
            # A refusal has to point at the evidence that proves it.  Ordinary
            # composition cites the highest-ranked handful, so a filing that
            # states a blank field can be served and still go uncited, leaving a
            # correct refusal with nothing to show for it.  Add exactly those
            # served items back, through the citation contract composition
            # already uses.  Retrieval, ranking and the served set are untouched.
            draft = _with_field_evidence_citations(draft, field_evidence, evidence)

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
                    *(
                        (f"holding_report_relative:{report_relative.status}",)
                        if report_relative is not None
                        and not report_relative.resolved
                        else ()
                    ),
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
            correction_pair=correction_pair_trace,
            evidence_results=tuple(evidence_input.results),
            evidence_chunks=tuple(evidence_input.chunks),
            evidence_overridden=bool(
                report_relative is not None or coverage.rescued or reporter_scoped
            ),
            report_relative_execution=report_relative,
            field_evidence=field_evidence,
        )


def _with_field_evidence_citations(
    draft: AnswerDraft, field_evidence: Sequence[FieldEvidence], evidence: EvidenceSet
) -> AnswerDraft:
    """Make the evidence a field refusal rests on citable.

    Bounded on purpose.  Only an authoritative ``UNAVAILABLE`` record adds
    anything: that is the one state whose whole meaning is "this served filing
    states no value", and a refusal saying so has to be able to point at it.  A
    stated value is already cited by whatever section reports it; an absence has
    no source to cite and must not invent one; a conflict has no single source
    and keeps the existing insufficient wording.

    At most one item per field, and only items already served and already
    carrying provenance -- no chunk is fetched, promoted, reordered or
    fabricated here.
    """

    by_id = {item.chunk_id: item for item in evidence.served_items}
    cited = {citation.chunk_id for citation in draft.citations}
    added: list[EvidenceCitation] = []
    for field in dict.fromkeys(record.field for record in field_evidence):
        source = next(
            (
                record
                for record in field_evidence
                if record.field == field
                and record.authoritative
                and record.status is FieldStatus.UNAVAILABLE
                and record.chunk_id
                and record.chunk_id not in cited
                and record.chunk_id in by_id
            ),
            None,
        )
        if source is None:
            continue
        cited.add(str(source.chunk_id))
        added.append(_general_citation(by_id[str(source.chunk_id)]))
    if not added:
        return draft
    return replace(draft, citations=(*draft.citations, *added))


def _field_evidence(
    *,
    question: str,
    plan: Any,
    execution: Any,
    evidence: EvidenceSet,
    resolution: Any,
    multi_document: Any = None,
    authoritative_holding_report: Any = None,
) -> tuple[FieldEvidence, ...]:
    """Collect both producers' findings for this question.

    Each producer owns one domain and answers only for the canonical fields its
    own domain defines, so their outputs never compete: a contract-amount
    question reaches no holding row and an acquisition-unit-price question
    reaches no corporate event.  Both are handed the same served evidence, and
    neither may look anywhere else.
    """

    items = evidence.served_items
    return (
        *corporate_event_field_evidence(
            question=question,
            plan=plan,
            execution=execution,
            evidence_items=items,
            multi_document=multi_document,
        ),
        *holding_field_evidence(
            question=question,
            plan=plan,
            resolution=resolution,
            evidence_items=items,
            authoritative_report=authoritative_holding_report,
        ),
    )


def _authoritative_holding_report(
    question: str,
    plan: Any,
    index: HoldingReportIndex | None,
    *,
    active_corpus_identity: Mapping[str, Any] | None,
) -> Any | None:
    """Select an exact report only for the unit-price negative-evidence lane."""

    if index is None or requested_holding_field_evidence(question) != (
        ACQUISITION_UNIT_PRICE,
    ):
        return None
    corp_code = str(getattr(plan, "corp_code", "") or "").strip()
    reporter = str(getattr(plan, "reporter", "") or "").strip()
    period = getattr(plan, "period", None)
    if hasattr(period, "to_dict"):
        period = period.to_dict()
    values = dict(period) if isinstance(period, Mapping) else {}
    start = values.get("from") or values.get("from_date")
    end = values.get("to") or values.get("to_date")
    if not corp_code or not reporter or not start or start != end:
        return None
    selection = index.select_report(
        corp_code,
        reporter,
        SELECTOR_EXACT_REFERENCE_DATE,
        reference_date=str(start),
        active_corpus_identity=active_corpus_identity,
    )
    return selection.selected if selection.resolved else None


def _holding_reporter_scope(
    execution: Any,
    plan: Any,
    *,
    routed_task_type: str | None,
    resolver: HoldingCompanyRoleResolver | None,
) -> Any | None:
    """Keep only retrieved evidence from the indexed issuer/reporter pair.

    Scoped to plans whose reporter this system derived from the corpus itself.
    A reporter the asker named directly is left alone: those questions answered
    without this pruning before B.2 existed, and B.2 is not a licence to change
    them.
    """

    if routed_task_type != "holding_event" or resolver is None:
        return None
    if not has_role_provenance(plan):
        return None
    corp_code = str(getattr(plan, "corp_code", "") or "").strip()
    reporter = str(getattr(plan, "reporter", "") or "").strip()
    document_ids = resolver.document_ids(corp_code, reporter)
    if not document_ids:
        return None
    chunks = tuple(
        candidate
        for candidate in execution.chunks
        if str(getattr(candidate, "doc_id", "") or "") in document_ids
    )
    results = tuple(
        result
        for result in execution.results
        if str(getattr(result, "doc_id", "") or "") in document_ids
    )
    if (
        list(chunks) == list(execution.chunks)
        and list(results) == list(execution.results)
    ):
        return None
    return SimpleNamespace(
        plan=execution.plan, chunks=chunks, results=_dense_ranks(results)
    )


def _dense_ranks(results: Sequence[RetrievalResult]) -> tuple[RetrievalResult, ...]:
    """Relabel surviving ranks 1..N without reordering or rescoring.

    Excluding evidence leaves holes in the served rank sequence, and every
    other evidence path in this pipeline hands out a contiguous one.  Only the
    label changes: order, scores, and provenance are carried through untouched.
    """

    return tuple(
        RetrievalResult(
            chunk_id=result.chunk_id,
            doc_id=result.doc_id,
            bm25_score=result.bm25_score,
            rank=rank,
            metadata_match=result.metadata_match,
        )
        for rank, result in enumerate(results, start=1)
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


def _served_chunks(
    execution: Any, evidence: EvidenceSet
) -> tuple[Mapping[str, Any], ...]:
    """Return raw retrieval chunks for the evidence that was actually served.

    ``HybridQueryExecution.chunks`` contains ``CandidateChunk`` objects rather
    than mappings.  Arithmetic readers need the candidate's raw ``chunk``
    mapping, while identity/date metadata is taken from the already-built
    ``EvidenceItem`` so the scalar stays bound to exactly the evidence the
    answer serves.
    """

    by_id: dict[str, Mapping[str, Any]] = {}

    for candidate in getattr(execution, "chunks", ()) or ():
        if isinstance(candidate, Mapping):
            chunk_id = str(candidate.get("chunk_id") or "")
            payload = dict(candidate)
        else:
            chunk_id = str(getattr(candidate, "chunk_id", "") or "")
            raw = getattr(candidate, "chunk", None)
            if not isinstance(raw, Mapping):
                continue
            payload = dict(raw)
            if chunk_id:
                payload.setdefault("chunk_id", chunk_id)
            doc_id = getattr(candidate, "doc_id", None)
            if doc_id:
                payload.setdefault("doc_id", str(doc_id))

        if chunk_id:
            by_id[chunk_id] = payload

    ordered: list[Mapping[str, Any]] = []

    for item in evidence.served_items:
        payload = by_id.get(item.chunk_id)
        if payload is None:
            continue

        # Candidate.chunk carries the table structure consumed by _amount_cells.
        # EvidenceItem carries the canonical identity/date metadata already used
        # by the answer and citation lanes.  Preserve both.
        hydrated = dict(payload)
        hydrated["chunk_id"] = item.chunk_id
        hydrated["doc_id"] = item.doc_id
        hydrated["corp_code"] = item.corp_code
        hydrated["rcept_dt"] = item.rcept_dt
        hydrated["report_nm"] = item.report_nm

        ordered.append(hydrated)

    return tuple(ordered)


def _compose_company_comparison(
    evidence: EvidenceSet,
    comparison: Any,
    order: Any,
    *,
    task_type: str,
    multi_document: Any = None,
) -> AnswerDraft:
    """State the comparison, citing every company that took part in it.

    All of them or none.  A ranking missing one company states a different
    order than the true one, and a larger-of-two missing one side is not a
    comparison at all, so an unresolved operand makes the whole answer
    unanswerable rather than a partial one presented as complete.
    """

    base = _compose_general_evidence(
        evidence, task_type=task_type, multi_document=multi_document
    )
    if order is None:
        return replace(
            base,
            answerable=False,
            warnings=tuple(
                dict.fromkeys((*base.warnings, "company_comparison_incomplete"))
            ),
        )
    wanted = [
        str(operand.source.chunk_id)
        for operand in order.operands
        if operand.source is not None
    ]
    by_id = {item.chunk_id: item for item in evidence.served_items}
    cited = [by_id[chunk_id] for chunk_id in wanted if chunk_id in by_id]
    if len(cited) != len(order.operands):
        # An operand whose document did not survive into served evidence cannot
        # be cited, and an uncitable comparison is not answerable.
        return replace(
            base,
            answerable=False,
            warnings=tuple(
                dict.fromkeys((*base.warnings, "company_comparison_uncited"))
            ),
        )
    statement = compose_comparison_text(comparison, order)
    section = AnswerSection(
        title="회사별 계약금액 비교",
        content={
            "summary": statement,
            "operands": [operand.to_dict() for operand in order.operands],
            "ordered": bool(comparison.ordered),
        },
        supporting_evidence_ids=tuple(item.chunk_id for item in cited),
    )
    return replace(
        base,
        answer_sections=(section, *base.answer_sections),
        evidence_references=tuple(
            dict.fromkeys((*(item.chunk_id for item in cited), *base.evidence_references))
        ),
        citations=tuple(_general_citation(item) for item in cited),
        answerable=True,
    )


def _compose_conditional_comparison_ranking(
    evidence: EvidenceSet,
    ranking: Any,
    *,
    task_type: str,
    multi_document: Any = None,
) -> AnswerDraft | None:
    """Lead with a ranking only when every selected operand remains citable."""

    base = _compose_general_evidence(
        evidence, task_type=task_type, multi_document=multi_document
    )
    by_id = {item.chunk_id: item for item in evidence.served_items}
    wanted = [operand.chunk_id for operand in ranking.operands]
    cited = [by_id[chunk_id] for chunk_id in wanted if chunk_id in by_id]
    if len(cited) != len(ranking.operands):
        # This is the last fail-closed boundary.  If selected retrieval evidence
        # did not survive into the evidence set, the caller must use the
        # stage-one draft rather than state an uncitable ranking.
        return None

    selected_citations = tuple(_general_citation(item) for item in cited)
    selected_keys = {
        (citation.chunk_id, citation.doc_id) for citation in selected_citations
    }
    citations = (
        *selected_citations,
        *(
            citation
            for citation in base.citations
            if (citation.chunk_id, citation.doc_id) not in selected_keys
        ),
    )
    statement = compose_conditional_ranking_text(
        ranking,
        tuple(f"[{index}]" for index in range(1, len(cited) + 1)),
    )
    section = AnswerSection(
        title="기업별 조건부 금액 비교",
        content={
            "summary": statement,
            "report_kind": ranking.report_kind,
            "base_year": ranking.base_year,
            "operands": [operand.to_dict() for operand in ranking.operands],
        },
        supporting_evidence_ids=tuple(item.chunk_id for item in cited),
    )
    return replace(
        base,
        answer_sections=(section, *base.answer_sections),
        evidence_references=tuple(
            dict.fromkeys(
                (*(item.chunk_id for item in cited), *base.evidence_references)
            )
        ),
        citations=citations,
        answerable=True,
    )


def _compose_amount_change(
    evidence: EvidenceSet,
    delta: Any,
    *,
    task_type: str,
    multi_document: Any = None,
) -> AnswerDraft:
    """State the change, citing the filing each of the two amounts came from."""

    base = _compose_general_evidence(
        evidence, task_type=task_type, multi_document=multi_document
    )
    wanted = {
        str(operand.source.chunk_id)
        for operand in (delta.initial, delta.final)
        if operand.source is not None
    }
    cited = [item for item in evidence.served_items if item.chunk_id in wanted]
    cited.extend(
        item
        for item in evidence.served_items
        if item.chunk_id not in wanted
        and len(cited) < _general_evidence_limit(evidence, task_type=task_type)
    )
    statement = compose_amount_change_text(delta)
    supporting_ids = tuple(
        dict.fromkeys(
            str(operand.source.chunk_id)
            for operand in (delta.initial, delta.final)
            if operand.source is not None and operand.source.chunk_id
        )
    )
    section = AnswerSection(
        title="계약금액 변경",
        content={"summary": statement},
        supporting_evidence_ids=supporting_ids,
    )

    # General evidence has a bounded display/citation set, but a scoped
    # arithmetic result may use operands that sit beyond that display limit.
    # The deterministic summary is only supported when both operand filings
    # remain in the citation registry.  Preserve all existing citations and
    # add only missing operand citations by exact chunk identity.
    item_by_id = {
        item.chunk_id: item
        for item in evidence.served_items
    }
    existing_citation_ids = {
        citation.chunk_id
        for citation in base.citations
    }
    operand_citations = tuple(
        _general_citation(item_by_id[chunk_id])
        for chunk_id in supporting_ids
        if chunk_id in item_by_id
        and chunk_id not in existing_citation_ids
    )
    citations = (*base.citations, *operand_citations)

    return replace(
        base,
        answer_sections=(section, *base.answer_sections),
        evidence_references=tuple(
            dict.fromkeys((*supporting_ids, *base.evidence_references))
        ),
        citations=citations,
    )


def _compose_lifecycle(
    evidence: EvidenceSet,
    outcome: Any,
    *,
    task_type: str,
    multi_document: Any = None,
) -> AnswerDraft:
    """State the contract's outcome, citing both ends of its lifecycle.

    Built on the ordinary general-evidence draft so nothing about rows,
    citation shape or limits changes; the two differences are that the
    lifecycle statement leads the answer, and that the origin and terminal
    filings are cited whatever rank retrieval gave them.
    """

    base = _compose_general_evidence(
        evidence, task_type=task_type, multi_document=multi_document
    )
    statement = compose_lifecycle_text(outcome)
    if not statement:
        return base
    supporting_ids = tuple(
        dict.fromkeys(
            str(item.chunk_id)
            for item in outcome.documents
            if getattr(item, "chunk_id", None)
        )
    )

    # Keep every citation required by the existing General evidence section.
    # A lifecycle document that sat beyond the general display limit is added,
    # never substituted for one of those existing citations.
    existing_refs = set(base.evidence_references)
    lifecycle_extra_citations = tuple(
        _general_citation(item)
        for item in outcome.documents
        if getattr(item, "chunk_id", None)
        and str(item.chunk_id) not in existing_refs
    )
    citations = (*base.citations, *lifecycle_extra_citations)
    section = AnswerSection(
        title="계약 최종 상태",
        content={
            "summary": statement,
            **outcome.to_dict(),
        },
        supporting_evidence_ids=supporting_ids,
    )
    return replace(
        base,
        answer_sections=(section, *base.answer_sections),
        evidence_references=tuple(
            dict.fromkeys((*supporting_ids, *base.evidence_references))
        ),
        citations=citations,
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
