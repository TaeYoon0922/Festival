"""Response contract for ``GET /answer``.

The response carries exactly five top-level fields.  Nothing else is added, so a
strict evaluation client can parse it without knowing anything about this
project's internals.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedContextItem(BaseModel):
    """One served Top-K chunk, with the provenance that supports a citation."""

    rank: int
    chunk_id: str
    doc_id: str
    bm25_score: float
    chunk_type: str | None = None
    section_path: list[str] = Field(default_factory=list)
    report_nm: str | None = None
    corp_code: str | None = None
    corp_name: str | None = None
    rcept_dt: str | None = None
    period: dict[str, Any] = Field(default_factory=dict)
    content: str | None = None
    retrieval_text: str | None = None
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)


class CorrectionTrace(BaseModel):
    """Which correction chain supplied evidence beyond what retrieval ranked.

    Identifiers and the intent that selected them, nothing more: this says which
    documents were added and where they came from, not why the answer says what
    it says.
    """

    correction_intent: str | None = None
    correction_group_id: str | None = None
    correction_root_doc_id: str | None = None
    correction_latest_doc_id: str | None = None
    correction_added_doc_ids: list[str] = Field(default_factory=list)


class MultiDocumentTrace(BaseModel):
    """Safe public P0-C summary; internal slots and identifiers are excluded."""

    applied: bool
    plan_type: str
    family_resolution: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    passes: int
    complete: bool
    stop_reason: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    logical_count: int
    unresolved_count: int
    lifecycle_answer: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    terminated_count: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    open_count: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    evidence_count: int


class ComparisonEvidenceTrace(BaseModel):
    """Which fan-out ran, in counts only.

    A company name never reaches the trace: this says how many companies were
    retrieved separately and why the layer engaged, not which ones.
    """

    applied: bool
    company_count: int
    subject_kind: str | None = None
    decline_reason: str | None = None


class QuerySlotTrace(BaseModel):
    value: Any = None
    source: str
    status: str
    candidates: list[str] = Field(default_factory=list)


class ClarificationOptionTrace(BaseModel):
    id: str
    label: str


class ClarificationTrace(BaseModel):
    question: str
    options: list[ClarificationOptionTrace] = Field(default_factory=list)


class BoundedClarificationCandidateTrace(BaseModel):
    id: str
    label: str
    semantic_type: str
    provenance: str


class BoundedClarificationDecisionTrace(BaseModel):
    state: str
    reason: str
    candidate_count: int
    candidates: list[BoundedClarificationCandidateTrace] = Field(default_factory=list)
    selected_candidate_id: str | None = None
    classifier_status: str
    truncated: bool = False


class HcxSemanticDiagnosticTrace(BaseModel):
    transport_status: str
    response_shape: str | None = None
    content_present: bool
    parse_status: str
    content_format: str | None = None
    prefix_text_present: bool = False
    suffix_text_present: bool = False
    schema_error_code: str | None = None
    schema_error_fields: list[str] = Field(default_factory=list)


class QueryUnderstandingTrace(BaseModel):
    status: str
    resolved_slots: dict[str, QuerySlotTrace] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    ambiguous_slots: list[str] = Field(default_factory=list)
    hcx_fallback_used: bool
    hcx_fallback_status: str
    hcx_elapsed_ms: float | None = None
    hcx_diagnostic: HcxSemanticDiagnosticTrace | None = None
    clarification_required: bool
    clarification: ClarificationTrace | None = None


class QueryValidationTrace(BaseModel):
    status: str
    retrieval_allowed: bool
    reason: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class AnswerabilityTrace(BaseModel):
    status: str
    evidence_count: int
    citation_count: int
    complete: bool | None = None
    confirmed_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    reason: str | None = None
    citable: bool = False
    relevant_to_request: bool | None = None
    answerable: bool = False
    #: STEP 11-C.  Present only when a domain producer answered for a canonical
    #: field, so every question outside the two fielded lanes keeps the exact
    #: key set it had.
    unavailable_fields: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )
    unavailable_evidence: list[dict[str, Any]] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
    )


class HoldingEvidenceCoverageTrace(BaseModel):
    status: str
    rescued: bool
    rescue_mode: str


class ThinkTrace(BaseModel):
    """Execution summary: which components ran and what they concluded.

    Deliberation is deliberately absent.  Only component names, counts, and
    statuses appear here.
    """

    task_type: str | None = None
    route: str
    stages: list[str] = Field(default_factory=list)
    retrieval_count: int
    selected_evidence_count: int
    answerable: bool
    warnings: list[str] = Field(default_factory=list)
    hcx_status: str
    #: Present only when the correction graph contributed documents; null
    #: otherwise, like every other optional field in this summary.
    correction: CorrectionTrace | None = None
    #: Present only when P0-C engaged.  ``exclude_if`` is intentionally scoped
    #: to this field so ordinary and Gold60 responses retain their exact prior
    #: key set, including the existing ``correction: null`` contract.
    multi_document_planner: MultiDocumentTrace | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    #: Present only on the P0-D serving path. Legacy unit fixtures that do not
    #: wire the verifier retain their byte-for-byte trace contract.
    query_understanding: QueryUnderstandingTrace | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    query_validation: QueryValidationTrace | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    answerability: AnswerabilityTrace | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    #: Present only when a comparison was retrieved company by company.
    #: ``exclude_if`` keeps every other response's key set unchanged.
    comparison_evidence: ComparisonEvidenceTrace | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    holding_evidence_coverage: HoldingEvidenceCoverageTrace | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    clarification: BoundedClarificationDecisionTrace | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )


class AnswerResponse(BaseModel):
    question_id: str
    question: str
    retrieved_context: list[RetrievedContextItem] = Field(default_factory=list)
    think_trace: ThinkTrace
    answer: str


class ErrorResponse(BaseModel):
    reason: str
    message: str
