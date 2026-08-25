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


class AnswerResponse(BaseModel):
    question_id: str
    question: str
    retrieved_context: list[RetrievedContextItem] = Field(default_factory=list)
    think_trace: ThinkTrace
    answer: str


class ErrorResponse(BaseModel):
    reason: str
    message: str
