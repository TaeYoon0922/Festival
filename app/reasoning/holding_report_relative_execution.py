"""Bounded production execution for frozen report-relative holding intent.

The report index decides *which* filing a question names.  Retrieval does not:
it ranks text, and its rank-one holding projection may belong to another date
or holder.  This adapter connects the deterministic selection to the existing
evidence path without turning the index into a second answer engine.

Only a deliberately narrow natural-language subset enters this lane.  The
question must already carry one issuer, an explicit reporter, genuine
report-relative wording, and holding-report fields.  Acquisition facts remain
owned by the existing row-level resolver.  Once the lane engages, every index
or hydration failure is authoritative; ordinary ranked evidence is never used
as a substitute report.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.reasoning.holding_correction_finality import (
    DEFAULT_ARTIFACT_PATH as DEFAULT_FINALITY_PATH,
)
from app.reasoning.holding_evidence_coverage import (
    is_citable,
    requested_holding_fields,
)
from app.reasoning.holding_report_index import (
    DEFAULT_ARTIFACT_PATH as DEFAULT_INDEX_PATH,
    RESOLVED,
    HoldingReportIndex,
    ReportExecution,
    execute_report_relative,
    load_index,
)
from app.reasoning.holding_report_relative import (
    SELECTOR_EXACT_RECEIPT_DATE,
    SELECTOR_EXACT_REFERENCE_DATE,
    SELECTOR_LATEST,
    SELECTOR_SELECTED_CONTEXT,
)
from app.reasoning.holding_reporter import canonical_reporter_key
from app.retrieval.interfaces import CandidateChunk, RetrievalResult


ROUTED_TASK_TYPE = "holding_event"

# These are canonical field names produced by the resolver's own request
# parser.  They are intentionally not words or phrase aliases: the adapter
# delegates language interpretation to ``requested_holding_fields`` and only
# checks which frozen field family the result belongs to.
ACQUISITION_REQUEST_FIELDS = frozenset({"acquisition_date", "acquired_shares"})

# Evidence-carrier provenance.  A rank is required by EvidenceBuilder's stable
# input contract, but this row was selected by enumeration, not ranked search.
PROVENANCE_KEY = "holding_report_relative_execution"

PROJECTION_CHUNK_MISSING = "projection_chunk_missing"
PROJECTION_CHUNK_AMBIGUOUS = "projection_chunk_ambiguous"
PROJECTION_IDENTITY_MISMATCH = "projection_identity_mismatch"
PROJECTION_NOT_CITABLE = "projection_not_citable"
HYDRATION_UNAVAILABLE = "exact_hydration_unavailable"

_EXECUTABLE_SELECTORS = frozenset(
    {
        SELECTOR_LATEST,
        SELECTOR_EXACT_REFERENCE_DATE,
        SELECTOR_EXACT_RECEIPT_DATE,
    }
)
_EXPLICIT_CORRECTION_FILTERS = frozenset({"original_only", "corrected_only"})

# The serving database is the frozen Structural v2.1 release represented by
# the tracked manifest.  Future releases fail closed until this identity is
# updated together with their artifacts.
DEFAULT_CORPUS_SNAPSHOT_ID = "structural_v2_1_full_4204"
DEFAULT_PROCESSED_SCHEMA_VERSIONS = ("2.0",)
DEFAULT_CHUNKING_VERSIONS = ("2.1",)
DEFAULT_PROJECTION_PROVENANCE_REVISIONS = ("v2.1-release-1",)
DEFAULT_HOLDING_REPORT_PROJECTION_COUNT = 1118


@dataclass(frozen=True)
class ReportRelativeEvidenceExecution:
    """The authoritative evidence input produced for one engaged question."""

    status: str
    report_execution: ReportExecution
    requested_fields: tuple[str, ...] = ()
    chunks: tuple[CandidateChunk, ...] = ()
    results: tuple[RetrievalResult, ...] = ()
    hydrated: bool = False
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return (
            self.status == RESOLVED
            and self.report_execution.executable
            and len(self.chunks) == 1
            and len(self.results) == 1
        )

    @property
    def selected_chunk_id(self) -> str | None:
        return self.results[0].chunk_id if self.results else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "resolved": self.resolved,
            "requested_fields": list(self.requested_fields),
            "selected_chunk_id": self.selected_chunk_id,
            "hydrated": self.hydrated,
            "detail": dict(self.detail),
            "report_execution": self.report_execution.to_dict(),
        }


class HoldingReportRelativeExecution:
    """Select and expose exactly one indexed holding-report projection."""

    def __init__(
        self,
        *,
        index: HoldingReportIndex | None,
        document_backend: Any = None,
        chunk_backend: Any = None,
        active_corpus_identity: Mapping[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.document_backend = document_backend
        self.chunk_backend = chunk_backend if chunk_backend is not None else document_backend
        self.active_corpus_identity = (
            dict(active_corpus_identity)
            if active_corpus_identity is not None
            else None
        )

    @classmethod
    def from_repository(
        cls,
        *,
        document_backend: Any,
        chunk_backend: Any | None = None,
        root: str | Path | None = None,
        index_path: str | Path = DEFAULT_INDEX_PATH,
        finality_path: str | Path = DEFAULT_FINALITY_PATH,
    ) -> "HoldingReportRelativeExecution":
        """Load the tracked artifacts and bind them to the active manifest."""

        repository = (
            Path(root)
            if root is not None
            else Path(__file__).resolve().parents[2]
        )
        index_location = _under(repository, index_path)
        finality_location = _under(repository, finality_path)
        return cls(
            index=load_index(index_location, finality_path=finality_location),
            document_backend=document_backend,
            chunk_backend=chunk_backend,
            active_corpus_identity=repository_corpus_identity(repository),
        )

    def adapt(
        self,
        question: str,
        plan: Any,
        execution: Any,
        *,
        routed_task_type: str | None,
    ) -> ReportRelativeEvidenceExecution | None:
        """Return an authoritative evidence copy, or ``None`` when gated out.

        ``None`` means the question never entered Phase 3 and must retain its
        prior behaviour.  A returned object means Phase 3 did engage; an empty
        ``results`` tuple is then an authoritative failure, not permission to
        fall back to ranked evidence.
        """

        intent = _report_relative_intent(plan)
        requested = requested_holding_fields(question, plan)
        if not _eligible(
            question,
            plan,
            intent,
            requested,
            routed_task_type=routed_task_type,
        ):
            return None

        corp_code = str(getattr(plan, "corp_code", "") or "").strip()
        reporter = str(getattr(plan, "reporter", "") or "").strip()
        period_date = _exact_period_date(plan)
        selected = execute_report_relative(
            intent,
            index=self.index,
            issuer_corp_code=corp_code,
            reporter=reporter,
            reference_date=period_date,
            receipt_date=period_date,
            active_corpus_identity=self.active_corpus_identity,
        )
        if not selected.executable or selected.record is None:
            return ReportRelativeEvidenceExecution(
                status=selected.status,
                report_execution=selected,
                requested_fields=requested,
            )

        candidate, hydrated, failure = self._projection_candidate(
            execution, selected.record
        )
        if candidate is None:
            return ReportRelativeEvidenceExecution(
                status=failure,
                report_execution=selected,
                requested_fields=requested,
                hydrated=hydrated,
            )

        validation_status = _validate_projection(candidate, selected.record)
        if validation_status is not None:
            return ReportRelativeEvidenceExecution(
                status=validation_status,
                report_execution=selected,
                requested_fields=requested,
                hydrated=hydrated,
            )

        result = _evidence_result(candidate, selected)
        return ReportRelativeEvidenceExecution(
            status=RESOLVED,
            report_execution=selected,
            requested_fields=requested,
            chunks=(candidate,),
            results=(result,),
            hydrated=hydrated,
            detail={
                "selector": selected.selection.selector,
                "projection_chunk_id": candidate.chunk_id,
                "source": "exact_hydration" if hydrated else "candidate_pool",
            },
        )

    def _projection_candidate(
        self, execution: Any, record: Any
    ) -> tuple[CandidateChunk | None, bool, str]:
        present = [
            candidate
            for candidate in (getattr(execution, "chunks", ()) or ())
            if candidate.chunk_id == record.projection_chunk_id
        ]
        if len(present) == 1:
            return present[0], False, RESOLVED
        if len(present) > 1:
            return None, False, PROJECTION_CHUNK_AMBIGUOUS

        fetch_documents = getattr(self.document_backend, "fetch_documents", None)
        get_chunks = getattr(self.chunk_backend, "get_candidate_chunks", None)
        if not callable(fetch_documents) or not callable(get_chunks):
            return None, False, HYDRATION_UNAVAILABLE
        try:
            documents = list(fetch_documents([record.doc_id]))
            hydrated = list(get_chunks(documents)) if len(documents) == 1 else []
        except Exception:  # noqa: BLE001 - deterministic lane fails closed
            return None, True, HYDRATION_UNAVAILABLE
        matches = [
            candidate
            for candidate in hydrated
            if candidate.chunk_id == record.projection_chunk_id
        ]
        if len(matches) == 1:
            return matches[0], True, RESOLVED
        if len(matches) > 1:
            return None, True, PROJECTION_CHUNK_AMBIGUOUS
        return None, True, PROJECTION_CHUNK_MISSING


def repository_corpus_identity(root: str | Path | None = None) -> dict[str, Any]:
    """Identity of the tracked manifest and frozen serving release.

    The manifest digest and holding-document count are recomputed rather than
    copied from the report index.  A missing or malformed manifest returns an
    empty identity, which makes B.2 report ``stale_index`` and therefore fails
    closed.
    """

    repository = (
        Path(root) if root is not None else Path(__file__).resolve().parents[2]
    )
    manifest = repository / "data" / "corpus" / "manifest.jsonl"
    try:
        raw = manifest.read_bytes()
        holding_count = sum(
            1
            for line in raw.decode("utf-8").splitlines()
            if line.strip() and json.loads(line).get("doc_group") == "holding"
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        return {}
    return {
        "corpus_snapshot_id": DEFAULT_CORPUS_SNAPSHOT_ID,
        "corpus_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "processed_schema_versions": list(DEFAULT_PROCESSED_SCHEMA_VERSIONS),
        "chunking_versions": list(DEFAULT_CHUNKING_VERSIONS),
        "projection_provenance_revisions": list(
            DEFAULT_PROJECTION_PROVENANCE_REVISIONS
        ),
        "source_holding_disclosure_count": holding_count,
        "source_holding_report_projection_count": (
            DEFAULT_HOLDING_REPORT_PROJECTION_COUNT
        ),
    }


def _eligible(
    question: str,
    plan: Any,
    intent: Mapping[str, Any] | None,
    requested: Sequence[str],
    *,
    routed_task_type: str | None,
) -> bool:
    if routed_task_type != ROUTED_TASK_TYPE or not intent:
        return False
    selector = str(intent.get("selector") or "")
    if (
        selector not in _EXECUTABLE_SELECTORS
        and selector != SELECTOR_SELECTED_CONTEXT
    ):
        return False
    if not _has_report_relative_wording(question, intent):
        return False
    if not _single_issuer(plan) or not str(getattr(plan, "reporter", "") or "").strip():
        return False
    if _comparison_firewall(plan):
        return False
    if str(getattr(plan, "correction_policy", "") or "") in _EXPLICIT_CORRECTION_FILTERS:
        return False
    if not requested or ACQUISITION_REQUEST_FIELDS.intersection(requested):
        return False
    return True


def _report_relative_intent(plan: Any) -> Mapping[str, Any] | None:
    evidence = getattr(plan, "evidence", None)
    payload = (
        dict(evidence).get("holding_report_relative")
        if isinstance(evidence, Mapping)
        else None
    )
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    return dict(payload) if isinstance(payload, Mapping) else None


def _has_report_relative_wording(
    question: str, intent: Mapping[str, Any]
) -> bool:
    if str(intent.get("evidence") or "").strip():
        return True
    selector = str(intent.get("selector") or "")
    if selector not in {
        SELECTOR_EXACT_REFERENCE_DATE,
        SELECTOR_EXACT_RECEIPT_DATE,
    }:
        return False
    # The frozen parser intentionally leaves ``evidence`` empty for an exact
    # date because the date owns selection.  Requiring the report noun here is
    # what distinguishes "2024-05-09 보고서의 보유비율" from a plain event-date
    # holding query.  Acquisition overlap is decided separately by the frozen
    # requested-field helper above.
    return "보고" in re.sub(r"\s+", "", str(question or ""))


def _single_issuer(plan: Any) -> bool:
    corp_code = str(getattr(plan, "corp_code", "") or "").strip()
    values = tuple(getattr(plan, "corp_codes", ()) or ())
    return bool(corp_code) and (not values or len(values) == 1)


def _comparison_firewall(plan: Any) -> bool:
    evidence = getattr(plan, "evidence", None)
    frame = (
        dict(evidence).get("comparison_frame")
        if isinstance(evidence, Mapping)
        else None
    )
    if frame in {"cross_company", "uncertain"}:
        return True
    comparison = getattr(plan, "comparison", None)
    return isinstance(comparison, Mapping) and comparison.get("type") == "company_comparison"


def _exact_period_date(plan: Any) -> str | None:
    period = getattr(plan, "period", None)
    if hasattr(period, "to_dict"):
        period = period.to_dict()
    values = dict(period) if isinstance(period, Mapping) else {}
    start = values.get("from") or values.get("from_date")
    end = values.get("to") or values.get("to_date")
    return str(start) if start and start == end else None


def _validate_projection(candidate: CandidateChunk, record: Any) -> str | None:
    chunk = candidate.chunk
    if candidate.doc_id != record.doc_id:
        return PROJECTION_IDENTITY_MISMATCH
    if str(chunk.get("doc_group") or "") != "holding":
        return PROJECTION_IDENTITY_MISMATCH
    if str(chunk.get("projection_type") or "") != "holding_report":
        return PROJECTION_IDENTITY_MISMATCH
    if str(chunk.get("corp_code") or "") != record.issuer_corp_code:
        return PROJECTION_IDENTITY_MISMATCH
    fields = dict(chunk.get("projection_fields") or {})
    reporter = fields.get("보고자/보유자") or chunk.get("reporter")
    if canonical_reporter_key(reporter) != record.reporter_key:
        return PROJECTION_IDENTITY_MISMATCH
    reference_date = fields.get("기준일/보고일") or chunk.get("reference_date")
    if _date_digits(reference_date) != record.reference_date:
        return PROJECTION_IDENTITY_MISMATCH
    if not is_citable(chunk):
        return PROJECTION_NOT_CITABLE
    return None


def _evidence_result(
    candidate: CandidateChunk, execution: ReportExecution
) -> RetrievalResult:
    metadata = dict(candidate.metadata_match.to_dict())
    metadata[PROVENANCE_KEY] = {
        "selected_for": "report_relative_holding",
        "selector": execution.selection.selector,
        "selection_status": execution.status,
        "ranked_retrieval": False,
    }
    return RetrievalResult(
        chunk_id=candidate.chunk_id,
        doc_id=candidate.doc_id,
        bm25_score=0.0,
        rank=1,
        metadata_match=metadata,
    )


def _date_digits(value: Any) -> str | None:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def _under(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


__all__ = [
    "ACQUISITION_REQUEST_FIELDS",
    "HYDRATION_UNAVAILABLE",
    "HoldingReportRelativeExecution",
    "PROJECTION_CHUNK_AMBIGUOUS",
    "PROJECTION_CHUNK_MISSING",
    "PROJECTION_IDENTITY_MISMATCH",
    "PROJECTION_NOT_CITABLE",
    "PROVENANCE_KEY",
    "ReportRelativeEvidenceExecution",
    "repository_corpus_identity",
]
