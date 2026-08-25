"""Backend-neutral query plans and their retrieval execution flow."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    ChunkBackend,
    MetadataBackend,
    RetrievalResult,
    Retriever,
)


_CORRECTION_POLICIES = {"any", "corrected_only", "original_only", "latest_preferred"}
_BASIS_VALUES = {"consolidated", "standalone", "unspecified"}


@dataclass(frozen=True)
class QueryPeriod:
    """Period semantics kept separate from the backend's legacy year/month inputs."""

    year: int | None = None
    quarter: int | None = None
    from_date: str | None = None
    to_date: str | None = None
    period_type: str | None = None

    def __post_init__(self) -> None:
        if self.year is not None and (
            not _is_integer(self.year) or not 1900 <= self.year <= 2100
        ):
            raise ValueError("period year must be an integer between 1900 and 2100")
        if self.quarter is not None and (
            not _is_integer(self.quarter) or not 1 <= self.quarter <= 4
        ):
            raise ValueError("quarter must be between 1 and 4")
        for value in (self.from_date, self.to_date):
            if value is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                raise ValueError("period dates must use YYYY-MM-DD")
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("period from_date must not be after to_date")

    @property
    def is_fiscal(self) -> bool:
        return self.period_type in {"fiscal_year", "fiscal_quarter"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "year": self.year,
            "quarter": self.quarter,
            "from": self.from_date,
            "to": self.to_date,
            "period_type": self.period_type,
        }


@dataclass(frozen=True)
class QueryPlan:
    """Deterministic, serializable understanding of one disclosure question.

    ``query`` is the lexical query retained for backward compatibility with the
    retrieval interface. ``raw_query`` is the untouched user question. Metadata
    consumed into this plan is safely removed from the lexical query by the
    query-understanding layer.
    """

    query: str
    raw_query: str | None = None
    company: str | None = None
    companies: tuple[str, ...] = ()
    corp_code: str | None = None
    corp_codes: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    period: QueryPeriod | int | tuple[int, int] | None = None
    task_type: str | None = None
    metric: str | None = None
    event_type: str | None = None
    disclosure_route: tuple[str, ...] | str = ()
    basis: str = "unspecified"
    reporter: str | None = None
    correction_policy: str = "any"
    comparison: Mapping[str, Any] | str | None = None
    doc_subtype: str | None = None
    section_path: str | None = None
    section_boosts: Mapping[str, float] = field(default_factory=dict)
    route_confidence: Mapping[str, float] = field(default_factory=dict)
    route_evidence: Mapping[str, Any] = field(default_factory=dict)
    top_k: int = 10
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lexical_query = self.query.strip()
        if not lexical_query:
            raise ValueError("query must not be empty")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool) or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.correction_policy not in _CORRECTION_POLICIES:
            raise ValueError(
                "correction_policy must be one of "
                + ", ".join(sorted(_CORRECTION_POLICIES))
            )

        basis = "standalone" if self.basis == "separate" else self.basis
        if basis not in _BASIS_VALUES:
            raise ValueError("basis must be consolidated, standalone, or unspecified")

        companies = _unique_strings((*self.companies, *((self.company,) if self.company else ())))
        corp_codes = _unique_strings(
            (*self.corp_codes, *((self.corp_code,) if self.corp_code else ()))
        )
        years = tuple(sorted({int(year) for year in self.years}))
        period = _coerce_period(self.period, years)
        if period.is_fiscal and period.year is not None:
            years = tuple(sorted({*years, period.year}))

        routes = (
            (self.disclosure_route,)
            if isinstance(self.disclosure_route, str) and self.disclosure_route
            else _unique_strings(self.disclosure_route)
        )
        confidence = {str(key): float(value) for key, value in self.route_confidence.items()}
        if any(not 0.0 <= value <= 1.0 for value in confidence.values()):
            raise ValueError("route confidence values must be between 0 and 1")
        section_boosts = {
            str(key): float(value) for key, value in self.section_boosts.items()
        }
        if any(not 0.0 <= value <= 1.0 for value in section_boosts.values()):
            raise ValueError("section boost values must be between 0 and 1")

        object.__setattr__(self, "query", lexical_query)
        object.__setattr__(self, "raw_query", (self.raw_query or lexical_query).strip())
        object.__setattr__(self, "companies", companies)
        object.__setattr__(self, "company", companies[0] if len(companies) == 1 else None)
        object.__setattr__(self, "corp_codes", corp_codes)
        object.__setattr__(self, "corp_code", corp_codes[0] if len(corp_codes) == 1 else None)
        object.__setattr__(self, "years", years)
        object.__setattr__(self, "period", period)
        object.__setattr__(self, "disclosure_route", routes)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "comparison", _copy_comparison(self.comparison))
        object.__setattr__(self, "section_boosts", section_boosts)
        object.__setattr__(self, "route_confidence", confidence)
        object.__setattr__(self, "route_evidence", dict(self.route_evidence))
        object.__setattr__(self, "evidence", dict(self.evidence))

    @property
    def lexical_query(self) -> str:
        return self.query

    @property
    def base_year(self) -> int | None:
        return self.period.year if self.period.is_fiscal else None

    @property
    def doc_group(self) -> str | None:
        return self.disclosure_route[0] if self.disclosure_route else None

    @property
    def is_correction(self) -> bool | None:
        if self.correction_policy == "corrected_only":
            return True
        if self.correction_policy == "original_only":
            return False
        return None

    @property
    def event(self) -> str | None:
        return self.event_type

    def backend_filters(self) -> dict[str, Any]:
        """Translate into the existing ``MetadataBackend`` contract unchanged."""

        period_month = self.period.quarter * 3 if self.period.quarter else None
        return {
            "company": list(self.companies) or None,
            "year": list(self.years) if self.period.is_fiscal and self.years else None,
            "period": period_month,
            "doc_group": self.doc_group,
            "doc_subtype": self.doc_subtype,
            "is_correction": self.is_correction,
            "corp_code": list(self.corp_codes) or None,
            "section_path": self.section_path,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "companies": list(self.companies),
            "corp_code": self.corp_code,
            "corp_codes": list(self.corp_codes),
            "task_type": self.task_type,
            "period": self.period.to_dict(),
            "metric": self.metric,
            "event_type": self.event_type,
            "disclosure_route": list(self.disclosure_route),
            "basis": self.basis,
            "reporter": self.reporter,
            "correction_policy": self.correction_policy,
            "comparison": _copy_comparison(self.comparison),
            "raw_query": self.raw_query,
            "lexical_query": self.lexical_query,
            "doc_subtype": self.doc_subtype,
            "section_path": self.section_path,
            "section_boosts": dict(self.section_boosts),
            "route_confidence": dict(self.route_confidence),
            "route_evidence": dict(self.route_evidence),
            "top_k": self.top_k,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True)
class QueryExecution:
    plan: QueryPlan
    documents: Sequence[CandidateDocument]
    chunks: Sequence[CandidateChunk]
    results: Sequence[RetrievalResult]
    routing: Mapping[str, Any] = field(default_factory=dict)
    correction_expansion: Mapping[str, Any] = field(default_factory=dict)


class QueryExecutor:
    """Execute a plan through stable backends plus the confidence-aware router."""

    def __init__(
        self,
        metadata_backend: MetadataBackend,
        chunk_backend: ChunkBackend | None = None,
        retriever: Retriever | None = None,
        *,
        router: Any | None = None,
        correction_expander: Any | None = None,
    ) -> None:
        from app.reasoning.router import QueryRouter

        self._metadata_backend = metadata_backend
        self._chunk_backend = (
            chunk_backend
            if chunk_backend is not None
            else _require_method(metadata_backend, "get_candidate_chunks")
        )
        self._retriever = (
            retriever
            if retriever is not None
            else _require_method(metadata_backend, "retrieve")
        )
        self._router = router or QueryRouter()
        self._correction_expander = correction_expander

    def execute(self, plan: QueryPlan) -> QueryExecution:
        route = self._router.route(plan)
        documents = self._metadata_backend.get_candidate_documents(**route.backend_filters)
        documents = self._router.filter_documents(documents, route)
        chunks = self._chunk_backend.get_candidate_chunks(documents)
        chunks = self._router.prepare_chunks(chunks, route)
        results = self._retriever.retrieve(
            plan.lexical_query,
            chunks,
            top_k=route.retrieval_limit,
        )
        results = self._router.rerank(
            results,
            route,
            chunks=chunks,
            document_metadata={document.doc_id: document.metadata for document in documents},
            top_k=plan.top_k,
        )
        # Documents the question's own metadata window excluded but the
        # correction graph links to what was retrieved.
        from app.retrieval.correction_expansion import (
            CorrectionExpansion,
            apply_expansion,
        )

        expansion = (
            self._correction_expander.expand(
                plan, documents=documents, chunks=chunks, results=results
            )
            if self._correction_expander is not None
            else CorrectionExpansion()
        )
        chunks, results = apply_expansion(expansion, chunks, results)
        return QueryExecution(
            plan=plan,
            documents=documents,
            chunks=chunks,
            results=results,
            routing=_routing(self._router, route, documents),
            correction_expansion=expansion.to_dict(),
        )


def _routing(
    router: Any,
    route: Any,
    documents: Sequence[CandidateDocument],
) -> dict[str, Any]:
    """Record the route plus the correction chains its candidates belong to.

    The correction block is omitted when no graph is wired, so an execution
    trace keeps its previous shape.
    """

    correction = router.correction_summary(documents, route)
    return {
        **route.to_dict(),
        **({"correction": correction} if correction else {}),
    }


def _coerce_period(
    value: QueryPeriod | int | tuple[int, int] | None,
    years: tuple[int, ...],
) -> QueryPeriod:
    if isinstance(value, QueryPeriod):
        return value
    if value is None:
        return QueryPeriod(
            year=years[0] if len(years) == 1 else None,
            period_type="fiscal_year" if len(years) == 1 else None,
        )
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("period tuple must contain (year, month)")
        year, month = value
    else:
        year = years[0] if len(years) == 1 else None
        month = value
    if not _is_integer(month) or month not in {3, 6, 9, 12}:
        raise ValueError("period month must be one of 3, 6, 9, or 12")
    return QueryPeriod(year=year, quarter=month // 3, period_type="fiscal_quarter")


def _unique_strings(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _copy_comparison(value: Mapping[str, Any] | str | None) -> Mapping[str, Any] | str | None:
    return dict(value) if isinstance(value, Mapping) else value


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_method(backend: object, method: str) -> Any:
    if not callable(getattr(backend, method, None)):
        raise TypeError(
            f"{type(backend).__name__} must implement {method}() or a separate "
            "backend must be supplied"
        )
    return backend
