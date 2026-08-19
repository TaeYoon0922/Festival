"""Confidence-aware routing and debuggable lexical score composition."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


@dataclass(frozen=True)
class RouteDecision:
    field: str
    value: Any
    confidence: float
    mode: str
    evidence: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "confidence": self.confidence,
            "mode": self.mode,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class RetrievalRoute:
    backend_filters: Mapping[str, Any]
    hard_filters: Mapping[str, Any]
    hard_routes: Mapping[str, Any]
    soft_boosts: Mapping[str, Any]
    decisions: Mapping[str, RouteDecision]
    route_candidates: Mapping[str, float]
    section_boosts: Mapping[str, float]
    lexical_query: str
    date_range: tuple[str, str] | None
    retrieval_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_filters": dict(self.backend_filters),
            "hard_filters": dict(self.hard_filters),
            "hard_routes": dict(self.hard_routes),
            "soft_boosts": dict(self.soft_boosts),
            "route_candidates": dict(self.route_candidates),
            "section_boosts": dict(self.section_boosts),
            "lexical_query": self.lexical_query,
            "date_range": list(self.date_range) if self.date_range else None,
            "decisions": {
                key: decision.to_dict() for key, decision in self.decisions.items()
            },
            "retrieval_limit": self.retrieval_limit,
        }


class QueryRouter:
    """Apply only high-confidence routes as exclusions; keep the rest as boosts."""

    SCORE_WEIGHTS = {
        "lexical": 0.58,
        "exact_term": 0.19,
        "section": 0.16,
        "metadata": 0.05,
        "retrieval_priority": 0.02,
        "date_relevance": 0.00,
    }

    def __init__(self, *, hard_threshold: float = 0.95) -> None:
        if not 0.0 <= hard_threshold <= 1.0:
            raise ValueError("hard_threshold must be between 0 and 1")
        self.hard_threshold = hard_threshold

    def route(self, plan: Any) -> RetrievalRoute:
        backend_filters = plan.backend_filters()
        hard_filters = {
            key: value
            for key, value in {
                "company": backend_filters.get("company"),
                "corp_code": backend_filters.get("corp_code"),
                "year": backend_filters.get("year"),
                "period": backend_filters.get("period"),
            }.items()
            if value not in (None, [], ())
        }
        date_range = None
        if plan.period.from_date and plan.period.to_date:
            date_range = (plan.period.from_date, plan.period.to_date)
            if plan.period.period_type in {"receipt_date", "date_range"}:
                hard_filters["rcept_dt"] = date_range

        route_candidates = {
            route: float(plan.route_confidence.get(f"disclosure_route.{route}", 0.5))
            for route in plan.disclosure_route
        }
        decisions: dict[str, RouteDecision] = {}
        hard_routes: dict[str, Any] = {}
        soft_boosts: dict[str, Any] = {}

        if route_candidates:
            top_route = max(route_candidates, key=route_candidates.get)
            confidence = route_candidates[top_route]
            mode = "hard" if confidence >= self.hard_threshold else "soft"
            decision = RouteDecision(
                field="doc_group",
                value=top_route,
                confidence=confidence,
                mode=mode,
                evidence=plan.route_evidence.get(f"disclosure_route.{top_route}"),
            )
            decisions["doc_group"] = decision
            (hard_routes if mode == "hard" else soft_boosts)["doc_group"] = top_route

        correction_value = plan.is_correction
        if correction_value is None and plan.route_confidence.get("is_correction", 0.0) > 0:
            correction_value = True
        optional_values = {
            "doc_subtype": plan.doc_subtype,
            "section_path": plan.section_path,
            "basis": None if plan.basis == "unspecified" else plan.basis,
            "is_correction": correction_value,
        }
        for field_name, value in optional_values.items():
            if value is None:
                continue
            confidence = float(plan.route_confidence.get(field_name, 0.8))
            mode = "hard" if confidence >= self.hard_threshold else "soft"
            decision = RouteDecision(
                field=field_name,
                value=value,
                confidence=confidence,
                mode=mode,
                evidence=plan.route_evidence.get(field_name),
            )
            decisions[field_name] = decision
            (hard_routes if mode == "hard" else soft_boosts)[field_name] = value

        has_ranking_boosts = bool(soft_boosts or route_candidates or plan.section_boosts)
        retrieval_limit = max(plan.top_k * 5, 50) if has_ranking_boosts else plan.top_k
        return RetrievalRoute(
            backend_filters=backend_filters,
            hard_filters=hard_filters,
            hard_routes=hard_routes,
            soft_boosts=soft_boosts,
            decisions=decisions,
            route_candidates=route_candidates,
            section_boosts=plan.section_boosts,
            lexical_query=plan.lexical_query,
            date_range=date_range,
            retrieval_limit=retrieval_limit,
        )

    def filter_documents(
        self,
        documents: Sequence[CandidateDocument],
        route: RetrievalRoute,
    ) -> list[CandidateDocument]:
        hard = route.hard_routes
        selected: list[CandidateDocument] = []
        for document in documents:
            metadata = document.metadata
            if "doc_group" in hard and not _same(metadata.get("doc_group"), hard["doc_group"]):
                continue
            if "doc_subtype" in hard and not _same(
                metadata.get("doc_subtype"), hard["doc_subtype"]
            ):
                continue
            if "is_correction" in hard and bool(metadata.get("is_correction")) is not bool(
                hard["is_correction"]
            ):
                continue
            if "rcept_dt" in route.hard_filters and not _date_in_range(
                metadata.get("rcept_dt"), *route.hard_filters["rcept_dt"]
            ):
                continue
            selected.append(document)
        return selected

    def prepare_chunks(
        self,
        chunks: Sequence[CandidateChunk],
        route: RetrievalRoute,
    ) -> list[CandidateChunk]:
        selected: list[CandidateChunk] = []
        for candidate in chunks:
            soft = dict(candidate.metadata_match.soft_boosts)
            soft_inputs = dict(candidate.metadata_match.soft_inputs)
            group = candidate.chunk.get("doc_group")
            for group_name in route.route_candidates:
                key = f"disclosure_route.{group_name}"
                soft[key] = _same(group, group_name)
                soft_inputs[key] = group_name
            if "is_correction" in route.decisions:
                value = bool(route.decisions["is_correction"].value)
                soft["is_correction"] = bool(candidate.chunk.get("is_correction")) is value
                soft_inputs["is_correction"] = value
            if "basis" in route.decisions:
                value = route.decisions["basis"].value
                soft["basis"] = _chunk_basis_matches(candidate.chunk, value)
                soft_inputs["basis"] = value
            if "section_path" in route.hard_routes and not _chunk_section_matches(
                candidate.chunk, route.hard_routes["section_path"]
            ):
                continue
            if "basis" in route.hard_routes and not soft.get("basis", False):
                continue
            match = MetadataMatch(
                hard_filters=candidate.metadata_match.hard_filters,
                soft_boosts=soft,
                soft_inputs=soft_inputs,
                soft_score=float(sum(bool(value) for value in soft.values())),
            )
            selected.append(
                CandidateChunk(
                    chunk_id=candidate.chunk_id,
                    doc_id=candidate.doc_id,
                    chunk=candidate.chunk,
                    metadata_match=match,
                )
            )
        return selected

    def rerank(
        self,
        results: Sequence[RetrievalResult],
        route: RetrievalRoute,
        *,
        chunks: Sequence[CandidateChunk],
        top_k: int,
    ) -> list[RetrievalResult]:
        if not results:
            return []
        chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        raw_lexical = [float(result.bm25_score) for result in results]
        low, high = min(raw_lexical), max(raw_lexical)
        scored: list[tuple[float, int, RetrievalResult, dict[str, float]]] = []

        for result in results:
            candidate = chunks_by_id.get(result.chunk_id)
            chunk = candidate.chunk if candidate else {}
            lexical = (
                (float(result.bm25_score) - low) / (high - low)
                if high > low
                else 1.0
            )
            components = {
                "lexical": lexical,
                "exact_term": _exact_term_score(route.lexical_query, chunk),
                "section": _section_score(route.section_boosts, chunk),
                "metadata": _metadata_score(route, chunk, result.metadata_match),
                "retrieval_priority": _priority_score(chunk.get("retrieval_priority")),
                "date_relevance": _date_relevance(route.date_range, chunk.get("rcept_dt")),
            }
            final_score = sum(
                components[name] * weight for name, weight in self.SCORE_WEIGHTS.items()
            )
            scored.append((final_score, result.rank, result, components))

        scored.sort(key=lambda item: (-item[0], item[1], item[2].chunk_id))
        output: list[RetrievalResult] = []
        for rank, (final_score, original_rank, result, components) in enumerate(
            scored[:top_k], start=1
        ):
            match = dict(result.metadata_match)
            match["score_components"] = {
                **components,
                "weights": dict(self.SCORE_WEIGHTS),
                "final_score": final_score,
                "original_lexical_rank": original_rank,
            }
            output.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    bm25_score=result.bm25_score,
                    rank=rank,
                    metadata_match=match,
                )
            )
        return output


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _same(left: Any, right: Any) -> bool:
    return _normalized(left) == _normalized(right)


def _date_in_range(value: Any, start: str, end: str) -> bool:
    compact = re.sub(r"[^0-9]", "", str(value or ""))
    if len(compact) < 8:
        return False
    iso = f"{compact[:4]}-{compact[4:6]}-{compact[6:8]}"
    return start <= iso <= end


def _chunk_section_matches(chunk: Mapping[str, Any], requested: Any) -> bool:
    return _normalized(requested) in _normalized(_section_text(chunk))


def _chunk_basis_matches(chunk: Mapping[str, Any], requested: Any) -> bool:
    aliases = {
        "consolidated": ("연결", "consolidated"),
        "standalone": ("별도", "개별", "separate", "standalone"),
    }
    expected = aliases.get(str(requested), (str(requested),))
    metadata = chunk.get("metadata")
    scope = str(
        chunk.get("statement_scope")
        or chunk.get("basis")
        or (metadata.get("statement_scope") if isinstance(metadata, Mapping) else "")
        or ""
    )
    normalized_scope = _normalized(scope)
    return any(_normalized(value) in normalized_scope for value in expected)


def _section_text(chunk: Mapping[str, Any]) -> str:
    path = chunk.get("section_path") or []
    if isinstance(path, str):
        return path
    return " > ".join(str(item) for item in path)


def _exact_term_score(query: str, chunk: Mapping[str, Any]) -> float:
    text = _normalized(
        f"{chunk.get('content') or ''} {chunk.get('retrieval_text') or ''}"
    )
    phrase = _normalized(query)
    if phrase and phrase in text:
        return 1.0
    terms = [_normalized(term) for term in re.findall(r"[0-9A-Za-z가-힣]+", query)]
    matched = sum(bool(term and term in text) for term in terms)
    return matched / len(terms) if terms else 0.0


def _section_score(boosts: Mapping[str, float], chunk: Mapping[str, Any]) -> float:
    section = _normalized(_section_text(chunk))
    return max(
        (weight for term, weight in boosts.items() if _normalized(term) in section),
        default=0.0,
    )


def _metadata_score(
    route: RetrievalRoute,
    chunk: Mapping[str, Any],
    metadata_match: Mapping[str, Any],
) -> float:
    scores: list[float] = []
    group = chunk.get("doc_group")
    for route_name, confidence in route.route_candidates.items():
        if _same(group, route_name):
            scores.append(confidence)
    flags = dict(metadata_match.get("soft_boosts") or {})
    for field_name in route.soft_boosts:
        if flags.get(field_name):
            scores.append(route.decisions[field_name].confidence)
    return max(scores, default=0.0)


def _priority_score(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return min(max(float(value), 0.0), 1.0)
    return {"high": 1.0, "normal": 0.5, "low": 0.0}.get(
        str(value or "").casefold(), 0.5
    )


def _date_relevance(date_range: tuple[str, str] | None, value: Any) -> float:
    if not date_range:
        return 0.0
    return 1.0 if _date_in_range(value, *date_range) else 0.0
