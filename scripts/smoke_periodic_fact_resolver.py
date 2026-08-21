"""Read-only production-data smoke test for the periodic fact pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.reasoning.evidence_builder import EvidenceBuilder, EvidenceSet
from app.reasoning.periodic_fact_resolver import (
    PeriodicFact,
    PeriodicFactResolution,
    PeriodicFactResolver,
)
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from app.retrieval.hybrid import HybridQueryExecutor, HybridRetrievalConfig
from app.retrieval.postgres_backend import PostgresBackend


DEFAULT_QUERIES = (
    "삼성전자 DX 부문의 주요 제품은 무엇인가",
    "레인보우로보틱스 HUBO 이족보행 로봇 사업 설명",
    "한미반도체 TC BONDER 반도체 제조 장비 사업",
)
_PERIODIC_GROUP_TYPES = {
    "periodic_repeated_fact",
    "document_evidence",
    "standalone_evidence",
}
_GENERIC_QUERY_TERMS = {
    "무엇인가",
    "무엇",
    "주요",
    "제품",
    "설명",
    "사업",
    "부문",
    "관련",
    "대한",
    "알려줘",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Query Understanding -> frozen Hybrid Retrieval -> Evidence Builder "
            "-> Periodic Fact Resolver without database writes."
        )
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Repeat to override the three default periodic smoke queries.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lexical-top-n", type=int, default=50)
    parser.add_argument("--vector-top-n", type=int, default=50)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "periodic_resolver_smoke"
            / "periodic_resolver_smoke.json"
        ),
    )
    args = parser.parse_args()

    embedding_config = EmbeddingConfig.from_env()
    backend = PostgresBackend()
    understanding = QueryUnderstanding(company_resolver=backend.resolve_company)
    executor = HybridQueryExecutor(
        backend,
        create_embedding_provider(embedding_config),
        embedding_config,
        config=HybridRetrievalConfig(
            lexical_top_n=args.lexical_top_n,
            vector_top_n=args.vector_top_n,
            final_top_k=args.top_k,
            rerank_mode="legacy",
        ),
    )
    evidence_builder = EvidenceBuilder()
    resolver = PeriodicFactResolver()

    rows = []
    for query in args.queries or DEFAULT_QUERIES:
        stage = "query_understanding"
        try:
            plan = understanding.understand(query, top_k=args.top_k)
            stage = "hybrid_retrieval"
            execution = executor.execute(plan)
            retrieval_before = _retrieval_snapshot(execution.results)
            stage = "evidence_builder"
            evidence = evidence_builder.build(execution)
            evidence_before = copy.deepcopy(evidence.to_dict())
            grouping_before = _grouping_snapshot(evidence)
            stage = "periodic_fact_resolver"
            resolution = resolver.resolve(evidence)
            rows.append(
                _query_report(
                    query=query,
                    plan=plan.to_dict(),
                    evidence=evidence,
                    resolution=resolution,
                    retrieval_before=retrieval_before,
                    retrieval_after=_retrieval_snapshot(execution.results),
                    evidence_before=evidence_before,
                    grouping_before=grouping_before,
                )
            )
        except Exception as error:  # diagnostic boundary; other queries still run
            rows.append(
                {
                    "query": query,
                    "status": "failed",
                    "error": {"stage": stage, "type": type(error).__name__},
                }
            )

    successful = [row for row in rows if row.get("status") == "ok"]
    report = {
        "mode": "read_only_periodic_resolver_smoke",
        "embedding": {
            "provider": embedding_config.provider,
            "model": embedding_config.model,
            "version": embedding_config.version,
            "dimensions": embedding_config.dimensions,
        },
        "hybrid": {
            "rerank_mode": "legacy",
            "top_k": args.top_k,
            "lexical_top_n": args.lexical_top_n,
            "vector_top_n": args.vector_top_n,
        },
        "query_count": len(rows),
        "successful_query_count": len(successful),
        "failed_query_count": len(rows) - len(successful),
        "all_invariants_preserved": bool(successful)
        and len(successful) == len(rows)
        and all(row["validation"]["all_invariants_preserved"] for row in successful),
        "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(_concise_summary(report), ensure_ascii=False, indent=2))
    print(f"[output] {args.output}", flush=True)
    if len(successful) != len(rows):
        raise SystemExit(1)


def _query_report(
    *,
    query: str,
    plan: Mapping[str, Any],
    evidence: EvidenceSet,
    resolution: PeriodicFactResolution,
    retrieval_before: Sequence[Mapping[str, Any]],
    retrieval_after: Sequence[Mapping[str, Any]],
    evidence_before: Mapping[str, Any],
    grouping_before: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fact_rows = [_fact_report(fact) for fact in resolution.facts]
    eligible_items = _eligible_periodic_items(evidence)
    provenance = _provenance_validation(resolution, eligible_items)
    answerability = _answerability(plan, resolution)
    retrieval_unchanged = list(retrieval_before) == list(retrieval_after)
    evidence_unchanged = dict(evidence_before) == evidence.to_dict()
    grouping_unchanged = list(grouping_before) == _grouping_snapshot(evidence)
    evidence_order_unchanged = tuple(
        row["chunk_id"] for row in retrieval_before
    ) == evidence.retrieval_order
    eligible_ids = [item.chunk_id for item in eligible_items]
    resolved_ids = [
        source.chunk_id for fact in resolution.facts for source in fact.sources
    ]
    periodic_sources_preserved = Counter(eligible_ids) == Counter(resolved_ids)
    source_values_preserved = _source_values_preserved(resolution, eligible_items)
    no_arbitrary_period_selection = _no_arbitrary_period_selection(
        resolution, evidence
    )
    invariants = {
        "retrieval_rank_score_order_unchanged": retrieval_unchanged,
        "evidence_set_unchanged": evidence_unchanged,
        "evidence_grouping_unchanged": grouping_unchanged,
        "retrieval_to_evidence_order_unchanged": evidence_order_unchanged,
        "eligible_periodic_sources_preserved": periodic_sources_preserved,
        "source_rank_score_identity_preserved": source_values_preserved,
        "unique_chunk_identity_preserved": provenance["unique_chunk_identity_preserved"],
        "provenance_complete": provenance["all_paths_preserved"],
        "no_arbitrary_period_selection": no_arbitrary_period_selection,
    }
    invariants["all_invariants_preserved"] = all(invariants.values())
    groups = evidence.evidence_groups
    return {
        "query": query,
        "status": "ok",
        "query_analysis": {
            "query": query,
            "company": plan.get("company") or plan.get("companies"),
            "task_type": plan.get("task_type"),
            "periodic_intent": _plan_evidence(plan).get("periodic_intent"),
            "requested_metric_or_fact": (
                resolution.requested_fact or plan.get("metric")
            ),
            "explicit_temporal_constraint": copy.deepcopy(
                evidence.ambiguity.get("temporal_constraint") or {}
            ),
        },
        "evidence_set": {
            "raw_candidate_count": evidence.raw_candidate_count,
            "evidence_group_count": len(groups),
            "periodic_repeated_fact_group_count": _group_count(
                groups, "periodic_repeated_fact"
            ),
            "document_evidence_group_count": _group_count(
                groups, "document_evidence"
            ),
            "standalone_periodic_group_count": sum(
                group.group_type == "standalone_evidence"
                and any(item.doc_group == "periodic" for item in group.items)
                for group in groups
            ),
            "temporal_ambiguity": bool(
                evidence.ambiguity.get("temporal_ambiguity")
            ),
            "warnings": list(evidence.warnings),
            "retrieval_order": list(evidence.retrieval_order),
            "groups": [_group_row(group) for group in groups],
        },
        "periodic_fact_resolution": {
            "fact_count": len(resolution.facts),
            "matching_fact_count": resolution.matching_fact_count,
            "temporal_ambiguity": resolution.temporal_ambiguity,
            "unresolved_requirements": list(resolution.unresolved_requirements),
            "warnings": list(resolution.warnings),
        },
        "facts": fact_rows,
        "answerability": answerability,
        "provenance_validation": provenance,
        "validation": invariants,
        "retrieval_snapshot": list(retrieval_before),
    }


def _fact_report(fact: PeriodicFact) -> dict[str, Any]:
    row = fact.to_dict()
    row["period_evolution"] = fact.period_evolution
    row["fact_conflict"] = fact.fact_conflict
    row["conflict_type"] = fact.conflict_type
    row["sources"] = copy.deepcopy(row["fact_provenance"])
    return row


def _retrieval_snapshot(results: Sequence[Any]) -> list[dict[str, Any]]:
    output = []
    for result in results:
        metadata = dict(result.metadata_match or {})
        hybrid = dict(metadata.get("hybrid") or {})
        output.append(
            {
                "chunk_id": result.chunk_id,
                "doc_id": result.doc_id,
                "rank": int(result.rank),
                "bm25_score": float(result.bm25_score),
                "final_score": _number(hybrid.get("final_score")),
                "retrieval_score": _number(hybrid.get("retrieval_score")),
                "rrf_score": _number(hybrid.get("rrf_score")),
            }
        )
    return output


def _grouping_snapshot(evidence: EvidenceSet) -> list[dict[str, Any]]:
    return [_group_row(group) for group in evidence.evidence_groups]


def _group_row(group: Any) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "group_type": group.group_type,
        "member_chunk_ids": list(group.member_chunk_ids),
        "primary_chunk_id": group.primary_evidence.chunk_id,
        "doc_ids": list(group.doc_ids),
    }


def _eligible_periodic_items(evidence: EvidenceSet) -> list[Any]:
    return [
        item
        for group in evidence.evidence_groups
        if group.group_type in _PERIODIC_GROUP_TYPES
        for item in group.items
        if item.doc_group == "periodic" and not _is_holding_item(item)
    ]


def _is_holding_item(item: Any) -> bool:
    holding = item.holding
    return bool(
        holding.get("projection_type") in {"holding_detail_row", "holding_report"}
        or holding.get("reporter")
        or holding.get("reference_date")
    )


def _source_values_preserved(
    resolution: PeriodicFactResolution, eligible_items: Sequence[Any]
) -> bool:
    expected = {
        item.chunk_id: (item.doc_id, item.retrieval_rank, item.retrieval_score)
        for item in eligible_items
    }
    actual = {
        source.chunk_id: (source.doc_id, source.retrieval_rank, source.retrieval_score)
        for fact in resolution.facts
        for source in fact.sources
    }
    return expected == actual


def _provenance_validation(
    resolution: PeriodicFactResolution, eligible_items: Sequence[Any]
) -> dict[str, Any]:
    item_by_id = {item.chunk_id: item for item in eligible_items}
    paths = []
    for fact_index, fact in enumerate(resolution.facts, start=1):
        for source in fact.sources:
            item = item_by_id.get(source.chunk_id)
            source_chunk = source.provenance.get("source_chunk") or {}
            refs_preserved = bool(item) and list(source.source_refs) == list(item.source_refs)
            path = {
                "fact_index": fact_index,
                "evidence_group_id": fact.evidence_group_id,
                "chunk_id": source.chunk_id,
                "doc_id": source.doc_id,
                "source_refs": copy.deepcopy(list(source.source_refs)),
                "evidence_item_found": item is not None,
                "source_chunk_id": source.provenance.get("source_chunk_id"),
                "source_chunk_doc_id": source.provenance.get("source_doc_id"),
                "original_source_chunk_id": source_chunk.get("chunk_id"),
                "original_source_doc_id": source_chunk.get("doc_id"),
                "source_refs_preserved": refs_preserved,
            }
            path["path_preserved"] = bool(
                item
                and source.provenance.get("source_chunk_id") == source.chunk_id
                and source.provenance.get("source_doc_id") == source.doc_id
                and (source_chunk.get("chunk_id") in {None, source.chunk_id})
                and (source_chunk.get("doc_id") in {None, source.doc_id})
                and refs_preserved
            )
            paths.append(path)
    actual_ids = [path["chunk_id"] for path in paths]
    same_doc_chunk_pairs = [(path["doc_id"], path["chunk_id"]) for path in paths]
    return {
        "has_paths": bool(paths),
        "all_paths_preserved": all(path["path_preserved"] for path in paths),
        "unique_chunk_identity_preserved": len(actual_ids) == len(set(actual_ids))
        and len(same_doc_chunk_pairs) == len(set(same_doc_chunk_pairs)),
        "paths": paths,
    }


def _no_arbitrary_period_selection(
    resolution: PeriodicFactResolution,
    evidence: EvidenceSet,
) -> bool:
    constraint = evidence.ambiguity.get("temporal_constraint") or {}
    if constraint.get("explicit"):
        return True
    group_by_id = {group.group_id: group for group in evidence.evidence_groups}
    for fact in resolution.facts:
        if any(match is not None for _, match in fact.temporal_matches):
            return False
        group = group_by_id.get(fact.evidence_group_id)
        if group is None:
            return False
        eligible = [
            item.chunk_id
            for item in group.items
            if item.doc_group == "periodic" and not _is_holding_item(item)
            and (item.corp_code or item.company_id or item.corp_name or item.doc_id)
            in {
                fact.corp_code,
                fact.company_id,
                fact.corp_name,
                fact.doc_ids[0] if fact.doc_ids else None,
            }
        ]
        if list(fact.evidence_chunk_ids) != eligible:
            return False
    return True


def _answerability(
    plan: Mapping[str, Any], resolution: PeriodicFactResolution
) -> dict[str, Any]:
    terms = _query_terms(plan)
    section_boosts = {
        _normalize(key)
        for key, value in dict(plan.get("section_boosts") or {}).items()
        if _number(value) is not None and float(value) > 0
    }
    supporting = []
    for index, fact in enumerate(resolution.facts, start=1):
        content = " ".join(source.fact_text for source in fact.sources)
        normalized_content = _normalize(content)
        normalized_section = _normalize(" ".join(fact.section_path))
        matched_terms = [term for term in terms if _normalize(term) in normalized_content]
        section_aligned = any(
            section and section in normalized_section for section in section_boosts
        )
        has_content = bool(normalized_content)
        supports = has_content and (
            bool(matched_terms) if terms else section_aligned or has_content
        )
        if supports:
            supporting.append(
                {
                    "fact_index": index,
                    "matched_query_terms": matched_terms,
                    "section_aligned": section_aligned,
                    "evidence_chunk_ids": list(fact.evidence_chunk_ids),
                }
            )
    return {
        "answerable_evidence": bool(supporting),
        "basis": "top_k_fact_content_and_query_terms",
        "query_evidence_terms": terms,
        "supporting_facts": supporting,
        "strict_gold_rank_used": False,
    }


def _query_terms(plan: Mapping[str, Any]) -> list[str]:
    query = str(plan.get("lexical_query") or plan.get("raw_query") or "")
    companies = {
        _normalize(value)
        for value in (
            plan.get("company"),
            *(plan.get("companies") or []),
        )
        if value
    }
    output = []
    for term in re.findall(r"[0-9A-Za-z가-힣]+", query):
        normalized = _normalize(term)
        if (
            len(normalized) < 2
            or normalized in {_normalize(value) for value in _GENERIC_QUERY_TERMS}
            or normalized in companies
        ):
            continue
        output.append(term)
    return list(dict.fromkeys(output))


def _group_count(groups: Sequence[Any], group_type: str) -> int:
    return sum(group.group_type == group_type for group in groups)


def _plan_evidence(plan: Mapping[str, Any]) -> dict[str, Any]:
    value = plan.get("evidence")
    return dict(value) if isinstance(value, Mapping) else {}


def _concise_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    queries = []
    for row in report["queries"]:
        if row.get("status") != "ok":
            queries.append(
                {
                    "query": row["query"],
                    "status": "failed",
                    "error": row.get("error"),
                }
            )
            continue
        facts = row["facts"]
        queries.append(
            {
                "query": row["query"],
                "status": "ok",
                "fact_count": row["periodic_fact_resolution"]["fact_count"],
                "matching_fact_count": row["periodic_fact_resolution"][
                    "matching_fact_count"
                ],
                "repeated_fact_count": sum(
                    bool(fact["repeated_across_periods"]) for fact in facts
                ),
                "conflict_count": sum(bool(fact["fact_conflict"]) for fact in facts),
                "period_evolution_count": sum(
                    bool(fact["period_evolution"]) for fact in facts
                ),
                "answerable_evidence": row["answerability"]["answerable_evidence"],
                "provenance_preserved": row["provenance_validation"][
                    "all_paths_preserved"
                ],
                "all_invariants_preserved": row["validation"][
                    "all_invariants_preserved"
                ],
            }
        )
    return {
        "query_count": report["query_count"],
        "successful_query_count": report["successful_query_count"],
        "failed_query_count": report["failed_query_count"],
        "all_invariants_preserved": report["all_invariants_preserved"],
        "queries": queries,
    }


def _normalize(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").casefold())


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
