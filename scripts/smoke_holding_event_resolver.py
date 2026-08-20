"""Read-only production-data smoke test for the holding resolution pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.reasoning.evidence_builder import EvidenceBuilder, EvidenceGroup, EvidenceSet
from app.reasoning.holding_event_resolver import HoldingEventResolver, HoldingResolution
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from app.retrieval.hybrid import HybridQueryExecutor, HybridRetrievalConfig
from app.retrieval.postgres_backend import PostgresBackend


DEFAULT_QUERIES = (
    "효성중공업 국민연금기금 변동일 변동후 주식수",
    "이마트 국민연금기금 변동일 감소 후 주식수",
    "LG생활건강 국민연금기금 변동일 감소 후 주식수",
)
_PROVENANCE_FIELDS = ("reference_date", "change_shares", "after_shares")
_MISSING_MARKERS = {"", "-", "--", "–", "—", "unknown", "미상", "해당없음"}
_NUMERIC_LABELS = {
    "직전 보유주식수": "before_shares",
    "증감주식수": "change_shares",
    "보유주식수": "after_shares",
    "직전 보유비율": "before_ratio",
    "보유비율": "after_ratio",
    "증감비율": "change_ratio",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Query Understanding -> frozen Hybrid Retrieval -> Evidence Builder "
            "-> Holding Event Resolver without database writes."
        )
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        help="Repeat to override the three default smoke queries.",
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
            / "holding_resolver_smoke"
            / "holding_resolver_smoke.json"
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
    resolver = HoldingEventResolver()

    rows = []
    for query in args.queries or DEFAULT_QUERIES:
        print(f"[smoke] {query}", flush=True)
        plan = understanding.understand(query, top_k=args.top_k)
        execution = executor.execute(plan)
        retrieval_before = _retrieval_snapshot(execution.results)
        evidence = evidence_builder.build(execution)
        evidence_before = copy.deepcopy(evidence.to_dict())
        grouping_before = _grouping_snapshot(evidence)
        resolution = resolver.resolve(evidence)
        retrieval_after = _retrieval_snapshot(execution.results)
        evidence_after = evidence.to_dict()
        grouping_after = _grouping_snapshot(evidence)
        rows.append(
            _query_report(
                query=query,
                plan=plan.to_dict(),
                evidence=evidence,
                resolution=resolution,
                retrieval_before=retrieval_before,
                retrieval_after=retrieval_after,
                evidence_unchanged=evidence_before == evidence_after,
                grouping_unchanged=grouping_before == grouping_after,
            )
        )

    report = {
        "mode": "read_only_holding_resolver_smoke",
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
        "all_invariants_preserved": all(
            row["validation"]["retrieval_unchanged"]
            and row["validation"]["evidence_set_unchanged"]
            and row["validation"]["grouping_unchanged"]
            for row in rows
        ),
        "queries": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[output] {args.output}", flush=True)


def _query_report(
    *,
    query: str,
    plan: Mapping[str, Any],
    evidence: EvidenceSet,
    resolution: HoldingResolution,
    retrieval_before: Sequence[Mapping[str, Any]],
    retrieval_after: Sequence[Mapping[str, Any]],
    evidence_unchanged: bool,
    grouping_unchanged: bool,
) -> dict[str, Any]:
    holding_groups = [
        group for group in evidence.evidence_groups if group.group_type == "holding_event"
    ]
    event_rows = [_event_report(event) for event in resolution.events]
    conflict_rows = _conflict_analysis(event_rows)
    provenance = [
        {
            "event_index": index,
            "doc_id": event.doc_id,
            "reference_date": event.reference_date,
            "fields": {
                field: _provenance_check(event.field_provenance.get(field))
                for field in _PROVENANCE_FIELDS
            },
        }
        for index, event in enumerate(resolution.events, start=1)
    ]
    duplicate_event_keys = _duplicate_event_keys(event_rows)
    different_date_groups = _groups_with_multiple_reference_dates(holding_groups)
    different_date_merge = bool(different_date_groups) or any(
        "reference_date" in event.conflicting_fields for event in resolution.events
    )
    missing_zero = _missing_value_zero_coercion(holding_groups, resolution)
    missing_provenance = [
        {
            "event_index": event["event_index"],
            "field": field_name,
        }
        for event in provenance
        for field_name, field in event["fields"].items()
        if field["has_value"] and not field["connected_to_source_row"]
    ]
    return {
        "query": query,
        "query_analysis": {
            "task_type": plan.get("task_type"),
            "requested_holding_fields": list(resolution.requested_fields),
            "explicit_temporal_constraint": copy.deepcopy(
                evidence.ambiguity.get("temporal_constraint") or {}
            ),
            "requested_direction": _requested_direction(query),
        },
        "evidence_set": {
            "raw_candidate_count": evidence.raw_candidate_count,
            "evidence_group_count": len(evidence.evidence_groups),
            "holding_event_group_count": len(holding_groups),
            "temporal_ambiguity": bool(
                evidence.ambiguity.get("temporal_ambiguity")
            ),
            "warnings": list(evidence.warnings),
            "retrieval_order": list(evidence.retrieval_order),
            "groups": [
                {
                    "group_id": group.group_id,
                    "group_type": group.group_type,
                    "member_chunk_ids": list(group.member_chunk_ids),
                    "doc_ids": list(group.doc_ids),
                }
                for group in evidence.evidence_groups
            ],
        },
        "holding_resolution": {
            "event_count": len(resolution.events),
            "matching_event_count": resolution.matching_event_count,
            "temporal_ambiguity": resolution.temporal_ambiguity,
            "unresolved_fields": list(resolution.unresolved_fields),
            "warnings": list(resolution.warnings),
        },
        "events": event_rows,
        "provenance_validation": provenance,
        "conflict_analysis": conflict_rows,
        "validation": {
            "retrieval_unchanged": list(retrieval_before) == list(retrieval_after),
            "evidence_set_unchanged": evidence_unchanged,
            "grouping_unchanged": grouping_unchanged,
            "duplicate_same_event_keys": duplicate_event_keys,
            "same_event_over_generation": bool(duplicate_event_keys),
            "different_dates_merged": different_date_merge,
            "groups_with_multiple_reference_dates": different_date_groups,
            "missing_value_zero_coercion_detected": missing_zero,
            "required_provenance_complete": not missing_provenance,
            "missing_required_provenance": missing_provenance,
        },
        "retrieval_snapshot": list(retrieval_before),
    }


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
                "final_score": hybrid.get("final_score"),
                "retrieval_score": hybrid.get("retrieval_score"),
                "rrf_score": hybrid.get("rrf_score"),
            }
        )
    return output


def _grouping_snapshot(evidence: EvidenceSet) -> list[dict[str, Any]]:
    return [
        {
            "group_id": group.group_id,
            "group_type": group.group_type,
            "members": list(group.member_chunk_ids),
            "primary": group.primary_evidence.chunk_id,
        }
        for group in evidence.evidence_groups
    ]


def _provenance_check(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "has_value": False,
            "field_conflict": False,
            "connected_to_source_row": False,
            "sources": [],
            "paths": [],
        }
    serialized = value.to_dict()
    sources = serialized["sources"]
    paths = [
        {
            "field": serialized["field_name"],
            "evidence_chunk_id": source.get("chunk_id"),
            "reference_kind": (
                "projection_field_ref"
                if source.get("direct_field_ref")
                else "source_ref"
            ),
            "table_id": ref.get("table_id"),
            "row_start": ref.get("row_start"),
            "row_end": ref.get("row_end"),
        }
        for source in sources
        for ref in source.get("source_refs") or []
    ]
    return {
        "has_value": serialized["value"] is not None,
        "field_conflict": serialized["field_conflict"],
        "connected_to_source_row": bool(sources)
        and all(
            any(
                ref.get("table_id") is not None
                and ref.get("row_start") is not None
                and ref.get("row_end") is not None
                for ref in source.get("source_refs") or []
            )
            for source in sources
        ),
        "sources": sources,
        "paths": paths,
        "alternatives": serialized["alternatives"],
    }


def _event_report(event: Any) -> dict[str, Any]:
    serialized = event.to_dict()
    serialized["query_direction_match"] = serialized.get("direction_match")
    return serialized


def _conflict_analysis(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, event in enumerate(events, start=1):
        provenance = dict(event.get("field_provenance") or {})
        for field in event.get("conflicting_fields") or []:
            alternatives = provenance.get(field, {}).get("alternatives") or []
            ref_sets = [
                {
                    (
                        ref.get("table_id"),
                        ref.get("row_start"),
                        ref.get("row_end"),
                    )
                    for source in alternative.get("sources") or []
                    for ref in source.get("source_refs") or []
                }
                for alternative in alternatives
            ]
            non_empty_ref_sets = [refs for refs in ref_sets if refs]
            same_refs = bool(non_empty_ref_sets) and all(
                refs == non_empty_ref_sets[0] for refs in non_empty_ref_sets
            )
            if not non_empty_ref_sets:
                cause = "untraceable_conflict"
            elif same_refs:
                cause = "projection_representation_conflict"
            else:
                cause = "distinct_source_value_conflict"
            output.append(
                {
                    "event_index": index,
                    "field": field,
                    "cause": cause,
                    "alternatives": alternatives,
                }
            )
    return output


def _duplicate_event_keys(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[int]] = {}
    for index, event in enumerate(events, start=1):
        reference_date = event.get("reference_date")
        if not reference_date:
            continue
        key = (
            event.get("corp_code"),
            event.get("reporter"),
            reference_date,
            event.get("doc_id"),
        )
        grouped.setdefault(key, []).append(index)
    return [
        {"event_key": list(key), "event_indexes": indexes}
        for key, indexes in grouped.items()
        if len(indexes) > 1
    ]


def _groups_with_multiple_reference_dates(
    groups: Sequence[EvidenceGroup],
) -> list[dict[str, Any]]:
    output = []
    for group in groups:
        dates = sorted(
            {
                str(item.holding.get("reference_date"))
                for item in group.items
                if item.holding.get("reference_date")
            }
        )
        if len(dates) > 1:
            output.append({"group_id": group.group_id, "reference_dates": dates})
    return output


def _missing_value_zero_coercion(
    groups: Sequence[EvidenceGroup], resolution: HoldingResolution
) -> bool:
    events_by_chunk = {
        chunk_id: event
        for event in resolution.events
        for chunk_id in event.evidence_chunk_ids
    }
    for group in groups:
        for item in group.items:
            event = events_by_chunk.get(item.chunk_id)
            if event is None:
                continue
            fields = dict(item.holding.get("projection_fields") or {})
            for label, event_field in _NUMERIC_LABELS.items():
                raw = str(fields.get(label) or "").strip().casefold()
                value = getattr(event, event_field)
                if raw in _MISSING_MARKERS and value is not None and value.normalized == 0:
                    return True
    return False


def _requested_direction(query: str) -> str | None:
    compact = re.sub(r"\s+", "", query)
    if "감소" in compact:
        return "decrease"
    if "증가" in compact:
        return "increase"
    if any(value in compact for value in ("변동없음", "증감없음", "변화없음")):
        return "unchanged"
    return None


if __name__ == "__main__":
    main()
