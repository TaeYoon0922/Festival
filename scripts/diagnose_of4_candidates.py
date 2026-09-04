"""Why does a question with matching filings retrieve nothing?

Prints the plan, the metadata filters it produces, and how many documents and
chunks survive each step, so the step that empties the candidate set is visible
rather than inferred.  Read-only.

    python scripts/diagnose_of4_candidates.py
    python scripts/diagnose_of4_candidates.py --question "..."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope
from app.reasoning.router import QueryRouter
from app.retrieval.postgres_backend import PostgresBackend


DEFAULT_QUESTION = (
    "카카오가 2025년에 실시한 자금조달 내역을 유형별(유상증자, CB, BW, EB)로 정리해줘"
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Trace candidate narrowing.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    args = parser.parse_args(argv)

    backend = PostgresBackend()
    scope = CorpusScope.repository_default()
    understanding = QueryUnderstanding(
        scope.company_aliases() if scope else None,
        company_resolver=backend.resolve_company,
    )
    plan = understanding.understand(args.question, top_k=10)

    print("질문:", args.question)
    print()
    print("── plan ──")
    for name in (
        "companies", "corp_codes", "task_type", "years", "metric",
        "event_type", "disclosure_route", "doc_subtype", "correction_policy",
    ):
        print(f"  {name:18s} {getattr(plan, name, None)!r}")
    print(f"  {'period':18s} {plan.period.to_dict()!r}")
    print(f"  {'doc_group':18s} {getattr(plan, 'doc_group', None)!r}")
    print(f"  {'is_correction':18s} {getattr(plan, 'is_correction', None)!r}")
    print()
    filters = plan.backend_filters()
    print("── backend_filters ──")
    print(" ", json.dumps(filters, ensure_ascii=False))
    print()

    documents = backend.get_candidate_documents(**filters)
    print(f"── 필터 적용 문서: {len(documents)}건 ──")
    for document in documents[:15]:
        metadata = document.metadata or {}
        print(
            f"  {metadata.get('rcept_dt')}  {metadata.get('doc_group'):8s}  "
            f"{metadata.get('report_nm')}"
        )
    print()

    router = QueryRouter()
    route = router.route(plan)
    print("── route ──")
    print(" ", json.dumps(route.to_dict(), ensure_ascii=False)[:500])
    print()

    filtered = router.filter_documents(documents, route)
    print(f"── router.filter_documents 이후: {len(filtered)}건 ──")
    for document in filtered[:15]:
        metadata = document.metadata or {}
        print(f"  {metadata.get('rcept_dt')}  {metadata.get('report_nm')}")
    print()

    chunks = backend.get_candidate_chunks(filtered)
    print(f"── 후보 청크: {len(chunks)}개 ──")
    prepared = router.prepare_chunks(chunks, route)
    print(f"── router.prepare_chunks 이후: {len(prepared)}개 ──")
    print()

    print("── 필터를 하나씩 풀어보기 ──")
    for drop in ("doc_group", "doc_subtype", "year", "period", "is_correction",
                 "section_path"):
        if filters.get(drop) is None:
            continue
        relaxed = dict(filters)
        relaxed[drop] = None
        count = len(backend.get_candidate_documents(**relaxed))
        print(f"  {drop:14s} 제거 → 문서 {count}건")
    corp_only = {"corp_code": filters.get("corp_code")}
    print(f"  {'기업만':14s}      → 문서 "
          f"{len(backend.get_candidate_documents(**corp_only))}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
