"""Smoke-test PostgreSQL query planning and lexical retrieval."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.reasoning import QueryExecutor, QueryUnderstanding
from app.retrieval.postgres_backend import PostgresBackend


def _preview(value: object, length: int = 240) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:length]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QueryPlan + PostgreSQL metadata/FTS retrieval."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    backend = PostgresBackend()
    plan = QueryUnderstanding(company_resolver=backend.resolve_company).understand(
        args.query, top_k=args.top_k
    )
    execution = QueryExecutor(backend).execute(plan)
    documents = execution.documents
    chunks = execution.chunks
    results = execution.results
    chunks_by_id = {chunk.chunk_id: chunk.chunk for chunk in chunks}

    print("query plan:")
    print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2))
    print("routing:")
    print(json.dumps(dict(execution.routing), ensure_ascii=False, indent=2))
    print(f"candidate disclosures: {len(documents)}")
    print(f"candidate chunks: {len(chunks)}")
    print("top chunks:")
    for result in results:
        chunk = chunks_by_id[result.chunk_id]
        section_path = chunk.get("section_path") or []
        section = " > ".join(str(item) for item in section_path)
        components = result.metadata_match.get("score_components") or {}
        print(f"\n[{result.rank}]")
        print(f"chunk_id: {result.chunk_id}")
        print(f"corp_name: {chunk.get('corp_name') or ''}")
        print(f"report_nm: {chunk.get('report_nm') or ''}")
        print(f"rcept_dt: {chunk.get('rcept_dt') or ''}")
        print(f"section: {section}")
        print(f"chunk_type: {chunk.get('chunk_type') or ''}")
        print(f"lexical score: {result.bm25_score:.6f}")
        print(f"score components: {json.dumps(components, ensure_ascii=False)}")
        print(f"retrieval_text: {_preview(chunk.get('retrieval_text'))}")


if __name__ == "__main__":
    main()
