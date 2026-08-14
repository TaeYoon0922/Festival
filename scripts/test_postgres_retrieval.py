"""Smoke-test PostgreSQL metadata filtering and lexical retrieval."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.metadata_filtered_retrieval import extract_metadata_filters
from app.retrieval.postgres_backend import PostgresBackend


def _preview(value: object, length: int = 240) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:length]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PostgreSQL metadata + FTS retrieval against the loaded corpus."
    )
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    backend = PostgresBackend()
    resolved = backend.resolve_company(args.query)
    aliases: dict[str, set[str]] = {}
    if resolved:
        canonical = str(resolved.get("corp_name") or "")
        for alias in (canonical, resolved.get("listed_name")):
            if alias:
                aliases.setdefault(str(alias), set()).add(canonical)
    filters = extract_metadata_filters(args.query, aliases)
    years = filters["years"] if filters["year_filter_applied"] else None
    documents = backend.filter_disclosures(
        year=years,
        doc_group=filters["doc_group"],
        doc_subtype=filters["doc_subtype"],
        corp_code=str(resolved["corp_code"]) if resolved else None,
    )
    chunks = backend.fetch_chunks(documents)
    results = backend.lexical_search(args.query, chunks, top_k=args.top_k)
    chunks_by_id = {chunk.chunk_id: chunk.chunk for chunk in chunks}

    print(f"candidate disclosures: {len(documents)}")
    print(f"candidate chunks: {len(chunks)}")
    print("top chunks:")
    for result in results:
        chunk = chunks_by_id[result.chunk_id]
        section_path = chunk.get("section_path") or []
        section = " > ".join(str(item) for item in section_path)
        print(f"\n[{result.rank}]")
        print(f"chunk_id: {result.chunk_id}")
        print(f"corp_name: {chunk.get('corp_name') or ''}")
        print(f"report_nm: {chunk.get('report_nm') or ''}")
        print(f"rcept_dt: {chunk.get('rcept_dt') or ''}")
        print(f"section: {section}")
        print(f"chunk_type: {chunk.get('chunk_type') or ''}")
        print(f"lexical score: {result.bm25_score:.6f}")
        print(f"retrieval_text: {_preview(chunk.get('retrieval_text'))}")


if __name__ == "__main__":
    main()
