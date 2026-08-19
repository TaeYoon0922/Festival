"""Print a dimension-specific HNSW migration without executing it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.retrieval.vector_index import generate_hnsw_index_sql


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (but do not execute) dimension-specific HNSW SQL."
    )
    parser.add_argument("--dimensions", type=int, required=True)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=128)
    parser.add_argument("--no-concurrently", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sql = generate_hnsw_index_sql(
        args.dimensions,
        concurrently=not args.no_concurrently,
        m=args.m,
        ef_construction=args.ef_construction,
    )
    if args.output:
        args.output.write_text(sql, encoding="utf-8")
        print(args.output)
    else:
        print(sql, end="")


if __name__ == "__main__":
    main()
