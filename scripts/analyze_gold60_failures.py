"""Analyze misses from an existing PostgreSQL hybrid Gold60 report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.reasoning.failure_analysis import (
    analyze_gold60_failures,
    write_failure_analysis,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Recall@10 failures in an existing Gold60 hybrid evaluation "
            "without running retrieval or connecting to PostgreSQL."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "postgres_hybrid_gold60"
            / "postgres_hybrid_gold60.json"
        ),
        help="Existing evaluate_postgres_hybrid_gold60.py JSON output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to the input report directory.",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"evaluation report does not exist: {args.input}")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    analysis = analyze_gold60_failures(report)
    output_dir = args.output_dir or args.input.parent
    write_failure_analysis(analysis, output_dir)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "output_dir": str(output_dir),
                **analysis["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
