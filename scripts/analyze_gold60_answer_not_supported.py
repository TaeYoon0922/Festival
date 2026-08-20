#!/usr/bin/env python3
"""Render a read-only analysis of Gold60 ``answer_not_supported`` failures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.agent.answer_failure_analysis import (  # noqa: E402
    analyze_answer_not_supported,
    write_answer_not_supported_report,
)


DEFAULT_INPUT = Path(
    "data/processed/postgres_agent_gold60/gold60_agent_evaluation.json"
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze answer_not_supported cases from a Gold60 agent evaluation "
            "or failure_analysis JSON without running retrieval."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Markdown output path (default: answer_not_supported_analysis.md "
            "beside the input file). A same-stem JSON file is also written."
        ),
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Fail if the number of answer_not_supported cases differs.",
    )
    return parser.parse_args(argv)


def load_report(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError("input JSON root must be an object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = load_report(args.input)
    analysis = analyze_answer_not_supported(report)
    actual_count = int(analysis["summary"]["case_count"])
    if args.expected_count is not None and actual_count != args.expected_count:
        raise RuntimeError(
            "answer_not_supported count mismatch: "
            f"expected={args.expected_count}, actual={actual_count}"
        )

    markdown_path = args.output or args.input.with_name(
        "answer_not_supported_analysis.md"
    )
    json_path = markdown_path.with_suffix(".json")
    written_markdown, written_json = write_answer_not_supported_report(
        analysis,
        markdown_path=markdown_path,
        json_path=json_path,
    )

    summary = analysis["summary"]
    print(f"input: {args.input}")
    print(f"answer_not_supported cases: {actual_count}")
    print(f"unsupported claims: {summary['unsupported_claim_count']}")
    print(f"markdown: {written_markdown}")
    print(f"json: {written_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
