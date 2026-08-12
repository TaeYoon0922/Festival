"""Run the final validation gate before any full-corpus reprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CORPUS_DIR, MANIFEST_PATH
from app.parsing.final_validation import run_final_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate structural chunking freeze readiness.")
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chunking_pilot_20",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chunking_final_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_final_validation(
        corpus_dir=CORPUS_DIR,
        manifest_path=MANIFEST_PATH,
        pilot_dir=args.pilot_dir,
        output_dir=args.output,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "length_stats": report["length_stats"]["overall"],
                "text_length_stats": report["length_stats"]["by_chunk_type"]["text"],
                "extreme_tables": {
                    key: report["extreme_tables"][key]
                    for key in (
                        "gt_5000_count",
                        "gt_10000_count",
                        "max_char_count",
                        "projection_count",
                        "projection_traceability_errors",
                    )
                },
                "excluded_tables": report["excluded_tables"]["counts_by_reason"],
                "high_risk_excluded_evidence": report["excluded_tables"][
                    "high_risk_excluded_evidence_count"
                ],
                "correction_pilot": {
                    key: report["correction_pilot"][key]
                    for key in (
                        "document_count",
                        "group_counts",
                        "valid_document_count",
                        "error_count",
                    )
                },
                "bm25": report["bm25"]["overall"],
                "bm25_holding_additional": report["bm25_holding_additional"][
                    "overall"
                ],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
