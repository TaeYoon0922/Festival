"""Run the read-only final release gate on the frozen full-corpus output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.release_validation import run_final_release_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen v2.1 full output.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "structural_v2_1_full_4204",
    )
    parser.add_argument(
        "--pilot-validation",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chunking_final_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_final_release_gate(args.output, args.pilot_validation)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "projection_audit": {
                    "length_stats": report["projection_audit"]["length_stats"],
                    "length_bins": report["projection_audit"]["length_bins"],
                    "structural_error_count": report["projection_audit"]["structural_error_count"],
                },
                "extreme_retrieval": report["extreme_retrieval_audit"]["counts"],
                "bm25_fixed_40": report["bm25_fixed_40"]["overall"],
                "bm25_holding_20": report["bm25_holding_20"]["overall"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
