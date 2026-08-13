"""Evaluate frozen v2.1 outputs with query-derived metadata filtering."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MANIFEST_PATH
from app.parsing.metadata_filtered_retrieval import run_metadata_filtered_bm25


if __name__ == "__main__":
    output = PROJECT_ROOT / "data" / "processed" / "structural_v2_1_full_4204"
    report = run_metadata_filtered_bm25(
        output, MANIFEST_PATH, output / "final_release_gate"
    )
    print(
        json.dumps(
            {
                "gold_40": report["gold_40"],
                "holding_20": report["holding_20"],
                "failure_counts": report["failure_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
