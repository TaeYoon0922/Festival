"""Parse a validated pilot sample of 20 DART disclosures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CORPUS_DIR, MANIFEST_PATH
from app.parsing.pipeline import run_sample_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse 5 documents from each of the four disclosure groups."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "sample_20",
    )
    parser.add_argument("--max-chars", type=int, default=1_200)
    parser.add_argument("--overlap", type=int, default=150)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_sample_pipeline(
        corpus_dir=CORPUS_DIR,
        manifest_path=MANIFEST_PATH,
        output_dir=args.output,
        per_group=5,
        max_chars=args.max_chars,
        overlap=args.overlap,
    )
    console_summary = {
        key: report[key]
        for key in (
            "document_count",
            "group_counts",
            "section_count",
            "table_count",
            "chunk_count",
            "warning_count",
        )
    }
    print(json.dumps(console_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
