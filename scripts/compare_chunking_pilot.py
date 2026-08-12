"""Compare legacy and structural chunking on the fixed 20-document pilot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CORPUS_DIR, MANIFEST_PATH
from app.parsing.comparison import run_chunking_pilot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 20-document legacy/structural chunk comparison."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "chunking_pilot_20",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_chunking_pilot(
        corpus_dir=CORPUS_DIR,
        manifest_path=MANIFEST_PATH,
        output_dir=args.output,
        per_group=5,
    )
    print(
        json.dumps(
            {
                "pilot_document_count": report["pilot_document_count"],
                "legacy": report["legacy"]["overall"],
                "structural": report["structural"]["overall"],
                "report": str(args.output / "comparison.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
