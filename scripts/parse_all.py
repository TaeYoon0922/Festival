"""Parse the complete DART corpus into compressed structured outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CORPUS_DIR, MANIFEST_PATH
from app.parsing.full_pipeline import run_full_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse all corpus XML/HTML sources.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-chars", type=int, default=1_200)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_full_pipeline(
        corpus_dir=CORPUS_DIR,
        manifest_path=MANIFEST_PATH,
        output_dir=args.output,
        workers=args.workers,
        max_chars=args.max_chars,
        overlap=args.overlap,
        resume=not args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
