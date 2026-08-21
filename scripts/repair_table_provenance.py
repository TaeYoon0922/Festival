"""Backfill ordinary table source references without parsing or rechunking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.table_provenance_repair import repair_table_provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair source_refs on existing ordinary table chunks only."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "structural_v2_1_full_4204",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = repair_table_provenance(args.output_dir, dry_run=args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

