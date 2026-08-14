"""Generate the release PostgreSQL COPY dataset at data/db_export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MANIFEST_PATH
from app.exporting.db_release_export import export_db_release


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "structural_v2_1_full_4204",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--export-dir", type=Path, default=PROJECT_ROOT / "data" / "db_export"
    )
    args = parser.parse_args()
    report = export_db_release(args.processed_dir, args.manifest, args.export_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
