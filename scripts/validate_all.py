"""Validate all full-corpus parsing outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import CORPUS_DIR
from app.parsing.full_validation import validate_full_output


def main() -> None:
    output_dir = PROJECT_ROOT / "data" / "processed" / "full"
    report = validate_full_output(output_dir=output_dir, corpus_dir=CORPUS_DIR)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
