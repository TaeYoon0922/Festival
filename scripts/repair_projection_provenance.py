"""Repair projection provenance without rerunning text/table chunking."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.projection_repair import (
    audit_holding_placeholders,
    repair_projection_provenance,
    validate_projection_provenance,
    validate_repaired_output_integrity,
)


if __name__ == "__main__":
    output = PROJECT_ROOT / "data" / "processed" / "structural_v2_1_full_4204"
    report = repair_projection_provenance(output)
    report["validation"] = validate_projection_provenance(output)
    report["integrity"] = validate_repaired_output_integrity(output)
    report["placeholder_audit"] = audit_holding_placeholders(output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
