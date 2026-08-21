"""Send one real HyperCLOVA X request and check the verbalizer contract.

This is a read-only connectivity check.  It touches no database, no corpus, and
no retrieval: it verbalizes a fixed, already-verified answer fixture so a
failure points at the HCX contract rather than at the agent.  Run it before
wiring the API to a live key.

    python scripts/smoke_hcx_verbalizer.py

The API key is read from ``FESTIVAL_HCX_API_KEY`` and is never printed, stored,
or included in the JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.answer_generator import (
    GeneratedAnswer,
    GeneratedCitation,
    GeneratedSection,
)
from app.generation.answer_validator import validate_verbalized_answer
from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer


#: A short verified answer.  Real numbers, dates, and a citation marker so the
#: validator has something meaningful to protect.
FIXTURE_TEXT = (
    "국민연금기금의 효성중공업 보유주식수는 2023년 03월 07일 기준 655,490주입니다.[1]"
)

FIXTURE_COMPANY = "효성중공업"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verbalize one fixed answer fixture through HyperCLOVA X and report "
            "whether the contract holds.  No database or corpus access."
        )
    )
    parser.add_argument(
        "--show-answer",
        action="store_true",
        help="Print the verbalized text as well as the contract checks.",
    )
    args = parser.parse_args(argv)

    settings = HcxSettings.from_env()
    if not settings.enabled:
        print("FESTIVAL_HCX_ENABLED is false; nothing to check.", file=sys.stderr)
        return 1
    if not settings.configured:
        print(
            "FESTIVAL_HCX_API_KEY is not set; the verbalizer would report "
            "not_configured and serve the deterministic answer.",
            file=sys.stderr,
        )
        return 1

    fixture = _fixture()
    verbalizer = HcxVerbalizer(settings)
    outcome = verbalizer.verbalize(fixture, required_terms=(FIXTURE_COMPANY,))

    validation = validate_verbalized_answer(
        outcome.text,
        reference=FIXTURE_TEXT,
        required_terms=(FIXTURE_COMPANY,),
    )
    report: dict[str, Any] = {
        "endpoint": settings.endpoint,
        "model": settings.model,
        "api_key_present": True,
        "hcx_status": outcome.status,
        "fallback_reason": outcome.reason,
        "used_hcx": outcome.used_hcx,
        "answer_non_empty": bool(outcome.text.strip()),
        "numbers_and_citations_preserved": validation.valid,
        "deterministic_fallback_served": outcome.text == FIXTURE_TEXT,
    }
    if args.show_answer:
        report["answer"] = outcome.text

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not report["answer_non_empty"]:
        print("FAILED: the verbalizer returned no text.", file=sys.stderr)
        return 1
    if outcome.status == "success":
        return 0

    print(
        f"HCX did not verbalize (status={outcome.status}, "
        f"reason={outcome.reason}); the deterministic answer was served.",
        file=sys.stderr,
    )
    return 2


def _fixture() -> GeneratedAnswer:
    return GeneratedAnswer(
        question="효성중공업 국민연금기금 변동일 변동후 주식수",
        answer_text=FIXTURE_TEXT,
        citations=(
            GeneratedCitation(
                citation_id="1",
                chunk_id="smoke:fixture",
                doc_id="smoke",
                source_refs=(),
                section="보유주식등의 수 및 보유비율",
                evidence_type="table",
            ),
        ),
        sections=(
            GeneratedSection(
                title="보유 현황",
                content=FIXTURE_TEXT,
                citations=("1",),
                metadata=("보고자: 국민연금기금",),
            ),
        ),
        warnings=(),
        confidence={"level": "high", "display_text": "높음"},
        answerable=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
