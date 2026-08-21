"""Send one real HyperCLOVA X request and check the verbalizer contract.

This is a read-only connectivity check.  It touches no database, no corpus, and
no retrieval: it verbalizes a fixed, already-verified answer fixture so a
failure points at the HCX contract rather than at the agent.  Run it before
wiring the API to a live key.

    python scripts/smoke_hcx_verbalizer.py
    python scripts/smoke_hcx_verbalizer.py --diagnose

The API key is read from ``FESTIVAL_HCX_API_KEY`` and is never printed, stored,
or included in the JSON report.  Request headers are never recorded.
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
from app.generation.answer_validator import (
    extract_citation_markers,
    extract_numeric_tokens,
    validate_verbalized_answer,
)

# ``_response_content`` is imported rather than reimplemented so the diagnostic
# reads the reply exactly the way production does.  A local copy could drift and
# then describe a candidate the verbalizer never saw.
from app.generation.hcx_verbalizer import (
    HcxSettings,
    HcxVerbalizer,
    _response_content,
)
from app.generation.protected_literals import (
    PLACEHOLDER_PATTERN,
    check_placeholder_integrity,
    protect_literals,
    restore_literals,
)
from app.retrieval.embeddings import UrllibJsonTransport


#: A short verified answer.  Real numbers, dates, and a citation marker so the
#: validator has something meaningful to protect.
FIXTURE_TEXT = (
    "국민연금기금의 효성중공업 보유주식수는 2023년 03월 07일 기준 655,490주입니다.[1]"
)

FIXTURE_COMPANY = "효성중공업"


class _RecordingTransport:
    """Pass a request through and keep only the response body.

    Headers carry the bearer token, and the request payload is already known, so
    neither is retained.  Only the reply is kept, and only so the raw candidate
    can be compared against the deterministic reference.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.response: Any = None

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.response = self._inner.post_json(
            url, headers=headers, payload=payload, timeout_seconds=timeout_seconds
        )
        return self.response


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
        help="Print the answer that would be served.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "Print the raw and restored HCX candidates next to the reference, "
            "with the placeholders, citation markers, and numeric tokens "
            "compared at each stage."
        ),
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

    recorder = _RecordingTransport(UrllibJsonTransport())
    verbalizer = HcxVerbalizer(settings, transport=recorder)
    outcome = verbalizer.verbalize(_fixture(), required_terms=(FIXTURE_COMPANY,))

    # ``protect_literals`` is deterministic, so the same protection the
    # verbalizer built can be rebuilt here without production exposing it.
    protection = protect_literals(FIXTURE_TEXT)
    raw = (
        _response_content(recorder.response)
        if isinstance(recorder.response, dict)
        else None
    )
    integrity = (
        check_placeholder_integrity(raw, protection) if raw is not None else None
    )
    restored = (
        restore_literals(raw, protection)
        if raw is not None and integrity is not None and integrity.valid
        else None
    )

    # Three different questions.  ``served`` is what a caller receives, which on
    # a fallback is the reference itself and therefore always valid.
    # ``integrity`` says whether the protected literals survived the model.
    # ``restored`` is the only one that judges what HCX actually wrote.
    served = validate_verbalized_answer(
        outcome.text, reference=FIXTURE_TEXT, required_terms=(FIXTURE_COMPANY,)
    )
    candidate_result = (
        validate_verbalized_answer(
            restored, reference=FIXTURE_TEXT, required_terms=(FIXTURE_COMPANY,)
        )
        if restored is not None
        else None
    )

    report: dict[str, Any] = {
        "endpoint": settings.endpoint,
        "model": settings.model,
        "api_key_present": True,
        "hcx_status": outcome.status,
        "fallback_reason": outcome.reason,
        "used_hcx": outcome.used_hcx,
        "answer_non_empty": bool(outcome.text.strip()),
        "deterministic_fallback_served": outcome.text == FIXTURE_TEXT,
        "served_answer_valid": served.valid,
        "hcx_candidate_received": raw is not None,
        "placeholder_integrity_valid": None if integrity is None else integrity.valid,
        "placeholder_integrity_reason": None if integrity is None else integrity.reason,
        "hcx_candidate_valid": (
            None if candidate_result is None else candidate_result.valid
        ),
        "hcx_candidate_reason": (
            None if candidate_result is None else candidate_result.reason
        ),
    }
    if args.show_answer:
        report["served_answer"] = outcome.text
    if args.diagnose:
        report["diagnostic"] = _diagnostic(
            raw, restored, protection, integrity, candidate_result
        )

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
    if not args.diagnose:
        print("Re-run with --diagnose to see what HCX changed.", file=sys.stderr)
    return 2


def _diagnostic(
    raw: str | None,
    restored: str | None,
    protection: Any,
    integrity: Any,
    result: Any,
) -> dict[str, Any]:
    """Show exactly what each stage compared.  No headers, no credentials."""

    reference_citations = extract_citation_markers(FIXTURE_TEXT)
    reference_numbers = extract_numeric_tokens(FIXTURE_TEXT)
    payload: dict[str, Any] = {
        "reference_text": FIXTURE_TEXT,
        "masked_reference": protection.masked,
        "raw_hcx_candidate": raw,
        "restored_hcx_candidate": restored,
        "expected_placeholders": list(protection.placeholders),
        "found_placeholders": None if raw is None else PLACEHOLDER_PATTERN.findall(raw),
        "placeholder_integrity_reason": None if integrity is None else integrity.reason,
        "placeholder_integrity_detail": None if integrity is None else integrity.detail,
        "reference_citations": sorted(reference_citations),
        "candidate_citations": None,
        "reference_numeric_tokens": sorted(reference_numbers),
        "candidate_numeric_tokens": None,
        "validator_reason": None if result is None else result.reason,
    }
    if restored is None:
        return payload

    # Tokens are read off the restored text, because that is what the validator
    # judged.  Reading them off the masked reply would report empty sets and
    # hide the real comparison.
    candidate_citations = extract_citation_markers(restored)
    candidate_numbers = extract_numeric_tokens(restored)
    payload["candidate_citations"] = sorted(candidate_citations)
    payload["candidate_numeric_tokens"] = sorted(candidate_numbers)
    payload["citations_only_in_reference"] = sorted(
        reference_citations - candidate_citations
    )
    payload["citations_only_in_candidate"] = sorted(
        candidate_citations - reference_citations
    )
    payload["numbers_only_in_reference"] = sorted(reference_numbers - candidate_numbers)
    payload["numbers_only_in_candidate"] = sorted(candidate_numbers - reference_numbers)
    return payload


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
