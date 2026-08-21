"""Explain why HyperCLOVA X failed placeholder integrity on a real question.

Read-only.  This runs the production path — query understanding, PostgreSQL
hybrid retrieval, the agent orchestrator, and the citation-aware generator —
then hands the resulting answer to the existing protected-literal layer and
reports what happened to the placeholders.  Nothing is written, no production
behaviour changes, and the served API response is not involved.

    python scripts/diagnose_hcx_answer.py
    python scripts/diagnose_hcx_answer.py --question-id P01 --question "..."

Credentials, headers, connection strings, and retrieved chunk text are never
printed.  The model's reply is reported as counts and short excerpts only.
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

from app.api.pipeline import AnswerPipeline
from app.generation.answer_validator import validate_verbalized_answer
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


DEFAULT_QUESTION_ID = "P07"
DEFAULT_QUESTION = "두산퓨얼셀 2023년 1분기 연료전지 주기기 매출액"

#: How much text to show around a divergence.  Never the whole reply.
EXCERPT_RADIUS = 80

CAUSE_LABELS = {
    "A": "placeholder missing",
    "B": "placeholder duplicated",
    "C": "placeholder reordered",
    "D": "placeholder spelling changed",
    "E": "output truncation",
    "F": "max_tokens too small",
    "G": "deterministic answer unsuitable for verbalization",
    "H": "other",
}


class _RecordingTransport:
    """Forward the request and keep only the response body."""

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
            "Run one question through the production path and report what the "
            "protected-literal layer observed.  Read-only."
        )
    )
    parser.add_argument("--question-id", default=DEFAULT_QUESTION_ID)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--show-excerpt",
        action="store_true",
        help="Include short excerpts around the first divergence.",
    )
    args = parser.parse_args(argv)

    settings = HcxSettings.from_env()
    if not settings.configured:
        print(
            "FESTIVAL_HCX_API_KEY is not set; cannot diagnose a live reply.",
            file=sys.stderr,
        )
        return 1

    pipeline = AnswerPipeline.from_env()
    plan = pipeline.understanding.understand(
        args.question, top_k=pipeline.settings.top_k
    )
    execution = pipeline.executor.execute(plan)
    result = pipeline.orchestrator.run(args.question, plan, execution)
    generated = pipeline.generator.generate(result.answer_draft)

    protection = protect_literals(generated.answer_text)
    recorder = _RecordingTransport(UrllibJsonTransport())
    outcome = HcxVerbalizer(settings, transport=recorder).verbalize(generated)

    raw = (
        _response_content(recorder.response)
        if isinstance(recorder.response, dict)
        else None
    )
    expected = list(protection.placeholders)
    found = PLACEHOLDER_PATTERN.findall(raw) if raw is not None else []
    integrity = (
        check_placeholder_integrity(raw, protection) if raw is not None else None
    )
    restored = (
        restore_literals(raw, protection)
        if raw is not None and integrity is not None and integrity.valid
        else None
    )
    validator = (
        validate_verbalized_answer(restored, reference=generated.answer_text)
        if restored is not None
        else None
    )

    report: dict[str, Any] = {
        "question_id": args.question_id,
        "answerable": generated.answerable,
        "deterministic_answer_chars": len(generated.answer_text),
        "protected_literal_count": len(protection.literals),
        "protected_literal_count_by_type": _counts_by_type(protection),
        "masked_answer_chars": len(protection.masked),
        "max_tokens": settings.max_tokens,
        "model": settings.model,
        "hcx_status": outcome.status,
        "fallback_reason": outcome.reason,
        "hcx_candidate_received": raw is not None,
        "raw_candidate_chars": None if raw is None else len(raw),
        "expected_placeholder_count": len(expected),
        "found_placeholder_count": len(found),
        "placeholder_integrity_valid": None if integrity is None else integrity.valid,
        "placeholder_integrity_reason": None if integrity is None else integrity.reason,
        "missing_placeholders": _sorted(set(expected) - set(found)),
        "unexpected_placeholders": _sorted(set(found) - set(expected)),
        "duplicated_placeholders": _sorted(
            {token for token in found if found.count(token) > 1}
        ),
        "reorder_summary": _reorder_summary(expected, found),
        "restored_candidate_chars": None if restored is None else len(restored),
        "validator_valid": None if validator is None else validator.valid,
        "validator_reason": None if validator is None else validator.reason,
        "response_finish_reason": _finish_reason(recorder.response),
        "response_usage": _usage(recorder.response),
        "found_is_prefix_of_expected": found == expected[: len(found)],
    }
    report["likely_cause"] = _classify(report)
    if args.show_excerpt:
        report["excerpt"] = _excerpt(raw, expected, found)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if outcome.status == "success" else 2


def _counts_by_type(protection: Any) -> dict[str, int]:
    counts = {"date": 0, "number": 0, "citation": 0}
    for literal in protection.literals:
        counts[literal.kind] = counts.get(literal.kind, 0) + 1
    return counts


def _sorted(values: set[str]) -> list[str]:
    return sorted(values)


def _reorder_summary(expected: list[str], found: list[str]) -> dict[str, Any] | None:
    """Report the first position where the two runs disagree."""

    if found == expected:
        return None
    for index in range(min(len(expected), len(found))):
        if expected[index] != found[index]:
            return {
                "first_divergence_index": index,
                "expected": expected[index],
                "found": found[index],
            }
    return {
        "first_divergence_index": min(len(expected), len(found)),
        "expected": (
            expected[len(found)] if len(found) < len(expected) else None
        ),
        "found": found[len(expected)] if len(expected) < len(found) else None,
        "note": "runs agree until one ends",
    }


def _finish_reason(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason") or choices[0].get("finishReason")
        if isinstance(reason, str):
            return reason
    result = response.get("result")
    if isinstance(result, dict):
        reason = result.get("stopReason") or result.get("finish_reason")
        if isinstance(reason, str):
            return reason
    return None


def _usage(response: Any) -> dict[str, Any] | None:
    """Token counts only.  Needed to judge whether max_tokens was the limit."""

    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return None
    return {
        key: usage.get(key)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        if key in usage
    }


def _classify(report: dict[str, Any]) -> dict[str, Any]:
    """Map the observations onto the candidate causes A-H."""

    codes: list[str] = []
    finish = (report.get("response_finish_reason") or "").lower()
    truncated = finish in {"length", "max_tokens", "stop_before", "token_limit"}

    if report["hcx_candidate_received"] is False:
        codes.append("H")
    else:
        if report["duplicated_placeholders"]:
            codes.append("B")
        if report["unexpected_placeholders"]:
            codes.append("D")
        if report["missing_placeholders"]:
            codes.append("A")
            if truncated or report["found_is_prefix_of_expected"]:
                # A clean prefix of the expected run means the reply stopped
                # early rather than the model editing a token.
                codes.extend(["E", "F"])
        summary = report.get("reorder_summary")
        if (
            summary
            and not report["missing_placeholders"]
            and not report["unexpected_placeholders"]
            and not report["duplicated_placeholders"]
        ):
            codes.append("C")

    if not codes and report.get("placeholder_integrity_valid"):
        # Nothing went wrong with the placeholders; a validator failure, if any,
        # is reported separately and is not a cause here.
        return {
            "codes": [],
            "labels": ["no placeholder failure"],
            "truncation_signature": truncated,
        }
    if not codes:
        codes.append("H")

    ordered = sorted(dict.fromkeys(codes))
    return {
        "codes": ordered,
        "labels": [CAUSE_LABELS[code] for code in ordered],
        "truncation_signature": truncated or report["found_is_prefix_of_expected"],
    }


def _excerpt(
    raw: str | None, expected: list[str], found: list[str]
) -> dict[str, Any] | None:
    """Show a small window around the first divergence, never the whole reply."""

    if raw is None:
        return None
    if found == expected:
        return {"note": "no divergence"}

    anchor = None
    for index in range(min(len(expected), len(found))):
        if expected[index] != found[index]:
            anchor = found[index]
            break
    if anchor is None:
        anchor = found[-1] if found else None

    window: dict[str, Any] = {"tail": raw[-EXCERPT_RADIUS:]}
    if anchor is not None:
        position = raw.find(anchor)
        if position >= 0:
            start = max(0, position - EXCERPT_RADIUS)
            window["around_divergence"] = raw[start : position + EXCERPT_RADIUS]
    return window


if __name__ == "__main__":
    raise SystemExit(main())
