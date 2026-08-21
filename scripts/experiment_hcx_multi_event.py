"""Measure how many verified events HyperCLOVA X will restate without editing.

The compact-claim adapter currently caps a claim at three events, which leaves
most holding questions on the deterministic path.  Raising that cap is only
defensible with evidence, so this script asks the model to restate claims of
increasing size and reports where it stops being faithful.

Everything here is an experiment.  It builds its own request with its own
system prompt so the production prompt stays untouched, it never writes
anything, and its results do not change how the API answers.  The production
caps are read for reference only.

    python scripts/experiment_hcx_multi_event.py
    python scripts/experiment_hcx_multi_event.py --events 4 6 10 --repeat 2

Credentials, headers, and connection strings are never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.answer_validator import validate_verbalized_answer
from app.generation.compact_claim import (
    MAX_CLAIM_EVENTS,
    MAX_CLAIM_LITERALS,
    ClaimCitation,
    ClaimField,
    CompactClaim,
    _render,
)
from app.generation.hcx_verbalizer import HcxSettings, _response_content
from app.generation.protected_literals import (
    PLACEHOLDER_PATTERN,
    check_placeholder_integrity,
    protect_literals,
    restore_literals,
)
from app.retrieval.embeddings import EmbeddingHttpError, UrllibJsonTransport


DEFAULT_EVENT_COUNTS = (4, 6, 10)

#: The literal kinds ``protect_literals`` assigns.
_KINDS = ("date", "number", "citation")

COMPANY = "효성중공업"
REPORTER = "국민연금기금"

#: Experimental only.  The production prompt in ``hcx_verbalizer`` is unchanged.
#: This wording adds an explicit instruction to keep every item, which is the
#: behaviour being measured.
EXPERIMENT_SYSTEM_PROMPT = """당신은 이미 검증된 공시 사실을 자연스러운 한국어로 다듬는 편집자입니다.

반드시 지킬 것:
- 입력에 있는 항목을 하나도 빠뜨리지 말고 전부 표현한다.
- 항목을 요약하거나, 대표값만 고르거나, 여러 항목을 하나로 합치지 않는다.
- 입력 순서를 그대로 유지한다.
- __FESTIVAL_...__ 형태의 토큰은 보호된 값이다. 글자 하나도 바꾸지 말고
  개수와 순서를 입력 그대로 유지한다. 삭제, 추가, 중복, 분할하지 않는다.
- 보호 토큰이 무엇을 뜻하는지 추측하거나 숫자, 날짜로 바꿔 쓰지 않는다.
- 새로운 사실, 추정, 해석, 결론, 전망, 투자 의견을 쓰지 않는다.
- "이를 통해 ~ 알 수 있습니다" 같은 문장을 덧붙이지 않는다.
- Markdown 서식을 새로 만들지 않는다.

출력은 다듬어진 본문만 쓴다. 설명이나 머리말을 붙이지 않는다."""

#: Phrases that mark the model drawing a conclusion instead of restating facts.
#: Diagnostic only; the production validator is not changed.
INFERENCE_MARKERS = (
    "이를 통해",
    "알 수 있습니다",
    "따라서",
    "결론적으로",
    "요약하면",
    "종합하면",
    "판단됩니다",
    "보입니다",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ask HyperCLOVA X to restate claims of increasing size and report "
            "where fidelity breaks.  Experimental; production is unchanged."
        )
    )
    parser.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=list(DEFAULT_EVENT_COUNTS),
        help="Event counts to try (default: 4 6 10).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Runs per size, to see whether a result is stable.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help=(
            "Include the assistant message for each run: the raw candidate, "
            "and the restored text when integrity allows restoration."
        ),
    )
    args = parser.parse_args(argv)

    settings = HcxSettings.from_env()
    if not settings.configured:
        print(
            "FESTIVAL_HCX_API_KEY is not set; this experiment needs a live key.",
            file=sys.stderr,
        )
        return 1

    transport = UrllibJsonTransport()
    runs = [
        run_once(
            transport, settings, count, show_output=args.show_output
        )
        for count in args.events
        for _ in range(max(1, args.repeat))
    ]

    print(
        json.dumps(
            {
                "model": settings.model,
                "max_tokens": settings.max_tokens,
                "production_caps": {
                    "MAX_CLAIM_EVENTS": MAX_CLAIM_EVENTS,
                    "MAX_CLAIM_LITERALS": MAX_CLAIM_LITERALS,
                },
                "runs": runs,
                "summary": summarize(runs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def run_once(
    transport: Any,
    settings: HcxSettings,
    event_count: int,
    *,
    show_output: bool = False,
) -> dict[str, Any]:
    """Send one claim of ``event_count`` events and report what came back."""

    claim = build_experiment_claim(event_count)
    protection = protect_literals(claim.deterministic_text)
    payload = {
        "model": settings.model,
        "messages": [
            {"role": "system", "content": EXPERIMENT_SYSTEM_PROMPT},
            {"role": "user", "content": protection.masked},
        ],
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }

    record: dict[str, Any] = {
        "event_count": event_count,
        "claim_chars": len(claim.deterministic_text),
        "masked_chars": len(protection.masked),
        "field_literal_count": len(claim.fields),
        "protected_literal_count": len(protection.literals),
    }

    try:
        response = transport.post_json(
            settings.endpoint,
            headers=settings.request_headers(),
            payload=payload,
            timeout_seconds=settings.timeout_seconds,
        )
    except EmbeddingHttpError as error:
        record["transport_error"] = (
            "timeout" if error.status_code is None else f"HTTP {error.status_code}"
        )
        return record
    except Exception as error:  # noqa: BLE001 - an experiment must not crash mid-sweep
        record["transport_error"] = type(error).__name__
        return record

    raw = _response_content(response) if isinstance(response, dict) else None
    record["finish_reason"] = _finish_reason(response)
    record["completion_tokens"] = _completion_tokens(response)
    record["candidate_received"] = raw is not None
    if show_output:
        # Emitted here, not at the end, so a run that fails integrity still
        # shows what the model actually wrote.  Only the assistant message is
        # exposed; the response envelope and request headers never are.
        record["raw_hcx_candidate"] = raw
    if raw is None:
        return record

    found = PLACEHOLDER_PATTERN.findall(raw)
    integrity = check_placeholder_integrity(raw, protection)
    breakdown = placeholder_type_breakdown(protection, found)
    record.update(
        {
            "output_chars": len(raw),
            "expected_placeholders": len(protection.placeholders),
            "found_placeholders": len(found),
            "placeholder_integrity_valid": integrity.valid,
            "placeholder_integrity_reason": integrity.reason,
            "all_events_kept": breakdown["field_placeholders_all_preserved"],
            **breakdown,
        }
    )
    if not integrity.valid:
        return record

    restored = restore_literals(raw, protection)
    result = validate_verbalized_answer(
        restored,
        reference=claim.deterministic_text,
        required_terms=claim.required_terms,
    )
    record.update(
        {
            "restored_chars": len(restored),
            "length_ratio": round(len(restored) / len(claim.deterministic_text), 2),
            "candidate_valid": result.valid,
            "validator_reason": result.reason,
            "inference_markers": [
                marker for marker in INFERENCE_MARKERS if marker in restored
            ],
        }
    )
    if show_output:
        record["restored_output"] = restored
    return record


def placeholder_type_breakdown(
    protection: Any, found: Sequence[str]
) -> dict[str, Any]:
    """Split placeholder survival by kind.

    Whether a model drops whole events or drops one kind of token across every
    event are different failures with different fixes, and the totals alone
    cannot tell them apart.
    """

    kinds = {literal.placeholder: literal.kind for literal in protection.literals}
    found_set = set(found)
    expected_counts = Counter(literal.kind for literal in protection.literals)
    found_counts = Counter(
        kinds[token] for token in found_set if token in kinds
    )

    expected_by_type = {kind: expected_counts.get(kind, 0) for kind in _KINDS}
    found_by_type = {kind: found_counts.get(kind, 0) for kind in _KINDS}
    missing_by_type = {
        kind: expected_by_type[kind] - found_by_type[kind] for kind in _KINDS
    }

    def preserved(*wanted: str) -> bool:
        return all(missing_by_type[kind] == 0 for kind in wanted)

    return {
        "expected_placeholders_by_type": expected_by_type,
        "found_placeholders_by_type": found_by_type,
        "missing_placeholders_by_type": missing_by_type,
        "unrecognized_placeholders": len(found_set - set(kinds)),
        "field_placeholders_all_preserved": preserved("date", "number"),
        "citation_placeholders_all_preserved": preserved("citation"),
    }


def build_experiment_claim(event_count: int) -> CompactClaim:
    """A claim shaped like the adapter's, with ``event_count`` distinct events."""

    dates = [
        f"2023-{month:02d}-{day:02d}"
        for month, day in _date_pairs(event_count)
    ]
    shares = [f"{(index + 1) * 137_411:,}주" for index in range(event_count)]

    fields: list[ClaimField] = []
    citations: list[ClaimCitation] = []
    for index in range(event_count):
        marker = f"[{index + 1}]"
        chunk_id = f"experiment:{index + 1}"
        citations.append(
            ClaimCitation(
                marker=marker,
                chunk_id=chunk_id,
                doc_id=f"experiment{index + 1}",
                source_refs=({"table_id": f"t{index + 1}", "row_start": 1, "row_end": 1},),
            )
        )
        fields.append(
            ClaimField(
                name="reference_date", label="변동일", value=dates[index],
                marker=marker, chunk_id=chunk_id,
            )
        )
        fields.append(
            ClaimField(
                name="after_shares", label="변동 후 주식수", value=shares[index],
                marker=marker, chunk_id=chunk_id,
            )
        )

    return CompactClaim(
        question=f"{COMPANY} {REPORTER} 변동일 변동후 주식수",
        company=COMPANY,
        reporter=REPORTER,
        fields=tuple(fields),
        citations=tuple(citations),
        deterministic_text=_render(COMPANY, REPORTER, fields),
    )


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Group runs by size so a cap can be read off the results."""

    by_size: dict[int, dict[str, Any]] = {}
    for run in runs:
        bucket = by_size.setdefault(
            run["event_count"],
            {"runs": 0, "integrity_ok": 0, "validator_ok": 0, "inference_seen": 0},
        )
        bucket["runs"] += 1
        if run.get("placeholder_integrity_valid"):
            bucket["integrity_ok"] += 1
        if run.get("candidate_valid"):
            bucket["validator_ok"] += 1
        if run.get("inference_markers"):
            bucket["inference_seen"] += 1

    largest_clean = max(
        (
            size
            for size, bucket in by_size.items()
            if bucket["runs"] == bucket["validator_ok"]
        ),
        default=None,
    )
    return {
        "by_event_count": {str(size): bucket for size, bucket in sorted(by_size.items())},
        "largest_fully_clean_event_count": largest_clean,
        "note": (
            "A cap should be set from these observations, not from the coverage "
            "it would unlock."
        ),
    }


def _date_pairs(count: int) -> list[tuple[int, int]]:
    return [((index % 12) + 1, ((index * 7) % 28) + 1) for index in range(count)]


def _finish_reason(response: Any) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        if isinstance(reason, str):
            return reason
    result = response.get("result")
    if isinstance(result, dict):
        reason = result.get("stopReason")
        if isinstance(reason, str):
            return reason
    return None


def _completion_tokens(response: Any) -> int | None:
    if not isinstance(response, dict):
        return None
    usage = response.get("usage")
    if isinstance(usage, dict):
        value = usage.get("completion_tokens")
        if isinstance(value, int):
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
