"""Ask which report, when the question names a holder but not a filing.

"에스엠 하이브 이번 보고 보유 주식수와 비율" reads as one question, but the
corpus holds seven 대량보유상황보고서 for that issuer and that holder, and
"이번 보고" selects a row inside a report -- 변동 전 against 변동 후 -- not a
report out of seven. Nothing in the question says which one.

Retrieval cannot close that gap and should not be asked to: the phrase appears
in 52,869 of this corpus's 91,387 holding chunks, so matching it selects more
than half the lane. Measured on the gold set, every question that named a date
resolved and every question that used the relative wording without one did not.

The system already answers this shape of problem by asking rather than by
choosing -- the alternative is picking a filing the asker never named and
presenting its figures as though they had. So this enumerates the filings the
question could mean and hands them to the existing clarification layer as
candidates. It is a candidate provider, not a new flow.

Refusing to guess is the point. Every gate here fails to ``None``, which leaves
the question on whatever path it was already taking.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from app.reasoning.clarification_request import (
    HOLDING_REPORT_INSTANCE,
    MAX_CANDIDATES,
    ClarificationCandidate,
    ClarificationRequest,
    ClarificationState,
)
from app.reasoning.holding_report_relative import parse as parse_report_relative


#: The selector the parser assigns when the wording points at a report the
#: context is expected to have fixed.  A one-shot question has no such context.
_CONTEXT_SELECTOR = "selected_context"

#: A date anywhere in the question means the asker did name a filing, and the
#: deterministic selector can use it.  Matched loosely on purpose: this decides
#: only whether to stay out of the way.
_NAMES_A_DATE = re.compile(
    r"(?:19|20)\d{2}\s*[.\-/년]|"
    r"\d{1,2}\s*월\s*\d{1,2}\s*일|"
    r"(?<!\d)\d{8}(?!\d)"
)

REASON = "holding_report_not_named"


def _dated(question: str) -> bool:
    return bool(_NAMES_A_DATE.search(str(question or "")))


def _label(record: Any) -> str | None:
    """What the asker would say back to choose this filing."""

    date = str(getattr(record, "reference_date", "") or "").strip()
    if not date:
        return None
    if len(date) == 8 and date.isdigit():
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    report = str(getattr(record, "report_nm", "") or "").strip()
    return f"{date} 기준 보고" if not report else f"{date} 기준 {report}"


def holding_report_clarification_request(
    question: str,
    plan: Any,
    *,
    report_index: Any,
    answerable: bool,
) -> ClarificationRequest | None:
    """Enumerate the filings a dateless relative-report question could mean.

    Returns ``None`` -- leaving the question exactly where it was -- when the
    wording is not relative, when a date was named, when the index cannot
    enumerate this pair, or when the question was answered anyway. Only a
    question that asked for a report it did not name, and did not get one,
    reaches the asker as a question back.
    """

    if answerable or report_index is None:
        return None
    text = str(question or "").strip()
    if not text or _dated(text):
        return None

    # Understanding already stored what it parsed; that is the reading the rest
    # of the pipeline gated on, so it is the one this must agree with. Parsing
    # is the fallback for a plan that carries none.
    stored = dict(getattr(plan, "evidence", {}) or {}).get("holding_report_relative")
    if isinstance(stored, Mapping):
        selector = str(stored.get("selector") or "")
    else:
        intent = parse_report_relative(text)
        selector = str(getattr(intent, "selector", "") or "") if intent else ""
    if selector != _CONTEXT_SELECTOR:
        return None

    corp_code = str(getattr(plan, "corp_code", "") or "").strip()
    reporter = str(getattr(plan, "reporter", "") or "").strip()
    if not corp_code or not reporter:
        return None

    try:
        records = report_index.enumerate_reports(corp_code, reporter)
    except Exception:  # noqa: BLE001 - a provider must never break the answer
        return None
    candidates = _candidates(records)
    if len(candidates) < 2:
        # One filing is not a choice, and none is not this layer's problem.
        return None

    return ClarificationRequest(
        question=text,
        candidates=candidates,
        reason=REASON,
        fallback_state=ClarificationState.INSUFFICIENT_EVIDENCE,
        truncated=len(records) > len(candidates),
        preserve_candidates=True,
    )


def _candidates(
    records: Sequence[Any],
) -> tuple[ClarificationCandidate, ...]:
    """One candidate per filing, newest first, each bound to its own chunk."""

    ordered = sorted(
        records,
        key=lambda record: str(getattr(record, "reference_date", "") or ""),
        reverse=True,
    )
    built: list[ClarificationCandidate] = []
    seen: set[str] = set()
    for record in ordered:
        label = _label(record)
        doc_id = str(getattr(record, "doc_id", "") or "").strip()
        chunk_id = str(getattr(record, "projection_chunk_id", "") or "").strip()
        if not label or not doc_id or not chunk_id or label in seen:
            continue
        seen.add(label)
        built.append(
            ClarificationCandidate(
                id=f"report::{doc_id}",
                label=label,
                semantic_type=HOLDING_REPORT_INSTANCE,
                provenance="holding_report_index",
                source_doc_id=doc_id,
                source_chunk_id=chunk_id,
            )
        )
        if len(built) == MAX_CANDIDATES:
            break
    return tuple(built)
