"""Ask which contract, when the served filings disagree about the answer.

"한미반도체 대만 장비 수주 계약금액" is one question and several filings. The
issuer disclosed more than one 단일판매·공급계약 involving that customer, each
with its own amount, and the question names none of them. The answerability
guard sees that as a conflict -- ``requested_field_conflict`` -- and declines,
which is right: picking one of several disclosed amounts and presenting it as
the answer would be a wrong number stated confidently.

Declining is where it stopped, and the served answer said the evidence was
insufficient. It was not. The amounts are there, more than one of them, and
what is missing is in the question. So the filings are enumerated and handed
back, the same way an under-specified holding question is.

The candidates reuse ``EVENT_INSTANCE``: its wording already asks which 공시일
was meant, which is exactly the choice on offer. A filing only becomes a
candidate when a chunk retrieval actually served states the field, so the list
is the corpus's, not this module's.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from app.reasoning.clarification_request import (
    EVENT_INSTANCE,
    MAX_CANDIDATES,
    ClarificationCandidate,
    ClarificationRequest,
    ClarificationState,
)


REASON = "contract_instance_not_named"

#: The wording a filing uses for each field the guard can report in conflict.
#: A filing that states none of its field's words is not offered as a choice
#: for that field, however highly it ranked.
_FIELD_WORDING: Mapping[str, tuple[str, ...]] = {
    "contract_amount": ("계약금액", "해지금액", "공급계약"),
    "contract_counterparty": ("계약상대", "상대방"),
    "contract_period": ("계약기간",),
    "termination_reason": ("해지사유", "해지주요사유", "종료사유"),
}

#: A date in the question means the asker did name a filing, and the ordinary
#: path should be left to use it.
_NAMES_A_DATE = re.compile(
    r"(?:19|20)\d{2}\s*[.\-/년]|"
    r"\d{1,2}\s*월\s*\d{1,2}\s*일|"
    r"(?<!\d)\d{8}(?!\d)"
)


def conflicting_field(answerability: Any) -> str | None:
    """The one field the guard proved several served filings disagree on.

    Exactly one, because a question whose several fields all conflict is not a
    question about one filing that needs naming.
    """

    records = getattr(answerability, "unavailable_evidence", ()) or ()
    fields = {
        str(record.get("field") or "").strip()
        for record in records
        if isinstance(record, Mapping)
        and str(record.get("status") or "") == "conflict"
        and str(record.get("field") or "").strip() in _FIELD_WORDING
    }
    return next(iter(fields)) if len(fields) == 1 else None


def contract_instance_clarification_request(
    question: str,
    execution: Any,
    answerability: Any,
) -> ClarificationRequest | None:
    """Enumerate the filings a conflicted question could have meant."""

    text = str(question or "").strip()
    if not text or _NAMES_A_DATE.search(text):
        return None
    field = conflicting_field(answerability)
    if field is None:
        return None

    candidates = _candidates(execution, _FIELD_WORDING[field])
    if len(candidates) < 2:
        # One filing cannot conflict with itself, and none is not a choice.
        return None
    return ClarificationRequest(
        question=text,
        candidates=candidates,
        reason=REASON,
        fallback_state=ClarificationState.INSUFFICIENT_EVIDENCE,
        preserve_candidates=True,
    )


def _label(metadata: Mapping[str, Any]) -> str | None:
    """What the asker would say back to choose this filing."""

    date = str(metadata.get("rcept_dt") or "").strip()
    report = str(metadata.get("report_nm") or "").strip()
    if not date:
        return report or None
    if len(date) == 8 and date.isdigit():
        date = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    return f"{date} {report}".strip()


def _candidates(
    execution: Any, wording: Sequence[str]
) -> tuple[ClarificationCandidate, ...]:
    """One candidate per served filing that states the conflicting field."""

    metadata_by_doc = {
        str(document.doc_id): dict(document.metadata)
        for document in (getattr(execution, "documents", ()) or ())
    }
    built: list[ClarificationCandidate] = []
    seen_docs: set[str] = set()
    seen_labels: set[str] = set()
    for candidate in getattr(execution, "chunks", ()) or ():
        doc_id = str(getattr(candidate, "doc_id", "") or "")
        chunk_id = str(getattr(candidate, "chunk_id", "") or "")
        if not doc_id or not chunk_id or doc_id in seen_docs:
            continue
        chunk = getattr(candidate, "chunk", None)
        content = str((chunk or {}).get("content") or "") if isinstance(chunk, Mapping) else ""
        if not any(word in content for word in wording):
            continue
        label = _label(metadata_by_doc.get(doc_id) or {})
        if not label or label in seen_labels:
            continue
        seen_docs.add(doc_id)
        seen_labels.add(label)
        built.append(
            ClarificationCandidate(
                id=f"contract::{doc_id}",
                label=label,
                semantic_type=EVENT_INSTANCE,
                provenance="served_contract_evidence",
                source_doc_id=doc_id,
                source_chunk_id=chunk_id,
            )
        )
        if len(built) == MAX_CANDIDATES:
            break
    return tuple(built)
