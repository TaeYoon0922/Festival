"""Refuse a question before it can reach the corpus or the model.

Everything else in this pipeline decides what to answer. This decides what not
to look at, and it runs first for that reason: a question carrying someone's
resident registration number must not be embedded, must not be used as a lexical
query, must not appear in a retrieval log, and must not be echoed back inside
``retrieved_context``. Refusing after retrieval would be too late for all four.

The hard part is not detection but restraint. Disclosure filings name executives,
largest shareholders and reporting parties by law, and a system that treats every
personal name as sensitive would refuse the questions it exists to answer. So
this blocks two narrow things -- an identifier that only a private record
carries, and a request for contact or address details that filings do not
publish -- and lets everything else through to the ordinary path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


#: The question carries an identifier that belongs to a private record.
CATEGORY_IDENTIFIER = "personal_identifier"
#: The question asks for details a disclosure filing does not publish.
CATEGORY_CONTACT_REQUEST = "personal_contact_request"

_BLOCKED_MESSAGE = (
    "개인정보가 포함되었거나 개인정보를 요청하는 질문에는 답변하지 않습니다. "
    "이 시스템은 공시 원문에 공개된 사실만을 근거로 답변합니다."
)

#: Identifiers a filing never states.  Each is anchored so an ordinary figure --
#: a share count, an amount, a date -- cannot match it.
_IDENTIFIER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # 주민등록번호: six digits, separator, then a century digit 1-4 and six more.
    ("resident_registration", re.compile(r"(?<!\d)\d{6}\s*[-–—]\s*[1-4]\d{6}(?!\d)")),
    # 휴대전화: 010-1234-5678 and the 011/016/017/018/019 forms.
    ("mobile_phone", re.compile(r"(?<!\d)01[016789]\s*[-–—]\s*\d{3,4}\s*[-–—]\s*\d{4}(?!\d)")),
    # 이메일.
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    # 여권번호: one letter and eight digits, as issued in Korea.
    ("passport", re.compile(r"(?<![A-Za-z0-9])[A-Z]\d{8}(?![A-Za-z0-9])")),
    # 카드번호: four groups of four.
    ("payment_card", re.compile(r"(?<!\d)\d{4}(?:\s*[-–—]\s*\d{4}){3}(?!\d)")),
)

#: Wording that asks for something filings do not publish.  A subject word alone
#: is not enough -- "대표이사" is an ordinary disclosure question -- so each
#: pattern pairs the private detail with a request for it.
_CONTACT_REQUEST = re.compile(
    r"(?:주민\s*등록\s*번호|주민번호|여권\s*번호|운전면허\s*번호"
    r"|개인\s*연락처|휴대\s*전화\s*번호|휴대폰\s*번호|전화\s*번호"
    r"|이메일\s*주소|메일\s*주소|자택\s*주소|집\s*주소|거주지"
    r"|계좌\s*번호|카드\s*번호)"
)
#: The request verb.  Present tense, imperative and interrogative forms alike.
_REQUEST_VERB = re.compile(
    r"(?:알려|가르쳐|조회|확인해|찾아|검색해|정리해|공개|보여|말해|뭐(?:야|니)|무엇)"
)


@dataclass(frozen=True)
class InputGuardDecision:
    """Whether this question may proceed, and why not when it may not."""

    blocked: bool
    category: str | None = None
    matched: str | None = None

    def to_dict(self) -> dict[str, Any]:
        # The matched value is never carried: reporting *what* was detected
        # would put the identifier back into the trace this exists to keep it
        # out of.  Only the kind is reported.
        return {
            "blocked": self.blocked,
            "category": self.category,
            "detector": self.matched,
        }


def guard_message() -> str:
    """What a blocked question is answered with."""

    return _BLOCKED_MESSAGE


def inspect_question(question: str) -> InputGuardDecision:
    """Decide whether this question may reach retrieval, without storing it.

    Runs before understanding, so nothing downstream ever sees a question this
    refuses -- not the embedder, not the lexical query, not the model.
    """

    text = str(question or "")
    if not text.strip():
        return InputGuardDecision(blocked=False)

    for name, pattern in _IDENTIFIER_PATTERNS:
        if pattern.search(text):
            return InputGuardDecision(
                blocked=True, category=CATEGORY_IDENTIFIER, matched=name
            )

    # A private detail named *and* asked for. Requiring both keeps "대표이사가
    # 누구인가" and "최대주주 현황" -- questions filings answer -- on the
    # ordinary path, while "임원 개인 연락처 알려줘" does not reach it.
    compact = re.sub(r"\s+", " ", text)
    if _CONTACT_REQUEST.search(compact) and _REQUEST_VERB.search(compact):
        return InputGuardDecision(
            blocked=True, category=CATEGORY_CONTACT_REQUEST, matched="contact_detail"
        )

    return InputGuardDecision(blocked=False)
