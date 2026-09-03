"""Deterministic phrase semantics for two metric names the corpus overloads.

Some metric words in this corpus are not metrics on their own.  ``매출액`` names
the periodic revenue line, but ``매출액 대비`` inside a contract question names
the filing's own ratio field -- a different number, in a different disclosure
family, answered by a different lane.  ``상장`` names a listing event, but
``상장과 비상장`` inside a 계열회사 question names two columns of one affiliate
count table.

Independent Eval v2 failed C059 and C086 on exactly that: the generic token was
found first and the phrase containing it never got to speak.  So the rule here
is an ordering, not a new vocabulary --

    a strong multi-token domain phrase outranks the generic single-token
    metric inside it

-- and each phrase carries the domain context that licenses it, because a phrase
read out of context is how the opposite mistake gets made.  ``매출액 대비`` in a
periodic question is still the periodic ratio; only a named contract makes it a
contract field.

Nothing here reads a document, retrieves anything, or asks a model.  Both
functions take a question and return either one intent or nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


#: The contract filing's own ``매출액대비(%)`` cell.  Named for the field, not
#: for the metric word inside it, so nothing downstream confuses the two.
CONTRACT_SALES_RATIO = "sales_ratio"

#: How many companies the 기업집단 has, listed and unlisted together.
AFFILIATE_COUNT = "affiliate_count"

_WHITESPACE = re.compile(r"\s+")
#: The interpunct family a disclosure title uses -- ``단일판매ㆍ공급계약`` is one
#: word to a reader and three characters to a matcher.
_INTERPUNCT = re.compile(r"[ㆍ·・‧]")


def _compact(value: Any) -> str:
    return _INTERPUNCT.sub("", _WHITESPACE.sub("", str(value or ""))).casefold()


@dataclass(frozen=True)
class PhraseIntent:
    """One metric a multi-token phrase named, and what licensed reading it so.

    ``phrase`` is the domain phrase that decided the metric; ``context`` is the
    word that made this question its domain.  Both are kept because a routing
    decision that cannot say which two words produced it is not debuggable.
    """

    metric: str
    phrase: str
    context: str


# ------------------------------------------------------------ 매출액 대비


#: The contract ratio, written every way a question writes it.  Whitespace is
#: compacted first, so this one entry covers ``매출액 대비``, ``매출액대비``,
#: ``매출액 대비 비율`` and the label's own ``매출액대비(%)``.
_SALES_RATIO_PHRASE = "매출액대비"

#: What makes a ``매출액 대비`` question a contract question.  Ordered most
#: specific first, so the recorded context names the filing family when the
#: question does.  ``계약`` and ``수주`` are last: they are the bare nouns, and
#: they are enough -- a question that names a contract and asks for its ratio
#: against revenue is asking about that contract.
_CONTRACT_CONTEXT = (
    "단일판매",
    "공급계약",
    "수주계약",
    "체결계약",
    "계약",
    "수주",
)


def contract_sales_ratio_intent(question: Any) -> PhraseIntent | None:
    """The contract's own revenue ratio, when the question names a contract.

    Returns ``None`` for a periodic ratio question.  ``매출액 대비 영업이익
    비율`` is a real periodic metric and must keep the route it has; only a
    named contract moves the phrase into the exchange lane.
    """

    compact = _compact(question)
    if _SALES_RATIO_PHRASE not in compact:
        return None
    context = next(
        (token for token in _CONTRACT_CONTEXT if token in compact), None
    )
    if context is None:
        return None
    return PhraseIntent(
        metric=CONTRACT_SALES_RATIO,
        phrase=_SALES_RATIO_PHRASE,
        context=context,
    )


# --------------------------------------------------------- 계열회사 수


#: The nouns that make a question about a corporate group's members.  Ordered
#: most specific first so the recorded phrase is the one the asker used.
_AFFILIATE_NOUNS = ("계열회사", "계열사", "소속회사", "기업집단")

#: What turns naming a group into asking for its size.  A question that merely
#: mentions 계열회사 -- a transaction with one, membership in a group -- is not
#: this metric and keeps whatever routing it had.
_COUNT_REQUEST = (
    "몇개",
    "몇곳",
    "개수",
    "수는",
    "수가",
    "총수",
    "합쳐",
    "합하면",
    "현황",
)

#: The listed/unlisted split, in every form a question writes it.  Compaction
#: already collapses ``상장과 비상장``, ``상장 및 비상장``, ``상장/비상장`` and
#: ``상장ㆍ비상장`` onto the one token that distinguishes them all.
_LISTING_SPLIT = "비상장"

#: The listing-history field, asked for by its own name.  This is a strong
#: multi-token phrase in its own right, so it outranks the affiliate reading by
#: exactly the rule this module is built on: ``계열회사 중 상장일이 가장 이른
#: 곳`` asks for a listing date and must keep the listing-history intent.
_LISTING_DATE = re.compile(r"상장일자?")


def affiliate_count_intent(question: Any) -> PhraseIntent | None:
    """How many affiliates the group has, when that is what was asked.

    The combination decides it: a group or affiliate noun, plus either the
    listed/unlisted split or an outright request for a number.  ``상장`` on its
    own never reaches this function's answer -- it is the token that produced
    the wrong one.
    """

    compact = _compact(question)
    if _LISTING_DATE.search(compact):
        return None
    noun = next((token for token in _AFFILIATE_NOUNS if token in compact), None)
    if noun is None:
        return None
    if _LISTING_SPLIT in compact:
        context = _LISTING_SPLIT
    else:
        context = next(
            (token for token in _COUNT_REQUEST if token in compact), None
        )
    if context is None:
        return None
    return PhraseIntent(metric=AFFILIATE_COUNT, phrase=noun, context=context)


__all__ = [
    "AFFILIATE_COUNT",
    "CONTRACT_SALES_RATIO",
    "PhraseIntent",
    "affiliate_count_intent",
    "contract_sales_ratio_intent",
]
