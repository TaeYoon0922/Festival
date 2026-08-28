"""One place that decides whether two holder labels name the same reporter.

A filing writes its holder the way the filer typed it, so the same entity
arrives as ``(주)하이브``, ``㈜하이브``, ``주식회사 하이브`` or plain ``하이브``.
A question names it one way.  Matching those is a string question, not an
identity lookup: 99.8% of the holders in this corpus are external -- asset
managers, funds, employee associations, individuals, foreign entities -- and
have no ``corp_code`` to resolve against.

The designator is removed from the **raw** string before normalization, and only
where it is syntactically recognizable as a legal form: a bracketed ``(주)``, the
composed ``㈜``, or a full word like ``주식회사``.  Normalizing first would be
wrong -- it deletes the brackets and leaves a bare ``주`` that is
indistinguishable from the first syllable of ``주성엔지니어링``, and stripping
that would rename real companies.

Matching is exact on the canonical key plus the pre-existing suffix family; it
is never containment.  ``영풍`` and ``영풍정밀`` are different holders, and any
rule that merges them has failed regardless of what it recovers.
"""

from __future__ import annotations

import re

#: Bracketed and composed abbreviations: ``(주)``, ``( 유 )``, ``㈜``, ``㈝``.
_BRACKETED = r"\(\s*(?:주|유)\s*\)|[㈜㈝]"

#: Spelled-out legal forms.  Longer alternatives precede their own prefixes so
#: ``유한책임회사`` is never truncated to ``유한회사``.
_SPELLED = r"주식회사|유한책임회사|유한회사|합자회사|합명회사|사단법인|재단법인"

_PREFIX = re.compile(rf"^\s*(?:{_BRACKETED}|{_SPELLED})\s*")
_SUFFIX = re.compile(rf"\s*(?:{_BRACKETED}|{_SPELLED})\s*$")

#: Retained from the original matcher: a holder may be named with or without the
#: word describing what kind of body it is, as in 국민연금 / 국민연금공단.
_FAMILY_SUFFIXES = ("공단", "기금", "조합", "법인", "회사")


def canonical_reporter_key(value: object) -> str:
    """Comparison key for a holder label.  Never used for display."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    for pattern in (_PREFIX, _SUFFIX):
        # At most one designator per edge, and never the whole name: a holder
        # recorded only as its legal form still has to compare as itself.
        stripped = pattern.sub("", raw, count=1)
        if stripped.strip():
            raw = stripped
    return re.sub(r"[^0-9a-z가-힣]+", "", raw.casefold())


def reporter_matches(value: object, constraint: object) -> bool:
    """Whether a holder label answers to the holder a question asked for.

    An empty key matches nothing.  Placeholder cells such as ``-`` and ``…``
    both reduce to nothing, and treating them as equal would assert that two
    filings with no stated holder describe the same one.
    """

    left = canonical_reporter_key(value)
    right = canonical_reporter_key(constraint)
    if not left or not right:
        return False
    if left == right:
        return True
    return any(
        left == right + suffix or right == left + suffix
        for suffix in _FAMILY_SUFFIXES
    )
