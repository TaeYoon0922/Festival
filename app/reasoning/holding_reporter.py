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


#: What may sit directly against a holder's name without being part of it.
#: Korean writes its postpositions onto the noun, so a name's right edge cannot
#: be found by looking for a space: ``영풍의`` and ``영풍정밀`` differ only in
#: whether what follows the name is grammar or more name, and the module's own
#: rule is that those two are different holders.  Only the postpositions a
#: holder actually carries in a holding question are listed; anything else
#: after the name means the name did not end there.
_HOLDER_POSTPOSITIONS = frozenset(
    {
        "", "의", "이", "가", "은", "는", "을", "를", "에", "와", "과", "도",
        "만", "께서", "에서", "으로", "로", "에게", "한테",
    }
)

#: The characters a name is made of.  Matches what ``canonical_reporter_key``
#: keeps, so a key can be searched for in text compacted the same way.
_NAME_CHARS = re.compile(r"[0-9a-z가-힣]")


def _compacted(value: str) -> tuple[str, tuple[int, ...]]:
    """The name characters of ``value``, plus where each came from."""

    characters: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(value):
        # Folded one character at a time so an index always refers to the
        # original string.  A fold that expands (``ß`` becomes ``ss``) stops
        # matching the single-character class and is dropped, which loses a
        # match rather than misreporting where one is.
        folded = character.casefold()
        if _NAME_CHARS.fullmatch(folded):
            characters.append(folded)
            positions.append(index)
    return "".join(characters), tuple(positions)


def reporter_surface_spans(question: str, reporter: object) -> tuple[tuple[int, int], ...]:
    """Where a known holder's name is written in a question, if it is.

    The corpus supplies the identity and the question supplies the text: this
    only asks whether the two are the same name, using the canonical key the
    rest of this module compares with, so spacing, punctuation and legal forms
    are already accounted for.

    It is deliberately not containment.  A match survives only when the name
    occupies a whole name-run of the question -- nothing naming before it, and
    after it nothing but one of the postpositions above.  That is what keeps
    ``영풍`` out of ``영풍정밀`` and, just as importantly, keeps an issuer's own
    name out of a holder's slot when the two share a prefix.
    """

    key = canonical_reporter_key(reporter)
    text = str(question or "")
    if not key or not text:
        return ()
    compact, positions = _compacted(text)
    spans: list[tuple[int, int]] = []
    start = compact.find(key)
    while start >= 0:
        end = start + len(key)
        if _bounded(positions, text, start, end):
            spans.append((positions[start], positions[end - 1] + 1))
        start = compact.find(key, start + 1)
    return tuple(spans)


def _bounded(
    positions: tuple[int, ...], text: str, start: int, end: int
) -> bool:
    """Whether the match is the whole name rather than part of a longer one."""

    if start > 0 and positions[start] - positions[start - 1] == 1:
        # A name character runs straight into the match: the name began earlier.
        return False
    tail = text[positions[end - 1] + 1 :]
    # Everything between the match and the next break has to be grammar. A
    # space, punctuation or the end of the question breaks; another name
    # character does not, which is the whole point.
    run = len(tail)
    for offset, character in enumerate(tail):
        if not _NAME_CHARS.fullmatch(character.casefold()):
            run = offset
            break
    return tail[:run] in _HOLDER_POSTPOSITIONS
