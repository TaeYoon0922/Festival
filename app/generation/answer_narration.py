"""Rewrite a finished answer into readable Korean, without showing it a value.

The deterministic generator produces a correct answer that reads like a dump:
label lines, an ordinal, and a markdown table row where a sentence belongs. The
competition asks for a 자연어 답변, and HyperCLOVA X is the model that has to
write it -- but handing a model the figures is exactly what this project's
safety design spent its effort preventing, and what made earlier attempts state
wrong numbers.

So the model is handed the answer with every citation marker, date and number
already replaced by a digit-free token. It cannot read a figure, and a reply
containing a digit is discarded on that fact alone. It rewrites the prose
around the tokens; the tokens are then checked for exact survival -- same set,
same order, once each -- and swapped back for the original characters, byte for
byte.

What this can change is wording. What it cannot change is any number, any date,
any citation, or which filing a figure came from. Every failure -- a dropped
token, a written digit, an invented issuer, a timeout, a disabled model --
returns no narration, and the answer is served exactly as it was built.

The citation block is never sent. Identifiers are not prose, and nothing is
gained by letting a model near them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Mapping, Sequence

from app.generation.answer_lead import _looks_like_a_company
from app.generation.hcx_verbalizer import HcxSettings, _response_content
from app.generation.protected_literals import (
    PLACEHOLDER_PATTERN,
    ProtectedText,
    check_placeholder_integrity,
    contains_placeholder_syntax,
    protect_literals,
    restore_literals,
)
from app.retrieval.embeddings import (
    EmbeddingHttpError,
    JsonHttpTransport,
    UrllibJsonTransport,
)


NARRATION_SYSTEM_PROMPT = """You rewrite a Korean disclosure answer so that a
person can read it.

Every citation marker, date and number in the text has already been replaced by
a token of the form __FESTIVAL_KIND_LABEL__. You cannot see any figure, and you
must not write one.

Rewrite the text as plain Korean prose:

- Copy every token exactly as given, in the same order, each one exactly once.
- Turn label lines and table rows into sentences: a row reading
  "매출액 | TOKEN" becomes "매출액은 TOKEN입니다".
- Keep every fact the text states, including which filing and which period it
  came from, and including any statement that something could not be confirmed
  or that a unit is not stated.
- Open with the sentence that answers what the text answers, using only what
  the text says.

Never do any of these:

- Write a digit, in any form.
- Invent, drop, reorder or duplicate a token.
- Write a citation marker such as a bracketed number.
- Name a company the text does not name.
- Conclude, compare, rank, estimate, recommend or predict anything.
- Add a heading, a bullet list, a quotation mark or a Markdown fence.

Reply with the rewritten Korean text and nothing else."""


#: The heading the deterministic generator writes above the citation block.
#: Everything from it onward is identifiers, and is never sent to the model.
CITATION_HEADING = "인용"

#: Below this the answer is a one-line refusal or a bare statement; rewriting it
#: risks damage and gains nothing.
MIN_NARRATION_CHARS = 80

#: Above this the answer is long enough that placeholder survival stops being
#: reliable, and a fallback is cheaper than a mangled rewrite.
MAX_NARRATION_CHARS = 6000

#: Prose that restates cannot grow much. A reply well past its input has started
#: adding, and what it added is not in the evidence.
MAX_GROWTH_RATIO = 1.6

STATUS_SUCCESS = "success"
STATUS_DISABLED = "disabled"
STATUS_NOT_CONFIGURED = "not_configured"
STATUS_NOT_ELIGIBLE = "not_eligible"
STATUS_TOO_LONG = "skipped_too_long"
STATUS_TRANSPORT_FAILURE = "transport_failure"
STATUS_ERROR = "narration_error"
STATUS_EMPTY = "empty_reply"
STATUS_REJECTED = "rejected"

#: Wording that turns a restatement into a judgement. The model is shown no
#: figure, so a comparison or a forecast in its reply is invented by definition.
_BANNED = (
    "더 큰",
    "더 많",
    "더 적",
    "가장",
    "제일",
    "최대",
    "최소",
    "전망",
    "예상",
    "추정",
    "예측",
    "권고",
    "추천",
    "매수",
    "매도",
    "유망",
    "우수",
    "부진",
    "긍정적",
    "부정적",
    "목표주가",
    "결론적으로",
)

_CITATION_MARKER = re.compile(r"\[\d+\]")
_DIGIT = re.compile(r"\d")
_FENCE = re.compile(r"```")
_WORD = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9]*")


#: Its own switch, separate from ``FESTIVAL_HCX_ENABLED``. The rewrite is the
#: newest thing HCX does here and the only one whose live behaviour is not yet
#: measured, so it can be turned off by restarting the server without giving up
#: the verbalizer, the semantic fallback, or the opening line.
NARRATION_ENV_FLAG = "FESTIVAL_HCX_NARRATION_ENABLED"


def _narration_enabled(environment: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environment is None else environment
    return str(values.get(NARRATION_ENV_FLAG, "true")).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


class NarrationRejected(Exception):
    """A reply that failed a check, named by the rule that refused it."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class NarrationOutcome:
    text: str | None
    status: str
    elapsed_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status == STATUS_SUCCESS and bool(self.text)

    def to_public_dict(self) -> dict[str, Any]:
        return {"status": self.status, "applied": self.succeeded}


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000.0, 3)


def split_citation_block(answer: str) -> tuple[str, str]:
    """Split an answer into the prose to rewrite and the citations to keep.

    The citation block is last, so the search runs from the end: a filing whose
    text happens to contain the word 인용 cannot be mistaken for the heading.
    """

    text = str(answer or "")
    lines = text.split("\n")
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].strip() == CITATION_HEADING:
            return "\n".join(lines[:index]).rstrip(), "\n".join(lines[index:])
    return text, ""


#: Ordinals the generator writes to bind metadata to an evidence block. They
#: are digits, so masking turns each one into a token the model must carry
#: through unchanged -- and a rewrite of one short answer was carrying thirteen
#: tokens of which nine were scaffolding. Written as words they leave the
#: placeholder set entirely, which is both a smaller ask and better Korean.
_ORDINAL_WORDS = (
    "첫 번째",
    "두 번째",
    "세 번째",
    "네 번째",
    "다섯 번째",
    "여섯 번째",
    "일곱 번째",
    "여덟 번째",
    "아홉 번째",
    "열 번째",
)

_EVIDENCE_ORDINAL = re.compile(r"근거\s*(\d+)(?=\s|$)")
_BARE_ORDINAL_LINE = re.compile(r"^\s*\d+\.\s*$")
_CONTENT_PREFIX = re.compile(r"^\s*내용:\s*")


def narration_source(body: str) -> str:
    """The answer as the model should see it, before literals are masked.

    Only presentation scaffolding is removed: the ordinal that numbers an
    evidence block, the bare "1." that precedes a source, and the "내용:" label
    in front of a chunk. No fact is touched. The served fallback stays the
    original answer, so this shapes what is asked for and never what is kept
    when the rewrite is refused.
    """

    lines: list[str] = []
    for line in str(body or "").split("\n"):
        if _BARE_ORDINAL_LINE.match(line):
            continue
        line = _CONTENT_PREFIX.sub("", line)
        line = _EVIDENCE_ORDINAL.sub(_ordinal_word, line)
        lines.append(line)
    return "\n".join(lines).strip()


def _ordinal_word(match: re.Match[str]) -> str:
    index = int(match.group(1))
    if 1 <= index <= len(_ORDINAL_WORDS):
        return f"{_ORDINAL_WORDS[index - 1]} 근거"
    return match.group(0)


def accept_narration(
    reply: str,
    protection: ProtectedText,
    *,
    corpus_companies: Sequence[str] = (),
) -> str:
    """Return the restored prose, or raise ``NarrationRejected`` saying why not.

    Separated from the call so every rule is testable without a transport, and
    so the rules read as one list rather than as branches around an HTTP result.
    """

    text = str(reply or "").strip().strip('"').strip("'")
    if not text:
        raise NarrationRejected("empty")
    if _FENCE.search(text):
        raise NarrationRejected("markdown_fence")
    limit = max(MIN_NARRATION_CHARS, int(len(protection.masked) * MAX_GROWTH_RATIO))
    if len(text) > limit:
        raise NarrationRejected("expanded")
    if _CITATION_MARKER.search(text):
        raise NarrationRejected("citation_marker")

    integrity = check_placeholder_integrity(text, protection)
    if not integrity.valid:
        raise NarrationRejected(str(integrity.reason))

    # Checked before restoration: afterwards the restored figures are digits
    # that belong, and the rule "no digit" would no longer be decidable.
    if _DIGIT.search(PLACEHOLDER_PATTERN.sub("", text)):
        raise NarrationRejected("digit")
    for word in _BANNED:
        if word in text:
            raise NarrationRejected("evaluative_wording")

    _refuse_unsupplied_companies(text, protection.original, corpus_companies)
    return restore_literals(text, protection)


def _refuse_unsupplied_companies(
    text: str, source: str, corpus_companies: Sequence[str]
) -> None:
    """Refuse a reply naming an issuer the answer being rewritten did not name.

    Korean agglutinates, so a supplied name arrives inside a longer token. The
    supplied vocabulary is removed first -- longest first, so a name containing
    another is not half-erased -- and only what is left is judged.
    """

    residue = PLACEHOLDER_PATTERN.sub(" ", text)
    for name in sorted(
        (str(name).strip() for name in corpus_companies if str(name).strip()),
        key=len,
        reverse=True,
    ):
        if name in residue and name not in source:
            raise NarrationRejected("unsupplied_company")
        residue = residue.replace(name, " ")
    for token in _WORD.findall(residue):
        if token not in source and _looks_like_a_company(token):
            raise NarrationRejected("unsupplied_company")


class AnswerNarrator:
    """Ask HCX to rewrite an answer readably, or return no narration."""

    def __init__(
        self,
        settings: HcxSettings | None = None,
        *,
        transport: JsonHttpTransport | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.settings = settings or HcxSettings.from_env()
        self.transport = transport or UrllibJsonTransport()
        self.enabled = _narration_enabled() if enabled is None else enabled
        self.call_count = 0

    def narrate(
        self, answer: str, *, corpus_companies: Sequence[str] = ()
    ) -> NarrationOutcome:
        start = perf_counter()
        body, citations = split_citation_block(answer)
        if len(body.strip()) < MIN_NARRATION_CHARS:
            return NarrationOutcome(None, STATUS_NOT_ELIGIBLE, _elapsed_ms(start))
        if contains_placeholder_syntax(body):
            return NarrationOutcome(None, STATUS_NOT_ELIGIBLE, _elapsed_ms(start))
        if not self.enabled or not self.settings.enabled:
            return NarrationOutcome(None, STATUS_DISABLED, _elapsed_ms(start))
        if not self.settings.configured:
            return NarrationOutcome(None, STATUS_NOT_CONFIGURED, _elapsed_ms(start))
        if len(body) > MAX_NARRATION_CHARS:
            return NarrationOutcome(None, STATUS_TOO_LONG, _elapsed_ms(start))

        protection = protect_literals(narration_source(body))
        self.call_count += 1
        try:
            response = self.transport.post_json(
                self.settings.endpoint,
                headers=self.settings.request_headers(),
                payload=self._payload(protection),
                timeout_seconds=self.settings.timeout_seconds,
            )
        except (EmbeddingHttpError, TimeoutError):
            return NarrationOutcome(None, STATUS_TRANSPORT_FAILURE, _elapsed_ms(start))
        except Exception:  # noqa: BLE001 - the answer is served as it was built
            return NarrationOutcome(None, STATUS_ERROR, _elapsed_ms(start))

        content = _response_content(response) if isinstance(response, Mapping) else None
        if not content:
            return NarrationOutcome(None, STATUS_EMPTY, _elapsed_ms(start))
        try:
            narrated = accept_narration(
                content, protection, corpus_companies=corpus_companies
            )
        except NarrationRejected as rejected:
            return NarrationOutcome(
                None, f"{STATUS_REJECTED}:{rejected.reason}", _elapsed_ms(start)
            )
        joined = f"{narrated}\n\n{citations}" if citations else narrated
        return NarrationOutcome(joined, STATUS_SUCCESS, _elapsed_ms(start))

    def _payload(self, protection: ProtectedText) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": NARRATION_SYSTEM_PROMPT},
                {"role": "user", "content": protection.masked},
            ],
            "temperature": 0.0,
            "max_tokens": self.settings.max_tokens,
        }
