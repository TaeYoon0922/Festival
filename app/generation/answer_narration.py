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

from app.generation.answer_generator import UNIT_ABSENT_NOTICE
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


ANSWER_SYSTEM_PROMPT = """You write the answer to a Korean disclosure question.

You are given verified facts as short Korean lines, followed by a TAGS line.
A tag looks like __FESTIVAL_NUMBER_A__ or __FESTIVAL_CITATION_B__ and stands
for a figure, a date or a source reference that has been hidden from you.

Your reply MUST contain every tag on the TAGS line, spelled exactly as written,
in the same order, each exactly once. A __FESTIVAL_CITATION__ tag is one of
them: it is not punctuation and not a footnote you may leave out. Write it
after the closing full stop of the sentence whose figure it supports, never
inside a clause -- "...입니다. __FESTIVAL_CITATION_B__", not "...이는
__FESTIVAL_CITATION_B__에 따른 것입니다". A reply missing a tag is thrown away,
so check the TAGS line before you answer.

Write one or two natural Korean sentences that answer the question the facts
answer. Lead with the company, the period, the measure and its value tag. Keep
any parenthetical remark the lines make about what the filing does not state.

Never do any of these:

- Write a digit of your own, in any form.
- Invent, drop, reorder or duplicate a tag.
- Write your own bracketed number; the citation tag already stands for one.
- Name a company, a period or a measure the lines do not name.
- Explain, interpret or comment on what a figure means.
- Decide for yourself which value is larger, or by how much. You cannot see the
  figures. If one of the lines already states a comparison, restate that line
  and change nothing about which side it names.
- Estimate, recommend or predict anything.
- Mention how confident the answer is.
- Add a heading, a bullet list, a table, a quotation mark or a Markdown fence.

Reply with those sentences and nothing else."""


#: The heading the deterministic generator writes above the citation block.
#: Everything from it onward is identifiers, and is never sent to the model.
CITATION_HEADING = "인용"

#: The heading above the sentences that answer the question. When it is there,
#: it is the only thing the model is asked to write: the competition requires
#: the answer to be generated by HyperCLOVA X, and asking it for one sentence
#: carrying two tokens is a question it can get right, where asking it to carry
#: a whole table's worth of figures was not.
ANSWER_HEADING = "답변"

#: Below this the answer is a one-line refusal or a bare statement; rewriting it
#: risks damage and gains nothing.
MIN_NARRATION_CHARS = 80

#: Above this the answer is long enough that placeholder survival stops being
#: reliable, and a fallback is cheaper than a mangled rewrite.
MAX_NARRATION_CHARS = 6000

#: Prose that restates cannot grow much. A reply well past its input has started
#: adding, and what it added is not in the evidence.
#:
#: Measured on the text with its placeholders removed, because a placeholder is
#: twenty-odd characters standing in for four and swamps the comparison: the
#: masked form of a one-sentence answer is long enough that a reply which
#: doubled its prose still looked like a modest rewrite. It was
#: "삼성전자의 매출액은 X원으로, 이는 회사의 재무 상태와 시장에서의 경쟁력을
#: 나타내는 중요한 지표 중 하나입니다" -- a judgement about a figure the model
#: could not see, served as though the filing had said it.
MAX_GROWTH_RATIO = 1.4

#: Rewording needs a little room even when the input is one short line.
GROWTH_ALLOWANCE_CHARS = 24

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
    # Commentary. A model with no figure in front of it explaining what a figure
    # means is writing about the company, not restating the filing.
    "나타내",
    "의미합니다",
    "의미하는",
    "중요한",
    "볼 수 있",
    "판단됩니다",
    "평가됩니다",
    "반영합니다",
    "반영한",
    "시사",
    "해석",
    "경쟁력",
    "수익성이",
    "성과를",
)

_CITATION_MARKER = re.compile(r"\[\d+\]")
_DIGIT = re.compile(r"\d")
_FENCE = re.compile(r"```")
_WORD = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9]*")


#: Its own switch, separate from ``FESTIVAL_HCX_ENABLED``. The rewrite is the
#: newest thing HCX does here and the only one whose live behaviour is not yet
#: measured, so it can be turned off by restarting the server without giving up
#: the verbalizer, the semantic fallback, or the opening line.
#: Default on, for the same reason as the synthesis layer beside it: the
#: pipeline only reaches it for a narrative answer.
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


def split_answer_section(body: str) -> tuple[str, str, str] | None:
    """Split the body around its 답변 section: before, the answer, after.

    Returns ``None`` when the body has no such section, which is how an answer
    with no resolved figure -- a refusal, a clarification, an evidence-only
    reply -- opts into having the whole body rewritten instead.
    """

    lines = str(body or "").split("\n")
    for index, line in enumerate(lines):
        if line.strip() != ANSWER_HEADING:
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip():
            end += 1
        answer = "\n".join(lines[index + 1 : end]).strip()
        if not answer:
            return None
        return (
            "\n".join(lines[:index]).strip(),
            answer,
            "\n".join(lines[end:]).strip(),
        )
    return None


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
    if _CITATION_MARKER.search(text):
        raise NarrationRejected("citation_marker")

    integrity = check_placeholder_integrity(text, protection)
    if not integrity.valid:
        raise NarrationRejected(str(integrity.reason))

    written = PLACEHOLDER_PATTERN.sub("", text)
    given = PLACEHOLDER_PATTERN.sub("", protection.masked)
    if len(written) > max(
        len(given) + GROWTH_ALLOWANCE_CHARS, int(len(given) * MAX_GROWTH_RATIO)
    ):
        raise NarrationRejected("expanded")

    # Checked before restoration: afterwards the restored figures are digits
    # that belong, and the rule "no digit" would no longer be decidable.
    if _DIGIT.search(written):
        raise NarrationRejected("digit")

    # Source-relative on purpose. A filing that itself says 예상 or 경쟁력 may be
    # restated saying it; the rule is about what the model added, not about
    # which words exist.
    source = protection.original
    for word in _BANNED:
        if word in text and word not in source:
            raise NarrationRejected("evaluative_wording")

    _refuse_unsupplied_companies(text, source, corpus_companies)
    return _keep_required_notices(restore_literals(text, protection), source)


#: Statements the deterministic answer makes that a rewrite may not lose. They
#: are prose, not tokens, so placeholder integrity does not protect them, and a
#: live reply dropped the unit caveat and served a grouped figure with nothing
#: saying the filing had not stated a unit for it.
REQUIRED_NOTICES = (UNIT_ABSENT_NOTICE,)


def _keep_required_notices(text: str, source: str) -> str:
    """Put back a notice the rewrite dropped, rather than discarding the rewrite.

    The notice is deterministic text about what the filing does not say, so
    appending it restores a fact rather than asserting a new one.
    """

    missing = [
        notice
        for notice in REQUIRED_NOTICES
        if notice in source and notice not in text
    ]
    return " ".join([text.rstrip(), *missing]) if missing else text


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

        # When the deterministic layer resolved a figure it writes a 답변
        # section, and that section is the answer. HyperCLOVA X is asked to
        # write that and nothing else: the evidence below it is already
        # readable, and every figure left in it is a token the model would
        # otherwise have to carry for no gain. Without such a section there is
        # no single sentence to write, so the whole body is rewritten as before.
        parts = split_answer_section(body)
        target = parts[1] if parts is not None else narration_source(body)
        prompt = ANSWER_SYSTEM_PROMPT if parts is not None else NARRATION_SYSTEM_PROMPT

        if parts is None and len(body.strip()) < MIN_NARRATION_CHARS:
            return NarrationOutcome(None, STATUS_NOT_ELIGIBLE, _elapsed_ms(start))
        if contains_placeholder_syntax(body):
            return NarrationOutcome(None, STATUS_NOT_ELIGIBLE, _elapsed_ms(start))
        if not self.enabled or not self.settings.enabled:
            return NarrationOutcome(None, STATUS_DISABLED, _elapsed_ms(start))
        if not self.settings.configured:
            return NarrationOutcome(None, STATUS_NOT_CONFIGURED, _elapsed_ms(start))
        if len(target) > MAX_NARRATION_CHARS:
            return NarrationOutcome(None, STATUS_TOO_LONG, _elapsed_ms(start))

        protection = protect_literals(target)
        self.call_count += 1
        try:
            response = self.transport.post_json(
                self.settings.endpoint,
                headers=self.settings.request_headers(),
                payload=self._payload(protection, prompt),
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
        if parts is not None:
            before, _, after = parts
            narrated = "\n\n".join(
                block
                for block in (before, f"{ANSWER_HEADING}\n{narrated}", after)
                if block
            )
        joined = f"{narrated}\n\n{citations}" if citations else narrated
        return NarrationOutcome(joined, STATUS_SUCCESS, _elapsed_ms(start))

    def _payload(self, protection: ProtectedText, prompt: str) -> dict[str, Any]:
        # The tags are listed back to the model because the check that discards
        # a reply is exactly "are they all here, in this order". A reply that
        # dropped one was the commonest failure, and the citation tag was the
        # one most often dropped -- read, it seems, as a footnote marker rather
        # than as something to carry.
        content = protection.masked
        if protection.placeholders:
            content = f"{content}\n\nTAGS: {' '.join(protection.placeholders)}"
        return {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            "temperature": 0.0,
            "max_tokens": self.settings.max_tokens,
        }
