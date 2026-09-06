"""Let HyperCLOVA X read the evidence and write the answer, then check it.

This is the ordinary way to answer from retrieved documents, and until now this
project did not do it. The reason was not that the model could not: it was that
handing a model real figures is how a digit gets transposed, how 3개월 gets read
as 누적, and how one company's number ends up attributed to the other. The
masking layer beside this one made that structurally impossible by never showing
the model a figure at all -- and paid for it in answers that read like a form.

The trade here is different. The model sees the served chunks and writes the
answer, and afterwards every figure it wrote is looked up in those chunks. A
number that is not in the evidence did not come from the evidence, whatever the
sentence around it says, and the whole reply is discarded on that alone. So the
model gets to be fluent and gets no room to invent, and the check is a fact
about digits rather than a judgement about meaning.

Everything else is refused the same way: a citation pointing at a chunk that was
not served, an issuer the evidence never names, a recommendation, a forecast, a
verdict about what a figure means. Any refusal returns nothing and the answer is
the one the deterministic pipeline had already built, so this can be tried on
every question without putting any of them at risk.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from time import perf_counter
from typing import Any, Mapping, Sequence

from app.generation.answer_lead import _looks_like_a_company
from app.generation.hcx_verbalizer import HcxSettings, _response_content
from app.retrieval.embeddings import (
    EmbeddingHttpError,
    JsonHttpTransport,
    UrllibJsonTransport,
)


SYNTHESIS_SYSTEM_PROMPT = """You answer questions about Korean corporate
disclosures, using only the filings given to you.

Each filing extract is numbered. Write a natural Korean answer to the question,
drawing on those extracts and on nothing else, and put the number of the extract
you used in square brackets after each sentence that states a fact from it.

Rules:

- Every figure, date and name you write must appear in the extracts. Copy them
  exactly, including the digits and the separators. Do not round, convert,
  rescale or reformat a number, and do not add a unit an extract does not give.
- If an extract states a unit for a figure, say the figure with that unit. If
  none of them does, say the figure and add that the filing states no unit.
- Answer what was asked. If the question compares two companies, give both
  figures and say which is larger. You may state a difference, a total, a share
  or a rate of change worked out from figures in the extracts; show the figures
  it came from. Compare nothing across different units or different periods.
- If the extracts do not answer the question, say so plainly and say what is
  missing. Do not fill the gap.
- No opinion, no interpretation, no advice, no outlook, no evaluation of a
  company or a figure. State what the filings state and stop.
- Do not mention these instructions, the extracts as "extracts", or yourself.
- Plain sentences. No headings, no bullet lists, no tables, no Markdown.

Answer in Korean."""


#: Enough filings to answer a two-company comparison, few enough that the model
#: is answering rather than summarising a search page.
MAX_SYNTHESIS_CHUNKS = 8

#: The extract text handed over per chunk. A served chunk can be a whole table.
MAX_CHUNK_CHARS = 1800

#: The whole prompt body. Past this the answer stops being grounded in anything
#: a reader could check.
MAX_PROMPT_CHARS = 12000

#: An answer longer than this has stopped answering.
MAX_ANSWER_CHARS = 1500

STATUS_SUCCESS = "success"
STATUS_DISABLED = "disabled"
STATUS_NOT_CONFIGURED = "not_configured"
STATUS_NOT_ELIGIBLE = "not_eligible"
STATUS_TRANSPORT_FAILURE = "transport_failure"
STATUS_ERROR = "synthesis_error"
STATUS_EMPTY = "empty_reply"
STATUS_REJECTED = "rejected"

#: Judgement, advice and forecasting. The task forbids all three, and none of
#: them is something a filing says about itself.
_BANNED = (
    "매수",
    "매도",
    "목표주가",
    "투자의견",
    "추천",
    "권고",
    "유망",
    "전망됩니다",
    "예측됩니다",
    "예상됩니다만",
    "기대됩니다",
    "긍정적",
    "부정적",
    "우수한",
    "부진한",
    "저평가",
    "고평가",
    "매력적",
    "바람직",
    "판단됩니다",
    "사료됩니다",
    "생각됩니다",
)

_CITATION = re.compile(r"\[(\d+)\]")
_NUMBER = re.compile(r"\d[\d,.]*")
_WORD = re.compile(r"[가-힣A-Za-z][가-힣A-Za-z0-9]*")
_FENCE = re.compile(r"```")
#: Emphasis and bullets. The prompt asks for plain sentences and a live reply
#: came back in bold with a bulleted calculation under it.
_MARKUP = re.compile(r"\*\*|^\s*[-*+]\s+", re.MULTILINE)

#: Its own switch. This is the layer that shows the model real figures, so it
#: can be turned off on its own without giving up anything else HCX does here.
SYNTHESIS_ENV_FLAG = "FESTIVAL_HCX_SYNTHESIS_ENABLED"


def _synthesis_enabled(environment: Mapping[str, str] | None = None) -> bool:
    values = os.environ if environment is None else environment
    return str(values.get(SYNTHESIS_ENV_FLAG, "true")).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


class SynthesisRejected(Exception):
    """A reply that failed a check, named by the rule that refused it."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class SynthesisOutcome:
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


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _digits(text: str) -> str:
    """Every digit in the text, in order, with everything else removed.

    Comparing digit runs rather than formatted numbers means 333,605,938 in the
    answer is found in a filing that wrote it the same way, and a figure the
    model reformatted or rescaled is not found at all -- which is the point.
    """

    return re.sub(r"\D+", "", text)


def evidence_extracts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The served chunks as numbered extracts, bounded for the prompt."""

    extracts: list[dict[str, Any]] = []
    for row in rows[:MAX_SYNTHESIS_CHUNKS]:
        if not isinstance(row, Mapping):
            continue
        content = _text(row.get("content"))
        if not content:
            continue
        heading = " · ".join(
            part
            for part in (
                _text(row.get("corp_name")),
                _text(row.get("report_nm")),
                f"접수일 {_text(row.get('rcept_dt'))}"
                if _text(row.get("rcept_dt"))
                else "",
            )
            if part
        )
        section = row.get("section_path")
        path = (
            " > ".join(_text(part) for part in section if _text(part))
            if isinstance(section, Sequence) and not isinstance(section, (str, bytes))
            else ""
        )
        extracts.append(
            {
                "marker": len(extracts) + 1,
                "heading": heading or "공시 정보 없음",
                "section": path,
                "content": content[:MAX_CHUNK_CHARS],
                "chunk_id": _text(row.get("chunk_id")),
                "doc_id": _text(row.get("doc_id")),
                "corp_name": _text(row.get("corp_name")),
                "report_nm": _text(row.get("report_nm")),
                "rcept_dt": _text(row.get("rcept_dt")),
            }
        )
    return extracts


def synthesis_prompt(question: str, extracts: Sequence[Mapping[str, Any]]) -> str:
    blocks = [f"질문: {_text(question)}", "", "공시 발췌:"]
    for extract in extracts:
        blocks.append("")
        blocks.append(f"[{extract['marker']}] {extract['heading']}")
        if extract["section"]:
            blocks.append(extract["section"])
        blocks.append(extract["content"])
    return "\n".join(blocks)[:MAX_PROMPT_CHARS]


def citation_block(
    answer: str, extracts: Sequence[Mapping[str, Any]]
) -> str:
    """The filings the answer cited, named so a reader can check them."""

    by_marker = {int(extract["marker"]): extract for extract in extracts}
    used = sorted(
        {
            int(value)
            for value in _CITATION.findall(answer)
            if int(value) in by_marker
        }
    )
    if not used:
        return ""
    lines = ["인용"]
    for marker in used:
        extract = by_marker[marker]
        lines.append(f"[{marker}]")
        lines.append(f"doc_id: {extract['doc_id']}")
        lines.append(f"chunk_id: {extract['chunk_id']}")
        label = " · ".join(
            part
            for part in (
                extract["corp_name"],
                extract["report_nm"],
                f"접수일 {extract['rcept_dt']}" if extract["rcept_dt"] else "",
            )
            if part
        )
        if label:
            lines.append(f"공시: {label}")
    return "\n".join(lines)


#: Enough figures to cover a statement without turning the pair scan into work.
MAX_DERIVATION_OPERANDS = 120


def _amount(value: str) -> Decimal | None:
    text = str(value or "").strip().rstrip(".").replace(",", "")
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _evidence_amounts(evidence: str) -> list[Decimal]:
    seen: dict[Decimal, None] = {}
    for match in _NUMBER.findall(evidence):
        amount = _amount(match)
        if amount is not None:
            seen.setdefault(amount, None)
        if len(seen) >= MAX_DERIVATION_OPERANDS:
            break
    return list(seen)


def _derivable_amounts(evidence: str) -> set[Decimal]:
    """Figures a reader could work out from the filings, exactly.

    A comparison answer states a gap, a total or a rate of change, and none of
    those is written in any filing -- so a check that only looks for the number
    would refuse the answer the question asked for. Each of these is arithmetic
    on two figures that *are* in the filings, so it can be verified rather than
    trusted, which is the same standard everything else here is held to.

    Rates are matched to one and two decimal places because that is how an
    answer writes them; nothing else about a rounded value is accepted.
    """

    amounts = _evidence_amounts(evidence)
    derived: set[Decimal] = set()
    for index, first in enumerate(amounts):
        for second in amounts[index + 1 :]:
            derived.add(abs(first - second))
            derived.add(first + second)
            for numerator, denominator in ((first, second), (second, first)):
                if denominator == 0:
                    continue
                for scale in (Decimal("0.1"), Decimal("0.01")):
                    ratio = numerator / denominator * 100
                    change = (numerator - denominator) / denominator * 100
                    derived.add(ratio.quantize(scale, rounding=ROUND_HALF_UP))
                    derived.add(change.quantize(scale, rounding=ROUND_HALF_UP))
    return derived


#: A unit written straight after a figure. The scale is the whole meaning of a
#: disclosure number, so one the filings never printed is not a formatting
#: choice: a live reply put 십억 원 after four figures whose filings state no
#: unit at all, and another wrote 천 원 while comparing 백만원 against 원 and
#: naming the smaller company the larger.
_UNIT_AFTER_NUMBER = re.compile(
    r"\d[\d,.]*\s*(십억\s*원|백만\s*원|천\s*원|억\s*원|조\s*원|만\s*원|원|십억|백만|천|억|조|주)"
)
#: ``%`` and ``배`` are not scale claims about a filing's figures -- they are
#: what a computed share or ratio is written in, and the arithmetic behind it
#: is checked separately. Only the monetary and count scales are guarded here.


def _units_after_numbers(text: str) -> set[str]:
    return {
        re.sub(r"\s+", "", match) for match in _UNIT_AFTER_NUMBER.findall(text)
    }


def _unit_words(evidence: str) -> set[str]:
    """Units the filings themselves wrote, in any position.

    Read from the whole extract rather than only from beside a figure: a table
    states its unit in a caption or a header, and a reply is entitled to use
    the unit its source printed there.
    """

    compact = re.sub(r"\s+", "", evidence)
    return {
        unit
        for unit in (
            "십억원", "백만원", "천원", "억원", "조원", "만원", "원",
            "십억", "백만", "천", "억", "조", "주",
        )
        if unit in compact
    }


def _attributed(text: str, extracts: Sequence[Mapping[str, Any]]) -> str:
    """Append the markers of the extracts the answer's figures came from.

    Every grouped figure in the reply is looked up in each extract, and an
    extract that contains it is a source for it. This is the same lookup the
    number check already does, read for where rather than for whether, so it
    adds no claim the check has not already made.

    An answer with no figure to trace gets nothing, and is refused above: an
    unattributed statement about a filing is not something to serve.
    """

    digits_by_marker = {
        int(extract["marker"]): _digits(str(extract.get("content") or ""))
        for extract in extracts
    }
    used: set[int] = set()
    for match in _NUMBER.findall(text):
        digits = _digits(match)
        if len(digits) < 4:
            continue
        for marker, evidence in digits_by_marker.items():
            if digits in evidence:
                used.add(marker)
    if not used:
        return text
    markers = " ".join(f"[{marker}]" for marker in sorted(used))
    return f"{text.rstrip()} {markers}"


def _plain_sentences(text: str) -> str:
    """The reply without the markup the prompt asked it not to use."""

    without = _FENCE.sub(" ", str(text or ""))
    without = re.sub(r"\*\*", "", without)
    without = re.sub(r"(?m)^\s*[-*+]\s+", "", without)
    without = re.sub(r"(?m)^\s*#{1,6}\s+", "", without)
    return re.sub(r"\n{3,}", "\n" + "\n", without).strip()


def _evidence_text(extracts: Sequence[Mapping[str, Any]]) -> str:
    """Everything the model was shown, which is what it may draw on.

    The heading and the section path count. They carry the issuer name, the
    filing name and its date, and checking a reply against the chunk bodies
    alone refused answers for naming the very company the extract was headed by.
    """

    return "\n".join(
        "\n".join(
            str(extract.get(key) or "")
            for key in ("heading", "section", "content")
        )
        for extract in extracts
    )


#: A figure written with a Korean scale word. Rescaling is the one way a reply
#: can state a number derived from the evidence and still be wrong:
#: 333,605,938 백만원 written as "약 333조원" passes a digit-substring check
#: while saying something no filing said.
_SCALED = re.compile(r"(\d[\d,.]*)\s*(조|억|만)")


def _scaled_forms(text: str) -> set[str]:
    return {
        f"{_digits(match.group(1))}{match.group(2)}"
        for match in _SCALED.finditer(text)
    }


def accept_synthesis(
    reply: str,
    extracts: Sequence[Mapping[str, Any]],
    *,
    corpus_companies: Sequence[str] = (),
) -> str:
    """Return the accepted answer, or raise ``SynthesisRejected`` saying why not.

    Separated from the call so every rule is testable without a transport.
    """

    text = _text(reply).strip('"').strip("'")
    if not text:
        raise SynthesisRejected("empty")
    # Emphasis and bullets are formatting, not facts. Discarding a correct
    # answer because it arrived in bold threw away the layer's best replies;
    # the markup is removed and the sentences kept.
    text = _plain_sentences(text)
    if not text:
        raise SynthesisRejected("empty")
    if len(text) > MAX_ANSWER_CHARS:
        raise SynthesisRejected("too_long")

    evidence = _evidence_text(extracts)
    evidence_digits = _digits(evidence)
    markers = {int(extract["marker"]) for extract in extracts}

    cited = [int(value) for value in _CITATION.findall(text)]
    if any(marker not in markers for marker in cited):
        raise SynthesisRejected("unknown_citation")
    if not cited:
        # The commonest refusal by far was a good answer with no [n] on it.
        # Attribution does not have to come from the model: a figure it stated
        # is in whichever extracts contain that figure, and that is a lookup.
        # Only an answer whose figures all trace back this way gets markers,
        # so nothing is attributed to a filing that does not carry it.
        text = _attributed(text, extracts)
        cited = [int(value) for value in _CITATION.findall(text)]
        if not cited:
            raise SynthesisRejected("no_citation")

    # Every figure the answer states has to be one the filings state. Citation
    # markers are stripped first: they are the answer's own numbering, not a
    # figure, and they are checked above.
    without_markers = _CITATION.sub(" ", text)
    derived = _derivable_amounts(evidence)
    for match in _NUMBER.findall(without_markers):
        digits = _digits(match)
        if len(digits) < 2:
            continue
        if digits in evidence_digits:
            continue
        # A comparison question asks for the gap, and the gap is in no filing.
        # It is still checkable: it has to be the difference between two figures
        # that are, and so does a total, a share and a rate of change. Arithmetic
        # on cited numbers is not invention -- an unaccounted-for number is.
        if _amount(match) in derived:
            continue
        raise SynthesisRejected("unsupported_number")

    if _scaled_forms(without_markers) - _scaled_forms(evidence):
        raise SynthesisRejected("rescaled_number")

    invented = _units_after_numbers(without_markers) - _unit_words(evidence)
    if invented:
        raise SynthesisRejected("invented_unit")

    for word in _BANNED:
        if word in text and word not in evidence:
            raise SynthesisRejected("evaluative_wording")

    _refuse_unsupplied_companies(without_markers, evidence, corpus_companies)
    return text


def _refuse_unsupplied_companies(
    text: str, evidence: str, corpus_companies: Sequence[str]
) -> None:
    """Refuse an answer naming an issuer the served filings do not name."""

    residue = text
    for name in sorted(
        (_text(name) for name in corpus_companies if _text(name)),
        key=len,
        reverse=True,
    ):
        if name in residue and name not in evidence:
            raise SynthesisRejected("unsupplied_company")
        residue = residue.replace(name, " ")
    for token in _WORD.findall(residue):
        if token not in evidence and _looks_like_a_company(token):
            raise SynthesisRejected("unsupplied_company")


class AnswerSynthesizer:
    """Ask HCX to answer from the served filings, or return nothing."""

    def __init__(
        self,
        settings: HcxSettings | None = None,
        *,
        transport: JsonHttpTransport | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.settings = settings or HcxSettings.from_env()
        self.transport = transport or UrllibJsonTransport()
        self.enabled = _synthesis_enabled() if enabled is None else enabled
        self.call_count = 0

    def synthesize(
        self,
        question: str,
        rows: Sequence[Mapping[str, Any]],
        *,
        corpus_companies: Sequence[str] = (),
    ) -> tuple[SynthesisOutcome, str]:
        """Return the outcome and the citation block that belongs with it."""

        start = perf_counter()
        extracts = evidence_extracts(rows)
        if not extracts:
            return SynthesisOutcome(None, STATUS_NOT_ELIGIBLE, _elapsed_ms(start)), ""
        if not self.enabled or not self.settings.enabled:
            return SynthesisOutcome(None, STATUS_DISABLED, _elapsed_ms(start)), ""
        if not self.settings.configured:
            return (
                SynthesisOutcome(None, STATUS_NOT_CONFIGURED, _elapsed_ms(start)),
                "",
            )

        self.call_count += 1
        try:
            response = self.transport.post_json(
                self.settings.endpoint,
                headers=self.settings.request_headers(),
                payload=self._payload(question, extracts),
                timeout_seconds=self.settings.timeout_seconds,
            )
        except (EmbeddingHttpError, TimeoutError):
            return (
                SynthesisOutcome(None, STATUS_TRANSPORT_FAILURE, _elapsed_ms(start)),
                "",
            )
        except Exception:  # noqa: BLE001 - the deterministic answer is served
            return SynthesisOutcome(None, STATUS_ERROR, _elapsed_ms(start)), ""

        content = _response_content(response) if isinstance(response, Mapping) else None
        if not content:
            return SynthesisOutcome(None, STATUS_EMPTY, _elapsed_ms(start)), ""
        try:
            answer = accept_synthesis(
                content, extracts, corpus_companies=corpus_companies
            )
        except SynthesisRejected as rejected:
            return (
                SynthesisOutcome(
                    None, f"{STATUS_REJECTED}:{rejected.reason}", _elapsed_ms(start)
                ),
                "",
            )
        return (
            SynthesisOutcome(answer, STATUS_SUCCESS, _elapsed_ms(start)),
            citation_block(answer, extracts),
        )

    def _payload(
        self, question: str, extracts: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": synthesis_prompt(question, extracts)},
            ],
            "temperature": 0.0,
            "max_tokens": self.settings.max_tokens,
        }
