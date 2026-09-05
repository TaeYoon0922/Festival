"""One sentence saying what the answer below is about, written by HCX.

An evidence answer opens straight into its first filing. The reader reaches a
markdown table before learning which companies were consulted or what was being
looked for, and the answer reads as a dump even after the retrieval scaffolding
is cleaned up.

This asks HyperCLOVA X for the missing opening line, under the same discipline
the verbalizer uses: the model is handed no figures, no citations and no filing
text -- only company names, filing kinds, and placeholders standing in for every
number. It can therefore phrase the framing but cannot state a fact, because it
was never shown one.

Every number is restored deterministically after the call. The accepted reply
must contain no digits at all, which makes the check a single unambiguous rule
rather than a judgement about whether a figure was altered: a digit in the reply
means the model produced one, and a reply that produced one is discarded.

A lead is a nicety, so every failure -- a refused check, a timeout, a disabled
model, an exception -- returns no lead and the answer is served exactly as it
was built.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping, Sequence

from app.generation.hcx_verbalizer import HcxSettings, _response_content
from app.retrieval.embeddings import (
    EmbeddingHttpError,
    JsonHttpTransport,
    UrllibJsonTransport,
)


LEAD_SYSTEM_PROMPT = """You write one Korean sentence introducing a disclosure
answer. You are given company names, filing kinds, and placeholders of the form
{{TOKEN}}. Reply with that one sentence and nothing else.

Rules. Use only the supplied companies and filing kinds. Copy any placeholder
exactly as given, braces included. Never write a digit. Never write a citation
marker such as [1]. Never state, compare, rank, estimate or predict any amount,
change or outcome. Do not answer the question. Do not add a heading, a list, a
quotation mark or a Markdown fence.

Write only that the answer below presents the disclosure evidence found for the
named companies."""

#: A lead is framing, not content.  Anything longer has started summarising.
MAX_LEAD_CHARS = 120
#: Enough companies and filing kinds to frame an answer; past this the sentence
#: stops being an opening line.
MAX_LEAD_COMPANIES = 5
MAX_LEAD_REPORTS = 4

STATUS_SUCCESS = "success"
STATUS_DISABLED = "disabled"
STATUS_NOT_CONFIGURED = "not_configured"
STATUS_NOT_ELIGIBLE = "not_eligible"
STATUS_TRANSPORT_FAILURE = "transport_failure"
STATUS_ERROR = "lead_error"
STATUS_EMPTY = "empty_reply"
STATUS_REJECTED = "rejected"

#: Wording that would turn framing into a claim.  The model is not shown any
#: figure, so a comparison or a forecast in its reply is invented by definition.
_BANNED = (
    "더 큰", "더 많", "더 적", "가장", "제일", "최대", "최소",
    "증가", "감소", "상승", "하락", "개선", "악화", "성장", "둔화",
    "전망", "예상", "추정", "예측", "권고", "추천", "유망", "우수",
    "따라서", "결론적으로", "즉,",
)
_DIGIT = re.compile(r"\d")
_PLACEHOLDER = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")
_CITATION = re.compile(r"\[\s*\d*\s*\]")


class LeadRejected(ValueError):
    """The reply failed a check, so it is discarded rather than repaired."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class LeadRequest:
    """What the model may see: names and filing kinds, never a figure."""

    companies: tuple[str, ...]
    reports: tuple[str, ...]
    #: Content words from the asker's own question.  Echoing the asker back is
    #: not a claim, and without them the lead cannot name what was looked for.
    topic: tuple[str, ...] = ()
    placeholders: Mapping[str, str] = field(default_factory=dict)

    @property
    def vocabulary(self) -> frozenset[str]:
        """Every name the reply is permitted to use."""

        return frozenset({*self.companies, *self.reports, *self.topic})

    def to_payload(self) -> dict[str, Any]:
        return {
            "companies": list(self.companies),
            "filing_kinds": list(self.reports),
            "subject": list(self.topic),
            "placeholders": sorted(self.placeholders),
        }


@dataclass(frozen=True)
class LeadOutcome:
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


#: Question words that are grammar rather than subject.  Removed so the topic
#: carries what was asked about and not how it was phrased.
_QUESTION_NOISE = frozenset(
    """
    중 더 큰 작은 가장 제일 어디 어느 무엇 누구 얼마 알려줘 알려 정리해 확인해
    비교해 인가 인가요 입니까 있나 있는지 해줘 좀 그리고 및 등 의 은 는 이 가
    """.split()
)


def question_topic(query: str) -> tuple[str, ...]:
    """Content words from the question, in order, without its interrogative half."""

    words = []
    for token in re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9]*", str(query or "")):
        if len(token) < 2 or token in _QUESTION_NOISE or token in words:
            continue
        words.append(token)
    return tuple(words[:6])


def lead_request(
    answer: str,
    *,
    period: str | None = None,
    topic: Sequence[str] = (),
) -> LeadRequest | None:
    """Read the companies and filing kinds out of an already-presented answer.

    Reads the headings ``answer_presentation`` produced, so the model is framing
    the answer that will actually be served rather than an earlier draft of it.
    Returns ``None`` when the answer has no such headings, which is how every
    refusal, clarification and single-fact answer opts out.
    """

    companies: list[str] = []
    reports: list[str] = []
    for line in str(answer or "").split("\n"):
        head = re.match(r"^\s*\d+\.\s+(.+?)\s+·\s+(.+?)\s*$", line)
        if head is None:
            continue
        company, report = head.groups()
        if company not in companies:
            companies.append(company)
        # "반기보고서 (2025.06)" names a kind and a period; only the kind is
        # framing, and the period is a figure the model must not receive.
        kind = re.sub(r"\s*\(.*?\)\s*$", "", report).strip()
        if kind and kind not in reports:
            reports.append(kind)
    if not companies:
        return None

    placeholders = {"{{PERIOD_1}}": period} if period else {}
    return LeadRequest(
        companies=tuple(companies[:MAX_LEAD_COMPANIES]),
        reports=tuple(reports[:MAX_LEAD_REPORTS]),
        topic=tuple(topic),
        placeholders={key: value for key, value in placeholders.items() if value},
    )


def accept_lead(reply: str, request: LeadRequest) -> str:
    """Return the restored sentence, or raise ``LeadRejected`` saying why not.

    Separated from the call so every check is testable without a transport, and
    so the rules read as one list rather than as branches around an HTTP result.
    """

    text = " ".join(str(reply or "").split()).strip().strip('"').strip("'")
    if not text:
        raise LeadRejected("empty")
    if "\n" in text or len(text) > MAX_LEAD_CHARS:
        raise LeadRejected("too_long")
    if _CITATION.search(text):
        raise LeadRejected("citation_marker")

    supplied = set(request.placeholders)
    for token in _PLACEHOLDER.findall(text):
        if token not in supplied:
            raise LeadRejected("unknown_placeholder")
    # Checked before restoration: after it the period's own digits are present
    # legitimately, and the rule "no digits" would no longer be decidable.
    if _DIGIT.search(_PLACEHOLDER.sub("", text)):
        raise LeadRejected("digit")
    for word in _BANNED:
        if word in text:
            raise LeadRejected("evaluative_wording")

    # Korean agglutinates, so "LG에너지솔루션과" is one token carrying a supplied
    # name.  Remove the supplied vocabulary first -- longest first, so a name
    # containing another is not half-erased -- and judge only what is left.
    residue = _PLACEHOLDER.sub(" ", text)
    for name in sorted(request.vocabulary, key=len, reverse=True):
        residue = residue.replace(name, " ")
    for token in re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9]*", residue):
        if not _is_ordinary_wording(token):
            raise LeadRejected("unsupplied_wording")

    for token, value in request.placeholders.items():
        text = text.replace(token, value)
    return text


#: Function words and framing nouns a lead sentence is made of.  A token outside
#: this set and outside the supplied names is treated as a name the model
#: invented, which is the failure this check exists to catch.
_ORDINARY = frozenset(
    """
    공시 근거 자료 내용 답변 결과 아래 다음 관련 대한 대해 확인된 확인 정리한
    제시한 제공된 기재된 보고서 보고 사항 현황 항목 기준 기간 부문 사업 회사 기업
    과 와 의 은 는 이 가 을 를 에 서 에서 로 으로 및 등 그 이하 각 해당 총 건 개
    년 월 분기 반기 두 세 네 여러 관하여 관한 따른 대하여 나온 담긴 실린
    입니다 있습니다 담았습니다 정리했습니다 나열했습니다 요약 목록 제시
    """.split()
)


def _is_ordinary_wording(token: str) -> bool:
    if token in _ORDINARY:
        return True
    # Korean agglutination means a framing noun arrives with particles attached.
    return any(
        token.startswith(word) and len(token) - len(word) <= 3
        for word in _ORDINARY
        if len(word) >= 2
    )


class AnswerLeadWriter:
    """Ask HCX for the opening line, or return no lead."""

    def __init__(
        self,
        settings: HcxSettings | None = None,
        *,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        self.settings = settings or HcxSettings.from_env()
        self.transport = transport or UrllibJsonTransport()
        self.call_count = 0

    def write(self, request: LeadRequest | None) -> LeadOutcome:
        start = perf_counter()
        if request is None:
            return LeadOutcome(None, STATUS_NOT_ELIGIBLE, _elapsed_ms(start))
        if not self.settings.enabled:
            return LeadOutcome(None, STATUS_DISABLED, _elapsed_ms(start))
        if not self.settings.configured:
            return LeadOutcome(None, STATUS_NOT_CONFIGURED, _elapsed_ms(start))

        self.call_count += 1
        try:
            response = self.transport.post_json(
                self.settings.endpoint,
                headers=self.settings.request_headers(),
                payload=self._payload(request),
                timeout_seconds=self.settings.timeout_seconds,
            )
        except (EmbeddingHttpError, TimeoutError):
            return LeadOutcome(None, STATUS_TRANSPORT_FAILURE, _elapsed_ms(start))
        except Exception:  # noqa: BLE001 - the answer is served without a lead
            return LeadOutcome(None, STATUS_ERROR, _elapsed_ms(start))

        content = _response_content(response) if isinstance(response, Mapping) else None
        if not content:
            return LeadOutcome(None, STATUS_EMPTY, _elapsed_ms(start))
        try:
            return LeadOutcome(
                accept_lead(content, request), STATUS_SUCCESS, _elapsed_ms(start)
            )
        except LeadRejected as rejected:
            return LeadOutcome(None, f"{STATUS_REJECTED}:{rejected.reason}", _elapsed_ms(start))

    def _payload(self, request: LeadRequest) -> dict[str, Any]:
        return {
            "model": self.settings.model,
            "messages": [
                {"role": "system", "content": LEAD_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        request.to_payload(), ensure_ascii=False, sort_keys=True
                    ),
                },
            ],
            "temperature": 0.0,
            "max_tokens": min(self.settings.max_tokens, 128),
        }


def with_lead(answer: str, lead: str | None) -> str:
    """Put the lead above the answer, or return the answer untouched."""

    body = str(answer or "")
    text = str(lead or "").strip()
    return f"{text}\n\n{body}" if text and body else body


def lead_period(plan: Any) -> str | None:
    """The year the question named, as the one value the lead may carry."""

    years: Sequence[Any] = tuple(getattr(plan, "years", ()) or ())
    if years:
        return str(years[0])
    period = getattr(plan, "period", None)
    year = getattr(period, "year", None)
    return str(year) if year is not None else None
