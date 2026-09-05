"""Show what HCX actually replied for a lead, and which rule judged it.

``think_trace`` reports the refusing rule but never the reply, because a refused
sentence is not evidence of anything and must not be persisted. Tuning the
allowed vocabulary still needs to see it once, so this asks for the same lead the
pipeline asks for and prints the raw reply beside the verdict.

    python scripts/diagnose_answer_lead.py --port 8000
    python scripts/diagnose_answer_lead.py --question "..." --repeat 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.generation.answer_lead import (
    AnswerLeadWriter,
    LeadRejected,
    accept_lead,
    lead_request,
    question_topic,
)
from app.generation.hcx_verbalizer import _response_content


DEFAULT_QUESTIONS = (
    "LG에너지솔루션과 삼성SDI 중 2025년 설비투자 규모가 더 큰 기업은 어디인가?",
    "삼성전자의 2025년 연결기준 매출액은 얼마인가?",
    "카카오가 2025년에 실시한 자금조달 내역을 유형별로 정리해줘",
)


def _served(base: str, question: str) -> Mapping[str, Any]:
    query = urllib.parse.urlencode({"question_id": "LEAD", "question": question})
    with urllib.request.urlopen(f"{base}/answer?{query}", timeout=300) as response:
        return json.load(response)


def _raw_reply(writer: AnswerLeadWriter, request: Any) -> str | None:
    response = writer.transport.post_json(
        writer.settings.endpoint,
        headers=writer.settings.request_headers(),
        payload=writer._payload(request),  # noqa: SLF001 - diagnostic
        timeout_seconds=writer.settings.timeout_seconds,
    )
    return _response_content(response) if isinstance(response, Mapping) else None


def _residue(reply: str, request: Any) -> list[str]:
    """The tokens the vocabulary check would judge, so the gap is visible."""

    text = re.sub(r"\{\{[A-Z][A-Z0-9_]*\}\}", " ", reply)
    for name in sorted(request.vocabulary, key=len, reverse=True):
        text = text.replace(name, " ")
    return re.findall(r"[가-힣A-Za-z][가-힣A-Za-z0-9]*", text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose the answer lead.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--question", action="append", default=None)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    writer = AnswerLeadWriter()
    if not writer.settings.configured:
        print("HCX is not configured; nothing to ask.", file=sys.stderr)
        return 1

    for question in (args.question or list(DEFAULT_QUESTIONS)):
        print("=" * 78)
        print(f"Q  {question}")
        payload = _served(base, question)
        request = lead_request(
            payload["answer"],
            period=None,
            topic=question_topic(question),
        )
        if request is None:
            print("   not eligible: the answer has no evidence headings")
            continue
        print(f"   companies {list(request.companies)}")
        print(f"   reports   {list(request.reports)}")
        print(f"   topic     {list(request.topic)}")
        for attempt in range(1, args.repeat + 1):
            reply = _raw_reply(writer, request)
            print(f"\n   [{attempt}] reply: {reply!r}")
            if not reply:
                continue
            try:
                print(f"       ACCEPTED: {accept_lead(reply, request)}")
            except LeadRejected as rejected:
                print(f"       REJECTED: {rejected.reason}")
                unknown = [
                    token
                    for token in _residue(reply, request)
                    if rejected.reason == "unsupplied_wording"
                ]
                if unknown:
                    print(f"       tokens judged: {unknown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
