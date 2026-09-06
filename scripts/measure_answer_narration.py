"""Measure how often HyperCLOVA X's rewrite survives its checks, on a live server.

The narration path is fail-closed: a refused rewrite serves the deterministic
answer, so a low success rate costs nothing but gains nothing either. That
number is not knowable from unit tests, because it depends on whether the live
model keeps every masked token in order. This asks the served endpoint and
reports the distribution, so the decision to keep the rewrite on is made on a
measurement rather than on a hope.

Run it against the submission endpoint after deploying:

    python scripts/measure_answer_narration.py --base-url http://HOST:PORT

``--show`` prints the first answer of each status, which is how a
``rejected:...`` status gets diagnosed: the reason names the rule that refused
the reply.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter


#: The six task shapes the organiser published as reference questions, plus the
#: two that exercise the refusal paths. Company names are corpus issuers.
QUESTIONS: tuple[tuple[str, str], ...] = (
    ("N01", "삼성전자의 2025년 연결기준 매출액은 얼마인가?"),
    ("N02", "SK하이닉스의 2025년 연결기준 영업이익은 얼마인가?"),
    ("N03", "LG에너지솔루션의 2026년 1분기 분기보고서를 기준으로 주요 투자 계획을 정리해줘"),
    ("N04", "LG에너지솔루션과 삼성SDI 중 2025년 설비투자 규모가 더 큰 기업은 어디인가?"),
    ("N05", "현대자동차가 2025년에 실시한 자금조달 내역을 유형별로 정리해줘"),
    ("N06", "현대자동차가 2025년에 체결한 주요 계약 이후 해지된 계약이 존재하는가?"),
    ("N07", "SK하이닉스의 2023년 사업보고서와 2025년 사업보고서를 비교했을 때 핵심 사업은 어떻게 변화했는지 설명해줘"),
    ("N08", "삼성전자 2027년 매출 전망은 얼마야?"),
)

#: ``think_trace`` is a string on the wire, one ``key: value`` per line.
_TRACE_LINE = re.compile(r"^([a-z_]+):\s*(.*)$")


def trace_value(think_trace: object, key: str) -> str:
    """Read one value out of the rendered trace, whichever form it arrives in."""

    if isinstance(think_trace, dict):
        value = think_trace.get(key)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return "" if value is None else str(value)
    for line in str(think_trace or "").split("\n"):
        match = _TRACE_LINE.match(line)
        if match is not None and match.group(1) == key:
            return match.group(2)
    return ""


def synthesis_status(think_trace: object) -> str:
    """``answer_synthesis`` is a JSON object on one line; read its status."""

    raw = trace_value(think_trace, "answer_synthesis")
    if not raw:
        return "-"
    try:
        return str(json.loads(raw).get("status") or "-")
    except (ValueError, AttributeError):
        return "unparsed"


def narration_status(think_trace: object) -> str:
    """``answer_narration`` is a JSON object on one line; read its status."""

    raw = trace_value(think_trace, "answer_narration")
    if not raw:
        return "absent"
    try:
        return str(json.loads(raw).get("status") or "absent")
    except (ValueError, AttributeError):
        return "unparsed"


def ask(base_url: str, question_id: str, question: str, timeout: float) -> dict:
    url = base_url.rstrip("/") + "/answer?" + urllib.parse.urlencode(
        {"question_id": question_id, "question": question}
    )
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--show", action="store_true", help="print the first answer of each status"
    )
    arguments = parser.parse_args(argv)

    statuses: Counter[str] = Counter()
    shown: set[str] = set()
    for question_id, question in QUESTIONS:
        try:
            payload = ask(arguments.base_url, question_id, question, arguments.timeout)
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            statuses["request_failed"] += 1
            print(f"{question_id}  request_failed  {error}")
            continue

        trace = payload.get("think_trace")
        status = narration_status(trace)
        statuses[status] += 1
        print(
            f"{question_id}  synthesis={synthesis_status(trace):<26}"
            f"narration={status:<28}"
        )
        if arguments.show and status not in shown:
            shown.add(status)
            print("-" * 70)
            print(str(payload.get("answer") or "")[:900])
            print("-" * 70)

    print("\n=== narration status ===")
    for status, count in statuses.most_common():
        print(f"  {count:>3}  {status}")
    applied = statuses.get("success", 0)
    print(f"\napplied {applied}/{len(QUESTIONS)}")
    print(
        "A status other than success means the deterministic answer was served,"
        " which is the behaviour without this layer at all."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
