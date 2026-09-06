"""Ask the served endpoint the two-company comparison set, the way the judges will.

Thirty questions, each one a comparison: two issuers, one figure, and a "which
is larger" that the answer has to actually settle. They are sent one at a time
over ``GET /answer``, exactly as the organiser's notice describes, and every
response is saved verbatim -- the five string fields, nothing added, nothing
reformatted -- so what is on disk is what a judge would collect.

    python scripts/ask_comparison_set.py --base-url http://101.79.20.171:8000

Each response is written to ``--out-dir`` as ``<question_id>.json``, and a
summary line per question reports the route, whether the answer was supported,
and whether HyperCLOVA X's rewrite survived its checks. ``--only`` runs a subset
by id while iterating.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


#: (id, question). Ordered by sector so a run reads like the corpus does.
QUESTIONS: tuple[tuple[str, str], ...] = (
    ("C01", "삼성전자와 SK하이닉스 2024 사업보고서 연결 매출액·영업이익을 비교해, 어느 쪽이 더 크고 매출·영업이익 차이는?"),
    ("C02", "LG이노텍과 삼성전기 2024 사업보고서 연결 영업이익률(영업이익÷매출액)을 비교하면 어느 쪽이 더 높아?"),
    ("C03", "한미반도체와 LG이노텍 2024년 exchange 공시 건수와 계약·시설투자 금액 합계를 비교해 어느 쪽이 더 많아?"),
    ("C04", "현대자동차와 기아 2024 사업보고서 연결 매출액 전기 대비 증감률을 비교하면 어느 브랜드가 더 높아?"),
    ("C05", "현대모비스와 현대오토에버 2024 사업보고서 연결 영업이익 중 더 큰 쪽은?"),
    ("C06", "레인보우로보틱스와 두산로보틱스 2024 사업보고서 연결 매출액·영업이익을 비교해 어느 쪽이 더 크고 차이는?"),
    ("C07", "LG에너지솔루션과 삼성SDI 2024년 신규시설투자 공시 투자금액 합계를 비교하면 어느 쪽이 더 커?"),
    ("C08", "LG에너지솔루션과 에코프로비엠 2024 사업보고서 연결 당기순이익 부호를 비교해, 어느 쪽이 흑자·적자인지?"),
    ("C09", "POSCO홀딩스와 현대제철 2024 사업보고서 연결 부채비율(부채총계÷자본총계)을 비교하면 어느 쪽이 더 높아?"),
    ("C10", "한화솔루션과 OCI홀딩스 2024 사업보고서 태양광(또는 Qcells) segment 매출 전기 대비 증감률을 비교하면 어느 쪽이 더 높아?"),
    ("C11", "KB금융과 신한지주 2024 사업보고서 연결 당기순이익을 비교해 어느 금융지주가 더 크고 차이는?"),
    ("C12", "삼성생명과 삼성화재 2024 사업보고서 보험료수익(또는 영업수익) 전기 대비 증감률을 비교하면 어느 쪽이 더 높아?"),
    ("C13", "NAVER와 카카오 2023·2024 사업보고서 연결 매출액 증가율을 비교하면 어느 쪽이 더 높아?"),
    ("C14", "하이브와 에스엠 2024 사업보고서 연결 매출액·영업이익을 비교해 어느 엔터사가 더 크고 차이는?"),
    ("C15", "SK텔레콤과 KT 2024 사업보고서 연결 영업이익률(영업이익÷매출)을 비교하면 어느 통신사가 더 높아?"),
    ("C16", "LG유플러스와 SK텔레콤 2024년 1·2·3분기 연결 매출액 추이에서 최고 분기 매출을 비교하면 어느 쪽이 더 커?"),
    ("C17", "크래프톤과 엔씨소프트 2024 사업보고서 연결 영업이익·당기순이익을 비교해 어느 게임사가 더 수익성이 좋아?"),
    ("C18", "크래프톤과 시프트업 2024년 4분기 영업이익 합계와 각 사 2024 사업보고서 영업이익을 나란히 비교해 차이는?"),
    ("C19", "삼성바이오로직스와 셀트리온 2024 사업보고서 연결 매출액·영업이익 중 어느 CMO·바이오사가 더 크고 차이는?"),
    ("C20", "삼성바이오로직스와 셀트리온 2024년 exchange 공급계약 금액 합계를 비교하면 어느 쪽이 더 많아?"),
    ("C21", "알테오젠과 한미약품 2024 사업보고서 연결 영업이익 전기 대비 증감률을 비교하면 어느 제약사가 더 높아?"),
    ("C22", "HD현대중공업과 한화오션 2024 사업보고서 수주잔고(또는 수주실적) 전기 대비 증감률을 비교하면 어느 조선사가 더 높아?"),
    ("C23", "HD현대중공업과 삼성중공업 2024년 exchange 공급·수주 관련 계약금액 합계를 비교하면 어느 쪽이 더 커?"),
    ("C24", "두산에너빌리티와 한전기술 2024·2025년 원전 관련 exchange 공급계약 건수와 최대 계약금액을 비교하면 어느 쪽이 더 많아?"),
    ("C25", "한화에어로스페이스와 LIG넥스원 2024 사업보고서 연결 영업이익·매출액 증가율 중 어느 방산사가 더 높아?"),
    ("C26", "현대로템과 한국항공우주 2024년 exchange 공급계약 수량·금액 중 더 큰 건은 어느 쪽이야?"),
    ("C27", "아모레퍼시픽과 LG생활건강 2024 사업보고서 국내 vs 해외 매출 비중(%)을 비교하면 해외 비중이 더 높은 쪽은?"),
    ("C28", "현대건설과 삼성E&A 2024년 exchange 공급계약 건수와 건당 평균 계약금액을 비교하면 어느 건설사가 더 커?"),
    ("C29", "HMM과 현대글로비스 2023·2024 사업보고서 연결 당기순이익을 비교해, 2024년 흑자 전환 여부가 어느 쪽인지?"),
    ("C30", "LS ELECTRIC과 HD현대일렉트릭 2024년 신규시설투자 투자금액과 자기자본 대비 비율을 비교하면 어느 전력기기사가 더 커?"),
)

#: ``think_trace`` is a string on the wire, one ``key: value`` per line. It was
#: a mapping before the string contract, so both forms are read.
_TRACE_LINE = re.compile(r"^([a-z_]+):\s*(.*)$")


def trace_value(think_trace: object, key: str) -> str:
    if isinstance(think_trace, dict):
        value = think_trace.get(key)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            # The rendered form writes JSON's spelling, so the mapping form has
            # to as well: str(True) is "True" and would never compare equal.
            return "true" if value else "false"
        return "" if value is None else str(value)
    for line in str(think_trace or "").split("\n"):
        match = _TRACE_LINE.match(line)
        if match is not None and match.group(1) == key:
            return match.group(2)
    return ""


def narration_status(think_trace: object) -> str:
    raw = trace_value(think_trace, "answer_narration")
    if not raw:
        return "-"
    try:
        return str(json.loads(raw).get("status") or "-")
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
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--out-dir", default="reports/comparison_set")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--only", default="", help="comma-separated ids, e.g. C01,C04"
    )
    parser.add_argument(
        "--show", action="store_true", help="print each answer as it arrives"
    )
    arguments = parser.parse_args(argv)

    wanted = {part.strip().upper() for part in arguments.only.split(",") if part.strip()}
    selected = [row for row in QUESTIONS if not wanted or row[0] in wanted]

    out_dir = Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    routes: Counter[str] = Counter()
    narrations: Counter[str] = Counter()
    answerable = 0
    failures = 0

    for question_id, question in selected:
        start = time.monotonic()
        try:
            payload = ask(
                arguments.base_url, question_id, question, arguments.timeout
            )
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            failures += 1
            print(f"{question_id}  REQUEST FAILED  {error}")
            continue
        elapsed = time.monotonic() - start

        (out_dir / f"{question_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        trace = payload.get("think_trace")
        route = trace_value(trace, "route") or "-"
        supported = trace_value(trace, "answerable") == "true"
        routes[route] += 1
        narrations[narration_status(trace)] += 1
        answerable += int(supported)

        print(
            f"{question_id}  {elapsed:5.1f}s  route={route:<26}"
            f"answerable={str(supported):<5} narration={narration_status(trace)}"
        )
        if arguments.show:
            print("-" * 74)
            print(str(payload.get("answer") or ""))
            print("-" * 74)

    total = len(selected)
    print(f"\n=== {total} questions, saved to {out_dir} ===")
    print(f"answerable   {answerable}/{total}")
    if failures:
        print(f"failed       {failures}")
    print("\nroute:")
    for route, count in routes.most_common():
        print(f"  {count:>3}  {route}")
    print("\nanswer_narration:")
    for status, count in narrations.most_common():
        print(f"  {count:>3}  {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
