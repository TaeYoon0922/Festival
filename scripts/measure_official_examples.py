"""Measure the official sample-question types against a running Festival API.

The six cases mirror the task types in the competition brief (검색·정보추출 /
다중조회·비교연산 / 복합문서추론, each Closed and Open), rewritten to name
companies that exist in this corpus.  Read-only: it calls GET /answer and
reports what came back.  It does not change the contract or the pipeline.

    python scripts/measure_official_examples.py
    python scripts/measure_official_examples.py --port 8010
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "official_examples"

#: (id, 과제 유형, Closed/Open, 질문).  Companies verified present in
#: data/corpus/universe.csv.
CASES = (
    (
        "OF1", "검색 및 정보 추출", "Closed",
        "삼성전자의 2025년 연결기준 매출액은 얼마인가?",
    ),
    (
        "OF2", "검색 및 정보 추출", "Open",
        "LG에너지솔루션의 2026년 1분기 분기보고서를 기준으로 주요 투자 계획을 정리해줘",
    ),
    (
        "OF3", "다중 조회 및 비교·연산", "Closed",
        "LG에너지솔루션과 삼성SDI 중 2025년 설비투자 규모가 더 큰 기업은 어디인가?",
    ),
    (
        "OF4", "다중 조회 및 비교·연산", "Open",
        "카카오가 2025년에 실시한 자금조달 내역을 유형별(유상증자, CB, BW, EB)로 정리해줘",
    ),
    (
        "OF5", "복합 문서 추론", "Closed",
        "한화오션이 2025년에 체결한 주요 계약 이후 해지된 계약이 존재하는가?",
    ),
    (
        "OF6", "복합 문서 추론", "Open",
        "카카오의 2023년 사업보고서와 2025년 사업보고서를 비교했을 때 "
        "핵심 사업은 어떻게 변화했는지 설명해줘",
    ),
)


def _get_json(url: str, timeout_seconds: float):
    with urlopen(url, timeout=timeout_seconds) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _ask(base: str, question_id: str, question: str, timeout_seconds: float):
    query = urlencode({"question_id": question_id, "question": question})
    try:
        return _get_json(f"{base}/answer?{query}", timeout_seconds), None
    except HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        return None, f"HTTP {error.code}: {body[:300]}"
    except URLError as error:
        return None, f"URLError: {error.reason}"
    except Exception as error:  # noqa: BLE001 - reported, not raised
        return None, f"{type(error).__name__}: {error}"


def _summarize(payload) -> dict:
    trace = payload.get("think_trace") or {}
    answer = payload.get("answer") or ""
    context = payload.get("retrieved_context") or []
    corps = []
    for row in context:
        name = row.get("corp_name")
        if name and name not in corps:
            corps.append(name)
    return {
        "task_type": trace.get("task_type"),
        "route": trace.get("route"),
        "stages": trace.get("stages") or [],
        "answerable": trace.get("answerable"),
        "hcx_status": trace.get("hcx_status"),
        "warnings": trace.get("warnings") or [],
        "retrieval_count": trace.get("retrieval_count"),
        "selected_evidence_count": trace.get("selected_evidence_count"),
        "context_rows": len(context),
        "context_corps": corps,
        "answer_chars": len(answer),
        "answer": answer,
        "query_understanding": trace.get("query_understanding"),
        "clarification": trace.get("clarification"),
        "answerability": trace.get("answerability"),
        "multi_document_planner": trace.get("multi_document_planner"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the six official-style cases against GET /answer."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    try:
        health = _get_json(f"{base}/healthz", timeout_seconds=5.0)
    except Exception as error:  # noqa: BLE001
        print(f"healthz 실패 ({base}): {type(error).__name__}: {error}")
        print("먼저 API를 띄우십시오:  FESTIVAL_API_PORT=8010 python -m app.api")
        return 2
    if not isinstance(health, dict) or health.get("status") != "ok":
        print(f"healthz 비정상 응답: {health}")
        return 2

    results = []
    for question_id, task, mode, question in CASES:
        print(f"\n{'='*78}")
        print(f"{question_id} · {task} · {mode}")
        print(f"질문: {question}")
        print("-" * 78)
        payload, error = _ask(base, question_id, question, args.timeout_seconds)
        if error is not None:
            print(f"  실패: {error}")
            results.append(
                {"id": question_id, "task": task, "mode": mode,
                 "question": question, "error": error}
            )
            continue
        summary = _summarize(payload)
        print(f"  task_type   : {summary['task_type']}")
        print(f"  route       : {summary['route']}")
        print(f"  answerable  : {summary['answerable']}")
        print(f"  hcx_status  : {summary['hcx_status']}")
        print(f"  검색/근거   : {summary['retrieval_count']} / "
              f"{summary['selected_evidence_count']}  "
              f"(context {summary['context_rows']}행)")
        print(f"  등장 기업   : {', '.join(summary['context_corps']) or '(없음)'}")
        if summary["warnings"]:
            print(f"  warnings    : {summary['warnings']}")
        if summary["clarification"]:
            print(f"  clarification: {summary['clarification']}")
        print(f"  stages      : {' → '.join(summary['stages'])}")
        print(f"\n  [답변 {summary['answer_chars']}자]")
        for line in summary["answer"].splitlines():
            print(f"    {line}")
        results.append(
            {"id": question_id, "task": task, "mode": mode,
             "question": question, "payload": payload, "summary": summary}
        )

    print(f"\n{'='*78}")
    print("요약")
    print("=" * 78)
    print(f"{'ID':5s} {'유형':22s} {'모드':7s} {'task_type':18s} "
          f"{'answerable':11s} {'hcx':22s} {'근거':>4s}")
    print("-" * 78)
    for row in results:
        if "error" in row:
            print(f"{row['id']:5s} {row['task'][:21]:22s} {row['mode']:7s} "
                  f"ERROR: {row['error'][:40]}")
            continue
        s = row["summary"]
        print(f"{row['id']:5s} {row['task'][:21]:22s} {row['mode']:7s} "
              f"{str(s['task_type'])[:17]:18s} {str(s['answerable']):11s} "
              f"{str(s['hcx_status'])[:21]:22s} "
              f"{str(s['selected_evidence_count']):>4s}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output_dir / f"official_examples_{stamp}.json"
    out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n전체 응답 저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
