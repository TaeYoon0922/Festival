"""Ask one question against a running Festival API (GET /answer).

Prints think_trace first, then retrieval ranks, then the answer.
Saves the full JSON response for later inspection.

    python scripts/ask_one_question.py
    python scripts/ask_one_question.py --port 8010
    python scripts/ask_one_question.py --question "..." --question-id BG01

Does not start the API, does not print credentials, and does not change
the /answer contract.  Requires a host API already listening on --port.
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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTION_ID = "BG01"
DEFAULT_QUESTION = (
    "고려아연이 최근 공시한 신규시설투자 금액은 자기자본 대비 어느 정도 수준이며, "
    "현재 보유 중인 현금성 자산으로 자체 조달이 가능한가요?"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ask_one_question"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Call GET /answer and print think_trace before the answer."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--question-id", default=DEFAULT_QUESTION_ID)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="Retrieval plus generation can take more than a few seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    health = _get_json(f"{base}/healthz", timeout_seconds=5.0)
    if not isinstance(health, dict) or health.get("status") != "ok":
        print(
            f"healthz failed on {base}. Start the API first, for example:\n"
            f"  FESTIVAL_API_PORT={args.port} FESTIVAL_HCX_ENABLED=false "
            f"python -m app.api",
            file=sys.stderr,
        )
        return 2

    url = f"{base}/answer?" + urlencode(
        {"question_id": args.question_id, "question": args.question}
    )
    payload = _get_json(url, timeout_seconds=args.timeout_seconds)
    if payload is None:
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{args.question_id}_{stamp}.json"
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    _print_summary(payload, output_path)
    return 0


def _get_json(url: str, *, timeout_seconds: float) -> dict | None:
    try:
        with urlopen(url, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP {exc.code} from {url.split('?', 1)[0]}", file=sys.stderr)
        print(_safe_error_body(body), file=sys.stderr)
        return None
    except URLError as exc:
        print(f"request failed: {exc.reason}", file=sys.stderr)
        return None
    except TimeoutError:
        print(f"timed out calling {url.split('?', 1)[0]}", file=sys.stderr)
        return None

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        print(f"HTTP {status}: response was not JSON", file=sys.stderr)
        print(body[:500], file=sys.stderr)
        return None
    if not isinstance(parsed, dict):
        print("response JSON was not an object", file=sys.stderr)
        return None
    return parsed


def _safe_error_body(body: str) -> str:
    lowered = body.casefold()
    if any(token in lowered for token in ("password", "postgres://", "bearer ", "api_key")):
        return "(error body omitted because it may contain a secret)"
    return body[:1000]


def _print_summary(payload: dict, output_path: Path) -> None:
    trace = payload.get("think_trace")
    trace = trace if isinstance(trace, dict) else {}
    context = payload.get("retrieved_context")
    context = context if isinstance(context, list) else []

    print("=== think_trace ===")
    print(
        json.dumps(
            {
                "task_type": trace.get("task_type"),
                "route": trace.get("route"),
                "answerable": trace.get("answerable"),
                "warnings": trace.get("warnings"),
                "hcx_status": trace.get("hcx_status"),
                "retrieval_count": trace.get("retrieval_count"),
                "selected_evidence_count": trace.get("selected_evidence_count"),
                "stages": trace.get("stages"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print()
    print("=== retrieved_context (rank only) ===")
    if not context:
        print("(empty)")
    for row in context:
        if not isinstance(row, dict):
            continue
        print(
            json.dumps(
                {
                    "rank": row.get("rank"),
                    "corp_name": row.get("corp_name"),
                    "report_nm": row.get("report_nm"),
                    "rcept_dt": row.get("rcept_dt"),
                    "chunk_type": row.get("chunk_type"),
                    "chunk_id": row.get("chunk_id"),
                },
                ensure_ascii=False,
            )
        )
    print()
    print("=== answer ===")
    print(payload.get("answer") or "(empty)")
    print()
    print(f"full JSON: {output_path}")


if __name__ == "__main__":
    raise SystemExit(main())
