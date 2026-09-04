"""Warm the answer path so the first real request is not the slow one.

BGE-M3 loads lazily, so the first question after a restart pays for reading the
model into memory on top of answering. Measured cold, that first call can take a
minute; every call after it takes ten to twenty seconds. An evaluator whose
client times out at thirty seconds would see a failure that the service does not
actually have.

Run from ``ExecStartPost`` so a restart pays that cost itself, before anyone
asks. It waits for the port to answer, sends one ordinary question, and exits
zero no matter what happened -- a warm-up that failed must never keep the
service from starting.

    python scripts/warmup_api.py
    python scripts/warmup_api.py --port 8000 --wait-seconds 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen


#: Ordinary enough to exercise the whole path -- understanding, retrieval,
#: reasoning, generation -- without depending on any one company's filings.
WARMUP_QUESTION = "삼성전자의 2025년 매출액은 얼마인가?"


def _healthy(base: str, deadline: float) -> bool:
    """Wait for the port to answer, rather than assuming it already does."""

    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base}/healthz", timeout=5) as response:
                if json.loads(response.read()).get("status") == "ok":
                    return True
        except (URLError, ValueError, OSError, json.JSONDecodeError):
            pass
        time.sleep(2)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Warm the answer path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("FESTIVAL_API_PORT", "8000"))
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=120.0,
        help="how long to wait for the port before giving up",
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--question", default=WARMUP_QUESTION)
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    started = time.monotonic()
    if not _healthy(base, started + args.wait_seconds):
        print(f"warmup: {base} did not become healthy; skipping", file=sys.stderr)
        return 0

    query = urlencode({"question_id": "WARMUP", "question": args.question})
    try:
        with urlopen(f"{base}/answer?{query}", timeout=args.timeout_seconds):
            pass
    except Exception as error:  # noqa: BLE001 - warm-up never blocks startup
        print(f"warmup: request failed ({type(error).__name__}); skipping",
              file=sys.stderr)
        return 0

    print(f"warmup: answer path ready in {time.monotonic() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
