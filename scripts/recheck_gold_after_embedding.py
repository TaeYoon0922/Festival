"""Re-run the gold set against the live API now that every chunk is embedded.

The deferred and open items in FREEZE_LOG were each decided on evidence
gathered when vector coverage was a fraction of the corpus. Coverage is now
complete, so the evidence those decisions rest on is stale -- not necessarily
wrong, but no longer the state of the system. Re-reading the reasoning cannot
tell which; only asking the questions again can.

So this asks all sixty, through the served endpoint rather than through an
in-process harness, and reports for each whether the gold document was
retrieved, at what rank, and whether the answer came back supported.

    python scripts/recheck_gold_after_embedding.py --port 8000
    python scripts/recheck_gold_after_embedding.py --group holding
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.parsing.final_validation import (
    GOLD_QUESTIONS,
    HOLDING_ADDITIONAL_QUESTIONS,
)


def _ask(base: str, question_id: str, query: str) -> Mapping[str, Any]:
    url = f"{base}/answer?" + urllib.parse.urlencode(
        {"question_id": question_id, "question": query}
    )
    with urllib.request.urlopen(url, timeout=300) as response:
        return json.load(response)


def _gold_rank(payload: Mapping[str, Any], doc_id: str) -> int | None:
    for row in payload.get("retrieved_context") or ():
        if str(row.get("doc_id")) == doc_id:
            return int(row.get("rank"))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-check the gold set.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--group", default=None, help="only this doc_group")
    parser.add_argument("--out", default=None, help="write the rows as JSON here")
    args = parser.parse_args(argv)

    base = f"http://{args.host}:{args.port}"
    questions = [*GOLD_QUESTIONS, *HOLDING_ADDITIONAL_QUESTIONS]
    if args.group:
        questions = [q for q in questions if q.get("doc_group") == args.group]

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    print(f"{'ID':6} {'group':9} {'rank':>5} {'answerable':>10}  route")
    print("-" * 78)
    for question in questions:
        question_id = str(question.get("question_id"))
        try:
            payload = _ask(base, question_id, str(question.get("query")))
        except Exception as error:  # noqa: BLE001 - one failure must not stop the run
            print(f"{question_id:6} {'':9} {'':>5} {'ERROR':>10}  {type(error).__name__}")
            rows.append({"question_id": question_id, "error": type(error).__name__})
            continue
        trace = payload.get("think_trace") or {}
        rank = _gold_rank(payload, str(question.get("doc_id")))
        row = {
            "question_id": question_id,
            "doc_group": question.get("doc_group"),
            "query": question.get("query"),
            "gold_doc_id": question.get("doc_id"),
            "gold_rank": rank,
            "answerable": bool(trace.get("answerable")),
            "route": trace.get("route"),
            "task_type": trace.get("task_type"),
            "hcx_status": trace.get("hcx_status"),
            "retrieval_count": trace.get("retrieval_count"),
        }
        rows.append(row)
        print(
            f"{question_id:6} {str(row['doc_group']):9} "
            f"{(rank if rank else '-'):>5} {str(row['answerable']):>10}  {row['route']}"
        )

    print()
    print("=" * 78)
    served = [row for row in rows if "error" not in row]
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in served:
        by_group[str(row["doc_group"])].append(row)

    print(f"{'group':10} {'n':>3} {'hit@10':>7} {'hit@1':>6} {'answerable':>11}")
    print("-" * 42)
    for group in sorted(by_group):
        group_rows = by_group[group]
        total = len(group_rows)
        hit10 = sum(1 for row in group_rows if row["gold_rank"])
        hit1 = sum(1 for row in group_rows if row["gold_rank"] == 1)
        answerable = sum(1 for row in group_rows if row["answerable"])
        print(
            f"{group:10} {total:>3} {hit10 / total:>7.2f} "
            f"{hit1 / total:>6.2f} {answerable / total:>11.2f}"
        )
    total = len(served)
    if total:
        print("-" * 42)
        print(
            f"{'전체':10} {total:>3} "
            f"{sum(1 for r in served if r['gold_rank']) / total:>7.2f} "
            f"{sum(1 for r in served if r['gold_rank'] == 1) / total:>6.2f} "
            f"{sum(1 for r in served if r['answerable']) / total:>11.2f}"
        )

    missed = [row for row in served if not row["gold_rank"]]
    if missed:
        print()
        print("gold 문서를 못 물어온 질문")
        for row in missed:
            print(f"  {row['question_id']:6} {row['doc_group']:9} {row['query']}")

    unsupported = [row for row in served if row["gold_rank"] and not row["answerable"]]
    if unsupported:
        print()
        print("gold 문서는 왔는데 답하지 못한 질문")
        for row in unsupported:
            print(
                f"  {row['question_id']:6} rank {row['gold_rank']:2}  "
                f"{row['route']:20} {row['query']}"
            )

    print()
    print(f"경로 분포: {dict(Counter(str(row['route']) for row in served))}")
    print(f"소요 {time.monotonic() - started:.0f}s")
    if args.out:
        Path(args.out).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
