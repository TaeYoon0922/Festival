"""P1-R baseline under a real, pinned BGE-M3 -- never under hash.

Reuses the frozen ``QueryPlanHybridEvaluator`` and the production retrieval
defaults, so the only difference from a normal hybrid evaluation is that both
the corpus vectors and the query vectors come from the pinned BGE-M3 snapshot.
Results are labelled LIVE_BGE_M3 and are never merged with hash figures.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.parsing.final_validation import GOLD_QUESTIONS, HOLDING_ADDITIONAL_QUESTIONS
from app.reasoning.hybrid_evaluation import QueryPlanHybridEvaluator
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from app.retrieval.hybrid import (
    HybridQueryExecutor,
    HybridRetrievalConfig,
    RRFConfig,
)
from app.retrieval.postgres_backend import PostgresBackend
from scripts.bge_eval_preflight import (
    assert_embedding_identity,
    describe,
    pinned_encoder_factory,
)

#: Frozen production retrieval defaults. This run establishes a baseline; it
#: does not tune anything.
RETRIEVAL_CONFIG = {
    "lexical_top_n": 50, "vector_top_n": 50, "final_top_k": 10, "rrf_k": 60,
    "fusion_weight": 0.60, "deterministic_weight": 0.40,
    "rerank_mode": "bounded", "rerank_window_size": 2,
}

#: The six P0-D declines are not evaluable for ranking: retrieval never runs.
P0D_DECLINED = ("H01", "H02", "HX01", "HX02", "HX03", "HX04")


def _ndcg(rank: int | None, cutoff: int) -> float:
    """One relevant document per question, so IDCG is 1."""

    if not rank or rank > cutoff:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _aggregate(rows, field):
    ranks = [r.get(field) for r in rows]
    found = [r for r in ranks if isinstance(r, int) and r > 0]
    n = len(rows)
    return {
        "questions": n,
        **{f"recall_at_{k}": round(sum(1 for r in found if r <= k) / n, 4) if n else 0.0
           for k in (1, 3, 5, 10)},
        "mrr": round(sum(1.0 / r for r in found) / n, 4) if n else 0.0,
        "ndcg_at_5": round(sum(_ndcg(r, 5) for r in ranks) / n, 4) if n else 0.0,
        "ndcg_at_10": round(sum(_ndcg(r, 10) for r in ranks) / n, 4) if n else 0.0,
    }


def _git_commit() -> str:
    """The container has no git; the runner passes the commit in explicitly."""

    import os

    value = os.environ.get("FESTIVAL_EVAL_GIT_COMMIT")
    if value:
        return value
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/out/bge/p1r_live_bge_m3.json")
    parser.add_argument("--manifest", default="/out/bge/manifest.json")
    parser.add_argument("--rerank-mode", choices=("legacy", "bounded"), default=None,
                        help="defaults to the frozen RETRIEVAL_CONFIG value")
    parser.add_argument("--all-groups", action="store_true",
                        help="report every P0-D-accepted question, grouped by doc_group")
    args = parser.parse_args()

    if args.rerank_mode:
        RETRIEVAL_CONFIG["rerank_mode"] = args.rerank_mode
    config = EmbeddingConfig.from_env()
    assert_embedding_identity(config, require_device="cuda")
    embedder = create_embedding_provider(
        config, bge_encoder_factory=pinned_encoder_factory)

    backend = PostgresBackend()
    understanding = QueryUnderstanding(company_resolver=backend.resolve_company)
    hybrid_config = HybridRetrievalConfig(
        lexical_top_n=RETRIEVAL_CONFIG["lexical_top_n"],
        vector_top_n=RETRIEVAL_CONFIG["vector_top_n"],
        final_top_k=RETRIEVAL_CONFIG["final_top_k"],
        fusion_weight=RETRIEVAL_CONFIG["fusion_weight"],
        deterministic_weight=RETRIEVAL_CONFIG["deterministic_weight"],
        rerank_mode=RETRIEVAL_CONFIG["rerank_mode"],
        rerank_window_size=RETRIEVAL_CONFIG["rerank_window_size"],
        rrf=RRFConfig(k=RETRIEVAL_CONFIG["rrf_k"]),
    )
    executor = HybridQueryExecutor(backend, embedder, config, config=hybrid_config)
    evaluator = QueryPlanHybridEvaluator(understanding, executor, top_k=10)

    started = time.perf_counter()
    result = evaluator.evaluate({
        "gold_40": GOLD_QUESTIONS,
        "holding_20": HOLDING_ADDITIONAL_QUESTIONS,
    })
    wall = time.perf_counter() - started

    rows = result.get("questions") or result.get("rows") or []
    by_id = {str(r.get("question_id")): r for r in rows}

    # Per-question latency, measured separately so evaluation bookkeeping is not
    # counted as retrieval time.
    latencies: dict[str, float] = {}
    for question in [*HOLDING_ADDITIONAL_QUESTIONS, *GOLD_QUESTIONS]:
        qid = str(question["question_id"])
        if qid in P0D_DECLINED or by_id.get(qid, {}).get("doc_group") != "holding":
            continue
        plan = understanding.understand(str(question["query"]))
        clock = time.perf_counter()
        executor.execute(plan)
        latencies[qid] = round((time.perf_counter() - clock) * 1000, 1)

    holding = [r for r in rows
               if str(r.get("doc_group")) == "holding"
               and str(r.get("question_id")) not in P0D_DECLINED]
    for row in holding:
        row["retrieval_latency_ms"] = latencies.get(str(row["question_id"]))

    lat = [v for v in latencies.values() if v is not None]
    accepted = [r for r in rows if str(r.get("question_id")) not in P0D_DECLINED]
    by_group = {}
    for row in accepted:
        by_group.setdefault(str(row.get("doc_group")), []).append(row)

    payload = {
        "label": "LIVE_BGE_M3",
        "all_accepted": _aggregate(accepted, "hybrid_gold_rank") if args.all_groups else None,
        "by_doc_group": ({g: _aggregate(v, "hybrid_gold_rank")
                          for g, v in sorted(by_group.items())}
                         if args.all_groups else None),
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "corpus_snapshot": "structural_v2_1_full_4204",
            "question_set": "gold_40 + holding_20 (Gold60)",
            "db_identifier": "festival-verify:55433/festival_verify",
            "retrieval_config": RETRIEVAL_CONFIG,
            **describe(config),
        },
        "p1r_scope": {
            "evaluable_holding_questions": len(holding),
            "excluded_p0d_declines": list(P0D_DECLINED),
        },
        "aggregate_hybrid": _aggregate(holding, "hybrid_gold_rank"),
        "aggregate_lexical_same_run": _aggregate(holding, "lexical_gold_rank"),
        "aggregate_vector_same_run": _aggregate(holding, "vector_gold_rank"),
        "latency_ms": {
            "mean": round(statistics.mean(lat), 1) if lat else None,
            "p95": round(sorted(lat)[min(len(lat) - 1, int(len(lat) * 0.95))], 1) if lat else None,
            "max": max(lat) if lat else None,
        },
        "wall_seconds": round(wall, 1),
        "questions": holding,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "questions"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
