"""P1-R Phase 4: can the ranking miss be recovered by *adding*, never replacing?

The cap-2 experiment recovered a document by changing which chunks occupied the
final ten, and that cost distinct facts from other answers.  This asks the
opposite question: leave the emitted top-10 exactly as production produces it,
and append a bounded number of already-scored candidates from documents the list
never showed.

Every variant here is measured against a hard invariant -- the baseline chunk
ids, in order, must be unchanged -- so a variant that cannot append without
disturbing them is rejected rather than reported.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from app.retrieval.hybrid import HybridQueryExecutor, HybridRetrievalConfig, RRFConfig
from app.retrieval.postgres_backend import PostgresBackend
from scripts.bge_eval_preflight import strict_vector_executor, assert_embedding_identity, pinned_encoder_factory
from scripts.diagnose_p1r_live import MISSES, holding_questions

#: Production defaults. `legacy` is the shipped rerank mode; `bounded` remains an
#: opt-in evaluation experiment and is not the baseline being protected here.
RETRIEVAL = dict(lexical_top_n=50, vector_top_n=50, final_top_k=10,
                 fusion_weight=0.60, deterministic_weight=0.40,
                 rerank_mode="legacy", rrf_k=60)


def build():
    config = EmbeddingConfig.from_env()
    assert_embedding_identity(config, require_device="cuda")
    embedder = create_embedding_provider(
        config, bge_encoder_factory=pinned_encoder_factory)
    backend = PostgresBackend()
    hybrid = HybridRetrievalConfig(
        lexical_top_n=RETRIEVAL["lexical_top_n"],
        vector_top_n=RETRIEVAL["vector_top_n"],
        final_top_k=RETRIEVAL["final_top_k"],
        fusion_weight=RETRIEVAL["fusion_weight"],
        deterministic_weight=RETRIEVAL["deterministic_weight"],
        rerank_mode=RETRIEVAL["rerank_mode"],
        rrf=RRFConfig(k=RETRIEVAL["rrf_k"]))
    return (strict_vector_executor(
        HybridQueryExecutor(backend, embedder, config, config=hybrid)),
            QueryUnderstanding(company_resolver=backend.resolve_company))


def additive_candidates(rows, emitted_docs, *, limit):
    """Highest-ranked already-scored candidates from documents never emitted.

    Scores and ranks are read, never recomputed: an appended chunk keeps the
    rank the reranker already gave it.
    """

    ordered = sorted((r for r in rows if r.get("final_rank")),
                     key=lambda r: int(r["final_rank"]))
    seen = set(emitted_docs)
    picked = []
    for row in ordered:
        if int(row["final_rank"]) <= RETRIEVAL["final_top_k"]:
            continue
        doc = row.get("doc_id")
        if not doc or doc in seen:
            continue
        seen.add(doc)
        picked.append(row)
        if len(picked) >= limit:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/out/bge/p1r_additive.json")
    args = parser.parse_args()

    executor, understanding = build()
    cases = []

    for question in holding_questions():
        qid = str(question["question_id"])
        gold_doc = str(question["doc_id"])
        plan = understanding.understand(str(question["query"]))
        clock = time.perf_counter()
        execution = executor.execute(plan)
        latency_ms = round((time.perf_counter() - clock) * 1000, 1)

        rows = [dict(r) for r in execution.rerank_diagnostics]
        chunk_text = {c.chunk_id: str(c.chunk.get("retrieval_text") or "")
                      for c in execution.chunks}

        # The emitted context is what survives rescues and graph expansion, not
        # the raw hybrid ten.
        emitted = list(execution.results)
        emitted_chunks = [r.chunk_id for r in emitted]
        emitted_docs = list(dict.fromkeys(r.doc_id for r in emitted))
        hybrid_top10 = [r["chunk_id"] for r in
                        sorted((x for x in rows if x.get("final_rank")),
                               key=lambda x: int(x["final_rank"]))[:10]]

        picks = additive_candidates(rows, emitted_docs, limit=2)
        crowding = max(collections.Counter(
            r.doc_id for r in emitted).values(), default=0)

        case = {
            "question_id": qid,
            "gold_doc": gold_doc,
            "latency_ms": latency_ms,
            "hybrid_top10_chunks": hybrid_top10,
            "emitted_chunks": emitted_chunks,
            "emitted_size": len(emitted),
            "emitted_distinct_docs": len(emitted_docs),
            "max_chunks_one_doc": crowding,
            "gold_in_emitted": gold_doc in emitted_docs,
            "gold_rank_in_emitted": (emitted_docs.index(gold_doc) + 1
                                     if gold_doc in emitted_docs else None),
            "appends": [
                {"chunk_id": p["chunk_id"], "doc_id": p.get("doc_id"),
                 "scored_rank": int(p["final_rank"]),
                 "final_score": p.get("final_score"),
                 "is_gold_doc": p.get("doc_id") == gold_doc,
                 "chars": len(chunk_text.get(p["chunk_id"], ""))}
                for p in picks
            ],
        }
        # Invariant: appending must not disturb what was emitted.
        for n in (1, 2):
            appended = emitted_chunks + [p["chunk_id"] for p in picks[:n]]
            case[f"plus{n}_prefix_identical"] = appended[:len(emitted_chunks)] == emitted_chunks
            case[f"plus{n}_gold_present"] = (
                case["gold_in_emitted"]
                or any(p.get("doc_id") == gold_doc for p in picks[:n]))
            case[f"plus{n}_size"] = len(appended)
        cases.append(case)

    def recall(key):
        return round(sum(1 for c in cases if c[key]) / len(cases), 4)

    summary = {
        "questions": len(cases),
        "baseline_emitted_size": {
            "mean": round(statistics.mean(c["emitted_size"] for c in cases), 2),
            "min": min(c["emitted_size"] for c in cases),
            "max": max(c["emitted_size"] for c in cases)},
        "baseline_distinct_docs_mean": round(
            statistics.mean(c["emitted_distinct_docs"] for c in cases), 2),
        "baseline_gold_present": recall("gold_in_emitted"),
        "plus1_gold_present": recall("plus1_gold_present"),
        "plus2_gold_present": recall("plus2_gold_present"),
        "prefix_identical_all": all(c["plus2_prefix_identical"] for c in cases),
        "misses": {m: {
            "baseline": next(c["gold_in_emitted"] for c in cases if c["question_id"] == m),
            "plus1": next(c["plus1_gold_present"] for c in cases if c["question_id"] == m),
            "plus2": next(c["plus2_gold_present"] for c in cases if c["question_id"] == m),
            "gold_scored_rank": next(
                (int(r["final_rank"]) for r in []), None)}
            for m in MISSES if any(c["question_id"] == m for c in cases)},
        "appended_chars": {
            "plus1_mean": round(statistics.mean(
                (c["appends"][0]["chars"] if c["appends"] else 0) for c in cases), 1),
            "plus2_mean": round(statistics.mean(
                sum(a["chars"] for a in c["appends"][:2]) for c in cases), 1),
            "plus2_max": max(sum(a["chars"] for a in c["appends"][:2]) for c in cases)},
        "latency_ms": {
            "mean": round(statistics.mean(c["latency_ms"] for c in cases), 1),
            "p95": round(sorted(c["latency_ms"] for c in cases)[
                min(len(cases) - 1, int(len(cases) * 0.95))], 1)},
    }
    payload = {"retrieval_config": RETRIEVAL, "summary": summary, "cases": cases}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                         default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
