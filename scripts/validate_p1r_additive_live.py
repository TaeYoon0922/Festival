"""Live pinned BGE validation of the additive document recovery.

The single thing that must hold is the one the rejected cap broke: the emitted
results, before any rescue, are the same chunks in the same order as production
produces today.  Everything else -- what was appended, what the answer gained --
is only interesting once that holds.

Baseline is obtained by disabling the rescue through its own bound, not by
editing production: a limit of zero is the identity case.
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

import app.reasoning.answer_composer as composer
import app.retrieval.hybrid as hybrid
from scripts.bge_eval_preflight import pinned_encoder_factory, strict_vector_executor
from scripts.diagnose_p1r_live import holding_questions


def summarize_answer(result):
    resolution = getattr(result, "resolution", None)
    events = list(getattr(resolution, "events", None) or [])
    matching = [e for e in events if e.matches_query is True]
    rendered = composer._reported_holding_events(resolution) if resolution else ()
    draft = getattr(result, "answer_draft", None)
    ambiguity = getattr(draft, "ambiguity", None) or {}
    return {
        "matching_event_count": len(matching),
        "rendered_event_count": len(rendered),
        "dates": sorted(e.reference_date or "?" for e in matching),
        "facts": sorted(f"{e.reference_date}|{e.after_shares}|{e.after_ratio}"
                        for e in rendered),
        "citations": len(getattr(draft, "citations", ()) or ()),
        "selection_mode": ambiguity.get("selection_mode"),
        "semantic_unique": bool(ambiguity.get("semantic_unique")),
        "under_specified": bool(ambiguity.get("under_specified")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/out/bge/p1r_additive_live.json")
    args = parser.parse_args()

    import app.api.pipeline as pipeline_module

    original = pipeline_module.create_embedding_provider
    pipeline_module.create_embedding_provider = (
        lambda cfg, **kw: original(cfg, bge_encoder_factory=pinned_encoder_factory, **kw))
    try:
        from app.api.pipeline import AnswerPipeline

        pipeline = AnswerPipeline.from_env()
    finally:
        pipeline_module.create_embedding_provider = original

    # A real-vector validation must not be satisfiable by lexical fallback.
    executor = strict_vector_executor(pipeline.executor)
    understanding = pipeline
    orch = pipeline.orchestrator
    # The limit is a default argument bound at definition time, so rebinding the
    # module constant would not disable anything. Swap the function instead.
    shipped_rescue = hybrid._additive_document_rescue

    def disabled(results, tail, **kwargs):
        return list(results), {"attempted": False, "crowding_detected": False,
                               "max_chunks_from_document": 0, "appended": False,
                               "appended_chunk_id": None, "appended_doc_id": None,
                               "original_candidate_rank": None}

    cases, latencies = [], []
    for question in holding_questions():
        qid = str(question["question_id"])
        text = str(question["query"])
        gold_doc = str(question["doc_id"])
        plan, _validation = understanding._validated_understanding(text)

        # Baseline: identical code path with the rescue's own limit at zero.
        hybrid._additive_document_rescue = disabled
        base_exec = executor.execute(plan)
        base_ids = [r.chunk_id for r in base_exec.results]
        base_answer = summarize_answer(orch.run(text, plan, base_exec))

        hybrid._additive_document_rescue = shipped_rescue
        clock = time.perf_counter()
        new_exec = executor.execute(plan)
        latency = round((time.perf_counter() - clock) * 1000, 1)
        latencies.append(latency)
        new_ids = [r.chunk_id for r in new_exec.results]
        new_answer = summarize_answer(orch.run(text, plan, new_exec))

        trace = (new_exec.routing.get("hybrid") or {}).get(
            "additive_document_recovery") or {}
        base_docs = list(dict.fromkeys(r.doc_id for r in base_exec.results))
        new_docs = list(dict.fromkeys(r.doc_id for r in new_exec.results))
        appended = new_ids[len(base_ids):]

        cases.append({
            "question_id": qid,
            "gold_doc": gold_doc,
            "prefix_identical": new_ids[:len(base_ids)] == base_ids,
            "baseline_size": len(base_ids),
            "new_size": len(new_ids),
            "crowding": trace.get("max_chunks_from_document"),
            "triggered": bool(trace.get("crowding_detected")),
            "appended": bool(trace.get("appended")),
            "appended_chunk_id": trace.get("appended_chunk_id"),
            "appended_doc_id": trace.get("appended_doc_id"),
            "original_candidate_rank": trace.get("original_candidate_rank"),
            "gold_in_baseline": gold_doc in base_docs,
            "gold_in_new": gold_doc in new_docs,
            "latency_ms": latency,
            "base_answer": base_answer,
            "new_answer": new_answer,
            "lost_facts": sorted(set(base_answer["facts"]) - set(new_answer["facts"])),
            "gained_facts": sorted(set(new_answer["facts"]) - set(base_answer["facts"])),
            "uniqueness_broken": (base_answer["semantic_unique"]
                                  and not new_answer["semantic_unique"]),
        })

    n = len(cases)
    summary = {
        "questions": n,
        "prefix_identical_all": all(c["prefix_identical"] for c in cases),
        "triggered": sorted(c["question_id"] for c in cases if c["triggered"]),
        "appended": sorted(c["question_id"] for c in cases if c["appended"]),
        "base_R@10_gold_document": round(
            sum(1 for c in cases if c["gold_in_baseline"]) / n, 4),
        "expanded_R@11_gold_document": round(
            sum(1 for c in cases if c["gold_in_new"]) / n, 4),
        "recovered": sorted(c["question_id"] for c in cases
                            if c["gold_in_new"] and not c["gold_in_baseline"]),
        "questions_with_lost_facts": sorted(
            c["question_id"] for c in cases if c["lost_facts"]),
        "questions_with_broken_uniqueness": sorted(
            c["question_id"] for c in cases if c["uniqueness_broken"]),
        "questions_with_gained_facts": sorted(
            c["question_id"] for c in cases if c["gained_facts"]),
        "context_growth_chunks": sum(c["new_size"] - c["baseline_size"] for c in cases),
        "latency_ms": {"mean": round(statistics.mean(latencies), 1),
                       "p95": round(sorted(latencies)[
                           min(n - 1, int(n * 0.95))], 1)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False,
                   indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    for c in cases:
        if c["gained_facts"] or c["lost_facts"] or c["uniqueness_broken"]:
            print(f"\n{c['question_id']}: appended={c['appended_doc_id']} "
                  f"(scored rank {c['original_candidate_rank']})")
            print(f"   match {c['base_answer']['matching_event_count']}"
                  f" -> {c['new_answer']['matching_event_count']}"
                  f"   unique {c['base_answer']['semantic_unique']}"
                  f" -> {c['new_answer']['semantic_unique']}")
            if c["lost_facts"]:
                print(f"   LOST: {c['lost_facts']}")
            if c["gained_facts"]:
                print(f"   gained: {c['gained_facts']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
