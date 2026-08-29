"""Sections 8-9: what an appended document does to the frozen holding answer.

Losing a fact is impossible here by contract -- the emitted ten are untouched --
so the question is the opposite one: does an extra document introduce a matching
event that turns a confident single-event answer into an ambiguous one?  That
would be a regression even though nothing was removed.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.reasoning.answer_composer as composer
from scripts.diagnose_p1r_additive import RETRIEVAL, additive_candidates, build
from scripts.diagnose_p1r_live import holding_questions

WIDE_TOP_K = 50


def summarize(result):
    resolution = getattr(result, "resolution", None)
    events = list(getattr(resolution, "events", None) or [])
    matching = [e for e in events if e.matches_query is True]
    rendered = composer._reported_holding_events(resolution) if resolution else ()
    draft = getattr(result, "answer_draft", None)
    ambiguity = getattr(draft, "ambiguity", None) or {}
    return {
        "events": len(events),
        "matching_event_count": len(matching),
        "rendered_event_count": len(rendered),
        "dates": sorted(e.reference_date or "?" for e in matching),
        "facts": sorted(f"{e.reference_date}|{e.after_shares}|{e.after_ratio}"
                        for e in rendered),
        "citations": len(getattr(draft, "citations", ()) or ()),
        "selection_mode": ambiguity.get("selection_mode"),
        "semantic_unique": bool(ambiguity.get("semantic_unique")),
        "under_specified": bool(ambiguity.get("under_specified")),
        "exact_multi_match": bool(ambiguity.get("exact_multi_match")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appends", type=int, default=1)
    parser.add_argument("--out", default="/out/bge/p1r_additive_semantics.json")
    args = parser.parse_args()

    import app.api.pipeline as pipeline_module
    from scripts.bge_eval_preflight import pinned_encoder_factory

    original = pipeline_module.create_embedding_provider
    pipeline_module.create_embedding_provider = (
        lambda cfg, **kw: original(cfg, bge_encoder_factory=pinned_encoder_factory, **kw))
    try:
        from app.api.pipeline import AnswerPipeline

        pipeline = AnswerPipeline.from_env()
    finally:
        pipeline_module.create_embedding_provider = original
    orch = pipeline.orchestrator

    executor, understanding = build()
    wide = replace(executor.config, final_top_k=WIDE_TOP_K)
    wide_executor = type(executor)(
        executor._metadata_backend, executor._embedder, executor.embedding_config,
        config=wide)

    rows_out, harmful = [], []
    for question in holding_questions():
        qid = str(question["question_id"])
        text = str(question["query"])
        plan = understanding.understand(text)

        base_exec = executor.execute(plan)
        base_results = list(base_exec.results)
        emitted_docs = list(dict.fromkeys(r.doc_id for r in base_results))
        rows = [dict(r) for r in base_exec.rerank_diagnostics]
        picks = additive_candidates(rows, emitted_docs, limit=args.appends)
        if not picks:
            continue

        # execute() takes min(plan.top_k, final_top_k), so the plan must be
        # widened too or the wide executor still returns only ten.
        wide_exec = wide_executor.execute(replace(plan, top_k=WIDE_TOP_K))
        by_chunk = {r.chunk_id: r for r in wide_exec.results}
        appended = [by_chunk[p["chunk_id"]] for p in picks if p["chunk_id"] in by_chunk]
        if len(appended) != len(picks):
            rows_out.append({"question_id": qid, "skipped": "append not resolvable"})
            continue

        before = summarize(orch.run(text, plan, base_exec))
        injected = replace(base_exec, results=tuple([*base_results, *appended]))
        after = summarize(orch.run(text, plan, injected))

        prefix_ok = [r.chunk_id for r in injected.results][:len(base_results)] == [
            r.chunk_id for r in base_results]
        lost_facts = sorted(set(before["facts"]) - set(after["facts"]))
        gained_facts = sorted(set(after["facts"]) - set(before["facts"]))
        uniqueness_broken = before["semantic_unique"] and not after["semantic_unique"]
        if lost_facts or uniqueness_broken:
            harmful.append(qid)

        rows_out.append({
            "question_id": qid,
            "prefix_identical": prefix_ok,
            "appended": [{"chunk_id": p["chunk_id"], "doc_id": p.get("doc_id"),
                          "scored_rank": int(p["final_rank"])} for p in picks],
            "before": before, "after": after,
            "lost_facts": lost_facts, "gained_facts": gained_facts,
            "uniqueness_broken": uniqueness_broken,
        })

    payload = {
        "appends": args.appends,
        "prefix_identical_all": all(r.get("prefix_identical", True) for r in rows_out),
        "questions_with_lost_facts": sorted(
            r["question_id"] for r in rows_out if r.get("lost_facts")),
        "questions_with_broken_uniqueness": sorted(
            r["question_id"] for r in rows_out if r.get("uniqueness_broken")),
        "questions_with_gained_facts": sorted(
            r["question_id"] for r in rows_out if r.get("gained_facts")),
        "harmful": sorted(set(harmful)),
        "cases": rows_out,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                         default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "cases"},
                     ensure_ascii=False, indent=1, default=str))
    for row in rows_out:
        if row.get("lost_facts") or row.get("uniqueness_broken") or row.get("gained_facts"):
            print(f"\n{row['question_id']}: appended={[a['doc_id'] for a in row['appended']]}")
            print(f"   match {row['before']['matching_event_count']} -> {row['after']['matching_event_count']}"
                  f"   unique {row['before']['semantic_unique']} -> {row['after']['semantic_unique']}"
                  f"   mode {row['before']['selection_mode']} -> {row['after']['selection_mode']}")
            if row["lost_facts"]:
                print(f"   LOST: {row['lost_facts']}")
            if row["gained_facts"]:
                print(f"   gained: {row['gained_facts']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
