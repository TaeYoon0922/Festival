"""Phase 5.1: is the shared additive rescue safe outside holding?

The rescue lives in the retrieval path every domain uses, but it was only ever
validated on holding.  This runs all 54 P0-D-accepted Gold60 questions twice --
once with the rescue disabled, once as implemented -- with nothing else
different, and compares what each domain's downstream stage produced.

Baseline is obtained by swapping the rescue function, not by editing production
or rebinding its limit: that limit is a default argument bound at definition
time, so rebinding it would silently change nothing.
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
from app.parsing.final_validation import GOLD_QUESTIONS, HOLDING_ADDITIONAL_QUESTIONS
from scripts.bge_eval_preflight import pinned_encoder_factory

P0D_DECLINED = {"H01", "H02", "HX01", "HX02", "HX03", "HX04"}

_DISABLED_TRACE = {
    "attempted": False, "crowding_detected": False, "max_chunks_from_document": 0,
    "appended": False, "appended_chunk_id": None, "appended_doc_id": None,
    "original_candidate_rank": None,
}


def accepted_questions():
    out = []
    for question in [*GOLD_QUESTIONS, *HOLDING_ADDITIONAL_QUESTIONS]:
        if str(question["question_id"]) not in P0D_DECLINED:
            out.append(question)
    return sorted(out, key=lambda q: str(q["question_id"]))


def downstream(orch, text, plan, execution):
    """Whatever the domain's resolver produced, described uniformly."""

    result = orch.run(text, plan, execution)
    resolution = getattr(result, "resolution", None)
    draft = getattr(result, "answer_draft", None)
    ambiguity = getattr(draft, "ambiguity", None) or {}
    events = list(getattr(resolution, "events", None) or [])
    facts = list(getattr(resolution, "facts", None) or [])
    rendered = ()
    if events:
        rendered = composer._reported_holding_events(resolution)
    citations = getattr(draft, "citations", ()) or ()

    def describe(item):
        for attrs in (("reference_date", "after_shares", "after_ratio"),
                      ("metric", "period", "value", "unit")):
            if all(hasattr(item, a) for a in attrs):
                return "|".join(str(getattr(item, a)) for a in attrs)
        return repr(item)[:160]

    return {
        "resolution_type": type(resolution).__name__ if resolution else None,
        "events": len(events),
        "facts": len(facts),
        "matching_event_count": sum(1 for e in events if e.matches_query is True),
        "rendered_event_count": len(rendered),
        "grounded": sorted({describe(x) for x in [*rendered, *facts]}),
        "citations": len(citations),
        "citation_chunks": sorted({str(c) for c in citations}),
        "answerable": bool(getattr(result, "answerable", None)
                           if hasattr(result, "answerable")
                           else getattr(draft, "answerable", None)),
        "selection_mode": ambiguity.get("selection_mode"),
        "semantic_unique": bool(ambiguity.get("semantic_unique")),
        "under_specified": bool(ambiguity.get("under_specified")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/out/bge/p1r_global_domains.json")
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

    executor, orch = pipeline.executor, pipeline.orchestrator
    shipped = hybrid._additive_document_rescue

    def disabled(results, tail, **kwargs):
        return list(results), dict(_DISABLED_TRACE)

    cases = []
    for question in accepted_questions():
        qid = str(question["question_id"])
        text = str(question["query"])
        gold_doc = str(question["doc_id"])
        group = str(question.get("doc_group") or "?")
        plan, validation = pipeline._validated_understanding(text)
        if not validation.retrieval_allowed:
            cases.append({"question_id": qid, "doc_group": group,
                          "skipped": "retrieval not allowed"})
            continue

        hybrid._additive_document_rescue = disabled
        clock = time.perf_counter()
        base_exec = executor.execute(plan)
        base_latency = round((time.perf_counter() - clock) * 1000, 1)
        base_ids = [r.chunk_id for r in base_exec.results]
        base_docs = list(dict.fromkeys(r.doc_id for r in base_exec.results))
        base_down = downstream(orch, text, plan, base_exec)

        hybrid._additive_document_rescue = shipped
        clock = time.perf_counter()
        new_exec = executor.execute(plan)
        new_latency = round((time.perf_counter() - clock) * 1000, 1)
        new_ids = [r.chunk_id for r in new_exec.results]
        new_docs = list(dict.fromkeys(r.doc_id for r in new_exec.results))
        new_down = downstream(orch, text, plan, new_exec)

        trace = (new_exec.routing.get("hybrid") or {}).get(
            "additive_document_recovery") or {}
        expansion_docs = sorted(
            set((new_exec.correction_expansion or {}).get("added_doc_ids", []))
            | set((new_exec.event_expansion or {}).get("added_doc_ids", [])))
        chunk_chars = {c.chunk_id: len(str(c.chunk.get("retrieval_text") or ""))
                       for c in new_exec.chunks}

        cases.append({
            "question_id": qid, "doc_group": group, "gold_doc": gold_doc,
            "prefix_identical": new_ids[:len(base_ids)] == base_ids,
            "baseline_size": len(base_ids), "new_size": len(new_ids),
            "baseline_distinct_docs": len(base_docs),
            "new_distinct_docs": len(new_docs),
            "max_chunks_one_doc": trace.get("max_chunks_from_document"),
            "triggered": bool(trace.get("crowding_detected")),
            "appended": bool(trace.get("appended")),
            "appended_chunk_id": trace.get("appended_chunk_id"),
            "appended_doc_id": trace.get("appended_doc_id"),
            "appended_is_gold": trace.get("appended_doc_id") == gold_doc,
            "appended_chars": chunk_chars.get(trace.get("appended_chunk_id"), 0),
            "original_candidate_rank": trace.get("original_candidate_rank"),
            "expansion_added_docs": expansion_docs,
            "append_duplicates_expansion": bool(
                trace.get("appended_doc_id")
                and trace.get("appended_doc_id") in expansion_docs),
            "gold_doc_rank_base": (base_docs.index(gold_doc) + 1
                                   if gold_doc in base_docs else None),
            "gold_in_base": gold_doc in base_docs,
            "gold_in_new": gold_doc in new_docs,
            "base_latency_ms": base_latency, "new_latency_ms": new_latency,
            "base_downstream": base_down, "new_downstream": new_down,
            "lost_grounded": sorted(set(base_down["grounded"]) - set(new_down["grounded"])),
            "gained_grounded": sorted(set(new_down["grounded"]) - set(base_down["grounded"])),
            "lost_citations": sorted(set(base_down["citation_chunks"])
                                     - set(new_down["citation_chunks"])),
            "uniqueness_broken": (base_down["semantic_unique"]
                                  and not new_down["semantic_unique"]),
        })

    live = [c for c in cases if "skipped" not in c]
    by_group: dict[str, list] = collections.defaultdict(list)
    for c in live:
        by_group[c["doc_group"]].append(c)

    def block(rows):
        n = len(rows)
        base_ranks = [r["gold_doc_rank_base"] for r in rows]
        found = [r for r in base_ranks if isinstance(r, int)]
        return {
            "questions": n,
            "triggered": sum(1 for r in rows if r["triggered"]),
            "appended": sum(1 for r in rows if r["appended"]),
            "prefix_identical": all(r["prefix_identical"] for r in rows),
            "base_gold_doc_recall_at_10": round(
                sum(1 for r in rows if r["gold_in_base"]) / n, 4) if n else 0.0,
            "expanded_evidence_recall_at_11": round(
                sum(1 for r in rows if r["gold_in_new"]) / n, 4) if n else 0.0,
            "base_doc_MRR": round(sum(1.0 / r for r in found) / n, 4) if n else 0.0,
            **{f"base_doc_R@{k}": round(sum(1 for r in found if r <= k) / n, 4) if n else 0.0
               for k in (1, 3, 5, 10)},
            "questions_with_lost_grounded": sorted(
                r["question_id"] for r in rows if r["lost_grounded"]),
            "questions_with_gained_grounded": sorted(
                r["question_id"] for r in rows if r["gained_grounded"]),
            "questions_with_lost_citations": sorted(
                r["question_id"] for r in rows if r["lost_citations"]),
            "uniqueness_broken": sorted(
                r["question_id"] for r in rows if r["uniqueness_broken"]),
            "mean_distinct_docs": round(
                statistics.mean(r["baseline_distinct_docs"] for r in rows), 2) if n else 0,
            "latency_base_mean": round(
                statistics.mean(r["base_latency_ms"] for r in rows), 1) if n else 0,
            "latency_new_mean": round(
                statistics.mean(r["new_latency_ms"] for r in rows), 1) if n else 0,
        }

    payload = {
        "overall": block(live),
        "by_doc_group": {g: block(rows) for g, rows in sorted(by_group.items())},
        "skipped": [c for c in cases if "skipped" in c],
        "appended_chars_total": sum(c["appended_chars"] for c in live if c["appended"]),
        "cases": cases,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                         default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "cases"},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
