"""Run failure attribution over an evaluation set, one company at a time.

Scoped by company on purpose.  Preparing every company's corpus at once has
already been measured at ~11.5 GB during embedding, so the runner prepares one
group, evaluates it, writes its results, releases them, and moves on.  Each
group's output is a separate file, so an interrupted run resumes by skipping the
groups that already finished rather than starting over.

MODE A needs only the disclosure table and is embedding-independent.  MODE B
additionally needs a retrieval run, and records which embedding provider
produced it so a ranking conclusion is never mistaken for a live one.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.retrieval_failure_evaluator import (  # noqa: E402
    Observation,
    aggregate,
    classify,
    markdown_table,
)

DEFAULT_GOLD = (
    "reports/evaluation/gold60/2026-08-21-agent-90pct/gold60_agent_questions.jsonl"
)
DEFAULT_MANIFEST = "data/corpus/manifest.jsonl"
PROCESSED = Path("data/processed/structural_v2_1_full_4204")


@lru_cache(maxsize=64)
def _corpus_index():
    """doc_id -> processed part records, read once per process."""

    index = defaultdict(list)
    for line in open(PROCESSED / "index.jsonl", encoding="utf-8"):
        if line.strip():
            record = json.loads(line)
            index[record["doc_id"]].append(record)
    return index


@lru_cache(maxsize=256)
def doc_chunks(doc_id):
    """chunk_id -> (table_id, text) for one document, from the built corpus.

    Table membership is a property of chunking, so it reads the same whatever
    embedded the corpus.  That is what keeps a sibling verdict embedding-
    independent.
    """

    mapping = {}
    for record in _corpus_index().get(doc_id, []):
        path = PROCESSED / record["output_path"].replace("\\", "/")
        if not path.exists():
            continue
        data = json.load(gzip.open(path, "rt", encoding="utf-8"))
        for chunk in data.get("chunks") or []:
            mapping[chunk.get("chunk_id")] = (
                chunk.get("table_id"),
                str(chunk.get("retrieval_text") or chunk.get("content") or ""),
            )
    return mapping


def table_of(doc_id):
    return {cid: tid for cid, (tid, _text) in doc_chunks(doc_id).items()}


def _normalise(value):
    return "".join(str(value or "").split()).casefold()


def gold_chunks(row):
    """The chunk ids the evaluation set marks relevant -- read after the run."""

    gold = row.get("gold") or {}
    ids = [c.get("chunk_id") for c in (gold.get("relevant_chunks") or [])]
    return tuple(c for c in ids if c)


def gold_terms(row):
    """The strings the evaluation set treats as the answer-bearing evidence."""

    gold = row.get("gold") or {}
    return tuple(t for t in (gold.get("evidence_terms") or []) if str(t).strip())


def load_manifest(path):
    return {r["doc_id"]: r for r in (json.loads(l) for l in
            open(path, encoding="utf-8") if l.strip())}


def gold_doc(row, manifest):
    value = row.get("gold_doc") or (row.get("gold") or {}).get("doc_id")
    return value if value in manifest else None


def observe(row, manifest, pipeline, *, mode):
    """Run production, then look for the required document in what it returned."""

    qid = row["question_id"]
    question = row["question"]
    required = tuple(d for d in (gold_doc(row, manifest),) if d)

    try:
        plan, validation = pipeline._validated_understanding(question)
    except Exception as error:  # noqa: BLE001 - recorded, not raised
        return Observation(question_id=qid, question=question,
                           retrieval_allowed=False,
                           understanding_state=f"error:{type(error).__name__}",
                           required_docs=required)
    if not validation.retrieval_allowed:
        return Observation(question_id=qid, question=question,
                           retrieval_allowed=False,
                           understanding_state=str(validation.state),
                           required_docs=required)

    router = pipeline.executor.router
    route = router.route(plan)
    documents = pipeline.executor._metadata_backend.get_candidate_documents(
        **route.backend_filters)
    documents = router.filter_documents(documents, route)
    eligible = tuple(d.doc_id for d in documents)

    excluded = {}
    if required and not set(required) & set(eligible):
        excluded = {k: v for k, v in route.backend_filters.items()
                    if v not in (None, [], ())}
        excluded["hard_routes"] = dict(route.hard_routes)

    if mode == "A":
        return Observation(question_id=qid, question=question,
                           required_docs=required, eligible_docs=eligible,
                           excluded_by=excluded)

    execution = pipeline.executor.execute(plan)
    served = tuple(dict.fromkeys(r.doc_id for r in execution.results))
    rank = next((r.rank for r in execution.results if r.doc_id in set(required)),
                None)
    provider = getattr(pipeline.executor.embedding_config, "provider", None)
    siblings, coverage, lost = _evidence_misses(row, execution)
    return Observation(question_id=qid, question=question,
                       required_docs=required, eligible_docs=eligible,
                       excluded_by=excluded, ranking_available=True,
                       served_docs=served, first_required_rank=rank,
                       evidence_available=bool(gold_chunks(row)),
                       missing_fields=lost,
                       sibling_misses=siblings, coverage_misses=coverage,
                       embedding_provider=provider)


def _evidence_misses(row, execution):
    """Report an evidence loss only when a fact the answer needed is absent.

    An unserved gold chunk is not by itself a loss.  The evaluation set marks
    one chunk that carries a fact; a structured projection can carry the same
    fact, and the frozen holding stack routinely delivers it that way.  Blaming
    P1-C for a fact that in truth arrived would reopen a phase on exactly the
    kind of wrong evidence this evaluator exists to prevent.

    So the fact decides.  A gold evidence term missing from everything served is
    a loss; it is P1-C's only when an unserved chunk of an *already served*
    table in the same document is what carried it.
    """

    required, terms = gold_chunks(row), gold_terms(row)
    if not required:
        return (), (), ()

    served_ids = {r.chunk_id for r in execution.results}
    served_text = _normalise(" ".join(
        doc_chunks(r.doc_id).get(r.chunk_id, (None, ""))[1]
        for r in execution.results))
    lost = tuple(t for t in terms if _normalise(t) not in served_text)

    served_tables = defaultdict(set)
    for result in execution.results:
        tid = doc_chunks(result.doc_id).get(result.chunk_id, (None, ""))[0]
        if tid:
            served_tables[result.doc_id].add(tid)

    siblings, coverage = [], []
    for chunk_id in required:
        if chunk_id in served_ids:
            continue
        doc = chunk_id.split(":", 1)[0]
        tid, text = doc_chunks(doc).get(chunk_id, (None, ""))
        carried = tuple(t for t in lost if _normalise(t) in _normalise(text))
        if not carried:
            # Unserved, but nothing the answer needed was only here.
            continue
        record = {"sibling_chunk_id": chunk_id, "doc_id": doc, "table_id": tid,
                  "lost_terms": list(carried)}
        if tid and tid in served_tables.get(doc, ()):
            record["anchor_chunk_id"] = next(
                (r.chunk_id for r in execution.results if r.doc_id == doc
                 and doc_chunks(doc).get(r.chunk_id, (None, ""))[0] == tid), None)
            siblings.append(record)
        else:
            coverage.append({"chunk_id": chunk_id, "doc_id": doc,
                             "table_id": tid, "lost_terms": list(carried)})
    return tuple(siblings), tuple(coverage), lost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("A", "B"), default="A")
    parser.add_argument("--gold", default=DEFAULT_GOLD)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default="reports/evaluation/failure_attribution")
    parser.add_argument("--company", action="append", default=None,
                        help="restrict to these corp codes; repeatable")
    parser.add_argument("--resume", action="store_true",
                        help="skip groups whose result file already exists")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    rows = [json.loads(l) for l in open(args.gold, encoding="utf-8") if l.strip()]

    groups = defaultdict(list)
    for row in rows:
        doc = gold_doc(row, manifest)
        corp = manifest[doc]["corp_code"] if doc else "unresolved"
        if args.company and corp not in set(args.company):
            continue
        groups[corp].append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from app.api.pipeline import AnswerPipeline  # noqa: E402 - after path setup

    pipeline = AnswerPipeline.from_env()
    doc_groups, tasks, attributions = {}, {}, []

    for corp in sorted(groups):
        target = out / f"group_{corp}.mode{args.mode}.jsonl"
        if args.resume and target.exists():
            for line in open(target, encoding="utf-8"):
                if line.strip():
                    attributions.append(_rehydrate(json.loads(line)))
            print(f"  {corp}: resumed {target.name}")
            continue
        records = []
        for row in sorted(groups[corp], key=lambda r: r["question_id"]):
            observation = observe(row, manifest, pipeline, mode=args.mode)
            attribution = classify(observation)
            records.append(attribution)
            doc = gold_doc(row, manifest)
            if doc:
                doc_groups[row["question_id"]] = manifest[doc]["doc_group"]
            tasks[row["question_id"]] = row.get("task_type") or "unknown"
        with open(target, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        attributions.extend(records)
        print(f"  {corp}: {len(records)} questions -> {target.name}")

    summary = aggregate(attributions, groups=doc_groups, tasks=tasks,
                        metadata={
                            "mode": args.mode,
                            "gold": args.gold,
                            "questions": len(attributions),
                            "embedding_provider": os.environ.get(
                                "FESTIVAL_EMBEDDING_PROVIDER", "hash"),
                            "commit": _commit(),
                            "corpus": str(PROCESSED),
                            "companies": sorted(groups),
                        })
    (out / f"summary.mode{args.mode}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / f"summary.mode{args.mode}.md").write_text(
        markdown_table(attributions), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False))
    return 0


def _commit():
    """The code the run was produced by -- half of what makes it reproducible."""

    import subprocess

    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        return "unknown"


def _rehydrate(record):
    from scripts.retrieval_failure_evaluator import Attribution

    return Attribution(
        question_id=record["question_id"], primary=record["primary_attribution"],
        owner=record["owner_phase"], mode=record["mode"],
        embedding_confidence=record["embedding_confidence"],
        caveats=tuple(record.get("caveats") or ()),
        details=record.get("details") or {})


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
