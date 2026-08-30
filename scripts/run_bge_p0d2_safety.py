"""P0-D.2 activation safety, measured under live BGE-M3.

Production P0-D is untouched: it still declines these questions.  What is
simulated here is only the *representation* the diagnosed role resolver would
produce -- company = issuer, reporter = holder -- which is then passed through
the frozen holding stack unchanged.

The issuer/reporter direction is derived from holding projection metadata in the
evaluation database, exactly as P0-D.2 specified.  Gold is read afterwards to
score the outcome and never to choose evidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.reasoning.answer_composer as composer
from app.api.pipeline import AnswerPipeline
from app.reasoning.holding_reporter import canonical_reporter_key, reporter_matches
from app.retrieval.embeddings import EmbeddingConfig, create_embedding_provider
from scripts.bge_eval_preflight import (
    assert_embedding_identity,
    describe,
    pinned_encoder_factory,
)

SIX = [
    ("H01", "에스엠 하이브 이번 보고 보유 주식수와 비율"),
    ("H02", "에스엠 하이브 풋옵션 주식 취득 증감 수량"),
    ("HX01", "에스엠 하이브 2024년 3월 14일 현재 보유 수량 비율"),
    ("HX02", "에스엠 하이브 직전보고 보유주식 수 비율"),
    ("HX03", "에스엠 하이브 보유주식 증가 수량 증가 비율"),
    ("HX04", "에스엠 하이브 풋옵션 행사 주식 취득일과 취득 수량"),
]
MIN_SUPPORT_DOCS = 2


def directed_support(backend) -> dict[tuple[str, str], int]:
    """(issuer_code, reporter_key) -> distinct holding documents supporting it."""

    rows = backend._fetch_all(
        """
        SELECT d.corp_code AS issuer,
               c.metadata->'projection_fields'->>'보고자/보유자' AS reporter,
               c.doc_id AS doc_id
        FROM chunks c JOIN disclosures d ON d.doc_id = c.doc_id
        WHERE d.doc_group = 'holding'
          AND c.metadata->'projection_fields' ? '보고자/보유자'
        """,
        [],
    )
    docs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = canonical_reporter_key(row["reporter"])
        if key:
            docs[(str(row["issuer"]), key)].add(str(row["doc_id"]))
    return {k: len(v) for k, v in docs.items()}


def resolve_roles(a_code, a_name, b_code, b_name, support):
    """Exactly one direction with >= MIN_SUPPORT_DOCS wins; otherwise decline."""

    forward = support.get((a_code, canonical_reporter_key(b_name)), 0)
    reverse = support.get((b_code, canonical_reporter_key(a_name)), 0)
    if forward >= MIN_SUPPORT_DOCS and reverse < MIN_SUPPORT_DOCS:
        return (a_code, a_name, b_name, forward, reverse)
    if reverse >= MIN_SUPPORT_DOCS and forward < MIN_SUPPORT_DOCS:
        return (b_code, b_name, a_name, reverse, forward)
    return (None, None, None, forward, reverse)


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
    parser.add_argument("--out", default="/out/bge/p0d2_live_safety.json")
    args = parser.parse_args()

    config = EmbeddingConfig.from_env()
    assert_embedding_identity(config, require_device="cuda")

    import app.api.pipeline as pipeline_module

    original = pipeline_module.create_embedding_provider

    def pinned(cfg, **kwargs):
        return original(cfg, bge_encoder_factory=pinned_encoder_factory, **kwargs)

    pipeline_module.create_embedding_provider = pinned
    try:
        pipeline = AnswerPipeline.from_env()
    finally:
        pipeline_module.create_embedding_provider = original

    orch = pipeline.orchestrator
    backend = pipeline.executor._metadata_backend
    support = directed_support(backend)

    gold = {}
    gold_path = Path("reports/evaluation/gold60/2026-08-21-agent-90pct/"
                     "gold60_agent_questions.jsonl")
    for line in gold_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            gold[row["question_id"]] = row["gold"]["doc_id"]

    results = []
    for qid, question in SIX:
        plan, validation = pipeline._validated_understanding(question)
        current = {"state": str(validation.state),
                   "retrieval_allowed": bool(validation.retrieval_allowed)}

        names = list(getattr(plan, "companies", ()) or ())
        codes = list(getattr(plan, "corp_codes", ()) or ())
        record = {"question_id": qid, "question": question,
                  "current_mode": current, "parsed_companies": names}
        if len(names) != 2 or len(codes) != 2:
            record["simulated"] = {"resolved": False,
                                   "reason": "not a two-company holding query"}
            results.append(record)
            continue

        issuer_code, issuer, reporter, fwd, rev = resolve_roles(
            codes[0], names[0], codes[1], names[1], support)
        record["role_resolution"] = {
            "issuer": issuer, "reporter": reporter,
            "forward_support_docs": fwd, "reverse_support_docs": rev,
            "min_support_docs": MIN_SUPPORT_DOCS,
            "resolved": issuer is not None,
        }
        if issuer is None:
            record["simulated"] = {"resolved": False,
                                   "reason": "no unique supported direction"}
            results.append(record)
            continue

        simulated = replace(plan, companies=(issuer,), corp_codes=(issuer_code,),
                            reporter=reporter)
        clock = time.perf_counter()
        execution = pipeline.executor.execute(simulated)
        latency_ms = round((time.perf_counter() - clock) * 1000, 1)
        result = orch.run(question, simulated, execution)

        served = list(dict.fromkeys(r.doc_id for r in execution.results))
        intended = gold.get(qid)
        rank = next((r.rank for r in execution.results if r.doc_id == intended), None)

        resolution = result.resolution
        events = list(getattr(resolution, "events", None) or [])
        matching = [e for e in events if e.matches_query is True]
        rendered = composer._reported_holding_events(resolution)
        draft = getattr(result, "answer_draft", None)
        ambiguity = getattr(draft, "ambiguity", None) or {}
        citations = len(getattr(draft, "citations", ()) or ()) if draft else 0

        intended_top10 = intended in served[:10]
        selected_docs = {getattr(e, "doc_id", None) for e in matching}
        # Confidently wrong: a single event is presented as the answer while the
        # document that carries the asked-about evidence never arrived.
        confidently_wrong = bool(
            not intended_top10 and len(rendered) == 1
            and intended not in selected_docs)

        competing = sorted({
            getattr(e, "doc_id", None) for e in events
            if reporter_matches(getattr(e, "reporter", "") or "", reporter)
        } - {intended})

        record["simulated"] = {
            "candidate_docs": len(served),
            "intended_doc_rank": rank,
            "intended_doc_in_top10": intended_top10,
            "matching_event_count": len(matching),
            "selected_event_dates": sorted(
                e.reference_date or "?" for e in matching),
            "rendered_event_count": len(rendered),
            "selection_mode": ambiguity.get("selection_mode"),
            "under_specified": bool(ambiguity.get("under_specified")),
            "exact_multi_match": bool(ambiguity.get("exact_multi_match")),
            "semantic_unique": bool(ambiguity.get("semantic_unique")),
            "citations": citations,
            "competing_same_reporter_docs": competing[:5],
            "confidently_wrong": confidently_wrong,
            "retrieval_latency_ms": latency_ms,
        }
        results.append(record)

    gate = {
        "1_intended_evidence_in_top10": all(
            r.get("simulated", {}).get("intended_doc_in_top10") for r in results),
        "2_no_confidently_wrong_answer": not any(
            r.get("simulated", {}).get("confidently_wrong") for r in results),
    }
    payload = {
        "label": "LIVE_BGE_M3",
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "corpus_snapshot": "structural_v2_1_full_4204",
            "db_identifier": "festival-verify:55433/festival_verify",
            "production_p0d_modified": False,
            **describe(config),
        },
        "activation_gate": gate,
        "cases": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
