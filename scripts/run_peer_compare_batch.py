"""Run peer-compare questions (PC01–PC30) through the Festival answer stack.

Two modes mirror how taeyoon actually executes ``GET /answer``:

``routing`` (no Postgres)
    P0-D only — understanding, validation, multi-document *plan*, task route,
    and orchestrator on empty retrieval.  Matches local probe when DB is absent.

``e2e`` (Postgres + embeddings)
    Full ``AnswerPipeline.from_env()`` — the same path Gold60 and ``/answer`` use.

Actual serving order (taeyoon, ``app/api/pipeline.py``)::

    query_understanding          → QueryPlan
    query_validation (P0-D)    → RESOLVED only proceeds; else clarification gate
    [hcx_semantic_fallback]     → at most once, then revalidation
    hybrid retrieval            → lexical + vector → RRF → rerank
    [correction_expansion]      → P0-A, inside retrieval
    [corporate_event_expansion] → P0-B, inside retrieval
    multi_document_planner      → P0-C, after retrieval (additive)
    multi_document_executor
    [multi_document_evidence]
    orchestrator                → task_router → resolver → answer_composer
    answer_generator
    answerability_guard         → P0-D
    hcx_verbalizer              → optional; fails back to deterministic answer

Examples::

    python scripts/run_peer_compare_batch.py --mode routing
    FESTIVAL_HCX_ENABLED=false python scripts/run_peer_compare_batch.py --mode e2e
    python scripts/run_peer_compare_batch.py --ids PC01 PC03 --save-full
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.orchestrator import AgentOrchestrator
from app.agent.task_router import route_task
from app.api.pipeline import AnswerPipeline, AnswerPipelineError
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.reasoning.multi_document_planner import MultiDocumentPlanner
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope, QueryState, QueryValidator
from app.retrieval.postgres_backend import PostgresBackend

DEFAULT_BATCH = PROJECT_ROOT / "qa-tool" / "examples" / "peer-compare-batch.txt"
DEFAULT_OUTPUT_ROUTING = (
    PROJECT_ROOT / "qa-tool" / "output" / "peer-compare-probe-full.json"
)
DEFAULT_OUTPUT_E2E = PROJECT_ROOT / "qa-tool" / "output" / "peer-compare-e2e.json"

PIPELINE_REFERENCE = "origin/taeyoon + feat/qa-driven-routing (P0-D/C/A/B serving path)"


def load_questions(
    batch_path: Path,
    *,
    ids: set[str] | None = None,
) -> list[tuple[str, str]]:
    lines = [
        line.strip()
        for line in batch_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 30:
        raise ValueError(
            f"expected 30 questions in {batch_path}, found {len(lines)}"
        )
    rows: list[tuple[str, str]] = []
    for index, question in enumerate(lines, start=1):
        question_id = f"PC{index:02d}"
        if ids is not None and question_id not in ids:
            continue
        rows.append((question_id, question))
    if ids is not None and not rows:
        raise ValueError(f"no questions matched ids={sorted(ids)}")
    return rows


def _corp_names(retrieved_context: list[dict[str, Any]]) -> list[str]:
    seen: list[str] = []
    for row in retrieved_context:
        name = str(row.get("corp_name") or "").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def _plan_public(plan: Any) -> dict[str, Any]:
    evidence = dict(getattr(plan, "evidence", {}) or {})
    return {
        "task_type": getattr(plan, "task_type", None),
        "company": getattr(plan, "company", None),
        "companies": list(getattr(plan, "companies", ()) or ()),
        "corp_code": getattr(plan, "corp_code", None),
        "corp_codes": list(getattr(plan, "corp_codes", ()) or ()),
        "comparison_frame": evidence.get("comparison_frame"),
        "derived_metric": evidence.get("derived_metric"),
        "exchange_aggregate": evidence.get("exchange_aggregate"),
    }


def _summarize_e2e(
    question_id: str,
    question: str,
    payload: dict[str, Any] | None,
    *,
    error: str | None = None,
) -> dict[str, Any]:
    if payload is None:
        return {
            "question_id": question_id,
            "question": question,
            "mode": "e2e",
            "error": error or "unknown_error",
        }

    trace = payload.get("think_trace") or {}
    validation = trace.get("query_validation") or {}
    understanding = trace.get("query_understanding") or {}
    multi = trace.get("multi_document_planner") or {}
    answerability = trace.get("answerability") or {}
    retrieved = list(payload.get("retrieved_context") or [])
    answer = str(payload.get("answer") or "")

    return {
        "question_id": question_id,
        "question": question,
        "mode": "e2e",
        "pipeline_stages": list(trace.get("stages") or []),
        "route": trace.get("route"),
        "task_type": trace.get("task_type"),
        "validation_state": validation.get("status") or understanding.get("status"),
        "retrieval_allowed": validation.get("retrieval_allowed"),
        "comparison_frame": understanding.get("comparison_frame")
        or (understanding.get("resolved_slots") or {}).get("comparison_frame"),
        "corp_codes": understanding.get("corp_codes") or [],
        "retrieval_count": trace.get("retrieval_count", 0),
        "retrieved_corps": _corp_names(retrieved),
        "selected_evidence_count": trace.get("selected_evidence_count", 0),
        "answerable": trace.get("answerable"),
        "answerability_status": answerability.get("status"),
        "multi_document_applied": bool(multi.get("applied")),
        "multi_document_reason": multi.get("stop_reason") or multi.get("reason"),
        "correction_expanded": bool((trace.get("correction") or {}).get("correction_expanded")),
        "event_expanded": bool(trace.get("corporate_event")),
        "hcx_status": trace.get("hcx_status"),
        "warnings": list(trace.get("warnings") or []),
        "answer_preview": answer[:240],
        "answer_len": len(answer),
    }


def _summarize_routing(
    question_id: str,
    question: str,
    probe: dict[str, Any],
) -> dict[str, Any]:
    validation = probe.get("validation") or {}
    plan = probe.get("plan") or {}
    multi = probe.get("multi_document_planner") or {}
    route = probe.get("route") or {}

    stages = ["query_understanding", "query_validation", "multi_document_planner"]
    if validation.get("retrieval_allowed"):
        stages.extend(["task_router", "orchestrator_empty_retrieval", "answer_generator"])

    return {
        "question_id": question_id,
        "question": question,
        "mode": "routing",
        "pipeline_stages": stages,
        "route": route.get("resolver_type") or route.get("task_type"),
        "task_type": plan.get("task_type"),
        "validation_state": validation.get("status"),
        "retrieval_allowed": validation.get("retrieval_allowed"),
        "comparison_frame": plan.get("comparison_frame"),
        "corp_codes": plan.get("corp_codes") or [],
        "retrieval_count": 0,
        "retrieved_corps": [],
        "selected_evidence_count": probe.get("selected_evidence_count", 0),
        "answerable": probe.get("answerable"),
        "answerability_status": None,
        "multi_document_applied": bool(multi.get("applied")),
        "multi_document_reason": multi.get("stop_reason"),
        "correction_expanded": False,
        "event_expanded": False,
        "hcx_status": "skipped_routing_probe",
        "warnings": list(probe.get("warnings") or []),
        "answer_preview": str(probe.get("answer_preview") or "")[:240],
        "answer_len": len(str(probe.get("answer_preview") or "")),
        "route_signals": list(route.get("matched_signals") or []),
        "derived_metric": plan.get("derived_metric"),
        "exchange_aggregate": plan.get("exchange_aggregate"),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in rows:
        if row.get("error"):
            counts["error"] += 1
            continue
        counts[f"route:{row.get('route')}"] += 1
        counts[f"validation:{row.get('validation_state')}"] += 1
        frame = row.get("comparison_frame")
        if frame:
            counts[f"comparison_frame:{frame}"] += 1
        if row.get("retrieval_allowed"):
            counts["retrieval_allowed"] += 1
        if row.get("answerable"):
            counts["answerable"] += 1
        if row.get("multi_document_applied"):
            counts["multi_document_applied"] += 1
        corps = row.get("retrieved_corps") or []
        if len(corps) >= 2:
            counts["dual_corp_retrieval"] += 1
        elif corps:
            counts["single_corp_retrieval"] += 1
        elif row.get("mode") == "e2e" and row.get("retrieval_allowed"):
            counts["no_corp_retrieval"] += 1
    return dict(counts)


def _build_routing_stack() -> tuple[
    QueryUnderstanding, QueryValidator, MultiDocumentPlanner, AgentOrchestrator
]:
    scope = CorpusScope.repository_default()
    if scope is None:
        raise RuntimeError(
            "CorpusScope.repository_default() failed — "
            "data/corpus/universe.csv and manifest.jsonl required"
        )
    understanding = QueryUnderstanding(
        scope.company_aliases(),
        company_resolver=scope.resolve_company,
    )
    planner = MultiDocumentPlanner()
    validator = QueryValidator(
        corpus_scope=scope,
        multi_document_planner=planner,
    )
    orchestrator = AgentOrchestrator()
    return understanding, validator, planner, orchestrator


def _empty_execution() -> SimpleNamespace:
    return SimpleNamespace(results=(), chunks=())


def _run_routing_probe(
    question: str,
    *,
    understanding: QueryUnderstanding,
    validator: QueryValidator,
    planner: MultiDocumentPlanner,
    orchestrator: AgentOrchestrator,
) -> dict[str, Any]:
    plan = understanding.understand(question, top_k=10)
    validation = validator.validate(plan)
    public = validation.to_public_dict()
    multi_plan = planner.plan(question, validation.plan)
    multi_trace = multi_plan.to_dict()

    route = route_task(question, query_plan=validation.plan)
    answerable = False
    selected = 0
    warnings: list[str] = []
    answer_preview = ""

    if validation.retrieval_allowed:
        result = orchestrator.run(
            question,
            validation.plan,
            _empty_execution(),
            multi_document=None,
        )
        generated = CitationAwareAnswerGenerator().generate(result.answer_draft)
        answerable = generated.answerable
        selected = result.evidence_set.selected_evidence_count
        warnings = list(generated.warnings)
        answer_preview = generated.answer_text

    return {
        "plan": _plan_public(validation.plan),
        "validation": {
            "status": validation.state.value,
            "retrieval_allowed": validation.retrieval_allowed,
            "issues": list(validation.issues),
            "clarification": (
                validation.clarification.to_dict()
                if validation.clarification is not None
                else None
            ),
        },
        "multi_document_planner": {
            "applied": multi_trace.get("applied"),
            "stop_reason": multi_trace.get("stop_reason"),
            "plan_type": multi_trace.get("plan_type"),
        },
        "route": route.to_dict(),
        "answerable": answerable,
        "selected_evidence_count": selected,
        "warnings": warnings,
        "answer_preview": answer_preview,
        "query_state_resolved": validation.state is QueryState.RESOLVED,
    }


def _preflight_database() -> str | None:
    backend = PostgresBackend(connect_kwargs={"connect_timeout": 5})
    try:
        with backend.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception:  # noqa: BLE001
        return "database_unavailable"
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run PC01–PC30 peer-compare batch. "
            "Use --mode routing locally; --mode e2e on the server with Postgres."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("routing", "e2e"),
        default="routing",
        help="routing = P0-D + planner + route (no DB); e2e = full AnswerPipeline.",
    )
    parser.add_argument(
        "--batch",
        type=Path,
        default=DEFAULT_BATCH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to qa-tool/output/peer-compare-probe-full.json or -e2e.json.",
    )
    parser.add_argument("--ids", nargs="*", default=None)
    parser.add_argument(
        "--save-full",
        action="store_true",
        help="Include full /answer or probe payloads under 'details'.",
    )
    args = parser.parse_args(argv)

    output = args.output or (
        DEFAULT_OUTPUT_E2E if args.mode == "e2e" else DEFAULT_OUTPUT_ROUTING
    )
    id_filter = {value.upper() for value in args.ids} if args.ids else None
    questions = load_questions(args.batch, ids=id_filter)

    summaries: list[dict[str, Any]] = []
    details: dict[str, Any] = {}

    if args.mode == "e2e":
        db_error = _preflight_database()
        if db_error is not None:
            print(
                json.dumps(
                    {
                        "status": "preflight_failed",
                        "reason": db_error,
                        "hint": (
                            "Postgres unavailable. Run --mode routing locally, "
                            "or set DATABASE_URL on the server."
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
        try:
            pipeline = AnswerPipeline.from_env()
        except AnswerPipelineError as error:
            print(
                json.dumps(
                    {
                        "status": "setup_failed",
                        "reason": error.reason,
                        "message": str(error),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
    else:
        try:
            routing_stack = _build_routing_stack()
        except RuntimeError as error:
            print(json.dumps({"status": "setup_failed", "message": str(error)}), file=sys.stderr)
            return 2

    total = len(questions)
    for index, (question_id, question) in enumerate(questions, start=1):
        print(f"[{index:02d}/{total:02d}] {question_id} ({args.mode})", flush=True)
        if args.mode == "routing":
            understanding, validator, planner, orchestrator = routing_stack
            probe = _run_routing_probe(
                question,
                understanding=understanding,
                validator=validator,
                planner=planner,
                orchestrator=orchestrator,
            )
            summary = _summarize_routing(question_id, question, probe)
            if args.save_full:
                details[question_id] = probe
        else:
            try:
                payload = pipeline.answer(question_id, question)
                summary = _summarize_e2e(question_id, question, payload)
                if args.save_full:
                    details[question_id] = payload
            except AnswerPipelineError as error:
                summary = _summarize_e2e(
                    question_id, question, None, error=error.reason
                )
        summaries.append(summary)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_reference": PIPELINE_REFERENCE,
        "mode": args.mode,
        "question_count": len(summaries),
        "batch_path": str(args.batch),
        "summary": _aggregate(summaries),
        "rows": summaries,
    }
    if args.save_full:
        report["details"] = details

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "mode": args.mode,
                "question_count": report["question_count"],
                "summary": report["summary"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
