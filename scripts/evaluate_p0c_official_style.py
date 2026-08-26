"""Read-only P0-C official-style evaluation on a real PostgreSQL corpus.

The database connection is forced into ``default_transaction_read_only`` mode.
No embedding provider or HCX call is used: an empty frozen-retrieval stub lets
the production P0-C planner, executor, hydration, composer, generator, final
semantic guard, and five-field response contract run without external systems.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.pipeline import AnswerPipeline
from app.api.settings import ApiSettings
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer
from app.reasoning.multi_document_evidence import MultiDocumentEvidenceBuilder
from app.reasoning.multi_document_executor import MultiDocumentExecutor
from app.reasoning.multi_document_plan import SlotType
from app.reasoning.multi_document_planner import MultiDocumentPlanner
from app.reasoning.multi_document_semantics import check_answer
from app.reasoning.query_understanding import QueryUnderstanding
from app.retrieval.corporate_event_repository import (
    PostgresCorporateEventRepository,
)
from app.retrieval.correction_repository import PostgresCorrectionRepository
from app.retrieval.postgres_backend import PostgresBackend


DEFAULT_CASES = PROJECT_ROOT / "data/evaluation/p0c_official_style_v1.json"
SUMMARY_TITLE = "계약 확인 결과"
RESPONSE_FIELDS = {
    "question_id", "question", "retrieved_context", "think_trace", "answer"
}


class _AuditBackend(PostgresBackend):
    def __init__(self, dsn: str | None) -> None:
        super().__init__(
            dsn,
            connect_kwargs={"options": "-c default_transaction_read_only=on"},
        )
        self.raw_enumeration_count: int | None = None

    def enumerate_disclosures(self, **kwargs):
        rows = super().enumerate_disclosures(**kwargs)
        self.raw_enumeration_count = len(rows)
        return rows


class _EmptyRetrieval:
    """Keep P0-C real while avoiding embedding or unrelated retrieval calls."""

    def execute(self, plan):
        return SimpleNamespace(
            plan=plan,
            documents=(),
            chunks=(),
            results=(),
            correction_expansion={},
            event_expansion={},
        )


class _CapturingUnderstanding(QueryUnderstanding):
    last_plan: Any = None

    def understand(self, *args, **kwargs):
        self.last_plan = super().understand(*args, **kwargs)
        return self.last_plan


class _CapturingPlanner(MultiDocumentPlanner):
    last_plan: Any = None

    def plan(self, *args, **kwargs):
        self.last_plan = super().plan(*args, **kwargs)
        return self.last_plan


class _CapturingExecutor(MultiDocumentExecutor):
    last_execution: Any = None

    def execute(self, plan):
        self.last_execution = super().execute(plan)
        return self.last_execution


class _CapturingEvidence(MultiDocumentEvidenceBuilder):
    last_evidence: Any = None

    def build(self, *args, **kwargs):
        self.last_evidence = super().build(*args, **kwargs)
        return self.last_evidence


class _CapturingGenerator(CitationAwareAnswerGenerator):
    last_generated: Any = None

    def generate(self, draft):
        self.last_generated = super().generate(draft)
        return self.last_generated


def _pipeline(dsn: str | None):
    backend = _AuditBackend(dsn)
    events = PostgresCorporateEventRepository(backend)
    corrections = PostgresCorrectionRepository(backend)
    understanding = _CapturingUnderstanding(company_resolver=backend.resolve_company)
    planner = _CapturingPlanner()
    executor = _CapturingExecutor(
        event_repository=events,
        correction_repository=corrections,
        disclosure_backend=backend,
    )
    evidence = _CapturingEvidence(
        event_repository=events,
        metadata_backend=backend,
        chunk_backend=backend,
        retriever=backend,
    )
    generator = _CapturingGenerator()
    pipeline = AnswerPipeline(
        settings=ApiSettings(),
        understanding=understanding,
        executor=_EmptyRetrieval(),
        generator=generator,
        verbalizer=HcxVerbalizer(HcxSettings(enabled=False)),
        multi_document_planner=planner,
        multi_document_executor=executor,
        multi_document_evidence=evidence,
    )
    return pipeline, backend, understanding, planner, executor, evidence, generator


def _expected(case: Mapping[str, Any], key: str, actual: Any, failures: list[str]) -> None:
    expected = case.get(key)
    if expected is not None and actual != expected:
        failures.append(f"{key}: expected {expected!r}, got {actual!r}")


def _slot(plan: Any, slot_type: SlotType) -> Any:
    return next((slot for slot in plan.slots if slot.slot_type is slot_type), None)


def _citation_audit(generated: Any, evidence: Any, builder: Any) -> dict[str, int]:
    summary = next(
        (section for section in generated.sections if section.title == SUMMARY_TITLE),
        None,
    )
    cited_ids = set(summary.citations if summary is not None else ())
    cited_docs = {
        citation.doc_id
        for citation in generated.citations
        if citation.citation_id in cited_ids
    }
    execution = evidence.execution
    lifecycle = next(
        (slot for slot in execution.plan.slots if slot.slot_type is SlotType.EVENT_STATE),
        None,
    )
    outcome = execution.outcome(lifecycle.slot_id) if lifecycle is not None else None
    terminated_ids = tuple(outcome.terminated_ids if outcome is not None else ())
    opening_docs = {
        execution.opening_documents[event_id]
        for event_id in terminated_ids
        if event_id in execution.opening_documents
    }
    termination_docs = set(
        builder._termination_documents(execution.plan, terminated_ids)
    )
    unresolved_ids = {
        str(value)
        for slot in execution.plan.slots
        for value in slot.unresolved_ids
    }
    document_members = {
        str(doc_id)
        for doc_ids in execution.document_ids.values()
        for doc_id in doc_ids
    }
    unresolved_docs = {
        execution.opening_documents.get(event_id, event_id)
        for event_id in unresolved_ids
        if event_id in execution.opening_documents or event_id in document_members
    }
    return {
        "termination_expected": len(termination_docs),
        "termination_cited": len(termination_docs & cited_docs),
        "opening_expected": len(opening_docs),
        "opening_cited": len(opening_docs & cited_docs),
        "unresolved_available": len(unresolved_docs),
        "unresolved_cited": len(unresolved_docs & cited_docs),
    }
def _evaluate(case: Mapping[str, Any], components) -> dict[str, Any]:
    pipeline, backend, understanding, planner, executor, evidence, generator = components
    backend.raw_enumeration_count = None
    planner.last_plan = None
    executor.last_execution = None
    evidence.last_evidence = None
    generator.last_generated = None

    payload = pipeline.answer(case["case_id"], case["question"])
    plan = planner.last_plan
    failures: list[str] = []
    applied = bool(plan and plan.applied)
    actual_plan_type = plan.plan_type if applied else "not_applicable"
    _expected(case, "expected_plan_type", actual_plan_type, failures)
    if set(payload) != RESPONSE_FIELDS:
        failures.append(f"response fields: {sorted(payload)}")

    facts = evidence.last_evidence.facts if evidence.last_evidence is not None else None
    actual = {
        "logical_count": getattr(facts, "logical_count", None),
        "terminated_count": getattr(facts, "terminated_count", None),
        "unresolved_count": getattr(facts, "unresolved_count", None),
        "lifecycle_answer": getattr(facts, "lifecycle_answer", None),
        "complete": getattr(facts, "complete", None),
    }
    for key in (
        "logical_count", "terminated_count", "unresolved_count",
        "lifecycle_answer", "complete",
    ):
        _expected(case, f"expected_{key}", actual[key], failures)

    if case.get("expected_lifecycle_answer") is None and applied:
        if actual["lifecycle_answer"] is not None:
            failures.append("unexpected lifecycle answer")
    if case.get("expected_family_resolution") is not None:
        _expected(
            case, "expected_family_resolution", plan.family_resolution, failures
        )

    enumeration = None
    if applied:
        enumeration = next(
            (
                slot for slot in plan.slots
                if slot.slot_type in (
                    SlotType.ENUMERATE_EVENTS, SlotType.ENUMERATE_DOCUMENTS
                )
            ),
            None,
        )
    if case.get("expected_tier") is not None and enumeration is not None:
        tier = (
            "tier1" if enumeration.slot_type is SlotType.ENUMERATE_EVENTS else "tier2"
        )
        _expected(case, "expected_tier", tier, failures)
    if case.get("expected_date_field") is not None and enumeration is not None:
        _expected(case, "expected_date_field", enumeration.date_field, failures)
    if case.get("raw_receipt_filings") is not None:
        if backend.raw_enumeration_count is None and enumeration is not None:
            # Tier 1 enumerates logical events.  Derive the separate raw filing
            # audit count through the public disclosure enumeration primitive.
            backend.enumerate_disclosures(
                corp_code=enumeration.corp_code,
                doc_group="exchange",
                doc_subtype="단일판매공급계약체결",
                date_from=enumeration.date_from,
                date_to=enumeration.date_to,
            )
        _expected(
            case, "raw_receipt_filings", backend.raw_enumeration_count, failures
        )

    semantic = None
    state = actual["lifecycle_answer"]
    if state is not None:
        verdict = check_answer(state, payload["answer"])
        semantic = verdict.ok
        if not verdict.ok:
            failures.append(f"semantic: {verdict.reason}")

    trace_text = json.dumps(payload["think_trace"], ensure_ascii=False)
    for forbidden in ("expected_ids", "found_ids", "terminated_ids", "open_ids"):
        if forbidden in trace_text:
            failures.append(f"identifier field leaked: {forbidden}")
    if executor.last_execution is not None:
        for event_id in executor.last_execution.opening_documents:
            if event_id in trace_text:
                failures.append("event id leaked into trace")
                break
    for internal in ("bare_contract_fallback", "family_resolution", "slot_id"):
        if internal in payload["answer"]:
            failures.append(f"internal wording leaked: {internal}")
    if facts is not None and "확인되지 않은 정보가 있습니다" in payload["answer"]:
        failures.append("generic retrieval fallback appended to deterministic answer")

    citation = {
        "termination_expected": 0, "termination_cited": 0,
        "opening_expected": 0, "opening_cited": 0,
        "unresolved_available": 0, "unresolved_cited": 0,
    }
    if evidence.last_evidence is not None and generator.last_generated is not None:
        citation = _citation_audit(
            generator.last_generated, evidence.last_evidence, evidence
        )
        if citation["termination_cited"] != citation["termination_expected"]:
            failures.append("termination citation incomplete")
        if citation["opening_cited"] != citation["opening_expected"]:
            failures.append("opening citation incomplete")
        if (
            state == "undetermined"
            and citation["unresolved_available"]
            and citation["unresolved_cited"] != citation["unresolved_available"]
        ):
            failures.append("unresolved provenance citation incomplete")

    checked = None
    if executor.last_execution is not None:
        lifecycle = _slot(executor.last_execution.plan, SlotType.EVENT_STATE)
        source = lifecycle or next(iter(executor.last_execution.plan.slots), None)
        checked = source.found_count if source is not None else 0
    return {
        "case_id": case["case_id"],
        "group": case["group"],
        "question": case["question"],
        "plan_type": actual_plan_type,
        **actual,
        "checked_count": checked,
        "semantic": semantic,
        "citation": citation,
        "answer_first_line": payload["answer"].splitlines()[1]
        if len(payload["answer"].splitlines()) > 1 else payload["answer"],
        "failures": failures,
    }


def _render(rows: list[dict[str, Any]]) -> str:
    lines = [
        "P0-C OFFICIAL-STYLE EVALUATION (read-only real corpus)",
        "=" * 112,
        f"{'case':<5} {'grp':<3} {'plan':<24} {'n':>4} {'term':>4} {'unr':>4} "
        f"{'chk':>4} {'cmpl':<5} {'lifecycle':<12} {'sem':<4} failures",
        "-" * 112,
    ]
    for row in rows:
        value = lambda key: "-" if row[key] is None else str(row[key])
        lines.append(
            f"{row['case_id']:<5} {row['group']:<3} {row['plan_type']:<24} "
            f"{value('logical_count'):>4} {value('terminated_count'):>4} "
            f"{value('unresolved_count'):>4} {value('checked_count'):>4} "
            f"{value('complete'):<5} {value('lifecycle_answer'):<12} "
            f"{('-' if row['semantic'] is None else 'OK' if row['semantic'] else 'FAIL'):<4} "
            f"{'; '.join(row['failures']) or '-'}"
        )

    plan_ok = sum(not any("expected_plan_type" in item for item in row["failures"])
                  for row in rows)
    lifecycle_rows = [row for row in rows if row["lifecycle_answer"] is not None]
    positive_supply = [
        row for row in rows
        if row["group"] == "B" and row["lifecycle_answer"] == "exists"
        and row["semantic"] and not row["failures"]
    ]
    termination_expected = sum(row["citation"]["termination_expected"] for row in rows)
    termination_cited = sum(row["citation"]["termination_cited"] for row in rows)
    opening_expected = sum(row["citation"]["opening_expected"] for row in rows)
    opening_cited = sum(row["citation"]["opening_cited"] for row in rows)
    unresolved_available = sum(
        row["citation"]["unresolved_available"] for row in rows
    )
    unresolved_cited = sum(row["citation"]["unresolved_cited"] for row in rows)
    checked = sum(row["checked_count"] or 0 for row in rows)
    logical = sum(row["logical_count"] or 0 for row in rows)
    failures = sum(len(row["failures"]) for row in rows)
    lines.extend(
        [
            "", "=" * 112,
            f"plan accuracy                  : {plan_ok}/{len(rows)}",
            f"lifecycle semantic accuracy   : "
            f"{sum(row['semantic'] is True for row in lifecycle_rows)}/{len(lifecycle_rows)}",
            f"real supply-positive cases    : {len(positive_supply)}",
            f"retrieval members checked     : {checked}/{logical}",
            f"termination claims cited      : {termination_cited}/{termination_expected}",
            f"opening identity claims cited : {opening_cited}/{opening_expected}",
            f"unresolved provenance cited   : {unresolved_cited}/{unresolved_available}",
            f"failures                       : {failures}",
            "HCX                           : unavailable/not invoked; deterministic fallback validated",
            "", "Real supply-positive cases:",
        ]
    )
    for row in positive_supply:
        citation = row["citation"]
        lines.append(
            f"- {row['case_id']} {row['question']} "
            f"(logical={row['logical_count']}, terminated={row['terminated_count']}, "
            f"termination citations={citation['termination_cited']}/"
            f"{citation['termination_expected']}, opening citations="
            f"{citation['opening_cited']}/{citation['opening_expected']})"
        )
    unresolved = next(
        (row for row in rows if row["lifecycle_answer"] == "undetermined"), None
    )
    if unresolved is not None:
        lines.extend(
            [
                "", "Real unresolved release gate:",
                f"- {unresolved['case_id']} {unresolved['question']} "
                f"(logical={unresolved['logical_count']}, checked="
                f"{unresolved['checked_count']}, unresolved="
                f"{unresolved['unresolved_count']}, complete="
                f"{unresolved['complete']}, semantic="
                f"{'OK' if unresolved['semantic'] else 'FAIL'}, tier2=true, "
                f"unresolved citations="
                f"{unresolved['citation']['unresolved_cited']}/"
                f"{unresolved['citation']['unresolved_available']})",
            ]
        )
    lines.extend(["", "Rendered semantic core:"])
    for state in ("exists", "none", "undetermined", "no_members"):
        row = next((item for item in rows if item["lifecycle_answer"] == state), None)
        if row is not None:
            lines.append(f"{state:<12} {row['answer_first_line']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--dsn", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    components = _pipeline(args.dsn)
    rows = [_evaluate(case, components) for case in cases]
    report = _render(rows)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    if any(row["failures"] for row in rows):
        raise SystemExit(1)
    if sum(
        row["group"] == "B" and row["lifecycle_answer"] == "exists"
        for row in rows
    ) < 2:
        raise SystemExit("fewer than two real supply-positive cases")


if __name__ == "__main__":
    main()
