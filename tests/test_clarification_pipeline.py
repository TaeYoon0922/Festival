from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from app.api.pipeline import AnswerPipeline
from app.api.schemas import AnswerResponse
from app.api.settings import ApiSettings
from app.generation.hcx_verbalizer import HcxSettings, HcxVerbalizer
from app.reasoning.answerability import AnswerabilityGuard
from app.reasoning.clarification_resolver import ClarificationResolver
from app.reasoning.holding_company_role_resolution import (
    ROLE_PROVENANCE_KEY,
    ROLE_PROVENANCE_SOURCE,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.reasoning.query_validation import CorpusScope, QueryValidator
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


class _Understanding:
    def __init__(self, plan: QueryPlan) -> None:
        self.plan = plan

    def understand(self, question, *, top_k):
        del question, top_k
        return self.plan


class _Executor:
    def __init__(self, execution) -> None:
        self.execution = execution
        self.calls = 0

    def execute(self, plan):
        self.calls += 1
        self.execution.plan = plan
        return self.execution


class _ExplodingClassifier:
    def classify(self, question, candidates):
        raise AssertionError(f"classifier must be bypassed: {question!r} {candidates!r}")


class _SemanticFallback:
    """Stand-in for the pre-retrieval semantic slot fallback."""

    def __init__(self, result) -> None:
        self.result = result
        self.succeeded = True
        self.status = "success"
        self.elapsed_ms = 1.0
        self.calls = 0

    def interpret(self, question, validation):
        del question, validation
        self.calls += 1
        return self

    def diagnostic(self):
        return {}


CORP_CODE_SCOPE = CorpusScope(
    companies={"테스트회사": ("테스트 회사", "00126380")},
    receipt_from="2020-01-01",
    receipt_to="2025-12-31",
    fiscal_years=(2024, 2025),
    event_from="2020-01-01",
    event_to="2025-12-31",
)


def _chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    *,
    doc_group: str,
    rank: int,
    rcept_dt: str = "20250102",
    report_nm: str = "사업보고서",
) -> tuple[CandidateChunk, RetrievalResult]:
    payload = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_group": doc_group,
        "chunk_type": "table",
        "retrieval_text": text,
        "content": text,
        "corp_code": "00000001",
        "corp_name": "테스트 회사",
        "rcept_dt": rcept_dt,
        "report_nm": report_nm,
        "section_path": ["포괄손익계산서"],
        "period": {"fiscal_year": 2025, "period_type": "fiscal_year"},
        "source_refs": [{"table_id": doc_id, "row_start": 1, "row_end": 2}],
    }
    return (
        CandidateChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            chunk=payload,
            metadata_match=MetadataMatch(),
        ),
        RetrievalResult(
            chunk_id=chunk_id,
            doc_id=doc_id,
            bm25_score=1.0,
            rank=rank,
            metadata_match={},
        ),
    )


def _execution(plan, pairs=(), *, event_expansion=None, documents=()):
    return SimpleNamespace(
        plan=plan,
        documents=tuple(documents),
        chunks=tuple(pair[0] for pair in pairs),
        results=tuple(pair[1] for pair in pairs),
        routing={},
        correction_expansion={},
        event_expansion=dict(event_expansion or {}),
    )


def _pipeline(plan, execution, *, classifier=None, validator=None, semantic_fallback=None):
    executor = _Executor(execution)
    pipeline = AnswerPipeline(
        understanding=_Understanding(plan),
        executor=executor,
        settings=ApiSettings(top_k=10),
        verbalizer=HcxVerbalizer(HcxSettings(enabled=False)),
        query_validator=validator or QueryValidator(),
        semantic_fallback=semantic_fallback,
        answerability_guard=AnswerabilityGuard(),
        clarification_resolver=ClarificationResolver(classifier),
    )
    return pipeline, executor


class ClarificationPipelineTests(unittest.TestCase):
    def test_already_resolved_query_bypasses_clarification(self) -> None:
        plan = QueryPlan(
            query="매출액",
            raw_query="테스트 회사 2025년 매출액은?",
            company="테스트 회사",
            task_type="financial_metric",
            metric="매출액",
            period=QueryPeriod(year=2025, period_type="fiscal_year"),
            disclosure_route=("periodic",),
            evidence={"operation": "lookup_metric"},
        )
        pair = _chunk(
            "p1:c1",
            "p1",
            "| 매출액 | 1,000 |",
            doc_group="periodic",
            rank=1,
        )
        pipeline, executor = _pipeline(
            plan,
            _execution(plan, (pair,)),
            classifier=_ExplodingClassifier(),
        )

        payload = pipeline.answer("Q1", plan.raw_query)

        self.assertEqual(executor.calls, 1)
        self.assertNotIn("clarification", payload["think_trace"])
        self.assertNotEqual(payload["think_trace"]["route"], "clarification")

    def test_holding_metric_ambiguity_returns_bounded_clarification(self) -> None:
        plan = QueryPlan(
            query="지분 얼마나",
            raw_query="테스트 회사의 지분을 얼마나 가지고 있어?",
            company="테스트 회사",
            reporter="테스트 투자자",
            task_type="holding_change",
            disclosure_route=("holding",),
            evidence={
                "operation": "lookup_holding",
                "holding_ownership_intent": "company_has_company_shares",
                ROLE_PROVENANCE_KEY: {
                    "source": ROLE_PROVENANCE_SOURCE,
                    "resolved": True,
                },
            },
        )
        pipeline, executor = _pipeline(plan, _execution(plan))

        payload = pipeline.answer("Q2", plan.raw_query)

        self.assertEqual(executor.calls, 0)
        self.assertEqual(payload["think_trace"]["route"], "clarification")
        self.assertEqual(payload["think_trace"]["clarification"]["state"], "clarify")
        self.assertEqual(
            [item["label"] for item in payload["think_trace"]["clarification"]["candidates"]],
            ["보유주식수", "보유비율"],
        )
        self.assertIn("보유주식수", payload["answer"])
        self.assertIn("보유비율", payload["answer"])
        self.assertEqual(payload["retrieved_context"], [])
        self.assertEqual(
            set(payload),
            {"question_id", "question", "retrieved_context", "think_trace", "answer"},
        )
        AnswerResponse.model_validate(payload)

    def test_no_evidence_remains_insufficient_and_not_clarification(self) -> None:
        plan = QueryPlan(
            query="매출액",
            raw_query="테스트 회사 2025년 매출액은?",
            company="테스트 회사",
            task_type="financial_metric",
            metric="매출액",
            period=QueryPeriod(year=2025, period_type="fiscal_year"),
            disclosure_route=("periodic",),
            evidence={"operation": "lookup_metric"},
        )
        pipeline, _executor = _pipeline(plan, _execution(plan))

        payload = pipeline.answer("Q3", plan.raw_query)

        self.assertNotIn("clarification", payload["think_trace"])
        self.assertEqual(
            payload["think_trace"]["answerability"]["status"],
            "insufficient_evidence",
        )
        self.assertFalse(payload["think_trace"]["answerable"])

    def test_periodic_rows_do_not_create_clarification(self) -> None:
        plan = QueryPlan(
            query="영업외비용",
            raw_query="테스트 회사 2025년 사업보고서의 영업외비용은?",
            company="테스트 회사",
            task_type="disclosure_lookup",
            disclosure_route=("periodic",),
            evidence={"operation": "lookup_disclosure"},
        )
        pair = _chunk(
            "p2:c1",
            "p2",
            "| 금융비용 | 100 |\n| 기타비용 | 20 |\n| 매출액 | 999 |",
            doc_group="periodic",
            rank=1,
        )
        pipeline, _executor = _pipeline(plan, _execution(plan, (pair,)))

        payload = pipeline.answer("Q4", plan.raw_query)

        self.assertNotEqual(payload["think_trace"]["route"], "clarification")
        self.assertNotIn("clarification", payload["think_trace"])

    def test_multiple_event_instances_return_filing_clarification(self) -> None:
        plan = QueryPlan(
            query="공급계약 계약금액",
            raw_query="테스트 회사 공급계약 계약금액 알려줘",
            company="테스트 회사",
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
            evidence={"operation": "inspect_event"},
        )
        first = _chunk(
            "e1:c1",
            "e1",
            "| 계약금액 | 100 |",
            doc_group="exchange",
            rank=1,
            rcept_dt="20240102",
            report_nm="공급계약 A",
        )
        second = _chunk(
            "e2:c1",
            "e2",
            "| 계약금액 | 200 |",
            doc_group="exchange",
            rank=2,
            rcept_dt="20240603",
            report_nm="공급계약 B",
        )
        events = {
            "event_count": 2,
            "corporate_event_expansion": {
                "events": [
                    {
                        "event_id": "evt_a",
                        "seed_doc_id": "e1",
                        "seed_member_doc_id": "e1",
                    },
                    {
                        "event_id": "evt_b",
                        "seed_doc_id": "e2",
                        "seed_member_doc_id": "e2",
                    },
                ]
            },
        }
        documents = (
            CandidateDocument(
                "e1",
                {"rcept_dt": "20240102", "report_nm": "공급계약 A"},
                MetadataMatch(),
            ),
            CandidateDocument(
                "e2",
                {"rcept_dt": "20240603", "report_nm": "공급계약 B"},
                MetadataMatch(),
            ),
        )
        pipeline, _executor = _pipeline(
            plan,
            _execution(
                plan,
                (first, second),
                event_expansion=events,
                documents=documents,
            ),
        )

        payload = pipeline.answer("Q5", plan.raw_query)

        self.assertEqual(payload["think_trace"]["route"], "clarification")
        self.assertIn("공급계약 A", payload["answer"])
        self.assertIn("공급계약 B", payload["answer"])
        self.assertNotIn("100", payload["answer"])
        self.assertNotIn("200", payload["answer"])
        self.assertEqual(payload["retrieved_context"], [])
        self.assertNotIn("evt_a", payload["answer"])
        self.assertNotIn("evt_b", payload["answer"])
        self.assertTrue(
            all(
                "evt_" not in candidate["label"]
                and "e1" not in candidate["label"]
                and "e2" not in candidate["label"]
                for candidate in payload["think_trace"]["clarification"]["candidates"]
            )
        )

    def test_corp_code_conflict_keeps_the_validator_question(self) -> None:
        plan = QueryPlan(
            query="매출액",
            raw_query="테스트 회사 2025년 매출액은?",
            company="테스트 회사",
            corp_codes=("99999999",),
            task_type="financial_metric",
            metric="매출액",
            period=QueryPeriod(year=2025, period_type="fiscal_year"),
            disclosure_route=("periodic",),
            evidence={"operation": "lookup_metric"},
        )
        pipeline, executor = _pipeline(
            plan,
            _execution(plan),
            classifier=_ExplodingClassifier(),
            validator=QueryValidator(corpus_scope=CORP_CODE_SCOPE),
        )

        payload = pipeline.answer("Q6", plan.raw_query)

        self.assertEqual(executor.calls, 0)
        self.assertEqual(payload["think_trace"]["route"], "clarification")
        self.assertEqual(payload["answer"], "어느 회사에 대한 공시를 확인할까요?")
        self.assertNotIn("clarification", payload["think_trace"])
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        for internal in ("00126380", "99999999"):
            self.assertNotIn(internal, payload["answer"])
            self.assertNotIn(internal, serialized)
        AnswerResponse.model_validate(payload)

    def test_locked_task_type_conflict_keeps_the_validator_question(self) -> None:
        plan = QueryPlan(
            query="공급계약",
            raw_query="테스트 회사 공급계약 취소 알려줘",
            company="테스트 회사",
            task_type="corporate_event",
            event_type="supply_contract",
            period=QueryPeriod(year=2025, period_type="fiscal_year"),
            disclosure_route=("exchange",),
            evidence={},
        )
        fallback = _SemanticFallback({"task_type": "holding_change"})
        pipeline, executor = _pipeline(
            plan,
            _execution(plan),
            classifier=_ExplodingClassifier(),
            semantic_fallback=fallback,
        )

        payload = pipeline.answer("Q7", plan.raw_query)

        # The conflict is real: the same validation records the two internal
        # task_type values as its only ambiguity...
        conflicted = QueryValidator().validate(
            plan, semantic={"task_type": "holding_change"}, fallback_used=True
        )
        self.assertEqual(
            conflicted.slots["task_type"].candidates,
            ("corporate_event", "holding_change"),
        )
        # ...and validation did publish labels, but for a different slot.  Options
        # existing is not licence to print this slot's raw values.
        understanding = payload["think_trace"]["query_understanding"]
        self.assertEqual(
            [option["id"] for option in understanding["clarification"]["options"]],
            ["contract_termination", "investment_cancellation"],
        )
        self.assertEqual(fallback.calls, 1)
        self.assertEqual(executor.calls, 0)
        self.assertEqual(payload["think_trace"]["route"], "clarification")
        self.assertEqual(
            payload["answer"],
            "말씀하신 '취소'가 체결된 계약의 이후 해지를 뜻하나요, "
            "아니면 투자 계획의 철회를 뜻하나요?",
        )
        self.assertNotIn("clarification", payload["think_trace"])
        for internal in ("corporate_event", "holding_change"):
            self.assertNotIn(internal, payload["answer"])
        # ``corporate_event`` is this plan's real task type and legitimately
        # appears in the trace; the semantic rival never should.
        self.assertNotIn(
            "holding_change", json.dumps(payload, ensure_ascii=False, default=str)
        )
        AnswerResponse.model_validate(payload)

if __name__ == "__main__":
    unittest.main()
