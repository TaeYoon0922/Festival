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
from app.reasoning.clarification_candidates import (
    _protected_semantics,
    validation_clarification_request,
)
from app.reasoning.holding_company_role_resolution import (
    RESOLVED,
    HoldingCompanyRoleResolution,
    has_role_provenance,
)
from app.reasoning.holding_event_resolver import CURRENT_HOLDING_STATE_FIELDS
from app.reasoning.holding_evidence_coverage import requested_holding_fields
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.reasoning.query_understanding import understand_query
from app.reasoning.query_validation import (
    CorpusScope,
    QuerySlotStatus,
    QueryState,
    QueryValidator,
)
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

    def test_generic_current_possession_answers_without_clarification(self) -> None:
        """The current-state pair is an answer, so nothing is asked back."""

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

        # Superseded: this used to stop and offer 보유주식수 / 보유비율 as a
        # choice.  The deterministic stack now proves the question asks for the
        # current-state pair, so retrieval runs and clarification never engages.
        self.assertEqual(executor.calls, 1)
        self.assertNotEqual(payload["think_trace"]["route"], "clarification")
        self.assertNotIn("clarification", payload["think_trace"])
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
            query="Pfizer Ireland Pharmaceuticals 위탁생산계약 계약금액",
            raw_query=(
                "테스트 회사가 Pfizer Ireland Pharmaceuticals와 체결한 "
                "위탁생산계약의 계약금액은?"
            ),
            company="테스트 회사",
            corp_code="00000001",
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
                        "seed_corp_code": "00000001",
                        "event_family": "supply_contract",
                        "seed_identity": {
                            "counterparty": "Pfizer Ireland Pharmaceuticals",
                            "subject": "의약품 위탁생산계약",
                        },
                    }
                ],
                # An independent open contract has no related document to
                # expand.  The expansion layer therefore carries its graph
                # identity as a resolved singleton diagnostic rather than an
                # expanded lifecycle.
                "skipped": [
                    {
                        "event_id": "evt_b",
                        "seed_doc_id": "e2",
                        "seed_member_doc_id": "e2",
                        "reason": "lifecycle_not_resolved",
                        "resolution_status": "resolved",
                        "member_count": 1,
                        "seed_corp_code": "00000001",
                        "event_family": "supply_contract",
                        "seed_identity": {
                            "counterparty": "PFIZER IRELAND PHARMACEUTICALS",
                            "subject": "의약품 위탁생산계약",
                        },
                    }
                ],
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

    def test_correction_versions_of_one_logical_contract_do_not_clarify(self) -> None:
        plan = QueryPlan(
            query="상대회사 공급계약 계약금액",
            raw_query="테스트 회사가 상대회사와 체결한 공급계약의 계약금액은?",
            company="테스트 회사",
            corp_code="00000001",
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
            evidence={"operation": "inspect_event"},
        )
        events = {
            "corporate_event_expansion": {
                "events": [
                    {"event_id": "evt_same", "seed_member_doc_id": "original"},
                    {"event_id": "evt_same", "seed_member_doc_id": "correction"},
                ]
            }
        }
        pipeline, _executor = _pipeline(
            plan, _execution(plan, event_expansion=events)
        )

        payload = pipeline.answer("Q5-correction", plan.raw_query)

        self.assertNotEqual(payload["think_trace"]["route"], "clarification")
        self.assertNotIn("clarification", payload["think_trace"])

    def test_explicit_contract_date_keeps_normal_answer_path(self) -> None:
        plan = QueryPlan(
            query="상대회사 공급계약 계약금액",
            raw_query=(
                "테스트 회사가 2024년 1월 2일 공시한 상대회사 공급계약의 "
                "계약금액은?"
            ),
            company="테스트 회사",
            corp_code="00000001",
            task_type="corporate_event",
            event_type="supply_contract",
            period=QueryPeriod(
                from_date="2024-01-02",
                to_date="2024-01-02",
                period_type="event_date",
            ),
            disclosure_route=("exchange",),
            evidence={"operation": "inspect_event"},
        )
        events = {
            "corporate_event_expansion": {
                "events": [
                    {"event_id": "evt_a", "seed_member_doc_id": "e1"},
                    {"event_id": "evt_b", "seed_member_doc_id": "e2"},
                ]
            }
        }
        pipeline, _executor = _pipeline(
            plan, _execution(plan, event_expansion=events)
        )

        payload = pipeline.answer("Q5-date", plan.raw_query)

        self.assertNotEqual(payload["think_trace"]["route"], "clarification")
        self.assertNotIn("clarification", payload["think_trace"])

    def test_unique_contract_keeps_normal_answer_path(self) -> None:
        plan = QueryPlan(
            query="상대회사 공급계약 계약금액",
            raw_query="테스트 회사가 상대회사와 체결한 공급계약의 계약금액은?",
            company="테스트 회사",
            corp_code="00000001",
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
            evidence={"operation": "inspect_event"},
        )
        events = {
            "corporate_event_expansion": {
                "events": [{"event_id": "evt_only", "seed_member_doc_id": "e1"}]
            }
        }
        pipeline, _executor = _pipeline(
            plan, _execution(plan, event_expansion=events)
        )

        payload = pipeline.answer("Q5-unique", plan.raw_query)

        self.assertNotEqual(payload["think_trace"]["route"], "clarification")
        self.assertNotIn("clarification", payload["think_trace"])

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

#: Two corpus companies in a proven issuer/reporter relation.  Named for the
#: role each one plays, so no real filer or filing date is encoded here.
ISSUER, ISSUER_CODE = "가상발행사", "00000011"
REPORTER, REPORTER_CODE = "가상투자사", "00000022"
HOLDING_ALIASES = {ISSUER: [ISSUER], REPORTER: [REPORTER]}
HOLDING_SCOPE = CorpusScope(
    companies={ISSUER: (ISSUER, ISSUER_CODE), REPORTER: (REPORTER, REPORTER_CODE)},
    receipt_from="2020-01-01",
    receipt_to="2025-12-31",
    fiscal_years=tuple(range(2020, 2026)),
    event_from="2020-01-01",
    event_to="2025-12-31",
)


class _RoleResolver:
    """Stands in for the corpus-backed issuer/reporter role resolver."""

    def resolve(self, first, first_code, second, second_code):
        del first, first_code, second, second_code
        return HoldingCompanyRoleResolution(
            status=RESOLVED,
            issuer=ISSUER,
            issuer_corp_code=ISSUER_CODE,
            reporter=REPORTER,
            reporter_key=REPORTER,
            direction_1_report_count=3,
        )


class HoldingAcquisitionFirewallTests(unittest.TestCase):
    """Acquisition wording is never re-read as shares-vs-ratio ambiguity."""

    def _resolved_holding_validation(self, question):
        """Drive the real understanding and validation path for a question."""

        plan = understand_query(question, HOLDING_ALIASES)
        validator = QueryValidator(
            corpus_scope=HOLDING_SCOPE,
            holding_company_role_resolver=_RoleResolver(),
        )
        return plan, validator.validate(plan)

    def _answer(self, question, plan):
        pipeline, executor = _pipeline(
            plan,
            _execution(plan),
            classifier=_ExplodingClassifier(),
            validator=QueryValidator(
                corpus_scope=HOLDING_SCOPE,
                holding_company_role_resolver=_RoleResolver(),
            ),
        )
        return pipeline.answer("Q", question), executor

    def test_acquisition_unit_price_resolves_to_the_acquisition_metric(self) -> None:
        """Named outright, the field is answered rather than asked about."""

        for question in (
            f"{REPORTER}가 보유한 {ISSUER} 주식의 2024년 8월 4일 기준 취득 단가는?",
            f"{REPORTER}가 보유한 {ISSUER} 주식의 취득단가는?",
        ):
            with self.subTest(question=question):
                plan, validation = self._resolved_holding_validation(question)

                # Every bare-ownership ingredient is still present, so a pass
                # here cannot come from the ambiguity having disappeared.
                self.assertEqual(
                    validation.plan.evidence["holding_ownership_intent"],
                    "company_holds_company_shares",
                )
                self.assertTrue(has_role_provenance(validation.plan))
                self.assertIs(
                    validation.slots["company"].status, QuerySlotStatus.RESOLVED
                )
                # Superseded: the metric slot used to be held open so the
                # validator would ask which figure was meant.  The wording names
                # the field outright, so it resolves to the acquisition metric
                # and the field-evidence lane answers it.
                self.assertIs(
                    validation.slots["metric"].status, QuerySlotStatus.RESOLVED
                )
                self.assertEqual(validation.plan.metric, "acquisition_unit_price")
                # The firewall still holds: acquisition wording is never re-read
                # as a shares-or-ratio choice.
                self.assertTrue(_protected_semantics(question, validation.plan))
                self.assertIsNone(
                    validation_clarification_request(question, validation)
                )

                payload, executor = self._answer(question, plan)

                # Retrieval runs; nothing is asked back.
                self.assertEqual(executor.calls, 1)
                self.assertNotEqual(
                    payload["think_trace"]["route"], "clarification"
                )
                self.assertNotIn("clarification", payload["think_trace"])
                self.assertNotEqual(
                    payload["answer"], "확인하려는 재무 지표를 구체적으로 알려주세요."
                )
                for label in ("보유주식수", "보유비율"):
                    self.assertNotIn(label, payload["answer"])
                AnswerResponse.model_validate(payload)

    def test_bare_ownership_resolves_to_the_current_state_pair(self) -> None:
        """A possession question without acquisition wording still resolves."""

        for question in (
            f"{REPORTER}가 {ISSUER} 주식을 2023년 5월 9일 기준으로 얼마나 들고 있어?",
            f"{REPORTER}의 {ISSUER} 지분은 2025년 3월 24일 보고 기준 얼마나 되나?",
        ):
            with self.subTest(question=question):
                _plan, validation = self._resolved_holding_validation(question)

                # Still not acquisition wording -- the firewall is silent here.
                self.assertFalse(_protected_semantics(question, validation.plan))
                request = validation_clarification_request(question, validation)

                # Superseded: this used to offer the two metrics as a choice.
                # The current-state pair is now proved and returned whole.
                self.assertIs(validation.state, QueryState.RESOLVED)
                self.assertIsNone(request)
                self.assertEqual(
                    requested_holding_fields(question, validation.plan),
                    CURRENT_HOLDING_STATE_FIELDS,
                )

    def test_answerable_acquisition_fields_stay_protected(self) -> None:
        for question in (
            f"{REPORTER}가 보유한 {ISSUER} 주식의 취득일은?",
            f"{REPORTER}가 보유한 {ISSUER} 주식의 취득 주식수는?",
        ):
            with self.subTest(question=question):
                _plan, validation = self._resolved_holding_validation(question)

                self.assertTrue(_protected_semantics(question, validation.plan))
                self.assertIsNone(
                    validation_clarification_request(question, validation)
                )


if __name__ == "__main__":
    unittest.main()
