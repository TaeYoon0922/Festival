from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from app.reasoning.clarification_candidates import (
    execution_clarification_request,
    validation_clarification_request,
)
from app.reasoning.clarification_request import (
    ClarificationCandidate,
    ClarificationRequest,
    ClarificationState,
    clarification_text,
)
from app.reasoning.clarification_resolver import ClarificationResolver
from app.reasoning.hcx_clarification_classifier import (
    HcxClarificationOutcome,
    HcxClarificationResult,
)
from app.reasoning.holding_company_role_resolution import (
    ROLE_PROVENANCE_KEY,
    ROLE_PROVENANCE_SOURCE,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.reasoning.query_validation import (
    Clarification,
    ClarificationOption,
    CorpusScope,
    QuerySlot,
    QuerySlotSource,
    QuerySlotStatus,
    QueryState,
    QueryValidator,
)


def _candidate(identifier: str, label: str | None = None) -> ClarificationCandidate:
    return ClarificationCandidate(
        identifier,
        label or f"항목 {identifier}",
        "metric",
        "synthetic_taxonomy",
        identifier,
    )


class _Classifier:
    def __init__(self, outcome: HcxClarificationOutcome) -> None:
        self.outcome = outcome
        self.calls = 0

    def classify(self, question, candidates):
        del question, candidates
        self.calls += 1
        return self.outcome


CORP_CODE_SCOPE = CorpusScope(
    companies={"테스트회사": ("테스트 회사", "00126380")},
    receipt_from="2020-01-01",
    receipt_to="2025-12-31",
    fiscal_years=(2024, 2025),
    event_from="2020-01-01",
    event_to="2025-12-31",
)


def _date_basis_validation():
    """A real validator ambiguity whose candidates carry explicit public labels."""

    plan = QueryPlan(
        query="계약",
        raw_query="테스트 회사 2024년 계약 알려줘",
        company="테스트 회사",
        task_type="disclosure_lookup",
        period=QueryPeriod(year=2024, period_type="reference_year"),
        disclosure_route=("exchange",),
        evidence={"operation": "lookup_disclosure"},
    )
    return plan, QueryValidator().validate(plan)


def _outcome(decision: str, ids=()) -> HcxClarificationOutcome:
    return HcxClarificationOutcome(
        HcxClarificationResult(decision, tuple(ids)),
        "success",
        1.0,
        transport_status="success",
        parse_status="success",
    )


class ClarificationStateMachineTests(unittest.TestCase):
    def test_zero_candidates_is_not_clarification(self) -> None:
        decision = ClarificationResolver().resolve(
            ClarificationRequest("무엇인가요?", ())
        )

        self.assertIs(decision.state, ClarificationState.INSUFFICIENT_EVIDENCE)
        self.assertNotEqual(decision.state, ClarificationState.CLARIFY)

    def test_fallback_state_cannot_claim_resolution_without_a_candidate(self) -> None:
        with self.assertRaises(ValueError):
            ClarificationRequest(
                "무엇인가요?",
                (),
                fallback_state=ClarificationState.RESOLVED,
            )

    def test_one_candidate_resolves_only_when_provider_declares_it_safe(self) -> None:
        candidate = _candidate("M1")
        safe = ClarificationResolver().resolve(
            ClarificationRequest(
                "어느 항목인가요?", (candidate,), single_candidate_safe=True
            )
        )
        unsafe = ClarificationResolver().resolve(
            ClarificationRequest("어느 항목인가요?", (candidate,))
        )

        self.assertIs(safe.state, ClarificationState.RESOLVED)
        self.assertEqual(safe.selected_candidate_id, "M1")
        self.assertIs(unsafe.state, ClarificationState.INSUFFICIENT_EVIDENCE)

    def test_two_candidates_clarify_without_hcx(self) -> None:
        candidates = (_candidate("M1", "금융비용"), _candidate("M2", "기타비용"))
        decision = ClarificationResolver().resolve(
            ClarificationRequest("영업외비용은?", candidates)
        )

        self.assertIs(decision.state, ClarificationState.CLARIFY)
        self.assertEqual(
            clarification_text(decision),
            "금융비용을 말씀하시는 건가요, 아니면 기타비용을 말씀하시는 건가요?",
        )

    def test_too_many_candidates_use_a_bounded_generic_question(self) -> None:
        candidates = tuple(_candidate(f"M{index}") for index in range(1, 6))
        decision = ClarificationResolver().resolve(
            ClarificationRequest("어느 항목인가요?", candidates)
        )
        text = clarification_text(decision)

        self.assertIn("여러 가능한 해석", text)
        self.assertNotIn("항목 M1", text)

    def test_classifier_resolution_is_used_only_when_provider_allows_it(self) -> None:
        candidates = (_candidate("M1"), _candidate("M2"))
        classifier = _Classifier(_outcome("resolved", ("M1",)))
        resolver = ClarificationResolver(classifier)

        safe = resolver.resolve(
            ClarificationRequest(
                "어느 항목인가요?",
                candidates,
                classifier_resolution_safe=True,
            )
        )
        unsafe = resolver.resolve(ClarificationRequest("어느 항목인가요?", candidates))

        self.assertIs(safe.state, ClarificationState.RESOLVED)
        self.assertEqual(safe.selected_candidate_id, "M1")
        self.assertIs(unsafe.state, ClarificationState.CLARIFY)

    def test_classifier_failure_falls_back_to_original_candidates(self) -> None:
        candidates = (_candidate("M1"), _candidate("M2"))
        failure = HcxClarificationOutcome(
            None,
            "malformed_response",
            1.0,
            transport_status="success",
            parse_status="invalid_json",
        )
        decision = ClarificationResolver(_Classifier(failure)).resolve(
            ClarificationRequest("어느 항목인가요?", candidates)
        )

        self.assertIs(decision.state, ClarificationState.CLARIFY)
        self.assertEqual([item.id for item in decision.candidates], ["M1", "M2"])
        self.assertEqual(decision.classifier_status, "malformed_response")

    def test_classifier_runtime_failures_fall_back_to_clarification(self) -> None:
        candidates = (_candidate("M1"), _candidate("M2"))
        for status in ("transport_failure", "classifier_error"):
            with self.subTest(status=status):
                failure = HcxClarificationOutcome(
                    None,
                    status,
                    1.0,
                    transport_status="failure",
                )
                decision = ClarificationResolver(_Classifier(failure)).resolve(
                    ClarificationRequest("어느 항목인가요?", candidates)
                )

                self.assertIs(decision.state, ClarificationState.CLARIFY)
                self.assertEqual(decision.classifier_status, status)
                self.assertEqual(decision.candidates, candidates)

    def test_legacy_classifier_unsupported_cannot_override_valid_candidates(self) -> None:
        candidates = (_candidate("M1"), _candidate("M2"))
        decision = ClarificationResolver(
            _Classifier(_outcome("unsupported", ()))
        ).resolve(ClarificationRequest("어느 항목인가요?", candidates))

        self.assertIs(decision.state, ClarificationState.CLARIFY)
        self.assertEqual(decision.candidates, candidates)


class ClarificationCandidateProviderTests(unittest.TestCase):
    def test_bounded_ownership_provenance_enables_holding_options(self) -> None:
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
        validation = QueryValidator().validate(plan)

        request = validation_clarification_request(plan.raw_query, validation)

        self.assertIs(validation.state, QueryState.INCOMPLETE)
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(
            [(item.value, item.label) for item in request.candidates],
            [("holding_shares", "보유주식수"), ("holding_ratio", "보유비율")],
        )
        self.assertTrue(request.classifier_resolution_safe)

    def test_holding_metric_without_bounded_ownership_provenance_is_ignored(self) -> None:
        plan = QueryPlan(
            query="보유자 이름",
            raw_query="테스트 회사의 보유자 이름은?",
            company="테스트 회사",
            task_type="holding_change",
            disclosure_route=("holding",),
            evidence={"operation": "lookup_holding"},
        )
        validation = QueryValidator().validate(plan)

        self.assertIs(validation.state, QueryState.INCOMPLETE)
        self.assertIsNone(
            validation_clarification_request(plan.raw_query, validation)
        )

    def test_explicit_holding_metrics_bypass_clarification(self) -> None:
        for metric in ("holding_shares", "holding_ratio"):
            with self.subTest(metric=metric):
                plan = QueryPlan(
                    query=metric,
                    raw_query="테스트 회사의 명시적 보유 지표는?",
                    company="테스트 회사",
                    reporter="테스트 투자자",
                    task_type="holding_change",
                    metric=metric,
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
                validation = QueryValidator().validate(plan)

                self.assertIs(validation.state, QueryState.RESOLVED)
                self.assertIsNone(
                    validation_clarification_request(plan.raw_query, validation)
                )

    def test_acquisition_and_comparison_semantics_are_not_intercepted(self) -> None:
        cases = (
            QueryPlan(
                query="취득 수량",
                raw_query="테스트 회사의 취득 수량은?",
                company="테스트 회사",
                task_type="holding_change",
                disclosure_route=("holding",),
                evidence={"operation": "lookup_holding"},
            ),
            QueryPlan(
                query="비교",
                raw_query="A와 B의 보유비율을 비교해줘",
                companies=("A", "B"),
                task_type="holding_change",
                disclosure_route=("holding",),
                comparison={"type": "company_comparison", "companies": ["A", "B"]},
                evidence={
                    "operation": "compare",
                    "comparison_frame": "cross_company",
                },
            ),
            QueryPlan(
                query="이번 보고 지분 얼마나",
                raw_query="이 회사의 이번 보고 기준 지분은 얼마나 돼?",
                company="테스트 회사",
                task_type="holding_change",
                disclosure_route=("holding",),
                evidence={
                    "operation": "lookup_holding",
                    "holding_report_relative": {
                        "selector": "selected_context",
                    },
                },
            ),
        )
        for plan in cases:
            with self.subTest(question=plan.raw_query):
                validation = QueryValidator().validate(plan)
                self.assertIsNone(
                    validation_clarification_request(plan.raw_query, validation)
                )

    def test_unresolved_holding_company_roles_take_precedence(self) -> None:
        plan = QueryPlan(
            query="지분 얼마나",
            raw_query="가상투자사가 가상발행사 주식을 얼마나 보유하고 있어?",
            companies=("가상투자사", "가상발행사"),
            task_type="holding_change",
            disclosure_route=("holding",),
            evidence={
                "operation": "lookup_holding",
                "comparison_frame": None,
            },
        )
        validation = QueryValidator().validate(plan)

        self.assertIs(validation.state, QueryState.AMBIGUOUS)
        self.assertEqual(
            validation.slots["company"].status.value,
            "ambiguous",
        )
        self.assertIsNone(
            validation_clarification_request(plan.raw_query, validation)
        )

    def test_periodic_rows_never_establish_candidate_eligibility(self) -> None:
        cases = (
            ("계약금액 알려줘", ("계약상대", "계약기간", "직원현황")),
            ("보고일 알려줘", ("보고자", "보고서", "매출액")),
            ("2024년 알려줘", ("2024년말", "2024년도", "매출액")),
            ("없는지표 알려줘", ("매출액", "자산총계", "직원현황")),
        )
        for question, labels in cases:
            with self.subTest(question=question):
                item = SimpleNamespace(
                    chunk_id="p1:table",
                    doc_group="periodic",
                    evidence_text="\n".join(
                        f"| {label} | 값 |" for label in labels
                    ),
                )
                result = SimpleNamespace(
                    resolution=None,
                    evidence_set=SimpleNamespace(
                        evidence_groups=(SimpleNamespace(items=(item,)),)
                    ),
                )
                plan = QueryPlan(
                    query=question,
                    raw_query=question,
                    company="테스트 회사",
                    task_type="disclosure_lookup",
                    disclosure_route=("periodic",),
                    evidence={"operation": "lookup_disclosure"},
                )

                self.assertIsNone(
                    execution_clarification_request(
                        question,
                        plan,
                        result,
                        SimpleNamespace(event_expansion={}, documents=()),
                    )
                )

    def test_event_candidates_use_distinct_corporate_event_graph_instances(self) -> None:
        plan = QueryPlan(
            query="공급계약 계약금액",
            raw_query="테스트 회사 공급계약 계약금액 알려줘",
            company="테스트 회사",
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
            evidence={"operation": "inspect_event"},
        )
        result = SimpleNamespace(resolution=None, evidence_set=None)
        execution = SimpleNamespace(
            event_expansion={
                "corporate_event_expansion": {
                    "events": [
                        {
                            "event_id": "evt_a",
                            "seed_doc_id": "ex_a",
                            "seed_member_doc_id": "ex_a",
                        },
                        {
                            "event_id": "evt_b",
                            "seed_doc_id": "ex_b",
                            "seed_member_doc_id": "ex_b",
                        },
                    ]
                }
            },
            documents=(
                SimpleNamespace(
                    doc_id="ex_a",
                    metadata={"rcept_dt": "20240102", "report_nm": "공급계약 A"},
                ),
                SimpleNamespace(
                    doc_id="ex_b",
                    metadata={"rcept_dt": "20240603", "report_nm": "공급계약 B"},
                ),
            ),
        )

        request = execution_clarification_request(
            plan.raw_query, plan, result, execution
        )

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual([item.value for item in request.candidates], ["evt_a", "evt_b"])
        self.assertIn("2024-01-02", request.candidates[0].label)
        self.assertTrue(
            all(item.provenance == "corporate_event_graph" for item in request.candidates)
        )

    def test_event_candidates_require_unique_public_metadata(self) -> None:
        plan = QueryPlan(
            query="공급계약 계약금액",
            raw_query="테스트 회사 공급계약 계약금액 알려줘",
            company="테스트 회사",
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
            evidence={"operation": "inspect_event"},
        )
        result = SimpleNamespace(resolution=None, evidence_set=None)
        events = [
            {"event_id": "evt_a", "seed_doc_id": "internal_a"},
            {"event_id": "evt_b", "seed_doc_id": "internal_b"},
        ]

        missing = SimpleNamespace(
            event_expansion={"corporate_event_expansion": {"events": events}},
            documents=(),
        )
        identical = SimpleNamespace(
            event_expansion={"corporate_event_expansion": {"events": events}},
            documents=(
                SimpleNamespace(
                    doc_id="internal_a",
                    metadata={"report_nm": "공급계약", "rcept_dt": "20240102"},
                ),
                SimpleNamespace(
                    doc_id="internal_b",
                    metadata={"report_nm": "공급계약", "rcept_dt": "20240102"},
                ),
            ),
        )

        self.assertIsNone(
            execution_clarification_request(plan.raw_query, plan, result, missing)
        )
        self.assertIsNone(
            execution_clarification_request(plan.raw_query, plan, result, identical)
        )

    def test_correction_members_of_one_event_do_not_create_choices(self) -> None:
        plan = QueryPlan(
            query="공급계약 계약금액",
            raw_query="테스트 회사 공급계약 계약금액 알려줘",
            company="테스트 회사",
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
            evidence={"operation": "inspect_event"},
        )
        result = SimpleNamespace(resolution=None, evidence_set=None)
        execution = SimpleNamespace(
            event_expansion={
                "corporate_event_expansion": {
                    "events": [
                        {"event_id": "evt_same", "seed_doc_id": "original"},
                        {"event_id": "evt_same", "seed_doc_id": "changed"},
                        {"event_id": "evt_same", "seed_doc_id": "terminated"},
                    ]
                }
            },
            documents=(),
        )

        self.assertIsNone(
            execution_clarification_request(plan.raw_query, plan, result, execution)
        )

    def test_unlabelled_slot_candidates_never_become_public_labels(self) -> None:
        corp_code_plan = QueryPlan(
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
        corp_code = QueryValidator(corpus_scope=CORP_CODE_SCOPE).validate(corp_code_plan)
        task_plan = QueryPlan(
            query="공급계약",
            raw_query="테스트 회사 공급계약 알려줘",
            company="테스트 회사",
            task_type="corporate_event",
            event_type="supply_contract",
            period=QueryPeriod(year=2025, period_type="fiscal_year"),
            disclosure_route=("exchange",),
            evidence={"operation": "inspect_event"},
        )
        task_conflict = QueryValidator().validate(
            task_plan, semantic={"task_type": "holding_change"}, fallback_used=True
        )

        # The slots really do carry machine values, so the assertions below are
        # about label safety rather than about an ambiguity that stopped existing.
        self.assertEqual(
            corp_code.slots["company"].candidates, ("00126380", "99999999")
        )
        self.assertEqual(
            task_conflict.slots["task_type"].candidates,
            ("corporate_event", "holding_change"),
        )
        self.assertIsNone(
            validation_clarification_request(corp_code_plan.raw_query, corp_code)
        )
        self.assertIsNone(
            validation_clarification_request(task_plan.raw_query, task_conflict)
        )

    def test_explicitly_labelled_slot_candidates_still_clarify(self) -> None:
        plan, validation = _date_basis_validation()

        request = validation_clarification_request(plan.raw_query, validation)

        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request.target_slot, "date_basis")
        self.assertEqual(
            [(item.value, item.label) for item in request.candidates],
            [
                ("contract_date", "계약 체결일 기준"),
                ("receipt_date", "공시 접수일 기준"),
            ],
        )
        for item in request.candidates:
            self.assertNotIn(item.value, item.label)
            self.assertNotIn("value", item.to_public_dict())

    def test_blank_option_labels_are_not_public_labels(self) -> None:
        plan, validation = _date_basis_validation()
        for blank in ("", "   ", "	"):
            with self.subTest(label=repr(blank)):
                blanked = replace(
                    validation,
                    clarification=Clarification(
                        validation.clarification.question,
                        (
                            ClarificationOption("contract_date", blank),
                            ClarificationOption("receipt_date", "공시 접수일 기준"),
                        ),
                    ),
                )

                self.assertIsNone(
                    validation_clarification_request(plan.raw_query, blanked)
                )

    def test_unlabelled_first_ambiguity_does_not_defer_to_a_later_slot(self) -> None:
        plan, validation = _date_basis_validation()
        slots = dict(validation.slots)
        slots["task_type"] = QuerySlot(
            "task_type",
            "disclosure_lookup",
            QuerySlotSource.DETERMINISTIC,
            QuerySlotStatus.AMBIGUOUS,
            True,
            ("corporate_event", "holding_change"),
        )
        spiked = replace(validation, slots=slots)

        self.assertEqual(spiked.ambiguous_slots[0], "task_type")
        self.assertIn("date_basis", spiked.ambiguous_slots)
        # date_basis is safely labelled, but answering it would move the question
        # off the first ambiguity the validator recorded.
        self.assertIsNone(validation_clarification_request(plan.raw_query, spiked))

if __name__ == "__main__":
    unittest.main()
