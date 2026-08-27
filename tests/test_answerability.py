from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.reasoning.answerability import (
    AnswerabilityGuard,
    AnswerabilityStatus,
    contains_categorical_negative,
    guarded_answer_text,
)
from app.reasoning.multi_document_evidence import (
    LIFECYCLE_NONE,
    LIFECYCLE_UNDETERMINED,
    MultiDocumentEvidence,
    MultiDocumentFacts,
)
from app.reasoning.query_plan import QueryPlan


def _generated(*, answerable: bool, citations: int = 1):
    return SimpleNamespace(
        answerable=answerable,
        citations=tuple(object() for _ in range(citations)),
        answer_text="확인된 사실입니다.",
    )


def _execution(count: int = 1):
    return SimpleNamespace(results=tuple(object() for _ in range(count)))


class AnswerabilityGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.guard = AnswerabilityGuard()

    def test_complete_citable_evidence_is_answerable(self) -> None:
        result = self.guard.evaluate(
            _generated(answerable=True), execution=_execution()
        )

        self.assertIs(result.status, AnswerabilityStatus.ANSWERABLE)

    def test_complete_zero_domain_is_not_found(self) -> None:
        multi = MultiDocumentEvidence(
            facts=MultiDocumentFacts(
                plan_type="enumeration",
                complete=True,
                logical_count=0,
            )
        )

        result = self.guard.evaluate(
            _generated(answerable=True),
            execution=_execution(0),
            multi_document=multi,
        )

        self.assertIs(result.status, AnswerabilityStatus.NOT_FOUND)
        answer = guarded_answer_text(result, "", multi_document=multi)
        self.assertTrue(contains_categorical_negative(answer))

    def test_complete_lifecycle_with_zero_terminations_is_not_found(self) -> None:
        multi = MultiDocumentEvidence(
            facts=MultiDocumentFacts(
                plan_type="enumeration_plus_event",
                complete=True,
                logical_count=13,
                lifecycle_answer=LIFECYCLE_NONE,
            )
        )

        result = self.guard.evaluate(
            _generated(answerable=True), execution=_execution(13), multi_document=multi
        )

        self.assertIs(result.status, AnswerabilityStatus.NOT_FOUND)

    def test_incomplete_enumeration_is_insufficient_not_not_found(self) -> None:
        multi = MultiDocumentEvidence(
            facts=MultiDocumentFacts(
                plan_type="enumeration",
                complete=False,
                logical_count=8,
                unresolved_count=0,
            )
        )

        result = self.guard.evaluate(
            _generated(answerable=True), execution=_execution(8), multi_document=multi
        )

        self.assertIs(result.status, AnswerabilityStatus.INSUFFICIENT_EVIDENCE)
        answer = guarded_answer_text(result, "없습니다.", multi_document=multi)
        self.assertFalse(contains_categorical_negative(answer))

    def test_unresolved_event_is_insufficient(self) -> None:
        multi = MultiDocumentEvidence(
            facts=MultiDocumentFacts(
                plan_type="enumeration_plus_event",
                complete=False,
                logical_count=13,
                unresolved_count=5,
                lifecycle_answer=LIFECYCLE_UNDETERMINED,
            )
        )

        result = self.guard.evaluate(
            _generated(answerable=False), execution=_execution(8), multi_document=multi
        )

        self.assertIs(result.status, AnswerabilityStatus.INSUFFICIENT_EVIDENCE)

    def test_some_supported_fields_produce_partial_answer(self) -> None:
        resolution = SimpleNamespace(
            requested_fields=("contract_amount", "termination_reason"),
            unresolved_fields=("termination_reason",),
        )
        agent = SimpleNamespace(resolution=resolution)

        result = self.guard.evaluate(
            _generated(answerable=False, citations=1),
            agent_result=agent,
            execution=_execution(),
        )

        self.assertIs(result.status, AnswerabilityStatus.PARTIALLY_ANSWERABLE)
        self.assertEqual(result.confirmed_fields, ("contract_amount",))
        self.assertEqual(result.missing_fields, ("termination_reason",))
        self.assertIn("일부 요청 항목", guarded_answer_text(result, "계약금액은 10억원입니다."))

    def test_missing_citation_is_insufficient_even_with_retrieved_chunks(self) -> None:
        result = self.guard.evaluate(
            _generated(answerable=True, citations=0), execution=_execution(5)
        )

        self.assertIs(result.status, AnswerabilityStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.reason, "citation_capable_evidence_missing")

    def test_single_document_field_coverage_can_be_partial(self) -> None:
        plan = QueryPlan(
            query="계약금액 해지 사유",
            raw_query="삼성중공업 계약금액과 해지 사유를 알려줘",
            company="삼성중공업",
            task_type="corporate_event",
            event_type="contract_termination",
        )
        generated = _generated(answerable=True, citations=1)
        generated.answer_text = "계약금액은 10억원입니다."

        result = self.guard.evaluate(
            generated,
            plan=plan,
            execution=_execution(),
        )

        self.assertIs(result.status, AnswerabilityStatus.PARTIALLY_ANSWERABLE)
        self.assertEqual(result.confirmed_fields, ("contract_amount",))
        self.assertEqual(result.missing_fields, ("termination_reason",))

    def test_citable_periodic_evidence_is_not_relevant_to_contract_event(self) -> None:
        plan = QueryPlan(
            query="2025년 공급계약",
            raw_query="삼성중공업의 2025년 공급계약을 알려줘",
            company="삼성중공업",
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
        )
        generated = SimpleNamespace(
            answerable=True,
            citations=(SimpleNamespace(chunk_id="periodic:accounting"),),
            answer_text="중요한 회계정책에 관한 내용입니다.",
        )
        execution = SimpleNamespace(
            results=(object(),),
            chunks=(
                SimpleNamespace(
                    chunk_id="periodic:accounting",
                    chunk={
                        "doc_group": "periodic",
                        "section_path": ["중요한 회계정책"],
                    },
                ),
            ),
        )

        result = self.guard.evaluate(generated, plan=plan, execution=execution)

        self.assertTrue(result.citable)
        self.assertFalse(result.relevant_to_request)
        self.assertFalse(result.answerable)
        self.assertIs(result.status, AnswerabilityStatus.INSUFFICIENT_EVIDENCE)
        self.assertEqual(result.reason, "evidence_semantic_mismatch")
        self.assertEqual(
            guarded_answer_text(result, generated.answer_text),
            "현재 확보된 공시 근거만으로는 해당 내용을 확인하기 어렵습니다.",
        )

    def test_cited_event_route_is_relevant_and_answerable(self) -> None:
        plan = QueryPlan(
            query="2025년 공급계약",
            company="삼성중공업",
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
        )
        generated = SimpleNamespace(
            answerable=True,
            citations=(SimpleNamespace(chunk_id="exchange:contract"),),
            answer_text="공급계약 공시 내용입니다.",
        )
        execution = SimpleNamespace(
            results=(object(),),
            chunks=(
                SimpleNamespace(
                    chunk_id="exchange:contract",
                    chunk={"doc_group": "exchange"},
                ),
            ),
        )

        result = self.guard.evaluate(generated, plan=plan, execution=execution)

        self.assertTrue(result.citable)
        self.assertTrue(result.relevant_to_request)
        self.assertTrue(result.answerable)
        self.assertIs(result.status, AnswerabilityStatus.ANSWERABLE)


if __name__ == "__main__":
    unittest.main()
