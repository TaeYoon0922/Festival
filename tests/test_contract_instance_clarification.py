"""A conflicted contract question gets asked which filing, not refused.

Measured on E01, "한미반도체 대만 장비 수주 계약금액": the guard returned
``requested_field_conflict`` and the served answer said the evidence was
insufficient. It was not -- 계약금액 appears twenty times across the served
chunks -- and what was missing was in the question, which names no filing.

The tests pin both halves. A conflicted, dateless question is handed the
filings that state the field; everything else is left where it was, because
turning an answered question into a question back is the worse failure.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.reasoning.clarification_request import EVENT_INSTANCE
from app.reasoning.contract_instance_clarification import (
    REASON,
    conflicting_field,
    contract_instance_clarification_request,
)


@dataclass(frozen=True)
class _Document:
    doc_id: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _Chunk:
    chunk_id: str
    doc_id: str
    chunk: Mapping[str, Any]


@dataclass(frozen=True)
class _Execution:
    documents: tuple[_Document, ...] = ()
    chunks: tuple[_Chunk, ...] = ()


@dataclass(frozen=True)
class _Answerability:
    unavailable_evidence: tuple[Mapping[str, Any], ...] = ()


def _conflict(field_name: str = "contract_amount") -> _Answerability:
    return _Answerability(
        ({"field": field_name, "status": "conflict", "authoritative": True},)
    )


def _execution(count: int = 3, content: str = "계약금액 1,234억원") -> _Execution:
    return _Execution(
        documents=tuple(
            _Document(
                f"exchange_{index}",
                {"rcept_dt": f"2025010{index}", "report_nm": "단일판매ㆍ공급계약체결"},
            )
            for index in range(1, count + 1)
        ),
        chunks=tuple(
            _Chunk(f"exchange_{index}:ch_a", f"exchange_{index}", {"content": content})
            for index in range(1, count + 1)
        ),
    )


QUESTION = "한미반도체 대만 장비 수주 계약금액"


class ConflictTests(unittest.TestCase):
    def test_one_conflicting_field_is_recognised(self) -> None:
        self.assertEqual(conflicting_field(_conflict()), "contract_amount")

    def test_a_field_the_module_has_no_wording_for_is_ignored(self) -> None:
        self.assertIsNone(conflicting_field(_conflict("investment_amount")))

    def test_two_conflicting_fields_are_not_one_choice(self) -> None:
        answerability = _Answerability(
            (
                {"field": "contract_amount", "status": "conflict"},
                {"field": "contract_period", "status": "conflict"},
            )
        )
        self.assertIsNone(conflicting_field(answerability))

    def test_an_unavailable_field_that_is_not_a_conflict_is_ignored(self) -> None:
        answerability = _Answerability(
            ({"field": "contract_amount", "status": "absent"},)
        )
        self.assertIsNone(conflicting_field(answerability))


class RequestTests(unittest.TestCase):
    def test_a_conflicted_question_is_asked_back(self) -> None:
        request = contract_instance_clarification_request(
            QUESTION, _execution(), _conflict()
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.reason, REASON)
        self.assertEqual(len(request.candidates), 3)
        self.assertTrue(
            all(c.semantic_type == EVENT_INSTANCE for c in request.candidates)
        )

    def test_each_candidate_is_bound_to_a_served_chunk(self) -> None:
        request = contract_instance_clarification_request(
            QUESTION, _execution(), _conflict()
        )
        for candidate in request.candidates:
            with self.subTest(candidate=candidate.id):
                self.assertIsNotNone(candidate.source)

    def test_the_filings_survive_the_classifier(self) -> None:
        request = contract_instance_clarification_request(
            QUESTION, _execution(), _conflict()
        )
        self.assertTrue(request.preserve_candidates)

    def test_the_label_names_the_filing_date(self) -> None:
        request = contract_instance_clarification_request(
            QUESTION, _execution(), _conflict()
        )
        self.assertEqual(
            request.candidates[0].label, "2025-01-01 단일판매ㆍ공급계약체결"
        )


class StayOutOfTheWayTests(unittest.TestCase):
    def test_a_question_that_named_a_date_is_left_alone(self) -> None:
        for question in (
            "한미반도체 2025년 1월 1일 공급계약 계약금액",
            "한미반도체 2025.01.01 계약금액",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    contract_instance_clarification_request(
                        question, _execution(), _conflict()
                    )
                )

    def test_no_conflict_means_nothing_to_ask(self) -> None:
        self.assertIsNone(
            contract_instance_clarification_request(
                QUESTION, _execution(), _Answerability()
            )
        )

    def test_a_filing_that_does_not_state_the_field_is_not_offered(self) -> None:
        # The list is the corpus's, not this module's: a chunk ranking highly
        # while saying nothing about the field is not a choice.
        self.assertIsNone(
            contract_instance_clarification_request(
                QUESTION, _execution(content="주주총회 소집 결의"), _conflict()
            )
        )

    def test_one_filing_is_not_a_conflict(self) -> None:
        self.assertIsNone(
            contract_instance_clarification_request(
                QUESTION, _execution(1), _conflict()
            )
        )

    def test_an_empty_execution_is_not_an_error(self) -> None:
        self.assertIsNone(
            contract_instance_clarification_request(
                QUESTION, _Execution(), _conflict()
            )
        )

    def test_one_filing_served_twice_is_still_one_choice(self) -> None:
        execution = _Execution(
            documents=(
                _Document(
                    "exchange_1",
                    {"rcept_dt": "20250101", "report_nm": "단일판매ㆍ공급계약체결"},
                ),
            ),
            chunks=(
                _Chunk("exchange_1:ch_a", "exchange_1", {"content": "계약금액 1억"}),
                _Chunk("exchange_1:ch_b", "exchange_1", {"content": "계약금액 1억"}),
            ),
        )
        self.assertIsNone(
            contract_instance_clarification_request(QUESTION, execution, _conflict())
        )


if __name__ == "__main__":
    unittest.main()
