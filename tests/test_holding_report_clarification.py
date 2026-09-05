"""A question that did not name a filing gets asked which one, not guessed at.

Measured on the gold set: every holding question naming a date resolved, and
every one using "이번 보고" or "직전보고" without a date did not. Those are not
ranking failures. "이번 보고" selects a row inside a report -- 변동 전 against
변동 후 -- and the corpus holds seven filings for the pair the question named,
so nothing in the text says which. Retrieval cannot close that: the phrase is
in 52,869 of 91,387 holding chunks.

The tests below pin the two halves that matter. A question that left the filing
unnamed is handed back the filings it could have meant, and a question that did
name one is left alone -- because the cost of getting the second half wrong is
turning working answers into questions.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from app.reasoning.clarification_request import (
    HOLDING_REPORT_INSTANCE,
    ClarificationDecision,
    ClarificationState,
    clarification_text,
)
from app.reasoning.holding_report_clarification import (
    REASON,
    holding_report_clarification_request,
)


@dataclass(frozen=True)
class _Record:
    doc_id: str
    projection_chunk_id: str
    reference_date: str
    report_nm: str | None = "주식등의대량보유상황보고서"


@dataclass(frozen=True)
class _Plan:
    corp_code: str = "00126380"
    reporter: str = "하이브"


class _Index:
    """Enumerates one pair, the way the generated artifact does."""

    def __init__(self, records: tuple[_Record, ...]) -> None:
        self.records = records
        self.asked: list[tuple[str, str]] = []

    def enumerate_reports(self, corp_code: str, reporter: str):
        self.asked.append((corp_code, reporter))
        return self.records


def _index(count: int = 3) -> _Index:
    return _Index(
        tuple(
            _Record(
                doc_id=f"holding_{index}",
                projection_chunk_id=f"holding_{index}:ch_a",
                reference_date=f"2024030{index}",
            )
            for index in range(1, count + 1)
        )
    )


class RequestTests(unittest.TestCase):
    def test_a_dateless_relative_question_is_asked_back(self) -> None:
        request = holding_report_clarification_request(
            "에스엠 하이브 이번 보고 보유 주식수와 비율",
            _Plan(),
            report_index=_index(),
            answerable=False,
        )

        self.assertIsNotNone(request)
        self.assertEqual(request.reason, REASON)
        self.assertEqual(len(request.candidates), 3)
        self.assertTrue(
            all(
                candidate.semantic_type == HOLDING_REPORT_INSTANCE
                for candidate in request.candidates
            )
        )

    def test_the_previous_report_wording_is_asked_back_too(self) -> None:
        self.assertIsNotNone(
            holding_report_clarification_request(
                "에스엠 하이브 직전보고 보유주식 수 비율",
                _Plan(),
                report_index=_index(),
                answerable=False,
            )
        )

    def test_candidates_are_newest_first_and_carry_their_own_filing(self) -> None:
        request = holding_report_clarification_request(
            "에스엠 하이브 이번 보고 보유 주식수와 비율",
            _Plan(),
            report_index=_index(),
            answerable=False,
        )
        labels = [candidate.label for candidate in request.candidates]

        self.assertEqual(labels[0], "2024-03-03 기준 주식등의대량보유상황보고서")
        for candidate in request.candidates:
            with self.subTest(candidate=candidate.id):
                self.assertIsNotNone(candidate.source)


class StayOutOfTheWayTests(unittest.TestCase):
    """Turning an answered question into a question back is the worse failure."""

    def test_a_question_that_named_a_date_is_left_alone(self) -> None:
        for question in (
            "에스엠 하이브 2024년 3월 14일 현재 보유 수량 비율",
            "효성중공업 국민연금 2023.03.07 보유 수량",
            "파마리서치 국민연금 20221205 기준 보유 비율",
        ):
            with self.subTest(question=question):
                self.assertIsNone(
                    holding_report_clarification_request(
                        question, _Plan(), report_index=_index(), answerable=False
                    )
                )

    def test_an_answered_question_is_left_alone(self) -> None:
        self.assertIsNone(
            holding_report_clarification_request(
                "에스엠 하이브 이번 보고 보유 주식수와 비율",
                _Plan(),
                report_index=_index(),
                answerable=True,
            )
        )

    def test_a_question_without_relative_wording_is_left_alone(self) -> None:
        self.assertIsNone(
            holding_report_clarification_request(
                "에스엠 하이브 보유 주식수와 비율",
                _Plan(),
                report_index=_index(),
                answerable=False,
            )
        )

    def test_one_filing_is_not_a_choice(self) -> None:
        self.assertIsNone(
            holding_report_clarification_request(
                "에스엠 하이브 이번 보고 보유 주식수와 비율",
                _Plan(),
                report_index=_index(1),
                answerable=False,
            )
        )

    def test_a_missing_index_is_not_an_error(self) -> None:
        self.assertIsNone(
            holding_report_clarification_request(
                "에스엠 하이브 이번 보고 보유 주식수와 비율",
                _Plan(),
                report_index=None,
                answerable=False,
            )
        )

    def test_an_unnamed_holder_is_left_alone(self) -> None:
        # Without a holder there is no timeline to enumerate, and guessing one
        # would answer for a position nobody asked about.
        self.assertIsNone(
            holding_report_clarification_request(
                "에스엠 이번 보고 보유 주식수와 비율",
                _Plan(reporter=""),
                report_index=_index(),
                answerable=False,
            )
        )

    def test_an_index_that_raises_does_not_break_the_answer(self) -> None:
        class _Broken:
            def enumerate_reports(self, *args, **kwargs):
                raise RuntimeError("index unavailable")

        self.assertIsNone(
            holding_report_clarification_request(
                "에스엠 하이브 이번 보고 보유 주식수와 비율",
                _Plan(),
                report_index=_Broken(),
                answerable=False,
            )
        )


class TextTests(unittest.TestCase):
    def _decision(self, count: int) -> ClarificationDecision:
        request = holding_report_clarification_request(
            "에스엠 하이브 이번 보고 보유 주식수와 비율",
            _Plan(),
            report_index=_index(count),
            answerable=False,
        )
        return ClarificationDecision(
            state=ClarificationState.CLARIFY,
            reason=request.reason,
            candidates=request.candidates,
        )

    def test_it_says_it_did_not_choose_and_lists_the_filings(self) -> None:
        text = clarification_text(self._decision(3))

        self.assertIn("질문에 어느 보고서인지가 없어 하나를 고르지 않았습니다", text)
        self.assertIn("- 2024-03-03 기준", text)
        self.assertIn("어느 보고서 기준으로 확인할지 알려주세요", text)

    def test_more_filings_than_the_prose_limit_are_still_listed(self) -> None:
        # The three-candidate limit protects against a wall of sentences. These
        # labels are dates, so listing seven stays readable and is the only
        # form the asker can actually answer from.
        text = clarification_text(self._decision(7))

        self.assertEqual(text.count("\n- "), 7)
        self.assertNotIn("여러 가능한 해석이 확인되었습니다", text)


if __name__ == "__main__":
    unittest.main()
