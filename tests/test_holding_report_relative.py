import unittest

from app.reasoning import holding_report_relative as rr
from app.reasoning.holding_report_relative import (
    ROLE_CHANGE,
    ROLE_CURRENT,
    ROLE_PREVIOUS,
    SELECTOR_EXACT_RECEIPT_DATE,
    SELECTOR_EXACT_REFERENCE_DATE,
    SELECTOR_LATEST,
    SELECTOR_SELECTED_CONTEXT,
)
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.query_validation import CorpusScope


def _understanding() -> QueryUnderstanding:
    scope = CorpusScope.repository_default()
    assert scope is not None
    return QueryUnderstanding(scope.company_aliases())


class ReportRelativeParsingTests(unittest.TestCase):
    """Which report a holding question means, and which of its fields.

    Companies appear only as data.  Nothing in the parser may branch on a
    company, a question id, a document id, or an expected value.
    """

    #: A corpus company, used only so the question routes to holding.
    CO = "에스엠"

    def intent(self, query: str):
        plan = _understanding().understand(query)
        return plan.evidence["holding_report_relative"]

    # ------------------------------------------------------- selector reading
    def test_latest_wording_asks_for_an_ordering_over_reports(self) -> None:
        for query in (
            f"{self.CO} 최신 보고 보유주식수",
            f"{self.CO} 최신 보고서 보유비율",
            f"{self.CO} 최근 보고서 보유비율",
        ):
            with self.subTest(query=query):
                intent = self.intent(query)
                self.assertEqual(intent["selector"], SELECTOR_LATEST)
                self.assertEqual(intent["projection_role"], ROLE_CURRENT)

    def test_unanchored_now_is_the_same_ordering_as_latest(self) -> None:
        """"현재" with no date asks for the newest known state, not today's."""

        for query in (f"{self.CO} 현재 보유비율", f"{self.CO} 현재 기준 보유주식수"):
            with self.subTest(query=query):
                self.assertEqual(self.intent(query)["selector"], SELECTOR_LATEST)

    def test_this_report_is_deictic_and_never_becomes_latest(self) -> None:
        """"이번 보고" points at a report the question does not name."""

        for query in (
            f"{self.CO} 이번 보고 보유주식수",
            f"{self.CO} 이번보고서 보유비율",
        ):
            with self.subTest(query=query):
                intent = self.intent(query)
                self.assertEqual(intent["selector"], SELECTOR_SELECTED_CONTEXT)
                self.assertNotEqual(intent["selector"], SELECTOR_LATEST)
                self.assertEqual(intent["projection_role"], ROLE_CURRENT)

    def test_an_explicit_reference_date_names_its_report(self) -> None:
        for query in (
            f"{self.CO} 2024년 3월 14일 보고서 기준 보유비율",
            f"{self.CO} 2024년 3월 14일 현재 보유비율",
        ):
            with self.subTest(query=query):
                intent = self.intent(query)
                self.assertEqual(intent["selector"], SELECTOR_EXACT_REFERENCE_DATE)
                self.assertTrue(intent["executable"])
                self.assertFalse(intent["dynamic"])

    def test_receipt_wording_stays_on_the_receipt_axis(self) -> None:
        """A filing's arrival date is not the date its holdings are stated for."""

        for query in (
            f"{self.CO} 2024년 3월 14일 공시 보유비율",
            f"{self.CO} 2024년 3월 14일 접수 보유비율",
            f"{self.CO} 2024년 3월 14일 제출 보유비율",
        ):
            with self.subTest(query=query):
                self.assertEqual(
                    self.intent(query)["selector"], SELECTOR_EXACT_RECEIPT_DATE
                )

    # ------------------------------------------------- projection role reading
    def test_previous_wording_asks_for_the_previous_fields(self) -> None:
        for query in (
            f"{self.CO} 직전보고 보유비율",
            f"{self.CO} 직전 보고 보유비율",
            f"{self.CO} 이전 보고 보유비율",
            f"{self.CO} 이전 보고서 보유비율",
        ):
            with self.subTest(query=query):
                self.assertEqual(
                    self.intent(query)["projection_role"], ROLE_PREVIOUS
                )

    def test_this_report_and_previous_report_are_not_the_same_question(self) -> None:
        """The distinction the old representation could not express.

        Both phrasings used to produce one undifferentiated label, so a system
        reading only that label answered them identically.
        """

        this_report = self.intent(f"{self.CO} 이번 보고 보유비율")
        previous_report = self.intent(f"{self.CO} 직전보고 보유비율")

        self.assertEqual(this_report["projection_role"], ROLE_CURRENT)
        self.assertEqual(previous_report["projection_role"], ROLE_PREVIOUS)
        self.assertNotEqual(
            this_report["projection_role"], previous_report["projection_role"]
        )

    def test_against_the_previous_report_asks_for_the_difference(self) -> None:
        for query in (
            f"{self.CO} 직전 보고 대비 증감 주식수",
            f"{self.CO} 직전 보고 대비 지분율 차이",
        ):
            with self.subTest(query=query):
                self.assertEqual(self.intent(query)["projection_role"], ROLE_CHANGE)

    # ------------------------------------------ the two axes stay independent
    def test_selector_and_role_are_read_separately(self) -> None:
        """``최신 보고의 직전보고 비율`` selects the latest report and reads its
        previous ratio.  The previous ratio is a field of the latest filing, not
        a different filing."""

        intent = self.intent(f"{self.CO} 최신 보고의 직전보고 보유비율")

        self.assertEqual(intent["selector"], SELECTOR_LATEST)
        self.assertEqual(intent["projection_role"], ROLE_PREVIOUS)

    def test_an_anchored_previous_question_is_executable(self) -> None:
        intent = self.intent(f"{self.CO} 2024년 3월 14일 보고서의 직전보고 보유비율")

        self.assertEqual(intent["selector"], SELECTOR_EXACT_REFERENCE_DATE)
        self.assertEqual(intent["projection_role"], ROLE_PREVIOUS)
        self.assertTrue(intent["executable"])

    # ------------------------------------------------------------ fail closed
    def test_unprovable_selectors_are_marked_not_executable(self) -> None:
        """Latest cannot be proven without reporter-aware enumeration.

        Marking it declares the gap.  Selecting a plausible filing instead would
        answer about a report nobody named.
        """

        for query in (
            f"{self.CO} 최신 보고 보유주식수",
            f"{self.CO} 현재 보유비율",
            f"{self.CO} 이번 보고 보유주식수",
            f"{self.CO} 직전보고 보유비율",
        ):
            with self.subTest(query=query):
                self.assertFalse(self.intent(query)["executable"])

    def test_latest_is_dynamic_and_an_explicit_date_is_not(self) -> None:
        """A "latest" answer moves when the corpus gains a newer filing."""

        self.assertTrue(self.intent(f"{self.CO} 최신 보고 보유주식수")["dynamic"])
        self.assertFalse(
            self.intent(f"{self.CO} 2024년 3월 14일 보고서 기준 보유비율")["dynamic"]
        )

    def test_a_question_saying_nothing_about_reports_gets_no_intent(self) -> None:
        """Absent must stay distinct from "current fields of an unnamed report"."""

        plan = _understanding().understand(f"{self.CO} 국민연금 보유 주식수")
        self.assertIsNone(plan.evidence["holding_report_relative"])

    def test_non_holding_questions_carry_no_report_intent(self) -> None:
        """The contract is about holding reports, so it stays on that route.

        These questions carry the same report wording and must still produce no
        intent: a periodic or event filing has no previous-report holding
        fields for the role to point at.
        """

        for query in (
            f"{self.CO} 2023년 사업보고서 매출액",
            f"{self.CO} 최신 보고서 매출액",
            f"{self.CO} 이번 보고서 유상증자 금액",
        ):
            with self.subTest(query=query):
                plan = _understanding().understand(query)
                self.assertNotIn("holding", plan.disclosure_route)
                self.assertIsNone(plan.evidence["holding_report_relative"])


class ReportRelativeUnitTests(unittest.TestCase):
    """The parser alone, with the period parser's decision supplied directly."""

    def test_receipt_role_wins_the_axis_for_an_exact_date(self) -> None:
        intent = rr.parse(
            "2024년 3월 14일 공시 보유비율",
            date_semantics={"role": "receipt"},
            has_exact_date=True,
        )
        self.assertEqual(intent.selector, SELECTOR_EXACT_RECEIPT_DATE)

    def test_reference_role_wins_the_axis_for_an_exact_date(self) -> None:
        intent = rr.parse(
            "2024년 3월 14일 기준 보유비율",
            date_semantics={"role": "holding_reference"},
            has_exact_date=True,
        )
        self.assertEqual(intent.selector, SELECTOR_EXACT_REFERENCE_DATE)

    def test_an_exact_date_outranks_a_relative_word(self) -> None:
        """The date names the report; the relative word names the fields."""

        intent = rr.parse(
            "2024년 3월 14일 보고서의 직전보고 보유비율",
            date_semantics={"role": "holding_reference"},
            has_exact_date=True,
        )
        self.assertEqual(intent.selector, SELECTOR_EXACT_REFERENCE_DATE)
        self.assertEqual(intent.projection_role, ROLE_PREVIOUS)

    def test_an_exact_date_of_unknown_axis_defaults_to_the_reference_axis(self) -> None:
        """A holding question's bare date states when the holding was held.

        Receipt semantics require receipt wording; without it the date is not
        promoted to the arrival axis, because doing so would answer a different
        question than the one asked.
        """

        for semantics in ({}, {"role": None}, None):
            with self.subTest(semantics=semantics):
                intent = rr.parse(
                    "2024년 3월 14일 직전보고 보유비율",
                    date_semantics=semantics,
                    has_exact_date=True,
                )
                self.assertEqual(intent.selector, SELECTOR_EXACT_REFERENCE_DATE)
                self.assertNotEqual(intent.selector, SELECTOR_EXACT_RECEIPT_DATE)
                self.assertEqual(intent.projection_role, ROLE_PREVIOUS)

    def test_no_report_wording_returns_none(self) -> None:
        self.assertIsNone(rr.parse("보유 주식수와 비율"))


if __name__ == "__main__":
    unittest.main()
