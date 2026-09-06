"""The sentence stating a figure may not state a unit the row does not have.

A 손익계산서 writes "기본주당이익 (단위 : 원)" on the EPS row while the statement
above it is in 백만원. Reading the first 단위 anywhere in the chunk attached 원 to
매출액 and reported a figure a million times too small -- a wrong answer that
looked entirely confident. These fix where a unit may come from, and when a
figure is unambiguous enough to state at all.
"""

from __future__ import annotations

import unittest

from decimal import Decimal

from app.generation.answer_generator import (
    GeneratedSection,
    StatedFigure,
    _comparison_line,
    _decimal_amount,
    _single_metric_cell,
    _stated_figure,
    _stated_sections,
    _stated_unit,
    _subject_particle,
    _topic_particle,
)


#: A 연결 손익계산서 as it is chunked: no table-level unit tag, and a row that
#: carries its own unit because it differs from the rest.
MIXED_UNITS = (
    "| 열 1 | 제 57 기 |\n"
    "| --- | --- |\n"
    "| 매출액 (주30) | 333,605,938 |\n"
    "| 기본주당이익 (단위 : 원) | 6,605 |"
)

TAGGED = (
    "[단위] 백만원\n"
    "| 열 1 | 제 57 기 |\n"
    "| --- | --- |\n"
    "| 매출액 | 333,605,938 |"
)


def _source(fact_text: str) -> dict:
    return {
        "fact_text": fact_text,
        "reporting_period": {"base_year": 2025, "base_month": 12},
        "section_path": ["III. 재무에 관한 사항", "2-2. 연결 손익계산서"],
    }


def _line(fact_text: str, metric: str = "매출액") -> str | None:
    figure = _stated_figure(
        {"corp_name": "삼성전자"},
        _source(fact_text),
        "[1]",
        request={"metric": metric, "basis": "consolidated"},
    )
    return None if figure is None else figure.sentence


class StatedUnitTests(unittest.TestCase):
    def test_only_the_table_tag_is_a_table_unit(self) -> None:
        self.assertEqual(_stated_unit(_source(TAGGED)), "백만원")

    def test_a_row_unit_is_not_read_as_the_table_unit(self) -> None:
        self.assertIsNone(_stated_unit(_source(MIXED_UNITS)))


class SingleMetricCellTests(unittest.TestCase):
    def test_one_row_and_one_column_give_a_figure(self) -> None:
        cell = _single_metric_cell(
            "| 열 1 | 제 57 기 |\n| --- | --- |\n| 매출액 (주30) | 333,605,938 |"
        )

        self.assertEqual(cell, ("매출액", "333,605,938", None))

    def test_a_row_carries_its_own_unit(self) -> None:
        cell = _single_metric_cell(
            "| 열 1 | 제 57 기 |\n| --- | --- |\n| 기본주당이익 (단위 : 원) | 6,605 |"
        )

        self.assertEqual(cell, ("기본주당이익", "6,605", "원"))

    def test_two_columns_are_not_one_figure(self) -> None:
        """3개월 and 누적 are both true, and neither is the answer on its own."""

        self.assertIsNone(
            _single_metric_cell(
                "| 열 1 | 제 78 기 3개월 | 제 78 기 누적 |\n"
                "| --- | --- | --- |\n"
                "| 영업이익 | 9,212,851 | 16,653,355 |"
            )
        )

    def test_statement_numbering_is_not_part_of_the_label(self) -> None:
        cell = _single_metric_cell(
            "| 열 1 | 제 24 기 |\n| --- | --- |\n| Ⅵ.당기순이익 | 4,558,170 |"
        )

        self.assertEqual(cell, ("당기순이익", "4,558,170", None))

    def test_prose_in_the_value_column_is_not_a_figure(self) -> None:
        self.assertIsNone(
            _single_metric_cell(
                "| 열 1 | 제 1 기 |\n| --- | --- |\n| 설명 | 해당사항 없음 |"
            )
        )


class DirectAnswerLineTests(unittest.TestCase):
    def test_another_rows_unit_never_reaches_this_row(self) -> None:
        line = _line(MIXED_UNITS)

        self.assertIsNotNone(line)
        self.assertIn("333,605,938", line)
        self.assertNotIn("333,605,938원", line)
        self.assertIn("단위 표기 없음", line)

    def test_a_table_unit_is_stated(self) -> None:
        line = _line(TAGGED)

        self.assertIn("333,605,938백만원", line)
        self.assertNotIn("단위 표기 없음", line)

    def test_the_sentence_names_company_period_and_basis(self) -> None:
        line = _line(TAGGED)

        self.assertTrue(line.startswith("삼성전자의 2025년 연결기준 매출액은"))
        self.assertTrue(line.endswith("[1]"))

    def test_consecutive_fiscal_terms_read_the_current_one(self) -> None:
        """제 57 기 beside 제 56 기 is this year and last, not an ambiguity."""

        line = _line(
            "| 열 1 | 제 57 기 | 제 56 기 |\n"
            "| --- | --- | --- |\n"
            "| 매출액 | 333,605,938 | 300,870,903 |"
        )

        self.assertIn("333,605,938", line)
        self.assertNotIn("300,870,903", line)

    def test_a_duration_split_states_nothing(self) -> None:
        """3개월 beside 누적 are both true and neither is the answer alone."""

        self.assertIsNone(
            _line(
                "| 열 1 | 제 78 기 반기 / 3 개월 | 제 78 기 반기 / 누적 |\n"
                "| --- | --- | --- |\n"
                "| 매출액 | 9,212,851 | 16,653,355 |"
            )
        )


def _figure(
    company: str,
    value: str,
    *,
    unit: str | None = "백만원",
    label: str = "당기순이익",
    period: str = "2024년",
    basis: str | None = "연결",
    marker: str = "[1]",
) -> StatedFigure:
    return StatedFigure(
        company=company,
        period=period,
        basis=basis,
        label=label,
        value=value,
        unit=unit,
        marker=marker,
        amount=_decimal_amount(value),
    )


class AmountTests(unittest.TestCase):
    def test_thousands_separators_come_off(self) -> None:
        self.assertEqual(_decimal_amount("5,028,606"), Decimal("5028606"))

    def test_parentheses_are_a_negative(self) -> None:
        self.assertEqual(_decimal_amount("(1,234)"), Decimal("-1234"))

    def test_a_ratio_keeps_its_magnitude(self) -> None:
        self.assertEqual(_decimal_amount("42.5%"), Decimal("42.5"))

    def test_prose_is_not_an_amount(self) -> None:
        self.assertIsNone(_decimal_amount("해당사항 없음"))


class ComparisonTests(unittest.TestCase):
    """The question asks which is larger, so the answer works it out in code."""

    def test_the_larger_side_and_the_gap_are_stated(self) -> None:
        line = _comparison_line(
            [
                _figure("KB금융", "5,028,606"),
                _figure("신한지주", "4,558,170", marker="[2]"),
            ]
        )

        self.assertIn("KB금융이 신한지주보다", line)
        self.assertIn("470,436백만원 더 큽니다", line)
        self.assertIn("[1] [2]", line)

    def test_the_order_of_the_two_does_not_change_the_winner(self) -> None:
        pair = [
            _figure("신한지주", "4,558,170"),
            _figure("KB금융", "5,028,606", marker="[2]"),
        ]

        self.assertIn("KB금융이 신한지주보다", _comparison_line(pair))

    def test_equal_figures_are_reported_equal(self) -> None:
        line = _comparison_line(
            [_figure("A전자", "100"), _figure("B전자", "100", marker="[2]")]
        )

        self.assertIn("같습니다", line)
        self.assertNotIn("큽니다", line)

    def test_different_units_are_not_compared(self) -> None:
        """매출액 in 원 against 매출액 in 백만원 is not a comparison."""

        self.assertIsNone(
            _comparison_line(
                [
                    _figure("A전자", "100", unit="원"),
                    _figure("B전자", "100", unit="백만원", marker="[2]"),
                ]
            )
        )

    def test_different_periods_are_not_compared(self) -> None:
        self.assertIsNone(
            _comparison_line(
                [
                    _figure("A전자", "100", period="2023년"),
                    _figure("B전자", "200", period="2024년", marker="[2]"),
                ]
            )
        )

    def test_different_measures_are_not_compared(self) -> None:
        self.assertIsNone(
            _comparison_line(
                [
                    _figure("A전자", "100", label="매출액"),
                    _figure("B전자", "200", label="영업이익", marker="[2]"),
                ]
            )
        )

    def test_one_company_is_not_a_comparison(self) -> None:
        self.assertIsNone(_comparison_line([_figure("A전자", "100")]))

    def test_the_same_company_twice_is_not_a_comparison(self) -> None:
        self.assertIsNone(
            _comparison_line(
                [_figure("A전자", "100"), _figure("A전자", "200", marker="[2]")]
            )
        )

    def test_a_negative_compares_below_a_positive(self) -> None:
        line = _comparison_line(
            [
                _figure("흑자사", "1,000"),
                _figure("적자사", "(2,000)", marker="[2]"),
            ]
        )

        self.assertIn("흑자사가 적자사보다", line)
        self.assertIn("3,000백만원", line)

    def test_unknown_units_are_not_compared(self) -> None:
        """삼성전기 wrote 원 and LG이노텍 wrote 백만원, and neither said so.

        Comparing the digits made the smaller company the larger one by a
        factor of a million. Two unknown units are two unknowns, not a match.
        """

        line = _comparison_line(
            [
                _figure("A전자", "10,294,102,976,435", unit=None),
                _figure("B전자", "21,200,755", unit=None, marker="[2]"),
            ]
        )

        self.assertNotIn("더 큽니다", line)
        self.assertIn("단위 표기가 없어", line)
        self.assertIn("[1] [2]", line)

    def test_a_stated_unit_is_carried_into_the_gap(self) -> None:
        line = _comparison_line(
            [
                _figure("A전자", "300", unit="백만원"),
                _figure("B전자", "100", unit="백만원", marker="[2]"),
            ]
        )

        self.assertIn("200백만원 더 큽니다", line)


class SubjectParticleTests(unittest.TestCase):
    def test_a_closed_syllable_takes_i(self) -> None:
        self.assertEqual(_subject_particle("KB금융"), "이")

    def test_an_open_syllable_takes_ga(self) -> None:
        self.assertEqual(_subject_particle("삼성전자"), "가")


class StatedSectionsTests(unittest.TestCase):
    """``answer`` is the answer; the documents are ``retrieved_context``."""

    def _sections(self, *titles: str) -> list[GeneratedSection]:
        return [
            GeneratedSection(title=title, content=f"{title} 본문", citations=())
            for title in titles
        ]

    def test_evidence_blocks_go_once_the_answer_is_stated(self) -> None:
        kept = _stated_sections(
            self._sections("답변", "Periodic fact 1", "Periodic fact 2", "주의")
        )

        self.assertEqual([section.title for section in kept], ["답변", "주의"])

    def test_the_confidence_line_never_reaches_the_answer(self) -> None:
        """It answers nothing, and it is the generator grading itself."""

        for sections in (
            self._sections("답변", "신뢰도"),
            self._sections("General evidence", "신뢰도"),
        ):
            with self.subTest(titles=[s.title for s in sections]):
                kept = _stated_sections(sections)
                self.assertNotIn("신뢰도", [s.title for s in kept])

    def test_general_and_holding_blocks_go_too(self) -> None:
        kept = _stated_sections(
            self._sections("답변", "General evidence", "Holding events")
        )

        self.assertEqual([section.title for section in kept], ["답변"])

    def test_a_limitation_notice_stays(self) -> None:
        """정보한계 대응 is scored, and it is not evidence."""

        kept = _stated_sections(
            self._sections("답변", "Periodic fact 1", "주의", "확인 필요")
        )

        self.assertEqual(
            [section.title for section in kept], ["답변", "주의", "확인 필요"]
        )

    def test_evidence_stays_without_an_answer(self) -> None:
        titles = ("Periodic fact 1", "General evidence", "주의")

        kept = _stated_sections(self._sections(*titles))

        self.assertEqual([section.title for section in kept], list(titles))


class ParticleTests(unittest.TestCase):
    def test_a_closed_syllable_takes_eun(self) -> None:
        self.assertEqual(_topic_particle("매출액"), "은")

    def test_an_open_syllable_takes_neun(self) -> None:
        self.assertEqual(_topic_particle("부채"), "는")

    def test_a_latin_label_takes_neun(self) -> None:
        self.assertEqual(_topic_particle("EBITDA"), "는")


if __name__ == "__main__":
    unittest.main()
