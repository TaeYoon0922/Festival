"""The sentence stating a figure may not state a unit the row does not have.

A 손익계산서 writes "기본주당이익 (단위 : 원)" on the EPS row while the statement
above it is in 백만원. Reading the first 단위 anywhere in the chunk attached 원 to
매출액 and reported a figure a million times too small -- a wrong answer that
looked entirely confident. These fix where a unit may come from, and when a
figure is unambiguous enough to state at all.
"""

from __future__ import annotations

import unittest

from app.generation.answer_generator import (
    _direct_answer_line,
    _single_metric_cell,
    _stated_unit,
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
    return _direct_answer_line(
        {"corp_name": "삼성전자"},
        _source(fact_text),
        "[1]",
        request={"metric": metric, "basis": "consolidated"},
    )


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

    def test_an_ambiguous_projection_states_nothing(self) -> None:
        line = _line(
            "| 열 1 | 제 57 기 | 제 56 기 |\n"
            "| --- | --- | --- |\n"
            "| 매출액 | 333,605,938 | 300,870,903 |"
        )

        self.assertIsNone(line)


class ParticleTests(unittest.TestCase):
    def test_a_closed_syllable_takes_eun(self) -> None:
        self.assertEqual(_topic_particle("매출액"), "은")

    def test_an_open_syllable_takes_neun(self) -> None:
        self.assertEqual(_topic_particle("부채"), "는")

    def test_a_latin_label_takes_neun(self) -> None:
        self.assertEqual(_topic_particle("EBITDA"), "는")


if __name__ == "__main__":
    unittest.main()
