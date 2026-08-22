from app.reasoning.periodic_metric_view import (
    has_exact_metric_row,
    is_income_statement_section,
    project_periodic_metric_table,
)


_TABLE = """\
| 열 1 | 제 58 기 1분기 / 3개월 | 제 57 기 1분기 / 3개월 |
| --- | --- | --- |
| 매출액 | 44,407,761 | 40,658,539 |
| 매출원가 | 35,428,253 | 32,230,756 |
| 재화의 판매로 인한 수익(매출액) | 36,287,439 | 33,824,721 |
"""


def test_project_keeps_exact_metric_row_only() -> None:
    projected = project_periodic_metric_table(_TABLE, metric="매출액")
    assert projected is not None
    assert "44,407,761" in projected
    assert "매출액" in projected
    assert "매출원가" not in projected
    assert "재화의 판매" not in projected
    assert has_exact_metric_row(_TABLE, "매출액")


def test_project_returns_none_without_exact_row() -> None:
    text = "| 구분 | 금액 |\n| --- | --- |\n| 재화의 판매로 인한 수익(매출액) | 1 |"
    assert project_periodic_metric_table(text, metric="매출액") is None
    assert not has_exact_metric_row(text, "매출액")


def test_income_statement_section_markers() -> None:
    assert is_income_statement_section(["연결포괄손익계산서"])
    assert not is_income_statement_section(["수익의 구분"])
