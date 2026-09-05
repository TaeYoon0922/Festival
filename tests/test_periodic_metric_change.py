from __future__ import annotations

import copy
from dataclasses import replace
from decimal import Decimal

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import generate_answer
from app.reasoning.periodic_evidence_selector import PeriodicEvidenceSelector
from app.reasoning.periodic_fact_resolver import resolve_periodic_facts
from app.reasoning.periodic_metric_change import (
    PERIODIC_METRIC_CHANGE_KEY,
    PeriodicMetricChangeRequest,
    periodic_metric_change_claims,
    requested_periodic_metric_change,
    resolve_periodic_metric_change,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.reasoning.query_understanding import QueryUnderstanding
from tests.test_evidence_builder import _candidate
from tests.test_orchestrator import _execution
from tests.test_periodic_fact_resolver import _evidence, _group, _item


_COMPARISON_TABLE = """\
| 열 1 | 제 58 기 1분기 / 3개월 | 제 57 기 1분기 / 3개월 |
| --- | --- | --- |
| 매출액 | 44,407,761 | 40,658,539 |
| 영업이익 | 3,633,609 | 3,557,362 |
"""


def _change_plan(*, years: tuple[int, ...] = (2024, 2025)) -> dict:
    return {
        "company": "테스트회사",
        "companies": ["테스트회사"],
        "task_type": "financial_metric",
        "metric": "매출액",
        "basis": "consolidated",
        "period": {
            "year": None,
            "quarter": 1,
            "from": None,
            "to": None,
            "period_type": "fiscal_quarter",
        },
        "comparison": {"type": "year_over_year", "years": list(years)},
        "raw_query": "테스트회사 2025년 1분기 매출액 전년 동기 대비 증가율",
        "lexical_query": "매출액",
        "evidence": {
            PERIODIC_METRIC_CHANGE_KEY: {
                "metric": "매출액",
                "years": list(years),
                "comparison_type": "year_over_year",
            }
        },
    }


def _selected_resolution(
    *,
    table: str = _COMPARISON_TABLE,
    unit: str | None = "백만원",
    plan: dict | None = None,
):
    active_plan = copy.deepcopy(plan or _change_plan())
    item = _item(
        "p:ch_income",
        "p",
        rank=1,
        text=table,
        year=2025,
        quarter=1,
        section_path=("연결포괄손익계산서",),
        statement_scope="연결",
        temporal_match=False,
    )
    item = replace(
        item,
        period={"base_year": 2025, "base_month": 3, "statement_scope": "연결"},
    )
    if unit is not None:
        provenance = copy.deepcopy(dict(item.provenance))
        provenance["source_chunk"]["unit"] = unit
        item = replace(item, provenance=provenance)
    evidence = _evidence(
        [_group("income", item)],
        question=active_plan["raw_query"],
        task_type="financial_metric",
    )
    resolution = resolve_periodic_facts(evidence, query_plan=active_plan)
    selection = PeriodicEvidenceSelector().select(
        resolution, query_plan=active_plan
    )
    return active_plan, selection.resolution


def test_query_understanding_records_periodic_growth_and_removes_rate_noise() -> None:
    plan = QueryUnderstanding({"삼성전자": {"삼성전자"}}).understand(
        "삼성전자 2025년 1분기 매출액 전년 동기 대비 몇 퍼센트 증가했어?"
    )

    request = requested_periodic_metric_change(plan)

    assert plan.task_type == "financial_metric"
    assert plan.comparison == {"type": "year_over_year", "years": [2024, 2025]}
    assert request == PeriodicMetricChangeRequest(
        metric="매출액",
        years=(2024, 2025),
        comparison_type="year_over_year",
    )
    assert plan.lexical_query == "매출액"


def test_rate_wording_variants_do_not_pollute_metric_retrieval_query() -> None:
    understanding = QueryUnderstanding({"삼성전자": {"삼성전자"}})

    for question in (
        "삼성전자 2024년 대비 2025년 매출액 증가 퍼센트",
        "삼성전자 2025년 매출액 전년 대비 몇 % 증가",
        "삼성전자 2024년과 2025년 매출액 증감률은?",
    ):
        plan = understanding.understand(question)

        assert requested_periodic_metric_change(plan) is not None
        assert plan.lexical_query.endswith("매출액")
        assert not any(
            term in plan.lexical_query
            for term in ("증감률", "증가율", "퍼센트", "%", "증가")
        )


def test_exact_comparable_row_resolves_delta_and_growth_rate() -> None:
    plan, resolution = _selected_resolution()
    request = requested_periodic_metric_change(plan)

    change = resolve_periodic_metric_change(
        request,
        resolution,
        query_plan=plan,
    )

    assert change is not None
    assert change.initial.year == 2024
    assert change.initial.value == Decimal("40658539")
    assert change.final.year == 2025
    assert change.final.value == Decimal("44407761")
    assert change.difference == Decimal("3749222")
    assert change.pct_change == Decimal("9.22")
    assert change.unit == "백만원"
    assert change.initial.chunk_id == change.final.chunk_id == "p:ch_income"


def test_a_four_column_quarterly_row_resolves() -> None:
    """The shape Korean quarterly filings actually use.

    A 분기보고서 states the prior year beside the current one and splits each
    into 3개월 and 누적, so the row carries four period columns. Requiring
    exactly two left every such filing unresolved -- measured live on 삼성전자
    2025 1분기, where all four gates passed and the row was refused for its
    width alone.
    """

    table = """| 열 1 | 제 58 기 1분기 / 3개월 | 제 58 기 1분기 / 누적 | 제 57 기 1분기 / 3개월 | 제 57 기 1분기 / 누적 |
| --- | --- | --- | --- | --- |
| 매출액 (주26) | 44,407,761 | 44,407,761 | 40,658,539 | 40,658,539 |
"""
    plan, resolution = _selected_resolution(table=table)

    change = resolve_periodic_metric_change(
        requested_periodic_metric_change(plan), resolution, query_plan=plan
    )

    assert change is not None
    assert change.initial.year == 2024
    assert change.final.year == 2025
    assert change.pct_change == Decimal("9.22")


def test_disagreeing_period_windows_still_fail_closed() -> None:
    """Two candidate groups that differ are the mixing this module refuses.

    Past Q1 the 3개월 and 누적 windows carry different figures, and nothing in
    the question says which was meant, so neither is chosen.
    """

    table = """| 열 1 | 제 58 기 반기 / 3개월 | 제 58 기 반기 / 누적 | 제 57 기 반기 / 3개월 | 제 57 기 반기 / 누적 |
| --- | --- | --- | --- | --- |
| 매출액 | 22,000,000 | 44,407,761 | 20,000,000 | 40,658,539 |
"""
    plan, resolution = _selected_resolution(table=table)

    assert (
        resolve_periodic_metric_change(
            requested_periodic_metric_change(plan), resolution, query_plan=plan
        )
        is None
    )


def test_a_row_with_no_group_for_the_asked_years_fails_closed() -> None:
    # Three fiscal terms, none of them pairing into the requested two.
    table = """| 열 1 | 제 58 기 | 제 56 기 | 제 55 기 |
| --- | --- | --- | --- |
| 매출액 | 44,407,761 | 40,658,539 | 38,000,000 |
"""
    plan, resolution = _selected_resolution(table=table)

    assert (
        resolve_periodic_metric_change(
            requested_periodic_metric_change(plan), resolution, query_plan=plan
        )
        is None
    )


def test_serialized_claims_are_recomputed_before_rendering() -> None:
    plan, resolution = _selected_resolution()
    change = resolve_periodic_metric_change(
        requested_periodic_metric_change(plan),
        resolution,
        query_plan=plan,
    )

    claims = periodic_metric_change_claims(change.to_dict())
    tampered = change.to_dict()
    tampered["pct_change"] = "99.99"

    assert claims is not None
    assert claims[-2][0] == "증감액: +3,749,222백만원"
    assert claims[-1][0] == "증감률: +9.22%"
    assert periodic_metric_change_claims(tampered) is None


def test_missing_unit_fails_closed() -> None:
    plan, resolution = _selected_resolution(unit=None)

    change = resolve_periodic_metric_change(
        requested_periodic_metric_change(plan),
        resolution,
        query_plan=plan,
    )

    assert change is None


def test_mismatched_period_windows_fail_closed() -> None:
    table = """\
| 열 1 | 제 58 기 1분기 / 3개월 | 제 57 기 1분기 / 누적 |
| --- | --- | --- |
| 매출액 | 44,407,761 | 40,658,539 |
"""
    plan, resolution = _selected_resolution(table=table)

    change = resolve_periodic_metric_change(
        requested_periodic_metric_change(plan),
        resolution,
        query_plan=plan,
    )

    assert change is None


def test_non_positive_baseline_fails_closed() -> None:
    table = """\
| 열 1 | 제 58 기 | 제 57 기 |
| --- | --- | --- |
| 영업이익 | 100 | (50) |
"""
    plan = _change_plan()
    plan["metric"] = "영업이익"
    plan["lexical_query"] = "영업이익"
    plan["evidence"][PERIODIC_METRIC_CHANGE_KEY]["metric"] = "영업이익"
    plan["period"]["quarter"] = None
    plan["period"]["period_type"] = "fiscal_year"
    active_plan, resolution = _selected_resolution(table=table, plan=plan)

    change = resolve_periodic_metric_change(
        requested_periodic_metric_change(active_plan),
        resolution,
        query_plan=active_plan,
    )

    assert change is None


def test_more_than_two_requested_years_fail_closed() -> None:
    plan, resolution = _selected_resolution(plan=_change_plan(years=(2023, 2024, 2025)))

    change = resolve_periodic_metric_change(
        requested_periodic_metric_change(plan),
        resolution,
        query_plan=plan,
    )

    assert change is None


def test_orchestrator_renders_calculation_with_selected_source_citation() -> None:
    pair = _candidate(
        "p:ch_income",
        "p",
        rank=1,
        doc_group="periodic",
        content=_COMPARISON_TABLE,
        section="연결포괄손익계산서",
        base_year=2025,
        base_month=3,
        quarter=1,
        period_type="fiscal_quarter",
        statement_scope="연결",
        unit="백만원",
        report_nm="분기보고서 (2025.03)",
        source_refs=[{"table_id": "income", "row_start": 1, "row_end": 4}],
    )
    plan = QueryPlan(
        query="매출액",
        raw_query="테스트회사 2025년 1분기 매출액 전년 동기 대비 증가율",
        company="테스트회사",
        years=(2024, 2025),
        period=QueryPeriod(quarter=1, period_type="fiscal_quarter"),
        task_type="financial_metric",
        metric="매출액",
        disclosure_route=("periodic",),
        basis="consolidated",
        comparison={"type": "year_over_year", "years": [2024, 2025]},
        evidence=_change_plan()["evidence"],
    )

    result = AgentOrchestrator().run(
        plan.raw_query,
        plan,
        _execution(plan, pair),
    )
    generated = generate_answer(result.answer_draft)

    assert "periodic_metric_change" in result.execution_trace
    assert generated.answerable
    assert generated.sections[0].title == "정기공시 지표 증감"
    assert "2024년 매출액: 40,658,539백만원 [1]" in generated.answer_text
    assert "2025년 매출액: 44,407,761백만원 [1]" in generated.answer_text
    assert "증감액: +3,749,222백만원 [1]" in generated.answer_text
    assert "증감률: +9.22% [1]" in generated.answer_text
    assert not any(
        warning.startswith("unsupported_periodic")
        for warning in generated.warnings
    )


def test_orchestrator_marks_requested_rate_unanswerable_without_unit() -> None:
    pair = _candidate(
        "p:ch_income",
        "p",
        rank=1,
        doc_group="periodic",
        content=_COMPARISON_TABLE,
        section="연결포괄손익계산서",
        base_year=2025,
        base_month=3,
        quarter=1,
        period_type="fiscal_quarter",
        statement_scope="연결",
        report_nm="분기보고서 (2025.03)",
        source_refs=[{"table_id": "income", "row_start": 1, "row_end": 4}],
    )
    plan = QueryPlan(
        query="매출액",
        raw_query="테스트회사 2025년 1분기 매출액 전년 동기 대비 증가율",
        company="테스트회사",
        years=(2024, 2025),
        period=QueryPeriod(quarter=1, period_type="fiscal_quarter"),
        task_type="financial_metric",
        metric="매출액",
        disclosure_route=("periodic",),
        basis="consolidated",
        comparison={"type": "year_over_year", "years": [2024, 2025]},
        evidence=_change_plan()["evidence"],
    )

    result = AgentOrchestrator().run(
        plan.raw_query,
        plan,
        _execution(plan, pair),
    )
    generated = generate_answer(result.answer_draft)

    assert "periodic_metric_change_unresolved" in result.execution_trace
    assert not result.answer_draft.answerable
    assert not generated.answerable
    assert "periodic_metric_change_unresolved" in generated.warnings
