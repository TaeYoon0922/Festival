from __future__ import annotations

import copy
from dataclasses import replace

from app.reasoning.periodic_evidence_selector import PeriodicEvidenceSelector
from app.reasoning.periodic_fact_resolver import resolve_periodic_facts
from tests.test_periodic_fact_resolver import _evidence, _group, _item


def test_selector_limits_relevant_evidence_to_three_without_mutation() -> None:
    items = [
        _item(
            f"p1:ch_{index}",
            "p1",
            rank=index,
            text=f"연료전지 주기기 매출액 표 {index}",
            year=2023,
            quarter=1,
            section="매출 및 수주상황",
            temporal_match=True,
        )
        for index in range(1, 6)
    ]
    evidence = _evidence(
        [_group(f"g{index}", item) for index, item in enumerate(items, start=1)],
        question="두산퓨얼셀 2023년 1분기 연료전지 주기기 매출액",
        year=2023,
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan["metric"] = "매출액"
    plan["lexical_query"] = "연료전지 주기기 매출액"
    plan["period"]["quarter"] = 1
    resolution = resolve_periodic_facts(evidence, query_plan=plan)
    before = copy.deepcopy(resolution.to_dict())

    selected = PeriodicEvidenceSelector(max_evidence=3).select(
        resolution, query_plan=plan
    )

    assert selected.selected_chunk_ids == (
        "p1:ch_1",
        "p1:ch_2",
        "p1:ch_3",
    )
    assert sum(len(fact.sources) for fact in selected.resolution.facts) == 3
    assert selected.excluded_chunk_ids == ("p1:ch_4", "p1:ch_5")
    assert "periodic_evidence_limited:max=3" in selected.warnings
    assert resolution.to_dict() == before


def test_selector_excludes_other_period_even_when_text_is_relevant() -> None:
    matching = _item(
        "p23q1:ch_sales",
        "p23q1",
        rank=1,
        text="연료전지 주기기 매출액 23,848",
        year=2023,
        quarter=1,
        temporal_match=True,
    )
    other = _item(
        "p23q2:ch_sales",
        "p23q2",
        rank=2,
        text="연료전지 주기기 매출액 48,000",
        year=2023,
        quarter=2,
        temporal_match=False,
    )
    evidence = _evidence(
        [_group("q1", matching), _group("q2", other)],
        question="2023년 1분기 연료전지 주기기 매출액",
        year=2023,
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan.update({"metric": "매출액", "lexical_query": "연료전지 주기기 매출액"})
    plan["period"]["quarter"] = 1
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == ("p23q1:ch_sales",)
    assert selected.excluded_chunk_ids == ("p23q2:ch_sales",)


def test_selector_returns_unanswerable_view_when_explicit_period_has_no_match() -> None:
    other = _item(
        "p23q2:ch_sales",
        "p23q2",
        rank=1,
        text="연료전지 주기기 매출액 48,000",
        year=2023,
        quarter=2,
        temporal_match=False,
    )
    evidence = _evidence(
        [_group("q2", other)],
        question="2023년 1분기 연료전지 주기기 매출액",
        year=2023,
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan.update({"metric": "매출액", "lexical_query": "연료전지 주기기 매출액"})
    plan["period"]["quarter"] = 1
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.resolution.facts == ()
    assert "explicit_period" in selected.resolution.unresolved_requirements
    assert "explicit_period_evidence_unmatched" in selected.warnings


def test_p02_fiscal_year_prefers_annual_report_over_half_year_source() -> None:
    half_year = replace(
        _item(
            "periodic_20230814002534:ch_half",
            "periodic_20230814002534",
            rank=1,
            text="연결대상회사수 228 주요종속회사 수 145",
            year=2023,
            section="연결대상 종속회사 현황",
            temporal_match=True,
        ),
        report_nm="반기보고서 (2023.06)",
    )
    annual = replace(
        _item(
            "periodic_20240312000736:ch_gold",
            "periodic_20240312000736",
            rank=2,
            text="연결대상회사수 232 주요종속회사 수 146",
            year=2023,
            section="연결대상 종속회사 현황",
            temporal_match=True,
        ),
        report_nm="사업보고서 (2023.12)",
    )
    evidence = _evidence(
        [_group("half", half_year), _group("annual", annual)],
        question="삼성전자 2023년 연결대상회사 기말 수와 주요종속회사 수",
        year=2023,
        task_type="periodic_fact",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan["lexical_query"] = "연결대상회사 기말 수 주요종속회사 수"
    resolution = resolve_periodic_facts(evidence, query_plan=plan)
    before = copy.deepcopy(resolution.to_dict())

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == (
        "periodic_20240312000736:ch_gold",
    )
    assert selected.excluded_chunk_ids == (
        "periodic_20230814002534:ch_half",
    )
    assert "annual_report_source_preferred" in selected.warnings
    assert len(resolution.facts) == 2
    assert resolution.to_dict() == before


def test_p08_explicit_quarter_keeps_matching_production_evidence() -> None:
    gold = replace(
        _item(
            "periodic_20230512001368:ch_production",
            "periodic_20230512001368",
            rank=1,
            text="익산공장 생산능력 58.0 생산실적 평균가동률 56%",
            year=2023,
            quarter=1,
            section="생산 및 설비",
            temporal_match=True,
        ),
        report_nm="분기보고서 (2023.03)",
    )
    other = replace(
        _item(
            "periodic_20230814002534:ch_production",
            "periodic_20230814002534",
            rank=2,
            text="익산공장 생산능력 60.0 생산실적 평균가동률 57%",
            year=2023,
            quarter=2,
            section="생산 및 설비",
            temporal_match=False,
        ),
        report_nm="반기보고서 (2023.06)",
    )
    evidence = _evidence(
        [_group("q1", gold), _group("q2", other)],
        question="두산퓨얼셀 익산공장 1분기 생산능력 생산실적 평균가동률",
        year=2023,
        task_type="periodic_fact",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan["lexical_query"] = "익산공장 생산능력 생산실적 평균가동률"
    plan["period"]["quarter"] = 1
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == (
        "periodic_20230512001368:ch_production",
    )


def test_p10_periodless_listing_query_keeps_exact_listing_source() -> None:
    gold = replace(
        _item(
            "periodic_20250319000952:ch_listing",
            "periodic_20250319000952",
            rank=1,
            text="유가증권시장 상장일 2024년 07월 11일",
            year=2024,
            section="회사의 연혁",
        ),
        report_nm="사업보고서 (2024.12)",
    )
    evidence = _evidence(
        [_group("listing", gold)],
        question="시프트업 유가증권시장 상장일",
        task_type="listing_history",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan["period"] = {
        "year": None,
        "quarter": None,
        "from": None,
        "to": None,
        "period_type": None,
    }
    plan["lexical_query"] = "유가증권시장 상장일"
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == (
        "periodic_20250319000952:ch_listing",
    )
