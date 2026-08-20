from __future__ import annotations

import copy

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
