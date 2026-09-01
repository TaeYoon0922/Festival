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


def test_p07_product_phrase_outranks_generic_sales_metric_chunks() -> None:
    generic_sales = [
        _item(
            f"periodic_20230512001368:ch_generic_{index}",
            "periodic_20230512001368",
            rank=index,
            text=text,
            year=2023,
            quarter=1,
            section="매출 및 수주상황",
            temporal_match=True,
        )
        for index, text in enumerate(
            (
                "연료전지 서비스 매출액 12,000",
                "기타 주기기 매출액 18,000",
                "연료전지 부품 매출액 9,000",
            ),
            start=1,
        )
    ]
    gold = _item(
        "periodic_20230512001368:ch_55913a2857ccdcfb104f",
        "periodic_20230512001368",
        rank=8,
        text="구분 | 연료전지 주기기 | 매출액 | 23,848",
        year=2023,
        quarter=1,
        section="매출 및 수주상황",
        temporal_match=True,
    )
    evidence = _evidence(
        [
            *(
                _group(f"generic-{index}", item)
                for index, item in enumerate(generic_sales, start=1)
            ),
            _group("gold", gold),
        ],
        question="두산퓨얼셀 2023년 1분기 연료전지 주기기 매출액",
        year=2023,
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan.update({"metric": "매출액", "lexical_query": "연료전지 주기기 매출액"})
    plan["period"]["quarter"] = 1
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids[0] == gold.chunk_id
    selected_sources = [
        source
        for fact in selected.resolution.facts
        for source in fact.sources
    ]
    assert selected_sources[0].source_refs


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
    same_period_competitors = [
        replace(
            _item(
                f"periodic_20230512001368:ch_competing_{index}",
                "periodic_20230512001368",
                rank=index,
                text=text,
                year=2023,
                quarter=1,
                section="생산 및 설비",
                temporal_match=True,
            ),
            report_nm="분기보고서 (2023.03)",
        )
        for index, text in enumerate(
            (
                "창원공장 생산능력 생산실적 평균가동률 71%",
                "익산공장 생산설비 현황",
                "익산공장 생산능력 58.0",
            ),
            start=1,
        )
    ]
    gold = replace(gold, retrieval_rank=8, retrieval_score=0.92)
    evidence = _evidence(
        [
            *(
                _group(f"competing-{index}", item)
                for index, item in enumerate(same_period_competitors, start=1)
            ),
            _group("q1", gold),
            _group("q2", other),
        ],
        question="두산퓨얼셀 익산공장 1분기 생산능력 생산실적 평균가동률",
        year=2023,
        task_type="periodic_fact",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan["lexical_query"] = "익산공장 생산능력 생산실적 평균가동률"
    plan["period"]["quarter"] = 1
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids[0] == (
        "periodic_20230512001368:ch_production"
    )
    assert all(
        source.source_refs
        for fact in selected.resolution.facts
        for source in fact.sources
        if source.chunk_id == "periodic_20230512001368:ch_production"
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


_INCOME_STATEMENT_TABLE = """\
| 열 1 | 제 58 기 1분기 / 3개월 | 제 57 기 1분기 / 3개월 |
| --- | --- | --- |
| 매출액 | 44,407,761 | 40,658,539 |
| 매출원가 | 35,428,253 | 32,230,756 |
| 영업이익 | 3,633,609 | 3,557,362 |
| 보통주기본주당이익(손실) (단위 : 원) | 12,076 | 12,287 |
"""

_REVENUE_NOTE_TABLE = """\
| 열 1 | 공시금액 |
| --- | --- |
| 매출액 | 44,407,761 |
| 재화의 판매로 인한 수익(매출액) | 36,287,439 |
| 금융업매출액 | 5,322,134 |
"""

_STANDALONE_TABLE = """\
| 열 1 | 공시금액 |
| --- | --- |
| 매출액 | 12,000,000 |
| 매출원가 | 8,000,000 |
"""


def test_selector_keeps_consolidated_income_statement_metric_row() -> None:
    standalone = _item(
        "p:ch_standalone",
        "p-sep",
        rank=1,
        text=_STANDALONE_TABLE,
        year=2025,
        quarter=1,
        section_path=("재무제표", "손익계산서"),
        statement_scope="별도",
        temporal_match=True,
    )
    note = _item(
        "p:ch_note",
        "p-con",
        rank=2,
        text=_REVENUE_NOTE_TABLE,
        year=2025,
        quarter=1,
        section="수익",
        statement_scope="연결",
        temporal_match=True,
    )
    income = _item(
        "p:ch_income",
        "p-con",
        rank=4,
        text=_INCOME_STATEMENT_TABLE,
        year=2025,
        quarter=1,
        section_path=("연결포괄손익계산서",),
        statement_scope="연결",
        temporal_match=True,
    )
    evidence = _evidence(
        [
            _group("standalone", standalone),
            _group("note", note),
            _group("income", income),
        ],
        question="테스트회사 2025년 1분기 연결 매출액",
        year=2025,
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan.update(
        {
            "metric": "매출액",
            "basis": "consolidated",
            "lexical_query": "연결 매출액",
        }
    )
    plan["period"]["quarter"] = 1
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == ("p:ch_income",)
    assert "periodic_metric_row_preferred" in selected.warnings
    assert "p:ch_standalone" in selected.excluded_chunk_ids
    assert "p:ch_note" in selected.excluded_chunk_ids


def test_selector_limits_balance_sheet_metric_to_best_exact_row() -> None:
    balance = _item(
        "p:ch_balance",
        "p-con",
        rank=1,
        text="""\
| 과 목 | 주 석 | 제 56 (당) 기 | 제 55 (전) 기 |
| --- | --- | --- | --- |
| 자 산 총 계 |  | 514,531,948 | 455,905,980 |
""",
        year=2024,
        section_path=("(첨부)연 결 재 무 제 표",),
        statement_scope="연결",
        temporal_match=True,
    )
    equity = _item(
        "p:ch_equity",
        "p-con",
        rank=2,
        text="""\
| 과 목 | 지배기업 소유주지분 / 자본금 | 총 계 |
| --- | --- | --- |
| 2023.1.1(전기초) | 897,514 | 354,749,604 |
""",
        year=2024,
        section_path=("(첨부)연 결 재 무 제 표",),
        statement_scope="연결",
        temporal_match=True,
    )
    evidence = _evidence(
        [_group("balance", balance), _group("equity", equity)],
        question="삼성전자 2024년 사업보고서 연결 자산총계",
        year=2024,
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan.update(
        {
            "metric": "자산총계",
            "basis": "consolidated",
            "lexical_query": "자산총계 자산 총계 자 산 총 계",
        }
    )
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == ("p:ch_balance",)
    assert "periodic_metric_row_preferred" in selected.warnings
    assert "p:ch_equity" in selected.excluded_chunk_ids


def test_selector_prefers_footnoted_income_statement_over_segment_note() -> None:
    segment = _item(
        "p:ch_segment",
        "p-con",
        rank=1,
        text="""\
| 열 1 | 영업부문 / DX 부문 | 영업부문 / DS 부문 | 기업 전체 총계 합계 |
| --- | --- | --- | --- |
| 매출액 | 47,292,742 | 23,137,290 | 71,915,601 |
""",
        year=2025,
        quarter=1,
        section_path=("연결재무제표 주석", "부문별 보고 (연결)",),
        statement_scope="연결",
        temporal_match=True,
    )
    income = _item(
        "p:ch_income_footnoted",
        "p-con",
        rank=4,
        text="""\
| 열 1 | 제 57 기 1분기 / 3개월 | 제 56 기 1분기 / 3개월 |
| --- | --- | --- |
| 매출액 (주26) | 79,140,503 | 71,915,601 |
| 영업이익 (주26) | 6,685,272 | 6,606,009 |
""",
        year=2025,
        quarter=1,
        section_path=("재무제표", "연결 손익계산서"),
        statement_scope="연결",
        temporal_match=True,
    )
    evidence = _evidence(
        [_group("segment", segment), _group("income", income)],
        question="테스트회사 2025년 1분기 연결 매출액",
        year=2025,
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan.update(
        {
            "metric": "매출액",
            "basis": "consolidated",
            "lexical_query": "연결 매출액",
        }
    )
    plan["period"]["quarter"] = 1
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == ("p:ch_income_footnoted",)
    assert "p:ch_segment" in selected.excluded_chunk_ids


def test_period_comparison_keeps_metric_row_without_single_period_match() -> None:
    prior_income = _item(
        "p:ch_income_prior",
        "p-con",
        rank=1,
        text="""\
| 열 1 | 제 57 기 1분기 / 3개월 | 제 56 기 1분기 / 3개월 |
| --- | --- | --- |
| 매출액 | 40,658,539 | 37,770,005 |
""",
        year=2024,
        quarter=1,
        section_path=("연결포괄손익계산서",),
        statement_scope="연결",
        temporal_match=False,
    )
    prior_income = replace(
        prior_income,
        period={"base_year": 2024, "base_month": 3, "statement_scope": "연결"},
    )
    income = _item(
        "p:ch_income_compare",
        "p-con",
        rank=2,
        text=_INCOME_STATEMENT_TABLE,
        year=2025,
        quarter=1,
        section_path=("연결포괄손익계산서",),
        statement_scope="연결",
        temporal_match=False,
    )
    income = replace(
        income,
        period={"base_year": 2025, "base_month": 3, "statement_scope": "연결"},
    )
    evidence = _evidence(
        [_group("prior", prior_income), _group("income", income)],
        question="테스트회사 2025년 1분기 매출액과 2024년 1분기 매출액 비교",
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan.update(
        {
            "metric": "매출액",
            "basis": "unspecified",
            "lexical_query": "매출액과 1분기 매출액 비교",
            "comparison": {"type": "period_comparison", "years": [2024, 2025]},
        }
    )
    plan["period"] = {
        "year": None,
        "quarter": 1,
        "from": None,
        "to": None,
        "period_type": "fiscal_quarter",
    }
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == ("p:ch_income_compare",)
    assert "p:ch_income_prior" in selected.excluded_chunk_ids
    assert selected.resolution.unresolved_requirements == ()
    assert "explicit_period_unmatched" not in selected.warnings


def test_selector_prefers_income_statement_net_income_over_cash_flow() -> None:
    cash_flow = _item(
        "p:ch_cash_flow",
        "p-con",
        rank=1,
        text="""\
| 열 1 | 열 2 | 차량 | 금융 |
| --- | --- | --- | --- |
| 영업활동현금흐름 | 당기순이익(손실) | 5,574,692 | 349,298 |
| 영업활동현금흐름 | 조정 | 156,165 | 3,027,690 |
""",
        year=2025,
        quarter=1,
        section_path=("연결현금흐름표",),
        statement_scope="연결",
        temporal_match=True,
    )
    income = _item(
        "p:ch_income_net",
        "p-con",
        rank=5,
        text="""\
| 열 1 | 제 58 기 1분기 / 3개월 | 제 57 기 1분기 / 3개월 |
| --- | --- | --- |
| 계속영업연결분기순이익 | 3,382,174 | 3,695,111 |
| 연결분기순이익 | 3,382,174 | 3,376,001 |
| 보통주기본주당이익(손실) (단위 : 원) | 12,076 | 12,287 |
""",
        year=2025,
        quarter=1,
        section_path=("재무제표", "연결 손익계산서"),
        statement_scope="연결",
        temporal_match=True,
    )
    evidence = _evidence(
        [_group("cash_flow", cash_flow), _group("income", income)],
        question="테스트회사 2025년 1분기 연결 당기순이익",
        year=2025,
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan.update(
        {
            "metric": "당기순이익",
            "basis": "consolidated",
            "lexical_query": "연결 당기순이익",
        }
    )
    plan["period"]["quarter"] = 1
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == ("p:ch_income_net",)
    assert "p:ch_cash_flow" in selected.excluded_chunk_ids


def test_peer_rate_company_compare_selects_exact_metric_rows_per_company() -> None:
    income_table = (
        "| 열 1 | 제 56 (당) 기 | 제 55 (전) 기 |\n"
        "| --- | --- | --- |\n"
        "| 보험료수익 | 10,000 | 9,000 |"
    )
    life_noise = replace(
        _item(
            "life:noise",
            "life-doc",
            rank=1,
            text="보험료수익이 전기 대비 증가하였습니다.",
            year=2024,
            section="사업의 내용",
        ),
        corp_code="LIFE",
        corp_name="삼성생명",
        company_id="LIFE",
    )
    life_income = replace(
        _item(
            "life:income",
            "life-doc",
            rank=3,
            text=income_table,
            year=2024,
            section_path=("연결 손익계산서",),
            statement_scope="연결",
        ),
        corp_code="LIFE",
        corp_name="삼성생명",
        company_id="LIFE",
    )
    fire_noise = replace(
        _item(
            "fire:noise",
            "fire-doc",
            rank=2,
            text="영업수익이 전기 대비 증가하였습니다.",
            year=2024,
            section="사업의 내용",
        ),
        corp_code="FIRE",
        corp_name="삼성화재",
        company_id="FIRE",
    )
    fire_income = replace(
        _item(
            "fire:income",
            "fire-doc",
            rank=4,
            text=income_table.replace("보험료수익", "영업수익"),
            year=2024,
            section_path=("연결 손익계산서",),
            statement_scope="연결",
        ),
        corp_code="FIRE",
        corp_name="삼성화재",
        company_id="FIRE",
    )
    evidence = _evidence(
        [
            _group("life_noise", life_noise),
            _group("life_income", life_income),
            _group("fire_noise", fire_noise),
            _group("fire_income", fire_income),
        ],
        question=(
            "삼성생명과 삼성화재 2024 사업보고서 보험료수익(또는 영업수익) "
            "전기 대비 증감률을 비교하면 어느 쪽이 더 높아?"
        ),
        year=2024,
        task_type="financial_metric",
    )
    plan = copy.deepcopy(dict(evidence.query_plan))
    plan.update(
        {
            "metric": "보험료수익",
            "lexical_query": "보험료수익 증감률",
            "comparison": {"type": "company_comparison"},
            "companies": ["삼성생명", "삼성화재"],
            "corp_codes": ["LIFE", "FIRE"],
            "evidence": {
                "derived_metric": "peer_rate",
                "metric_fallback": "영업수익",
            },
        }
    )
    resolution = resolve_periodic_facts(evidence, query_plan=plan)

    selected = PeriodicEvidenceSelector().select(resolution, query_plan=plan)

    assert selected.selected_chunk_ids == ("life:income", "fire:income")
    assert selected.resolution.facts
    assert "no_periodic_fact_evidence" not in selected.warnings
