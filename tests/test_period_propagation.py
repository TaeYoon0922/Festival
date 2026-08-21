from __future__ import annotations

from app.reasoning.evidence_builder import build_evidence_set
from app.reasoning.periodic_evidence_selector import PeriodicEvidenceSelector
from app.reasoning.periodic_fact_resolver import resolve_periodic_facts
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


def _candidate(
    chunk_id: str,
    *,
    rank: int,
    year: int,
    month: int,
    content: str,
) -> tuple[CandidateChunk, RetrievalResult]:
    doc_id = chunk_id.split(":ch_", 1)[0]
    source_ref = {"table_id": "t1", "row_start": rank, "row_end": rank}
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": "01412725",
        "corp_name": "두산퓨얼셀",
        "doc_group": "periodic",
        "doc_subtype": "quarter",
        "chunk_type": "table",
        "section_path": ["II. 사업의 내용", "생산 및 매출"],
        "content": content,
        "retrieval_text": content,
        "report_nm": f"분기보고서 ({year}.{month:02d})",
        "base_year": year,
        "base_month": month,
        "table_id": "t1",
        "source_refs": [source_ref],
    }
    return (
        CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch()),
        RetrievalResult(
            chunk_id,
            doc_id,
            1.0 / rank,
            rank,
            {"hybrid": {"final_score": 1.0 - rank / 100.0}},
        ),
    )


def _pipeline(
    pairs: list[tuple[CandidateChunk, RetrievalResult]],
    *,
    question: str,
    period: QueryPeriod,
    metric: str | None,
    lexical_query: str,
):
    plan = QueryPlan(
        query=lexical_query,
        raw_query=question,
        company="두산퓨얼셀",
        task_type="financial_metric" if metric else "periodic_fact",
        metric=metric,
        period=period,
        disclosure_route=("periodic",),
    )
    candidates, results = zip(*pairs)
    evidence = build_evidence_set(
        question=question,
        query_plan=plan,
        candidates=candidates,
        results=results,
    )
    resolution = resolve_periodic_facts(evidence, query_plan=plan)
    selection = PeriodicEvidenceSelector().select(resolution, query_plan=plan)
    return evidence, resolution, selection


def _items(evidence):
    return [item for group in evidence.evidence_groups for item in group.items]


def test_p07_base_month_propagates_quarter_and_preserves_provenance() -> None:
    chunk_id = "periodic_20230512001368:ch_55913a2857ccdcfb104f"
    pair = _candidate(
        chunk_id,
        rank=1,
        year=2023,
        month=3,
        content="연료전지 주기기 매출액 23,848",
    )

    evidence, resolution, selection = _pipeline(
        [pair],
        question="두산퓨얼셀 2023년 1분기 연료전지 주기기 매출액",
        period=QueryPeriod(year=2023, quarter=1, period_type="fiscal_quarter"),
        metric="매출액",
        lexical_query="연료전지 주기기 매출액",
    )

    item = _items(evidence)[0]
    assert item.period["base_month"] == 3
    assert item.period["quarter"] == 1
    assert item.temporal_match is True
    assert selection.selected_chunk_ids == (chunk_id,)
    assert selection.resolution.facts[0].sources[0].source_refs == item.source_refs


def test_wrong_base_month_is_not_selected_for_explicit_quarter() -> None:
    pair = _candidate(
        "periodic_20230814002534:ch_q2",
        rank=1,
        year=2023,
        month=6,
        content="연료전지 주기기 매출액 48,000",
    )

    evidence, _, selection = _pipeline(
        [pair],
        question="두산퓨얼셀 2023년 1분기 연료전지 주기기 매출액",
        period=QueryPeriod(year=2023, quarter=1, period_type="fiscal_quarter"),
        metric="매출액",
        lexical_query="연료전지 주기기 매출액",
    )

    item = _items(evidence)[0]
    assert item.period["quarter"] == 2
    assert item.temporal_match is False
    assert selection.selected_chunk_ids == ()


def test_year_only_temporal_matching_is_unchanged() -> None:
    chunk_id = "periodic_20240312000736:ch_p02"
    pair = _candidate(
        chunk_id,
        rank=1,
        year=2023,
        month=12,
        content="연결대상회사수 232 주요종속회사 수 146",
    )

    evidence, _, selection = _pipeline(
        [pair],
        question="삼성전자 2023년 연결대상회사 기말 수와 주요종속회사 수",
        period=QueryPeriod(year=2023, period_type="fiscal_year"),
        metric=None,
        lexical_query="연결대상회사 기말 수 주요종속회사 수",
    )

    assert _items(evidence)[0].temporal_match is True
    assert selection.selected_chunk_ids == (chunk_id,)


def test_periodless_selection_behavior_is_unchanged() -> None:
    chunk_id = "periodic_20250319000952:ch_p10"
    pair = _candidate(
        chunk_id,
        rank=1,
        year=2024,
        month=12,
        content="유가증권시장 상장일 2024년 07월 11일",
    )

    evidence, _, selection = _pipeline(
        [pair],
        question="시프트업 유가증권시장 상장일",
        period=QueryPeriod(),
        metric=None,
        lexical_query="유가증권시장 상장일",
    )

    assert _items(evidence)[0].temporal_match is None
    assert selection.selected_chunk_ids == (chunk_id,)


def test_quarter_without_year_preserves_all_years_and_ambiguity() -> None:
    pairs = [
        _candidate(
            f"periodic_{year}0512:ch_q1",
            rank=rank,
            year=year,
            month=3,
            content=(
                f"익산공장 생산능력 {year - 1965}.0 "
                f"생산실적 {year - 1990}.0 평균가동률 {50 + rank}%"
            ),
        )
        for rank, year in enumerate((2023, 2024, 2025), start=1)
    ]

    evidence, resolution, selection = _pipeline(
        pairs,
        question="두산퓨얼셀 익산공장 1분기 생산능력 생산실적 평균가동률",
        period=QueryPeriod(quarter=1, period_type="fiscal_quarter"),
        metric=None,
        lexical_query="익산공장 생산능력 생산실적 평균가동률",
    )

    items = _items(evidence)
    assert [item.period["base_year"] for item in items] == [2023, 2024, 2025]
    assert all(item.period["quarter"] == 1 for item in items)
    assert all(item.temporal_match is True for item in items)
    assert len(selection.selected_chunk_ids) == 3
    assert set(selection.selected_chunk_ids) == {pair[0].chunk_id for pair in pairs}
    assert resolution.temporal_ambiguity is True
    assert "multiple_periodic_fact_alternatives" in resolution.warnings
