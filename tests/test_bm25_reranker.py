from __future__ import annotations

import copy

from app.parsing.final_validation import (
    _evaluation_document,
    bm25_rerank_bonus,
    build_bm25_rerank_policy,
    rerank_bm25_candidates,
)
from app.parsing.release_validation import _Bm25Collector


def _table_chunk() -> dict:
    return {
        "chunk_id": "d1:ch_table",
        "chunk_type": "table",
        "table_id": "t1",
        "section_title": "1. 회사의 개요",
        "section_path": ["1. 회사의 개요"],
        "column_headers": ["부문", "주 요 제 품"],
        "content": "DX 부문 | TV, 모니터, 냉장고, 세탁기",
        "retrieval_text": "frozen retrieval text",
        "source_refs": [{"table_id": "t1", "row_start": 1, "row_end": 1}],
    }


def test_table_header_alignment_reorders_without_mutating_raw_results() -> None:
    query = "삼성전자 DX 부문의 주요 제품은 무엇인가"
    table = _table_chunk()
    text = {
        "chunk_id": "d2:ch_text",
        "chunk_type": "text",
        "section_title": "원재료 및 생산설비",
        "section_path": ["원재료 및 생산설비"],
        "content": "DX 부문의 주요 품목별 생산능력을 산출합니다.",
        "retrieval_text": "another frozen retrieval text",
    }
    raw = [(16.37, text), (4.85, table)]
    before = copy.deepcopy(raw)

    ranked = rerank_bm25_candidates(raw, query)

    assert [chunk["chunk_id"] for _, chunk in ranked] == [
        "d1:ch_table",
        "d2:ch_text",
    ]
    assert {chunk["chunk_id"]: score for score, chunk in ranked} == {
        "d1:ch_table": 4.85,
        "d2:ch_text": 16.37,
    }
    assert raw == before
    assert ranked[0][1] is table
    assert table["source_refs"] == [
        {"table_id": "t1", "row_start": 1, "row_end": 1}
    ]


def test_explanation_query_preserves_exact_bm25_order() -> None:
    raw = [
        (9.0, {"chunk_id": "text", "chunk_type": "text"}),
        (8.0, {"chunk_id": "table", "chunk_type": "table"}),
    ]

    ranked = rerank_bm25_candidates(raw, "매출 증가 이유와 향후 전망")

    assert ranked == raw
    assert [chunk for _, chunk in ranked] == [chunk for _, chunk in raw]


def test_table_prior_is_bounded_and_projection_prior_is_weaker() -> None:
    policy = build_bm25_rerank_policy("보유 주식수와 보유 비율 현황")
    table = {
        "chunk_id": "table",
        "chunk_type": "table",
        "content": "관련 없는 표",
    }
    projection = {
        "chunk_id": "projection",
        "chunk_type": "table_projection",
        "content": "관련 없는 투영",
    }

    assert policy.table_intent is True
    assert bm25_rerank_bonus(table, policy) == 2.5
    assert bm25_rerank_bonus(projection, policy) == 1.0


def test_full_corpus_collector_uses_same_reranker_and_preserves_raw_rank() -> None:
    question = {
        "question_id": "Q01",
        "doc_group": "periodic",
        "query": "삼성전자 DX 부문의 주요 제품은 무엇인가",
        "doc_id": "d1",
        "target_type": "table",
        "target_id": "t1",
        "evidence_terms": ["TV"],
    }
    collector = _Bm25Collector((question,))
    collector.add(
        {
            "doc_id": "d1",
            "doc_group": "periodic",
            "corp_name": "삼성전자",
            "report_nm": "사업보고서",
        },
        _table_chunk(),
    )
    for index in range(20):
        collector.add(
            {
                "doc_id": f"d{index + 2}",
                "doc_group": "periodic",
                "corp_name": "삼성전자",
                "report_nm": "분기보고서",
            },
            {
                "chunk_id": f"d{index + 2}:ch_text",
                "chunk_type": "text",
                "section_title": "생산설비",
                "section_path": ["생산설비"],
                "content": "DX 부문의 주요 품목별 생산능력 " * 3,
            },
        )

    result = collector.evaluate("shared-reranker-test")["questions"][0]

    assert result["structural_raw_first_relevant_rank"] > 10
    assert result["structural_first_relevant_rank"] == 1
    assert result["structural_raw_top1_chunk_id"] != "d1:ch_table"
    assert result["structural_top1_chunk_id"] == "d1:ch_table"
    assert result["structural_top1_rerank_bonus"] > 0


def test_query_derived_group_and_listed_name_alignment_are_generic() -> None:
    policy = build_bm25_rerank_policy("KT 자기주식 처분 예정 보통주 수와 가격")
    matching = {
        "chunk_type": "table",
        "listed_name": "KT",
        "doc_group": "major",
        "content": "처분예정주식 보통주식 131,690",
    }
    wrong_group = {**matching, "doc_group": "periodic"}
    wrong_company = {**matching, "listed_name": "다른회사"}

    assert policy.preferred_doc_group == "major"
    assert bm25_rerank_bonus(matching, policy) > bm25_rerank_bonus(
        wrong_group, policy
    )
    assert bm25_rerank_bonus(matching, policy) > bm25_rerank_bonus(
        wrong_company, policy
    )


def test_holding_reference_year_is_not_used_as_report_year_bonus() -> None:
    holding = build_bm25_rerank_policy(
        "파마리서치 국민연금 2022년 12월 5일 현재 보유 비율"
    )
    periodic = build_bm25_rerank_policy("두산퓨얼셀 2023년 1분기 매출액")

    assert holding.preferred_doc_group == "holding"
    assert holding.fiscal_year is None
    assert periodic.preferred_doc_group == "periodic"
    assert periodic.fiscal_year == 2023
    assert periodic.preferred_doc_subtype == "quarter"


def test_evaluation_document_adds_alias_without_mutating_chunk() -> None:
    source = _table_chunk()
    before = copy.deepcopy(source)

    evaluated = _evaluation_document(
        {
            "doc_id": "d1",
            "doc_group": "major",
            "corp_name": "케이티",
            "listed_name": "KT",
            "report_nm": "주요사항보고서(자기주식처분결정)",
        },
        source,
    )

    assert evaluated["listed_name"] == "KT"
    assert evaluated["chunk_id"] == source["chunk_id"]
    assert source == before


def test_legal_entity_alignment_prefers_exact_law_without_question_id_rule() -> None:
    query = "산업안전보건법 위반 과태료 고용노동부 제재 내역"
    policy = build_bm25_rerank_policy(query)
    exact = {
        "chunk_id": "d1:ch_exact",
        "chunk_type": "table",
        "section_title": "제재 등과 관련된 사항",
        "table_title": "행정기관 제재 현황",
        "column_headers": ["제재기관", "처분", "근거 법령"],
        "content": (
            "고용노동부 | 과태료 | 산업안전보건법 위반에 따른 처분"
        ),
    }
    other = {
        **exact,
        "chunk_id": "d2:ch_other",
        "content": "법원 | 벌금 | 다른 법령에 따른 제재",
    }

    assert policy.legal_law_signals == ("산업안전보건법",)
    assert policy.legal_institution_signals == ("고용노동부",)
    assert policy.legal_disposition_signals == ("과태료",)
    assert set(policy.legal_violation_signals) == {"위반", "제재"}
    assert bm25_rerank_bonus(exact, policy) > bm25_rerank_bonus(other, policy)

    ranked = rerank_bm25_candidates([(10.0, other), (9.0, exact)], query)

    assert [chunk["chunk_id"] for _, chunk in ranked] == [
        "d1:ch_exact",
        "d2:ch_other",
    ]


def test_repeated_exact_law_alignment_is_bounded() -> None:
    policy = build_bm25_rerank_policy("산업안전보건법 위반 제재 내역")
    one_row = {
        "chunk_type": "table",
        "content": "산업안전보건법에 따른 제재",
    }
    repeated_rows = {
        "chunk_type": "table",
        "content": " | ".join(["산업안전보건법에 따른 제재"] * 20),
    }

    assert bm25_rerank_bonus(repeated_rows, policy) > bm25_rerank_bonus(
        one_row, policy
    )
    assert (
        bm25_rerank_bonus(repeated_rows, policy)
        - bm25_rerank_bonus(one_row, policy)
        == 8.0
    )


def test_non_legal_table_query_receives_no_legal_entity_bonus() -> None:
    policy = build_bm25_rerank_policy("보유 주식수와 보유 비율 현황")
    chunk = {
        "chunk_id": "d1:ch_table",
        "chunk_type": "table",
        "content": "산업안전보건법 위반 과태료 고용노동부",
    }

    assert policy.legal_law_signals == ()
    assert policy.legal_institution_signals == ()
    assert policy.legal_disposition_signals == ()
    assert policy.legal_violation_signals == ()
    assert bm25_rerank_bonus(chunk, policy) == 2.5
