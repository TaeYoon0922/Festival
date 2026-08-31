#!/usr/bin/env python3
"""동일 산업군 peer 비교 질문 → qa-tool examples YAML (로컬 재생성용)."""

from __future__ import annotations

import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
BATCH = Path(__file__).resolve().parents[1] / "peer-compare-batch.txt"

# (id, sector_no, sector, corp_a, listed_a, corp_b, listed_b, focus, doc_group, task, must_include, negative, notes)
PROBES = [
    (
        "PC01", 1, "반도체·전자부품",
        "삼성전자", "삼성전자", "SK하이닉스", "SK하이닉스",
        "메모리·파운드리 연결 손익 peer compare",
        "periodic",
        "cross_company · financial_metric · comparative · multi_corp",
        ["삼성전자 2024 연결 매출·영업이익", "SK하이닉스 2024 연결 매출·영업이익", "차이(파생) — 각사 표 숫자"],
        "단일 기업만 답 · 두사 숫자 혼동",
        "comparison_frame=cross_company · dual periodic retrieval.",
    ),
    (
        "PC02", 1, "반도체·전자부품",
        "LG이노텍", "LG이노텍", "삼성전기", "삼성전기",
        "부품사 영업이익률 peer compare",
        "periodic",
        "cross_company · derived_ratio · multi_corp",
        ["각사 2024 연결 매출·영업이익", "영업이익률(파생) 비교", "더 높은 쪽"],
        "한사 IS만 · LLM 임의 %",
        "derived ratio per company then compare.",
    ),
    (
        "PC03", 1, "반도체·전자부품",
        "한미반도체", "한미반도체", "LG이노텍", "LG이노텍",
        "exchange activity peer compare",
        "exchange",
        "cross_company · exchange_aggregate · multi_corp",
        ["각사 2024 exchange 건수", "계약·시설투자 금액 합(파생)", "더 많은 쪽"],
        "단일 corp enumeration",
        "P0-C must run twice (corp-scoped).",
    ),
    (
        "PC04", 2, "자동차·모빌리티",
        "현대자동차", "현대차", "기아", "기아",
        "완성차 매출 증감률 peer compare",
        "periodic",
        "cross_company · derived_rate · multi_corp",
        ["현대차·기아 2024 연결 매출 전기·당기", "각사 증감률(파생)", "더 높은 쪽"],
        "그룹 합산 · 단일 브랜드만",
        "YoY rate per OEM.",
    ),
    (
        "PC05", 2, "자동차·모빌리티",
        "현대모비스", "현대모비스", "현대오토에버", "현대오토에버",
        "모빌리티 밸류체인 영업이익 peer compare",
        "periodic",
        "cross_company · financial_metric · multi_corp",
        ["각사 2024 연결 영업이익", "더 큰 쪽 · 차이(파생)"],
        "완성차 표 혼입",
        "parts vs IT mobility subsidiary.",
    ),
    (
        "PC06", 3, "로봇",
        "레인보우로보틱스", "레인보우로보틱스", "두산로보틱스", "두산로보틱스",
        "로봇 손익 scale peer compare",
        "periodic",
        "cross_company · financial_metric · multi_corp",
        ["각사 2024 연결 매출·영업이익", "차이(파생)"],
        "로봇 segment 단일행만",
        "robotics pure-play pair.",
    ),
    (
        "PC07", 4, "2차전지",
        "LG에너지솔루션", "LG에너지솔루션", "삼성SDI", "삼성SDI",
        "셀·소재 capex peer compare",
        "exchange",
        "cross_company · facility_investment · exchange_aggregate · multi_corp",
        ["각사 2024 신규시설투자 합(파생)", "더 큰 쪽"],
        "단일 filing · annual CAPEX 표만",
        "investment_amount sum by corp.",
    ),
    (
        "PC08", 4, "2차전지",
        "LG에너지솔루션", "LG에너지솔루션", "에코프로비엠", "에코프로비엠",
        "2차전지 순이익 부호 peer compare",
        "periodic",
        "cross_company · sign_flip · multi_corp",
        ["각사 2024 연결 당기순이익", "흑자/적자 판정(파생)"],
        "영업이익만 · 부호 혼동",
        "sign classification per company.",
    ),
    (
        "PC09", 5, "철강",
        "POSCO홀딩스", "POSCO홀딩스", "현대제철", "현대제철",
        "철강 부채비율 peer compare",
        "periodic",
        "cross_company · balance_ratio · multi_corp",
        ["각사 2024 연결 부채·자본", "부채비율(파생)", "더 높은 쪽"],
        "단일 BS · LLM 임의 %",
        "balance_ratio per steelmaker.",
    ),
    (
        "PC10", 19, "신재생에너지",
        "한화솔루션", "한화솔루션", "OCI홀딩스", "OCI홀딩스",
        "태양광 segment YoY peer compare",
        "periodic",
        "cross_company · segment_rate · metric_fallback · multi_corp",
        ["각사 태양광/Qcells segment 매출 전기·당기", "증감률(파생)", "더 높은 쪽"],
        "총매출 단일행 · module 가격 invent",
        "renewable segment rate pair.",
    ),
    (
        "PC11", 7, "금융·보험",
        "KB금융", "KB금융", "신한지주", "신한지주",
        "금융지주 순이익 peer compare",
        "periodic",
        "cross_company · financial_metric · multi_corp",
        ["KB·신한 2024 연결 당기순이익", "차이(파생)"],
        "은행 단독 vs 지주 연결 혼동",
        "financial holding peer.",
    ),
    (
        "PC12", 7, "금융·보험",
        "삼성생명", "삼성생명", "삼성화재해상보험", "삼성화재",
        "생·손보 수익 증감률 peer compare",
        "periodic",
        "cross_company · derived_rate · metric_fallback · multi_corp",
        ["각사 보험료수익/영업수익 전기·당기", "증감률(파생)", "더 높은 쪽"],
        "동일 표 혼동 · 손보 별도표",
        "life vs P&C within Samsung FS.",
    ),
    (
        "PC13", 8, "AI소프트웨어·플랫폼",
        "NAVER", "NAVER", "카카오", "카카오",
        "플랫폼 매출 성장 peer compare",
        "periodic",
        "cross_company · derived_rate · multi_doc · multi_corp",
        ["각사 2023·2024 연결 매출", "증가율(파생)", "더 높은 쪽"],
        "단일 연도 · segment만",
        "dual-year revenue growth compare.",
    ),
    (
        "PC14", 20, "엔터테인먼트",
        "하이브", "하이브", "에스엠", "에스엠",
        "엔터 손익 scale peer compare",
        "periodic",
        "cross_company · financial_metric · multi_corp",
        ["각사 2024 연결 매출·영업이익", "차이(파생)"],
        "앨범·공연 뉴스 invent",
        "K-pop major vs mid-cap label.",
    ),
    (
        "PC15", 9, "통신",
        "SK텔레콤", "SK텔레콤", "KT", "KT",
        "통신사 영업이익률 peer compare",
        "periodic",
        "cross_company · derived_ratio · multi_corp",
        ["SKT·KT 2024 연결 매출·영업이익", "영업이익률(파생)", "더 높은 쪽"],
        "무선 segment만 단독",
        "telco margin compare.",
    ),
    (
        "PC16", 9, "통신",
        "LG유플러스", "LG유플러스", "SK텔레콤", "SK텔레콤",
        "분기 매출 peak peer compare",
        "periodic",
        "cross_company · quarter_timeseries · multi_corp",
        ["각사 2024 Q1–Q3 매출", "최고 분기(파생)", "더 큰 쪽"],
        "annual 단일행만",
        "quarter peak across two telcos.",
    ),
    (
        "PC17", 10, "게임",
        "크래프톤", "크래프톤", "NC", "엔씨소프트",
        "게임사 손익 peer compare",
        "periodic",
        "cross_company · financial_metric · multi_corp",
        ["각사 2024 연결 영업·순이익", "더 수익성 좋은 쪽(서술+숫자)"],
        "IP별 segment만",
        "publisher profitability pair.",
    ),
    (
        "PC18", 10, "게임",
        "크래프톤", "크래프톤", "시프트업", "시프트업",
        "분기합 vs annual peer compare",
        "periodic",
        "cross_company · quarter_sum_vs_annual · multi_corp",
        ["각사 4분기 합 vs annual(파생)", "나란히 비교"],
        "단일 분기 · 한사만",
        "reconciliation pattern × two corps.",
    ),
    (
        "PC19", 11, "바이오·제약",
        "삼성바이오로직스", "삼성바이오로직스", "셀트리온", "셀트리온",
        "CDMO·바이오 손익 scale peer compare",
        "periodic",
        "cross_company · financial_metric · multi_corp",
        ["각사 2024 연결 매출·영업이익", "차이(파생)"],
        "CMO contract만 · pipeline 추측",
        "large-cap biotech pair.",
    ),
    (
        "PC20", 11, "바이오·제약",
        "삼성바이오로직스", "삼성바이오로직스", "셀트리온", "셀트리온",
        "바이오 supply contract peer compare",
        "exchange+periodic",
        "cross_company · exchange_aggregate · multi_corp",
        ["각사 2024 공급계약 합(파생)", "더 큰 쪽"],
        "단일 계약 · R&D비 혼동",
        "exchange sum per biotech.",
    ),
    (
        "PC21", 11, "바이오·제약",
        "알테오젠", "알테오젠", "한미약품", "한미약품",
        "제약 영업이익 증감률 peer compare",
        "periodic",
        "cross_company · derived_rate · multi_corp",
        ["각사 2024 영업이익 전기·당기", "증감률(파생)", "더 높은 쪽"],
        "파이프라인 뉴스 invent",
        "mid-cap pharma pair.",
    ),
    (
        "PC22", 12, "조선",
        "HD현대중공업", "HD현대중공업", "한화오션", "한화오션",
        "수주잔고 증감률 peer compare",
        "periodic",
        "cross_company · derived_rate · metric_fallback · multi_corp",
        ["각사 수주잔고/수주실적 전기·당기", "증감률(파생)", "더 높은 쪽"],
        "수주 segment 없는 표만",
        "order backlog YoY pair.",
    ),
    (
        "PC23", 12, "조선",
        "HD현대중공업", "HD현대중공업", "삼성중공업", "삼성중공업",
        "조선 exchange contract peer compare",
        "exchange",
        "cross_company · exchange_aggregate · multi_corp",
        ["각사 2024 계약금액 합(파생)", "더 큰 쪽"],
        "단일 yard · 건조进度 invent",
        "shipbuilder contract sum.",
    ),
    (
        "PC24", 13, "원전",
        "두산에너빌리티", "두산에너빌리티", "한전기술", "한전기술",
        "원전 supply count·max peer compare",
        "exchange",
        "cross_company · exchange_aggregate · max · multi_corp",
        ["각사 2024·2025 건수", "최대 계약금액(파생)", "더 많은/큰 쪽"],
        "원전 segment 없이 단일 계약",
        "nuclear supply peer.",
    ),
    (
        "PC25", 14, "방산·항공우주",
        "한화에어로스페이스", "한화에어로스페이스", "LIG디펜스앤에어로스페이스", "LIG넥스원",
        "방산 growth peer compare",
        "periodic",
        "cross_company · compare_rates · multi_corp",
        ["각사 매출·영업이익 증가율(파생)", "더 높은 쪽"],
        "수출 계약 단건으로 대체",
        "defense prime vs mid-tier.",
    ),
    (
        "PC26", 14, "방산·항공우주",
        "현대로템", "현대로템", "한국항공우주", "한국항공우주",
        "방산 exchange qty·amount peer compare",
        "exchange",
        "cross_company · exchange_quantity_compare · multi_corp",
        ["각사 수량·금액(파생)", "더 큰 건"],
        "금액만 · 단일 프로그램",
        "rolling stock vs KAI contracts.",
    ),
    (
        "PC27", 15, "소비재·유통",
        "아모레퍼시픽", "아모레퍼시픽", "LG생활건강", "LG생활건강",
        "K뷰티 해외비중 peer compare",
        "periodic",
        "cross_company · breakdown_share · multi_corp",
        ["각사 국내·해외 비중(파생)", "해외 비중 더 높은 쪽"],
        "총매출 단일행",
        "beauty overseas mix pair.",
    ),
    (
        "PC28", 16, "건설",
        "현대건설", "현대건설", "삼성E&A", "삼성E&A",
        "건설 supply contract intensity peer compare",
        "exchange+periodic",
        "cross_company · exchange_aggregate · average · multi_corp",
        ["각사 2024 건수·건당 평균(파생)", "더 큰 쪽"],
        "단일 mega project",
        "EPC contract stats pair.",
    ),
    (
        "PC29", 17, "운송·물류",
        "HMM", "HMM", "현대글로비스", "현대글로비스",
        "운송·물류 순이익·부호 peer compare",
        "periodic",
        "cross_company · sign_flip · multi_corp",
        ["각사 2023·2024 연결 당기순이익", "흑자전환/적자 판정(파생)"],
        "운임 rate invent",
        "shipping vs logistics peer.",
    ),
    (
        "PC30", 18, "전력기기",
        "엘에스일렉트릭", "LS ELECTRIC", "HD현대일렉트릭", "HD현대일렉트릭",
        "전력기기 capex·equity ratio peer compare",
        "exchange",
        "cross_company · facility_investment · equity_compare · multi_corp",
        ["각사 2024 시설투자 금액", "자기자본 대비 %(파생)", "더 큰 쪽"],
        "단일 변압기 건만",
        "grid equipment capex pair.",
    ),
]

REASONING_KO = {
    "compare": "비교",
    "compute": "연산",
    "multi_doc": "복수문서",
    "multi_corp": "복수기업",
}


def _batch_questions() -> list[str]:
    if not BATCH.is_file():
        return []
    return [
        line.strip()
        for line in BATCH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _probe_dict(
    probe_id: str,
    sector_no: int,
    sector: str,
    corp_a: str,
    listed_a: str,
    corp_b: str,
    listed_b: str,
    focus: str,
    doc_group: str,
    task: str,
    must_include: list[str],
    negative: str,
    notes: str,
    question: str,
) -> dict:
    reasoning = ["compare", "compute", "multi_corp"]
    if "multi_doc" in task or "quarter" in task or "2023" in question:
        reasoning.append("multi_doc")
    basis = "연결" if doc_group.startswith("periodic") or "+" in doc_group else "해당없음"
    if doc_group == "periodic":
        basis = "연결"
    elif "periodic" in doc_group:
        basis = "연결"
    return {
        "id": probe_id,
        "slug": f"{listed_a}vs{listed_b}",
        "sector_no": sector_no,
        "sector": sector,
        "listed_a": listed_a,
        "corp_a": corp_a,
        "listed_b": listed_b,
        "corp_b": corp_b,
        "focus": focus,
        "question": question,
        "reasoning": reasoning,
        "doc_group": doc_group.replace("+periodic", "+periodic"),
        "report": "각 corp manifest 확인 후 FY·분기 보완",
        "basis": basis,
        "task": task,
        "comparison_frame": "cross_company",
        "must_include": must_include,
        "must_not": "코퍼스 밖 invent · 두 기업 숫자 혼동 · 단일 corp만 답",
        "evidence": {
            "manifest_a": f"corp_name={corp_a}",
            "manifest_b": f"corp_name={corp_b}",
        },
        "negative": negative,
        "notes": notes,
        "gaps": ["cross_company multi-corp execution"],
        "improve": "comparison_frame=cross_company · per-corp retrieval + side-by-side answer",
    }


def render(probe: dict) -> str:
    lines = [
        f"# {probe['sector_no']}. {probe['sector']} · {probe['listed_a']} vs {probe['listed_b']}",
        "",
        f"question_id: {probe['id']}",
        f"question: {probe['question']}",
        "",
        f"listed_name: {probe['listed_a']}  # peer: {probe['listed_b']}",
        f"corp_name: {probe['corp_a']}",
        f"peer_listed_name: {probe['listed_b']}",
        f"peer_corp_name: {probe['corp_b']}",
        "",
        f"sector_no: {probe['sector_no']}",
        f"sector: {probe['sector']}",
        f"sector_focus: {probe['focus']}",
        "",
        "reasoning_types:",
    ]
    for item in probe["reasoning"]:
        lines.append(f"  - {item}  # {REASONING_KO.get(item, item)}")
    lines.extend(
        [
            "",
            f"doc_group: {probe['doc_group']}",
            f"report_nm · period: {probe['report']}",
            f"basis: {probe['basis']}",
            "",
            f"comparison_frame (기대): {probe['comparison_frame']}",
            f"task_type (기대): {probe['task']}",
            "",
            "expected_behavior: normal_answer  # 현재 cross_company → fail-closed/clarification 가능",
            "",
            "must_include_in_answer:",
        ]
    )
    for item in probe["must_include"]:
        lines.append(f"  - {item}")
    lines.extend(
        [
            "",
            f"must_NOT_invent: {probe['must_not']}",
            "",
            "evidence:",
        ]
    )
    for key, value in probe["evidence"].items():
        lines.append(f"  {key}: {value}")
    lines.extend(
        [
            "",
            f"negative_example: {probe['negative']}",
            "",
            "corpus_sketch: |",
            f"  {probe['notes']}",
            "",
            "related_agent_gaps:",
        ]
    )
    for gap in probe["gaps"]:
        lines.append(f"  - {gap}")
    lines.append("")
    lines.append(f"improve_hint (참고): {probe['improve']}")
    lines.extend(
        [
            "",
            "related_pipeline (taeyoon):",
            "  - query_validation comparison_frame=cross_company → P0-D fail-closed",
            "  - multi_document_planner REASON_UNSUPPORTED_CALCULATION (cross-company)",
            "  - per-corp answer compose + side-by-side render (미구현)",
            "",
            "notes: |",
            "  공시 확인 후 must_include·doc_id를 corp별로 보완.",
        ]
    )
    return "\n".join(lines) + "\n"


def all_probes() -> list[dict]:
    questions = _batch_questions()
    probes: list[dict] = []
    for index, row in enumerate(PROBES):
        question = questions[index] if index < len(questions) else ""
        probes.append(_probe_dict(*row, question=question))
    return probes


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for probe in all_probes():
        slug = re.sub(r"[^\w가-힣]+", "", probe["slug"])[:24]
        path = OUT / f"{probe['id'].lower()}-{slug}.yaml"
        path.write_text(render(probe), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
