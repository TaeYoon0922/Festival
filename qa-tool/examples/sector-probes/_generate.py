#!/usr/bin/env python3
"""sector-complex-reasoning-probes → qa-tool examples YAML (로컬 재생성용)."""

from __future__ import annotations

import re
from pathlib import Path

OUT = Path(__file__).resolve().parent

PROBES = [
    {
        "id": "SP01", "slug": "삼성전자", "sector_no": 1, "sector": "반도체·전자부품",
        "listed": "삼성전자", "corp": "삼성전자", "focus": "연결 손익·부문·Capa",
        "question": "삼성전자 2023·2024 사업보고서 연결 매출액을 비교해서 증가율(%)을 계산해줘",
        "reasoning": ["compare", "compute", "multi_doc"], "doc_group": "periodic",
        "report": "사업보고서 (2023.12 · 2024.12)", "basis": "연결",
        "task": "periodic_fact · financial_metric · comparative · derived_rate · multi_doc",
        "must_include": ["2023·2024 연결 매출액", "증가율(%) — 표 숫자로만"],
        "must_not": "LLM 임의 증가율 · 단일 연도만",
        "evidence": {"section_path": "연결 손익계산서 > 매출액", "manifest": "corp_name=삼성전자, annual"},
        "negative": "매출 단일행 · derived rate 미지원",
        "notes": "comparison 파싱 + derived rate compute.",
        "gaps": [], "improve": "derived_metric(rate) (implemented)",
    },
    {
        "id": "SP02", "slug": "현대자동차", "sector_no": 2, "sector": "자동차·모빌리티",
        "listed": "현대차", "corp": "현대자동차", "focus": "연결 손익·부문·전동화",
        "question": "현대자동차 2025 사업보고서 연결 기준 영업이익률(영업이익÷매출액)이 전기 대비 몇 %p 변했어?",
        "reasoning": ["compare", "compute"], "doc_group": "periodic",
        "report": "사업보고서 (2025.12)", "basis": "연결",
        "task": "periodic_fact · derived_ratio · delta_pp",
        "must_include": ["당기·전기 매출·영업이익", "%p 변화"],
        "must_not": "별도 표 혼동",
        "evidence": {"section_path": "연결 손익계산서", "manifest": "corp_name=현대자동차"},
        "negative": "영업이익 단일행",
        "notes": "same-table ratio + %p.",
        "gaps": [], "improve": "derived_metric(ratio/delta_pp) (implemented)",
    },
    {
        "id": "SP03", "slug": "레인보우로보틱스", "sector_no": 3, "sector": "로봇·자동화",
        "listed": "레인보우로보틱스", "corp": "레인보우로보틱스", "focus": "분기 매출 비교",
        "question": "레인보우로보틱스 2023년 1분기와 2024년 1분기 연결 매출액 중 어느 분기가 더 크고 차이는?",
        "reasoning": ["compare", "compute"], "doc_group": "periodic",
        "report": "분기보고서 Q1 2023 · Q1 2024", "basis": "연결",
        "task": "periodic_fact · quarter_comparison",
        "must_include": ["두 분기 연결 매출액", "차이(파생)"],
        "must_not": "annual 단일행만",
        "evidence": {"manifest": "corp_name=레인보우로보틱스, quarter"},
        "negative": "단일 분기만",
        "notes": "multi-quarter column pick + delta.",
        "gaps": [], "improve": "derived_metric(quarter_compare) (implemented)",
    },
    {
        "id": "SP04", "slug": "LG에너지솔루션", "sector_no": 4, "sector": "2차전지",
        "listed": "LG에너지솔루션", "corp": "LG에너지솔루션", "focus": "exchange 시설투자 집계",
        "question": "LG에너지솔루션 2024년과 2025년에 공시한 신규시설투자 금액 합계를 비교하고, 2025년 합계가 자기자본 대비 더 큰지?",
        "reasoning": ["compare", "compute", "multi_doc"], "doc_group": "exchange",
        "report": "신규시설투자등 · FY2024·FY2025", "basis": "해당없음",
        "task": "facility_investment · exchange_aggregate · compare",
        "must_include": ["2024·2025 투자금액 합계(파생)", "자기자본 대비(%) per filing"],
        "must_not": "단일 시설투자 건만",
        "evidence": {"manifest": "corp_name=LG에너지솔루션, doc_subtype=신규시설투자등"},
        "negative": "합계 미집계",
        "notes": "P0-C enumeration + investment_amount sum.",
        "gaps": [], "improve": "exchange_year_compare (implemented)",
    },
    {
        "id": "SP05", "slug": "POSCO홀딩스", "sector_no": 5, "sector": "철강",
        "listed": "POSCO홀딩스", "corp": "POSCO홀딩스", "focus": "부채비율 ratio",
        "question": "POSCO홀딩스 2024 사업보고서 연결 부채총계와 자본총계로 부채비율(부채÷자본)을 전기·당기 각각 계산해줘",
        "reasoning": ["compare", "compute"], "doc_group": "periodic",
        "report": "사업보고서 (2024.12)", "basis": "연결",
        "task": "financial_metric · derived_ratio · balance_sheet",
        "must_include": ["부채총계·자본총계 (연결)", "전기·당기 부채비율(파생)"],
        "must_not": "단일 행만 · LLM 임의 비율",
        "evidence": {"section_path": "연결 재무상태표", "manifest": "corp_name=POSCO홀딩스"},
        "negative": "부채총계 단일행",
        "notes": "two-row ratio from balance sheet.",
        "gaps": [], "improve": "periodic_derived_metrics balance_ratio (implemented)",
    },
    {
        "id": "SP06", "slug": "고려아연", "sector_no": 6, "sector": "비철금속",
        "listed": "고려아연", "corp": "고려아연", "focus": "exchange 시설투자 비교",
        "question": "고려아연 최근 신규시설투자 공시의 투자금액은 자기자본 대비 몇 %이고, 직전 시설투자 공시 대비 금액이 커졌어?",
        "reasoning": ["compare", "compute"], "doc_group": "exchange",
        "report": "신규시설투자등 (최근 2건)", "basis": "해당없음",
        "task": "facility_investment · multi_event_compare",
        "must_include": ["투자금액", "자기자본 대비(%)", "직전 대비 증감"],
        "must_not": "다른 시설투자 건 혼동",
        "evidence": {"manifest": "corp_name=고려아연, doc_subtype=신규시설투자등"},
        "negative": "단일 filing만",
        "notes": "recent-two compare; no explicit date in question → partial.",
        "gaps": [], "improve": "exchange_recent_pair (implemented)",
    },
    {
        "id": "SP07", "slug": "삼성생명", "sector_no": 7, "sector": "보험",
        "listed": "삼성생명", "corp": "삼성생명", "focus": "보험료수익 YoY",
        "question": "삼성생명 2024 사업보고서 연결 보험료수익(또는 영업수익)이 전기 대비 얼마나 변했는지 증감률까지",
        "reasoning": ["compare", "compute"], "doc_group": "periodic",
        "report": "사업보고서 (2024.12)", "basis": "연결",
        "task": "financial_metric · derived_rate · year_over_year",
        "must_include": ["당기·전기 보험료수익/영업수익", "증감률(파생)"],
        "must_not": "별도 혼입",
        "evidence": {"section_path": "연결 손익계산서", "manifest": "corp_name=삼성생명"},
        "negative": "단일 연도만",
        "notes": "metric alias 영업수익 fallback + rate.",
        "gaps": [], "improve": "metric_fallback + derived_metric(rate) (implemented)",
    },
    {
        "id": "SP11", "slug": "삼성바이오로직스", "sector_no": 11, "sector": "바이오",
        "listed": "삼성바이오로직스", "corp": "삼성바이오로직스", "focus": "공급계약 합 vs 매출",
        "question": "삼성바이오로직스 2024년 공시한 단일판매·공급계약 금액 합과 2024 사업보고서 연결 매출액 대비 비율은?",
        "reasoning": ["compute", "multi_doc"], "doc_group": "exchange+periodic",
        "report": "exchange 2024 + annual 2024", "basis": "연결",
        "task": "supply_contract · exchange_aggregate · cross_domain_ratio",
        "must_include": ["공급계약 금액 합(파생)", "연결 매출액", "비율(파생)"],
        "must_not": "단일 계약만",
        "evidence": {"manifest": "exchange supply_contract + periodic annual"},
        "negative": "cross-domain ratio 미구현",
        "notes": "exchange sum + periodic denominator via cross_domain_ratio.",
        "gaps": [], "improve": "cross_domain_ratio (implemented)",
    },
    {
        "id": "SP16", "slug": "현대건설", "sector_no": 16, "sector": "건설",
        "listed": "현대건설", "corp": "현대건설", "focus": "공급계약 건수·평균",
        "question": "현대건설 2024년 exchange 공급계약 공시 건수와 2024 연결 매출액 대비, 건당 평균 계약금액은?",
        "reasoning": ["compute", "multi_doc"], "doc_group": "exchange+periodic",
        "report": "exchange 2024 + annual 2024", "basis": "연결",
        "task": "supply_contract · exchange_aggregate · average",
        "must_include": ["건수", "건당 평균 계약금액(파생)"],
        "must_not": "단일 계약만",
        "evidence": {"manifest": "exchange supply_contract 2024"},
        "negative": "count without average",
        "notes": "P0-C count + contract_amount average.",
        "gaps": [], "improve": "exchange_field_aggregate average (implemented)",
    },
    {
        "id": "SP08", "slug": "NAVER", "sector_no": 8, "sector": "IT·플랫폼",
        "listed": "NAVER", "corp": "NAVER", "focus": "dual growth-rate compare",
        "question": "NAVER 2023·2024 사업보고서 연결 영업이익을 비교해, 매출액 증가율과 영업이익 증가율 중 어느 쪽이 더 높아?",
        "reasoning": ["compare", "compute", "multi_doc"], "doc_group": "periodic",
        "report": "사업보고서 2023·2024", "basis": "연결",
        "task": "financial_metric · derived_metric · compare_rates",
        "must_include": ["매출 증가율(파생)", "영업이익 증가율(파생)", "더 큰 쪽"],
        "must_not": "단일 지표만",
        "evidence": {"section_path": "연결 손익계산서"},
        "negative": "compare_rates 미구현",
        "notes": "periodic_derived_metrics compare_rates.",
        "gaps": [], "improve": "derived_metric(compare_rates) (implemented)",
    },
    {
        "id": "SP15", "slug": "아모레퍼시픽", "sector_no": 15, "sector": "소비재",
        "listed": "아모레퍼시픽", "corp": "아모레퍼시픽", "focus": "국내/해외 비중·%p",
        "question": "아모레퍼시픽 2024 사업보고서 국내 vs 해외 매출 비중(%)을 계산하고, 전기 대비 해외 비중이 몇 %p 늘었는지",
        "reasoning": ["compare", "compute"], "doc_group": "periodic",
        "report": "사업보고서 (2024.12)", "basis": "연결",
        "task": "financial_metric · breakdown_share · delta_pp",
        "must_include": ["국내·해외 비중(파생)", "해외 비중 %p 변화(파생)"],
        "must_not": "총매출 단일행만",
        "evidence": {"section_path": "매출 구분/지역별"},
        "negative": "breakdown_share 미구현",
        "notes": "metric_view=breakdown + delta_pp.",
        "gaps": [], "improve": "periodic_derived_metrics (implemented)",
    },
    {
        "id": "SP18", "slug": "엘에스일렉트릭", "sector_no": 18, "sector": "전력기기",
        "listed": "LS ELECTRIC", "corp": "엘에스일렉트릭", "focus": "exchange 시설투자",
        "question": "LS ELECTRIC 2024년 초고압 변압기 시설투자 금액(Gold E03)과 자기자본 대비 비율, 종료일(E04)까지 한 번에",
        "reasoning": ["compute"], "doc_group": "exchange",
        "report": "신규시설투자등 · rcept_dt=20240521", "basis": "해당없음",
        "task": "facility_investment · multi_field_single_event",
        "must_include": ["투자금액 (Gold E03)", "자기자본 대비(%)", "투자종료일 (Gold E04)"],
        "must_not": "다른 시설투자 건 혼동",
        "evidence": {"doc_id": "exchange_20240521800037", "manifest": "corp_name=엘에스일렉트릭"},
        "negative": "양성 벤치 — 단건 multi-field",
        "notes": "Gold E03/E04 검증.",
        "gaps": [], "improve": "양성 패턴",
        "gold_refs": ["E03", "E04"],
    },
    {
        "id": "SP20", "slug": "하이브", "sector_no": 20, "sector": "엔터테인먼트",
        "listed": "하이브", "corp": "하이브", "focus": "holding·exchange·연결",
        "question": "에스엠의 하이브 보유비율(H01)과 풋옵션 행사 후 증감(H02)을 합쳐, 순증가 주식수와 증가율(%)을 계산할 수 있어?",
        "reasoning": ["compute", "compare", "multi_doc"], "doc_group": "holding",
        "report": "대량보유상황보고서 · H01/H02", "basis": "해당없음",
        "task": "holding_change · multi_event_compute",
        "must_include": ["H01·H02 doc_id", "순증가 주식수·증가율(%) — verified delta only"],
        "must_not": "연결 재무표 혼입",
        "evidence": {"manifest": "corp_name=하이브"},
        "negative": "holding cross-table math",
        "notes": "Gold H01/H02.",
        "gaps": ["holding multi-event math"], "improve": "holding_event aggregate",
        "gold_refs": ["H01", "H02"],
    },
]

# Remaining sector probes (SP05–SP07, SP09–SP10, SP12–SP14, SP17, SP19): same batch, card skeleton.
_EXTRA = [
    ("SP09", 9, "통신", "SK텔레콤", "SK텔레콤", "periodic", "quarter timeseries"),
    ("SP10", 10, "게임", "크래프톤", "크래프톤", "periodic", "quarter sum vs annual"),
    ("SP12", 12, "조선", "HD현대중공업", "HD현대중공업", "periodic", "order backlog delta"),
    ("SP13", 13, "원전·에너지", "두산에너빌리티", "두산에너빌리티", "exchange", "supply count max"),
    ("SP14", 14, "방산", "한화에어로스페이스", "한화에어로스페이스", "exchange", "order qty compare"),
    ("SP17", 17, "해운", "HMM", "HMM", "periodic", "sign flip compare"),
    ("SP19", 19, "에너지", "한화솔루션", "한화솔루션", "periodic", "segment rate"),
]

BATCH = Path(__file__).resolve().parents[1] / "sector-probes-batch.txt"


def _batch_questions() -> list[str]:
    if not BATCH.is_file():
        return []
    return [
        line.strip()
        for line in BATCH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _extra_probe(
    probe_id: str,
    sector_no: int,
    sector: str,
    listed: str,
    corp: str,
    doc_group: str,
    focus: str,
    question: str,
) -> dict:
    return {
        "id": probe_id,
        "slug": corp,
        "sector_no": sector_no,
        "sector": sector,
        "listed": listed,
        "corp": corp,
        "focus": focus,
        "question": question,
        "reasoning": ["compare", "compute"],
        "doc_group": doc_group,
        "report": "manifest 확인 후 보완",
        "basis": "연결" if doc_group == "periodic" else "해당없음",
        "task": f"{doc_group} · complex_reasoning",
        "must_include": ["must_include·doc_id는 corpus 확인 후 보완"],
        "must_not": "코퍼스 밖 invent",
        "evidence": {"manifest": f"corp_name={corp}"},
        "negative": "partial / answerable_false 변형 병행",
        "notes": focus,
        "gaps": [focus],
        "improve": focus,
    }


REASONING_KO = {
    "compare": "비교",
    "compute": "연산",
    "timeseries": "시계열",
    "multi_doc": "복수문서",
}


def render(probe: dict) -> str:
    lines = [
        f"# {probe['sector_no']}. {probe['sector']} · {probe['listed']}",
        "",
        f"question_id: {probe['id']}",
        f"question: {probe['question']}",
        "",
        f"listed_name: {probe['listed']}",
        f"corp_name: {probe['corp']}",
        "",
        f"sector_no: {probe['sector_no']}",
        f"sector: {probe['sector']}",
        f"sector_focus: {probe['focus']}",
        "",
        "reasoning_types:",
    ]
    for r in probe["reasoning"]:
        lines.append(f"  - {r}  # {REASONING_KO.get(r, r)}")
    lines.extend(
        [
            "",
            f"doc_group: {probe['doc_group']}",
            f"report_nm · period: {probe['report']}",
            f"basis: {probe['basis']}",
            "",
            f"task_type (기대): {probe['task']}",
            "",
            "expected_behavior: normal_answer  # partial 또는 answerable_false 변형 카드 병행",
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
    for k, v in probe["evidence"].items():
        lines.append(f"  {k}: {v}")
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
    for g in probe["gaps"]:
        lines.append(f"  - {g}")
    lines.append("")
    lines.append(f"improve_hint (참고): {probe['improve']}")
    if probe.get("gold_refs"):
        lines.append("")
        lines.append("gold_refs:")
        for ref in probe["gold_refs"]:
            lines.append(f"  - {ref}")
    lines.extend(
        [
            "",
            "related_pipeline (taeyoon):",
            "  - comparison_frame uncertain → P0-D fail-closed 가능 (query_validation)",
            "  - clarification → think_trace.query_understanding (P0-D)",
            "",
            "notes: |",
            "  공시 확인 후 must_include·doc_id 보완.",
        ]
    )
    return "\n".join(lines) + "\n"


def all_probes() -> list[dict]:
    questions = _batch_questions()
    extras: list[dict] = []
    for index, (probe_id, sector_no, sector, listed, corp, doc_group, focus) in enumerate(
        _EXTRA
    ):
        q_index = sector_no - 1
        question = questions[q_index] if q_index < len(questions) else listed
        extras.append(
            _extra_probe(probe_id, sector_no, sector, listed, corp, doc_group, focus, question)
        )
    detailed_ids = {probe["id"] for probe in PROBES}
    merged = list(PROBES)
    for probe in extras:
        if probe["id"] not in detailed_ids:
            merged.append(probe)
    merged.sort(key=lambda item: item["sector_no"])
    return merged


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for probe in all_probes():
        slug = re.sub(r"[^\w가-힣]+", "", probe["slug"])[:16]
        path = OUT / f"{probe['id'].lower()}-{slug}.yaml"
        path.write_text(render(probe), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
