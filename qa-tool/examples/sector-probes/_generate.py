#!/usr/bin/env python3
"""sector-complex-reasoning-probes → qa-tool examples YAML (로컬 재생성용)."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent

PROBES = [
    {
        "id": "SP01",
        "sector_no": 1,
        "sector": "반도체·전자부품",
        "listed": "삼성전자",
        "corp": "삼성전자",
        "focus": "연결 손익·부문·Capa",
        "question": "삼성전자 2023·2024 사업보고서 연결 매출액을 비교해서 증가율(%)을 계산해줘",
        "reasoning": ["compare", "compute", "multi_doc"],
        "doc_group": "periodic",
        "report": "사업보고서 (2023.12 · 2024.12) · FY2023·FY2024",
        "basis": "연결",
        "task": "periodic_fact · financial_metric · comparative · derived_rate · multi_doc",
        "must_include": [
            "report_nm: 사업보고서 (2023.12) · 사업보고서 (2024.12) — 2건",
            "재무제표 기준: 연결",
            "section: 연결 손익계산서 > 매출액",
            "2023·2024 연결 매출액 (공시 숫자)",
            "증가율(%) — 표 숫자로만 계산",
            "단위: 백만원 등 공시 그대로",
        ],
        "must_not": "코퍼스 밖·성장성 해석·별도 혼입·단일 연도만·LLM 임의 증가율",
        "evidence": {
            "doc_id": "periodic_(2023·2024 annual)",
            "section_path": "연결 손익계산서 > 매출액",
            "manifest": "corp_name=삼성전자, doc_subtype=annual",
        },
        "negative": "매출 단일행 1건만 · multi-doc·derived rate 미지원",
        "notes": "① 각 annual 연결 매출 ② 증가율 산술 ③ 단위. taeyoon: comparison 파싱됨, derived compute는 미구현.",
        "gaps": ["multi-doc retrieval", "derived_metric(rate)"],
        "improve": "comparative + derived_metric(rate) · 2 doc_id 바인딩",
    },
    {
        "id": "SP02",
        "sector_no": 2,
        "sector": "자동차·모빌리티",
        "listed": "현대차",
        "corp": "현대자동차",
        "focus": "연결 손익·부문·전동화",
        "question": "현대자동차 2025 사업보고서 연결 기준 영업이익률(영업이익÷매출액)이 전기 대비 몇 %p 변했어?",
        "reasoning": ["compare", "compute"],
        "doc_group": "periodic",
        "report": "사업보고서 (2025.12) · FY2025",
        "basis": "연결",
        "task": "periodic_fact · financial_metric · comparative · derived_ratio · delta_pp",
        "must_include": [
            "당기·전기 매출·영업이익 (연결)",
            "당기·전기 영업이익률(%)",
            "%p 변화",
        ],
        "must_not": "별도 표 혼동·주관적 해석",
        "evidence": {
            "doc_id": "periodic_(2025 annual)",
            "section_path": "연결 손익계산서",
            "manifest": "corp_name=현대자동차",
        },
        "negative": "영업이익 단일행 · %p 파생 미등록",
        "notes": "한 표 4셀+2비율+%p.",
        "gaps": ["derived_ratio", "delta_pp"],
        "improve": "same-table 4-cell atomic fact",
    },
    {
        "id": "SP18",
        "sector_no": 18,
        "sector": "전력기기",
        "listed": "LS ELECTRIC",
        "corp": "엘에스일렉트릭",
        "focus": "exchange 시설투자",
        "question": "LS ELECTRIC 2024년 초고압 변압기 시설투자 금액(Gold E03)과 자기자본 대비 비율, 종료일(E04)까지 한 번에",
        "reasoning": ["compute"],
        "doc_group": "exchange",
        "report": "신규시설투자등 · rcept_dt=20240521",
        "basis": "해당없음",
        "task": "facility_investment · multi_field_single_event",
        "must_include": [
            "doc_id: exchange_20240521800037",
            "투자금액 (Gold E03)",
            "자기자본 대비(%)",
            "투자종료일 (Gold E04)",
        ],
        "must_not": "다른 시설투자 건 혼동",
        "evidence": {
            "doc_id": "exchange_20240521800037",
            "rcept_dt": "20240521",
            "section_path": "신규시설투자",
            "manifest": "corp_name=엘에스일렉트릭",
        },
        "negative": "양성 벤치 — 단건 multi-field",
        "notes": "Gold E03/E04 검증.",
        "gaps": [],
        "improve": "양성 패턴을 comparative로 확장",
        "gold_refs": ["E03", "E04"],
    },
    {
        "id": "SP20",
        "sector_no": 20,
        "sector": "엔터테인먼트",
        "listed": "하이브",
        "corp": "하이브",
        "focus": "holding·exchange·연결",
        "question": "에스엠의 하이브 보유비율(H01)과 풋옵션 행사 후 증감(H02)을 합쳐, 순증가 주식수와 증가율(%)을 계산할 수 있어?",
        "reasoning": ["compute", "compare", "multi_doc"],
        "doc_group": "holding",
        "report": "대량보유상황보고서 · H01/H02",
        "basis": "해당없음",
        "task": "holding_change · multi_event_compute",
        "must_include": [
            "H01·H02 doc_id 각각",
            "순증가 주식수·증가율(%) — verified delta only",
        ],
        "must_not": "연결 재무표 혼입·보고자/피투자 혼동",
        "evidence": {
            "doc_id": "holding_20240314001102 + H02",
            "section_path": "보유주식수·보유비율",
            "manifest": "corp_name=하이브",
        },
        "negative": "holding cross-table math",
        "notes": "Gold H01/H02.",
        "gaps": ["holding multi-event math"],
        "improve": "holding_event aggregate",
        "gold_refs": ["H01", "H02"],
    },
]

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
        lines.append(f"  - {r}  # {REASONING_KO[r]}")
    lines.extend(
        [
            "",
            f"doc_group: {probe['doc_group']}",
            f"report_nm · period: {probe['report']}",
            f"basis: {probe['basis']}",
            "",
            f"task_type (기대): {probe['task']}",
            "",
            "expected_behavior: normal_answer  # partial 또는 answerable_false 변형 카드 별도",
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for probe in PROBES:
        path = OUT / f"{probe['id'].lower()}-{probe['corp'][:8].replace(' ', '')}.yaml"
        path.write_text(render(probe), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
