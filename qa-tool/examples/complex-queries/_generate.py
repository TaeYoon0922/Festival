#!/usr/bin/env python3
"""복합·정정·다기간 QA 질문 → qa-tool examples YAML (로컬 재생성용)."""

from __future__ import annotations

import re
from pathlib import Path

OUT = Path(__file__).resolve().parent
BATCH = Path(__file__).resolve().parents[1] / "complex-queries-batch.txt"

# (id, slug, theme, corps, focus, doc_groups, task_tags, must_include, gaps, notes)
PROBES = [
    (
        "CQ01",
        "KB금융-정정당기순이익",
        "정정공시 · 금융 periodic",
        ("KB금융",),
        "사업보고서 기재정정 전·후 당기순이익",
        ("periodic",),
        "correction · financial_metric · before_after",
        [
            "정정 전·후 연결 당기순이익(공시 수치)",
            "변동액·변동률(파생)",
            "최종(정정 후) 수치 명시",
        ],
        ["P0-A correction_graph", "canonical vs superseded doc"],
        "KB금융 2025 [기재정정] 사업보고서 — corpus에 correction 병행 수집.",
    ),
    (
        "CQ02",
        "LGES-정정시설투자",
        "정정공시 · exchange 시설투자",
        ("LG에너지솔루션",),
        "신규시설투자 정정 건 최종 합계 vs 정정 전",
        ("exchange",),
        "correction · facility_investment · exchange_aggregate · re-aggregate",
        [
            "정정 체인 최종 investment_amount 합(파생)",
            "정정 전 합(파생)",
            "차이(파생)",
        ],
        ["correction dedupe in enumeration", "latest canonical amount"],
        "exchange 기재정정 631건 중 시설투자 subset.",
    ),
    (
        "CQ03",
        "삼성전자vsSK하이닉스-3개년CAGR",
        "2사 × 3개년 periodic",
        ("삼성전자", "SK하이닉스"),
        "3개년 매출 CAGR peer compare",
        ("periodic",),
        "cross_company · financial_metric · multi_period · CAGR · multi_corp",
        [
            "각사 2023·2024·2025 연결 매출액",
            "2023→2025 CAGR(파생) per corp",
            "더 높은 쪽",
        ],
        ["multi-year dual-corp retrieval", "period-scoped table projection"],
        "3개년 × 2사 = 6 filing surfaces.",
    ),
    (
        "CQ04",
        "HD현대중공업vs한화오션-연도별exchange",
        "2사 × 3개년 exchange",
        ("HD현대중공업", "한화오션"),
        "연도별 공급·수주 계약금액 dual-corp matrix",
        ("exchange",),
        "cross_company · supply_contract · exchange_aggregate · receipt_date · multi_corp",
        [
            "각사·각 연도(2023·2024·2025) 합계(파생)",
            "최대 연도·최대 corp 식별",
        ],
        ["multi-year receipt_date span", "dual-corp P0-C enumeration"],
        "조선 sector exchange 풍부.",
    ),
    (
        "CQ05",
        "한화오cean-정정1분기",
        "정정공시 · 분기 periodic",
        ("한화오cean",),
        "1분기 분기보고서 정정 전·후 IS diff",
        ("periodic",),
        "correction · financial_metric · quarter · before_after",
        [
            "정정 전·후 연결 매출·영업이익",
            "변동액·변동률(파생)",
        ],
        ["quarterly correction chain", "temporal + correction resolution"],
        "한화오cean [기재정정] 분기보고서(2024.03) corpus 명시.",
    ),
    (
        "CQ06",
        "크래프톤vs엔씨-분기합vs연간",
        "2사 분기합 vs 연간",
        ("크래프톤", "엔씨소프트"),
        "4분기 영업이익 합 vs 사업보고서 diff peer",
        ("periodic",),
        "cross_company · quarter_sum_vs_annual · multi_corp",
        [
            "각사 4분기 영업이익 합(파생)",
            "각사 2024 사업보고서 영업이익",
            "차이(파생) · 더 큰 차이 corp",
        ],
        ["dual-corp quarter_sum_vs_annual", "annual vs quarter subtype filter"],
        "PC18 패턴 2사 확장.",
    ),
    (
        "CQ07",
        "삼성SDIvsLGES-투자비중",
        "2사 cross-domain ratio",
        ("삼성SDI", "LG에너지솔루션"),
        "시설투자 합 / 자산총계 비중 peer",
        ("exchange", "periodic"),
        "cross_company · facility_investment · balance_ratio · multi_corp",
        [
            "각사 2024 시설투자 합(파생)",
            "각사 2024 연결 자산총계",
            "투자비중 %(파생) · 더 높은 쪽",
        ],
        ["cross-domain ratio", "exchange + periodic dual retrieval"],
        "exchange aggregate + periodic BS same year.",
    ),
    (
        "CQ08",
        "국민연금-삼성전자-정정보유",
        "정정 · 지분공시 복합연산",
        ("삼성전자",),
        "대량보유보고서 정정 + 순증감·증감률",
        ("holding",),
        "correction · holding_change · multi_event_compute",
        [
            "최종 보유주식수·보유비율",
            "직전 대비 순증감·증감률(파생)",
            "정정 반영 최신 보고 기준",
        ],
        ["holding correction chain", "reporter-scoped retrieval"],
        "국민연금 → 삼성전자 보유비율 시계열 + 정정.",
    ),
    (
        "CQ09",
        "HMMvs현대글로비스-부호증감",
        "2사 sign-flip multi-year",
        ("HMM", "현대글로비스"),
        "당기순이익 부호·증감률 peer compare",
        ("periodic",),
        "cross_company · sign_flip · peer_rate · multi_period · multi_corp",
        [
            "각사 2023·2024 당기순이익·부호",
            "전기 대비 증감률(파생)",
            "2024 개선폭 더 큰 corp",
        ],
        ["dual-corp sign_flip", "multi-year periodic compare"],
        "SP17 패턴 2사 확장.",
    ),
    (
        "CQ10",
        "두산에너빌리티vs한전기술-정정원전",
        "2사 정정 dedupe exchange",
        ("두산에너빌리티", "한전기술"),
        "원전 공급계약 정정 반영 aggregate peer",
        ("exchange",),
        "cross_company · correction · supply_contract · exchange_aggregate · multi_corp",
        [
            "각사 최종 건수·합계·max(파생)",
            "정정 전 합 vs 정정 후 합 차이(파생)",
        ],
        ["correction-aware enumeration", "dual-corp exchange aggregate"],
        "PC23·24 + correction layer.",
    ),
]


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^\w가-힣]+", "-", value.strip())
    return cleaned.strip("-") or "probe"


def _yaml_block(title: str, lines: list[str]) -> str:
    if not lines:
        return f"{title}: []\n"
    body = "\n".join(f"  - {line}" for line in lines)
    return f"{title}:\n{body}\n"


def render_card(probe: tuple) -> str:
    (
        qid,
        _slug_name,
        theme,
        corps,
        focus,
        doc_groups,
        task_tags,
        must_include,
        gaps,
        notes,
    ) = probe
    questions = [
        line.strip()
        for line in BATCH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index = int(qid[2:]) - 1
    question = questions[index]
    corp_lines = "\n".join(f"corp_name: {corp}" for corp in corps[:1])
    peer = ""
    if len(corps) > 1:
        peer = (
            f"peer_listed_name: {corps[1]}\n"
            f"peer_corp_name: {corps[1]}\n"
        )
    return f"""# {theme}

question_id: {qid}
question: {question}

listed_name: {corps[0]}
{corp_lines}{peer}
theme: {theme}
focus: {focus}

reasoning_types:
  - compare
  - compute
  - multi_doc
  - correction

doc_group: {' · '.join(doc_groups)}
task_type (기대): {task_tags}

expected_behavior: stretch_goal  # partial_answer 또는 answerable_false 변형 카드 병행

{_yaml_block('must_include_in_answer', must_include)}
must_NOT_invent:
  - 정정 전 수치를 최종값으로 혼동
  - 코퍼스 밖 invent · 두 기업 숫자 혼동

{_yaml_block('related_agent_gaps', gaps)}

notes: |
  {notes}
"""


def main() -> None:
    questions = [
        line.strip()
        for line in BATCH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(questions) != len(PROBES):
        raise SystemExit(f"expected {len(PROBES)} questions, found {len(questions)}")
    OUT.mkdir(parents=True, exist_ok=True)
    for probe in PROBES:
        qid = probe[0]
        path = OUT / f"{qid.lower()}-{_slug(probe[1])}.yaml"
        path.write_text(render_card(probe), encoding="utf-8")
        print("wrote", path.name)


if __name__ == "__main__":
    main()
