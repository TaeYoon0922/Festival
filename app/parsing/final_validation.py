"""Final pre-full-corpus validation for structural disclosure chunking."""

from __future__ import annotations

import csv
import gzip
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.parsing.chunking import (
    CHUNKING_VERSION,
    DOCUMENT_METADATA_FIELDS,
    REQUIRED_CHUNK_FIELDS,
    _BASIS_PERIOD,
    _find_unit,
    _table_chunks_for_section,
    build_chunks,
    build_legacy_chunks,
    get_chunking_strategy,
)
from app.parsing.dart_xml import parse_dart_document
from app.parsing.models import ParsedDocument, Table
from app.parsing.sampling import (
    GROUPS,
    load_manifest,
    resolve_primary_xml,
    resolve_unicode_path,
)


GOLD_QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "question_id": "P01",
        "doc_group": "periodic",
        "query": "삼성전자 DX 부문의 주요 제품은 무엇인가",
        "doc_id": "periodic_20240312000736",
        "target_type": "table",
        "target_id": "t0006",
        "evidence_terms": ["DX 부문", "TV, 모니터", "스마트폰"],
    },
    {
        "question_id": "P02",
        "doc_group": "periodic",
        "query": "삼성전자 2023년 연결대상회사 기말 수와 주요종속회사 수",
        "doc_id": "periodic_20240312000736",
        "target_type": "table",
        "target_id": "t0009",
        "evidence_terms": ["연결대상회사수", "232", "146"],
    },
    {
        "question_id": "P03",
        "doc_group": "periodic",
        "query": "삼성전자 Harman이 업계 최초 수주한 5G 전장부품",
        "doc_id": "periodic_20240312000736",
        "target_type": "text",
        "target_id": "s0102",
        "evidence_terms": ["5G TCU", "업계 최초"],
    },
    {
        "question_id": "P04",
        "doc_group": "periodic",
        "query": "삼성전자 산업안전보건법 위반 제재 내역",
        "doc_id": "periodic_20240312000736",
        "target_type": "text",
        "target_id": "s0119",
        "evidence_terms": ["산업안전보건법", "과태료"],
    },
    {
        "question_id": "P05",
        "doc_group": "periodic",
        "query": "레인보우로보틱스 기업신용평가 BBB+ 평가일",
        "doc_id": "periodic_20240814003228",
        "target_type": "table",
        "target_id": "t0011",
        "evidence_terms": ["2023.06.05", "BBB+", "이크레더블"],
    },
    {
        "question_id": "P06",
        "doc_group": "periodic",
        "query": "레인보우로보틱스 HUBO 이족보행 로봇 사업 설명",
        "doc_id": "periodic_20240814003228",
        "target_type": "text",
        "target_id": "s0011",
        "evidence_terms": ["HUBO", "이족보행"],
    },
    {
        "question_id": "P07",
        "doc_group": "periodic",
        "query": "두산퓨얼셀 2023년 1분기 연료전지 주기기 매출액",
        "doc_id": "periodic_20230512001368",
        "target_type": "table",
        "target_id": "t0006",
        "evidence_terms": ["연료전지 주기기", "23,848"],
    },
    {
        "question_id": "P08",
        "doc_group": "periodic",
        "query": "두산퓨얼셀 익산공장 1분기 생산능력 생산실적 평균가동률",
        "doc_id": "periodic_20230512001368",
        "target_type": "table",
        "target_id": "t0010",
        "evidence_terms": ["생산능력", "58.0", "56%"],
    },
    {
        "question_id": "P09",
        "doc_group": "periodic",
        "query": "두산퓨얼셀 PAFC 연료전지 기술의 특징",
        "doc_id": "periodic_20230512001368",
        "target_type": "text",
        "target_id": "s0017",
        "evidence_terms": ["PAFC", "고효율"],
    },
    {
        "question_id": "P10",
        "doc_group": "periodic",
        "query": "시프트업 유가증권시장 상장일",
        "doc_id": "periodic_20250319000952",
        "target_type": "table",
        "target_id": "t0010",
        "evidence_terms": ["유가증권시장 상장", "2024년 07월 11일"],
    },
    {
        "question_id": "P11",
        "doc_group": "periodic",
        "query": "시프트업 테이블원 흡수합병 합병기일",
        "doc_id": "periodic_20250319000952",
        "target_type": "table",
        "target_id": "t0013",
        "evidence_terms": ["주식회사 테이블원", "합병기일", "2024년 12월 5일"],
    },
    {
        "question_id": "P12",
        "doc_group": "periodic",
        "query": "시프트업 승리의 여신 니케 게임 장르와 출시 국가",
        "doc_id": "periodic_20250319000952",
        "target_type": "text",
        "target_id": "s0012",
        "evidence_terms": ["수집형 RPG", "153개국"],
    },
    {
        "question_id": "P13",
        "doc_group": "periodic",
        "query": "한미반도체 TC BONDER 반도체 제조 장비 사업",
        "doc_id": "periodic_20230512000911",
        "target_type": "text",
        "target_id": "s0011",
        "evidence_terms": ["TC BONDER", "반도체"],
    },
    {
        "question_id": "P14",
        "doc_group": "periodic",
        "query": "한미반도체 장비제조판매 수익 핵심감사사항",
        "doc_id": "periodic_20230512000911",
        "target_type": "text",
        "target_id": "s0037",
        "evidence_terms": ["장비제조판매 수익", "핵심감사사항"],
    },
    {
        "question_id": "M01",
        "doc_group": "major",
        "query": "두산로보틱스 유상증자 보통주 신주 수",
        "doc_id": "major_20230823000334",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["신주의 종류와 수", "16,200,000"],
    },
    {
        "question_id": "M02",
        "doc_group": "major",
        "query": "두산로보틱스 유상증자 시설자금 운영자금 조달액",
        "doc_id": "major_20230823000334",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["시설자금", "31,000,000,000", "운영자금"],
    },
    {
        "question_id": "M03",
        "doc_group": "major",
        "query": "셀트리온 합병 존속회사와 소멸회사",
        "doc_id": "major_20230817000203",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["존속회사", "셀트리온헬스케어", "흡수합병"],
    },
    {
        "question_id": "M04",
        "doc_group": "major",
        "query": "셀트리온 합병 목적 원가경쟁력 거래구조 단순화",
        "doc_id": "major_20230817000203",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["합병목적", "원가경쟁력", "거래구조 단순화"],
    },
    {
        "question_id": "M05",
        "doc_group": "major",
        "query": "하이브 전환사채 권면총액과 채무상환자금",
        "doc_id": "major_20241015000082",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["400,000,000,000", "채무상환자금"],
    },
    {
        "question_id": "M06",
        "doc_group": "major",
        "query": "하이브 전환사채 만기일과 표면이자율",
        "doc_id": "major_20241015000082",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["2029년 10월 17일", "표면이자율", "0.0"],
    },
    {
        "question_id": "M07",
        "doc_group": "major",
        "query": "KT 자기주식 처분 예정 보통주 수와 가격",
        "doc_id": "major_20230511000446",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["131,690", "31,350", "처분 대상 주식가격"],
    },
    {
        "question_id": "M08",
        "doc_group": "major",
        "query": "KT 자기주식 처분 목적 장기성과급 주식보상",
        "doc_id": "major_20230511000446",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["장기성과급", "사외이사 주식보상"],
    },
    {
        "question_id": "M09",
        "doc_group": "major",
        "query": "한화에어로스페이스 분할신설회사 분할비율",
        "doc_id": "major_20240405000022",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["분할신설회사", "0.0997203"],
    },
    {
        "question_id": "M10",
        "doc_group": "major",
        "query": "한화에어로스페이스 분할대상 시큐리티 칩마운터 반도체장비",
        "doc_id": "major_20240405000022",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["시큐리티", "칩마운터", "반도체장비"],
    },
    {
        "question_id": "E01",
        "doc_group": "exchange",
        "query": "한미반도체 대만 장비 수주 계약금액",
        "doc_id": "exchange_20230612800470",
        "target_type": "table",
        "target_id": "t0001",
        "evidence_terms": ["1,417,130,000", "대만"],
    },
    {
        "question_id": "E02",
        "doc_group": "exchange",
        "query": "한미반도체 계약 상대방 Unimicron 계약기간",
        "doc_id": "exchange_20230612800470",
        "target_type": "table",
        "target_id": "t0001",
        "evidence_terms": ["Unimicron Technology Corp.", "2023-09-20"],
    },
    {
        "question_id": "E03",
        "doc_group": "exchange",
        "query": "LS ELECTRIC 초고압 변압기 시설 투자금액",
        "doc_id": "exchange_20240521800037",
        "target_type": "table",
        "target_id": "t0001",
        "evidence_terms": ["초고압 변압기", "80,300,000,000"],
    },
    {
        "question_id": "E04",
        "doc_group": "exchange",
        "query": "LS ELECTRIC 시설투자 종료일과 투자목적",
        "doc_id": "exchange_20240521800037",
        "target_type": "table",
        "target_id": "t0001",
        "evidence_terms": ["2025-09-30", "수주 증가 물량 대응"],
    },
    {
        "question_id": "E05",
        "doc_group": "exchange",
        "query": "하이브 방탄소년단 멤버 전속계약 재계약",
        "doc_id": "exchange_20230920800479",
        "target_type": "table",
        "target_id": "t0001",
        "evidence_terms": ["방탄소년단", "멤버 7인", "재계약"],
    },
    {
        "question_id": "E06",
        "doc_group": "exchange",
        "query": "한미약품 제넥신 코로나 백신 계약 해지금액",
        "doc_id": "exchange_20230227800485",
        "target_type": "table",
        "target_id": "t0001",
        "evidence_terms": ["제넥신", "23,419,586,850"],
    },
    {
        "question_id": "E07",
        "doc_group": "exchange",
        "query": "한미약품 GX-19N 계약 해지 주요 사유",
        "doc_id": "exchange_20230227800485",
        "target_type": "table",
        "target_id": "t0001",
        "evidence_terms": ["GX-19N", "개발 중단"],
    },
    {
        "question_id": "E08",
        "doc_group": "exchange",
        "query": "한화에어로스페이스 호주 레드백 장갑차 우선협상 수량",
        "doc_id": "exchange_20230727800097",
        "target_type": "table",
        "target_id": "t0001",
        "evidence_terms": ["레드백 장갑차", "129대"],
    },
    {
        "question_id": "H01",
        "doc_group": "holding",
        "query": "에스엠 하이브 이번 보고 보유 주식수와 비율",
        "doc_id": "holding_20240314001102",
        "target_type": "table",
        "target_id": "t0002",
        "evidence_terms": ["2,967,759", "12.45"],
    },
    {
        "question_id": "H02",
        "doc_group": "holding",
        "query": "에스엠 하이브 풋옵션 주식 취득 증감 수량",
        "doc_id": "holding_20240314001102",
        "target_type": "table",
        "target_id": "t0019",
        "evidence_terms": ["풋옵션권리행사", "868,948", "2,967,759"],
    },
    {
        "question_id": "H03",
        "doc_group": "holding",
        "query": "효성중공업 국민연금 이번보고서 보유 수와 비율",
        "doc_id": "holding_20230404000489",
        "target_type": "table",
        "target_id": "t0012",
        "evidence_terms": ["655,490", "7.03", "국민연금공단"],
    },
    {
        "question_id": "H04",
        "doc_group": "holding",
        "query": "파마리서치 국민연금 보유주식 증가 후 수량 비율",
        "doc_id": "holding_20230103000120",
        "target_type": "table",
        "target_id": "t0012",
        "evidence_terms": ["720,039", "7.12", "106,281"],
    },
    {
        "question_id": "H05",
        "doc_group": "holding",
        "query": "이마트 국민연금 이번보고 보유 수와 비율",
        "doc_id": "holding_20230704000318",
        "target_type": "table",
        "target_id": "t0012",
        "evidence_terms": ["2,202,050", "7.90"],
    },
    {
        "question_id": "H06",
        "doc_group": "holding",
        "query": "이마트 국민연금 직전보고 대비 감소 주식수",
        "doc_id": "holding_20230704000318",
        "target_type": "table",
        "target_id": "t0012",
        "evidence_terms": ["-283,151", "-1.02"],
    },
    {
        "question_id": "H07",
        "doc_group": "holding",
        "query": "LG생활건강 국민연금 이번보고 보유 수와 비율",
        "doc_id": "holding_20230704000260",
        "target_type": "table",
        "target_id": "t0012",
        "evidence_terms": ["1,092,455", "6.99"],
    },
    {
        "question_id": "H08",
        "doc_group": "holding",
        "query": "LG생활건강 국민연금 보유 감소 수량과 비율",
        "doc_id": "holding_20230704000260",
        "target_type": "table",
        "target_id": "t0012",
        "evidence_terms": ["-162,143", "-1.04"],
    },
)


# Separate expansion set. GOLD_QUESTIONS above remains the fixed 40-question baseline.
HOLDING_ADDITIONAL_QUESTIONS: tuple[dict[str, Any], ...] = (
    {"question_id": "HX01", "doc_group": "holding", "query": "에스엠 하이브 2024년 3월 14일 현재 보유 수량 비율", "doc_id": "holding_20240314001102", "target_type": "table", "target_id": "t0012", "evidence_terms": ["2024년 03월 14일", "2,967,759", "12.45"]},
    {"question_id": "HX02", "doc_group": "holding", "query": "에스엠 하이브 직전보고 보유주식 수 비율", "doc_id": "holding_20240314001102", "target_type": "table", "target_id": "t0012", "evidence_terms": ["2,098,811", "8.81"]},
    {"question_id": "HX03", "doc_group": "holding", "query": "에스엠 하이브 보유주식 증가 수량 증가 비율", "doc_id": "holding_20240314001102", "target_type": "table", "target_id": "t0012", "evidence_terms": ["868,948", "3.64"]},
    {"question_id": "HX04", "doc_group": "holding", "query": "에스엠 하이브 풋옵션 행사 주식 취득일과 취득 수량", "doc_id": "holding_20240314001102", "target_type": "table", "target_id": "t0019", "evidence_terms": ["2024.03.07", "풋옵션권리행사", "868,948"]},
    {"question_id": "HX05", "doc_group": "holding", "query": "효성중공업 국민연금 2023년 3월 7일 보유 수량 비율", "doc_id": "holding_20230404000489", "target_type": "table", "target_id": "t0012", "evidence_terms": ["2023년 03월 07일", "655,490", "7.03"]},
    {"question_id": "HX06", "doc_group": "holding", "query": "효성중공업 국민연금 직전보고 보유주식 수 비율", "doc_id": "holding_20230404000489", "target_type": "table", "target_id": "t0012", "evidence_terms": ["555,510", "5.96"]},
    {"question_id": "HX07", "doc_group": "holding", "query": "효성중공업 국민연금 증가 주식수 증가 비율", "doc_id": "holding_20230404000489", "target_type": "table", "target_id": "t0012", "evidence_terms": ["99,980", "1.07"]},
    {"question_id": "HX08", "doc_group": "holding", "query": "효성중공업 국민연금기금 변동일 변동후 주식수", "doc_id": "holding_20230404000489", "target_type": "table", "target_id": "t0019", "evidence_terms": ["국민연금기금", "2023년 03월 07일", "655,490"]},
    {"question_id": "HX09", "doc_group": "holding", "query": "파마리서치 국민연금 2022년 12월 5일 현재 보유 비율", "doc_id": "holding_20230103000120", "target_type": "table", "target_id": "t0012", "evidence_terms": ["2022년 12월 05일", "720,039", "7.12"]},
    {"question_id": "HX10", "doc_group": "holding", "query": "파마리서치 국민연금 직전보고 수량 지분율", "doc_id": "holding_20230103000120", "target_type": "table", "target_id": "t0012", "evidence_terms": ["613,758", "6.07"]},
    {"question_id": "HX11", "doc_group": "holding", "query": "파마리서치 국민연금 증가 주식수 증가율", "doc_id": "holding_20230103000120", "target_type": "table", "target_id": "t0012", "evidence_terms": ["106,281", "1.05"]},
    {"question_id": "HX12", "doc_group": "holding", "query": "파마리서치 국민연금기금 변동일 변동전 변동후", "doc_id": "holding_20230103000120", "target_type": "table", "target_id": "t0019", "evidence_terms": ["2022년 12월 05일", "613,758", "720,039"]},
    {"question_id": "HX13", "doc_group": "holding", "query": "이마트 국민연금 2023년 6월 13일 보유 수와 비율", "doc_id": "holding_20230704000318", "target_type": "table", "target_id": "t0012", "evidence_terms": ["2023년 06월 13일", "2,202,050", "7.90"]},
    {"question_id": "HX14", "doc_group": "holding", "query": "이마트 국민연금 직전보고 보유주식 수 비율", "doc_id": "holding_20230704000318", "target_type": "table", "target_id": "t0012", "evidence_terms": ["2,485,201", "8.92"]},
    {"question_id": "HX15", "doc_group": "holding", "query": "이마트 국민연금 감소 주식수 감소 비율", "doc_id": "holding_20230704000318", "target_type": "table", "target_id": "t0012", "evidence_terms": ["-283,151", "-1.02"]},
    {"question_id": "HX16", "doc_group": "holding", "query": "이마트 국민연금기금 변동일 감소 후 주식수", "doc_id": "holding_20230704000318", "target_type": "table", "target_id": "t0019", "evidence_terms": ["2023년 06월 13일", "-283,151", "2,202,050"]},
    {"question_id": "HX17", "doc_group": "holding", "query": "LG생활건강 국민연금 2023년 6월 30일 보유 수와 비율", "doc_id": "holding_20230704000260", "target_type": "table", "target_id": "t0012", "evidence_terms": ["2023년 06월 30일", "1,092,455", "6.99"]},
    {"question_id": "HX18", "doc_group": "holding", "query": "LG생활건강 국민연금 직전보고 보유주식 수 비율", "doc_id": "holding_20230704000260", "target_type": "table", "target_id": "t0012", "evidence_terms": ["1,254,598", "8.03"]},
    {"question_id": "HX19", "doc_group": "holding", "query": "LG생활건강 국민연금 감소 주식수 감소 비율", "doc_id": "holding_20230704000260", "target_type": "table", "target_id": "t0012", "evidence_terms": ["-162,143", "-1.04"]},
    {"question_id": "HX20", "doc_group": "holding", "query": "LG생활건강 국민연금기금 변동일 감소 후 주식수", "doc_id": "holding_20230704000260", "target_type": "table", "target_id": "t0019", "evidence_terms": ["2023년 06월 30일", "-162,143", "1,092,455"]},
)


@dataclass
class PilotRecord:
    row: dict[str, Any]
    parsed: ParsedDocument
    structural_chunks: list[dict[str, Any]]
    legacy_chunks: list[dict[str, Any]]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _length_metrics(chunks: Iterable[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(str(chunk["content"])) for chunk in chunks]
    total = len(lengths)

    def ratio(predicate: Any) -> float:
        return round(sum(predicate(value) for value in lengths) / total, 6) if total else 0

    return {
        "chunk_count": total,
        "min": min(lengths, default=0),
        "mean": round(statistics.mean(lengths), 3) if lengths else 0,
        "median": round(statistics.median(lengths), 3) if lengths else 0,
        "p90": _percentile(lengths, 0.90),
        "p95": _percentile(lengths, 0.95),
        "p99": _percentile(lengths, 0.99),
        "max": max(lengths, default=0),
        "le_100_count": sum(value <= 100 for value in lengths),
        "le_100_ratio": ratio(lambda value: value <= 100),
        "le_200_count": sum(value <= 200 for value in lengths),
        "le_200_ratio": ratio(lambda value: value <= 200),
        "gt_1500_count": sum(value > 1_500 for value in lengths),
        "gt_1500_ratio": ratio(lambda value: value > 1_500),
    }


def build_length_report(records: list[PilotRecord]) -> dict[str, Any]:
    all_chunks = [chunk for record in records for chunk in record.structural_chunks]
    chunk_types = ("text", "table", "table_projection")
    by_type = {
        kind: _length_metrics(
            chunk for chunk in all_chunks if chunk["chunk_type"] == kind
        )
        for kind in chunk_types
    }
    by_group = {
        group: _length_metrics(
            chunk
            for record in records
            if record.row["doc_group"] == group
            for chunk in record.structural_chunks
        )
        for group in GROUPS
    }
    by_group_and_type = {
        group: {
            kind: _length_metrics(
                chunk
                for record in records
                if record.row["doc_group"] == group
                for chunk in record.structural_chunks
                if chunk["chunk_type"] == kind
            )
            for kind in chunk_types
        }
        for group in GROUPS
    }
    short_texts = sorted(
        (
            chunk
            for chunk in all_chunks
            if chunk["chunk_type"] == "text" and len(str(chunk["content"])) <= 100
        ),
        key=lambda chunk: (len(str(chunk["content"])), str(chunk["chunk_id"])),
    )
    return {
        "scope": "20-document non-correction structural pilot",
        "previous_v2_0_text_ratios": {"le_100_ratio": 0.530256, "le_200_ratio": 0.681026},
        "overall": _length_metrics(all_chunks),
        "by_chunk_type": by_type,
        "by_doc_group": by_group,
        "by_doc_group_and_type": by_group_and_type,
        "merged_context_label_count": sum(
            len(chunk.get("context_labels") or []) for chunk in all_chunks
        ),
        "short_text_samples": [
            {
                "chunk_id": chunk["chunk_id"],
                "doc_group": chunk["doc_group"],
                "corp_name": chunk["corp_name"],
                "section_path": " > ".join(chunk["section_path"]),
                "char_count": chunk["char_count"],
                "context_labels": chunk.get("context_labels") or [],
                "content": chunk["content"],
            }
            for chunk in short_texts[:30]
        ],
    }


def _length_markdown(report: dict[str, Any]) -> str:
    fields = ("chunk_count", "min", "mean", "median", "p90", "p95", "p99", "max")
    ratio_fields = ("le_100_ratio", "le_200_ratio", "gt_1500_ratio")
    rows: list[tuple[str, dict[str, Any]]] = [("overall", report["overall"])]
    rows.extend((f"type:{key}", value) for key, value in report["by_chunk_type"].items())
    rows.extend((f"group:{key}", value) for key, value in report["by_doc_group"].items())
    lines = [
        "# Chunk length validation",
        "",
        f"Scope: {report['scope']}",
        "",
        "| scope | count | min | mean | median | p90 | p95 | p99 | max | <=100 | <=200 | >1500 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in rows:
        values = [str(metrics[field]) for field in fields]
        ratios = [f"{metrics[field] * 100:.2f}%" for field in ratio_fields]
        lines.append(f"| {name} | {' | '.join([*values, *ratios])} |")
    lines.extend(
        [
            "",
            "## Group × type",
            "",
            "| scope | count | min | mean | median | p90 | p95 | p99 | max | <=100 | <=200 | >1500 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, kinds in report["by_doc_group_and_type"].items():
        for kind, metrics in kinds.items():
            values = [str(metrics[field]) for field in fields]
            ratios = [f"{metrics[field] * 100:.2f}%" for field in ratio_fields]
            lines.append(f"| {group}:{kind} | {' | '.join([*values, *ratios])} |")
    current_text = report["by_chunk_type"]["text"]
    previous = report["previous_v2_0_text_ratios"]
    lines.extend(
        [
            "",
            "## Short text change",
            "",
            f"- <=100: {previous['le_100_ratio'] * 100:.2f}% → {current_text['le_100_ratio'] * 100:.2f}%",
            f"- <=200: {previous['le_200_ratio'] * 100:.2f}% → {current_text['le_200_ratio'] * 100:.2f}%",
            f"- merged context labels: {report['merged_context_label_count']}",
            "",
            "## Remaining <=100-character text samples",
            "",
        ]
    )
    for item in report["short_text_samples"]:
        lines.append(
            f"- `{item['chunk_id']}` ({item['char_count']} chars): "
            f"{str(item['content']).replace(chr(10), ' / ')}"
        )
    return "\n".join(lines)


def build_extreme_table_report(records: list[PilotRecord]) -> dict[str, Any]:
    originals = [
        chunk
        for record in records
        for chunk in record.structural_chunks
        if chunk["chunk_type"] == "table"
    ]
    projections = [
        chunk
        for record in records
        for chunk in record.structural_chunks
        if chunk.get("projection_type") == "extreme_table_row"
    ]
    extreme = sorted(
        (chunk for chunk in originals if chunk["char_count"] > 5_000),
        key=lambda chunk: (-int(chunk["char_count"]), str(chunk["chunk_id"])),
    )
    return {
        "table_chunk_count": len(originals),
        "gt_5000_count": sum(chunk["char_count"] > 5_000 for chunk in originals),
        "gt_10000_count": sum(chunk["char_count"] > 10_000 for chunk in originals),
        "max_char_count": max((chunk["char_count"] for chunk in originals), default=0),
        "projection_count": len(projections),
        "projection_max_char_count": max(
            (chunk["char_count"] for chunk in projections), default=0
        ),
        "projection_traceability_errors": sum(
            not chunk.get("source_table_id")
            or not isinstance(chunk.get("source_row_start"), int)
            or not isinstance(chunk.get("source_row_end"), int)
            for chunk in projections
        ),
        "samples": [
            {
                key: chunk.get(key)
                for key in (
                    "chunk_id",
                    "doc_id",
                    "corp_name",
                    "table_id",
                    "table_title",
                    "section_path",
                    "row_start",
                    "row_end",
                    "char_count",
                )
            }
            for chunk in extreme[:20]
        ],
    }


def _extreme_table_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Extreme table chunk audit",
        "",
        f"- Original table chunks >5,000 chars: {report['gt_5000_count']}",
        f"- Original table chunks >10,000 chars: {report['gt_10000_count']}",
        f"- Maximum original size: {report['max_char_count']}",
        f"- Search-only row projections: {report['projection_count']}",
        f"- Maximum projection size: {report['projection_max_char_count']}",
        f"- Projection traceability errors: {report['projection_traceability_errors']}",
        "",
        "원본 table/row는 변경하거나 문자 중간에서 분할하지 않았습니다.",
        "",
        "| doc | table | rows | chars | title |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in report["samples"]:
        lines.append(
            f"| {item['doc_id']} | {item['table_id']} | "
            f"{item['row_start']}–{item['row_end']} | {item['char_count']} | "
            f"{item['table_title']} |"
        )
    return "\n".join(lines)


def _table_preview(table: Table) -> str:
    values = [
        cell.text.strip()
        for row in table.rows
        for cell in row
        if cell.text.strip()
    ]
    return " | ".join(dict.fromkeys(values))


def _exclusion_reason(table: Table, has_candidate: bool) -> tuple[str, str]:
    if has_candidate:
        return "exact_duplicate_content", "safe"
    preview = _table_preview(table)
    if _find_unit(preview):
        return "unit_wrapper", "safe"
    if re.fullmatch(r"【[^】]+】", preview):
        return "title_wrapper", "safe"
    if re.match(r"^[※*]\s*.+(?:참조|기재)\s*$", preview):
        return "reference_note", "review"
    if _BASIS_PERIOD.search(preview):
        return "basis_period_note", "high"
    return "context_rule_without_intrinsic_marker", "high"


def build_exclusion_audit(records: list[PilotRecord]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    recovered: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        row = record.row
        parsed = record.parsed
        table_map = {table.table_id: table for table in parsed.tables}
        indexed_ids = {
            str(chunk["table_id"])
            for chunk in record.structural_chunks
            if chunk["chunk_type"] == "table"
        }
        candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        strategy = get_chunking_strategy(str(row["doc_group"]))
        for section in parsed.sections:
            for chunk in _table_chunks_for_section(
                row,
                str(row["source_path"]),
                section,
                table_map,
                strategy,
            ):
                candidates[str(chunk["table_id"])].append(chunk)
        indexed_by_key: dict[tuple[Any, ...], str] = {}
        for chunk in record.structural_chunks:
            if chunk["chunk_type"] != "table":
                continue
            key = (
                tuple(chunk["section_path"]),
                str(chunk["content"]).strip(),
            )
            indexed_by_key[key] = str(chunk["table_id"])

        section_map = parsed.section_map()
        for chunk in record.structural_chunks:
            reason = chunk.get("former_exclusion_reason")
            if not reason:
                continue
            key = (str(row["doc_id"]), str(chunk["table_id"]))
            recovered[key] = {
                "doc_group": row["doc_group"],
                "doc_id": row["doc_id"],
                "corp_name": row["corp_name"],
                "report_nm": row["report_nm"],
                "table_id": chunk["table_id"],
                "section_id": chunk["section_id"],
                "section_path": " > ".join(chunk["section_path"]),
                "former_exclusion_reason": reason,
                "retrieval_priority": chunk.get("retrieval_priority"),
                "is_indexable": chunk.get("is_indexable", True),
                "preview": str(chunk["content"])[:2_000],
            }
        for table in parsed.tables:
            if table.table_id in indexed_ids:
                continue
            has_candidate = table.table_id in candidates
            reason, risk = _exclusion_reason(table, has_candidate)
            matching_table_id = ""
            if has_candidate:
                for candidate in candidates[table.table_id]:
                    key = (
                        tuple(candidate["section_path"]),
                        str(candidate["content"]).strip(),
                    )
                    if key in indexed_by_key:
                        matching_table_id = indexed_by_key[key]
                        break
            preview = _table_preview(table)
            rows.append(
                {
                    "doc_group": row["doc_group"],
                    "doc_id": row["doc_id"],
                    "corp_name": row["corp_name"],
                    "report_nm": row["report_nm"],
                    "table_id": table.table_id,
                    "section_id": table.section_id,
                    "section_path": " > ".join(section_map[table.section_id].path),
                    "reason": reason,
                    "risk": risk,
                    "matching_indexed_table_id": matching_table_id,
                    "row_count": len(table.rows),
                    "cell_count": sum(len(item) for item in table.rows),
                    "preview": preview[:2_000],
                }
            )

    counts_by_reason = Counter(item["reason"] for item in rows)
    counts_by_group_reason = Counter(
        (item["doc_group"], item["reason"]) for item in rows
    )
    counts_by_risk = Counter(item["risk"] for item in rows)
    recovered_rows = list(recovered.values())
    recovered_counts = Counter(
        item["former_exclusion_reason"] for item in recovered_rows
    )

    sample: list[dict[str, Any]] = []
    sample_ids: set[tuple[str, str]] = set()
    group_quotas = {"periodic": 30, "major": 5, "exchange": 0, "holding": 5}
    for group in GROUPS:
        candidates = [item for item in rows if item["doc_group"] == group]
        candidates.sort(
            key=lambda item: (
                {"high": 0, "review": 1, "safe": 2}[item["risk"]],
                item["reason"],
                item["doc_id"],
                item["table_id"],
            )
        )
        quota = min(group_quotas[group], len(candidates))
        reason_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in candidates:
            reason_groups[item["reason"]].append(item)
        while len([item for item in sample if item["doc_group"] == group]) < quota:
            progressed = False
            for reason in sorted(reason_groups):
                if reason_groups[reason]:
                    item = reason_groups[reason].pop(0)
                    key = (str(item["doc_id"]), str(item["table_id"]))
                    if key not in sample_ids:
                        sample.append(item)
                        sample_ids.add(key)
                        progressed = True
                    if len([entry for entry in sample if entry["doc_group"] == group]) >= quota:
                        break
            if not progressed:
                break
    if len(sample) < 40:
        for item in sorted(
            rows,
            key=lambda entry: (
                {"high": 0, "review": 1, "safe": 2}[entry["risk"]],
                entry["doc_group"],
                entry["reason"],
                entry["doc_id"],
                entry["table_id"],
            ),
        ):
            key = (str(item["doc_id"]), str(item["table_id"]))
            if key not in sample_ids:
                sample.append(item)
                sample_ids.add(key)
            if len(sample) == 40:
                break

    return {
        "excluded_table_count": len(rows),
        "counts_by_reason": dict(sorted(counts_by_reason.items())),
        "counts_by_risk": dict(sorted(counts_by_risk.items())),
        "high_risk_excluded_evidence_count": sum(
            item["risk"] == "high" for item in rows
        ),
        "recovered_evidence_table_count": len(recovered_rows),
        "recovered_counts_by_reason": dict(sorted(recovered_counts.items())),
        "counts_by_doc_group_and_reason": {
            f"{group}:{reason}": count
            for (group, reason), count in sorted(counts_by_group_reason.items())
        },
        "all_rows": rows,
        "recovered_rows": recovered_rows,
        "audit_sample": sample,
    }


def _exclusion_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Excluded table audit",
        "",
        f"Excluded source tables: {report['excluded_table_count']}",
        "",
        "## Counts by reason",
        "",
        "| reason | count | risk |",
        "| --- | ---: | --- |",
    ]
    risk_by_reason = {
        item["reason"]: item["risk"] for item in report["all_rows"]
    }
    for reason, count in report["counts_by_reason"].items():
        lines.append(f"| {reason} | {count} | {risk_by_reason[reason]} |")
    lines.extend(
        [
            "",
            "## Restored evidence tables",
            "",
            f"Recovered/indexable tables: {report['recovered_evidence_table_count']}",
            f"High-risk excluded evidence remaining: {report['high_risk_excluded_evidence_count']}",
            "",
            "| former reason | recovered |",
            "| --- | ---: |",
        ]
    )
    for reason, count in report["recovered_counts_by_reason"].items():
        lines.append(f"| {reason} | {count} |")
    lines.extend(
        [
            "",
            "## Counts by doc group and reason",
            "",
            "| doc group | reason | count |",
            "| --- | --- | ---: |",
        ]
    )
    for key, count in report["counts_by_doc_group_and_reason"].items():
        group, reason = key.split(":", 1)
        lines.append(f"| {group} | {reason} | {count} |")
    lines.extend(
        [
            "",
            "`high`는 표 자체의 실질적인 주석/증거가 검색 인덱스에서 사라질 수 있어 "
            "freeze 전에 반드시 해소해야 함을 뜻합니다.",
            "",
            f"## Audit sample ({len(report['audit_sample'])})",
            "",
        ]
    )
    for item in report["audit_sample"]:
        lines.extend(
            [
                f"### {item['doc_group']} · {item['doc_id']} · {item['table_id']}",
                "",
                f"- Reason/risk: `{item['reason']}` / `{item['risk']}`",
                f"- Company/report: {item['corp_name']} · {item['report_nm']}",
                f"- Section: {item['section_path']}",
                f"- Rows/cells: {item['row_count']} / {item['cell_count']}",
                "",
                "```text",
                str(item["preview"])[:2_000],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def select_correction_documents(
    manifest_path: Path, corpus_dir: Path, per_group: int = 3
) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {group: [] for group in GROUPS}
    for original in load_manifest(manifest_path):
        group = str(original.get("doc_group"))
        if group not in candidates or not original.get("is_correction"):
            continue
        try:
            source = resolve_primary_xml(corpus_dir, original)
        except FileNotFoundError:
            continue
        row = dict(original)
        row["source_path"] = str(source.relative_to(corpus_dir))
        row["source_size"] = source.stat().st_size
        candidates[group].append(row)
    selected: list[dict[str, Any]] = []
    for group in GROUPS:
        ordered = sorted(
            candidates[group],
            key=lambda row: (
                int(row["source_size"]),
                str(row["doc_id"]),
            ),
        )
        if len(ordered) < per_group:
            raise RuntimeError(f"not enough correction documents for {group}")
        used_companies: set[str] = set()
        used_ids: set[str] = set()
        for number in range(1, per_group + 1):
            target = round((len(ordered) - 1) * number / (per_group + 1))
            indexes = sorted(range(len(ordered)), key=lambda index: (abs(index - target), index))
            chosen = next(
                (
                    ordered[index]
                    for index in indexes
                    if ordered[index]["doc_id"] not in used_ids
                    and ordered[index]["corp_name"] not in used_companies
                ),
                None,
            )
            if chosen is None:
                chosen = next(row for row in ordered if row["doc_id"] not in used_ids)
            selected.append(chosen)
            used_ids.add(str(chosen["doc_id"]))
            used_companies.add(str(chosen["corp_name"]))
    return selected


def _validate_structural_document(
    row: dict[str, Any],
    parsed: ParsedDocument,
    chunks: list[dict[str, Any]],
    *,
    require_correction: bool = False,
) -> list[str]:
    errors: list[str] = []
    section_ids = {section.section_id for section in parsed.sections}
    table_ids = {table.table_id for table in parsed.tables}
    table_map = {table.table_id: table for table in parsed.tables}
    chunks_by_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        errors.append("duplicate chunk_id")
    for index, chunk in enumerate(chunks):
        chunk_id = str(chunk["chunk_id"])
        missing = [field for field in REQUIRED_CHUNK_FIELDS if field not in chunk]
        if missing:
            errors.append(f"{chunk_id}: missing metadata {missing}")
        for field in DOCUMENT_METADATA_FIELDS:
            expected = row.get(field)
            if chunk.get(field) != expected:
                errors.append(f"{chunk_id}: metadata mismatch {field}")
        if require_correction and chunk.get("is_correction") is not True:
            errors.append(f"{chunk_id}: correction flag lost")
        if chunk.get("section_id") not in section_ids:
            errors.append(f"{chunk_id}: orphan section")
        if chunk.get("chunk_type") in {"table", "table_projection"}:
            table_id = str(chunk.get("table_id") or "")
            if table_id not in table_ids:
                errors.append(f"{chunk_id}: orphan table")
            else:
                table = table_map[table_id]
                if chunk.get("chunk_type") == "table":
                    chunks_by_table[table_id].append(chunk)
                if chunk.get("section_id") != table.section_id:
                    errors.append(f"{chunk_id}: table/section mismatch")
                row_start = chunk.get("row_start")
                row_end = chunk.get("row_end")
                if (
                    not isinstance(row_start, int)
                    or not isinstance(row_end, int)
                    or row_start < 0
                    or row_end < row_start
                    or row_end >= len(table.rows)
                ):
                    errors.append(f"{chunk_id}: invalid table row range")
                headers = chunk.get("column_headers")
                if not isinstance(headers, list):
                    errors.append(f"{chunk_id}: invalid column headers")
                if chunk.get("is_indexable", True) and not chunk.get("table_rows"):
                    errors.append(f"{chunk_id}: indexable table without data rows")
                if chunk.get("chunk_type") == "table_projection":
                    if chunk.get("source_table_id") != table_id:
                        errors.append(f"{chunk_id}: projection source table mismatch")
                    refs = chunk.get("source_refs")
                    if not isinstance(refs, list) or not refs:
                        errors.append(f"{chunk_id}: missing projection traceability")
                    else:
                        for ref in refs:
                            ref_table_id = str(ref.get("table_id") or "")
                            ref_table = table_map.get(ref_table_id)
                            if ref_table is None:
                                errors.append(f"{chunk_id}: orphan projection source")
                                continue
                            ref_start = ref.get("row_start")
                            ref_end = ref.get("row_end")
                            if (
                                not isinstance(ref_start, int)
                                or not isinstance(ref_end, int)
                                or ref_start < 0
                                or ref_end < ref_start
                                or ref_end >= len(ref_table.rows)
                            ):
                                errors.append(f"{chunk_id}: invalid projection row range")
        retrieval_text = str(chunk.get("retrieval_text", ""))
        required_context = (
            str(row.get("corp_name") or ""),
            str(row.get("report_nm") or ""),
            " > ".join(chunk.get("section_path") or []),
            str(chunk.get("content") or ""),
        )
        if any(value and value not in retrieval_text for value in required_context):
            errors.append(f"{chunk_id}: incomplete retrieval_text")
        expected_previous = chunks[index - 1]["chunk_id"] if index else None
        expected_next = chunks[index + 1]["chunk_id"] if index + 1 < len(chunks) else None
        if chunk.get("prev_chunk_id") != expected_previous:
            errors.append(f"{chunk_id}: invalid prev link")
        if chunk.get("next_chunk_id") != expected_next:
            errors.append(f"{chunk_id}: invalid next link")
    for section in parsed.sections:
        if section.parent_id and section.parent_id not in section_ids:
            errors.append(f"{section.section_id}: orphan parent")
    for table in parsed.tables:
        if table.section_id not in section_ids:
            errors.append(f"{table.table_id}: orphan source table")
    for table_id, table_chunks in chunks_by_table.items():
        if len(table_chunks) <= 1:
            continue
        expected_headers = table_chunks[0].get("column_headers")
        expected_header_rows = table_chunks[0].get("header_rows")
        previous_end: int | None = None
        for chunk in sorted(table_chunks, key=lambda item: int(item["row_start"])):
            chunk_id = str(chunk["chunk_id"])
            if not expected_headers:
                errors.append(f"{chunk_id}: split table without repeated headers")
            elif chunk.get("column_headers") != expected_headers:
                errors.append(f"{chunk_id}: inconsistent repeated headers")
            if chunk.get("header_rows") != expected_header_rows:
                errors.append(f"{chunk_id}: inconsistent source header rows")
            row_start = int(chunk["row_start"])
            if previous_end is not None and row_start <= previous_end:
                errors.append(f"{chunk_id}: overlapping table row ranges")
            previous_end = int(chunk["row_end"])
    return errors


def _correction_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Correction-document pilot validation",
        "",
        f"Documents: {report['document_count']}",
        f"Valid: {report['valid_document_count']}",
        f"Validation errors: {report['error_count']}",
        "",
        "| group | company | doc_id | sections | tables | chunks | metadata | integrity | retrieval text | deterministic IDs |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for item in report["documents"]:
        lines.append(
            f"| {item['doc_group']} | {item['corp_name']} | {item['doc_id']} | "
            f"{item['section_count']} | {item['table_count']} | {item['chunk_count']} | "
            f"{item['metadata_valid']} | {item['integrity_valid']} | "
            f"{item['retrieval_text_valid']} | {item['deterministic_ids']} |"
        )
    return "\n".join(lines)


def run_correction_pilot(
    manifest_path: Path,
    corpus_dir: Path,
    output_dir: Path,
    per_group: int = 3,
) -> dict[str, Any]:
    selected = select_correction_documents(manifest_path, corpus_dir, per_group)
    document_dir = output_dir / "correction_documents"
    document_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        source_path = resolve_unicode_path(corpus_dir, str(row["source_path"]))
        print(
            f"correction [{index:02d}/{len(selected)}] {row['doc_group']} "
            f"{row['corp_name']} {row['report_nm']}",
            flush=True,
        )
        parsed = parse_dart_document(source_path, fallback_title=str(row["report_nm"]))
        first = build_chunks(
            str(row["doc_id"]),
            parsed,
            document_metadata=row,
            source_file=str(row["source_path"]),
        )
        second = build_chunks(
            str(row["doc_id"]),
            parsed,
            document_metadata=row,
            source_file=str(row["source_path"]),
        )
        deterministic = [chunk["chunk_id"] for chunk in first] == [
            chunk["chunk_id"] for chunk in second
        ]
        errors = _validate_structural_document(
            row,
            parsed,
            first,
            require_correction=True,
        )
        if not deterministic:
            errors.append("non-deterministic chunk IDs")
        payload = {
            "schema_version": "2.0",
            "chunking_version": CHUNKING_VERSION,
            "document": {field: row.get(field) for field in DOCUMENT_METADATA_FIELDS},
            "sections": [section.to_dict() for section in parsed.sections],
            "tables": [
                table.to_dict(parsed.section_map()[table.section_id].path)
                for table in parsed.tables
            ],
            "chunks": first,
            "parser_warnings": parsed.parser_warnings,
        }
        target = document_dir / f"{row['doc_id']}.json.gz"
        with gzip.open(target, "wt", encoding="utf-8", compresslevel=3) as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        results.append(
            {
                "doc_group": row["doc_group"],
                "doc_id": row["doc_id"],
                "corp_name": row["corp_name"],
                "report_nm": row["report_nm"],
                "rcept_no": row["rcept_no"],
                "rcept_dt": row["rcept_dt"],
                "source_path": row["source_path"],
                "source_size": row["source_size"],
                "section_count": len(parsed.sections),
                "table_count": len(parsed.tables),
                "chunk_count": len(first),
                "parser_warning_count": len(parsed.parser_warnings),
                "metadata_valid": not any("metadata" in error for error in errors),
                "integrity_valid": not any(
                    marker in error
                    for error in errors
                    for marker in (
                        "orphan",
                        "link",
                        "mismatch",
                        "range",
                        "traceability",
                        "headers",
                        "rows",
                    )
                ),
                "retrieval_text_valid": not any("retrieval_text" in error for error in errors),
                "deterministic_ids": deterministic,
                "valid": not errors,
                "errors": errors,
                "output_path": str(target.relative_to(output_dir)),
            }
        )
    return {
        "document_count": len(results),
        "group_counts": dict(Counter(item["doc_group"] for item in results)),
        "valid_document_count": sum(item["valid"] for item in results),
        "error_count": sum(len(item["errors"]) for item in results),
        "documents": results,
    }


_TOKEN = re.compile(r"[0-9A-Za-z가-힣]+")


def _tokens(value: str) -> list[str]:
    return [token.casefold() for token in _TOKEN.findall(value)]


class BM25Index:
    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies = [Counter(_tokens(str(doc["evaluation_text"]))) for doc in documents]
        self.lengths = [sum(counter.values()) for counter in self.term_frequencies]
        self.average_length = statistics.mean(self.lengths) if self.lengths else 0
        self.document_frequency: Counter[str] = Counter()
        for counter in self.term_frequencies:
            self.document_frequency.update(counter.keys())

    def search(self, query: str) -> list[tuple[float, dict[str, Any]]]:
        query_terms = Counter(_tokens(query))
        total = len(self.documents)
        scored: list[tuple[float, str, dict[str, Any]]] = []
        for index, (document, frequencies) in enumerate(
            zip(self.documents, self.term_frequencies)
        ):
            score = 0.0
            length = self.lengths[index]
            for term, query_frequency in query_terms.items():
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[term]
                inverse_frequency = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1 - self.b
                    + self.b * length / max(self.average_length, 1)
                )
                score += (
                    inverse_frequency
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                    * query_frequency
                )
            scored.append((score, str(document["chunk_id"]), document))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(score, document) for score, _, document in scored]


def _evaluation_document(
    row: dict[str, Any], chunk: dict[str, Any]
) -> dict[str, Any]:
    section_path = chunk.get("section_path") or []
    evaluation_text = "\n".join(
        [
            f"[기업명] {row.get('corp_name') or ''}",
            f"[공시명] {row.get('report_nm') or ''}",
            f"[Section Path] {' > '.join(section_path)}",
            "",
            str(chunk.get("content") or ""),
        ]
    )
    return {
        **chunk,
        "doc_id": row["doc_id"],
        "doc_group": row["doc_group"],
        "evaluation_text": evaluation_text,
    }


def _is_relevant(document: dict[str, Any], question: dict[str, Any]) -> bool:
    if document.get("doc_id") != question["doc_id"]:
        return False
    if question["target_type"] == "table":
        source_table_ids = document.get("source_table_ids") or []
        if (
            document.get("table_id") != question["target_id"]
            and question["target_id"] not in source_table_ids
        ):
            return False
    elif document.get("section_id") != question["target_id"]:
        return False
    content = str(document.get("content", ""))
    return all(term in content for term in question["evidence_terms"])


def run_bm25_evaluation(
    records: list[PilotRecord],
    questions: tuple[dict[str, Any], ...] = GOLD_QUESTIONS,
    *,
    evaluation_name: str = "fixed_40",
) -> dict[str, Any]:
    systems: dict[str, list[dict[str, Any]]] = {"legacy": [], "structural": []}
    for record in records:
        systems["legacy"].extend(
            _evaluation_document(record.row, chunk) for chunk in record.legacy_chunks
        )
        systems["structural"].extend(
            _evaluation_document(record.row, chunk)
            for chunk in record.structural_chunks
            if chunk.get("is_indexable", True)
        )
    indexes = {name: BM25Index(documents) for name, documents in systems.items()}
    result_rows: list[dict[str, Any]] = []
    for question in questions:
        row: dict[str, Any] = {
            **question,
            "evidence_terms": " | ".join(question["evidence_terms"]),
        }
        for system, index in indexes.items():
            ranked = index.search(str(question["query"]))
            relevant_ranks = [
                position
                for position, (_, document) in enumerate(ranked, start=1)
                if _is_relevant(document, question)
            ]
            first_rank = relevant_ranks[0] if relevant_ranks else None
            row[f"{system}_relevant_chunk_count"] = len(relevant_ranks)
            row[f"{system}_first_relevant_rank"] = first_rank
            row[f"{system}_hit_at_1"] = bool(first_rank and first_rank <= 1)
            row[f"{system}_hit_at_5"] = bool(first_rank and first_rank <= 5)
            row[f"{system}_hit_at_10"] = bool(first_rank and first_rank <= 10)
            row[f"{system}_top1_chunk_id"] = ranked[0][1]["chunk_id"] if ranked else ""
            row[f"{system}_top1_doc_id"] = ranked[0][1]["doc_id"] if ranked else ""
            row[f"{system}_top1_score"] = round(ranked[0][0], 6) if ranked else 0
        result_rows.append(row)

    def metrics(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
        total = len(rows)
        return {
            "question_count": total,
            "recall_at_1": round(
                sum(item[f"{system}_hit_at_1"] for item in rows) / total, 6
            ) if total else 0,
            "recall_at_5": round(
                sum(item[f"{system}_hit_at_5"] for item in rows) / total, 6
            ) if total else 0,
            "recall_at_10": round(
                sum(item[f"{system}_hit_at_10"] for item in rows) / total, 6
            ) if total else 0,
            "questions_without_relevant_chunk": sum(
                not item[f"{system}_relevant_chunk_count"] for item in rows
            ),
        }

    return {
        "method": {
            "algorithm": "BM25",
            "k1": 1.5,
            "b": 0.75,
            "tokenizer": "lowercased Korean/English/numeric word tokens",
            "evaluation_text": "identical corp/report/section context prefix + chunk content",
        },
        "evaluation_name": evaluation_name,
        "gold_question_count": len(questions),
        "overall": {
            system: metrics(result_rows, system) for system in ("legacy", "structural")
        },
        "by_doc_group": {
            group: {
                system: metrics(
                    [item for item in result_rows if item["doc_group"] == group],
                    system,
                )
                for system in ("legacy", "structural")
            }
            for group in GROUPS
        },
        "questions": result_rows,
    }


def _bm25_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BM25 legacy vs structural evaluation",
        "",
        f"Evaluation: {report['evaluation_name']}",
        f"Gold questions: {report['gold_question_count']}",
        "",
        "양쪽 모두 동일한 tokenizer/BM25와 기업명·공시명·section path prefix를 사용했습니다.",
        "",
        "| scope | system | Recall@1 | Recall@5 | Recall@10 | no relevant chunk |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    scopes = [
        ("overall", report["overall"]),
        *(
            (scope, systems)
            for scope, systems in report["by_doc_group"].items()
            if systems["legacy"]["question_count"]
        ),
    ]
    for scope, systems in scopes:
        for system in ("legacy", "structural"):
            metrics = systems[system]
            lines.append(
                f"| {scope} | {system} | {metrics['recall_at_1']:.3f} | "
                f"{metrics['recall_at_5']:.3f} | {metrics['recall_at_10']:.3f} | "
                f"{metrics['questions_without_relevant_chunk']} |"
            )
    lines.extend(["", "## Per-question result", ""])
    lines.extend(
        [
            "| id | group | question | legacy rank | structural rank |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for item in report["questions"]:
        lines.append(
            f"| {item['question_id']} | {item['doc_group']} | {item['query']} | "
            f"{item['legacy_first_relevant_rank'] or '-'} | "
            f"{item['structural_first_relevant_rank'] or '-'} |"
        )
    return "\n".join(lines)


def _load_pilot_records(
    corpus_dir: Path, pilot_dir: Path
) -> tuple[list[PilotRecord], list[str]]:
    selected = json.loads((pilot_dir / "selection.json").read_text(encoding="utf-8"))
    records: list[PilotRecord] = []
    errors: list[str] = []
    for row in selected:
        source_path = resolve_unicode_path(corpus_dir, str(row["source_path"]))
        parsed = parse_dart_document(source_path, fallback_title=str(row["report_nm"]))
        saved = json.loads(
            (pilot_dir / "documents" / f"{row['doc_id']}.json").read_text(
                encoding="utf-8"
            )
        )
        structural_chunks = saved["chunks"]
        rebuilt = build_chunks(
            str(row["doc_id"]),
            parsed,
            document_metadata=row,
            source_file=str(row["source_path"]),
        )
        if [chunk["chunk_id"] for chunk in rebuilt] != [
            chunk["chunk_id"] for chunk in structural_chunks
        ]:
            errors.append(f"{row['doc_id']}: saved pilot is stale or non-deterministic")
        errors.extend(_validate_structural_document(row, parsed, structural_chunks))
        records.append(
            PilotRecord(
                row=row,
                parsed=parsed,
                structural_chunks=structural_chunks,
                legacy_chunks=build_legacy_chunks(str(row["doc_id"]), parsed),
            )
        )
    return records, errors


def run_final_validation(
    corpus_dir: Path,
    manifest_path: Path,
    pilot_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, pilot_errors = _load_pilot_records(corpus_dir, pilot_dir)

    length_report = build_length_report(records)
    _write_json(output_dir / "chunk_length_stats.json", length_report)
    (output_dir / "chunk_length_stats.md").write_text(
        _length_markdown(length_report), encoding="utf-8"
    )
    extreme_tables = build_extreme_table_report(records)
    _write_json(output_dir / "extreme_table_audit.json", extreme_tables)
    (output_dir / "extreme_table_audit.md").write_text(
        _extreme_table_markdown(extreme_tables), encoding="utf-8"
    )

    exclusions = build_exclusion_audit(records)
    _write_json(
        output_dir / "excluded_tables_summary.json",
        {
            key: value
            for key, value in exclusions.items()
            if key not in {"all_rows", "recovered_rows"}
        },
    )
    exclusion_fields = [
        "doc_group",
        "doc_id",
        "corp_name",
        "report_nm",
        "table_id",
        "section_id",
        "section_path",
        "reason",
        "risk",
        "matching_indexed_table_id",
        "row_count",
        "cell_count",
        "preview",
    ]
    _write_csv(
        output_dir / "excluded_tables_all.csv", exclusions["all_rows"], exclusion_fields
    )
    _write_csv(
        output_dir / "excluded_tables_audit_40.csv",
        exclusions["audit_sample"],
        exclusion_fields,
    )
    _write_csv(
        output_dir / "restored_evidence_tables.csv",
        exclusions["recovered_rows"],
        [
            "doc_group",
            "doc_id",
            "corp_name",
            "report_nm",
            "table_id",
            "section_id",
            "section_path",
            "former_exclusion_reason",
            "retrieval_priority",
            "is_indexable",
            "preview",
        ],
    )
    (output_dir / "excluded_tables_audit_40.md").write_text(
        _exclusion_markdown(exclusions), encoding="utf-8"
    )

    correction_report = run_correction_pilot(
        manifest_path=manifest_path,
        corpus_dir=corpus_dir,
        output_dir=output_dir,
        per_group=3,
    )
    _write_json(output_dir / "correction_pilot.json", correction_report)
    (output_dir / "correction_pilot.md").write_text(
        _correction_markdown(correction_report), encoding="utf-8"
    )
    correction_fields = [
        "doc_group",
        "doc_id",
        "corp_name",
        "report_nm",
        "rcept_no",
        "rcept_dt",
        "source_path",
        "source_size",
        "section_count",
        "table_count",
        "chunk_count",
        "parser_warning_count",
        "metadata_valid",
        "integrity_valid",
        "retrieval_text_valid",
        "deterministic_ids",
        "valid",
        "output_path",
    ]
    _write_csv(
        output_dir / "correction_pilot.csv",
        correction_report["documents"],
        correction_fields,
    )

    bm25_report = run_bm25_evaluation(records)
    _write_json(output_dir / "bm25_evaluation.json", bm25_report)
    bm25_fields = [
        "question_id",
        "doc_group",
        "query",
        "doc_id",
        "target_type",
        "target_id",
        "evidence_terms",
        "legacy_relevant_chunk_count",
        "legacy_first_relevant_rank",
        "legacy_hit_at_1",
        "legacy_hit_at_5",
        "legacy_hit_at_10",
        "legacy_top1_doc_id",
        "legacy_top1_chunk_id",
        "legacy_top1_score",
        "structural_relevant_chunk_count",
        "structural_first_relevant_rank",
        "structural_hit_at_1",
        "structural_hit_at_5",
        "structural_hit_at_10",
        "structural_top1_doc_id",
        "structural_top1_chunk_id",
        "structural_top1_score",
    ]
    _write_csv(output_dir / "bm25_question_results.csv", bm25_report["questions"], bm25_fields)
    _write_csv(
        output_dir / "gold_questions.csv",
        [
            {
                **question,
                "evidence_terms": " | ".join(question["evidence_terms"]),
            }
            for question in GOLD_QUESTIONS
        ],
        [
            "question_id",
            "doc_group",
            "query",
            "doc_id",
            "target_type",
            "target_id",
            "evidence_terms",
        ],
    )
    (output_dir / "bm25_evaluation.md").write_text(
        _bm25_markdown(bm25_report), encoding="utf-8"
    )

    holding_bm25_report = run_bm25_evaluation(
        records,
        HOLDING_ADDITIONAL_QUESTIONS,
        evaluation_name="additional_holding_20",
    )
    _write_json(output_dir / "bm25_holding_additional.json", holding_bm25_report)
    _write_csv(
        output_dir / "bm25_holding_additional_results.csv",
        holding_bm25_report["questions"],
        bm25_fields,
    )
    _write_csv(
        output_dir / "holding_additional_gold_questions.csv",
        [
            {
                **question,
                "evidence_terms": " | ".join(question["evidence_terms"]),
            }
            for question in HOLDING_ADDITIONAL_QUESTIONS
        ],
        [
            "question_id",
            "doc_group",
            "query",
            "doc_id",
            "target_type",
            "target_id",
            "evidence_terms",
        ],
    )
    (output_dir / "bm25_holding_additional.md").write_text(
        _bm25_markdown(holding_bm25_report), encoding="utf-8"
    )

    fatal_issues: list[str] = []
    if pilot_errors:
        fatal_issues.append(f"pilot integrity errors: {len(pilot_errors)}")
    allowed_exclusion_reasons = {
        "exact_duplicate_content",
        "unit_wrapper",
        "title_wrapper",
    }
    unexpected_exclusion_reasons = sorted(
        set(exclusions["counts_by_reason"]) - allowed_exclusion_reasons
    )
    if unexpected_exclusion_reasons:
        fatal_issues.append(
            "non-conservative exclusion reasons remain: "
            + ", ".join(unexpected_exclusion_reasons)
        )
    high_risk_exclusions = exclusions["high_risk_excluded_evidence_count"]
    if high_risk_exclusions:
        fatal_issues.append(
            f"high-risk excluded tables requiring review: {high_risk_exclusions}"
        )
    if correction_report["error_count"]:
        fatal_issues.append(
            f"correction pilot validation errors: {correction_report['error_count']}"
        )
    legacy = bm25_report["overall"]["legacy"]
    structural = bm25_report["overall"]["structural"]
    if structural["questions_without_relevant_chunk"]:
        fatal_issues.append(
            "structural gold questions without a coherent relevant chunk: "
            f"{structural['questions_without_relevant_chunk']}"
        )
    if structural["recall_at_5"] < legacy["recall_at_5"]:
        fatal_issues.append("structural BM25 Recall@5 is below legacy")
    if structural["recall_at_10"] < legacy["recall_at_10"]:
        fatal_issues.append("structural BM25 Recall@10 is below legacy")
    holding_legacy = bm25_report["by_doc_group"]["holding"]["legacy"]
    holding_structural = bm25_report["by_doc_group"]["holding"]["structural"]
    if holding_structural["recall_at_5"] + 0.125 < holding_legacy["recall_at_5"]:
        fatal_issues.append("holding structural Recall@5 is not close to legacy")
    if holding_structural["recall_at_10"] <= 0.375:
        fatal_issues.append("holding structural Recall@10 did not improve from 0.375")
    added_legacy = holding_bm25_report["overall"]["legacy"]
    added_structural = holding_bm25_report["overall"]["structural"]
    if added_structural["questions_without_relevant_chunk"]:
        fatal_issues.append(
            "additional holding questions without a structural relevant chunk: "
            f"{added_structural['questions_without_relevant_chunk']}"
        )
    if extreme_tables["projection_traceability_errors"]:
        fatal_issues.append(
            "extreme table projection traceability errors: "
            f"{extreme_tables['projection_traceability_errors']}"
        )

    decision = {
        "chunking_version": CHUNKING_VERSION,
        "full_corpus_reprocessed": False,
        "freeze_candidate": not fatal_issues,
        "decision": "FREEZE_READY" if not fatal_issues else "FREEZE_BLOCKED",
        "fatal_issues": fatal_issues,
        "pilot_integrity_errors": pilot_errors,
        "high_risk_excluded_table_count": high_risk_exclusions,
        "correction_pilot_valid": correction_report["error_count"] == 0,
        "bm25_legacy": legacy,
        "bm25_structural": structural,
        "holding_bm25_legacy": holding_legacy,
        "holding_bm25_structural": holding_structural,
        "additional_holding_bm25_legacy": added_legacy,
        "additional_holding_bm25_structural": added_structural,
    }
    _write_json(output_dir / "freeze_decision.json", decision)
    decision_lines = [
        "# Chunking freeze decision",
        "",
        f"Decision: **{decision['decision']}**",
        "",
        "The full 4,204-document corpus was not reprocessed.",
        "",
    ]
    if fatal_issues:
        decision_lines.extend(["## Blocking findings", ""])
        decision_lines.extend(f"- {issue}" for issue in fatal_issues)
    else:
        decision_lines.extend(
            [
                "No fatal validation issue was found. The current implementation is FREEZE_READY.",
            ]
        )
    (output_dir / "freeze_decision.md").write_text(
        "\n".join(decision_lines), encoding="utf-8"
    )
    final_report = {
        "decision": decision,
        "length_stats": length_report,
        "extreme_tables": extreme_tables,
        "excluded_tables": {
            key: value
            for key, value in exclusions.items()
            if key not in {"all_rows", "recovered_rows"}
        },
        "correction_pilot": correction_report,
        "bm25": bm25_report,
        "bm25_holding_additional": holding_bm25_report,
    }
    _write_json(output_dir / "final_validation.json", final_report)
    return final_report
