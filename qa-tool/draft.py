#!/usr/bin/env python3
"""공시 Q&A 초안 YAML 생성 — 에이전트·Gold60과 무관한 팀 작성 도구."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UNIVERSE = ROOT / "data" / "corpus" / "universe.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _normalize(text: str) -> str:
    value = re.sub(r"\s+", "", text or "")
    return re.sub(r"[ㆍ·・‧]", "", value).casefold()


@dataclass(frozen=True)
class CompanyRow:
    corp_name: str
    listed_name: str
    sector: str
    n_periodic: int
    n_major: int
    n_exchange: int
    n_holding: int


def load_universe(path: Path) -> list[CompanyRow]:
    rows: list[CompanyRow] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                CompanyRow(
                    corp_name=str(row["corp_name"]).strip(),
                    listed_name=str(row["listed_name"]).strip(),
                    sector=str(row.get("sector") or "").strip(),
                    n_periodic=int(row.get("n_periodic") or 0),
                    n_major=int(row.get("n_major") or 0),
                    n_exchange=int(row.get("n_exchange") or 0),
                    n_holding=int(row.get("n_holding") or 0),
                )
            )
    return rows


def detect_company(query: str, universe: list[CompanyRow]) -> tuple[str, str, str | None]:
    compact = _normalize(query)
    best: tuple[int, CompanyRow] | None = None
    for row in universe:
        for label in (row.listed_name, row.corp_name):
            key = _normalize(label)
            if key and key in compact:
                if best is None or len(key) > best[0]:
                    best = (len(key), row)
    if best is None:
        return "(회사명)", "(corp_name)", None
    row = best[1]
    return row.listed_name, row.corp_name, row.sector


def detect_doc_group(query: str) -> str:
    compact = _normalize(query)
    if any(
        term in compact
        for term in (
            "소유상황보고서",
            "대량보유상황",
            "국민연금",
            "보유비율",
            "보유주식",
            "변동후",
            "변동전",
        )
    ):
        return "holding"
    major_terms = (
        "자기주식취득신탁계약해지",
        "자기주식취득신탁계약체결",
        "자기주식취득신탁",
        "자기주식취득",
        "자기주식처분",
        "자기주식소각",
        "유상증자",
        "전환사채",
        "회사분할",
        "흡수합병",
        "합병",
    )
    if any(term in compact for term in major_terms):
        return "major"
    if any(
        term in compact
        for term in ("시설투자", "신규시설", "단일판매", "공급계약", "수주계약", "투자판단")
    ):
        return "exchange"
    if "계약해지" in compact or "계약해지금액" in compact:
        if any(term in compact for term in ("자기주식", "신탁계약", "신탁")):
            return "major"
        return "exchange"
    if any(term in compact for term in ("사업보고서", "분기보고서", "반기보고서", "매출액", "영업이익")):
        return "periodic"
    return "periodic"


def detect_basis(query: str) -> str:
    if re.search(r"별도", query):
        return "별도"
    if re.search(r"연결", query):
        return "연결"
    if re.search(r"국민연금|보유|계약|시설", query):
        return "해당없음"
    return "(질문에 연결/별도 명시 권장)"


def detect_report_hint(query: str) -> str:
    if re.search(r"사업보고서", query):
        return "사업보고서 (연말 결산)"
    if re.search(r"반기보고서|상반기|하반기", query):
        return "반기보고서"
    quarter = re.search(r"(?<!\d)(20\d{2})\s*년?\s*([1-4])\s*분기", query)
    if quarter:
        year, q = quarter.group(1), int(quarter.group(2))
        return f"분기보고서 ({year}.{q * 3:02d}) · {year}년 {q}분기"
    year = re.search(r"(?<!\d)(20\d{2})\s*년", query)
    if year:
        return f"{year.group(1)}년 (보고서 종류 명시 필요)"
    return "(기간·보고서 종류 명시)"


def detect_task_type(query: str, doc_group: str) -> str:
    compact = _normalize(query)
    if doc_group == "holding":
        return "holding_change"
    if doc_group == "major":
        if "자기주식취득신탁계약해지" in compact or (
            "신탁" in compact and "해지" in compact
        ):
            return "corporate_event · treasury_share_trust_termination"
        if "자기주식처분" in compact:
            return "corporate_event · treasury_share_disposal"
        if "자기주식취득" in compact:
            return "corporate_event · treasury_share_acquisition"
        return "major_event"
    if doc_group == "exchange":
        if re.search(r"시설|투자\s*종료|투자목적|자기자본", query):
            return "facility_investment"
        return "supply_contract / exchange_event"
    if re.search(r"상장일", query):
        return "listing_history"
    if re.search(r"구성|내역|수익\s*구분|재화|용역", query):
        return "periodic_fact · metric_view=breakdown"
    if re.search(r"매출|영업이익|당기순|자산|부채|자본|영업외|금융비용|EPS|주당", query):
        return "periodic_fact · financial_metric"
    if re.search(r"부문|segment|사업부", query):
        return "periodic_fact · segment"
    return "periodic_fact / general_evidence"


def build_must_include(query: str, doc_group: str, basis: str) -> list[str]:
    lines: list[str] = []
    if basis in {"연결", "별도"}:
        lines.append(f"재무제표 기준: {basis}")
    if doc_group == "periodic" and re.search(r"구성|내역", query):
        lines.extend(
            [
                "섹션: 고객과의 계약에서 생기는 수익의 구분 (또는 동의 표)",
                "행: 재화·용역·로열티·금융·건설·기타 (공시에 있는 줄만)",
            ]
        )
    elif doc_group == "periodic" and re.search(r"매출|영업|순이익|자산|부채|자본", query):
        lines.append("손익/재무상태표 해당 과목 한 줄 (질문한 지표만)")
    elif doc_group == "major" and re.search(r"신탁.*해지|자기주식.*소각", query):
        lines.extend(
            [
                "report_nm: 주요사항보고서(자기주식취득신탁계약해지결정) 등",
                "신탁계약 해지·소각(예정) 관련 항목 (공시에 있는 경우만)",
            ]
        )
    elif doc_group == "exchange" and re.search(r"시설|투자", query):
        lines.append("투자금액, 자기자본 대비(%), 투자목적, 시작·종료일, 이사회 결의일")
    elif doc_group == "exchange":
        lines.append("계약상대, 계약금액, 계약기간, 최근매출 대비(%)")
    elif doc_group == "holding":
        lines.append("보고자, 변동일, 변동 전·후 주식수·보유비율")
    lines.append("숫자 단위 (백만원/주/% 등 공시 그대로)")
    return lines


def build_must_not(doc_group: str) -> str:
    common = "코퍼스 밖 공시·뉴스, 전년 대비·성장성 해석, 표에 없는 재분류"
    if doc_group == "periodic":
        return f"{common}, 질문과 다른 과목(매출원가·EPS 등) 혼입"
    if doc_group == "exchange":
        return f"{common}, 현금조달 가능 여부 등 공시에 없는 판단"
    if doc_group == "major":
        return f"{common}, exchange(공급계약 해지)로 오인"
    return common


def build_negative(query: str, doc_group: str) -> str:
    compact = _normalize(query)
    if doc_group == "periodic" and re.search(r"구성|내역", query):
        return "손익계산서 매출액 총액 한 줄만 답함 (구성 질문의 대표 오답)"
    if doc_group == "periodic" and re.search(r"연결", query) and not re.search(r"별도", query):
        return "별도 재무제표 숫자를 연결 질문에 답함"
    if doc_group == "major" and "신탁" in compact and "해지" in compact:
        return "doc_group=exchange, subtype=단일판매공급계약해지 (신탁계약해지의 계약해지 부분문자열)"
    if doc_group == "exchange":
        return "정정 전 숫자·종료일 (정정본 존재 시)"
    if doc_group == "holding":
        return "피투자회사와 보고자 혼동, 정기공시 재무표와 혼합"
    return "(공시 확인 후 기록)"


def build_evidence(doc_group: str) -> list[str]:
    prefix = {
        "periodic": "periodic_",
        "exchange": "exchange_",
        "holding": "holding_",
        "major": "major_",
    }[doc_group]
    return [
        f"doc_id: {prefix}(rcept_no)",
        "section_path: (표 제목)",
        "manifest: data/corpus/manifest.jsonl에서 corp_name·rcept_dt 검색",
    ]


def build_question_id(
    question: str,
    corp_name: str,
    listed_name: str,
    prefix: str,
    index: int,
) -> str:
    if prefix:
        suffix = f"{index:02d}" if index else ""
        return f"{prefix}{suffix or '01'}"
    slug_source = corp_name if corp_name != "(corp_name)" else listed_name
    slug = re.sub(r"[^0-9A-Za-z가-힣]", "", slug_source)[:4].upper() or "QA"
    seq = max(1, (len(question) % 9) + 1)
    return f"{slug}{seq:02d}"


@dataclass
class QaDraft:
    question_id: str
    question: str
    listed_name: str
    corp_name: str
    doc_group: str
    report_hint: str
    basis: str
    task_type: str
    must_include: list[str]
    must_not: str
    evidence: list[str]
    negative_example: str
    notes: str
    in_universe: bool
    corpus_note: str | None

    def to_yaml(self) -> str:
        lines = [
            f"question_id: {self.question_id}",
            f"question: {self.question}",
            f"listed_name: {self.listed_name}",
            f"corp_name: {self.corp_name}",
            f"doc_group: {self.doc_group}",
            f"report_nm · period: {self.report_hint}",
            f"basis: {self.basis}",
            f"task_type (기대): {self.task_type}",
            "must_include_in_answer:",
        ]
        for item in self.must_include:
            lines.append(f"  - {item}")
        lines.extend(
            [
                f"must_NOT_invent: {self.must_not}",
                "evidence:",
            ]
        )
        for item in self.evidence:
            lines.append(f"  {item}")
        lines.extend(
            [
                f"negative_example: {self.negative_example}",
                f"notes: {self.notes}",
            ]
        )
        if self.corpus_note:
            lines.append(f"corpus_note: {self.corpus_note}")
        return "\n".join(lines) + "\n"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "listed_name": self.listed_name,
            "corp_name": self.corp_name,
            "doc_group": self.doc_group,
            "report_nm_period": self.report_hint,
            "basis": self.basis,
            "task_type": self.task_type,
            "must_include_in_answer": self.must_include,
            "must_NOT_invent": self.must_not,
            "evidence": self.evidence,
            "negative_example": self.negative_example,
            "notes": self.notes,
            "in_universe": self.in_universe,
            "corpus_note": self.corpus_note,
        }


def build_draft(
    question: str,
    *,
    universe: list[CompanyRow],
    prefix: str = "",
    index: int = 0,
) -> QaDraft | None:
    text = question.strip()
    if not text:
        return None
    listed, corp, _sector = detect_company(text, universe)
    in_universe = corp != "(corp_name)"
    corpus_note = None
    if not in_universe:
        corpus_note = "70개 universe에 없는 회사 - manifest 확인 또는 질문 회사 교체"
    doc_group = detect_doc_group(text)
    basis = detect_basis(text)
    task_type = detect_task_type(text, doc_group)
    notes = (
        "정정본 우선 · 정정 전후 diff 확인"
        if re.search(r"정정", text)
        else "(공시 열람 후 doc_id·기대답 숫자 보완)"
    )
    return QaDraft(
        question_id=build_question_id(text, corp, listed, prefix, index),
        question=text,
        listed_name=listed,
        corp_name=corp,
        doc_group=doc_group,
        report_hint=detect_report_hint(text),
        basis=basis,
        task_type=task_type,
        must_include=build_must_include(text, doc_group, basis),
        must_not=build_must_not(doc_group),
        evidence=build_evidence(doc_group),
        negative_example=build_negative(text, doc_group),
        notes=notes,
        in_universe=in_universe,
        corpus_note=corpus_note,
    )


def export_universe_json(universe_path: Path, output_path: Path) -> None:
    rows = load_universe(universe_path)
    payload = [
        {
            "corp_name": row.corp_name,
            "listed_name": row.listed_name,
            "sector": row.sector,
            "n_periodic": row.n_periodic,
            "n_major": row.n_major,
            "n_exchange": row.n_exchange,
            "n_holding": row.n_holding,
        }
        for row in rows
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_questions(args: argparse.Namespace) -> list[str]:
    if args.question:
        return [args.question]
    if args.batch:
        path = Path(args.batch)
        source = sys.stdin if str(path) == "-" else path.open(encoding="utf-8")
        with source:
            return [line.strip() for line in source if line.strip()]
    return []


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="공시 Q&A 초안 YAML 생성 (에이전트 코드와 분리된 작성 도구)",
    )
    parser.add_argument("question", nargs="?", help="단일 질문")
    parser.add_argument(
        "--batch",
        metavar="FILE",
        help="질문 파일 (한 줄에 하나). '-' 이면 stdin",
    )
    parser.add_argument("--prefix", default="", help="question_id 접두사 (예: HY)")
    parser.add_argument(
        "--universe",
        type=Path,
        default=DEFAULT_UNIVERSE,
        help="universe.csv 경로",
    )
    parser.add_argument(
        "--format",
        choices=("yaml", "json"),
        default="yaml",
        help="출력 형식",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="저장 경로 (미지정 시 stdout). batch면 stem_N.yaml",
    )
    parser.add_argument(
        "--export-universe",
        type=Path,
        metavar="PATH",
        help="universe.csv → JSON (웹 UI용)",
    )
    args = parser.parse_args(argv)

    if args.export_universe:
        export_universe_json(args.universe, args.export_universe)
        print(args.export_universe, file=sys.stderr)
        return 0

    universe = load_universe(args.universe)
    questions = _read_questions(args)
    if not questions:
        parser.error("question 또는 --batch 가 필요합니다")

    drafts = [
        build_draft(q, universe=universe, prefix=args.prefix, index=i + 1)
        for i, q in enumerate(questions)
    ]
    drafts = [draft for draft in drafts if draft is not None]
    if not drafts:
        return 1

    if len(drafts) == 1 and not args.out:
        draft = drafts[0]
        payload = draft.to_yaml() if args.format == "yaml" else json.dumps(
            draft.to_dict(), ensure_ascii=False, indent=2
        ) + "\n"
        sys.stdout.write(payload)
        return 0

    out_dir = args.out or DEFAULT_OUTPUT_DIR
    if len(drafts) == 1:
        paths = [out_dir]
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "yaml" if args.format == "yaml" else "json"
        paths = [
            out_dir / f"{draft.question_id}.{ext}" if out_dir.is_dir() else out_dir
            for draft in drafts
        ]
        if not out_dir.is_dir() and len(drafts) > 1:
            parser.error("여러 질문 저장 시 --out 은 디렉터리여야 합니다")

    for draft, path in zip(drafts, paths, strict=True):
        if len(drafts) == 1:
            target = path
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target = path
        body = draft.to_yaml() if args.format == "yaml" else json.dumps(
            draft.to_dict(), ensure_ascii=False, indent=2
        ) + "\n"
        target.write_text(body, encoding="utf-8")
        print(target, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
