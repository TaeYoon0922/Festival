"""Metadata-filtered BM25 evaluation using only conditions extractable from a query."""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.parsing.final_validation import (
    BM25Index,
    GOLD_QUESTIONS,
    HOLDING_ADDITIONAL_QUESTIONS,
    _evaluation_document,
    _is_relevant,
)
from app.parsing.sampling import load_manifest


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _companies_from_query(
    query: str, company_aliases: dict[str, set[str]]
) -> list[str]:
    normalized_query = _normalized(query)
    matches: set[str] = set()
    for alias, canonical_names in company_aliases.items():
        normalized_alias = _normalized(alias)
        if not normalized_alias:
            continue
        if normalized_alias.isascii() and len(normalized_alias) <= 3:
            found = re.search(
                rf"(?<![0-9a-z]){re.escape(normalized_alias)}(?![0-9a-z])",
                query.casefold(),
            )
        else:
            found = normalized_alias in normalized_query
        if found:
            matches.update(canonical_names)
    return sorted(matches)


def _doc_group_from_query(query: str) -> tuple[str | None, str | None]:
    compact = _normalized(query)
    periodic_term = next(
        (
            term
            for term in (
                "사업보고서",
                "분기보고서",
                "반기보고서",
                "연결대상회사",
                "주요종속회사",
                "핵심감사사항",
            )
            if term in compact
        ),
        None,
    )
    if periodic_term:
        return "periodic", periodic_term
    if "보유" in compact and any(
        term in compact for term in ("주식", "수량", "비율", "증감", "증가", "감소")
    ):
        return "holding", "보유+주식/수량/비율/증감"
    if all(term in compact for term in ("풋옵션", "주식", "수량")):
        return "holding", "풋옵션+주식+수량"
    rules = (
        (
            "holding",
            (
                "보유주식",
                "보유수량",
                "보유수와비율",
                "보유비율",
                "현재보유",
                "증감주식",
                "증감수량",
                "증가주식수",
                "감소주식수",
                "변동후",
                "변동전",
                "국민연금기금",
                "직전보고",
            ),
        ),
        (
            "major",
            (
                "유상증자",
                "전환사채",
                "자기주식처분",
                "분할신설회사",
                "존속회사와소멸회사",
                "분할비율",
                "합병목적",
                "분할대상",
            ),
        ),
        ("exchange", ("단일판매", "공급계약", "계약해지", "시설투자", "우선협상수량", "수주계약금액")),
    )
    for group, terms in rules:
        matched = next((term for term in terms if term in compact), None)
        if matched:
            return group, matched
    return None, None


def _subtype_from_query(query: str, doc_group: str | None) -> tuple[str | None, str | None]:
    compact = _normalized(query)
    if doc_group == "periodic" or doc_group is None:
        if "분기" in compact:
            return "quarter", "분기"
        if "반기" in compact:
            return "half", "반기"
        if "사업보고서" in compact:
            return "annual", "사업보고서"
    if doc_group == "exchange":
        if "해지" in compact:
            return "단일판매공급계약해지", "해지"
        if "시설투자" in compact:
            return "신규시설투자등", "시설투자"
        if "공급계약" in compact or "수주계약" in compact:
            return "단일판매공급계약체결", "공급/수주계약"
    return None, None


def extract_metadata_filters(
    query: str, company_aliases: dict[str, set[str]]
) -> dict[str, Any]:
    companies = _companies_from_query(query, company_aliases)
    doc_group, group_evidence = _doc_group_from_query(query)
    subtype, subtype_evidence = _subtype_from_query(query, doc_group)
    if doc_group is None and subtype in {"quarter", "half", "annual"}:
        doc_group = "periodic"
        group_evidence = subtype_evidence
    years = sorted({int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", query)})
    apply_year_to_base_period = bool(years) and doc_group != "holding"
    return {
        "companies": companies,
        "years": years,
        "year_filter_applied": apply_year_to_base_period,
        "doc_group": doc_group,
        "doc_group_evidence": group_evidence,
        "doc_subtype": subtype,
        "doc_subtype_evidence": subtype_evidence,
    }


def _candidate_documents(
    rows: list[dict[str, Any]], filters: dict[str, Any]
) -> list[dict[str, Any]]:
    candidates = rows
    if filters["companies"]:
        companies = {_normalized(value) for value in filters["companies"]}
        candidates = [
            row for row in candidates if _normalized(row.get("corp_name")) in companies
        ]
    if filters["doc_group"]:
        candidates = [row for row in candidates if row.get("doc_group") == filters["doc_group"]]
    if filters["doc_subtype"]:
        candidates = [row for row in candidates if row.get("doc_subtype") == filters["doc_subtype"]]
    if filters["year_filter_applied"]:
        years = set(filters["years"])
        candidates = [
            row
            for row in candidates
            if row.get("base_year") is None or int(row["base_year"]) in years
        ]
    return candidates


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "question_count": count,
        "recall_at_1": round(sum(row["hit_at_1"] for row in rows) / count, 6) if count else 0,
        "recall_at_5": round(sum(row["hit_at_5"] for row in rows) / count, 6) if count else 0,
        "recall_at_10": round(sum(row["hit_at_10"] for row in rows) / count, 6) if count else 0,
        "metadata_filter_failure_count": sum(
            row["failure_class"] == "metadata_filter_failure" for row in rows
        ),
        "bm25_ranking_failure_count": sum(
            row["failure_class"] == "bm25_ranking_failure" for row in rows
        ),
        "gold_mapping_failure_count": sum(
            row["failure_class"] == "gold_mapping_failure" for row in rows
        ),
    }


def _ranked_chunk_summary(rank: int, chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk.get("chunk_id"),
        "doc_id": chunk.get("doc_id"),
        "report_nm": chunk.get("report_nm"),
        "rcept_dt": chunk.get("rcept_dt"),
        "chunk_type": chunk.get("chunk_type"),
        "projection_type": chunk.get("projection_type"),
        "section_title": chunk.get("section_title"),
        "content_preview": str(chunk.get("content") or "")[:240],
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Metadata-filtered BM25 evaluation",
        "",
        "Gold metadata was not used to construct candidate sets. Filters are extracted from question text.",
        "The pre-existing global BM25 files remain the baseline and were not overwritten.",
        "",
        "| set | mode | R@1 | R@5 | R@10 | metadata filter failures | BM25 failures | gold mapping failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (("gold_40", "Gold 40"), ("holding_20", "Holding 20")):
        filtered = report[name]
        baseline = report["global_baseline_preserved"][name]
        lines.append(
            f"| {label} | global baseline | {baseline['recall_at_1']:.3f} | "
            f"{baseline['recall_at_5']:.3f} | {baseline['recall_at_10']:.3f} | - | - | - |"
        )
        lines.append(
            f"| {label} | metadata-filtered | {filtered['recall_at_1']:.3f} | "
            f"{filtered['recall_at_5']:.3f} | {filtered['recall_at_10']:.3f} | "
            f"{filtered['metadata_filter_failure_count']} | "
            f"{filtered['bm25_ranking_failure_count']} | "
            f"{filtered['gold_mapping_failure_count']} |"
        )
    lines.extend(
        [
            "",
            "## Per-question results",
            "",
            "| id | set | company filter | year | group | subtype | candidate docs | gold doc included | candidate chunks | rank | R@1 | R@5 | R@10 | result |",
            "| --- | --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in report["questions"]:
        filters = row["extracted_filters"]
        rank = row["gold_chunk_rank"] if row["gold_chunk_rank"] is not None else "-"
        lines.append(
            f"| {row['question_id']} | {row['evaluation_set']} | "
            f"{', '.join(filters['companies']) or '-'} | "
            f"{', '.join(str(year) for year in filters['years']) or '-'} | "
            f"{filters['doc_group'] or '-'} | {filters['doc_subtype'] or '-'} | "
            f"{row['candidate_document_count']} | {row['gold_document_in_candidates']} | "
            f"{row['candidate_chunk_count']} | {rank} | {row['hit_at_1']} | "
            f"{row['hit_at_5']} | {row['hit_at_10']} | {row['failure_class']} |"
        )
    return "\n".join(lines)


def run_metadata_filtered_bm25(
    output_dir: Path, manifest_path: Path, global_baseline_dir: Path
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    company_aliases: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        canonical = str(row.get("corp_name") or "")
        if not canonical:
            continue
        company_aliases[canonical].add(canonical)
        listed_name = str(row.get("listed_name") or "")
        if listed_name:
            company_aliases[listed_name].add(canonical)
    questions = (*GOLD_QUESTIONS, *HOLDING_ADDITIONAL_QUESTIONS)
    plans: list[dict[str, Any]] = []
    candidate_doc_ids: set[str] = set()
    for question in questions:
        filters = extract_metadata_filters(str(question["query"]), company_aliases)
        candidates = _candidate_documents(manifest, filters)
        ids = {str(row["doc_id"]) for row in candidates}
        candidate_doc_ids.update(ids)
        plans.append({"question": question, "filters": filters, "candidate_doc_ids": ids})

    chunks_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    documents_by_doc: dict[str, dict[str, Any]] = {}
    records = [
        json.loads(line)
        for line in (output_dir / "index.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, record in enumerate(records, start=1):
        doc_id = str(record["doc_id"])
        if doc_id not in candidate_doc_ids:
            continue
        with gzip.open(output_dir / record["output_path"], "rt", encoding="utf-8") as source:
            payload = json.load(source)
        documents_by_doc[doc_id] = payload["document"]
        chunks_by_doc[doc_id].extend(
            chunk for chunk in payload["chunks"] if chunk.get("is_indexable", True)
        )
        if index % 500 == 0:
            print(f"metadata BM25 load [{index}/{len(records)}]", flush=True)

    result_rows: list[dict[str, Any]] = []
    for plan in plans:
        question = plan["question"]
        candidate_ids = plan["candidate_doc_ids"]
        candidate_chunks = [
            _evaluation_document(documents_by_doc[doc_id], chunk)
            for doc_id in sorted(candidate_ids)
            for chunk in chunks_by_doc.get(doc_id, [])
        ]
        gold_included = str(question["doc_id"]) in candidate_ids
        relevant = [chunk for chunk in candidate_chunks if _is_relevant(chunk, question)]
        if candidate_chunks:
            ranked = BM25Index(candidate_chunks).search(str(question["query"]))
            relevant_ranked = [
                (rank, chunk)
                for rank, (_, chunk) in enumerate(ranked, start=1)
                if _is_relevant(chunk, question)
            ]
            first_rank = relevant_ranked[0][0] if relevant_ranked else None
            top1 = ranked[0][1]["chunk_id"] if ranked else ""
            top_results = [
                _ranked_chunk_summary(rank, chunk)
                for rank, (_, chunk) in enumerate(ranked[:10], start=1)
            ]
            first_gold_result = (
                _ranked_chunk_summary(*relevant_ranked[0]) if relevant_ranked else None
            )
        else:
            first_rank = None
            top1 = ""
            top_results = []
            first_gold_result = None
        if not gold_included:
            failure_class = "metadata_filter_failure"
        elif not relevant:
            failure_class = "gold_mapping_failure"
        elif not first_rank or first_rank > 10:
            failure_class = "bm25_ranking_failure"
        else:
            failure_class = "success"
        result_rows.append(
            {
                "question_id": question["question_id"],
                "evaluation_set": "gold_40"
                if question["question_id"] in {item["question_id"] for item in GOLD_QUESTIONS}
                else "holding_20",
                "query": question["query"],
                "gold_doc_id": question["doc_id"],
                "extracted_filters": plan["filters"],
                "candidate_document_count": len(candidate_ids),
                "gold_document_in_candidates": gold_included,
                "candidate_chunk_count": len(candidate_chunks),
                "gold_relevant_chunk_count": len(relevant),
                "gold_chunk_rank": first_rank,
                "hit_at_1": bool(first_rank and first_rank <= 1),
                "hit_at_5": bool(first_rank and first_rank <= 5),
                "hit_at_10": bool(first_rank and first_rank <= 10),
                "failure_class": failure_class,
                "top1_chunk_id": top1,
                "top_10_results": top_results,
                "first_gold_result": first_gold_result,
            }
        )

    gold_rows = [row for row in result_rows if row["evaluation_set"] == "gold_40"]
    holding_rows = [row for row in result_rows if row["evaluation_set"] == "holding_20"]
    global_gold = json.loads((global_baseline_dir / "bm25_fixed_40.json").read_text(encoding="utf-8"))
    global_holding = json.loads((global_baseline_dir / "bm25_holding_20.json").read_text(encoding="utf-8"))
    report = {
        "method": {
            "flow": "Question -> Metadata Filter -> Candidate disclosures -> Candidate chunks -> BM25 -> Top-K",
            "allowed_filters": ["company mentions (OR)", "year/period", "doc_group", "doc_subtype when clear"],
            "gold_metadata_used_for_filtering": False,
            "company_extraction": "all exact normalized corp/listed-name mentions from manifest vocabulary, combined with OR",
            "year_policy": "base_year filter only when available; disabled for holding reference dates",
            "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "same as global baseline"},
        },
        "gold_40": _metrics(gold_rows),
        "holding_20": _metrics(holding_rows),
        "global_baseline_preserved": {
            "gold_40": global_gold["overall"],
            "holding_20": global_holding["overall"],
        },
        "failure_counts": dict(Counter(row["failure_class"] for row in result_rows)),
        "questions": result_rows,
    }
    report_dir = output_dir / "metadata_filtered_bm25"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metadata_filtered_bm25.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "metadata_filtered_bm25.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    fields = [
        "question_id", "evaluation_set", "query", "gold_doc_id", "extracted_filters",
        "candidate_document_count", "gold_document_in_candidates", "candidate_chunk_count",
        "gold_relevant_chunk_count", "gold_chunk_rank", "hit_at_1", "hit_at_5",
        "hit_at_10", "failure_class", "top1_chunk_id", "top_10_results",
        "first_gold_result",
    ]
    with (report_dir / "metadata_filtered_questions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for row in result_rows:
            writer.writerow(
                {
                    **row,
                    "extracted_filters": json.dumps(
                        row["extracted_filters"], ensure_ascii=False, separators=(",", ":")
                    ),
                    "top_10_results": json.dumps(
                        row["top_10_results"], ensure_ascii=False, separators=(",", ":")
                    ),
                    "first_gold_result": json.dumps(
                        row["first_gold_result"], ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )
    return report
