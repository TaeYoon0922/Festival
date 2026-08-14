"""Local manifest/gzip implementation of the retrieval contracts."""

from __future__ import annotations

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from app.parsing.final_validation import BM25Index, _evaluation_document
from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {_normalized(value)}
    return {_normalized(item) for item in value}


def _integers(value: int | Sequence[int] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


class LocalManifestBackend:
    """Hard-filter manifest rows, then annotate optional soft matches.

    Hard filters: company/corp_code and explicit year/period.
    Soft boosts: doc_group, doc_subtype, correction status, and section path.
    """

    def __init__(self, rows: Sequence[dict[str, Any]]) -> None:
        self._rows = list(rows)

    def get_candidate_documents(
        self,
        company: str | Sequence[str] | None = None,
        year: int | Sequence[int] | None = None,
        period: int | tuple[int, int] | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        is_correction: bool | None = None,
        *,
        corp_code: str | Sequence[str] | None = None,
        section_path: str | None = None,
    ) -> list[CandidateDocument]:
        companies = _values(company)
        corp_codes = _values(corp_code)
        years = _integers(year)
        period_year: int | None = None
        period_month: int | None = None
        if isinstance(period, tuple):
            period_year, period_month = int(period[0]), int(period[1])
        elif isinstance(period, int):
            period_month = period
        candidates: list[CandidateDocument] = []
        for row in self._rows:
            names = {_normalized(row.get("corp_name")), _normalized(row.get("listed_name"))}
            if companies and not companies.intersection(names):
                continue
            if corp_codes and _normalized(row.get("corp_code")) not in corp_codes:
                continue
            row_year = row.get("base_year")
            row_month = row.get("base_month")
            if years and (row_year is None or int(row_year) not in years):
                continue
            if period_year is not None and (row_year is None or int(row_year) != period_year):
                continue
            if period_month is not None and (row_month is None or int(row_month) != period_month):
                continue
            soft = {
                "doc_group": doc_group is not None and row.get("doc_group") == doc_group,
                "doc_subtype": doc_subtype is not None and row.get("doc_subtype") == doc_subtype,
                "is_correction": is_correction is not None
                and bool(row.get("is_correction")) is is_correction,
                "section_path": False if section_path is not None else False,
            }
            active_soft = {
                key: value
                for key, value in soft.items()
                if {
                    "doc_group": doc_group,
                    "doc_subtype": doc_subtype,
                    "is_correction": is_correction,
                    "section_path": section_path,
                }[key]
                is not None
            }
            hard = {
                "company": sorted(companies),
                "corp_code": sorted(corp_codes),
                "year": sorted(years),
                "period": period,
            }
            candidates.append(
                CandidateDocument(
                    doc_id=str(row["doc_id"]),
                    metadata=dict(row),
                    metadata_match=MetadataMatch(
                        hard_filters={key: value for key, value in hard.items() if value},
                        soft_boosts=active_soft,
                        soft_inputs={
                            key: value
                            for key, value in {
                                "doc_group": doc_group,
                                "doc_subtype": doc_subtype,
                                "is_correction": is_correction,
                                "section_path": section_path,
                            }.items()
                            if value is not None
                        },
                        soft_score=float(sum(active_soft.values())),
                    ),
                )
            )
        return candidates


class LocalChunkBackend:
    def __init__(self, processed_dir: Path) -> None:
        self.processed_dir = processed_dir
        self._records_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line in (processed_dir / "index.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                self._records_by_doc[str(record["doc_id"])].append(record)

    def get_candidate_chunks(
        self, documents: Iterable[CandidateDocument]
    ) -> list[CandidateChunk]:
        candidates: list[CandidateChunk] = []
        for document in documents:
            for record in self._records_by_doc.get(document.doc_id, []):
                with gzip.open(
                    self.processed_dir / record["output_path"], "rt", encoding="utf-8"
                ) as source:
                    payload = json.load(source)
                for chunk in payload.get("chunks") or []:
                    if not chunk.get("is_indexable", True):
                        continue
                    soft = dict(document.metadata_match.soft_boosts)
                    requested_path = document.metadata_match.soft_inputs.get("section_path")
                    if requested_path:
                        path = " > ".join(chunk.get("section_path") or [])
                        soft["section_path"] = _normalized(requested_path) in _normalized(path)
                    match = MetadataMatch(
                        hard_filters=document.metadata_match.hard_filters,
                        soft_boosts=soft,
                        soft_inputs=document.metadata_match.soft_inputs,
                        soft_score=float(sum(soft.values())),
                    )
                    evaluation_chunk = _evaluation_document(payload["document"], chunk)
                    candidates.append(
                        CandidateChunk(
                            chunk_id=str(chunk["chunk_id"]),
                            doc_id=document.doc_id,
                            chunk=evaluation_chunk,
                            metadata_match=match,
                        )
                    )
        return candidates


class LocalBM25Retriever:
    """Candidate Documents -> Candidate Chunks -> BM25 with stable result schema."""

    def retrieve(
        self,
        query: str,
        candidates: Sequence[CandidateChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []
        by_id = {candidate.chunk_id: candidate for candidate in candidates}
        documents: list[dict[str, Any]] = []
        for candidate in candidates:
            document = dict(candidate.chunk)
            document.setdefault(
                "evaluation_text",
                str(document.get("retrieval_text") or document.get("content") or ""),
            )
            documents.append(document)
        ranked = BM25Index(documents).search(query)
        if top_k is not None:
            ranked = ranked[:top_k]
        results: list[RetrievalResult] = []
        for rank, (score, chunk) in enumerate(ranked, start=1):
            candidate = by_id[str(chunk["chunk_id"])]
            results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    doc_id=candidate.doc_id,
                    bm25_score=float(score),
                    rank=rank,
                    metadata_match=candidate.metadata_match.to_dict(),
                )
            )
        return results
