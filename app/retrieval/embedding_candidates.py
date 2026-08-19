"""Collect embedding candidates from query routing without consulting gold answers."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.reasoning.router import QueryRouter
from app.retrieval.embeddings import chunk_embedding_text


ProgressCallback = Callable[[int, int, str, int], None]


@dataclass(frozen=True)
class CandidateCollection:
    chunks: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


class Gold60EmbeddingCandidateCollector:
    """Union pre-retrieval candidate universes produced only from question text."""

    def __init__(
        self,
        understanding: Any,
        backend: Any,
        *,
        router: QueryRouter | None = None,
    ) -> None:
        self.understanding = understanding
        self.backend = backend
        self.router = router or QueryRouter()

    def collect(
        self,
        question_sets: Mapping[str, Sequence[Mapping[str, Any]]],
        *,
        progress: ProgressCallback | None = None,
    ) -> CandidateCollection:
        scheduled = [
            (set_name, str(question["question_id"]), str(question["query"]))
            for set_name, questions in question_sets.items()
            for question in questions
        ]
        chunks_by_id: dict[str, dict[str, Any]] = {}
        candidate_counts: list[int] = []
        question_rows: list[dict[str, Any]] = []
        total_occurrences = 0

        for index, (set_name, question_id, query) in enumerate(scheduled, start=1):
            plan = self.understanding.understand(query, top_k=10)
            route = self.router.route(plan)
            documents = self.backend.get_candidate_documents(**route.backend_filters)
            documents = self.router.filter_documents(documents, route)
            chunks = self.backend.get_candidate_chunks(documents)
            chunks = self.router.prepare_chunks(chunks, route)
            count = len(chunks)
            candidate_counts.append(count)
            total_occurrences += count
            question_rows.append(
                {
                    "evaluation_set": set_name,
                    "question_id": question_id,
                    "candidate_count": count,
                }
            )
            for candidate in chunks:
                row = chunks_by_id.get(candidate.chunk_id)
                if row is None:
                    chunk = candidate.chunk
                    row = {
                        "chunk_id": candidate.chunk_id,
                        "doc_id": candidate.doc_id,
                        "corp_code": chunk.get("corp_code"),
                        "corp_name": chunk.get("corp_name"),
                        "doc_group": chunk.get("doc_group"),
                        "chunk_type": chunk.get("chunk_type"),
                        "retrieval_text": chunk_embedding_text(chunk),
                        "candidate_occurrences": 0,
                    }
                    chunks_by_id[candidate.chunk_id] = row
                row["candidate_occurrences"] += 1
            if progress:
                progress(index, len(scheduled), question_id, count)

        chunks = tuple(chunks_by_id[key] for key in sorted(chunks_by_id))
        summary = {
            "selection_basis": (
                "question text -> QueryUnderstanding -> Router -> metadata candidate "
                "documents -> candidate chunks; gold doc/chunk fields are not read"
            ),
            "total_questions": len(scheduled),
            "total_candidate_occurrences": total_occurrences,
            "unique_candidate_chunks": len(chunks),
            "candidate_count_stats": _count_stats(candidate_counts),
            "doc_group_distribution": _distribution(chunks, "doc_group"),
            "chunk_type_distribution": _distribution(chunks, "chunk_type"),
            "company_distribution": _distribution(chunks, "corp_name"),
            "questions": question_rows,
        }
        return CandidateCollection(chunks=chunks, summary=summary)


def write_candidate_collection(
    collection: CandidateCollection, output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = collection.chunks
    (output_dir / "candidate_chunk_ids.txt").write_text(
        "".join(f"{chunk['chunk_id']}\n" for chunk in chunks), encoding="utf-8"
    )
    with (output_dir / "candidate_chunks.jsonl").open("w", encoding="utf-8") as target:
        for chunk in chunks:
            target.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")))
            target.write("\n")
    (output_dir / "candidate_summary.json").write_text(
        json.dumps(collection.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "candidate_summary.md").write_text(
        _markdown_summary(collection.summary), encoding="utf-8"
    )


def _count_stats(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "median": 0.0, "p95": 0, "max": 0}
    ordered = sorted(values)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "min": ordered[0],
        "median": float(statistics.median(ordered)),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def _distribution(
    chunks: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    counts = Counter(str(chunk.get(field) or "unknown") for chunk in chunks)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    stats = summary["candidate_count_stats"]
    lines = [
        "# Gold60 embedding candidate summary",
        "",
        f"- Questions: {summary['total_questions']}",
        f"- Candidate occurrences: {summary['total_candidate_occurrences']}",
        f"- Unique candidate chunks: {summary['unique_candidate_chunks']}",
        "- Candidate count min/median/p95/max: "
        f"{stats['min']} / {stats['median']} / {stats['p95']} / {stats['max']}",
        "- Selection: question-derived metadata routing only; no gold answer leakage",
    ]
    for heading, key in (
        ("Document group", "doc_group_distribution"),
        ("Chunk type", "chunk_type_distribution"),
        ("Company", "company_distribution"),
    ):
        lines.extend(["", f"## {heading}", "", "| value | chunks |", "| --- | ---: |"])
        lines.extend(f"| {name} | {count} |" for name, count in summary[key].items())
    return "\n".join(lines) + "\n"
