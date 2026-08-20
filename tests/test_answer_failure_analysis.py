from __future__ import annotations

import json
from pathlib import Path

from app.agent.answer_failure_analysis import (
    analyze_answer_not_supported,
    render_answer_not_supported_markdown,
    write_answer_not_supported_report,
)
from scripts.analyze_gold60_answer_not_supported import main


def _evaluation_row(
    *,
    question_id: str,
    question: str,
    chunk_id: str,
    content: str,
    generated_content: str,
    unresolved: tuple[str, ...],
) -> dict[str, object]:
    source_ref = {
        "table_id": f"table-{chunk_id}",
        "row_start": 2,
        "row_end": 3,
    }
    return {
        "question_id": question_id,
        "question": question,
        "evaluation_set": "gold40",
        "doc_group": "periodic",
        "status": "ok",
        "hit_at_10": True,
        "hybrid_gold_rank": 1,
        "end_to_end_failure_class": "answer_not_supported",
        "retrieved_chunks": [
            {
                "rank": 1,
                "chunk_id": chunk_id,
                "doc_id": "periodic-doc-1",
                "section_path": "사업의 내용 > 주요 제품",
                "content": content,
                "retrieval_text": content,
                "source_refs": [source_ref],
            }
        ],
        "source_references": [
            {
                "chunk_id": chunk_id,
                "doc_id": "periodic-doc-1",
                "source_refs": [source_ref],
            }
        ],
        "agent": {
            "answerable": False,
            "warnings": ["answer_not_supported"],
            "task_decision": {
                "task_type": "periodic_fact",
                "resolver_type": "periodic_fact_resolver",
            },
            "resolution": {
                "facts": [],
                "unresolved_requirements": list(unresolved),
                "warnings": list(unresolved),
            },
            "answer_draft": {
                "answerable": False,
                "warnings": ["answer_not_supported"],
            },
            "generated_answer": {
                "question": question,
                "answer_text": generated_content,
                "answerable": False,
                "warnings": ["answer_not_supported"],
                "sections": [
                    {
                        "title": "확인된 사업 내용",
                        "content": generated_content,
                        "citations": ["[1]"],
                    }
                ],
                "citations": [
                    {
                        "citation_id": "[1]",
                        "chunk_id": chunk_id,
                        "doc_id": "periodic-doc-1",
                        "source_refs": [source_ref],
                    }
                ],
            },
        },
        "gold": {
            "relevant_chunks": [
                {
                    "chunk_id": chunk_id,
                    "doc_id": "periodic-doc-1",
                    "section": "사업의 내용 > 주요 제품",
                }
            ]
        },
        "answer_gold_comparison": {
            "retrieval_hit_at_10": True,
            "gold_document_in_evidence": True,
            "gold_chunk_in_evidence": True,
            "answerable": False,
            "answer_supported": False,
        },
    }


def _report() -> dict[str, object]:
    supported = _evaluation_row(
        question_id="P01",
        question="DX 주요 제품은 무엇인가",
        chunk_id="periodic-doc-1:ch_supported",
        content="DX 주요 제품은 TV와 스마트폰이며 판매량은 100,000대이다.",
        generated_content=(
            "DX 주요 제품은 TV와 스마트폰이며 판매량은 100,000대이다. [1]"
        ),
        unresolved=("periodic_fact",),
    )
    unsupported = _evaluation_row(
        question_id="P02",
        question="영업이익은 얼마인가",
        chunk_id="periodic-doc-1:ch_other",
        content="매출은 100억원이다.",
        generated_content="영업이익은 999억원이다. [1]",
        unresolved=(),
    )
    return {
        "evaluation": "postgres_agent_gold60",
        "question_count": 3,
        "questions": [
            supported,
            unsupported,
            {
                **supported,
                "question_id": "P03",
                "end_to_end_failure_class": None,
                "agent": {**supported["agent"], "answerable": True},
            },
        ],
    }


def test_analysis_preserves_evidence_answer_and_source_references() -> None:
    analysis = analyze_answer_not_supported(_report())

    assert analysis["summary"]["case_count"] == 2
    assert analysis["summary"]["unsupported_claim_count"] == 1
    first = analysis["cases"][0]
    assert first["retrieved_evidence"][0]["chunk_id"].endswith("ch_supported")
    assert first["retrieved_evidence"][0]["source_refs"][0]["row_start"] == 2
    assert first["retrieved_evidence"][0]["relevant_content"]
    assert "TV와 스마트폰" in first["generated_answer"]["answer_text"]
    assert first["unsupported_claims"] == []
    assert first["cause"]["primary"] == "resolver 오류"
    assert first["cause"]["llm_hallucination"] is False


def test_analysis_detects_claim_outside_cited_evidence() -> None:
    analysis = analyze_answer_not_supported(_report())
    second = analysis["cases"][1]

    assert len(second["unsupported_claims"]) == 1
    assert second["unsupported_claims"][0]["claim"] == "영업이익은 999억원이다."
    assert (
        second["unsupported_claims"][0]["reason"]
        == "numeric_value_absent_from_cited_evidence"
    )
    assert second["cause"]["primary"] == "context 조합 오류"
    assert any(
        "citation scope" in improvement
        for improvement in second["improvements"]
    )


def test_markdown_and_json_report_are_written(tmp_path: Path) -> None:
    analysis = analyze_answer_not_supported(_report())
    markdown = render_answer_not_supported_markdown(analysis)

    assert "# Gold60 answer_not_supported 분석" in markdown
    assert "## P01" in markdown
    assert "### 1. Retrieved evidence" in markdown
    assert "### 5. 개선 방향" in markdown

    md_path, json_path = write_answer_not_supported_report(
        analysis,
        markdown_path=tmp_path / "answer_not_supported_analysis.md",
    )
    assert md_path.exists()
    assert json_path.exists()
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["summary"]["case_count"] == 2


def test_cli_reads_saved_evaluation_without_running_retrieval(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "gold60_agent_evaluation.json"
    output_path = tmp_path / "answer_not_supported_analysis.md"
    input_path.write_text(
        json.dumps(_report(), ensure_ascii=False), encoding="utf-8"
    )

    assert main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--expected-count",
            "2",
        ]
    ) == 0
    assert output_path.exists()
    assert output_path.with_suffix(".json").exists()
