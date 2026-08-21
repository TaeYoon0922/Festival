"""Analyze Agent ``answer_not_supported`` failures from saved Gold60 output."""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from app.agent.gold60_evaluation import analyze_agent_gold60_failures


ANSWER_FAILURE_ANALYSIS_VERSION = "1"
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_MATERIAL_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_STATIC_LINE_MARKERS = (
    "확인된 사업 또는 공시 내용",
    "확인된 보유 변동 내역",
    "여러 기간의 공시에서 동일 사실이 확인됩니다",
    "여러 기간 또는 사실 후보를 자동으로 선택하지 않았습니다",
    "여러 변동 이벤트가 확인되어 특정 시점을 자동 선택하지 않았습니다",
    "확인되지 않은 정보가 있습니다",
    "답변 신뢰도",
    "추가 확인이 필요합니다",
    "provenance가 없어",
)
_CLAIM_LABELS = (
    "회사",
    "보고자",
    "변동일",
    "보고일",
    "접수일",
    "변동 전 주식수",
    "증감 주식수",
    "변동 후 주식수",
    "변동 전 비율",
    "변동 후 비율",
    "증감 비율",
    "변동 방향",
    "보고 기간",
    "보고서",
    "내용",
)
_TOKEN_STOPWORDS = {
    "내용",
    "보고서",
    "보고기간",
    "확인",
    "공시",
    "주요",
    "대한",
    "다음과",
    "같습니다",
    "입니다",
    "있습니다",
}


def analyze_answer_not_supported(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return grounded diagnostics without rerunning retrieval or generation."""

    failures = _answer_not_supported_rows(report)
    cases = [_analyze_case(row) for row in failures]
    cause_counts = Counter(case["cause"]["primary"] for case in cases)
    unsupported_count = sum(len(case["unsupported_claims"]) for case in cases)
    return {
        "analysis_version": ANSWER_FAILURE_ANALYSIS_VERSION,
        "scope": "answer_not_supported",
        "summary": {
            "case_count": len(cases),
            "unsupported_claim_count": unsupported_count,
            "cases_without_unsupported_claims": sum(
                not case["unsupported_claims"] for case in cases
            ),
            "cause_counts": dict(sorted(cause_counts.items())),
            "llm_hallucination_possible": False,
            "llm_hallucination_note": (
                "Current CitationAwareAnswerGenerator is deterministic and does not "
                "call an LLM; unsupported text is therefore classified as context, "
                "resolver, provenance, or validation behavior."
            ),
        },
        "cases": cases,
    }


def _answer_not_supported_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(report.get("questions"), Sequence):
        failure_report = analyze_agent_gold60_failures(report)
        by_id = {
            str(row.get("question_id")): row
            for row in report.get("questions") or []
            if isinstance(row, Mapping)
        }
        rows = []
        for failure in failure_report.get("failures") or []:
            if failure.get("failure_category") != "answer_not_supported":
                continue
            source = by_id.get(str(failure.get("question_id")), {})
            rows.append(_merge_full_evaluation_row(failure, source))
        return rows
    return [
        copy.deepcopy(dict(row))
        for row in report.get("failures") or []
        if isinstance(row, Mapping)
        and row.get("failure_category") == "answer_not_supported"
    ]


def _merge_full_evaluation_row(
    failure: Mapping[str, Any], source: Mapping[str, Any]
) -> dict[str, Any]:
    output = copy.deepcopy(dict(failure))
    agent = copy.deepcopy(dict(source.get("agent") or {}))
    output["answer"] = {
        **copy.deepcopy(dict(output.get("answer") or {})),
        "task_decision": agent.get("task_decision"),
        "resolution": agent.get("resolution"),
        "answer_draft": agent.get("answer_draft"),
        "generated_answer": agent.get("generated_answer"),
        "answerable": agent.get("answerable"),
        "warnings": agent.get("warnings") or [],
    }
    return output


def _analyze_case(row: Mapping[str, Any]) -> dict[str, Any]:
    retrieval = dict(row.get("retrieval") or {})
    answer = dict(row.get("answer") or {})
    generated = dict(answer.get("generated_answer") or {})
    source_references = [
        dict(value)
        for value in row.get("source_references") or []
        if isinstance(value, Mapping)
    ]
    retrieved_evidence = _retrieved_evidence(
        retrieval.get("retrieved_chunks") or [],
        source_references=source_references,
        gold=row.get("gold") or {},
    )
    unsupported = _unsupported_claims(
        generated,
        retrieved_evidence=retrieved_evidence,
    )
    cause = _classify_cause(
        answer,
        retrieved_evidence=retrieved_evidence,
        unsupported_claims=unsupported,
    )
    return {
        "question_id": row.get("question_id"),
        "evaluation_set": row.get("evaluation_set"),
        "doc_group": row.get("doc_group"),
        "question": row.get("question"),
        "retrieved_evidence": retrieved_evidence,
        "generated_answer": {
            "answerable": answer.get("answerable"),
            "answer_text": generated.get("answer_text") or "",
            "warnings": answer.get("warnings") or generated.get("warnings") or [],
            "confidence": generated.get("confidence") or {},
        },
        "unsupported_claims": unsupported,
        "cause": cause,
        "improvements": _improvements(cause, unsupported),
        "gold": copy.deepcopy(dict(row.get("gold") or {})),
        "answer_gold_comparison": copy.deepcopy(
            dict(row.get("answer_gold_comparison") or {})
        ),
    }


def _retrieved_evidence(
    chunks: Sequence[Any],
    *,
    source_references: Sequence[Mapping[str, Any]],
    gold: Mapping[str, Any],
) -> list[dict[str, Any]]:
    refs_by_chunk: dict[str, list[Mapping[str, Any]]] = {}
    for row in source_references:
        refs_by_chunk.setdefault(str(row.get("chunk_id") or ""), []).append(row)
    evidence_terms = [str(value) for value in gold.get("evidence_terms") or []]
    output = []
    for value in chunks:
        if not isinstance(value, Mapping):
            continue
        chunk = dict(value)
        chunk_id = str(chunk.get("chunk_id") or "")
        refs = _unique_mappings(
            [
                *(chunk.get("source_refs") or []),
                *(
                    ref
                    for citation in refs_by_chunk.get(chunk_id, [])
                    for ref in citation.get("source_refs") or []
                ),
            ]
        )
        provenance_paths = [
            copy.deepcopy(path)
            for citation in refs_by_chunk.get(chunk_id, [])
            for path in citation.get("provenance_path") or []
        ]
        content = str(chunk.get("content") or chunk.get("retrieval_text") or "")
        output.append(
            {
                "rank": chunk.get("rank"),
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id"),
                "section_path": copy.deepcopy(chunk.get("section_path") or []),
                "is_gold_relevant": bool(chunk.get("is_gold_relevant")),
                "source_refs": refs,
                "provenance_paths": provenance_paths,
                "relevant_content": _relevant_excerpt(content, evidence_terms),
                "content": content,
            }
        )
    return output


def _unsupported_claims(
    generated: Mapping[str, Any],
    *,
    retrieved_evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    citations = {
        str(row.get("citation_id")): dict(row)
        for row in generated.get("citations") or []
        if isinstance(row, Mapping) and row.get("citation_id")
    }
    chunks = {
        str(row.get("chunk_id")): row for row in retrieved_evidence
    }
    unsupported = []
    for section_value in generated.get("sections") or []:
        if not isinstance(section_value, Mapping):
            continue
        section = dict(section_value)
        section_citations = [str(value) for value in section.get("citations") or []]
        for line in str(section.get("content") or "").splitlines():
            claim = _claim_text(line)
            if not claim:
                continue
            inline_ids = [f"[{value}]" for value in _CITATION_PATTERN.findall(line)]
            citation_ids = inline_ids or section_citations
            cited_chunk_ids = [
                str(citations[citation_id].get("chunk_id") or "")
                for citation_id in citation_ids
                if citation_id in citations
            ]
            evidence_text = "\n".join(
                str(chunks[chunk_id].get("content") or "")
                for chunk_id in cited_chunk_ids
                if chunk_id in chunks
            )
            supported, reason = _claim_supported(claim, evidence_text, citation_ids)
            if not supported:
                unsupported.append(
                    {
                        "claim": claim,
                        "section": section.get("title"),
                        "citation_ids": citation_ids,
                        "cited_chunk_ids": cited_chunk_ids,
                        "reason": reason,
                    }
                )
    return unsupported


def _claim_text(line: str) -> str | None:
    text = _CITATION_PATTERN.sub("", line).strip()
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text).strip()
    if not text or text in {"주의", "인용"}:
        return None
    if any(marker in text for marker in _STATIC_LINE_MARKERS):
        return None
    if text.endswith(":") and len(text) < 40:
        return None
    material = bool(
        re.search(r"\d", text)
        or any(text.startswith(f"{label}:") for label in _CLAIM_LABELS)
        or len(_material_tokens(text)) >= 2
    )
    return text if material else None


def _claim_supported(
    claim: str, evidence_text: str, citation_ids: Sequence[str]
) -> tuple[bool, str]:
    if not citation_ids:
        return False, "claim_has_no_citation"
    if not evidence_text:
        return False, "cited_chunk_not_in_retrieved_evidence"
    payload = re.sub(r"^[^:]{1,30}:\s*", "", claim).strip()
    normalized_claim = _normalize(payload)
    normalized_evidence = _normalize(evidence_text)
    if normalized_claim and normalized_claim in normalized_evidence:
        return True, "exact_normalized_match"
    claim_numbers = re.findall(r"\d+(?:[.,]\d+)*", payload)
    evidence_numbers = {
        value.replace(",", "")
        for value in re.findall(r"\d+(?:[.,]\d+)*", evidence_text)
    }
    if any(value.replace(",", "") not in evidence_numbers for value in claim_numbers):
        return False, "numeric_value_absent_from_cited_evidence"
    tokens = _material_tokens(payload)
    if not tokens:
        return True, "no_material_token_to_contradict"
    evidence_tokens = set(_material_tokens(evidence_text))
    coverage = sum(token in evidence_tokens for token in tokens) / len(tokens)
    if coverage >= 0.8:
        return True, "material_token_coverage"
    return False, "claim_text_absent_from_cited_evidence"


def _material_tokens(value: Any) -> list[str]:
    return [
        token
        for token in _MATERIAL_TOKEN_PATTERN.findall(str(value or "").casefold())
        if token not in _TOKEN_STOPWORDS
    ]


def _classify_cause(
    answer: Mapping[str, Any],
    *,
    retrieved_evidence: Sequence[Mapping[str, Any]],
    unsupported_claims: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    warnings = [str(value) for value in answer.get("warnings") or []]
    resolution = dict(answer.get("resolution") or {})
    draft = dict(answer.get("answer_draft") or {})
    confidence = dict(draft.get("confidence") or {})
    unresolved = [
        str(value)
        for value in (
            resolution.get("unresolved_fields")
            or resolution.get("unresolved_requirements")
            or confidence.get("unresolved_requirements")
            or []
        )
    ]
    combined = " ".join([*warnings, *unresolved])
    contributing = []
    if unsupported_claims:
        primary = "context 조합 오류"
        explanation = (
            "Generated factual text is not supported by its cited retrieved chunks."
        )
        contributing.append("citation validation 필요")
    elif any(marker in combined for marker in ("provenance", "citation")):
        primary = "citation/provenance 누락"
        explanation = "Answerability failed because required source provenance is missing."
    elif any(
        marker in combined
        for marker in (
            "no_periodic_fact_evidence",
            "no_holding",
            "explicit_period_unmatched",
            "evidence_presence",
        )
    ):
        primary = "evidence 범위 초과"
        explanation = (
            "Retrieved evidence did not contain the resolver fields or requested period."
        )
    elif any(
        marker in combined
        for marker in (
            "complete_requested",
            "unconflicted_fact",
            "matching_holding_event",
            "periodic_fact",
            "holding_event",
        )
    ):
        primary = "resolver 오류"
        explanation = (
            "The resolver could not construct all required fields from retrieved evidence."
        )
    elif not retrieved_evidence:
        primary = "evidence 범위 초과"
        explanation = "No retrieved evidence was serialized for the answer."
    else:
        primary = "answer constraint"
        explanation = (
            "The deterministic answerability constraint rejected an otherwise grounded answer."
        )
    if "conflict" in combined or "period_evolution" in combined:
        contributing.append("resolver conflict handling")
    return {
        "primary": primary,
        "contributing": list(dict.fromkeys(contributing)),
        "explanation": explanation,
        "llm_hallucination": False,
        "prompt_problem": False,
        "unresolved_requirements": unresolved,
    }


def _improvements(
    cause: Mapping[str, Any], unsupported_claims: Sequence[Mapping[str, Any]]
) -> list[str]:
    primary = cause.get("primary")
    values = []
    if unsupported_claims:
        values.extend(
            [
                "Answer Generator의 claim별 citation scope를 검증한다.",
                "cited retrieved chunk에 없는 숫자ㆍ날짜는 출력하지 않는 constraint를 유지한다.",
                "unsupported claim 발견 시 해당 문장을 제거하고 answerable=false로 유지한다.",
            ]
        )
    if primary == "citation/provenance 누락":
        values.extend(
            [
                "projection_field_ref/source_ref가 원본 table row까지 연결되는지 보강한다.",
                "Answer Composer에서 provenance 없는 citation을 answerability 근거로 사용하지 않는다.",
            ]
        )
    elif primary == "resolver 오류":
        values.extend(
            [
                "resolver의 required-field 추출 및 alias mapping을 진단한다.",
                "동일 evidence group의 표ㆍprojection 값을 보존적으로 결합한다.",
            ]
        )
    elif primary == "evidence 범위 초과":
        values.extend(
            [
                "retrieved Top-K에 요구 필드와 기간을 포함한 chunk가 있는지 먼저 확인한다.",
                "근거가 없으면 답을 확장하지 않고 미확인 필드를 명시한다.",
            ]
        )
    elif primary == "answer constraint":
        values.extend(
            [
                "answerability required-field 규칙과 실제 AnswerDraft 필드를 비교한다.",
                "근거가 모두 존재할 때만 과도한 false negative constraint를 완화한다.",
            ]
        )
    values.append(
        "현재 생성 계층은 LLM을 호출하지 않으므로 prompt 수정은 우선순위가 아니다."
    )
    return list(dict.fromkeys(values))


def _relevant_excerpt(text: str, evidence_terms: Sequence[str], limit: int = 900) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    normalized = compact.casefold()
    positions = [
        normalized.find(term.casefold())
        for term in evidence_terms
        if term and normalized.find(term.casefold()) >= 0
    ]
    center = min(positions) if positions else 0
    start = max(0, center - limit // 4)
    end = min(len(compact), start + limit)
    prefix = "…" if start else ""
    suffix = "…" if end < len(compact) else ""
    return prefix + compact[start:end] + suffix


def _unique_mappings(values: Sequence[Any]) -> list[dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        copied = copy.deepcopy(dict(value))
        key = json.dumps(copied, ensure_ascii=False, sort_keys=True, default=str)
        output.setdefault(key, copied)
    return list(output.values())


def _normalize(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", str(value or "").casefold())


def write_answer_not_supported_report(
    analysis: Mapping[str, Any],
    output_path: Path | None = None,
    *,
    markdown_path: Path | None = None,
    json_path: Path | None = None,
) -> tuple[Path, Path]:
    target_markdown = markdown_path or output_path
    if target_markdown is None:
        raise ValueError("output_path or markdown_path is required")
    target_json = json_path or target_markdown.with_suffix(".json")
    target_markdown.parent.mkdir(parents=True, exist_ok=True)
    target_json.parent.mkdir(parents=True, exist_ok=True)
    target_markdown.write_text(
        render_answer_not_supported_markdown(analysis), encoding="utf-8"
    )
    target_json.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return target_markdown, target_json


def render_answer_not_supported_markdown(analysis: Mapping[str, Any]) -> str:
    summary = dict(analysis.get("summary") or {})
    lines = [
        "# Gold60 answer_not_supported 분석",
        "",
        "## Summary",
        "",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Unsupported claims: {summary.get('unsupported_claim_count', 0)}",
        f"- Cases without unsupported claims: {summary.get('cases_without_unsupported_claims', 0)}",
        "- LLM hallucination possible: false (deterministic generator, no LLM call)",
        "",
        "| Cause | Count |",
        "|---|---:|",
    ]
    counts = dict(summary.get("cause_counts") or {})
    lines.extend(f"| {name} | {count} |" for name, count in counts.items())
    if not counts:
        lines.append("| none | 0 |")

    for value in analysis.get("cases") or []:
        case = dict(value)
        generated = dict(case.get("generated_answer") or {})
        cause = dict(case.get("cause") or {})
        lines.extend(
            [
                "",
                f"## {case.get('question_id')}: {case.get('question')}",
                "",
                "### 1. Retrieved evidence",
                "",
            ]
        )
        evidence = case.get("retrieved_evidence") or []
        if not evidence:
            lines.append("검색된 evidence가 없습니다.")
        for item_value in evidence:
            item = dict(item_value)
            section = " > ".join(str(part) for part in item.get("section_path") or [])
            lines.extend(
                [
                    f"#### Rank {item.get('rank')}: `{item.get('chunk_id')}`",
                    "",
                    f"- doc_id: `{item.get('doc_id')}`",
                    f"- section: `{section}`",
                    f"- gold relevant: `{item.get('is_gold_relevant')}`",
                    f"- source reference: `{json.dumps(item.get('source_refs') or [], ensure_ascii=False)}`",
                    "",
                    "관련 근거 내용:",
                    "",
                    str(item.get("relevant_content") or ""),
                    "",
                ]
            )
        lines.extend(
            [
                "### 2. Generated answer",
                "",
                "```text",
                str(generated.get("answer_text") or ""),
                "```",
                "",
                f"- answerable: `{generated.get('answerable')}`",
                f"- warnings: `{json.dumps(generated.get('warnings') or [], ensure_ascii=False)}`",
                "",
                "### 3. Unsupported claim",
                "",
            ]
        )
        unsupported = case.get("unsupported_claims") or []
        if unsupported:
            for claim_value in unsupported:
                claim = dict(claim_value)
                lines.extend(
                    [
                        f"- `{claim.get('claim')}`",
                        f"  - reason: `{claim.get('reason')}`",
                        f"  - citations: `{', '.join(claim.get('citation_ids') or [])}`",
                        f"  - cited chunks: `{', '.join(claim.get('cited_chunk_ids') or [])}`",
                    ]
                )
        else:
            lines.append(
                "- 없음. 이 케이스는 hallucination이 아니라 answerability/provenance/resolver 조건 미충족입니다."
            )
        lines.extend(
            [
                "",
                "### 4. 원인 분류",
                "",
                f"- primary: `{cause.get('primary')}`",
                f"- explanation: {cause.get('explanation')}",
                f"- LLM hallucination: `{cause.get('llm_hallucination')}`",
                f"- prompt problem: `{cause.get('prompt_problem')}`",
                f"- unresolved requirements: `{json.dumps(cause.get('unresolved_requirements') or [], ensure_ascii=False)}`",
                "",
                "### 5. 개선 방향",
                "",
                *(f"- {item}" for item in case.get("improvements") or []),
            ]
        )
    return "\n".join(lines) + "\n"
