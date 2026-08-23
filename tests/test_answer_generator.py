import copy
import unittest
from dataclasses import FrozenInstanceError, replace

from app.generation.answer_generator import (
    CitationAwareAnswerGenerator,
    GeneratedAnswer,
    _periodic_claim_payload,
    generate_answer,
    validate_periodic_citation_scope,
)
from app.reasoning.answer_composer import (
    AnswerDraft,
    AnswerSection,
    EvidenceCitation,
    compose_holding_answer,
    compose_periodic_answer,
)
from app.reasoning.holding_event_resolver import resolve_holding_events
from app.reasoning.periodic_fact_resolver import resolve_periodic_facts
from tests.test_holding_event_resolver import (
    _evidence_set as _holding_evidence,
    _group as _holding_group,
    _item as _holding_item,
)
from tests.test_periodic_fact_resolver import (
    _evidence as _periodic_evidence,
    _group as _periodic_group,
    _item as _periodic_item,
)


def _multiple_holding_draft():
    older = _holding_item(
        "h23:ch_report",
        "h23",
        rank=1,
        table_id="t23",
        fields={
            "reporter": "국민연금기금",
            "reference_date": "2024-05-09",
            "after_shares": "1,138,905",
        },
    )
    newer = _holding_item(
        "h25:ch_report",
        "h25",
        rank=2,
        table_id="t25",
        fields={
            "reporter": "국민연금기금",
            "reference_date": "2025-07-28",
            "after_shares": "1,037,916",
        },
    )
    evidence = _holding_evidence(
        [_holding_group("g23", older), _holding_group("g25", newer)],
        question="국민연금기금 변동일 변동후 주식수",
    )
    return compose_holding_answer(resolve_holding_events(evidence), evidence)


def _repeated_periodic_draft():
    older = _periodic_item(
        "p24:ch_fact",
        "p24",
        rank=1,
        text="동일한 주요 제품 및 사업 내용",
        year=2024,
        table_id="t24",
    )
    newer = _periodic_item(
        "p25:ch_fact",
        "p25",
        rank=2,
        text="동일한 주요 제품 및 사업 내용",
        year=2025,
        table_id="t25",
    )
    evidence = _periodic_evidence(
        [
            _periodic_group(
                "g-repeat",
                older,
                newer,
                group_type="periodic_repeated_fact",
            )
        ]
    )
    return compose_periodic_answer(resolve_periodic_facts(evidence), evidence)


class AnswerGeneratorTests(unittest.TestCase):
    def test_periodic_claim_payload_ignores_markdown_separator(self):
        self.assertIsNone(_periodic_claim_payload("| --- | --- |"))

    def test_periodic_claim_payload_ignores_aligned_markdown_separator(self):
        self.assertIsNone(_periodic_claim_payload("| :--- | ---: | :---: |"))

    def test_periodic_claim_payload_keeps_factual_table_row(self):
        row = "| 익산공장 | 58.0 | 32.7 | 56% | [1]"
        self.assertEqual(
            _periodic_claim_payload(row),
            "| 익산공장 | 58.0 | 32.7 | 56% |",
        )

    def test_periodic_claim_payload_keeps_table_header(self):
        header = "| 사업소 | 생산능력 | 생산실적 | 평균가동률 | [1]"
        self.assertEqual(
            _periodic_claim_payload(header),
            "| 사업소 | 생산능력 | 생산실적 | 평균가동률 |",
        )

    def test_periodic_markdown_separator_does_not_fail_citation_scope(self):
        table_text = (
            "| 사업소 | 생산능력 | 생산실적 | 평균가동률 |\n"
            "| :--- | ---: | ---: | ---: |\n"
            "| 익산공장 | 58.0 | 32.7 | 56% |"
        )
        item = _periodic_item(
            "p07:ch_table",
            "p07",
            rank=1,
            text=table_text,
            year=2023,
            table_id="t07",
        )
        evidence = _periodic_evidence(
            [_periodic_group("g-table", item, group_type="document_evidence")]
        )

        generated = generate_answer(
            compose_periodic_answer(resolve_periodic_facts(evidence), evidence)
        )

        self.assertTrue(generated.answerable)
        self.assertIn("| :--- | ---: | ---: | ---: |", generated.answer_text)
        self.assertFalse(
            any(
                warning.startswith("unsupported_periodic_claim_removed")
                for warning in generated.warnings
            )
        )

    def test_holding_multiple_events_and_order_are_preserved(self):
        generated = CitationAwareAnswerGenerator().generate(
            _multiple_holding_draft()
        )

        self.assertIsInstance(generated, GeneratedAnswer)
        self.assertLess(
            generated.answer_text.index("2024-05-09"),
            generated.answer_text.index("2025-07-28"),
        )
        self.assertIn("1,138,905주", generated.answer_text)
        self.assertIn("1,037,916주", generated.answer_text)
        self.assertNotIn("최신", generated.answer_text)
        self.assertTrue(generated.answerable)
        with self.assertRaises(FrozenInstanceError):
            generated.answerable = False

    def test_holding_citations_and_ambiguity_warning_are_preserved(self):
        generated = generate_answer(_multiple_holding_draft())

        self.assertEqual(len(generated.citations), 2)
        self.assertEqual(
            [(citation.doc_id, citation.chunk_id) for citation in generated.citations],
            [("h23", "h23:ch_report"), ("h25", "h25:ch_report")],
        )
        # Each event keeps its own citation on its own line; the compact
        # multi-event form no longer prefixes the date with a field label.
        lines = generated.answer_text.splitlines()
        self.assertTrue(
            any(
                line.startswith("2024-05-09") and line.endswith("[1]")
                for line in lines
            )
        )
        self.assertTrue(
            any(
                line.startswith("2025-07-28") and line.endswith("[2]")
                for line in lines
            )
        )
        self.assertIn("특정 시점을 자동 선택하지 않았습니다", generated.answer_text)
        self.assertIn("multiple_matching_holding_events", generated.warnings)

    def test_periodic_repeated_periods_and_sources_are_preserved(self):
        generated = generate_answer(_repeated_periodic_draft())

        self.assertIn("2024년", generated.answer_text)
        self.assertIn("2025년", generated.answer_text)
        self.assertEqual(
            generated.answer_text.count("동일한 주요 제품 및 사업 내용"), 2
        )
        self.assertIn("여러 기간의 공시에서 동일 사실이 확인됩니다", generated.answer_text)
        self.assertNotIn("최신", generated.answer_text)
        self.assertEqual(len(generated.citations), 2)
        self.assertTrue(generated.answerable)

    def test_periodic_metadata_is_outside_citation_claim_scope(self):
        generated = generate_answer(_repeated_periodic_draft())
        fact_section = generated.sections[0]

        self.assertTrue(any("보고 기간" in row for row in fact_section.metadata))
        self.assertTrue(all("[" not in row for row in fact_section.metadata))
        self.assertNotIn("보고 기간", fact_section.content)
        factual_lines = [
            line for line in fact_section.content.splitlines() if line.startswith("내용:")
        ]
        self.assertTrue(factual_lines)
        self.assertTrue(all("[" in line for line in factual_lines))

    def test_periodic_citation_validator_removes_claim_outside_selected_source(self):
        draft = _repeated_periodic_draft()
        generated = generate_answer(draft)
        unsafe_section = replace(
            generated.sections[0],
            content=(
                generated.sections[0].content
                + "\n내용: 선택된 근거에 없는 매출액 999억원 [1]"
            ),
        )

        sections, warnings, valid = validate_periodic_citation_scope(
            draft,
            (unsafe_section,),
            generated.citations,
        )

        self.assertFalse(valid)
        self.assertNotIn("999억원", sections[0].content)
        self.assertTrue(
            any(value.startswith("unsupported_periodic_claim_removed") for value in warnings)
        )

    def test_periodic_projected_table_header_is_not_a_standalone_claim(self):
        header = "내용: | 열 1 | 제 58 기 1분기 / 누적 |"
        row = "| 매출액 | 44,407,761 | [1]"

        self.assertIsNone(_periodic_claim_payload(header))
        self.assertEqual(_periodic_claim_payload(row), "| 매출액 | 44,407,761 |")

    def test_periodic_long_source_text_is_bounded_for_display(self):
        item = _periodic_item(
            "p25:ch_long",
            "p25",
            rank=1,
            text="주요 제품 설명 " + ("가" * 2000),
            year=2025,
            table_id="t25",
        )
        evidence = _periodic_evidence(
            [_periodic_group("g-long", item, group_type="document_evidence")]
        )
        draft = compose_periodic_answer(resolve_periodic_facts(evidence), evidence)

        generated = generate_answer(draft)

        self.assertTrue(generated.answerable)
        self.assertLess(len(generated.answer_text), 1100)
        self.assertIn("[1]", generated.answer_text)

    def test_periodic_conflict_alternatives_are_preserved(self):
        first = _periodic_item(
            "p1:ch_a", "p1", rank=1, text="제품 수는 10개", year=2024
        )
        second = _periodic_item(
            "p1:ch_b", "p1", rank=2, text="제품 수는 12개", year=2024
        )
        evidence = _periodic_evidence(
            [
                _periodic_group(
                    "g-conflict",
                    first,
                    second,
                    group_type="document_evidence",
                )
            ]
        )
        draft = compose_periodic_answer(resolve_periodic_facts(evidence), evidence)

        generated = generate_answer(draft)

        self.assertIn("상충하는 대안", generated.answer_text)
        self.assertIn("제품 수는 10개", generated.answer_text)
        self.assertIn("제품 수는 12개", generated.answer_text)
        self.assertIn("확인되지 않은 정보가 있습니다", generated.answer_text)
        self.assertFalse(generated.answerable)

    def test_missing_provenance_prevents_citation_and_fact_rendering(self):
        draft = _multiple_holding_draft()
        citations = tuple(
            replace(citation, provenance_path=()) for citation in draft.citations
        )
        unsafe = replace(draft, citations=citations)

        generated = generate_answer(unsafe)

        self.assertEqual(generated.citations, ())
        self.assertNotIn("1,138,905", generated.answer_text)
        self.assertNotIn("1,037,916", generated.answer_text)
        self.assertIn("provenance가 없어", generated.answer_text)
        self.assertFalse(generated.answerable)
        self.assertTrue(
            any(value.startswith("citation_missing_provenance") for value in generated.warnings)
        )

    def test_general_generation_emits_only_draft_evidence_content(self):
        citation = EvidenceCitation(
            chunk_id="d1:ch_fact",
            doc_id="d1",
            source_refs=({"table_id": "t1", "row_start": 1, "row_end": 1},),
            provenance_path=(
                {
                    "resolver": None,
                    "source_chunk_id": "d1:ch_fact",
                    "source_doc_id": "d1",
                },
            ),
        )
        draft = AnswerDraft(
            question="일반 질문",
            task_type="general_evidence",
            answer_sections=(
                AnswerSection(
                    title="General evidence",
                    content={
                        "evidence": [
                            {
                                "chunk_id": "d1:ch_fact",
                                "evidence_text": "검증된 원문 ONLY_TOKEN",
                            }
                        ]
                    },
                    supporting_evidence_ids=("d1:ch_fact",),
                ),
            ),
            evidence_references=("d1:ch_fact",),
            citations=(citation,),
            ambiguity={},
            warnings=(),
            confidence={"level": "high", "score": 1.0},
            answerable=True,
        )

        generated = generate_answer(draft)

        evidence_section = generated.sections[0]
        self.assertEqual(evidence_section.content, "1. 검증된 원문 ONLY_TOKEN [1]")
        for unsupported in ("시장 전망", "투자 추천", "업계 1위", "향후 성장"):
            self.assertNotIn(unsupported, generated.answer_text)

    def test_answer_draft_is_not_mutated(self):
        draft = _repeated_periodic_draft()
        before = copy.deepcopy(draft.to_dict())

        generate_answer(draft)

        self.assertEqual(draft.to_dict(), before)

    def test_confidence_labels_follow_answer_draft(self):
        draft = _repeated_periodic_draft()
        high = generate_answer(replace(draft, confidence={"level": "high"}))
        medium = generate_answer(replace(draft, confidence={"level": "medium"}))
        low = generate_answer(
            replace(draft, confidence={"level": "low"}, answerable=False)
        )

        self.assertIn("답변 신뢰도: 높음", high.answer_text)
        self.assertIn("답변 신뢰도: 중간", medium.answer_text)
        self.assertIn("추가 확인이 필요합니다", low.answer_text)

    def test_periodic_renderer_keeps_named_metric_row_and_basis(self):
        table = (
            "| 열 1 | 제 58 기 1분기 / 3개월 | 제 57 기 1분기 / 3개월 |\n"
            "| --- | --- | --- |\n"
            "| 매출액 | 44,407,761 | 40,658,539 |\n"
            "| 매출원가 | 35,428,253 | 32,230,756 |\n"
            "| 보통주기본주당이익(손실) (단위 : 원) | 12,076 | 12,287 |\n"
        )
        item = _periodic_item(
            "p:ch_income",
            "p",
            rank=1,
            text=table,
            year=2025,
            quarter=1,
            section_path=("연결포괄손익계산서",),
            statement_scope="연결",
            temporal_match=True,
        )
        evidence = _periodic_evidence(
            [_periodic_group("g-income", item, group_type="document_evidence")],
            question="테스트회사 2025년 1분기 연결 매출액",
            year=2025,
            task_type="financial_metric",
        )
        plan = copy.deepcopy(dict(evidence.query_plan))
        plan.update(
            {
                "metric": "매출액",
                "basis": "consolidated",
                "lexical_query": "연결 매출액",
            }
        )
        plan["period"]["quarter"] = 1
        evidence = replace(evidence, query_plan=plan)
        generated = generate_answer(
            compose_periodic_answer(
                resolve_periodic_facts(evidence, query_plan=plan), evidence
            )
        )

        self.assertIn("44,407,761", generated.answer_text)
        self.assertIn("매출액", generated.answer_text)
        self.assertIn("재무제표 기준: 연결", generated.answer_text)
        self.assertNotIn("매출원가", generated.answer_text)
        self.assertNotIn("12,076", generated.answer_text)
        self.assertNotIn("unsupported_periodic_claim_removed", " ".join(generated.warnings))


if __name__ == "__main__":
    unittest.main()
