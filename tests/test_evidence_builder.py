import copy
import unittest
from types import SimpleNamespace

from app.reasoning.evidence_builder import EvidenceBuilder, build_evidence_set
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


def _candidate(
    chunk_id: str,
    doc_id: str,
    *,
    rank: int,
    doc_group: str,
    content: str,
    section: str = "사업의 내용",
    **metadata,
) -> tuple[CandidateChunk, RetrievalResult]:
    chunk = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "corp_code": "00123456",
        "corp_name": "테스트회사",
        "doc_group": doc_group,
        "chunk_type": metadata.pop("chunk_type", "text"),
        "section_path": [section],
        "content": content,
        "retrieval_text": content,
        "report_nm": metadata.pop("report_nm", None),
        **metadata,
    }
    candidate = CandidateChunk(chunk_id, doc_id, chunk, MetadataMatch())
    result = RetrievalResult(
        chunk_id,
        doc_id,
        1.0 / rank,
        rank,
        {"hybrid": {"final_score": 1.0 - rank / 100.0}},
    )
    return candidate, result


def _holding_pair(
    chunk_id: str,
    doc_id: str,
    *,
    rank: int,
    date: str,
    projection_type: str,
    table_id: str,
    source_table_ids: list[str] | None = None,
) -> tuple[CandidateChunk, RetrievalResult]:
    return _candidate(
        chunk_id,
        doc_id,
        rank=rank,
        doc_group="holding",
        content=f"국민연금기금 {date} 보유주식수 1,000",
        section="보유주식등의 수 및 보유비율",
        chunk_type="table" if projection_type == "original" else "table_projection",
        table_id=table_id,
        projection_type=None if projection_type == "original" else projection_type,
        source_table_ids=source_table_ids or [],
        source_refs=[{"table_id": table_id, "row_start": rank, "row_end": rank}],
        projection_fields={
            "보고자/보유자": "국민연금기금",
            "기준일/보고일": date,
            "보유주식수": "1,000",
            "보유비율": "7.0",
        },
    )


class EvidenceBuilderTests(unittest.TestCase):
    def test_same_doc_different_chunks_keep_unique_identity(self):
        first = _candidate(
            "d1:ch_alpha",
            "d1",
            rank=1,
            doc_group="major",
            content="서로 관련 없는 알파 사실",
            section="알파",
        )
        second = _candidate(
            "d1:ch_gold",
            "d1",
            rank=2,
            doc_group="major",
            content="서로 관련 없는 골드 사실",
            section="골드",
        )

        evidence = build_evidence_set(
            question="권리공모 결과",
            query_plan=QueryPlan(query="권리공모 결과", task_type="corporate_event"),
            candidates=[first[0], second[0]],
            results=[first[1], second[1]],
        )

        items = [item for group in evidence.evidence_groups for item in group.items]
        self.assertEqual(
            [(item.chunk_id, item.doc_id) for item in items],
            [("d1:ch_alpha", "d1"), ("d1:ch_gold", "d1")],
        )
        self.assertEqual(len(evidence.evidence_groups), 2)

    def test_holding_original_detail_and_report_group_as_same_event(self):
        original = _holding_pair(
            "h1:ch_table",
            "h1",
            rank=1,
            date="2024년 03월 07일",
            projection_type="original",
            table_id="t12",
        )
        detail = _holding_pair(
            "h1:ch_detail",
            "h1",
            rank=2,
            date="2024-03-07",
            projection_type="holding_detail_row",
            table_id="t19",
            source_table_ids=["t12"],
        )
        report = _holding_pair(
            "h1:ch_report",
            "h1",
            rank=3,
            date="2024.03.07",
            projection_type="holding_report",
            table_id="t20",
        )
        candidates, results = zip(original, detail, report)

        evidence = build_evidence_set(
            question="국민연금 보유 변동",
            query_plan=QueryPlan(
                query="국민연금 보유 변동",
                task_type="holding_change",
                period=QueryPeriod(period_type="latest_holding"),
            ),
            candidates=candidates,
            results=results,
        )

        self.assertEqual(len(evidence.evidence_groups), 1)
        group = evidence.evidence_groups[0]
        self.assertEqual(group.group_type, "holding_event")
        self.assertEqual(
            group.member_chunk_ids,
            ("h1:ch_table", "h1:ch_detail", "h1:ch_report"),
        )
        self.assertEqual(group.primary_evidence.chunk_id, "h1:ch_table")

    def test_holding_same_reporter_different_dates_stay_separate(self):
        bridge = _candidate(
            "h1:ch_original",
            "h1",
            rank=1,
            doc_group="holding",
            content="두 변동 행을 포함한 원본 표",
            section="보유주식등의 수 및 보유비율",
            chunk_type="table",
            table_id="t1",
            source_refs=[],
        )
        first = _holding_pair(
            "h1:ch_2023",
            "h1",
            rank=2,
            date="2023-06-30",
            projection_type="holding_report",
            table_id="t1",
        )
        second = _holding_pair(
            "h1:ch_2024",
            "h1",
            rank=3,
            date="2024-06-30",
            projection_type="holding_report",
            table_id="t2",
            source_table_ids=["t1"],
        )

        evidence = build_evidence_set(
            question="국민연금 보유 변동",
            query_plan=QueryPlan(
                query="국민연금 보유 변동",
                task_type="holding_change",
                period=QueryPeriod(period_type="latest_holding"),
            ),
            candidates=[bridge[0], first[0], second[0]],
            results=[bridge[1], first[1], second[1]],
        )

        self.assertEqual(len(evidence.evidence_groups), 2)
        grouped_ids = [set(group.member_chunk_ids) for group in evidence.evidence_groups]
        self.assertIn({"h1:ch_original", "h1:ch_2023"}, grouped_ids)
        self.assertIn({"h1:ch_2024"}, grouped_ids)
        self.assertFalse(
            any(
                {"h1:ch_2023", "h1:ch_2024"}.issubset(ids)
                for ids in grouped_ids
            )
        )

    def test_periodic_repeated_fact_retains_every_period_without_latest_selection(self):
        older = _candidate(
            "p2023:ch_business",
            "p2023",
            rank=1,
            doc_group="periodic",
            content="HUBO는 이족보행 휴머노이드 로봇 사업입니다.",
            fiscal_year=2023,
            period_type="fiscal_year",
        )
        newer = _candidate(
            "p2024:ch_business",
            "p2024",
            rank=2,
            doc_group="periodic",
            content="HUBO는 이족보행 휴머노이드 로봇 사업입니다.",
            fiscal_year=2024,
            period_type="fiscal_year",
        )

        evidence = build_evidence_set(
            question="HUBO 사업 설명",
            query_plan=QueryPlan(
                query="HUBO 사업 설명",
                task_type="business_product",
                disclosure_route=("periodic",),
            ),
            candidates=[older[0], newer[0]],
            results=[older[1], newer[1]],
        )

        self.assertEqual(len(evidence.evidence_groups), 1)
        group = evidence.evidence_groups[0]
        self.assertEqual(group.group_type, "periodic_repeated_fact")
        self.assertEqual(group.primary_evidence.chunk_id, "p2023:ch_business")
        self.assertEqual(len(group.items), 2)
        self.assertEqual(
            evidence.retrieval_order,
            ("p2023:ch_business", "p2024:ch_business"),
        )
        self.assertTrue(evidence.ambiguity["temporal_ambiguity"])
        self.assertEqual(evidence.ambiguity["matching_event_count"], 2)

    def test_explicit_period_is_marked_without_filtering_or_reordering(self):
        older = _candidate(
            "p2023:ch_business",
            "p2023",
            rank=1,
            doc_group="periodic",
            content="동일한 주요 제품 및 서비스의 구체적인 사업 설명입니다.",
            fiscal_year=2023,
            period_type="fiscal_year",
        )
        newer = _candidate(
            "p2024:ch_business",
            "p2024",
            rank=2,
            doc_group="periodic",
            content="동일한 주요 제품 및 서비스의 구체적인 사업 설명입니다.",
            fiscal_year=2024,
            period_type="fiscal_year",
        )

        evidence = build_evidence_set(
            question="2024년 주요 제품",
            query_plan=QueryPlan(
                query="주요 제품",
                raw_query="2024년 주요 제품",
                task_type="business_product",
                period=QueryPeriod(year=2024, period_type="fiscal_year"),
                disclosure_route=("periodic",),
            ),
            candidates=[older[0], newer[0]],
            results=[older[1], newer[1]],
        )

        group = evidence.evidence_groups[0]
        matches = {item.chunk_id: item.temporal_match for item in group.items}
        self.assertEqual(
            evidence.ambiguity["temporal_constraint"]["year"], 2024
        )
        self.assertEqual(matches, {"p2023:ch_business": False, "p2024:ch_business": True})
        self.assertEqual(group.primary_evidence.chunk_id, "p2023:ch_business")
        self.assertEqual(evidence.selected_evidence_count, 2)
        self.assertFalse(evidence.ambiguity["temporal_ambiguity"])
        self.assertEqual(evidence.ambiguity["matching_event_count"], 1)

    def test_holding_without_explicit_date_marks_temporal_ambiguity(self):
        first = _holding_pair(
            "h2023:ch_report",
            "h2023",
            rank=1,
            date="2023-06-30",
            projection_type="holding_report",
            table_id="t1",
        )
        second = _holding_pair(
            "h2024:ch_report",
            "h2024",
            rank=2,
            date="2024-06-30",
            projection_type="holding_report",
            table_id="t1",
        )

        evidence = build_evidence_set(
            question="국민연금 보유 비율",
            query_plan=QueryPlan(
                query="국민연금 보유 비율",
                task_type="holding_change",
                period=QueryPeriod(period_type="latest_holding"),
            ),
            candidates=[first[0], second[0]],
            results=[first[1], second[1]],
        )

        self.assertTrue(evidence.ambiguity["temporal_ambiguity"])
        self.assertEqual(evidence.ambiguity["matching_event_count"], 2)
        self.assertIn("multiple_temporal_alternatives", evidence.warnings)

    def test_grouping_preserves_chunk_and_source_provenance(self):
        original = _holding_pair(
            "h1:ch_table",
            "h1",
            rank=1,
            date="2024-03-07",
            projection_type="original",
            table_id="t12",
        )
        projection = _holding_pair(
            "h1:ch_projection",
            "h1",
            rank=2,
            date="2024-03-07",
            projection_type="holding_detail_row",
            table_id="t19",
            source_table_ids=["t12"],
        )

        evidence = build_evidence_set(
            question="국민연금 변동",
            query_plan=QueryPlan(query="국민연금 변동", task_type="holding_change"),
            candidates=[original[0], projection[0]],
            results=[original[1], projection[1]],
        )

        group = evidence.evidence_groups[0]
        serialized = group.to_dict()
        self.assertEqual(
            serialized["member_chunk_ids"], ["h1:ch_table", "h1:ch_projection"]
        )
        for item in group.items:
            self.assertEqual(item.provenance["source_chunk_id"], item.chunk_id)
            self.assertEqual(item.provenance["source_doc_id"], item.doc_id)
            self.assertEqual(item.source_refs[0]["table_id"], item.provenance["table_id"])
            self.assertEqual(
                item.provenance["source_chunk"]["chunk_id"], item.chunk_id
            )

    def test_builder_does_not_mutate_retrieval_execution(self):
        candidate, result = _candidate(
            "d1:ch_1",
            "d1",
            rank=1,
            doc_group="major",
            content="합병 관련 원문 증거",
            source_refs=[{"table_id": "t1", "row_start": 1, "row_end": 1}],
        )
        plan = QueryPlan(query="합병 관련 원문", task_type="corporate_event")
        execution = SimpleNamespace(plan=plan, chunks=[candidate], results=[result])
        before_chunk = copy.deepcopy(candidate.chunk)
        before_result = copy.deepcopy(result)

        evidence = EvidenceBuilder().build(execution)

        self.assertEqual(candidate.chunk, before_chunk)
        self.assertEqual(result, before_result)
        self.assertEqual(evidence.raw_candidate_count, 1)
        self.assertEqual(evidence.selected_evidence_count, 1)


if __name__ == "__main__":
    unittest.main()
