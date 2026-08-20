import copy
import unittest
from dataclasses import FrozenInstanceError

from app.reasoning.evidence_builder import (
    EvidenceGroup,
    EvidenceItem,
    EvidenceSet,
    build_evidence_set,
)
from app.reasoning.periodic_fact_resolver import (
    PeriodicFactResolver,
    resolve_periodic_facts,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from app.retrieval.interfaces import CandidateChunk, MetadataMatch, RetrievalResult


def _item(
    chunk_id: str,
    doc_id: str,
    *,
    rank: int,
    text: str,
    year: int | None = None,
    quarter: int | None = None,
    section: str = "주요 제품 및 서비스",
    doc_group: str = "periodic",
    temporal_match: bool | None = None,
    table_id: str | None = None,
    holding: dict | None = None,
) -> EvidenceItem:
    period = {}
    if year is not None:
        period["fiscal_year"] = year
        period["period_type"] = "fiscal_year"
    if quarter is not None:
        period["quarter"] = quarter
        period["period_type"] = "fiscal_quarter"
    ref = {
        "table_id": table_id or f"table_{rank}",
        "row_start": rank,
        "row_end": rank,
    }
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        company_id="00126380",
        corp_code="00126380",
        corp_name="테스트회사",
        doc_group=doc_group,
        chunk_type="text",
        section_path=(section,),
        evidence_text=f"검색 헤더\n{text}",
        retrieval_rank=rank,
        retrieval_score=1.0 - rank / 100,
        rcept_dt=f"{year or 2024}-03-31",
        report_nm=f"{year or 2024}년 사업보고서",
        period=period,
        source_refs=(ref,),
        provenance={
            "source_chunk_id": chunk_id,
            "source_doc_id": doc_id,
            "source_refs": [ref],
            "source_chunk": {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": text,
                "source_refs": [ref],
            },
        },
        holding=holding or {},
        temporal_match=temporal_match,
    )


def _group(
    group_id: str,
    *items: EvidenceItem,
    group_type: str = "standalone_evidence",
) -> EvidenceGroup:
    ordered = tuple(sorted(items, key=lambda item: (item.retrieval_rank, item.chunk_id)))
    return EvidenceGroup(
        group_id=group_id,
        group_type=group_type,
        member_chunk_ids=tuple(item.chunk_id for item in ordered),
        primary_evidence=ordered[0],
        supporting_evidence=ordered[1:],
        doc_ids=tuple(dict.fromkeys(item.doc_id for item in ordered)),
        reason="periodic resolver fixture",
    )


def _evidence(
    groups: list[EvidenceGroup],
    *,
    question: str = "테스트회사 주요 제품",
    year: int | None = None,
    task_type: str = "business_product",
) -> EvidenceSet:
    items = [item for group in groups for item in group.items]
    period = {
        "year": year,
        "quarter": None,
        "from": None,
        "to": None,
        "period_type": "fiscal_year" if year is not None else None,
    }
    return EvidenceSet(
        question=question,
        query_plan={
            "raw_query": question,
            "task_type": task_type,
            "metric": None,
            "period": period,
            "evidence": {
                "periodic_intent": task_type,
                "periodic_intent_evidence": "주요 제품",
            },
        },
        task_type=task_type,
        evidence_groups=tuple(groups),
        retrieval_order=tuple(item.chunk_id for item in items),
        raw_candidate_count=len(items),
        selected_evidence_count=len(items),
        warnings=(),
        ambiguity={
            "temporal_ambiguity": False,
            "temporal_constraint": {
                "explicit": year is not None,
                "year": year,
                "quarter": None,
                "from_date": None,
                "to_date": None,
                "period_type": "fiscal_year" if year is not None else None,
            },
        },
    )


class PeriodicFactResolverTests(unittest.TestCase):
    def test_evidence_builder_repeated_group_integrates_with_resolver(self):
        candidates = []
        results = []
        for rank, year in enumerate((2023, 2024), start=1):
            chunk_id = f"p{year}:ch_fact"
            doc_id = f"p{year}"
            chunk = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "company_id": "00126380",
                "corp_code": "00126380",
                "corp_name": "테스트회사",
                "doc_group": "periodic",
                "chunk_type": "text",
                "section_path": ["주요 제품 및 서비스"],
                "content": "동일한 주요 제품 및 서비스의 구체적인 사업 설명입니다.",
                "retrieval_text": "동일한 주요 제품 및 서비스의 구체적인 사업 설명입니다.",
                "fiscal_year": year,
                "period_type": "fiscal_year",
                "report_nm": f"{year}년 사업보고서",
                "source_refs": [
                    {"table_id": f"t{year}", "row_start": 1, "row_end": 1}
                ],
            }
            candidates.append(
                CandidateChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk=chunk,
                    metadata_match=MetadataMatch(),
                )
            )
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    bm25_score=10.0 - rank,
                    rank=rank,
                    metadata_match={},
                )
            )
        evidence = build_evidence_set(
            question="테스트회사 주요 제품",
            query_plan=QueryPlan(
                query="주요 제품",
                raw_query="테스트회사 주요 제품",
                task_type="business_product",
                period=QueryPeriod(),
                disclosure_route=("periodic",),
                evidence={"periodic_intent": "business_product"},
            ),
            candidates=candidates,
            results=results,
        )

        self.assertEqual(evidence.evidence_groups[0].group_type, "periodic_repeated_fact")
        fact = resolve_periodic_facts(evidence).facts[0]
        self.assertTrue(fact.repeated_across_periods)
        self.assertEqual(fact.evidence_chunk_ids, ("p2023:ch_fact", "p2024:ch_fact"))

    def test_single_periodic_fact(self):
        item = _item("p1:ch_alpha", "p1", rank=1, text="HUBO 로봇을 생산합니다.", year=2024)
        resolution = PeriodicFactResolver().resolve(_evidence([_group("g1", item)]))

        self.assertEqual(len(resolution.facts), 1)
        fact = resolution.facts[0]
        self.assertEqual(fact.fact_type, "business_product")
        self.assertEqual(fact.fact_text, "HUBO 로봇을 생산합니다.")
        self.assertEqual(fact.evidence_chunk_ids, ("p1:ch_alpha",))
        with self.assertRaises(FrozenInstanceError):
            fact.fact_text = "변경"

    def test_repeated_fact_keeps_all_period_sources_in_one_fact(self):
        older = _item("p23:ch_fact", "p23", rank=1, text="동일한 로봇 사업 설명", year=2023)
        newer = _item("p24:ch_fact", "p24", rank=2, text="동일한 로봇 사업 설명", year=2024)
        group = _group("g-repeat", older, newer, group_type="periodic_repeated_fact")

        fact = resolve_periodic_facts(_evidence([group])).facts[0]

        self.assertTrue(fact.repeated_across_periods)
        self.assertEqual(fact.supporting_evidence_count, 2)
        self.assertEqual(fact.doc_ids, ("p23", "p24"))
        self.assertEqual(len(fact.reporting_periods), 2)

    def test_periodless_question_never_selects_latest_source(self):
        older = _item("p23:ch_fact", "p23", rank=1, text="반복 사실", year=2023)
        newer = _item("p24:ch_fact", "p24", rank=2, text="반복 사실", year=2024)
        group = _group("g-repeat", older, newer, group_type="periodic_repeated_fact")

        resolution = resolve_periodic_facts(_evidence([group]))
        fact = resolution.facts[0]

        self.assertEqual(fact.evidence_chunk_ids, ("p23:ch_fact", "p24:ch_fact"))
        self.assertEqual(dict(fact.temporal_matches), {"p23:ch_fact": None, "p24:ch_fact": None})
        self.assertFalse(resolution.temporal_ambiguity)

    def test_explicit_period_marks_each_source_without_filtering(self):
        older = _item(
            "p23:ch_fact", "p23", rank=1, text="반복 사실", year=2023, temporal_match=False
        )
        newer = _item(
            "p24:ch_fact", "p24", rank=2, text="반복 사실", year=2024, temporal_match=True
        )
        group = _group("g-repeat", older, newer, group_type="periodic_repeated_fact")

        resolution = resolve_periodic_facts(_evidence([group], year=2024))
        fact = resolution.facts[0]

        self.assertEqual(dict(fact.temporal_matches), {"p23:ch_fact": False, "p24:ch_fact": True})
        self.assertEqual(fact.evidence_chunk_ids, ("p23:ch_fact", "p24:ch_fact"))
        self.assertEqual(resolution.matching_fact_count, 1)

    def test_different_facts_in_same_company_and_section_are_not_merged(self):
        first = _item("p1:ch_a", "p1", rank=1, text="로봇을 생산합니다.", year=2024)
        second = _item("p1:ch_b", "p1", rank=2, text="반도체를 생산합니다.", year=2024)

        resolution = resolve_periodic_facts(
            _evidence([_group("g-a", first), _group("g-b", second)])
        )

        self.assertEqual(len(resolution.facts), 2)
        self.assertEqual(
            [fact.evidence_chunk_ids for fact in resolution.facts],
            [("p1:ch_a",), ("p1:ch_b",)],
        )

    def test_normal_period_evolution_is_not_same_period_conflict(self):
        older = _item("p23:ch_fact", "p23", rank=1, text="제품 A를 생산", year=2023)
        newer = _item("p24:ch_fact", "p24", rank=2, text="제품 A와 B를 생산", year=2024)
        group = _group("g-evolution", older, newer, group_type="periodic_repeated_fact")

        fact = resolve_periodic_facts(_evidence([group])).facts[0]

        self.assertFalse(fact.fact_conflict)
        self.assertTrue(fact.period_evolution)
        self.assertIsNone(fact.fact_text)
        self.assertEqual(len(fact.alternatives), 2)

    def test_same_period_conflict_preserves_alternatives(self):
        first = _item("p1:ch_a", "p1", rank=1, text="제품 수는 10개", year=2024)
        second = _item("p1:ch_b", "p1", rank=2, text="제품 수는 12개", year=2024)
        group = _group("g-conflict", first, second, group_type="document_evidence")

        fact = resolve_periodic_facts(_evidence([group])).facts[0]

        self.assertTrue(fact.fact_conflict)
        self.assertEqual(fact.conflict_type, "same_period_source_conflict")
        self.assertIsNone(fact.fact_text)
        self.assertEqual(len(fact.alternatives), 2)
        self.assertIn("same_period_fact_conflict", fact.warnings)

    def test_provenance_keeps_every_chunk_and_source_row(self):
        first = _item(
            "p1:ch_a", "p1", rank=1, text="동일 사실", year=2023, table_id="t11"
        )
        second = _item(
            "p2:ch_a", "p2", rank=2, text="동일 사실", year=2024, table_id="t22"
        )
        group = _group("g-repeat", first, second, group_type="periodic_repeated_fact")

        serialized = resolve_periodic_facts(_evidence([group])).facts[0].to_dict()

        self.assertEqual(
            [source["chunk_id"] for source in serialized["fact_provenance"]],
            ["p1:ch_a", "p2:ch_a"],
        )
        refs = [source["source_refs"][0] for source in serialized["fact_provenance"]]
        self.assertEqual([ref["table_id"] for ref in refs], ["t11", "t22"])
        self.assertTrue(all(ref["row_start"] == ref["row_end"] for ref in refs))

    def test_unique_chunk_identity_survives_same_document(self):
        first = _item("p1:ch_alpha", "p1", rank=1, text="첫 번째 사실", year=2024)
        second = _item("p1:ch_beta", "p1", rank=2, text="두 번째 사실", year=2024)

        resolution = resolve_periodic_facts(
            _evidence([_group("g-alpha", first), _group("g-beta", second)])
        )

        self.assertEqual({fact.doc_ids for fact in resolution.facts}, {("p1",)})
        self.assertEqual(
            {fact.evidence_chunk_ids for fact in resolution.facts},
            {("p1:ch_alpha",), ("p1:ch_beta",)},
        )

    def test_input_evidence_set_is_not_mutated(self):
        item = _item("p1:ch_alpha", "p1", rank=1, text="원본 사실", year=2024)
        evidence = _evidence([_group("g1", item)])
        before = copy.deepcopy(evidence.to_dict())

        resolve_periodic_facts(evidence)

        self.assertEqual(evidence.to_dict(), before)

    def test_holding_evidence_is_isolated(self):
        holding = _item(
            "h1:ch_report",
            "h1",
            rank=1,
            text="국민연금 보유 변동",
            year=2024,
            doc_group="holding",
            holding={"reporter": "국민연금기금", "reference_date": "2024-03-07"},
        )
        group = _group("g-holding", holding, group_type="holding_event")

        resolution = resolve_periodic_facts(
            _evidence([group], task_type="holding_change")
        )

        self.assertEqual(resolution.facts, ())
        self.assertIn("no_periodic_fact_evidence", resolution.warnings)

    def test_retrieval_rank_score_and_order_are_unchanged(self):
        first = _item("p1:ch_a", "p1", rank=2, text="사실 A", year=2024)
        second = _item("p1:ch_b", "p1", rank=7, text="사실 B", year=2024)
        evidence = _evidence([_group("g-a", first), _group("g-b", second)])
        before_order = evidence.retrieval_order
        before_values = [
            (item.chunk_id, item.retrieval_rank, item.retrieval_score)
            for group in evidence.evidence_groups
            for item in group.items
        ]

        resolution = resolve_periodic_facts(evidence)

        after_values = [
            (source.chunk_id, source.retrieval_rank, source.retrieval_score)
            for fact in resolution.facts
            for source in fact.sources
        ]
        self.assertEqual(evidence.retrieval_order, before_order)
        self.assertEqual(after_values, before_values)


if __name__ == "__main__":
    unittest.main()
