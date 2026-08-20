import copy
import unittest

from app.reasoning.periodic_fact_resolver import resolve_periodic_facts
from scripts.smoke_periodic_fact_resolver import (
    _concise_summary,
    _grouping_snapshot,
    _query_report,
)
from tests.test_periodic_fact_resolver import _evidence, _group, _item


def _retrieval_rows(*items):
    return [
        {
            "chunk_id": item.chunk_id,
            "doc_id": item.doc_id,
            "rank": item.retrieval_rank,
            "bm25_score": item.retrieval_score,
            "final_score": item.retrieval_score,
            "retrieval_score": item.retrieval_score,
            "rrf_score": None,
        }
        for item in items
    ]


class PeriodicResolverSmokeTests(unittest.TestCase):
    def test_repeated_fact_report_preserves_sources_and_invariants(self):
        older = _item(
            "p23:ch_hubo",
            "p23",
            rank=1,
            text="HUBO 이족보행 로봇 사업을 설명합니다.",
            year=2023,
        )
        newer = _item(
            "p24:ch_hubo",
            "p24",
            rank=2,
            text="HUBO 이족보행 로봇 사업을 설명합니다.",
            year=2024,
        )
        evidence = _evidence(
            [
                _group(
                    "g-hubo",
                    older,
                    newer,
                    group_type="periodic_repeated_fact",
                )
            ],
            question="테스트회사 HUBO 이족보행 로봇 사업 설명",
        )
        resolution = resolve_periodic_facts(evidence)
        retrieval = _retrieval_rows(older, newer)

        report = _query_report(
            query=evidence.question,
            plan=evidence.query_plan,
            evidence=evidence,
            resolution=resolution,
            retrieval_before=retrieval,
            retrieval_after=copy.deepcopy(retrieval),
            evidence_before=copy.deepcopy(evidence.to_dict()),
            grouping_before=_grouping_snapshot(evidence),
        )

        self.assertTrue(report["validation"]["all_invariants_preserved"])
        self.assertTrue(report["answerability"]["answerable_evidence"])
        self.assertEqual(report["periodic_fact_resolution"]["fact_count"], 1)
        self.assertTrue(report["facts"][0]["repeated_across_periods"])
        self.assertEqual(
            [source["chunk_id"] for source in report["facts"][0]["sources"]],
            ["p23:ch_hubo", "p24:ch_hubo"],
        )
        self.assertTrue(report["provenance_validation"]["all_paths_preserved"])

    def test_same_doc_different_chunks_keep_unique_identity(self):
        first = _item(
            "p1:ch_dx", "p1", rank=1, text="DX 부문 제품 근거", year=2024
        )
        second = _item(
            "p1:ch_other", "p1", rank=2, text="다른 제품 근거", year=2024
        )
        evidence = _evidence(
            [_group("g-dx", first), _group("g-other", second)],
            question="테스트회사 DX 부문의 주요 제품은 무엇인가",
        )
        resolution = resolve_periodic_facts(evidence)
        retrieval = _retrieval_rows(first, second)

        report = _query_report(
            query=evidence.question,
            plan=evidence.query_plan,
            evidence=evidence,
            resolution=resolution,
            retrieval_before=retrieval,
            retrieval_after=copy.deepcopy(retrieval),
            evidence_before=copy.deepcopy(evidence.to_dict()),
            grouping_before=_grouping_snapshot(evidence),
        )

        self.assertTrue(
            report["provenance_validation"]["unique_chunk_identity_preserved"]
        )
        self.assertEqual(
            {path["chunk_id"] for path in report["provenance_validation"]["paths"]},
            {"p1:ch_dx", "p1:ch_other"},
        )

    def test_concise_summary_omits_full_fact_and_source_payloads(self):
        report = {
            "query_count": 1,
            "successful_query_count": 1,
            "failed_query_count": 0,
            "all_invariants_preserved": True,
            "queries": [
                {
                    "query": "질의",
                    "status": "ok",
                    "periodic_fact_resolution": {
                        "fact_count": 1,
                        "matching_fact_count": 1,
                    },
                    "facts": [
                        {
                            "repeated_across_periods": True,
                            "fact_conflict": False,
                            "period_evolution": False,
                            "sources": [{"large": "payload"}],
                        }
                    ],
                    "answerability": {"answerable_evidence": True},
                    "provenance_validation": {"all_paths_preserved": True},
                    "validation": {"all_invariants_preserved": True},
                }
            ],
        }

        summary = _concise_summary(report)

        self.assertNotIn("facts", summary["queries"][0])
        self.assertNotIn("sources", summary["queries"][0])
        self.assertEqual(summary["queries"][0]["repeated_fact_count"], 1)


if __name__ == "__main__":
    unittest.main()
