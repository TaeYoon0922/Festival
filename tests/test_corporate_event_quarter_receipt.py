"""Corporate-event calendar quarters must open a receipt window, not base_month."""

from __future__ import annotations

import unittest
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.reasoning.query_understanding import QueryUnderstanding
from app.reasoning.router import QueryRouter
from app.retrieval.interfaces import CandidateDocument, MetadataMatch
from tests.test_evidence_builder import _candidate


PURPOSE_TEXT = (
    "본 자기주식 처분 결정은 임원의 책임경영 강화 및 성과 창출을 위한 "
    "동기부여 제공 목적에 따라 자기주식 일부를 지급하기 위한 것임"
)


def _major_doc(
    doc_id: str,
    *,
    report_nm: str,
    rcept_dt: str,
    base_month: int | None = None,
) -> CandidateDocument:
    return CandidateDocument(
        doc_id=doc_id,
        metadata={
            "doc_id": doc_id,
            "corp_name": "기아",
            "doc_group": "major",
            "report_nm": report_nm,
            "rcept_dt": rcept_dt,
            "base_year": None,
            "base_month": base_month,
        },
        metadata_match=MetadataMatch(),
    )


class CorporateEventQuarterReceiptTests(unittest.TestCase):
    def test_kia_q1_disposal_keeps_only_receipt_window_docs(self) -> None:
        aliases = {"기아": {"기아"}}
        question = "올해 기아의 1분기 자기주식 처분의 목적이 뭐야?"
        with patch("app.reasoning.query_understanding.date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 24)
            plan = QueryUnderstanding(aliases).understand(question)
        route = QueryRouter().route(plan)

        kept = QueryRouter().filter_documents(
            [
                _major_doc(
                    "major_20251031000189",
                    report_nm="주요사항보고서(자기주식처분결정)",
                    rcept_dt="20251031",
                ),
                _major_doc(
                    "major_20260320001182",
                    report_nm="주요사항보고서(자기주식처분결정)",
                    rcept_dt="20260320",
                ),
                _major_doc(
                    "major_20250314001155",
                    report_nm="주요사항보고서(자기주식취득결정)",
                    rcept_dt="20250314",
                ),
                _major_doc(
                    "major_20230127000327",
                    report_nm="주요사항보고서(자기주식취득결정)",
                    rcept_dt="20230127",
                ),
            ],
            route,
        )

        self.assertEqual(plan.period.period_type, "receipt_date")
        self.assertIsNone(plan.backend_filters()["period"])
        self.assertEqual(
            [document.doc_id for document in kept],
            ["major_20260320001182"],
        )

    def test_retrieved_purpose_chunk_answers_without_확인필요(self) -> None:
        question = "올해 기아의 1분기 자기주식 처분의 목적이 뭐야?"
        with patch("app.reasoning.query_understanding.date") as mock_date:
            mock_date.today.return_value = date(2026, 8, 24)
            plan = QueryUnderstanding({"기아": {"기아"}}).understand(question)

        pair = _candidate(
            "major_20260320001182:ch_purpose",
            "major_20260320001182",
            rank=1,
            doc_group="major",
            content=PURPOSE_TEXT,
            section="기타 투자판단에 참고할 사항",
            corp_name="기아",
            report_nm="주요사항보고서(자기주식처분결정)",
            rcept_dt="20260320",
            source_refs=[{"section": "기타 투자판단에 참고할 사항"}],
        )
        execution = SimpleNamespace(
            plan=plan,
            chunks=[pair[0]],
            results=[pair[1]],
        )

        agent = AgentOrchestrator().run(question, plan, execution)
        generated = CitationAwareAnswerGenerator().generate(agent.answer_draft)

        self.assertEqual(agent.task_decision.task_type, "corporate_event")
        self.assertTrue(agent.answer_draft.answerable)
        self.assertTrue(generated.answerable)
        self.assertIn(PURPOSE_TEXT, generated.answer_text)
        self.assertIn("책임경영", generated.answer_text)
        self.assertIn("동기부여", generated.answer_text)
        self.assertNotIn("확인되지 않은 정보가 있습니다", generated.answer_text)
        self.assertGreaterEqual(generated.answer_text.count("[1]"), 1)


if __name__ == "__main__":
    unittest.main()
