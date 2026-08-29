import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA_TOOL = ROOT / "qa-tool"
if str(QA_TOOL) not in sys.path:
    sys.path.insert(0, str(QA_TOOL))

from draft import build_draft, detect_doc_group, load_universe
UNIVERSE = ROOT / "data" / "corpus" / "universe.csv"


def _load():
    return load_universe(UNIVERSE)


class QaToolDraftTests(unittest.TestCase):
    def test_treasury_trust_termination_routes_to_major(self) -> None:
        query = (
            "최근 lg전자 자기주식취득신탁계약해지결정에서 "
            "자기주식 소각 예정에 관한 내용이 있어?"
        )
        self.assertEqual(detect_doc_group(query), "major")
        draft = build_draft(query, universe=_load())
        assert draft is not None
        self.assertEqual(draft.doc_group, "major")
        self.assertIn("treasury_share_trust_termination", draft.task_type)
        self.assertFalse(draft.in_universe)

    def test_supply_contract_termination_stays_exchange(self) -> None:
        query = "한미약품 제넥신 코로나 백신 계약 해지금액"
        self.assertEqual(detect_doc_group(query), "exchange")
        draft = build_draft(query, universe=_load())
        assert draft is not None
        self.assertEqual(draft.doc_group, "exchange")

    def test_universe_company_name_resolution(self) -> None:
        draft = build_draft(
            "현대자동차 2025년 1분기 연결 매출액은?",
            universe=_load(),
        )
        assert draft is not None
        self.assertEqual(draft.corp_name, "현대자동차")
        self.assertEqual(draft.listed_name, "현대차")
        self.assertTrue(draft.in_universe)

    def test_revenue_breakdown_hint(self) -> None:
        draft = build_draft(
            "2025 현대자동차 사업보고서에서 매출액의 구성",
            universe=_load(),
        )
        assert draft is not None
        self.assertIn("breakdown", draft.task_type)
        self.assertIn("수익의 구분", draft.must_include[0])


if __name__ == "__main__":
    unittest.main()
