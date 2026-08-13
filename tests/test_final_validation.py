import unittest

from app.parsing.final_validation import _exclusion_reason, _is_empty_context_wrapper
from app.parsing.models import Table, TableCell


class EmptyContextWrapperTests(unittest.TestCase):
    def test_empty_unit_wrapper_has_no_evidence(self) -> None:
        self.assertTrue(_is_empty_context_wrapper("(단위 : )"))
        table = Table(
            table_id="t0001",
            section_id="s0001",
            rows=[[TableCell("(단위 : )")]],
        )
        self.assertEqual(
            _exclusion_reason(table, has_candidate=False),
            ("empty_context_wrapper", "safe"),
        )

    def test_wrapper_with_a_real_unit_is_not_empty(self) -> None:
        self.assertFalse(_is_empty_context_wrapper("(단위 : 백만원)"))

    def test_real_evidence_is_not_empty(self) -> None:
        self.assertFalse(_is_empty_context_wrapper("보유주식수 | 1,234"))


if __name__ == "__main__":
    unittest.main()
