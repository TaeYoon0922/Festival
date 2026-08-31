import unittest

from app.reasoning.periodic_segment_ranking import project_segment_ranking_table


class PeriodicSegmentRankingTests(unittest.TestCase):
    def test_picks_largest_segment_row(self) -> None:
        table = """
| 부문 | 영업이익 |
| --- | --- |
| 광학 | 100 |
| 기판 | 250 |
| 합계 | 350 |
"""
        result = project_segment_ranking_table(table, metric="영업이익")
        assert result is not None
        self.assertIn("기판", result)
        self.assertIn("최대", result)
        self.assertNotIn("| 합계 |", result.split("(")[0])


if __name__ == "__main__":
    unittest.main()
