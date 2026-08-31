from __future__ import annotations

import unittest

from app.reasoning.holding_event_aggregate import aggregate_holding_change_statement


class HoldingEventAggregateTests(unittest.TestCase):
    def test_net_change_and_rate(self) -> None:
        events = [
            {
                "before_shares": {"raw": "1,000", "normalized": 1000},
                "change_shares": {"raw": "200", "normalized": 200},
            },
            {
                "change_shares": {"raw": "300", "normalized": 300},
            },
        ]
        statement = aggregate_holding_change_statement(events)
        self.assertIsNotNone(statement)
        assert statement is not None
        self.assertIn("500주", statement)
        self.assertIn("50%", statement)


if __name__ == "__main__":
    unittest.main()
