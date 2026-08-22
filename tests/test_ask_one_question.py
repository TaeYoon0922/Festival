import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.ask_one_question import (
    DEFAULT_QUESTION,
    _print_summary,
    _safe_error_body,
    main,
)


class AskOneQuestionTests(unittest.TestCase):
    def test_error_body_omits_likely_secrets(self) -> None:
        self.assertIn("database_unavailable", _safe_error_body('{"reason": "database_unavailable"}'))
        self.assertEqual(
            _safe_error_body("postgresql://user:password@host/db"),
            "(error body omitted because it may contain a secret)",
        )

    def test_summary_prints_trace_before_answer(self) -> None:
        payload = {
            "think_trace": {
                "task_type": "corporate_event",
                "route": "general_evidence",
                "answerable": True,
                "warnings": [],
                "hcx_status": "disabled",
                "retrieval_count": 1,
                "selected_evidence_count": 1,
                "stages": ["task_router"],
            },
            "retrieved_context": [
                {
                    "rank": 1,
                    "corp_name": "고려아연",
                    "report_nm": "신규시설투자등",
                    "rcept_dt": "2024-01-02",
                    "chunk_type": "table",
                    "chunk_id": "c1",
                }
            ],
            "answer": "공시된 사실만 표시합니다.",
        }
        buffer = io.StringIO()
        with patch("sys.stdout", buffer):
            _print_summary(payload, Path("out.json"))
        text = buffer.getvalue()
        self.assertLess(text.index("=== think_trace ==="), text.index("=== answer ==="))
        self.assertIn("corporate_event", text)
        self.assertIn("신규시설투자등", text)
        self.assertIn("공시된 사실만 표시합니다.", text)

    def test_healthz_failure_does_not_call_answer(self) -> None:
        with TemporaryDirectory() as tmp:
            code = main(
                [
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "1",
                    "--output-dir",
                    tmp,
                    "--question",
                    DEFAULT_QUESTION,
                ]
            )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
