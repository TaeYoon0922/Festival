from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.generation.compact_claim import build_compact_claim
from app.generation.hcx_verbalizer import HcxSettings
from app.generation.protected_literals import PLACEHOLDER_PATTERN
from scripts.experiment_hcx_holding_live import (
    TARGET_QUESTION_IDS,
    prepare_actual_question,
    run_prepared_question,
    summarize_live_runs,
    target_question_rows,
)
from tests.test_compact_claim import _holding_case


def _settings() -> HcxSettings:
    return HcxSettings(
        enabled=True,
        endpoint="https://clova.example/v1/chat/completions",
        api_key="secret-test-key",
    )


class _Transport:
    def __init__(self, transform=lambda value: value) -> None:
        self.transform = transform
        self.calls = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        raw = self.transform(payload["messages"][1]["content"])
        return {
            "choices": [
                {"message": {"content": raw}, "finish_reason": "stop"}
            ],
            "usage": {"completion_tokens": 123},
        }


class _StaticPipeline:
    def __init__(self, result) -> None:
        self.settings = SimpleNamespace(top_k=10)
        self.understanding = SimpleNamespace(understand=self._understand)
        self.executor = SimpleNamespace(execute=self._execute)
        self.orchestrator = SimpleNamespace(run=self._run)
        self.result = result
        self.calls = []

    def _understand(self, question, *, top_k):
        self.calls.append(("understand", question, top_k))
        return SimpleNamespace(raw_query=question)

    def _execute(self, plan):
        self.calls.append(("execute", plan.raw_query))
        return SimpleNamespace()

    def _run(self, question, plan, execution):
        self.calls.append(("orchestrate", question))
        return self.result


def _prepared(events: int = 4):
    result, _, production_claim = _holding_case(events=events)
    pipeline = _StaticPipeline(result)
    row = {"question_id": "HX07", "query": result.answer_draft.question}
    prepared = prepare_actual_question(pipeline, row)
    return prepared, production_claim, pipeline


class ActualHoldingPreparationTests(unittest.TestCase):
    def test_target_list_is_the_requested_eleven_frozen_questions(self) -> None:
        rows = target_question_rows()

        self.assertEqual(len(rows), 11)
        self.assertEqual(
            tuple(str(row["question_id"]) for row in rows),
            TARGET_QUESTION_IDS,
        )

    def test_all_production_selected_events_are_kept_without_caps(self) -> None:
        prepared, production_claim, pipeline = _prepared(events=4)

        self.assertIsNone(production_claim)
        self.assertEqual(prepared.candidate_event_count, 4)
        self.assertEqual(
            len(prepared.claim.fields),
            4 * len(prepared.requested_fields),
        )
        self.assertEqual(len(prepared.detached.attachments), 4)
        self.assertEqual(
            [call[0] for call in pipeline.calls],
            ["understand", "execute", "orchestrate"],
        )
        self.assertIsNone(
            build_compact_claim(
                prepared.claim,
                None,
                task_type="holding_event",
            )
        )

    def test_real_structure_diagnostics_report_event_citations_and_duplicates(self) -> None:
        prepared, _, _ = _prepared(events=4)

        self.assertEqual(len(prepared.structure["event_citation_sets"]), 4)
        self.assertEqual(
            prepared.structure["requested_fields_per_event"],
            [list(prepared.requested_fields)] * 4,
        )
        self.assertEqual(prepared.structure["multi_source_event_count"], 0)
        self.assertEqual(prepared.structure["duplicate_event_count"], 0)


class ActualHoldingLiveRunTests(unittest.TestCase):
    def test_clean_echo_passes_every_required_check(self) -> None:
        prepared, _, _ = _prepared(events=4)
        transport = _Transport()

        record = run_prepared_question(
            prepared,
            transport=transport,
            settings=_settings(),
            repeat_index=1,
        )

        self.assertTrue(record["candidate_received"])
        self.assertEqual(record["finish_reason"], "stop")
        self.assertEqual(record["completion_tokens"], 123)
        self.assertTrue(record["field_placeholder_integrity_valid"])
        self.assertTrue(record["all_events_kept"])
        self.assertEqual(
            record["unprotected_numeric_tokens_in_masked_candidate"], []
        )
        self.assertTrue(record["candidate_valid_before_citation_attachment"])
        self.assertTrue(record["deterministic_citations_attached"])
        self.assertTrue(record["final_answer_valid"])
        self.assertTrue(record["hcx_success"])
        self.assertIsNone(record["hcx_failure_reason"])
        self.assertNotIn("raw_hcx_candidate", record)
        self.assertNotIn("secret-test-key", str(record))

    def test_show_output_is_the_only_mode_that_exposes_candidate_text(self) -> None:
        prepared, _, _ = _prepared(events=4)

        record = run_prepared_question(
            prepared,
            transport=_Transport(),
            settings=_settings(),
            repeat_index=1,
            show_output=True,
        )

        self.assertIn("raw_hcx_candidate", record)
        self.assertIn("restored_candidate_text", record)
        self.assertIn("final_output", record)

    def test_duplicate_placeholder_fails_before_attachment(self) -> None:
        prepared, _, _ = _prepared(events=4)

        def duplicate(masked: str) -> str:
            return masked + PLACEHOLDER_PATTERN.findall(masked)[0]

        record = run_prepared_question(
            prepared,
            transport=_Transport(duplicate),
            settings=_settings(),
            repeat_index=1,
        )

        self.assertFalse(record["field_placeholder_integrity_valid"])
        self.assertEqual(
            record["field_placeholder_integrity_reason"],
            "placeholder_duplicated",
        )
        self.assertFalse(record["deterministic_citations_attached"])
        self.assertEqual(record["hcx_failure_reason"], "placeholder_duplicated")

    def test_new_unprotected_number_fails_closed(self) -> None:
        prepared, _, _ = _prepared(events=4)
        record = run_prepared_question(
            prepared,
            transport=_Transport(lambda masked: masked + " 99,999"),
            settings=_settings(),
            repeat_index=1,
        )

        self.assertFalse(record["hcx_success"])
        self.assertEqual(
            record["unprotected_numeric_tokens_in_masked_candidate"],
            ["99,999"],
        )
        self.assertEqual(
            record["hcx_failure_reason"],
            "unprotected_numeric_generation",
        )

    def test_summary_counts_questions_repeats_sizes_and_failures(self) -> None:
        prepared, _, _ = _prepared(events=4)
        clean = run_prepared_question(
            prepared,
            transport=_Transport(),
            settings=_settings(),
            repeat_index=1,
        )
        failed = dict(
            clean,
            repeat_index=2,
            hcx_success=False,
            hcx_failure_reason="placeholder_missing",
        )

        summary = summarize_live_runs([clean, failed], repeat=2)

        self.assertEqual(summary["question_count"], 1)
        self.assertEqual(summary["hcx_success_count"], 0)
        self.assertEqual(summary["hcx_successful_run_count"], 1)
        self.assertEqual(summary["failure_reasons"], {"placeholder_missing": 1})
        self.assertEqual(
            summary["by_candidate_event_count"]["4"]["runs"],
            2,
        )
        self.assertEqual(summary["max_successful_event_count"], 4)
        self.assertFalse(summary["all_questions_clean"])


if __name__ == "__main__":
    unittest.main()
