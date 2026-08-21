from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest import mock

from app.generation.compact_claim import (
    MAX_CLAIM_EVENTS,
    build_compact_claim,
)
from scripts.experiment_hcx_holding_live import (
    HOLDING20_QUESTION_IDS,
    main as live_main,
    prepare_capped_question,
    run_prepared_question,
    summarize_capped_runs,
    target_question_rows,
)
from tests.test_compact_claim import _holding_case
from tests.test_experiment_hcx_holding_live import (
    _StaticPipeline,
    _structured_prepared,
    _Transport,
    _settings,
)


def _row(question_id: str, question: str) -> dict:
    return {"question_id": question_id, "query": question}


def _capped(events: int = 1, question_id: str = "HX01"):
    """Prepare a question through the capped path, as production would."""

    result, _, _ = _holding_case(events=events)
    pipeline = _StaticPipeline(result)
    row = _row(question_id, result.answer_draft.question)
    prepared, verdict = prepare_capped_question(pipeline, row)
    return prepared, verdict, pipeline


class EligibilityTests(unittest.TestCase):
    """Eligibility is decided from live objects, never assumed."""

    def test_a_question_within_the_caps_is_eligible(self) -> None:
        prepared, verdict, _ = _capped(events=1)

        self.assertIsNotNone(prepared)
        self.assertTrue(verdict["eligible_under_current_caps"])
        self.assertIsNone(verdict["eligibility_reason"])

    def test_eligibility_matches_the_production_builder(self) -> None:
        result, _, _ = _holding_case(events=1)
        pipeline = _StaticPipeline(result)

        prepared, verdict = prepare_capped_question(
            pipeline, _row("HX01", result.answer_draft.question)
        )
        claim = build_compact_claim(
            result.answer_draft,
            result.resolution,
            task_type=result.task_decision.task_type,
        )

        self.assertEqual(prepared is not None, claim is not None)
        self.assertTrue(verdict["audit_agrees_with_builder"])

    def test_the_prepared_claim_is_the_production_claim(self) -> None:
        result, _, _ = _holding_case(events=1)
        pipeline = _StaticPipeline(result)

        prepared, _ = prepare_capped_question(
            pipeline, _row("HX01", result.answer_draft.question)
        )
        production = build_compact_claim(
            result.answer_draft,
            result.resolution,
            task_type=result.task_decision.task_type,
        )

        self.assertEqual(
            prepared.claim.deterministic_text, production.deterministic_text
        )

    def test_a_question_past_the_event_cap_is_ineligible(self) -> None:
        prepared, verdict, pipeline = _capped(events=MAX_CLAIM_EVENTS + 1)

        self.assertIsNone(prepared)
        self.assertFalse(verdict["eligible_under_current_caps"])
        self.assertEqual(verdict["eligibility_reason"], "too_many_events")
        self.assertEqual(verdict["candidate_event_count"], MAX_CLAIM_EVENTS + 1)

    def test_the_caps_in_force_are_recorded(self) -> None:
        _, verdict, _ = _capped(events=1)

        self.assertEqual(verdict["MAX_CLAIM_EVENTS"], MAX_CLAIM_EVENTS)

    def test_a_pipeline_failure_is_reported_not_raised(self) -> None:
        class _Broken:
            settings = _StaticPipeline(None).settings

            class understanding:  # noqa: D106 - test stub
                @staticmethod
                def understand(question, *, top_k):
                    raise RuntimeError("dsn=postgresql://user:pw@host/db")

        prepared, verdict = prepare_capped_question(_Broken(), _row("HX01", "q"))

        self.assertIsNone(prepared)
        self.assertEqual(verdict["eligibility_reason"], "pipeline_error")
        self.assertEqual(verdict["exception_stage"], "pipeline")


class TargetSelectionTests(unittest.TestCase):
    def test_every_holding_question_can_be_targeted(self) -> None:
        self.assertEqual(len(HOLDING20_QUESTION_IDS), 20)
        for question_id in ("HX01", "HX03", "HX05", "HX08", "HX09"):
            self.assertIn(question_id, HOLDING20_QUESTION_IDS)

    def test_requested_ids_are_returned_in_order(self) -> None:
        rows = target_question_rows(["HX09", "HX01"])

        self.assertEqual(
            [str(row["question_id"]) for row in rows], ["HX09", "HX01"]
        )

    def test_capped_mode_defaults_to_the_whole_holding_set(self) -> None:
        rows = target_question_rows(None, default_ids=HOLDING20_QUESTION_IDS)

        self.assertEqual(len(rows), 20)

    def test_an_unknown_id_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            target_question_rows(["HX99"])


class SingleCallTests(unittest.TestCase):
    """One HCX call per run, over the whole capped claim."""

    def test_a_faithful_echo_succeeds_end_to_end(self) -> None:
        prepared, _, _ = _capped(events=1)
        transport = _Transport()

        record = run_prepared_question(
            prepared,
            transport=transport,
            settings=_settings(),
            repeat_index=1,
            check_structured_text_leakage=True,
        )

        self.assertEqual(len(transport.calls), 1)
        self.assertTrue(record["hcx_success"])
        self.assertTrue(record["deterministic_citations_attached"])
        self.assertTrue(record["final_answer_valid"])

    def test_the_request_carries_no_citation_placeholder(self) -> None:
        prepared, _, _ = _capped(events=1)
        transport = _Transport()

        run_prepared_question(
            prepared,
            transport=transport,
            settings=_settings(),
            repeat_index=1,
        )

        masked = transport.calls[0]["payload"]["messages"][1]["content"]
        self.assertNotIn("CITATION", masked)
        self.assertNotIn("[1]", masked)

    def test_elapsed_is_measured(self) -> None:
        prepared, _, _ = _capped(events=1)

        record = run_prepared_question(
            prepared,
            transport=_Transport(),
            settings=_settings(),
            repeat_index=1,
        )

        self.assertIsInstance(record["elapsed_ms"], int)
        self.assertGreaterEqual(record["elapsed_ms"], 0)

    def test_structured_text_leakage_fails_closed(self) -> None:
        """A TEXT value such as 감소 must appear only inside its placeholder."""

        prepared = _structured_prepared(
            [
                (
                    ("reference_date", "변동일", "2023-06-30", "[1]"),
                    ("change_direction", "변동 방향", "감소", "[1]"),
                ),
            ]
        )
        text_literal = next(
            literal.text
            for literal in prepared.detached.protection.literals
            if literal.kind == "text"
        )
        self.assertEqual(text_literal, "감소")

        record = run_prepared_question(
            prepared,
            transport=_Transport(lambda masked: f"{text_literal}하여 {masked}"),
            settings=_settings(),
            repeat_index=1,
            check_structured_text_leakage=True,
        )

        self.assertFalse(record["hcx_success"])
        self.assertEqual(
            record["unprotected_text_literals_in_masked_candidate"], ["감소"]
        )

    def test_a_clean_text_claim_still_succeeds(self) -> None:
        prepared = _structured_prepared(
            [
                (
                    ("reference_date", "변동일", "2023-06-30", "[1]"),
                    ("change_direction", "변동 방향", "감소", "[1]"),
                ),
            ]
        )

        record = run_prepared_question(
            prepared,
            transport=_Transport(),
            settings=_settings(),
            repeat_index=1,
            check_structured_text_leakage=True,
        )

        self.assertTrue(record["hcx_success"])
        self.assertEqual(record["unprotected_text_literals_in_masked_candidate"], [])

    def test_a_model_written_citation_fails_closed(self) -> None:
        prepared, _, _ = _capped(events=1)

        record = run_prepared_question(
            prepared,
            transport=_Transport(lambda masked: masked + "[1]"),
            settings=_settings(),
            repeat_index=1,
            check_structured_text_leakage=True,
        )

        self.assertFalse(record["hcx_success"])
        self.assertTrue(record["unexpected_citation_generation"])

    def test_a_dropped_placeholder_fails_closed(self) -> None:
        prepared, _, _ = _capped(events=1)
        first = prepared.detached.protection.placeholders[0]

        record = run_prepared_question(
            prepared,
            transport=_Transport(lambda masked: masked.replace(first, "")),
            settings=_settings(),
            repeat_index=1,
            check_structured_text_leakage=True,
        )

        self.assertFalse(record["hcx_success"])
        self.assertEqual(record["hcx_failure_reason"], "placeholder_missing")


class CappedSummaryTests(unittest.TestCase):
    def _run(self, question_id: str, *, success: bool, reason: str | None) -> dict:
        return {
            "question_id": question_id,
            "hcx_success": success,
            "hcx_failure_reason": reason,
            "deterministic_citations_attached": success,
            "final_answer_valid": success,
            "completion_tokens": 100,
            "elapsed_ms": 250,
        }

    def _verdict(self, question_id: str, *, eligible: bool, reason=None) -> dict:
        return {
            "question_id": question_id,
            "eligible_under_current_caps": eligible,
            "eligibility_reason": reason,
            "candidate_event_count": 2,
            "protected_literal_count": 8,
        }

    def test_a_clean_sweep_passes_the_gate(self) -> None:
        eligibility = [self._verdict("HX01", eligible=True)]
        runs = [self._run("HX01", success=True, reason=None) for _ in range(3)]

        summary = summarize_capped_runs(runs, eligibility, repeat=3)

        self.assertEqual(summary["eligible_question_count"], 1)
        self.assertEqual(summary["total_hcx_calls"], 3)
        self.assertEqual(summary["hcx_success_count"], 3)
        self.assertEqual(summary["fallback_count"], 0)
        self.assertTrue(summary["all_eligible_questions_clean"])
        self.assertEqual(summary["total_completion_tokens"], 300)
        self.assertEqual(summary["total_elapsed_ms"], 750)

    def test_one_failure_fails_the_gate(self) -> None:
        eligibility = [self._verdict("HX01", eligible=True)]
        runs = [
            self._run("HX01", success=True, reason=None),
            self._run("HX01", success=True, reason=None),
            self._run("HX01", success=False, reason="placeholder_missing"),
        ]

        summary = summarize_capped_runs(runs, eligibility, repeat=3)

        self.assertFalse(summary["all_eligible_questions_clean"])
        self.assertEqual(summary["fallback_count"], 1)
        self.assertEqual(summary["failure_reasons"], {"placeholder_missing": 1})
        self.assertAlmostEqual(summary["hcx_success_rate"], 2 / 3)

    def test_ineligible_questions_appear_without_runs(self) -> None:
        eligibility = [
            self._verdict("HX01", eligible=True),
            self._verdict("HX07", eligible=False, reason="too_many_events"),
        ]
        runs = [self._run("HX01", success=True, reason=None) for _ in range(3)]

        summary = summarize_capped_runs(runs, eligibility, repeat=3)

        self.assertEqual(summary["eligible_question_count"], 1)
        self.assertEqual(summary["ineligible_question_ids"], ["HX07"])
        self.assertEqual(summary["by_question"]["HX07"]["runs"], 0)
        self.assertEqual(
            summary["by_question"]["HX07"]["eligibility_reason"], "too_many_events"
        )
        # An ineligible question must not make the gate look dirty.
        self.assertTrue(summary["all_eligible_questions_clean"])

    def test_no_eligible_question_never_reports_clean(self) -> None:
        eligibility = [
            self._verdict("HX07", eligible=False, reason="too_many_events")
        ]

        summary = summarize_capped_runs([], eligibility, repeat=3)

        self.assertEqual(summary["eligible_question_count"], 0)
        self.assertFalse(summary["all_eligible_questions_clean"])
        self.assertEqual(summary["hcx_success_rate"], 0.0)

    def test_per_question_counts_are_reported(self) -> None:
        eligibility = [self._verdict("HX01", eligible=True)]
        runs = [self._run("HX01", success=True, reason=None) for _ in range(3)]

        row = summarize_capped_runs(runs, eligibility, repeat=3)["by_question"]["HX01"]

        self.assertEqual(row["runs"], 3)
        self.assertEqual(row["hcx_successes"], 3)
        self.assertEqual(row["hcx_failures"], 0)
        self.assertEqual(row["citation_attachment_successes"], 3)
        self.assertEqual(row["final_validation_successes"], 3)
        self.assertEqual(row["candidate_event_count"], 2)
        self.assertEqual(row["protected_literal_count"], 8)


class NoLiveCallTests(unittest.TestCase):
    """Without a key the experiment must stop before touching anything."""

    def test_capped_mode_refuses_to_run_without_a_key(self) -> None:
        # Pinned rather than inherited: on a machine that exports a real key
        # this test would otherwise build the pipeline and reach the database.
        with mock.patch.dict(os.environ, {"FESTIVAL_HCX_API_KEY": ""}), mock.patch(
            "scripts.experiment_hcx_holding_live.AnswerPipeline"
        ) as pipeline, contextlib.redirect_stderr(io.StringIO()) as captured:
            exit_code = live_main(
                ["--production-caps", "--question-id", "HX01", "--repeat", "1"]
            )

        self.assertEqual(exit_code, 1)
        pipeline.from_env.assert_not_called()
        self.assertIn("FESTIVAL_HCX_API_KEY", captured.getvalue())


if __name__ == "__main__":
    unittest.main()
