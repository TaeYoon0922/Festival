from __future__ import annotations

import json
import unittest

from app.generation.hcx_verbalizer import SYSTEM_PROMPT, HcxSettings
from app.generation.protected_literals import protect_literals, restore_literals
from scripts.experiment_hcx_multi_event import (
    EXPERIMENT_SYSTEM_PROMPT,
    build_experiment_claim,
    placeholder_type_breakdown,
    run_once,
    summarize,
)


def _settings() -> HcxSettings:
    return HcxSettings(
        enabled=True,
        endpoint="https://clova.example/v1/chat/completions",
        api_key="test-key",
    )


class _Transport:
    """Answers with whatever the caller decides, given the masked prompt."""

    def __init__(self, respond) -> None:
        self._respond = respond
        self.calls: list[dict] = []

    def post_json(self, url, *, headers, payload, timeout_seconds):
        self.calls.append({"url": url, "payload": payload})
        masked = payload["messages"][1]["content"]
        content = self._respond(masked)
        if isinstance(content, Exception):
            raise content
        return {
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": len(content) // 2},
        }


class ExperimentClaimTests(unittest.TestCase):
    def test_builds_one_date_and_one_share_count_per_event(self) -> None:
        claim = build_experiment_claim(4)

        self.assertEqual(len(claim.fields), 8)
        self.assertEqual(len(claim.citations), 4)

    def test_events_carry_distinct_values(self) -> None:
        claim = build_experiment_claim(6)

        values = [field.value for field in claim.fields]
        self.assertEqual(len(set(values)), len(values))

    def test_markers_are_numbered_per_event(self) -> None:
        claim = build_experiment_claim(3)

        self.assertEqual(
            [citation.marker for citation in claim.citations], ["[1]", "[2]", "[3]"]
        )

    def test_claim_grows_with_event_count(self) -> None:
        small = build_experiment_claim(4)
        large = build_experiment_claim(10)

        self.assertLess(
            len(small.deterministic_text), len(large.deterministic_text)
        )

    def test_share_counts_keep_thousands_separators(self) -> None:
        claim = build_experiment_claim(2)

        shares = [f.value for f in claim.fields if f.name == "after_shares"]
        for value in shares:
            self.assertIn(",", value)


class ExperimentPromptTests(unittest.TestCase):
    def test_the_experiment_prompt_is_not_the_production_prompt(self) -> None:
        self.assertNotEqual(EXPERIMENT_SYSTEM_PROMPT, SYSTEM_PROMPT)

    def test_the_experiment_prompt_forbids_summarising(self) -> None:
        self.assertIn("요약", EXPERIMENT_SYSTEM_PROMPT)
        self.assertIn("빠뜨리지", EXPERIMENT_SYSTEM_PROMPT)

    def test_the_request_sends_the_experiment_prompt(self) -> None:
        transport = _Transport(lambda masked: masked)

        run_once(transport, _settings(), 4)

        system = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertEqual(system, EXPERIMENT_SYSTEM_PROMPT)

    def test_the_request_sends_only_masked_text(self) -> None:
        transport = _Transport(lambda masked: masked)

        run_once(transport, _settings(), 4)

        user = transport.calls[0]["payload"]["messages"][1]["content"]
        self.assertIn("__FESTIVAL_", user)
        self.assertNotIn("2023-01-01", user)


class RunOutcomeTests(unittest.TestCase):
    def test_a_faithful_echo_passes_every_check(self) -> None:
        record = run_once(_Transport(lambda masked: masked), _settings(), 4)

        self.assertTrue(record["placeholder_integrity_valid"])
        self.assertTrue(record["all_events_kept"])
        self.assertTrue(record["candidate_valid"])
        self.assertEqual(record["inference_markers"], [])
        self.assertEqual(record["finish_reason"], "stop")

    def test_a_dropped_event_value_is_reported_as_event_loss(self) -> None:
        def respond(masked: str) -> str:
            protection = protect_literals(
                build_experiment_claim(4).deterministic_text
            )
            date = next(
                literal.placeholder
                for literal in protection.literals
                if literal.kind == "date"
            )
            return masked.replace(date, "")

        record = run_once(_Transport(respond), _settings(), 4)

        self.assertFalse(record["placeholder_integrity_valid"])
        self.assertFalse(record["all_events_kept"])
        self.assertEqual(record["placeholder_integrity_reason"], "placeholder_missing")
        self.assertNotIn("candidate_valid", record)

    def test_dropping_only_citations_is_not_reported_as_event_loss(self) -> None:
        """Integrity still fails, but the events themselves survived."""

        def respond(masked: str) -> str:
            protection = protect_literals(
                build_experiment_claim(4).deterministic_text
            )
            for literal in protection.literals:
                if literal.kind == "citation":
                    masked = masked.replace(literal.placeholder, "")
            return masked

        record = run_once(_Transport(respond), _settings(), 4)

        self.assertFalse(record["placeholder_integrity_valid"])
        self.assertTrue(record["all_events_kept"])
        self.assertTrue(record["field_placeholders_all_preserved"])
        self.assertFalse(record["citation_placeholders_all_preserved"])
        self.assertEqual(record["found_placeholders"], 8)

    def test_an_added_conclusion_is_surfaced(self) -> None:
        record = run_once(
            _Transport(lambda masked: masked + " 이를 통해 보유 추이를 알 수 있습니다."),
            _settings(),
            4,
        )

        self.assertTrue(record["placeholder_integrity_valid"])
        self.assertIn("이를 통해", record["inference_markers"])

    def test_a_hallucinated_number_fails_the_validator(self) -> None:
        record = run_once(
            _Transport(lambda masked: masked + " 총 99,999주"), _settings(), 4
        )

        self.assertTrue(record["placeholder_integrity_valid"])
        self.assertFalse(record["candidate_valid"])
        self.assertEqual(record["validator_reason"], "numeric_token_changed")

    def test_a_transport_failure_does_not_stop_the_sweep(self) -> None:
        record = run_once(
            _Transport(lambda masked: RuntimeError("boom")), _settings(), 4
        )

        self.assertEqual(record["transport_error"], "RuntimeError")
        self.assertNotIn("candidate_valid", record)

    def test_restoration_round_trips_the_literals(self) -> None:
        claim = build_experiment_claim(4)
        protection = protect_literals(claim.deterministic_text)

        self.assertEqual(
            restore_literals(protection.masked, protection), claim.deterministic_text
        )


class PlaceholderTypeBreakdownTests(unittest.TestCase):
    """Losing every citation and losing whole events look identical in totals."""

    def setUp(self) -> None:
        self.claim = build_experiment_claim(4)
        self.protection = protect_literals(self.claim.deterministic_text)
        self.by_kind = {
            kind: [
                literal.placeholder
                for literal in self.protection.literals
                if literal.kind == kind
            ]
            for kind in ("date", "number", "citation")
        }

    def test_four_events_expect_four_dates_four_numbers_eight_citations(self) -> None:
        breakdown = placeholder_type_breakdown(self.protection, [])

        self.assertEqual(
            breakdown["expected_placeholders_by_type"],
            {"date": 4, "number": 4, "citation": 8},
        )

    def test_a_full_echo_preserves_every_kind(self) -> None:
        breakdown = placeholder_type_breakdown(
            self.protection, list(self.protection.placeholders)
        )

        self.assertEqual(
            breakdown["missing_placeholders_by_type"],
            {"date": 0, "number": 0, "citation": 0},
        )
        self.assertTrue(breakdown["field_placeholders_all_preserved"])
        self.assertTrue(breakdown["citation_placeholders_all_preserved"])

    def test_the_observed_pattern_is_citation_loss_not_event_loss(self) -> None:
        """8 of 16 survive: every date and number, and no citation."""

        found = self.by_kind["date"] + self.by_kind["number"]

        breakdown = placeholder_type_breakdown(self.protection, found)

        self.assertEqual(
            breakdown["found_placeholders_by_type"],
            {"date": 4, "number": 4, "citation": 0},
        )
        self.assertEqual(breakdown["missing_placeholders_by_type"]["citation"], 8)
        self.assertTrue(breakdown["field_placeholders_all_preserved"])
        self.assertFalse(breakdown["citation_placeholders_all_preserved"])

    def test_dropping_a_whole_event_shows_as_field_loss(self) -> None:
        found = (
            self.by_kind["date"][:-1]
            + self.by_kind["number"][:-1]
            + self.by_kind["citation"][:-2]
        )

        breakdown = placeholder_type_breakdown(self.protection, found)

        self.assertFalse(breakdown["field_placeholders_all_preserved"])
        self.assertEqual(breakdown["missing_placeholders_by_type"]["date"], 1)
        self.assertEqual(breakdown["missing_placeholders_by_type"]["number"], 1)

    def test_invented_placeholders_are_counted_separately(self) -> None:
        breakdown = placeholder_type_breakdown(
            self.protection,
            [*self.protection.placeholders, "__FESTIVAL_NUMBER_ZZ__"],
        )

        self.assertEqual(breakdown["unrecognized_placeholders"], 1)


class ShowOutputTests(unittest.TestCase):
    """A run that fails integrity is exactly when the output matters most."""

    def test_raw_candidate_is_reported_when_integrity_fails(self) -> None:
        def respond(masked: str) -> str:
            protection = protect_literals(
                build_experiment_claim(4).deterministic_text
            )
            citations = [
                literal.placeholder
                for literal in protection.literals
                if literal.kind == "citation"
            ]
            for token in citations:
                masked = masked.replace(token, "")
            return masked

        record = run_once(_Transport(respond), _settings(), 4, show_output=True)

        self.assertFalse(record["placeholder_integrity_valid"])
        self.assertIn("raw_hcx_candidate", record)
        self.assertTrue(record["raw_hcx_candidate"])
        self.assertNotIn("restored_output", record)

    def test_raw_candidate_is_reported_on_success_too(self) -> None:
        record = run_once(
            _Transport(lambda masked: masked), _settings(), 4, show_output=True
        )

        self.assertIn("raw_hcx_candidate", record)
        self.assertIn("restored_output", record)

    def test_nothing_is_shown_without_the_flag(self) -> None:
        record = run_once(_Transport(lambda masked: masked), _settings(), 4)

        self.assertNotIn("raw_hcx_candidate", record)
        self.assertNotIn("restored_output", record)

    def test_the_report_never_carries_credentials(self) -> None:
        record = run_once(
            _Transport(lambda masked: masked), _settings(), 4, show_output=True
        )

        body = json.dumps(record, ensure_ascii=False).lower()
        for forbidden in ("authorization", "bearer", "test-key", "api_key"):
            self.assertNotIn(forbidden, body)


class SummaryTests(unittest.TestCase):
    def _run(self, size: int, *, integrity: bool, valid: bool, inference: bool) -> dict:
        return {
            "event_count": size,
            "placeholder_integrity_valid": integrity,
            "candidate_valid": valid,
            "inference_markers": ["이를 통해"] if inference else [],
        }

    def test_groups_by_event_count(self) -> None:
        runs = [
            self._run(4, integrity=True, valid=True, inference=False),
            self._run(4, integrity=True, valid=True, inference=False),
            self._run(10, integrity=False, valid=False, inference=False),
        ]

        summary = summarize(runs)

        self.assertEqual(summary["by_event_count"]["4"]["runs"], 2)
        self.assertEqual(summary["by_event_count"]["4"]["validator_ok"], 2)
        self.assertEqual(summary["by_event_count"]["10"]["validator_ok"], 0)

    def test_reports_the_largest_size_that_was_clean_in_every_run(self) -> None:
        runs = [
            self._run(4, integrity=True, valid=True, inference=False),
            self._run(6, integrity=True, valid=True, inference=False),
            self._run(6, integrity=False, valid=False, inference=False),
            self._run(10, integrity=False, valid=False, inference=False),
        ]

        self.assertEqual(summarize(runs)["largest_fully_clean_event_count"], 4)

    def test_no_clean_size_reports_none(self) -> None:
        runs = [self._run(4, integrity=False, valid=False, inference=False)]

        self.assertIsNone(summarize(runs)["largest_fully_clean_event_count"])


if __name__ == "__main__":
    unittest.main()
