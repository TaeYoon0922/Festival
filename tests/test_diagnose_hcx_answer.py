from __future__ import annotations

import json
import unittest

from app.generation.protected_literals import protect_literals
from scripts.diagnose_hcx_answer import (
    _classify,
    _counts_by_type,
    _excerpt,
    _finish_reason,
    _reorder_summary,
    _usage,
)


LONG_ANSWER = " ".join(
    f"{2020 + index}년 0{index + 1}월 0{index + 1}일 기준 {index},000주입니다.[{index + 1}]"
    for index in range(8)
)
PROTECTION = protect_literals(LONG_ANSWER)
EXPECTED = list(PROTECTION.placeholders)


def _report(found: list[str], finish: str, *, received: bool = True) -> dict:
    return {
        "hcx_candidate_received": received,
        "missing_placeholders": sorted(set(EXPECTED) - set(found)),
        "unexpected_placeholders": sorted(set(found) - set(EXPECTED)),
        "duplicated_placeholders": sorted(
            {token for token in found if found.count(token) > 1}
        ),
        "reorder_summary": _reorder_summary(EXPECTED, found),
        "response_finish_reason": finish,
        "found_is_prefix_of_expected": found == EXPECTED[: len(found)],
        "placeholder_integrity_valid": found == EXPECTED,
        "expected_placeholder_count": len(EXPECTED),
        "found_placeholder_count": len(found),
    }


class ClassificationTests(unittest.TestCase):
    def test_truncation_is_flagged_by_finish_reason(self) -> None:
        codes = _classify(_report(EXPECTED[:10], "length"))["codes"]

        self.assertEqual(codes, ["A", "E", "F"])

    def test_a_clean_prefix_is_treated_as_truncation(self) -> None:
        """A reply that simply stops is a different failure from an edited one."""

        result = _classify(_report(EXPECTED[:10], "stop"))

        self.assertIn("E", result["codes"])
        self.assertTrue(result["truncation_signature"])

    def test_a_dropped_middle_placeholder_is_not_truncation(self) -> None:
        found = EXPECTED[:5] + EXPECTED[6:]

        result = _classify(_report(found, "stop"))

        self.assertEqual(result["codes"], ["A"])
        self.assertFalse(result["truncation_signature"])

    def test_duplication_is_classified(self) -> None:
        codes = _classify(_report(EXPECTED + [EXPECTED[0]], "stop"))["codes"]

        self.assertEqual(codes, ["B"])

    def test_reordering_is_classified(self) -> None:
        found = [EXPECTED[1], EXPECTED[0], *EXPECTED[2:]]

        codes = _classify(_report(found, "stop"))["codes"]

        self.assertEqual(codes, ["C"])

    def test_a_renamed_placeholder_is_classified_as_spelling(self) -> None:
        found = EXPECTED[:-1] + ["__FESTIVAL_NUMBER_ZZ__"]

        codes = _classify(_report(found, "stop"))["codes"]

        self.assertIn("D", codes)

    def test_no_reply_is_other(self) -> None:
        codes = _classify(_report([], "stop", received=False))["codes"]

        self.assertEqual(codes, ["H"])

    def test_ignoring_every_placeholder_is_not_truncation(self) -> None:
        """The observed P07 failure: 96 expected, 0 returned, finish_reason=stop.

        An empty run is a prefix of every run, so the prefix test would call this
        truncation.  It is not: the model discarded the masked text and wrote its
        own prose, and more output room cannot fix that.
        """

        result = _classify(_report([], "stop"))

        self.assertEqual(result["codes"], ["A", "G"])
        self.assertEqual(
            result["labels"],
            ["placeholder missing", "long full-answer rewrite unsuitable"],
        )
        self.assertFalse(result["truncation_signature"])
        self.assertTrue(result["ignored_every_placeholder"])

    def test_a_genuinely_truncated_reply_still_reports_truncation(self) -> None:
        result = _classify(_report(EXPECTED[:1], "stop"))

        self.assertIn("E", result["codes"])
        self.assertNotIn("G", result["codes"])
        self.assertTrue(result["truncation_signature"])

    def test_a_clean_run_reports_no_cause(self) -> None:
        result = _classify(_report(EXPECTED, "stop"))

        self.assertEqual(result["codes"], [])
        self.assertEqual(result["labels"], ["no placeholder failure"])


class ReorderSummaryTests(unittest.TestCase):
    def test_identical_runs_have_no_summary(self) -> None:
        self.assertIsNone(_reorder_summary(EXPECTED, list(EXPECTED)))

    def test_reports_the_first_divergence(self) -> None:
        found = [EXPECTED[1], EXPECTED[0], *EXPECTED[2:]]

        summary = _reorder_summary(EXPECTED, found)

        self.assertEqual(summary["first_divergence_index"], 0)
        self.assertEqual(summary["expected"], EXPECTED[0])
        self.assertEqual(summary["found"], EXPECTED[1])

    def test_reports_a_short_run_as_agreeing_until_it_ends(self) -> None:
        summary = _reorder_summary(EXPECTED, EXPECTED[:3])

        self.assertEqual(summary["first_divergence_index"], 3)
        self.assertEqual(summary["expected"], EXPECTED[3])
        self.assertIn("note", summary)


class ResponseMetadataTests(unittest.TestCase):
    def test_reads_openai_finish_reason(self) -> None:
        response = {"choices": [{"finish_reason": "length", "message": {}}]}

        self.assertEqual(_finish_reason(response), "length")

    def test_reads_native_stop_reason(self) -> None:
        response = {"result": {"stopReason": "length", "message": {}}}

        self.assertEqual(_finish_reason(response), "length")

    def test_missing_finish_reason_is_none(self) -> None:
        self.assertIsNone(_finish_reason({"choices": []}))

    def test_usage_keeps_only_token_counts(self) -> None:
        usage = _usage(
            {
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                    "api_key": "leak",
                }
            }
        )

        self.assertEqual(
            usage, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        self.assertNotIn("api_key", usage)


class ExcerptTests(unittest.TestCase):
    def test_excerpt_is_bounded_and_not_the_whole_reply(self) -> None:
        reply = "가" * 4000 + EXPECTED[0]

        excerpt = _excerpt(reply, EXPECTED, [EXPECTED[0]])

        rendered = json.dumps(excerpt, ensure_ascii=False)
        self.assertLess(len(rendered), 600)
        self.assertNotIn("가" * 200, rendered)

    def test_no_divergence_is_reported_plainly(self) -> None:
        self.assertEqual(
            _excerpt("본문", EXPECTED, list(EXPECTED)), {"note": "no divergence"}
        )

    def test_missing_reply_has_no_excerpt(self) -> None:
        self.assertIsNone(_excerpt(None, EXPECTED, []))


class LiteralCountTests(unittest.TestCase):
    def test_counts_every_protected_kind(self) -> None:
        counts = _counts_by_type(PROTECTION)

        self.assertEqual(counts, {"date": 8, "number": 8, "citation": 8})


if __name__ == "__main__":
    unittest.main()
