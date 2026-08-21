from __future__ import annotations

import contextlib
import io
import os
import re
import unittest
from unittest import mock

from app.generation.hcx_verbalizer import SYSTEM_PROMPT
from scripts.experiment_hcx_holding_live import (
    LOSSLESS_VERBALIZER_SYSTEM_PROMPT,
    STRICT_EVENT_ORDER_SYSTEM_PROMPT,
    experiment_prompt_variant,
    main as live_main,
    run_prepared_question,
    select_experiment_prompt,
)
from scripts.experiment_hcx_multi_event import EXPERIMENT_SYSTEM_PROMPT
from tests.test_experiment_hcx_holding_live import _Transport, _settings
from tests.test_experiment_hcx_production_caps import _capped


PROMPT = LOSSLESS_VERBALIZER_SYSTEM_PROMPT


class LosslessPromptRuleTests(unittest.TestCase):
    """Each rule the experiment depends on must actually be in the prompt."""

    def test_states_it_is_not_answering_the_question(self) -> None:
        self.assertIn("독자적으로 답하지 않는다", PROMPT)

    def test_states_it_is_not_analysing_the_disclosure(self) -> None:
        self.assertIn("공시를 분석하지 않는다", PROMPT)

    def test_states_it_does_not_fill_missing_information(self) -> None:
        self.assertIn("빠진 정보를 채우지 않는다", PROMPT)

    def test_forbids_new_factual_fields(self) -> None:
        self.assertIn("새로운 사실 항목을 추가하지 않는다", PROMPT)

    def test_forbids_reconstructing_before_and_after_values(self) -> None:
        self.assertIn("변동 전 값", PROMPT)
        self.assertIn("변동 후 값", PROMPT)

    def test_forbids_inferring_change_amount_ratio_and_date(self) -> None:
        for phrase in ("증감 수량", "증감 비율", "기준일"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, PROMPT)

    def test_forbids_calculation(self) -> None:
        self.assertIn("어떤 계산도 하지 않는다", PROMPT)

    def test_forbids_formulas_examples_and_hypothetical_numbers(self) -> None:
        for phrase in ("수식", "예시", "가정된 숫자", "설명용 숫자", "번호 매긴 개요"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, PROMPT)

    def test_forbids_creating_new_placeholders(self) -> None:
        self.assertIn("새로운 placeholder를 만들지 않는다", PROMPT)

    def test_requires_exact_once_and_original_order(self) -> None:
        self.assertIn("정확히 한 번", PROMPT)
        self.assertIn("같은 순서", PROMPT)

    def test_forbids_renaming_the_placeholder_type(self) -> None:
        self.assertIn("의미 유형을 다시 해석하지 않는다", PROMPT)
        self.assertIn("이름을 바꾸지 않는다", PROMPT)

    def test_forbids_transaction_semantics(self) -> None:
        for term in ("매수", "매도", "매입", "처분", "취득", "사들이다"):
            with self.subTest(term=term):
                self.assertIn(term, PROMPT)

    def test_forbids_the_preamble_that_invites_reconstruction(self) -> None:
        self.assertIn("변동 현황은 다음과 같습니다", PROMPT)
        self.assertIn("머리말을 쓰지 않는다", PROMPT)

    def test_forbids_conclusions_and_explanations(self) -> None:
        self.assertIn("결론 문장이나 설명 문장을 덧붙이지 않는다", PROMPT)

    def test_forbids_citations_and_says_where_they_come_from(self) -> None:
        self.assertIn("citation을 쓰지 않는다", PROMPT)
        self.assertIn("결정적으로 붙는다", PROMPT)

    def test_asks_for_the_shortest_faithful_sentence(self) -> None:
        self.assertIn("가장 짧은", PROMPT)


class LosslessPromptExampleTests(unittest.TestCase):
    """Examples teach the shape, never a real answer."""

    def test_shows_a_good_and_bad_pair(self) -> None:
        self.assertIn("GOOD:", PROMPT)
        self.assertIn("BAD:", PROMPT)

    def test_bad_examples_cover_the_observed_failure_modes(self) -> None:
        # A reconstructed prior value, an invented formula, and a renamed
        # placeholder are exactly what the live run produced.
        self.assertIn("이전 보유 비율은", PROMPT)
        self.assertIn("* 100", PROMPT.replace("*100", "* 100"))
        self.assertIn("__FESTIVAL_PERCENTAGE_A__", PROMPT)

    def test_examples_name_no_real_company_or_question(self) -> None:
        for leaked in ("파마리서치", "국민연금공단", "효성중공업", "HX0", "HX1"):
            with self.subTest(leaked=leaked):
                self.assertNotIn(leaked, PROMPT)

    def test_examples_carry_no_real_verified_value(self) -> None:
        for leaked in ("2,000", "4.8", "655,490", "2023-", "2024-"):
            with self.subTest(leaked=leaked):
                self.assertNotIn(leaked, PROMPT)

    def test_the_only_placeholder_names_shown_are_schema_generic(self) -> None:
        names = set(re.findall(r"__FESTIVAL_[A-Z]+_[A-Z]+__", PROMPT))

        self.assertEqual(names, {"__FESTIVAL_NUMBER_A__", "__FESTIVAL_PERCENTAGE_A__"})


class PromptIsolationTests(unittest.TestCase):
    """The new wording must not disturb any prompt already in use."""

    def test_production_prompt_is_untouched(self) -> None:
        self.assertNotEqual(PROMPT, SYSTEM_PROMPT)
        self.assertNotIn("__FESTIVAL_PERCENTAGE_A__", SYSTEM_PROMPT)

    def test_default_experiment_prompt_is_untouched(self) -> None:
        self.assertNotEqual(PROMPT, EXPERIMENT_SYSTEM_PROMPT)
        self.assertNotIn("변환기", EXPERIMENT_SYSTEM_PROMPT)

    def test_strict_event_order_prompt_is_untouched(self) -> None:
        self.assertNotEqual(PROMPT, STRICT_EVENT_ORDER_SYSTEM_PROMPT)
        self.assertTrue(
            STRICT_EVENT_ORDER_SYSTEM_PROMPT.startswith(EXPERIMENT_SYSTEM_PROMPT)
        )


class PromptSelectionTests(unittest.TestCase):
    def test_lossless_flag_selects_the_lossless_prompt(self) -> None:
        self.assertEqual(
            select_experiment_prompt(
                strict_event_order=False, lossless_verbalizer=True
            ),
            PROMPT,
        )

    def test_lossless_wins_over_strict_event_order(self) -> None:
        self.assertEqual(
            select_experiment_prompt(
                strict_event_order=True, lossless_verbalizer=True
            ),
            PROMPT,
        )

    def test_without_the_flag_the_default_experiment_prompt_is_used(self) -> None:
        self.assertEqual(
            select_experiment_prompt(
                strict_event_order=False, lossless_verbalizer=False
            ),
            EXPERIMENT_SYSTEM_PROMPT,
        )

    def test_strict_event_order_still_selects_its_own_prompt(self) -> None:
        self.assertEqual(
            select_experiment_prompt(
                strict_event_order=True, lossless_verbalizer=False
            ),
            STRICT_EVENT_ORDER_SYSTEM_PROMPT,
        )

    def test_variant_names_are_reported(self) -> None:
        self.assertEqual(
            experiment_prompt_variant(
                strict_event_order=False, lossless_verbalizer=True
            ),
            "lossless_verbalizer",
        )
        self.assertEqual(
            experiment_prompt_variant(
                strict_event_order=False, lossless_verbalizer=False
            ),
            "default_experiment",
        )


class RunWiringTests(unittest.TestCase):
    def test_the_flag_sends_the_lossless_prompt(self) -> None:
        prepared, _, _ = _capped(events=1)
        transport = _Transport()

        record = run_prepared_question(
            prepared,
            transport=transport,
            settings=_settings(),
            repeat_index=1,
            lossless_verbalizer=True,
            check_structured_text_leakage=True,
        )

        sent = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertEqual(sent, PROMPT)
        self.assertEqual(record["system_prompt_variant"], "lossless_verbalizer")
        self.assertTrue(record["lossless_verbalizer"])

    def test_capped_mode_without_the_flag_is_unchanged(self) -> None:
        prepared, _, _ = _capped(events=1)
        transport = _Transport()

        record = run_prepared_question(
            prepared,
            transport=transport,
            settings=_settings(),
            repeat_index=1,
            check_structured_text_leakage=True,
        )

        sent = transport.calls[0]["payload"]["messages"][0]["content"]
        self.assertEqual(sent, EXPERIMENT_SYSTEM_PROMPT)
        self.assertEqual(record["system_prompt_variant"], "default_experiment")
        self.assertFalse(record["lossless_verbalizer"])

    def test_the_stricter_prompt_does_not_relax_any_check(self) -> None:
        """The observed HX09 failure must still fail under the new prompt."""

        prepared, _, _ = _capped(events=1)

        record = run_prepared_question(
            prepared,
            transport=_Transport(
                lambda masked: f"변동 전: 2,000주 변동 후: {masked} __FESTIVAL_NUMBER_B__%"
            ),
            settings=_settings(),
            repeat_index=1,
            lossless_verbalizer=True,
            check_structured_text_leakage=True,
        )

        self.assertFalse(record["hcx_success"])
        self.assertEqual(record["hcx_failure_reason"], "placeholder_unexpected")

    def test_an_invented_number_still_fails_under_the_new_prompt(self) -> None:
        prepared, _, _ = _capped(events=1)

        record = run_prepared_question(
            prepared,
            transport=_Transport(lambda masked: f"{masked} 이전에는 5%였습니다"),
            settings=_settings(),
            repeat_index=1,
            lossless_verbalizer=True,
            check_structured_text_leakage=True,
        )

        self.assertFalse(record["hcx_success"])
        self.assertTrue(record["unprotected_numeric_tokens_in_masked_candidate"])

    def test_a_faithful_reply_still_succeeds(self) -> None:
        prepared, _, _ = _capped(events=1)

        record = run_prepared_question(
            prepared,
            transport=_Transport(),
            settings=_settings(),
            repeat_index=1,
            lossless_verbalizer=True,
            check_structured_text_leakage=True,
        )

        self.assertTrue(record["hcx_success"])
        self.assertTrue(record["final_answer_valid"])


class CliTests(unittest.TestCase):
    def test_lossless_requires_production_caps(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as captured:
            with self.assertRaises(SystemExit):
                live_main(["--lossless-verbalizer", "--question-id", "HX09"])

        self.assertIn("--production-caps", captured.getvalue())

    def test_no_key_stops_before_any_call(self) -> None:
        with mock.patch.dict(os.environ, {"FESTIVAL_HCX_API_KEY": ""}), mock.patch(
            "scripts.experiment_hcx_holding_live.AnswerPipeline"
        ) as pipeline, contextlib.redirect_stderr(io.StringIO()):
            exit_code = live_main(
                [
                    "--production-caps",
                    "--lossless-verbalizer",
                    "--question-id",
                    "HX09",
                    "--repeat",
                    "5",
                ]
            )

        self.assertEqual(exit_code, 1)
        pipeline.from_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
