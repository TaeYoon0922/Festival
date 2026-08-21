from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace

from app.generation.answer_validator import FORBIDDEN_INVESTMENT_TERMS
from app.generation.compact_claim import (
    ClaimCitation,
    ClaimField,
    CompactClaim,
    _render,
    build_compact_claim,
)
from app.generation.hcx_verbalizer import HcxSettings
from app.generation.lossless_verbalization import (
    LOSSLESS_VERBALIZER_SYSTEM_PROMPT,
)
from app.generation.protected_literals import (
    PLACEHOLDER_PATTERN,
    check_placeholder_integrity,
    restore_literals,
)
from scripts.experiment_hcx_holding_live import (
    PreparedHoldingQuestion,
    STRICT_EVENT_ORDER_SYSTEM_PROMPT,
    TARGET_QUESTION_IDS,
    attach_live_detached_citations,
    detach_live_claim_citations,
    diagnose_prepare,
    orchestrator_exception_stage,
    placeholder_diagnostics,
    prepare_actual_question,
    real_structure_diagnostics,
    run_event_wise_question,
    run_prepared_question,
    split_prepared_events,
    summarize_event_wise_runs,
    summarize_live_runs,
    target_question_rows,
)
from scripts.experiment_hcx_multi_event import EXPERIMENT_SYSTEM_PROMPT
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


def _structured_claim(event_fields):
    """Build a claim from already-structured field values, like ClaimBuilder."""

    fields = []
    citations = []
    seen_markers = set()
    for event_index, event in enumerate(event_fields):
        for field_index, (name, label, value, marker) in enumerate(event):
            chunk_id = f"doc-{event_index}:ch-{field_index}"
            fields.append(
                ClaimField(
                    name=name,
                    label=label,
                    value=value,
                    marker=marker,
                    chunk_id=chunk_id,
                )
            )
            if marker not in seen_markers:
                seen_markers.add(marker)
                citations.append(
                    ClaimCitation(
                        marker=marker,
                        chunk_id=chunk_id,
                        doc_id=f"doc-{event_index}",
                        source_refs=(
                            {
                                "table_id": f"t{event_index}",
                                "row_start": 1,
                                "row_end": 1,
                            },
                        ),
                    )
                )
    text = _render("테스트회사", "국민연금기금", fields)
    return CompactClaim(
        question="테스트 질문",
        company="테스트회사",
        reporter="국민연금기금",
        fields=tuple(fields),
        citations=tuple(citations),
        deterministic_text=text,
    )


def _structured_prepared(event_fields) -> PreparedHoldingQuestion:
    prepared, _, _ = _prepared(events=len(event_fields))
    claim = _structured_claim(event_fields)
    detached = detach_live_claim_citations(claim)
    requested = tuple(field.name for field in claim.fields[: len(event_fields[0])])
    diagnostics = dict(prepared.prepare_diagnostics)
    diagnostics.update(
        {
            "candidate_event_count": len(event_fields),
            "claim_field_count": len(claim.fields),
            "claim_citation_count": len(claim.citations),
        }
    )
    return replace(
        prepared,
        requested_fields=requested,
        candidate_event_count=len(event_fields),
        claim=claim,
        detached=detached,
        structure=real_structure_diagnostics(
            claim,
            detached,
            requested_fields=requested,
            candidate_event_count=len(event_fields),
        ),
        prepare_diagnostics=diagnostics,
    )


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

    def test_orchestrator_error_from_resolver_traceback_is_named_resolver(self) -> None:
        try:
            exec(
                compile(
                    "raise ValueError('resolver failed')",
                    "app/reasoning/holding_event_resolver.py",
                    "exec",
                )
            )
        except ValueError as error:
            self.assertEqual(orchestrator_exception_stage(error), "resolver")


class TextFieldPlaceholderTests(unittest.TestCase):
    def test_change_direction_increase_is_masked_and_restored(self) -> None:
        claim = _structured_claim(
            [[("change_direction", "증감 방향", "증가", "[1]")]]
        )

        detached = detach_live_claim_citations(claim)

        self.assertIn("__FESTIVAL_TEXT_A__", detached.protection.masked)
        self.assertNotIn("증가", detached.protection.masked)
        self.assertEqual(
            restore_literals(detached.protection.masked, detached.protection),
            detached.text,
        )

    def test_change_direction_decrease_is_masked_and_restored(self) -> None:
        claim = _structured_claim(
            [[("change_direction", "증감 방향", "감소", "[1]")]]
        )

        detached = detach_live_claim_citations(claim)

        self.assertIn("__FESTIVAL_TEXT_A__", detached.protection.masked)
        self.assertNotIn("감소", detached.protection.masked)
        self.assertEqual(detached.protection.mapping()["__FESTIVAL_TEXT_A__"], "감소")

    def test_mixed_date_number_and_text_event_uses_all_placeholder_types(self) -> None:
        claim = _structured_claim(
            [[
                ("reference_date", "변동일", "2024-01-02", "[1]"),
                ("after_shares", "변동 후 주식수", "1,500주", "[1]"),
                ("change_direction", "증감 방향", "증가", "[1]"),
            ]]
        )

        detached = detach_live_claim_citations(claim)
        breakdown = placeholder_diagnostics(
            detached.protection,
            PLACEHOLDER_PATTERN.findall(detached.protection.masked),
        )

        self.assertEqual(
            {
                kind: len(tokens)
                for kind, tokens in breakdown[
                    "expected_placeholders_by_type"
                ].items()
            },
            {"date": 1, "number": 1, "text": 1},
        )
        self.assertTrue(breakdown["text_placeholders_all_preserved"])

    def test_text_placeholder_missing_fails_integrity(self) -> None:
        claim = _structured_claim(
            [[("change_direction", "증감 방향", "감소", "[1]")]]
        )
        detached = detach_live_claim_citations(claim)
        token = detached.protection.placeholders[0]

        result = check_placeholder_integrity(
            detached.protection.masked.replace(token, "", 1),
            detached.protection,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "placeholder_missing")

    def test_text_placeholder_duplicated_fails_integrity(self) -> None:
        claim = _structured_claim(
            [[("change_direction", "증감 방향", "감소", "[1]")]]
        )
        detached = detach_live_claim_citations(claim)
        token = detached.protection.placeholders[0]

        result = check_placeholder_integrity(
            detached.protection.masked + token,
            detached.protection,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "placeholder_duplicated")

    def test_text_placeholder_reordered_fails_integrity(self) -> None:
        claim = _structured_claim(
            [[
                ("reference_date", "변동일", "2024-01-02", "[1]"),
                ("change_direction", "증감 방향", "감소", "[1]"),
            ]]
        )
        detached = detach_live_claim_citations(claim)
        first, second = detached.protection.placeholders
        swapped = detached.protection.masked.replace(first, "__SWAP__", 1)
        swapped = swapped.replace(second, first, 1).replace("__SWAP__", second, 1)

        result = check_placeholder_integrity(swapped, detached.protection)

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "placeholder_reordered")

    def test_unrecognized_text_placeholder_fails_integrity(self) -> None:
        claim = _structured_claim(
            [[("change_direction", "증감 방향", "감소", "[1]")]]
        )
        detached = detach_live_claim_citations(claim)

        result = check_placeholder_integrity(
            detached.protection.masked + "__FESTIVAL_TEXT_Z__",
            detached.protection,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "placeholder_unexpected")

    def test_same_decrease_value_has_unique_placeholder_per_event(self) -> None:
        claim = _structured_claim(
            [
                [("change_direction", "증감 방향", "감소", "[1]")],
                [("change_direction", "증감 방향", "감소", "[2]")],
            ]
        )

        detached = detach_live_claim_citations(claim)
        text_tokens = [
            literal.placeholder
            for literal in detached.protection.literals
            if literal.kind == "text"
        ]

        self.assertEqual(text_tokens, ["__FESTIVAL_TEXT_A__", "__FESTIVAL_TEXT_B__"])
        self.assertEqual(len(set(text_tokens)), 2)
        self.assertEqual(
            [detached.protection.mapping()[token] for token in text_tokens],
            ["감소", "감소"],
        )

    def test_citation_attachment_uses_text_field_event_ownership(self) -> None:
        claim = _structured_claim(
            [[
                ("reference_date", "변동일", "2024-01-02", "[1]"),
                ("change_direction", "증감 방향", "감소", "[2]"),
            ]]
        )
        detached = detach_live_claim_citations(claim)

        result = attach_live_detached_citations(
            detached.protection.masked,
            detached,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.attached_citation_count, 2)
        self.assertIn("감소 [1][2]", result.final_answer)

    def test_hx07_like_three_requested_fields_keep_all_events(self) -> None:
        claim = _structured_claim(
            [
                [
                    (
                        "change_shares",
                        "증감 주식수",
                        f"{index + 1},000주",
                        f"[{index + 1}]",
                    ),
                    (
                        "change_ratio",
                        "증감 비율",
                        f"{index + 1}.0%",
                        f"[{index + 1}]",
                    ),
                    (
                        "change_direction",
                        "증감 방향",
                        "증가",
                        f"[{index + 1}]",
                    ),
                ]
                for index in range(4)
            ]
        )

        detached = detach_live_claim_citations(claim)

        self.assertEqual(len(detached.attachments), 4)
        self.assertEqual(len(detached.protection.placeholders), 12)
        self.assertEqual(
            [literal.kind for literal in detached.protection.literals].count("text"),
            4,
        )

    def test_hx16_like_date_number_direction_attaches_for_six_events(self) -> None:
        claim = _structured_claim(
            [
                [
                    (
                        "reference_date",
                        "변동일",
                        f"2024-01-{index + 1:02d}",
                        f"[{index + 1}]",
                    ),
                    (
                        "after_shares",
                        "변동 후 주식수",
                        f"{index + 1},500주",
                        f"[{index + 1}]",
                    ),
                    (
                        "change_direction",
                        "증감 방향",
                        "감소",
                        f"[{index + 1}]",
                    ),
                ]
                for index in range(6)
            ]
        )
        detached = detach_live_claim_citations(claim)

        result = attach_live_detached_citations(
            detached.protection.masked,
            detached,
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.attached_citation_count, 6)
        self.assertEqual(len(detached.attachments), 6)


class EventWiseHoldingExperimentTests(unittest.TestCase):
    def test_each_hcx_call_receives_exactly_one_event_in_original_order(self) -> None:
        prepared, _, _ = _prepared(events=3)
        transport = _Transport()
        clock_values = iter((0.0, 0.1, 0.1, 0.3, 0.3, 0.6))

        question, events = run_event_wise_question(
            prepared,
            transport=transport,
            settings=_settings(),
            repeat_index=1,
            clock=lambda: next(clock_values),
        )

        self.assertEqual(len(split_prepared_events(prepared)), 3)
        self.assertEqual(len(transport.calls), 3)
        self.assertEqual([event["event_index"] for event in events], [0, 1, 2])
        self.assertTrue(question["question_served_success"])
        self.assertTrue(question["question_all_events_hcx_success"])
        self.assertEqual(question["total_hcx_calls"], 3)
        self.assertEqual(question["total_completion_tokens"], 369)
        self.assertEqual(question["total_elapsed_ms"], 600.0)
        for call in transport.calls:
            system_message, user_message = call["payload"]["messages"]
            self.assertEqual(system_message["content"], STRICT_EVENT_ORDER_SYSTEM_PROMPT)
            self.assertEqual(len(PLACEHOLDER_PATTERN.findall(user_message["content"])), 2)

    def test_only_failed_event_uses_deterministic_fallback(self) -> None:
        prepared, _, _ = _prepared(events=3)
        call_index = 0

        def fail_second(masked: str) -> str:
            nonlocal call_index
            call_index += 1
            if call_index == 2:
                return masked.replace(PLACEHOLDER_PATTERN.findall(masked)[0], "", 1)
            return masked

        question, events = run_event_wise_question(
            prepared,
            transport=_Transport(fail_second),
            settings=_settings(),
            repeat_index=1,
            show_output=True,
        )

        self.assertEqual(
            [event["hcx_event_success"] for event in events],
            [True, False, True],
        )
        self.assertEqual(
            [event["fallback_used"] for event in events],
            [False, True, False],
        )
        self.assertEqual(events[1]["fallback_reason"], "placeholder_missing")
        self.assertEqual(question["fallback_event_count"], 1)
        self.assertFalse(question["question_all_events_hcx_success"])
        self.assertTrue(question["question_served_success"])
        self.assertEqual(question["served_event_indexes"], [0, 1, 2])
        self.assertIn("2023-06-30", question["served_answer"])
        self.assertIn("2024-06-30", question["served_answer"])
        self.assertIn("2025-06-30", question["served_answer"])

    def test_exact_structured_text_leakage_fails_closed(self) -> None:
        prepared = _structured_prepared(
            [[("change_direction", "증감 방향", "감소", "[1]")]]
        )

        question, events = run_event_wise_question(
            prepared,
            transport=_Transport(lambda masked: masked + " 감소"),
            settings=_settings(),
            repeat_index=1,
        )

        event = events[0]
        self.assertTrue(event["placeholder_integrity_valid"])
        self.assertEqual(
            event["unprotected_text_literals_in_masked_candidate"],
            ["감소"],
        )
        self.assertFalse(event["hcx_event_success"])
        self.assertEqual(
            event["fallback_reason"],
            "unprotected_structured_text_leakage",
        )
        self.assertTrue(event["fallback_used"])
        self.assertTrue(question["question_served_success"])

    def test_forbidden_investment_language_fails_only_that_event(self) -> None:
        prepared, _, _ = _prepared(events=1)
        forbidden = next(
            term
            for term in FORBIDDEN_INVESTMENT_TERMS
            if term not in prepared.detached.protection.masked
        )

        question, events = run_event_wise_question(
            prepared,
            transport=_Transport(lambda masked: masked + " " + forbidden),
            settings=_settings(),
            repeat_index=1,
        )

        event = events[0]
        self.assertTrue(event["forbidden_investment_language_detected"])
        self.assertEqual(event["fallback_reason"], "forbidden_investment_language")
        self.assertFalse(event["hcx_event_success"])
        self.assertTrue(event["fallback_used"])
        self.assertEqual(question["fallback_event_count"], 1)
        self.assertTrue(question["question_served_success"])

    def test_event_wise_summary_aggregates_repeats_latency_and_fallback(self) -> None:
        prepared, _, _ = _prepared(events=2)
        question_one, events_one = run_event_wise_question(
            prepared,
            transport=_Transport(),
            settings=_settings(),
            repeat_index=1,
        )
        question_two, events_two = run_event_wise_question(
            prepared,
            transport=_Transport(
                lambda masked: masked.replace(
                    PLACEHOLDER_PATTERN.findall(masked)[0], "", 1
                )
            ),
            settings=_settings(),
            repeat_index=2,
        )

        summary = summarize_event_wise_runs(
            [question_one, question_two],
            [*events_one, *events_two],
            repeat=2,
        )

        hx07 = summary["by_question"]["HX07"]
        self.assertEqual(hx07["runs"], 2)
        self.assertEqual(hx07["event_count"], 2)
        self.assertEqual(hx07["total_hcx_calls"], 4)
        self.assertEqual(hx07["hcx_event_success_count"], 2)
        self.assertEqual(hx07["hcx_event_failure_count"], 2)
        self.assertEqual(hx07["fallback_event_count"], 2)
        self.assertEqual(hx07["failure_reasons"], {"placeholder_missing": 2})
        self.assertTrue(hx07["question_served_success"])
        self.assertFalse(hx07["question_all_events_hcx_success"])


class ActualHoldingLiveRunTests(unittest.TestCase):
    def test_strict_prompt_requires_exact_once_placeholder_use(self) -> None:
        self.assertIn("모든 placeholder", STRICT_EVENT_ORDER_SYSTEM_PROMPT)
        self.assertIn("정확히 한 번씩", STRICT_EVENT_ORDER_SYSTEM_PROMPT)
        self.assertIn("누락하거나 중복하지 않는다", STRICT_EVENT_ORDER_SYSTEM_PROMPT)

    def test_strict_prompt_requires_exact_placeholder_order(self) -> None:
        self.assertIn("정확히 동일한 순서", STRICT_EVENT_ORDER_SYSTEM_PROMPT)
        self.assertIn("앞이나 뒤로 이동하지 않는다", STRICT_EVENT_ORDER_SYSTEM_PROMPT)
        self.assertIn("A → B → C → D", STRICT_EVENT_ORDER_SYSTEM_PROMPT)

    def test_strict_prompt_forbids_cross_event_field_regrouping(self) -> None:
        self.assertIn("나눌 수 없는 하나의 record", STRICT_EVENT_ORDER_SYSTEM_PROMPT)
        self.assertIn("필드를 재그룹화하지 않는다", STRICT_EVENT_ORDER_SYSTEM_PROMPT)

    def test_strict_prompt_forbids_event_merge_and_summary(self) -> None:
        self.assertIn("병합, 요약, 정렬", STRICT_EVENT_ORDER_SYSTEM_PROMPT)
        self.assertIn("모든 이벤트를 그대로 보존", STRICT_EVENT_ORDER_SYSTEM_PROMPT)

    def test_strict_prompt_forbids_factual_repetition_around_answer(self) -> None:
        self.assertIn("서론, 설명, 요약 또는 결론", STRICT_EVENT_ORDER_SYSTEM_PROMPT)
        self.assertIn("factual placeholder를 다시 반복하지 않는다", STRICT_EVENT_ORDER_SYSTEM_PROMPT)

    def test_default_prompt_and_production_prompt_are_unchanged(self) -> None:
        prepared, _, _ = _prepared(events=4)
        default_transport = _Transport()
        strict_transport = _Transport()
        production_prompt_before = LOSSLESS_VERBALIZER_SYSTEM_PROMPT

        run_prepared_question(
            prepared,
            transport=default_transport,
            settings=_settings(),
            repeat_index=1,
        )
        run_prepared_question(
            prepared,
            transport=strict_transport,
            settings=_settings(),
            repeat_index=1,
            strict_event_order=True,
        )

        self.assertEqual(
            default_transport.calls[0]["payload"]["messages"][0]["content"],
            EXPERIMENT_SYSTEM_PROMPT,
        )
        self.assertEqual(
            strict_transport.calls[0]["payload"]["messages"][0]["content"],
            STRICT_EVENT_ORDER_SYSTEM_PROMPT,
        )
        self.assertEqual(
            LOSSLESS_VERBALIZER_SYSTEM_PROMPT, production_prompt_before
        )
        self.assertNotEqual(
            STRICT_EVENT_ORDER_SYSTEM_PROMPT, LOSSLESS_VERBALIZER_SYSTEM_PROMPT
        )

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
        self.assertEqual(
            record["expected_field_placeholders"],
            record["found_field_placeholders"],
        )

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
        self.assertEqual(summary["by_question"]["HX07"]["runs"], 2)
        self.assertEqual(summary["by_question"]["HX07"]["successes"], 1)
        self.assertEqual(
            summary["by_question"]["HX07"]["failure_reasons"],
            {"placeholder_missing": 1},
        )
        self.assertEqual(
            summary["by_question"]["HX07"]["expected_placeholder_count"],
            8,
        )
        self.assertEqual(
            summary["by_question"]["HX07"]["found_placeholder_counts"],
            [8, 8],
        )
        self.assertFalse(summary["all_questions_clean"])


if __name__ == "__main__":
    unittest.main()
