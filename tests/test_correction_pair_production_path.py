"""T2-B.1: the two production-path mismatches T2-B's role binding fell through.

The verbalizer is handed a compact claim only when that claim covers exactly one
event.  A correction pair covers two by construction, so every pair question
takes the deterministic multi-event renderer instead -- which is where the role
has to be stated, and where T2-B did not state it.

The second mismatch is upstream: one execution can expand more than one
correction chain, and the trace named only the first chain's endpoints, so the
binder declined rather than risk attributing a filing to the wrong chain.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from app.agent.orchestrator import AgentOrchestrator
from app.generation.answer_generator import CitationAwareAnswerGenerator
from app.reasoning.correction_pair_roles import (
    REASON_BOUND,
    REASON_EVENT_COUNT_MISMATCH,
    REASON_FIELD_MISSING,
    REASON_INTENT_NOT_HISTORY,
    REASON_MULTIPLE_MATCHING_GROUPS,
    REASON_NO_MATCHING_GROUP,
    decide_correction_pair,
)
from app.reasoning.query_plan import QueryPeriod, QueryPlan
from tests.test_evidence_builder import _candidate


REPORTER = "국민연금기금"


def _pair(doc_id: str, *, rank: int, date: str, shares=None, ratio=None):
    fields = {"보고자/보유자": REPORTER, "기준일/보고일": date}
    if shares is not None:
        fields["보유주식수"] = shares
    if ratio is not None:
        fields["보유비율"] = ratio
    table_id = f"t{rank}"
    return _candidate(
        f"{doc_id}:ch",
        doc_id,
        rank=rank,
        doc_group="holding",
        content=f"{REPORTER} {date} 보유",
        section="보유주식등의 수 및 보유비율",
        chunk_type="table_projection",
        table_id=table_id,
        projection_type="holding_report",
        source_table_ids=[],
        source_refs=[{"table_id": table_id, "row_start": rank, "row_end": rank}],
        projection_fields=fields,
    )


def _group(group_id: str, root: str, latest: str) -> dict:
    return {
        "correction_group_id": group_id,
        "root_doc_id": root,
        "latest_doc_id": latest,
    }


def _trace(*groups: dict) -> dict:
    """The expansion trace, written the way the expander writes it."""

    first = groups[0] if groups else {}
    return {
        "correction_intent": "history",
        "correction_expanded": True,
        "correction_group_count": len(groups),
        "correction_group_id": first.get("correction_group_id"),
        "correction_root_doc_id": first.get("root_doc_id"),
        "correction_latest_doc_id": first.get("latest_doc_id"),
        "correction_groups": list(groups),
    }


def _run(
    *,
    question: str,
    metric: str,
    docs,
    trace: dict,
    correction_intent: str | None = "history",
    period: QueryPeriod | None = None,
):
    """Run the production reasoning path and return what the server would serve."""

    evidence = {"requested_holding_fields": []}
    if correction_intent:
        evidence["correction_intent"] = correction_intent
    plan = QueryPlan(
        query=question,
        task_type="holding_change",
        metric=metric,
        reporter=REPORTER,
        disclosure_route=("holding",),
        period=period,
        evidence=evidence,
    )
    kind = "shares" if metric == "holding_shares" else "ratio"
    pairs = [
        _pair(doc_id, rank=index + 1, date=date, **{kind: value})
        for index, (doc_id, date, value) in enumerate(docs)
    ]
    execution = SimpleNamespace(
        plan=plan,
        chunks=[pair[0] for pair in pairs],
        results=[pair[1] for pair in pairs],
        correction_expansion=trace,
    )
    result = AgentOrchestrator().run(plan.raw_query, plan, execution)
    answer = CitationAwareAnswerGenerator().generate(result.answer_draft).answer_text
    return result, answer


class DeterministicRoleRenderingTest(unittest.TestCase):
    """A: the answer the server actually serves states the roles.

    The verbalizer skips every multi-event claim, so these assertions are made
    against the deterministic renderer -- the path a pair question always takes.
    """

    ONE_CHAIN = _trace(_group("grp-1", "docROOT", "docFINAL"))

    def test_a_shares_pair_states_both_roles_deterministically(self) -> None:
        result, answer = _run(
            question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
            metric="holding_shares",
            docs=[("docROOT", "2025-05-20", "120"), ("docFINAL", "2025-05-20", "125")],
            trace=self.ONE_CHAIN,
        )

        self.assertIn("correction_pair_roles", result.execution_trace)
        self.assertIn("정정 전 보유주식수 120주", answer)
        self.assertIn("정정 후 보유주식수 125주", answer)

    def test_b_ratio_pair_states_both_roles_deterministically(self) -> None:
        _result, answer = _run(
            question="정정 전과 정정 후 보유비율이 각각 얼마야?",
            metric="holding_ratio",
            docs=[("docROOT", "2025-05-20", "32.14"), ("docFINAL", "2025-05-20", "32.16")],
            trace=self.ONE_CHAIN,
        )

        self.assertIn("정정 전 보유비율 32.14%", answer)
        self.assertIn("정정 후 보유비율 32.16%", answer)

    def test_the_role_rides_with_its_own_filings_citation(self) -> None:
        """Each role keeps the citation of the filing that states it."""

        _result, answer = _run(
            question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
            metric="holding_shares",
            docs=[("docROOT", "2025-05-20", "120"), ("docFINAL", "2025-05-20", "125")],
            trace=self.ONE_CHAIN,
        )
        before = next(line for line in answer.splitlines() if "정정 전" in line)
        after = next(line for line in answer.splitlines() if "정정 후" in line)

        self.assertIn("[1]", before)
        self.assertIn("[2]", after)
        self.assertIn("doc_id: docROOT", answer)
        self.assertIn("doc_id: docFINAL", answer)


class UnboundRenderingUnchangedTest(unittest.TestCase):
    """C: an event with no role renders exactly as it always did."""

    def test_c_multi_event_rows_without_a_role_are_unchanged(self) -> None:
        _result, answer = _run(
            question="보유주식수는?",
            metric="holding_shares",
            docs=[("docA", "2024-07-05", "120"), ("docB", "2025-03-24", "125")],
            trace={},
            correction_intent=None,
        )

        self.assertNotIn("정정 전", answer)
        self.assertNotIn("정정 후", answer)
        self.assertIn("2024-07-05 | 국민연금기금 | 120주", answer)
        self.assertIn("2025-03-24 | 국민연금기금 | 125주", answer)

    def test_a_single_event_answer_is_untouched(self) -> None:
        _result, answer = _run(
            question="보유주식수는?",
            metric="holding_shares",
            docs=[("docA", "2024-07-05", "120")],
            trace={},
            correction_intent=None,
        )

        self.assertNotIn("정정 전", answer)
        self.assertNotIn("정정 후", answer)


class MultipleCorrectionGroupsTest(unittest.TestCase):
    """D/E/F: which chain a pair question is about, when several are served.

    Two reference events of one reporter, each independently corrected -- the
    shape the C001 execution actually retrieved.
    """

    CHAIN_A = ("docA_ROOT", "docA_FINAL", "2024-07-05")
    CHAIN_B = ("docB_ROOT", "docB_FINAL", "2025-03-24")
    TWO_CHAINS = _trace(
        _group("grp-A", CHAIN_A[0], CHAIN_A[1]),
        _group("grp-B", CHAIN_B[0], CHAIN_B[1]),
    )
    DOCS = [
        (CHAIN_A[0], CHAIN_A[2], "183,202,431"),
        (CHAIN_A[1], CHAIN_A[2], "183,323,596"),
        (CHAIN_B[0], CHAIN_B[2], "200,000,000"),
        (CHAIN_B[1], CHAIN_B[2], "200,111,111"),
    ]

    def test_d_the_chain_the_question_matched_is_the_one_bound(self) -> None:
        """Only one chain's filings satisfy the question, so only it is bound."""

        result, answer = _run(
            question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
            metric="holding_shares",
            docs=self.DOCS,
            trace=self.TWO_CHAINS,
            period=QueryPeriod(
                period_type="reference_date",
                from_date=self.CHAIN_A[2],
                to_date=self.CHAIN_A[2],
            ),
        )

        self.assertIn("correction_pair_roles", result.execution_trace)
        self.assertEqual(result.correction_pair["reason"], REASON_BOUND)
        self.assertEqual(result.correction_pair["correction_group_id"], "grp-A")
        self.assertEqual(result.correction_pair["before_doc_id"], self.CHAIN_A[0])
        self.assertEqual(result.correction_pair["after_doc_id"], self.CHAIN_A[1])
        self.assertIn("정정 전 보유주식수 183,202,431주", answer)
        self.assertIn("정정 후 보유주식수 183,323,596주", answer)
        # The other chain is not relabelled and not dropped.
        self.assertNotIn("정정 전 보유주식수 200,000,000주", answer)

    def test_e_two_equally_matching_chains_bind_nothing(self) -> None:
        """Nothing singles one chain out, so the frozen behaviour is kept."""

        result, answer = _run(
            question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
            metric="holding_shares",
            docs=self.DOCS,
            trace=self.TWO_CHAINS,
        )

        self.assertNotIn("correction_pair_roles", result.execution_trace)
        self.assertEqual(
            result.correction_pair["reason"], REASON_MULTIPLE_MATCHING_GROUPS
        )
        self.assertNotIn("정정 전", answer)
        self.assertNotIn("정정 후", answer)

    def test_f_a_chain_for_another_reference_event_is_never_the_pair(self) -> None:
        """Only the unrelated chain is served; its filings are not the answer."""

        result, _answer = _run(
            question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
            metric="holding_shares",
            docs=[
                (self.CHAIN_A[0], self.CHAIN_A[2], "183,202,431"),
                (self.CHAIN_A[1], self.CHAIN_A[2], "183,323,596"),
            ],
            trace=_trace(_group("grp-B", self.CHAIN_B[0], self.CHAIN_B[1])),
        )

        self.assertNotIn("correction_pair_roles", result.execution_trace)
        self.assertFalse(result.correction_pair["bound"])


class PairDiagnosticTest(unittest.TestCase):
    """The decline reason is deterministic and changes no behaviour."""

    def _decide(self, **kwargs):
        result, _answer = _run(**kwargs)
        return result

    def test_every_decline_reports_a_reason(self) -> None:
        cases = (
            (
                REASON_INTENT_NOT_HISTORY,
                {"correction_intent": None, "trace": _trace(_group("g", "d1", "d2"))},
            ),
            (REASON_NO_MATCHING_GROUP, {"trace": {}}),
            (
                REASON_EVENT_COUNT_MISMATCH,
                {"trace": _trace(_group("g", "docROOT", "docFINAL"))},
            ),
        )
        for expected, overrides in cases:
            with self.subTest(reason=expected):
                docs = overrides.pop(
                    "docs",
                    [("docROOT", "2025-05-20", "120")]
                    if expected == REASON_EVENT_COUNT_MISMATCH
                    else [
                        ("docROOT", "2025-05-20", "120"),
                        ("docFINAL", "2025-05-20", "125"),
                    ],
                )
                result = self._decide(
                    question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
                    metric="holding_shares",
                    docs=docs,
                    **overrides,
                )
                self.assertEqual(result.correction_pair["reason"], expected)

    def test_a_missing_metric_on_one_side_reports_field_missing(self) -> None:
        plan_docs = [("docROOT", "2025-05-20", "120"), ("docFINAL", "2025-05-20", None)]
        result, _answer = _run(
            question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
            metric="holding_shares",
            docs=plan_docs,
            trace=_trace(_group("grp-1", "docROOT", "docFINAL")),
        )

        self.assertFalse(result.correction_pair["bound"])
        self.assertIn(
            result.correction_pair["reason"],
            {REASON_FIELD_MISSING, REASON_EVENT_COUNT_MISMATCH},
        )

    def test_the_diagnostic_never_changes_the_answer(self) -> None:
        """Reading the reason is the only thing the diagnostic is for."""

        result, answer = _run(
            question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
            metric="holding_shares",
            docs=[("docROOT", "2025-05-20", "120"), ("docFINAL", "2025-05-20", "125")],
            trace=_trace(_group("grp-1", "docROOT", "docFINAL")),
        )

        self.assertTrue(result.correction_pair["bound"])
        self.assertIn("정정 전 보유주식수 120주", answer)


class LegacyTraceCompatibilityTest(unittest.TestCase):
    """A trace written before per-group endpoints existed still binds."""

    def test_a_flat_single_group_trace_is_still_read(self) -> None:
        result, answer = _run(
            question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
            metric="holding_shares",
            docs=[("docROOT", "2025-05-20", "120"), ("docFINAL", "2025-05-20", "125")],
            trace={
                "correction_intent": "history",
                "correction_group_count": 1,
                "correction_group_id": "grp-1",
                "correction_root_doc_id": "docROOT",
                "correction_latest_doc_id": "docFINAL",
            },
        )

        self.assertIn("correction_pair_roles", result.execution_trace)
        self.assertIn("정정 전 보유주식수 120주", answer)


class ExpansionTracePerGroupTest(unittest.TestCase):
    """The expander records every chain's endpoints, not only the first."""

    def test_each_expanded_group_keeps_its_own_endpoints(self) -> None:
        from app.retrieval.correction_expansion import _trace as expansion_trace

        groups = [
            {"correction_group_id": "grp-A", "root_doc_id": "a1", "latest_doc_id": "a2"},
            {"correction_group_id": "grp-B", "root_doc_id": "b1", "latest_doc_id": "b2"},
        ]
        trace = expansion_trace("history", "expanded", groups=groups)

        self.assertEqual(trace["correction_group_count"], 2)
        # The pre-existing flat keys still describe the first group.
        self.assertEqual(trace["correction_root_doc_id"], "a1")
        self.assertEqual(trace["correction_latest_doc_id"], "a2")
        self.assertEqual(
            [group["correction_group_id"] for group in trace["correction_groups"]],
            ["grp-A", "grp-B"],
        )
        self.assertEqual(trace["correction_groups"][1]["root_doc_id"], "b1")


if __name__ == "__main__":
    unittest.main()
