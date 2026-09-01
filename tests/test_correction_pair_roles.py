import unittest

from app.generation.compact_claim import build_compact_claim
from app.reasoning.answer_composer import compose_holding_answer
from app.reasoning.correction_pair_roles import (
    ROLE_AFTER,
    ROLE_BEFORE,
    apply_correction_pair,
    bind_correction_pair,
)
from app.reasoning.evidence_builder import EvidenceGroup, EvidenceItem, EvidenceSet
from app.reasoning.holding_event_resolver import resolve_holding_events


REPORTER = "국민연금기금"
REFERENCE_DATE = "2025-05-20"


def _item(
    chunk_id: str,
    doc_id: str,
    *,
    rank: int,
    fields: dict,
    rcept_dt: str,
) -> EvidenceItem:
    labels = {
        "reporter": "보고자/보유자",
        "reference_date": "기준일/보고일",
        "before_shares": "직전 보유주식수",
        "change_shares": "증감주식수",
        "after_shares": "보유주식수",
        "before_ratio": "직전 보유비율",
        "after_ratio": "보유비율",
        "change_ratio": "증감비율",
    }
    projection_fields = {
        labels[key]: value for key, value in fields.items() if key in labels
    }
    ref = {"table_id": f"{doc_id}-t1", "row_start": 1, "row_end": 1}
    field_refs = {label: [ref] for label in projection_fields}
    holding = {
        **fields,
        "projection_type": "holding_detail_row",
        "projection_fields": projection_fields,
        "projection_field_refs": field_refs,
    }
    return EvidenceItem(
        chunk_id=chunk_id,
        doc_id=doc_id,
        company_id="00123456",
        corp_code="00123456",
        corp_name="테스트회사",
        doc_group="holding",
        chunk_type="table_projection",
        section_path=("보유주식등의 수 및 보유비율",),
        evidence_text="holding evidence",
        retrieval_rank=rank,
        retrieval_score=1.0 - rank / 100,
        rcept_dt=rcept_dt,
        report_nm="주식등의대량보유상황보고서",
        period={},
        source_refs=(ref,),
        provenance={
            "source_chunk_id": chunk_id,
            "source_doc_id": doc_id,
            "table_id": ref["table_id"],
            "source_refs": [ref],
            "projection_field_refs": field_refs,
            "source_chunk": {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "projection_type": "holding_detail_row",
                "projection_fields": projection_fields,
                "source_refs": [ref],
            },
        },
        holding=holding,
        temporal_match=None,
    )


def _group(item: EvidenceItem) -> EvidenceGroup:
    return EvidenceGroup(
        group_id=f"holding_event:{item.doc_id}",
        group_type="holding_event",
        member_chunk_ids=(item.chunk_id,),
        primary_evidence=item,
        supporting_evidence=(),
        doc_ids=(item.doc_id,),
        reason="fixture holding event",
    )


def _evidence_set(
    items: list[EvidenceItem],
    *,
    question: str,
    metric: str,
    correction_intent: str | None = "history",
) -> EvidenceSet:
    plan = {
        "task_type": "holding_change",
        "metric": metric,
        "reporter": REPORTER,
        "raw_query": question,
        "period": {"period_type": "latest_holding"},
        "evidence": {
            "requested_holding_fields": [],
            **({"correction_intent": correction_intent} if correction_intent else {}),
        },
    }
    return EvidenceSet(
        question=question,
        query_plan=plan,
        task_type="holding_change",
        evidence_groups=tuple(_group(item) for item in items),
        retrieval_order=tuple(item.chunk_id for item in items),
        raw_candidate_count=len(items),
        selected_evidence_count=len(items),
        warnings=(),
        ambiguity={
            "temporal_ambiguity": False,
            "temporal_constraint": {"explicit": False},
        },
    )


def _trace(
    *,
    root: str = "docA",
    latest: str = "docB",
    group_id: str = "group-1",
    group_count: int = 1,
) -> dict:
    return {
        "correction_intent": "history",
        "correction_group_count": group_count,
        "correction_group_id": group_id,
        "correction_root_doc_id": root,
        "correction_latest_doc_id": latest,
    }


class CorrectionPairRoleBindingTest(unittest.TestCase):
    """T2-B: a corrected report's two versions carry a before and an after role.

    The two filings state one event, so the resolver reports them as two
    indistinguishable events.  Which is "정정 전" and which is "정정 후" is a fact
    about the documents, and only the correction graph knows it.
    """

    QUESTION = "정정 전과 정정 후 보유주식수가 각각 얼마야?"

    def pair(
        self,
        *,
        original: dict,
        corrected: dict,
        question: str | None = None,
        metric: str = "holding_shares",
        correction_intent: str | None = "history",
        trace: dict | None = None,
        original_doc: str = "docA",
        corrected_doc: str = "docB",
    ):
        question = question or self.QUESTION
        evidence = _evidence_set(
            [
                _item("c1", original_doc, rank=1, fields=original, rcept_dt="2025-05-21"),
                _item("c2", corrected_doc, rank=2, fields=corrected, rcept_dt="2025-05-22"),
            ],
            question=question,
            metric=metric,
            correction_intent=correction_intent,
        )
        resolution = resolve_holding_events(evidence, query_plan=evidence.query_plan)
        claim = bind_correction_pair(
            resolution,
            correction_trace=_trace() if trace is None else trace,
            query_plan=evidence.query_plan,
        )
        return evidence, resolution, claim

    # ------------------------------------------------------------------ A

    def test_a_shares_pair_binds_the_requested_field_of_each_version(self) -> None:
        """The document-version axis, not the event-field axis.

        Both filings report a change from 100.  "정정 전" is the *original
        filing's* answer to the question, which is its 120 -- never the 100 that
        is the holding before the transaction both filings describe.
        """

        _evidence, resolution, claim = self.pair(
            original={
                "reporter": REPORTER,
                "reference_date": REFERENCE_DATE,
                "before_shares": "100",
                "after_shares": "120",
            },
            corrected={
                "reporter": REPORTER,
                "reference_date": REFERENCE_DATE,
                "before_shares": "100",
                "after_shares": "125",
            },
        )

        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual(claim.fields, ("after_shares",))
        self.assertEqual(claim.before_event.after_shares.normalized, 120)
        self.assertEqual(claim.after_event.after_shares.normalized, 125)
        # The event-field before must never be read as the correction before.
        self.assertEqual(claim.before_event.before_shares.normalized, 100)
        self.assertNotEqual(claim.before_event.after_shares.normalized, 100)

        bound = apply_correction_pair(resolution, claim)
        roles = [event.correction_role for event in bound.events]
        self.assertEqual(roles, [ROLE_BEFORE, ROLE_AFTER])

    # ------------------------------------------------------------------ B

    def test_b_ratio_pair_binds_the_ratio_of_each_version(self) -> None:
        _evidence, _resolution, claim = self.pair(
            original={
                "reporter": REPORTER,
                "reference_date": REFERENCE_DATE,
                "after_ratio": "32.14",
            },
            corrected={
                "reporter": REPORTER,
                "reference_date": REFERENCE_DATE,
                "after_ratio": "32.16",
            },
            question="정정 전과 정정 후 보유비율이 각각 얼마야?",
            metric="holding_ratio",
        )

        self.assertIsNotNone(claim)
        assert claim is not None
        self.assertEqual(claim.fields, ("after_ratio",))
        self.assertEqual(float(claim.before_event.after_ratio.normalized), 32.14)
        self.assertEqual(float(claim.after_event.after_ratio.normalized), 32.16)

    # ------------------------------------------------------------------ C

    def test_c_each_role_keeps_its_own_filings_citation(self) -> None:
        """The before value cites the original filing, the after the correction."""

        evidence, resolution, claim = self.pair(
            original={
                "reporter": REPORTER,
                "reference_date": REFERENCE_DATE,
                "before_shares": "100",
                "after_shares": "120",
            },
            corrected={
                "reporter": REPORTER,
                "reference_date": REFERENCE_DATE,
                "before_shares": "100",
                "after_shares": "125",
            },
        )
        assert claim is not None
        draft = compose_holding_answer(apply_correction_pair(resolution, claim), evidence)
        compact = build_compact_claim(
            draft, apply_correction_pair(resolution, claim), task_type="holding_event"
        )

        self.assertIsNotNone(compact)
        assert compact is not None
        by_chunk = {citation.chunk_id: citation for citation in compact.citations}
        before, after = compact.fields
        self.assertEqual(before.label, "정정 전 보유주식수")
        self.assertEqual(after.label, "정정 후 보유주식수")
        self.assertEqual(by_chunk[before.chunk_id].doc_id, "docA")
        self.assertEqual(by_chunk[after.chunk_id].doc_id, "docB")
        self.assertNotEqual(before.marker, after.marker)
        self.assertIn("정정 전 보유주식수 120주", compact.deterministic_text)
        self.assertIn("정정 후 보유주식수 125주", compact.deterministic_text)

    # ------------------------------------------------------------------ D

    def test_d_two_events_without_a_correction_relation_are_not_a_pair(self) -> None:
        """Same day, same holder, two filings -- but no chain names them."""

        for trace in (
            None,
            {},
            _trace(group_count=2),
            _trace(root="", latest="docB"),
            _trace(root="docA", latest="docA"),
            # A chain whose endpoints are not the served filings.
            _trace(root="docX", latest="docY"),
        ):
            with self.subTest(trace=trace):
                _evidence, _resolution, claim = self.pair(
                    original={
                        "reporter": REPORTER,
                        "reference_date": REFERENCE_DATE,
                        "after_shares": "120",
                    },
                    corrected={
                        "reporter": REPORTER,
                        "reference_date": REFERENCE_DATE,
                        "after_shares": "125",
                    },
                    trace=trace if trace is not None else {},
                )
                self.assertIsNone(claim)

    def test_d2_a_different_issuer_or_reference_date_is_not_a_pair(self) -> None:
        _evidence, _resolution, claim = self.pair(
            original={
                "reporter": REPORTER,
                "reference_date": REFERENCE_DATE,
                "after_shares": "120",
            },
            corrected={
                "reporter": REPORTER,
                "reference_date": "2024-01-02",
                "after_shares": "125",
            },
        )

        self.assertIsNone(claim)

    # ------------------------------------------------------------------ E

    def test_e_a_non_history_question_never_takes_the_pair_path(self) -> None:
        for intent in (None, "latest", "original"):
            with self.subTest(intent=intent):
                _evidence, _resolution, claim = self.pair(
                    original={
                        "reporter": REPORTER,
                        "reference_date": REFERENCE_DATE,
                        "after_shares": "120",
                    },
                    corrected={
                        "reporter": REPORTER,
                        "reference_date": REFERENCE_DATE,
                        "after_shares": "125",
                    },
                    correction_intent=intent,
                )
                self.assertIsNone(claim)

    # ------------------------------------------------------------------ F

    def test_f_an_incomplete_pair_stays_fail_closed(self) -> None:
        """One side missing the requested metric is not half an answer."""

        _evidence, _resolution, claim = self.pair(
            original={
                "reporter": REPORTER,
                "reference_date": REFERENCE_DATE,
                "after_shares": "120",
            },
            corrected={
                "reporter": REPORTER,
                "reference_date": REFERENCE_DATE,
                "before_shares": "100",
            },
        )

        self.assertIsNone(claim)

    def test_f2_a_lone_version_is_not_a_pair(self) -> None:
        evidence = _evidence_set(
            [
                _item(
                    "c1",
                    "docA",
                    rank=1,
                    fields={
                        "reporter": REPORTER,
                        "reference_date": REFERENCE_DATE,
                        "after_shares": "120",
                    },
                    rcept_dt="2025-05-21",
                )
            ],
            question=self.QUESTION,
            metric="holding_shares",
        )
        resolution = resolve_holding_events(evidence, query_plan=evidence.query_plan)

        self.assertIsNone(
            bind_correction_pair(
                resolution,
                correction_trace=_trace(),
                query_plan=evidence.query_plan,
            )
        )

    # ------------------------------------------------------- unrelated rows

    def test_an_unbound_event_keeps_its_frozen_shape_and_label(self) -> None:
        """No role, no new key, and the frozen label is unchanged."""

        evidence = _evidence_set(
            [
                _item(
                    "c1",
                    "docA",
                    rank=1,
                    fields={
                        "reporter": REPORTER,
                        "reference_date": REFERENCE_DATE,
                        "after_shares": "120",
                    },
                    rcept_dt="2025-05-21",
                )
            ],
            question="국민연금 보유주식수는?",
            metric="holding_shares",
            correction_intent=None,
        )
        resolution = resolve_holding_events(evidence, query_plan=evidence.query_plan)
        draft = compose_holding_answer(resolution, evidence)
        row = draft.answer_sections[0].content["events"][0]

        self.assertNotIn("correction_role", row)
        self.assertIsNone(resolution.events[0].correction_role)
        compact = build_compact_claim(draft, resolution, task_type="holding_event")
        assert compact is not None
        self.assertEqual(compact.fields[0].label, "변동 후 주식수")


class CorrectionPairLocalProbeTest(unittest.TestCase):
    """The C001/C002/C007 shapes, as structure rather than as those questions.

    Only the numeric shape of each failure is reproduced -- two versions of one
    filing differing in the requested metric.  No company, date, or document id
    from any benchmark question appears here, and none is read by the code.

    Values are written the way a filing writes them, because the renderer
    reproduces a number verbatim rather than reformatting it.
    """

    def probe(self, *, metric: str, question: str, original: str, corrected: str):
        field = "after_ratio" if metric == "holding_ratio" else "after_shares"
        evidence = _evidence_set(
            [
                _item(
                    "c1",
                    "docA",
                    rank=1,
                    fields={
                        "reporter": REPORTER,
                        "reference_date": REFERENCE_DATE,
                        field: original,
                    },
                    rcept_dt="2025-05-21",
                ),
                _item(
                    "c2",
                    "docB",
                    rank=2,
                    fields={
                        "reporter": REPORTER,
                        "reference_date": REFERENCE_DATE,
                        field: corrected,
                    },
                    rcept_dt="2025-05-22",
                ),
            ],
            question=question,
            metric=metric,
        )
        resolution = resolve_holding_events(evidence, query_plan=evidence.query_plan)
        claim = bind_correction_pair(
            resolution, correction_trace=_trace(), query_plan=evidence.query_plan
        )
        assert claim is not None
        bound = apply_correction_pair(resolution, claim)
        draft = compose_holding_answer(bound, evidence)
        compact = build_compact_claim(draft, bound, task_type="holding_event")
        assert compact is not None
        return compact

    def test_shares_pair_probe(self) -> None:
        compact = self.probe(
            metric="holding_shares",
            question="정정 전과 정정 후 보유주식수가 각각 얼마야?",
            original="183,202,431",
            corrected="183,323,596",
        )

        self.assertIn("정정 전 보유주식수 183,202,431주", compact.deterministic_text)
        self.assertIn("정정 후 보유주식수 183,323,596주", compact.deterministic_text)

    def test_ratio_pair_probe(self) -> None:
        compact = self.probe(
            metric="holding_ratio",
            question="정정 전과 정정 후 보유비율이 각각 얼마야?",
            original="32.14",
            corrected="32.16",
        )

        self.assertIn("정정 전 보유비율 32.14%", compact.deterministic_text)
        self.assertIn("정정 후 보유비율 32.16%", compact.deterministic_text)

    def test_second_shares_pair_probe(self) -> None:
        compact = self.probe(
            metric="holding_shares",
            question="정정 전후 보유주식수는 각각 몇 주야?",
            original="49,222,116",
            corrected="49,223,530",
        )

        self.assertIn("정정 전 보유주식수 49,222,116주", compact.deterministic_text)
        self.assertIn("정정 후 보유주식수 49,223,530주", compact.deterministic_text)


if __name__ == "__main__":
    unittest.main()
