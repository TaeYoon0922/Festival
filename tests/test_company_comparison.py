"""One field compared across N companies, each answered from its own scope."""

import unittest
from types import SimpleNamespace

from app.reasoning.company_comparison import (
    CompanyComparisonRequest,
    CompanyOperand,
    company_operands,
    executable_comparison,
    operand_subplan,
    comparison_requested,
    compose_comparison_text,
    execute_company_scopes,
    resolve_comparison,
)
from tests.test_scoped_operands import chunk

A, B, C, D = "00000001", "00000002", "00000003", "00000004"
NAMES = {A: "가상항공", B: "가상로템", C: "가상중공업", D: "가상디펜스"}


def request(*codes, ordered=False, clauses=None, dates=None):
    clauses = clauses or {}
    dates = dates or {}
    return CompanyComparisonRequest(
        companies=tuple(
            CompanyOperand(
                name=NAMES[code],
                corp_code=code,
                clause=clauses.get(code),
                on_date=dates.get(code),
            )
            for code in codes
        ),
        ordered=ordered,
    )


class RecognitionTest(unittest.TestCase):
    def test_two_companies_and_a_larger_side(self):
        found = comparison_requested(
            "가상항공의 수출계약과 가상로템의 부품 계약 중 계약금액이 더 큰 쪽은?",
            ["가상항공", "가상로템"],
            [A, B],
        )

        self.assertIsNotNone(found)
        self.assertEqual(found.size, 2)
        self.assertFalse(found.ordered)

    def test_four_companies_and_an_ordering(self):
        found = comparison_requested(
            "가상항공, 가상로템, 가상중공업, 가상디펜스가 각각 공시한 계약 중 "
            "계약금액이 큰 순서대로 나열해줘",
            list(NAMES.values()),
            [A, B, C, D],
        )

        self.assertEqual(found.size, 4)
        self.assertTrue(found.ordered)

    def test_a_difference_question_is_a_comparison(self):
        self.assertIsNotNone(
            comparison_requested(
                "가상항공의 계약과 가상로템의 계약 계약금액은 차이가 얼마나 돼?",
                ["가상항공", "가상로템"],
                [A, B],
            )
        )

    # ----------------------------------------------------------- declined
    def test_one_company_is_not_a_comparison(self):
        self.assertIsNone(
            comparison_requested("가상항공 계약금액이 가장 큰 건은?", ["가상항공"], [A])
        )

    def test_companies_without_comparison_intent_stay_ambiguous(self):
        self.assertIsNone(
            comparison_requested(
                "가상항공과 가상로템의 계약금액을 알려줘", ["가상항공", "가상로템"], [A, B]
            )
        )

    def test_a_holding_role_pair_is_not_a_comparison(self):
        """Two companies, one holding question: an issuer and its filer."""

        self.assertIsNone(
            comparison_requested(
                "가상로템이 보유한 가상항공 주식은 몇 주인가?",
                ["가상항공", "가상로템"],
                [A, B],
            )
        )

    def test_a_comparison_of_a_non_comparable_field_is_declined(self):
        self.assertIsNone(
            comparison_requested(
                "가상항공과 가상로템 중 보유주식수가 더 큰 쪽은?",
                ["가상항공", "가상로템"],
                [A, B],
            )
        )

    def test_an_unresolved_company_cannot_own_an_operand(self):
        self.assertIsNone(
            comparison_requested(
                "가상항공과 가상로템 중 계약금액이 더 큰 쪽은?",
                ["가상항공", "가상로템"],
                [A],
            )
        )

    def test_duplicate_codes_are_refused(self):
        self.assertIsNone(
            comparison_requested(
                "가상항공과 가상로템 중 계약금액이 더 큰 쪽은?",
                ["가상항공", "가상로템"],
                [A, A],
            )
        )


class ScopedExecutionTest(unittest.TestCase):
    """Each company retrieved on a plan narrowed to that company alone."""

    def plan(self, **kwargs):
        from app.reasoning.query_plan import QueryPeriod, QueryPlan

        defaults = dict(
            query="원본 질의",
            raw_query="원본 질의",
            companies=("가상항공", "가상로템"),
            corp_codes=(A, B),
            task_type="corporate_event",
            event_type="supply_contract",
            disclosure_route=("exchange",),
            period=QueryPeriod(
                year=2023, from_date="2023-02-28", to_date="2023-02-28",
                period_type="receipt_date",
            ),
        )
        defaults.update(kwargs)
        return QueryPlan(**defaults)

    def test_each_company_is_retrieved_on_its_own_narrowed_plan(self):
        seen = []

        def execute(plan):
            seen.append(plan)
            return SimpleNamespace(
                chunks=(chunk(plan.corp_codes[0], "100", corp_code=plan.corp_codes[0]),)
            )

        req = request(
            A, B,
            clauses={A: "보잉 착륙장치", B: "30mm차륜형대공포"},
            dates={A: "2023-02-28", B: "2023-01-20"},
        )
        execute_company_scopes(req, self.plan(), execute)

        self.assertEqual([p.corp_codes for p in seen], [(A,), (B,)])
        self.assertEqual([p.companies for p in seen], [("가상항공",), ("가상로템",)])

    def test_each_operand_carries_only_its_own_clause(self):
        seen = []

        def execute(plan):
            seen.append(plan.query)
            return SimpleNamespace(chunks=())

        req = request(A, B, clauses={A: "보잉 착륙장치", B: "30mm차륜형대공포"})
        execute_company_scopes(req, self.plan(), execute)

        self.assertEqual(seen, ["보잉 착륙장치 계약금액", "30mm차륜형대공포 계약금액"])

    def test_the_parent_global_date_is_not_inherited(self):
        """A plan dated 2023-02-28 must not ask every company for that day."""

        parent = self.plan()
        self.assertEqual(parent.period.from_date, "2023-02-28")

        req = request(A, B, dates={A: "2023-02-28", B: "2023-01-20"})
        first = operand_subplan(parent, req.companies[0])
        second = operand_subplan(parent, req.companies[1])

        self.assertEqual(first.period.from_date, "2023-02-28")
        self.assertEqual(second.period.from_date, "2023-01-20")
        self.assertEqual(second.period.to_date, "2023-01-20")

    def test_an_undated_operand_gets_an_empty_period(self):
        parent = self.plan()
        undated = operand_subplan(parent, request(A).companies[0])

        self.assertIsNone(undated.period.from_date)
        self.assertIsNone(undated.period.to_date)
        self.assertEqual(undated.years, ())

    def test_one_companys_failure_does_not_borrow_another(self):
        def execute(plan):
            if plan.corp_codes[0] == B:
                raise RuntimeError("retrieval failed")
            return SimpleNamespace(chunks=(chunk("a", "100", corp_code=A),))

        per_company = execute_company_scopes(request(A, B), self.plan(), execute)

        self.assertEqual(per_company[B], ())
        self.assertIsNone(resolve_comparison(request(A, B), per_company))


class ResolutionTest(unittest.TestCase):
    def per_company(self, *pairs):
        return {
            code: (chunk(code, amount, corp_code=code),) for code, amount in pairs
        }

    def test_two_companies_rank_by_their_own_amounts(self):
        order = resolve_comparison(
            request(A, B), self.per_company((A, "1,195,242,120,000"), (B, "300"))
        )

        self.assertEqual(order.largest.scope.company, NAMES[A])
        self.assertEqual(order.spread, 1195242119700)

    def test_four_companies_order_completely(self):
        order = resolve_comparison(
            request(A, B, C, D),
            self.per_company((A, "10"), (B, "40"), (C, "30"), (D, "20")),
        )

        self.assertEqual(
            [operand.scope.company for operand in order.operands],
            [NAMES[B], NAMES[C], NAMES[D], NAMES[A]],
        )

    def test_a_company_with_no_evidence_declines_the_whole_ranking(self):
        per_company = self.per_company((A, "10"), (B, "40"))
        per_company[C] = ()

        self.assertIsNone(resolve_comparison(request(A, B, C), per_company))

    def test_one_companys_evidence_never_satisfies_another_scope(self):
        """B's slice is empty; A's document must not fill it."""

        per_company = {A: (chunk("a", "10", corp_code=A),), B: ()}

        self.assertIsNone(resolve_comparison(request(A, B), per_company))

    def test_a_foreign_document_in_a_slice_is_ignored(self):
        """A chunk carrying another company's code cannot answer this scope."""

        per_company = {
            A: (chunk("a", "10", corp_code=A),),
            B: (chunk("stray", "40", corp_code=A),),
        }

        self.assertIsNone(resolve_comparison(request(A, B), per_company))

    def test_every_ranked_member_keeps_its_own_document(self):
        order = resolve_comparison(request(A, B), self.per_company((A, "10"), (B, "40")))
        docs = {o.scope.company: o.source.doc_id for o in order.operands}

        self.assertEqual(docs, {NAMES[A]: A, NAMES[B]: B})


class CompositionTest(unittest.TestCase):
    def per_company(self, *pairs):
        return {code: (chunk(code, amount, corp_code=code),) for code, amount in pairs}

    def test_two_companies_state_the_winner_and_the_difference(self):
        req = request(A, B)
        text = compose_comparison_text(
            req, resolve_comparison(req, self.per_company((A, "300"), (B, "100")))
        )

        self.assertIn(NAMES[A], text)
        self.assertIn("차이", text)
        self.assertIn("200", text)

    def test_an_ordering_lists_every_company(self):
        req = request(A, B, C, D, ordered=True)
        text = compose_comparison_text(
            req,
            resolve_comparison(
                req, self.per_company((A, "10"), (B, "40"), (C, "30"), (D, "20"))
            ),
        )

        for name in NAMES.values():
            self.assertIn(name, text)
        self.assertIn(">", text)


if __name__ == "__main__":
    unittest.main()


class OperandBindingTest(unittest.TestCase):
    """Order and clauses come from the question, never from the resolver."""

    def test_mention_order_is_restored_from_the_question(self):
        """The resolver hands them over sorted; the question decides."""

        operands = company_operands(
            "가상중공업의 LNGC 3척 계약과 가상일렉트릭의 변압기 계약 계약금액은 차이가 얼마나 돼?",
            [("가상일렉트릭", B), ("가상중공업", A)],
        )

        self.assertEqual([o.name for o in operands], ["가상중공업", "가상일렉트릭"])

    def test_each_company_keeps_its_own_clause(self):
        operands = company_operands(
            "가상항공의 말레이시아 수출계약과 가상로템의 알타이전차 부품 계약 중 계약금액이 더 큰 쪽은?",
            [("가상항공", A), ("가상로템", B)],
        )

        self.assertEqual(
            [o.lexical_query() for o in operands],
            ["말레이시아 수출계약 계약금액", "알타이전차 부품 계약 계약금액"],
        )

    def test_a_spelled_out_list_binds_dates_per_company(self):
        operands = company_operands(
            "가상항공, 가상중공업, 가상로템이 각각 공시한 계약 중 계약금액이 큰 순서대로 나열해줘 "
            "(가상항공 2023-02-28 보잉 착륙장치, 가상중공업 2023-01-20 VLGC 2척, "
            "가상로템 2023-02-28 대공포 2차 양산)",
            [("가상중공업", A), ("가상로템", B), ("가상항공", C)],
        )

        self.assertEqual(
            [(o.name, o.on_date, o.lexical_query()) for o in operands],
            [
                ("가상항공", "2023-02-28", "보잉 착륙장치 계약금액"),
                ("가상중공업", "2023-01-20", "VLGC 2척 계약금액"),
                ("가상로템", "2023-02-28", "대공포 2차 양산 계약금액"),
            ],
        )

    def test_an_abbreviated_list_entry_still_binds(self):
        operands = company_operands(
            "가상항공과 가상디펜스앤에어로가 공시한 계약금액을 순서대로 나열해줘 "
            "(가상항공 2023-02-28 착륙장치, 가상디펜스 2023-04-14 경찰 헬기)",
            [("가상항공", A), ("가상디펜스앤에어로", D)],
        )

        self.assertEqual(
            [(o.name, o.on_date) for o in operands],
            [("가상항공", "2023-02-28"), ("가상디펜스앤에어로", "2023-04-14")],
        )

    def test_a_company_the_question_never_writes_declines(self):
        self.assertIsNone(
            company_operands(
                "가상항공의 계약금액이 더 큰가?", [("가상항공", A), ("이름없음", B)]
            )
        )


class ValidatorActivationTest(unittest.TestCase):
    """Only a complete request is executable; everything else stays ambiguous."""

    def setUp(self):
        from app.reasoning.query_understanding import QueryUnderstanding
        from app.reasoning.query_validation import CorpusScope, QueryValidator

        self.scope = CorpusScope(
            companies={
                "가상항공": ("가상항공", A),
                "가상로템": ("가상로템", B),
            },
            receipt_from="2020-01-01",
            receipt_to="2030-12-31",
        )
        self.understanding = QueryUnderstanding(self.scope.company_aliases())
        self.validator = QueryValidator(corpus_scope=self.scope)

    def result(self, question):
        return self.validator.validate(self.understanding.understand(question))

    def test_a_complete_comparison_is_permitted(self):
        result = self.result(
            "가상항공의 수출계약과 가상로템의 부품 계약 중 계약금액이 더 큰 쪽은?"
        )

        self.assertTrue(result.retrieval_allowed)
        self.assertIsNotNone(executable_comparison(result.plan))

    def test_multiple_companies_without_comparison_stay_ambiguous(self):
        result = self.result("가상항공과 가상로템의 계약금액을 알려줘")

        self.assertFalse(result.retrieval_allowed)
        self.assertIsNone(executable_comparison(result.plan))

    def test_a_comparison_of_another_field_stays_ambiguous(self):
        result = self.result("가상항공과 가상로템 중 보유주식수가 더 큰 쪽은?")

        self.assertFalse(result.retrieval_allowed)
        self.assertIsNone(executable_comparison(result.plan))

    def test_a_holding_role_pair_is_unaffected(self):
        """Two companies in a holding question stay an issuer and a filer."""

        result = self.result("가상로템이 보유한 가상항공 주식은 몇 주인가?")

        self.assertIsNone(executable_comparison(result.plan))

    def test_a_request_whose_companies_left_the_plan_is_not_executable(self):
        from dataclasses import replace

        result = self.result(
            "가상항공의 수출계약과 가상로템의 부품 계약 중 계약금액이 더 큰 쪽은?"
        )
        drifted = replace(result.plan, companies=("가상항공",), corp_codes=(A,))

        self.assertIsNone(executable_comparison(drifted))


class ComparisonDraftTest(unittest.TestCase):
    """The comparison enters the ordinary answer draft, or nothing is answered."""

    def evidence_set(self, chunks):
        from app.reasoning.evidence_builder import (
            EvidenceGroup,
            EvidenceItem,
            EvidenceSet,
        )

        items = [
            EvidenceItem(
                chunk_id=c["chunk_id"], doc_id=c["doc_id"], company_id=None,
                corp_code=c["corp_code"], corp_name=None, doc_group="exchange",
                chunk_type="table", section_path=(), evidence_text="",
                retrieval_rank=index + 1, retrieval_score=1.0,
                rcept_dt=c["rcept_dt"], report_nm=c["report_nm"], period={},
                source_refs=(), provenance={}, holding={},
            )
            for index, c in enumerate(chunks)
        ]
        groups = tuple(
            EvidenceGroup(
                group_id=f"g{index}", group_type="document",
                member_chunk_ids=(item.chunk_id,), primary_evidence=item,
                supporting_evidence=(), doc_ids=(item.doc_id,), reason="test",
            )
            for index, item in enumerate(items)
        )
        return EvidenceSet(
            question="q", query_plan={}, task_type="general_evidence",
            evidence_groups=groups,
            retrieval_order=tuple(item.chunk_id for item in items),
            raw_candidate_count=len(items), selected_evidence_count=len(items),
            warnings=(), ambiguity={},
        )

    def draft(self, per_company, req=None):
        from app.agent.orchestrator import _compose_company_comparison

        req = req or request(A, B)
        order = resolve_comparison(req, per_company)
        chunks = [c for slice_ in per_company.values() for c in slice_]
        return _compose_company_comparison(
            self.evidence_set(chunks), req, order, task_type="general_evidence"
        )

    def slice_for(self, *pairs):
        return {code: (chunk(code, amount, corp_code=code),) for code, amount in pairs}

    def test_every_operand_is_cited(self):
        draft = self.draft(self.slice_for((A, "300"), (B, "100")))

        self.assertTrue(draft.answerable)
        self.assertEqual(len(draft.citations), 2)
        self.assertEqual({citation.doc_id for citation in draft.citations}, {A, B})

    def test_the_comparison_leads_the_answer_with_its_evidence_ids(self):
        draft = self.draft(self.slice_for((A, "300"), (B, "100")))
        section = draft.answer_sections[0]

        self.assertEqual(len(section.supporting_evidence_ids), 2)
        self.assertIn(NAMES[A], section.content["summary"])
        self.assertEqual(len(draft.evidence_references) >= 2, True)

    def test_four_companies_are_all_cited_in_order(self):
        req = request(A, B, C, D, ordered=True)
        draft = self.draft(
            self.slice_for((A, "10"), (B, "40"), (C, "30"), (D, "20")), req=req
        )
        summary = draft.answer_sections[0].content["summary"]

        self.assertTrue(draft.answerable)
        self.assertEqual(len(draft.citations), 4)
        self.assertLess(summary.index(NAMES[B]), summary.index(NAMES[C]))
        self.assertLess(summary.index(NAMES[C]), summary.index(NAMES[D]))
        self.assertLess(summary.index(NAMES[D]), summary.index(NAMES[A]))

    def test_a_missing_operand_is_not_answerable(self):
        per_company = self.slice_for((A, "300"))
        per_company[B] = ()

        draft = self.draft(per_company)

        self.assertFalse(draft.answerable)
        self.assertIn("company_comparison_incomplete", draft.warnings)


class CandidateChunkMergeTest(unittest.TestCase):
    """Real retrieval yields CandidateChunk dataclasses, not mappings."""

    def candidate(self, code, amount):
        from app.retrieval.interfaces import CandidateChunk, MetadataMatch

        payload = chunk(code, amount, corp_code=code)
        return CandidateChunk(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            chunk=payload,
            metadata_match=MetadataMatch(),
        )

    def pipeline(self, executions):
        from app.api.pipeline import AnswerPipeline

        return AnswerPipeline(
            understanding=SimpleNamespace(),
            executor=SimpleNamespace(execute=executions),
        )

    def plan(self):
        from app.reasoning.query_plan import QueryPlan

        return QueryPlan(
            query="원본 질의",
            raw_query="원본 질의",
            companies=("가상항공", "가상로템"),
            corp_codes=(A, B),
            task_type="corporate_event",
            disclosure_route=("exchange",),
        )

    def merged(self):
        from app.retrieval.interfaces import RetrievalResult

        def execute(plan):
            code = plan.corp_codes[0]
            candidate = self.candidate(code, "100" if code == A else "300")
            return SimpleNamespace(
                documents=(),
                chunks=(candidate,),
                results=(
                    RetrievalResult(
                        chunk_id=candidate.chunk_id, doc_id=candidate.doc_id,
                        bm25_score=1.0, rank=1, metadata_match={},
                    ),
                ),
            )

        return self.pipeline(execute)._comparison_execution(
            request(A, B), self.plan()
        )

    def test_merge_accepts_candidate_chunks(self):
        execution = self.merged()

        self.assertEqual(len(execution.chunks), 2)

    def test_merge_preserves_the_candidate_object(self):
        """Downstream reads CandidateChunk attributes; flattening breaks them."""

        from app.retrieval.interfaces import CandidateChunk

        for candidate in self.merged().chunks:
            self.assertIsInstance(candidate, CandidateChunk)
            self.assertIsInstance(candidate.chunk, dict)

    def test_merge_preserves_chunk_and_doc_provenance(self):
        execution = self.merged()

        self.assertEqual(
            {candidate.doc_id for candidate in execution.chunks}, {A, B}
        )
        self.assertEqual(
            {candidate.chunk_id for candidate in execution.chunks},
            {f"{A}:c", f"{B}:c"},
        )
        self.assertEqual([result.rank for result in execution.results], [1, 2])

    def test_mapping_chunks_still_merge(self):
        """The mapping shape unit fixtures use keeps working."""

        from app.retrieval.interfaces import RetrievalResult

        def execute(plan):
            code = plan.corp_codes[0]
            payload = chunk(code, "100", corp_code=code)
            return SimpleNamespace(
                documents=(),
                chunks=(payload,),
                results=(
                    RetrievalResult(
                        chunk_id=payload["chunk_id"], doc_id=payload["doc_id"],
                        bm25_score=1.0, rank=1, metadata_match={},
                    ),
                ),
            )

        execution = self.pipeline(execute)._comparison_execution(
            request(A, B), self.plan()
        )

        self.assertEqual(len(execution.chunks), 2)
        self.assertTrue(all(isinstance(c, dict) for c in execution.chunks))


class MergeOrderingTest(unittest.TestCase):
    """Every company's best document must survive the evidence limit."""

    #: What the evidence builder keeps for a general-evidence answer. The merge
    #: has to put each company's own rank-1 inside this many positions.
    GENERAL_EVIDENCE_LIMIT = 5

    def candidate(self, code, index):
        from app.retrieval.interfaces import CandidateChunk, MetadataMatch

        payload = chunk(f"{code}{index}", "100", corp_code=code)
        return CandidateChunk(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            chunk=payload,
            metadata_match=MetadataMatch(),
        )

    def execution_for(self, code, depth):
        from app.retrieval.interfaces import RetrievalResult

        candidates = [self.candidate(code, index) for index in range(1, depth + 1)]
        return SimpleNamespace(
            documents=(),
            chunks=tuple(candidates),
            results=tuple(
                RetrievalResult(
                    chunk_id=candidate.chunk_id, doc_id=candidate.doc_id,
                    bm25_score=1.0, rank=index + 1, metadata_match={},
                )
                for index, candidate in enumerate(candidates)
            ),
        )

    def merged(self, codes, depth=10):
        from app.api.pipeline import AnswerPipeline
        from app.reasoning.query_plan import QueryPlan

        pipeline = AnswerPipeline(
            understanding=SimpleNamespace(),
            executor=SimpleNamespace(
                execute=lambda plan: self.execution_for(plan.corp_codes[0], depth)
            ),
        )
        plan = QueryPlan(
            query="원본 질의",
            raw_query="원본 질의",
            companies=tuple(NAMES[code] for code in codes),
            corp_codes=tuple(codes),
            task_type="corporate_event",
            disclosure_route=("exchange",),
        )
        return pipeline._comparison_execution(request(*codes), plan)

    def test_two_companies_interleave_by_rank(self):
        execution = self.merged((A, B))
        order = [result.chunk_id for result in execution.results]

        self.assertEqual(
            order[:6],
            [f"{A}1:c", f"{B}1:c", f"{A}2:c", f"{B}2:c", f"{A}3:c", f"{B}3:c"],
        )

    def test_four_companies_lead_with_every_first_hit(self):
        execution = self.merged((A, B, C, D))
        order = [result.chunk_id for result in execution.results]

        self.assertEqual(
            order[:4], [f"{A}1:c", f"{B}1:c", f"{C}1:c", f"{D}1:c"]
        )

    def test_every_company_rank_one_survives_the_evidence_limit(self):
        """The starvation that made a two-company answer cite one company."""

        for codes in ((A, B), (A, B, C, D)):
            with self.subTest(companies=len(codes)):
                execution = self.merged(codes)
                kept = [
                    result.doc_id
                    for result in execution.results[: self.GENERAL_EVIDENCE_LIMIT]
                ]
                for code in codes:
                    self.assertIn(f"{code}1", kept)

    def test_ranks_are_renumbered_over_the_merged_order(self):
        execution = self.merged((A, B), depth=3)

        self.assertEqual(
            [result.rank for result in execution.results], [1, 2, 3, 4, 5, 6]
        )

    def test_each_companys_own_order_is_preserved(self):
        execution = self.merged((A, B))
        a_order = [
            result.chunk_id
            for result in execution.results
            if result.doc_id.startswith(A)
        ]

        self.assertEqual(a_order, [f"{A}{index}:c" for index in range(1, 11)])

    def test_chunks_follow_the_merged_result_order(self):
        from app.retrieval.interfaces import CandidateChunk

        execution = self.merged((A, B))

        self.assertEqual(
            [candidate.chunk_id for candidate in execution.chunks],
            [result.chunk_id for result in execution.results],
        )
        self.assertTrue(
            all(isinstance(c, CandidateChunk) for c in execution.chunks)
        )

    def test_a_shorter_company_does_not_stop_the_interleave(self):
        """One company returning fewer results must not truncate the others."""

        from app.api.pipeline import AnswerPipeline
        from app.reasoning.query_plan import QueryPlan

        depths = {A: 1, B: 4}
        pipeline = AnswerPipeline(
            understanding=SimpleNamespace(),
            executor=SimpleNamespace(
                execute=lambda plan: self.execution_for(
                    plan.corp_codes[0], depths[plan.corp_codes[0]]
                )
            ),
        )
        plan = QueryPlan(
            query="q", raw_query="q",
            companies=(NAMES[A], NAMES[B]), corp_codes=(A, B),
            task_type="corporate_event", disclosure_route=("exchange",),
        )
        execution = pipeline._comparison_execution(request(A, B), plan)

        self.assertEqual(
            [result.chunk_id for result in execution.results],
            [f"{A}1:c", f"{B}1:c", f"{B}2:c", f"{B}3:c", f"{B}4:c"],
        )
