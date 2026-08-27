# Freeze Log

Authoritative record of components that have passed implementation, regression,
and live-server verification. A component listed here is **closed**. It is not
reopened for cleanup, restructuring, or performance work.

Branch: `taeyoon` · Log current through `eaec179`.

---

## Reopen policy

A FINAL FREEZE component may be modified only when at least one of these holds:

1. a reproducible regression is demonstrated;
2. hidden/official-style evaluation exposes a correctness failure attributable
   to the frozen component;
3. a later architecture requires a change that cannot be implemented cleanly
   without modifying the frozen contract.

Before modifying a frozen component:

- reproduce the failure;
- record the failing question/test;
- identify the exact violated invariant;
- add a regression test;
- document why an additive solution *outside* the frozen component is not viable.

**Performance improvement alone is NOT sufficient reason to reopen a frozen
component.**

---

## P0-A — Correction Graph

**Status:** FINAL FREEZE

**Frozen commits:** `0e6542a` (graph) → `2554647` (retrieval integration) →
`9200ee1` (API trace)

### Problem solved
Which disclosure a correcting filing corrects was not resolvable deterministically.
Correction/latest resolution had to stop depending on search ranking heuristics.

### Final architecture / behavior
A persisted correction graph built by deterministic resolution rules
(`correction_notice`, `periodic_period_key`, `event_title_key`), with every
member classified resolved / ambiguous / unresolved. Retrieval consults the
persisted graph — not the `is_correction` flag alone — and expands the resolved
chain for latest/history intents. A database without the correction tables
degrades to prior behavior rather than failing.

### Production files
- `app/reasoning/correction_graph.py`, `app/reasoning/correction_policy.py`
- `app/retrieval/correction_repository.py`, `app/retrieval/correction_expansion.py`
- `app/reasoning/router.py`, `app/retrieval/hybrid.py`, `app/api/pipeline.py`
- `db/006_correction_graph.sql`, `scripts/build_correction_graph.py`

### Invariants that must remain true
- Correction/latest disclosure resolution is **deterministic**.
- The latest corrected disclosure is resolved **through the correction graph**.
- Naive search/ranking logic must not replace graph resolution.
- Expansion stays within a resolved chain: no cross-company expansion, cycle
  guarded, bounded.
- Chain integrity is enforced in the database, not only in Python: a member has
  a parent if and only if it has a chain position
  (`correction_group_members_chain_parent`).
- A missing correction schema degrades; it does not raise.

### Verification performed
- Graph construction against the real corpus: 0 cycles, 0 duplicates, 0
  self-references; idempotent backfill verified against live PostgreSQL 16.
- Depth-13 chain and `event_title_key` threshold audited.
- `tests/test_correction_graph.py` + `tests/test_correction_expansion.py`:
  **83 tests OK** at `53e480f`.

### Known residual issues NOT solved
- Members classified `ambiguous` / `unresolved` remain unresolved by design;
  they are reported, not guessed.

### Reopen conditions
Demonstrated wrong correction-chain resolution, proven regression, or a blocking
architectural incompatibility.

---

## P0-B — Corporate Event Timeline / Graph

**Status:** FINAL FREEZE

**Frozen commit:** `e21ee27`

### Problem solved
A question about one filing in a corporate-event lifecycle (a contract and its
termination) needed the rest of that lifecycle, without a second search pass.

### Final architecture / behavior
A persisted corporate-event timeline graph. The executor is the single owner of
event expansion on the serving path: the graph is followed once, never searched
again. A database without `db/007` degrades to prior behavior.

### Production files
- `app/reasoning/corporate_event.py`, `corporate_event_graph.py`,
  `corporate_event_resolver.py`
- `app/retrieval/corporate_event_repository.py`, `app/retrieval/event_expansion.py`
- `app/retrieval/hybrid.py`, `app/api/pipeline.py`
- `db/007_corporate_event_timeline.sql`, `scripts/build_corporate_event_graph.py`

### Invariants that must remain true
- Corporate event relationships and timeline resolution remain **deterministic**.
- Deterministic event relationships must not be replaced by LLM-only retrieval
  loops.
- Event expansion happens once, inside retrieval; no second search.
- A missing event schema degrades; it does not raise.

### Verification performed
- `tests/test_corporate_event.py`: **76 OK** (skipped 5);
  `tests/test_event_expansion.py`: **35 OK** (skipped 8), at `53e480f`.
  Skips are environment-gated (`db/007 not applied`).

### Known residual issues NOT solved
- Not recorded in this log; no residual claim is made.

### Reopen conditions
Proven regression or blocking architectural incompatibility.

---

## P0-C — Deterministic Multi-Document Planner / Completeness

**Status:** FINAL FREEZE

**Frozen commits:** `d04c587` (planner) → `e1c8489` (semantic evaluation) →
`a09f6e6` (trace in API response) → `ba6a7c3` (trace sanitization)

### Problem solved
Questions requiring several disclosures to be complete needed a deterministic
completeness layer that could not disturb the frozen ranking.

### Final architecture / behavior
Planning runs **after** retrieval, so it can only add completeness evidence on
top of the frozen ranking, never replace it. It engages only for questions that
name a company, an enumerable family, a bounded period, and an explicit date
basis. A declined question — which is every Gold60 question — takes exactly the
path it always did, including an unchanged `think_trace`. The trace carries
counts and statuses only; no identifier reaches it.

### Production files
- `app/reasoning/multi_document_plan.py`, `multi_document_planner.py`,
  `multi_document_executor.py`, `multi_document_evidence.py`
- `app/retrieval/enumeration.py`, `app/agent/orchestrator.py`,
  `app/api/pipeline.py`, `app/api/schemas.py`

### Invariants that must remain true
- Multi-document planning and completeness checks remain **deterministic**.
- The citation/evidence completeness contract remains intact.
- P0-C is additive: a declined question's response and trace are unchanged.
- Planning never reorders or replaces retrieval ranking.
- The trace stays sanitized — counts and statuses, no identifiers.

### Verification performed
- `tests/test_multi_document_{plan,planner,executor,evidence,semantics,serving}`:
  **161 tests OK** at `53e480f`.

### Known residual issues NOT solved
- Not recorded in this log; no residual claim is made.

### Reopen conditions
Proven regression or blocking architectural incompatibility.

---

## P0-D — Query Understanding & Verification

**Status:** FINAL FREEZE

**Frozen commit:** `7a7da17`

### Problem solved
Retrieval could run for questions that were not actually resolvable, and answers
could be emitted without a check that the requested fields were citable.

### Final architecture / behavior
Deterministic-first query understanding; conditional HCX semantic fallback **at
most once**; a clarification gate; deterministic validation producing
`retrieval_allowed`; and an `AnswerabilityGuard` that classifies answerability
from deterministic resolver and public P0 state. A non-RESOLVED state blocks
retrieval before the executor is reached. `task_router` production behavior is
intentionally preserved.

### Production files
- `app/reasoning/query_validation.py`, `semantic_query_fallback.py`,
  `query_understanding.py`, `answerability.py`
- `app/api/pipeline.py`, `app/api/schemas.py`

### Invariants that must remain true
- Deterministic understanding runs first; HCX fallback is conditional and fires
  **at most once** per query.
- A query that is not RESOLVED does not reach retrieval.
- `AnswerabilityGuard` is not a lexical relevance judge: it does not rerank or
  filter retrieval.
- `task_router` production behavior stays as frozen here.
- The public response exposes exactly five top-level fields.

### Verification performed
- `tests/test_query_validation.py`, `test_query_understanding.py`,
  `test_p0d_pipeline.py`, `test_semantic_query_fallback.py`: **80 tests OK** at
  `53e480f`.
- `tests/test_answerability.py`: **10 OK**.

### Known residual issues NOT solved
- Questions correctly classified `ambiguous` return a clarification instead of
  an answer. This is intended behavior, not a defect to be engineered away.

### Reopen conditions
Proven regression or blocking architectural incompatibility.

---

## P1-A2 — Holding Evidence Routing Consistency

**Status:** FINAL FREEZE

**Frozen commit:** `1b8d08f`

### Problem solved
`TaskRouter` could promote `disclosure_lookup` + holding route → `holding_event`,
while `EvidenceBuilder` grouped solely from `QueryPlan.task_type`, which was
still `disclosure_lookup`. No `holding_event` groups were built, so the holding
resolver received nothing to resolve.

### Final architecture / behavior
The orchestrator passes an explicit **execution grouping intent** derived from
the routed task type:

```python
_EXECUTION_GROUPING_INTENT = {"holding_event": "holding_change"}
```

The router's semantics and the grouping vocabulary stay separate; the mapping is
the only bridge between them.

### Production files
- `app/agent/orchestrator.py`
- `app/reasoning/evidence_builder.py`

### Invariants that must remain true
- `QueryPlan.task_type` remains **unchanged**.
- `EvidenceSet.task_type` remains **plan-derived**.
- `TaskRouter` semantics remain distinct from grouping semantics.
- The mapping remains exactly `{"holding_event": "holding_change"}`.
- Grouping intent is backward compatible: absent an intent, prior behavior holds.

### Verification performed
- Full suite at freeze: **1286 OK, skipped 13**.
- Live holding smoke verified; P0 regression verified.
- `tests/test_holding_grouping_intent.py`: **16 OK** at `53e480f`.

### Known residual issues NOT solved
- Structured holding projection coverage (addressed later by P1-A3).
- Exact holding-date narrowing.
- Multi-event over-inclusion.

### Reopen conditions
Proven regression or blocking architectural incompatibility.

---

## P1-A3 — Holding Structured Evidence Coverage Rescue

**Status:** FINAL FREEZE

**Frozen commit:** `53e480f`

### Problem solved
A `holding_event` execution could hold relevant structured holding projections in
the already-fetched candidate pool while the final served evidence lacked
resolver-consumable structured fields.

**The old P1-A design must NOT be restored.** It gated on *"does any holding
projection exist?"*, which let an unrelated projection suppress the rescue. Live
A/B falsified it: results were byte-identical to baseline.

### Final architecture / behavior

Coverage-based, **structurally anchored** holding evidence enrichment.

**Gate** — only when routed execution is `holding_event`; inspect query-visible
requested holding fields; use resolver-native field semantics; use only citable
structured projections; reporter-compatible evidence only.

**Structured projection types** — `holding_detail_row`, `holding_report`.

**Coverage** — served evidence is sufficient when the union of served, citable,
reporter-compatible holding projections covers all query-visible requested fields
according to existing resolver semantics. `change_direction` is treated by the
resolver's existing derivation semantics; **no separate label ontology is
created**.

**Candidate source** — the already-fetched candidate pool only.

**Structural anchoring**

| Tier | Rule |
|---|---|
| STRONG | same doc **and** same event-bearing source row/reference. Metadata-only refs (e.g. holding-purpose rows) do **not** qualify. |
| MEDIUM | same doc **and** same normalized `reference_date` **and** reporter-compatible. |

No weak or unanchored fallback. An unanchored candidate is declined
(`no_anchored_candidate`), not promoted.

**Candidate priority**

1. served anchor rank
2. STRONG before MEDIUM within the same anchor
3. coverage contribution
4. minimum candidates within the same anchor/tier

Global minimum-cardinality must **not** be used where it would let lower-ranked
served evidence displace completion of higher-ranked evidence.

**Merge** — retain the raw/served anchor; add the structured sibling; displace
the lowest-ranked non-contributing evidence; final count stays within existing
`top_k`; chunk IDs are never duplicated.

**Public evidence synchronization** — the original `HybridQueryExecution` stays
immutable, so the enriched list is *carried out* on `AgentResult` rather than
written back. `AnswerabilityGuard` and the public `retrieved_context` consume
that one final evidence set.

### Production files
- `app/agent/orchestrator.py`
- `app/api/pipeline.py`
- `app/reasoning/holding_evidence_coverage.py`

Tests: `tests/test_holding_evidence_coverage.py`

### Invariants that must remain true

**Retrieval isolation — strictly forbidden inside this lane:**
second lexical search · second vector search · DB query · external retrieval ·
global `top_k` increase · Gold doc targeting · question-ID hacks ·
company/reporter/doc/table hardcoding.

**Anchoring**
- No weak or unanchored rescue, ever.
- Metadata-only source refs are never event anchors.
- Served anchor rank dominates candidate count.

**Merge**
- The served anchor is retained, never replaced.
- Only non-contributing evidence may be displaced; if only contributors remain,
  the rescue is abandoned rather than trading one gap for another.
- Final count stays within existing `top_k`; no duplicate chunk IDs.

**Public synchronization**
- The original `HybridQueryExecution` remains **immutable**. The public output is
  repaired by carrying the final list, never by mutating retrieval.
- `AgentResult` carries the final evidence results.
- `AnswerabilityGuard` and public `retrieved_context` use the **same** final
  evidence set.
- `retrieved_context` contains every evidence chunk actually used for final
  citations.
- There is **no second public evidence universe**.
- Top-level API keys remain exactly: `question_id`, `question`,
  `retrieved_context`, `think_trace`, `answer`.
- Internal rescue metadata stays sanitized out of public rows; the evidence chunk
  itself is published in full.

**Ontology**
- No second field ontology: field candidates, labels, normalization and date
  handling are driven by the resolver's own definitions.

### Verification performed

- Full suite: **1346 OK, skipped 13** (confirmed at `53e480f`).
- `tests/test_holding_evidence_coverage.py`: **60 OK**.
- Mutation checks: removing the anchor requirement fails 7 tests; reverting the
  public context to the pre-enrichment list fails 7.

**Live BGE-M3 verification**

| | HX12 baseline | HX12 with P1-A3 |
|---|---|---|
| status | `insufficient_evidence` | answerable |
| citations | 0 | 2 |
| missing | `reference_date` | `[]` |

- `holding_evidence_coverage` stage executed.
- The live rank-1 raw table remained public; its rank-1 structured sibling was
  added to the public `retrieved_context`; context count remained 10.
- The answer's citations were represented in public evidence.
- The live server selected the **2024-02-19** event because that was the real
  BGE-M3 rank-1 served evidence — demonstrating the implementation is not tied to
  offline/hash ordering or a hardcoded Gold date.

**No-op behavior confirmed live**
- HX10 — production BGE-M3 baseline was already answerable; P1-A3 correctly
  remained a no-op.
- HX08 / HX16 / HX20 — remained answerable; P1-A3 correctly remained a no-op.

**P0 correction regression** — resolved, valid, latest correction preserved,
answerable.

### Known residual issues NOT solved

- A question such as HX12 can still be **semantically ambiguous**: multiple
  holding events satisfy the visible wording.
- P1-A3 makes evidence completion *principled*; it does **not** make the query
  unique.
- This belongs to later holding-date / multi-event ambiguity work.
- **Do NOT reopen P1-A3 merely to solve presentation ambiguity.**

### Reopen conditions
Proven regression or blocking architectural incompatibility, per the reopen
policy above.

---

## P1-A4 — Exact Holding Event Resolution

**Status:** FINAL FREEZE

**Frozen commit:** `d39a1b1`

### Problem solved
Explicit holding-event dates could degrade to year-level semantics after
`disclosure_lookup` was later promoted to `holding_event`, and one real holding
event could be split across complementary evidence groups.

Neither half was sufficient alone: the exact date narrowed the constraint but
changed no answer, and fusion without the date regressed an undated question to
unanswerable. The three parts shipped as one phase.

### Final architecture / behavior

**D1 — execution-scoped exact holding-reference date.** Reuses the existing
`_period_from_query`; no new date parser. Fires only for `holding_event`
execution, only on an exact full date; receipt wording, ranges, year-only and
year-month are excluded. The original `QueryPlan` is immutable, `QueryPlan.task_type`
is unchanged, and a native `holding_change` exact period is preserved rather than
re-derived.

**D2 — exact-date-only complementary same-event fusion.** Runs only for an
exact-date P1-A4 execution. Two groups are one event only when they share a
`doc_id` and a normalized `reference_date`, agree exactly on a populated and
arithmetically consistent before/change/after transition that is not all-zero,
draw on disjoint event-bearing source tables, and produce no non-reporter field
conflict when merged. Reporter text and projection type never establish identity.
The original `EvidenceSet` is immutable and no retrieval chunk is added or removed.

**D2b — conservative reporter-alternative matching.** No alias table. Reporter
alternatives retain their provenance; when they conflict, **every** retained
alternative must satisfy the existing query reporter constraint, and
cross-reporter fusion is declined outright without a reporter constraint.

### Production files
- `app/reasoning/holding_date_intent.py`
- `app/reasoning/holding_event_fusion.py`
- `app/agent/orchestrator.py`
- `app/reasoning/holding_event_resolver.py`

Tests: `tests/test_holding_exact_event_resolution.py`

### Invariants that must remain true

**D1**
- The existing `_period_from_query` is reused; no second date parser or alias list.
- Only `holding_event` execution, only an exact full date.
- Receipt, range, year-only and year-month semantics never yield a holding date.
- The original `QueryPlan` is never mutated and `QueryPlan.task_type` never changes.
- A native `holding_change` exact period is preserved, never rewritten.

**D2**
- Fusion runs **only** for an exact-date P1-A4 execution — it is not a general
  holding normalization pass.
- Identity requires: same `doc_id`; same normalized `reference_date`; exact
  before/change/after equality; all three populated; arithmetic consistent;
  not all-zero; disjoint event-bearing source tables; no non-reporter conflict.
- Reporter text and projection type never establish event identity.
- The original `EvidenceSet` is immutable; grouping changes, membership does not.

**D2b**
- No reporter alias table, ever.
- Reporter alternatives retain provenance.
- Conflicting alternatives satisfy a reporter constraint only when **all** of
  them do.
- Cross-reporter fusion is declined when the query names no reporter.

**Public / citation**
- Fusion changes logical grouping only.
- `retrieved_context` chunk IDs are unchanged by P1-A4.
- Every answer citation remains present in the public `retrieved_context`,
  preserving the P1-A3.1 synchronization invariant.

### Verification performed

- Full suite: **1391 OK, skipped 13** (confirmed at `d39a1b1`).
- `tests/test_holding_exact_event_resolution.py`: **45 OK**.
- Mutation checks: disabling D1 fails 4 tests, disabling D2 fails 6, flipping
  D2b's `all` to `any` fails 1 — each part is load-bearing.

**Live BGE-M3 verification**

| | HX13 | HX17 |
|---|---|---|
| reference date | 2023-06-13 | 2023-06-30 |
| shares | 2,202,050 | 1,092,455 |
| ratio | 7.90% | 6.99% |
| answerable | yes | yes |
| citations | preserved | preserved |

**No-date controls** — HX08, HX10, HX12, HX16, HX20 behaviourally unchanged.
HX10 remains outside P1-A4 fusion because no exact date is present. A later
P1-A5-B presentation fix may reduce rendered rows by removing events that the
resolver already marked non-matching; this does not violate the P1-A4 no-date
fusion invariant.

**P0 correction regression** — PASS.

### Known residual issues NOT solved

- Generic no-date multi-event ambiguity.
- Multi-metric wording such as "보유 수량 비율" may still be parsed as only one
  requested field in some cases.
- Year-month exact holding semantics.
- Same-holder identity cannot be proven from entity identifiers, because the
  corpus exposes no such identifiers. Identity is argued from structure and
  numbers alone, and the reporter rules bound the residual exposure.

### Reopen conditions
- a reproducible exact-date regression;
- unsafe fusion demonstrated by the real corpus or an evaluation;
- a citation/provenance break;
- a later architecture that cannot preserve this contract additively.

---

## P1-A5-B — Render Matching Holding Events Only

**Status:** FINAL FREEZE

**Frozen commit:** `eaec179`

### Problem solved
`HoldingEventResolver` already marked each event with `matches_query`, derived
from the question's reporter, temporal and direction constraints, but
`AnswerComposer` discarded that filtering whenever the matching count was not
exactly one:

```python
if len(matching) != 1:
    return events        # every retrieved event, matching or not
```

Non-matching events were therefore rendered: decrease questions included
increase events, and reporter-mismatched events appeared in answers.

### Final architecture / behavior

| `matching` | behaviour |
|---|---|
| **1** | unchanged — the existing single-event output P1-A4 depends on |
| **> 1** | render all and only `matches_query is True` events |
| **0** | unchanged — the previous fallback still shows what was retrieved |
| no requested fields (history path) | unchanged — the whole timeline is kept |

The executable change is one line: `if len(matching) != 1:` became
`if not matching:`.

**Selection semantics** — no rank-based event selection; no latest/earliest
default; no new temporal semantics. Retrieval rank may affect presentation
order only.

### Production files
- `app/reasoning/answer_composer.py`

Tests: `tests/test_holding_matching_event_presentation.py`, plus an updated
contract in `tests/test_holding_answer_scope.py` (one existing test asserted the
old behaviour directly and was rewritten).

### Invariants that must remain true

**Selection**
- Only `event.matches_query` decides what is rendered — it is already derived
  from query-visible reporter, temporal and direction constraints.
- No event is ever chosen for ranking first, for being newest, or for being
  oldest.
- Retrieval rank is presentation order only, never semantic selection.
- `matching == 1` still returns that one event, so P1-A4's exact-event output is
  unaffected.
- `matching == 0` still falls back to the retrieved events, and the pre-existing
  "holder absent from every event" guard still returns nothing.

**Answerability**
- `AnswerabilityGuard` is untouched. Multi-event ambiguity is **not** treated as
  insufficient evidence.

**Citation / public evidence**
- No evidence is added; `retrieved_context` is unchanged.
- Removing non-matching rows can only reduce or hold the citation count.
- Every displayed citation remains grounded in the public `retrieved_context`.

### Verification performed

- Full suite: **1407 OK, skipped 13**.
- `tests/test_holding_matching_event_presentation.py`: **16 OK**.
- Mutation check: restoring `if len(matching) != 1:` fails 6 of the 16 new tests
  and 1 in `test_holding_answer_scope`; restored → 60 OK.

**Live BGE-M3 verification**

| qid | result |
|---|---|
| HX20 | only decrease events rendered; no increase leakage |
| HX16 | only matching decrease events rendered |
| HX10 | reporter mismatch removed |
| HX13 / HX17 | P1-A4 exact-date behaviour preserved |
| HX12 | P1-A3 rescue behaviour preserved |

**P0 correction regression** — PASS.

### Known residual issues NOT solved

- Semantic ambiguity when the query itself does not identify a unique event.
- HX12 can still appear unique because P1-A3 may rescue only one structured
  event from the served evidence — an artifact of what retrieval served, not of
  what the question asked. This must never be read as semantic uniqueness.
- Latest / previous / current semantics remain incomplete: `직전보고`, `현재`,
  `최근`, `최초` and `마지막` all collapse to inert period labels the resolver
  does not read.
- A generic ambiguity notice is deferred to **P1-A5-A**.

### Reopen conditions
Proven regression, a citation/provenance break, or a later architecture that
cannot preserve this contract additively.

---

## Summary

| Phase | Status | Frozen commit |
|---|---|---|
| P0-A Correction Graph | FINAL FREEZE | `9200ee1` (from `0e6542a`) |
| P0-B Corporate Event Timeline | FINAL FREEZE | `e21ee27` |
| P0-C Multi-Document Planner | FINAL FREEZE | `ba6a7c3` (from `d04c587`) |
| P0-D Query Understanding & Verification | FINAL FREEZE | `7a7da17` |
| P1-A2 Holding Evidence Routing Consistency | FINAL FREEZE | `1b8d08f` |
| P1-A3 Holding Structured Evidence Coverage Rescue | FINAL FREEZE | `53e480f` |
| P1-A4 Exact Holding Event Resolution | FINAL FREEZE | `d39a1b1` |
| P1-A5-B Render Matching Holding Events Only | FINAL FREEZE | `eaec179` |
