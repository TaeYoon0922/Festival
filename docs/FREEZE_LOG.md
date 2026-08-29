# Freeze Log

Authoritative record of components that have passed implementation, regression,
and live-server verification. A component listed here is **closed**. It is not
reopened for cleanup, restructuring, or performance work.

Branch: `taeyoon` · Log current through `da65d23`.

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

## P0-D.1 — Multi-company Query Understanding Diagnosis

**Status:** **DIAGNOSIS COMPLETE — KEEP P0-D FROZEN**

This is **not** a new FINAL FREEZE. The **P0-D FINAL FREEZE above remains in
force**, and no production implementation was changed. This entry records why
six declining Gold questions were investigated and deliberately left declining.

### What prompted the diagnosis
P2-A attributed **6 of 60** Gold questions to `QUERY_UNDERSTANDING_DECLINE` —
the only non-`COMPLETE` category in the structural pass. All six name **both
에스엠 and 하이브**, and all six resolve to the same structural domain shape:
**PRIMARY_COMPANY_WITH_COUNTERPARTY**.

### Corpus evidence for that shape
The gold document's structured projection carries both roles as distinct fields:

| role | value |
|---|---|
| disclosure owner / issuer (`corp_name`) | **에스엠** |
| reporter / holder (`보고자/보유자`) | **(주)하이브** |

A 대량보유상황보고서 is filed *about* an issuer *by* a holder, so the two names
describe one document's two roles rather than two search targets.

### The current ambiguity rule is count-based
`_company_slot` in `app/reasoning/query_validation.py`:

```
len(plan.companies) > 1  and  no explicit company_comparison  =>  AMBIGUOUS
```

It is not competing aliases, not an unresolved canonical, not a confidence tie,
and not a corp_code conflict. A multi-company **accept** path already exists —
`comparison.type == "company_comparison"` — but it fires only on the literal
terms `비교` / `대비`, which none of the six contain.

> ### ⚠ These six are **not** ambiguous aliases
>
> **Corpus-wide alias collision count was measured as zero.** No alias key maps
> to more than one canonical company, so `len(plan.companies) > 1` can never
> mean "one mention with two candidate readings". In every such case **two
> distinct companies were intentionally named**.

### Structural capability vs safe consumption
`QueryPlan.companies` / `corp_codes` are tuples, `backend_filters()` emits both,
and the SQL predicate is `corp_code = ANY(%s)` — a **union**. The plan and the
backend can therefore *represent* multiple companies. **Downstream components do
not safely consume arbitrary multi-company holding queries**: the P0-C executor
keys slots by a single `corp_code`, and the holding resolver derives one
`corp_code` per group.

### Unrestricted two-company retrieval is rejected
Measured offline against the seeded pair, hash embeddings:

| routing | gold reached | wrong-owner documents in top-10 |
|---|---|---|
| **both companies (OR union)** | **2 / 6** | **2–5 per query** |
| owner + reporter | 3 / 6 | 0 |
| owner only | 3 / 6 | 0 |

OR-union retrieval introduces wrong-owner documents and is **worse than
declining** — on one question the counterparty's filings displace a gold hit
that owner-only routing finds. **Unrestricted multi-company acceptance is
rejected.**

### A deterministic direction does exist for the observed pair
Derived from corpus projections only — no Gold, no company literal:

| direction | supporting documents |
|---|---|
| 에스엠 issuer + 하이브 reporter | **10** |
| reverse | **0** |

### Why the natural representation is nevertheless unsafe today
The otherwise natural plan shape is:

```
company  = 에스엠
reporter = 하이브
```

Frozen P1-A4 reporter matching currently gives:

```
_reporter_matches("(주)하이브", "하이브")  ==  False
```

**This is a statement about the current contract, not a defect finding.** P1-A4
is frozen and is not asserted here to be wrong; what is recorded is that its
reporter matching **does not equate `"(주)하이브"` with `"하이브"`**, and that
this blocks the proposed P0-D role resolution.

Full orchestrator simulations showed that assigning the reporter can change
`matches_query=True` to `matches_query=False`, and that under frozen **P1-A5-B**
(render matching events only) this can render **zero events**.

### Decision
**KEEP P0-D FINAL FREEZE.** Reopening P0-D alone would convert an explicit
clarification into a potentially **silent evidence loss** — a strictly worse
outcome than the current behaviour.

P0-D may be reconsidered **only after reporter-role compatibility downstream is
independently proven safe**.

**Next prerequisite:** diagnose **P1-A4 reporter normalization**.

### Also recorded
- **Unrestricted multi-company acceptance is rejected** (measured above).
- **P0-C is not a general multi-company planner** — its executor keys slots by a
  single `corp_code`; it plans multiple *documents*, not multiple *companies*.
- **Gold60 has no true alias-ambiguity negative control.** With zero alias
  collisions corpus-wide, no case exists to prove a relaxation still declines
  genuine ambiguity.
- **The remaining 3 of 6 owner-routed misses under hash embeddings belong to
  blocked P1-R and are not P0-D evidence.** They are ranking failures measured
  with `DeterministicHashEmbedder`, not understanding failures.

### Ownership firewall
| symptom | owner |
|---|---|
| query declined before retrieval | **P0-D** |
| holder corporate-prefix normalization — a downstream blocker **at P0-D.1 diagnosis time** | **P1-A4** (frozen then) → subsequently resolved by **P1-A4.1** FINAL FREEZE |
| eligible but **ranked low** under a verified embedding | **P1-R** (blocked) |

---

## P0-D.2 — Issuer / Reporter Role Resolution

**Status:** Phase 1 **DIAGNOSIS COMPLETE — REOPEN TARGET EXISTS** ·
**ACTIVATION DEFERRED — LIVE BGE-M3 SAFETY GATE FAILED (INFRA-E1)** ·
Phase 3 **SAFETY RECHECK COMPLETE — KEEP DEFERRED** (`e8851d7`) ·
Target B **FIRST-LOSS DIAGNOSIS COMPLETE — BENCHMARK TARGET CONTEXT DEPENDENT** ·
**ACTIVATION DEFERRED**

**Target A — the comparison-intent firewall — is RESOLVED** by *Comparison
Intent Firewall* (`7a0921e`, FINAL FREEZE), which did **not** activate P0-D.2.
**Target B — historically labelled the H01/HX02 retrieval-ranking residual —
has completed first-loss diagnosis.** Both questions have measurable ranking
and final-top-k symptoms, but their frozen 2024 targets depend on report context
that the standalone questions do not provide. See *Target B — H01 / HX02
First-Loss Diagnosis* below. This result does **not** activate P0-D.2.

This is **not** a FINAL FREEZE. **No P0-D.2 role resolution runs in
production** — this entry records a target that was proven to exist, the narrow
contract that would serve it, and the specific safety condition that keeps it
switched off.

Query understanding has since been reopened once, in a bounded way, for the
Target A comparison firewall; that change is frozen separately below and is
**not** part of this target.

### Target shape
**PRIMARY_COMPANY_WITH_REPORTER** — exactly two distinct companies are named in
a holding-routed query; one is the disclosure issuer, the other the
reporter/holder.

Distinct from: company comparison · arbitrary two-company lookup · M&A ·
contract counterparty · parent/subsidiary relation · parallel multi-company
lookup.

### Deterministic corpus signal
Holding projection reporter metadata only. For candidate companies A and B:

```
support(A -> B) = distinct holding documents where
                  issuer = A and reporter canonically matches B
support(B -> A) = symmetric
```

**P1-A4.1 reporter normalization is the frozen comparison contract** used for
that match. No Gold, no document-id lookup, no embeddings, no ranking.

### Corpus-wide relation inventory

| | |
|---|---|
| holding reporter occurrences | **51,730** |
| linked to another corpus master company | **107** |
| distinct directed issuer→reporter pairs | **16** |
| asymmetric | **16** |
| reverse-direction pairs observed | **0** |

Representative examples of the relation's shape — strong support
(한화오션 → 한화에어로스페이스, 15 documents; 현대자동차 → 현대모비스, 10),
mid support (현대오토에버 → 현대자동차, 5), and single-document support
(삼성E&A → 삼성SDI). These illustrate the distribution; **none of them is a
special case**, and the rule treats every pair identically.

> ### ⚠ A corpus relation alone is NOT sufficient
>
> 한화오션 → 한화에어로스페이스 carries the strongest holding support in the
> corpus, yet a **contract** question naming those two companies must not be
> read as a holding issuer/reporter query. Relation support says the pair exists;
> it says nothing about what the question is asking.
>
> The role resolver must therefore additionally require
> **`disclosure_route == ["holding"]`**. Comparison handling retains precedence.

### Proposed narrow contract
Trigger only when **all** hold:

- the company slot would otherwise be AMBIGUOUS;
- exactly two distinct canonical companies;
- not `company_comparison`;
- `disclosure_route == ["holding"]`.

Then inspect directed holding support. A direction is eligible only when
supported by **≥2 distinct documents**. If exactly one eligible direction exists:

```
companies         = (issuer,)
corp_codes        = (issuer_code,)
reporter          = counterparty name
QueryState        may become RESOLVED
retrieval_allowed may become True
```

Otherwise the current AMBIGUOUS clarification is retained. Queries naming **3+
companies remain conservative and unchanged**.

### Why the threshold matters
Of the 16 observed relation pairs, **6 rely on a single supporting document**. A
`≥2` threshold retains **10 of 16** and rejects the six weakest relations. This
is a **safety threshold supported by the current small relation inventory**, and
it **must be reconsidered if larger evidence changes the distribution**.

### The six Gold60 declines
All six known 에스엠 / 하이브 declines resolve structurally as
**issuer = 에스엠, reporter = 하이브**, with **forward support = 10 documents**
and **reverse support = 0**.

Frozen P1-A4.1 now gives `reporter_matches("(주)하이브", "하이브") == True`,
so **the P0-D.1 downstream reporter blocker is removed**.

> ### ⚠ Activation blocker — the six are NOT fixed
>
> Offline downstream simulation used `DeterministicHashEmbedder`. Role
> resolution succeeded **6/6**, but only **H02, HX01 and HX04** reached the
> intended Gold document. **H01, HX02 and HX03 did not** under hash ranking.
>
> H01 and HX03 could therefore change from today's explicit clarification into a
> **confident answer based on a different filing** if P0-D.2 were activated
> without live retrieval validation. That is why activation remains deferred.
>
> Every ranking conclusion from this experiment is **HASH_DIAGNOSTIC, not live
> BGE-M3**.

### Ownership
| concern | owner |
|---|---|
| role determination | **P0-D.2** |
| reporter string compatibility | **P1-A4.1** (frozen) |
| eligible evidence ranked incorrectly | **P1-R** |
| undated multiple matching events | **P1-A5-A** (frozen) |

Ranking and event uniqueness must **not** be fixed inside P0-D.2.

### Runtime architecture finding
| approach | measured cost | verdict |
|---|---|---|
| per-query DB `GROUP BY` | **~311 ms** on the measured subset | rejected |
| full projection scan at startup | **~8.4 s** | rejected |

A production implementation would require a small **precomputed,
corpus-versioned issuer→reporter relation artifact** loaded with `CorpusScope`.
Measured artifact shape in the current corpus: **~16 rows / ~3.2 KB**.

**Stale-artifact protection and refresh semantics are mandatory** — a relation
that outlives its corpus silently mis-assigns roles.

### Risks
- only **16** linked relation pairs exist;
- **6** have one-document support;
- no symmetric pair was observed, but **absence is not a domain guarantee**;
- the company master covers only a tiny share of all holders;
- **live ranking is still unavailable**;
- the route gate has been tested on **constructed controls**, not a broad
  real-user set.

### Reopen / activation gate
Implementation may proceed only while the role-resolution algorithm and the
relation artifact remain **deterministic and generic**.

Activation that changes user-visible P0-D behavior additionally requires one of:

**A.** live BGE-M3 evaluation showing the newly accepted queries retrieve
reliable evidence; **or**

**B.** an equally strong deterministic evidence-safety mechanism proven not to
convert a clarification into a confident wrong answer.

**Gold and expected answers must not be used for that safety decision.**


### Live BGE-M3 activation result (INFRA-E1)

Production P0-D was **not** modified; only the diagnosed representation was
simulated downstream. All six roles resolved structurally from the corpus
relation — **issuer = 에스엠, reporter = 하이브** (forward support 10 documents,
reverse 0), at the `>=2` document threshold.

| question | intended document, final rank |
|---|---|
| H01 | **top-10 MISS** |
| H02 | 1 |
| HX01 | 2 |
| HX02 | **top-10 MISS** |
| HX03 | 7 |
| HX04 | 1 |

The gate requires the intended evidence in the final top-10 **for all six**. It
**FAILED** because H01 and HX02 miss.

**P0-D.2 therefore remains TARGET EXISTS · ACTIVATION DEFERRED.**

> **Update after HX04 Acquisition Semantics FINAL FREEZE (`7393842`).** One
> downstream blocker recorded against this target has been removed: HX04 failed
> *after* role resolution, which already produced the correct issuer and
> reporter, and that failure is now fixed in the holding acquisition lane.
>
> **This changes nothing here.** `PRIMARY_COMPANY_WITH_REPORTER` is **not**
> production-active, and P0-D.2 remains **ACTIVATION DEFERRED — SAFETY RECHECK
> REQUIRED**. The live ranking safety gate that deferred activation has not been
> re-measured, and the HX04 result above was obtained through the bounded role
> *simulation*, not through production P0-D, which still declines the question.

### Safety-positive findings
Despite those ranking misses:

- **no confidently wrong unique filing was produced**;
- exact-date **HX01 selected the correct structural date**;
- **H01 and HX02 remained under-specified multi-event shapes**;
- **P1-A4.1 reporter normalization worked** — `(주)하이브` matched `하이브`;
- **negative controls were unchanged**.

This confirms **frozen P1-A5-A is providing the intended safety behaviour**: the
two misses surfaced as visibly under-specified answers rather than confident
wrong ones.

### Ownership of the remaining loss
| concern | owner |
|---|---|
| issuer/reporter role resolution | **P0-D.2** |
| reporter string compatibility | **P1-A4.1** |
| remaining retrieval ranking loss | **P1-R** |
| undated multi-event safety | **P1-A5-A** |
| the observed misses | **not P1-B** |

### Safety recheck — Phase 3

**Result: P0-D.2 SAFETY RECHECK COMPLETE — KEEP DEFERRED.**

This recheck is **documentation only**. It is **not** a FINAL FREEZE activation.
`PRIMARY_COMPANY_WITH_REPORTER` remains **NOT production-active**, and
**two independent blockers remain**.

#### 1 · Recheck baseline

| | |
|---|---|
| branch | `taeyoon` |
| safety recheck HEAD | `e8851d7` |
| full suite | **1663 passed · 13 skipped · 1060 subtests** |

Frozen dependencies confirmed at recheck time:

| dependency | status | commit |
|---|---|---|
| HX04 Acquisition Semantics | FINAL FREEZE | `7393842` |
| Embedding Identity Hardening | FINAL FREEZE | `f592e9d` |
| Retrieval Vector-Availability Policy | FINAL FREEZE | `b1e31aa` |
| P1-R Bounded Additive Document Recovery | FINAL FREEZE | `6503c77` |

#### 2 · Frozen role contract under recheck

The activation candidate contract is unchanged. It is eligible **only** when all
of these hold:

- the current result would otherwise be **AMBIGUOUS** because of a multi-company
  company slot;
- exactly **two** distinct canonical companies;
- `company_comparison == false`;
- `disclosure_route == ["holding"]`;
- corpus relation support has **exactly one** eligible directed
  issuer → reporter relationship;
- that direction is supported by **≥2 distinct holding documents**.

Then, and only then:

```
companies  = (issuer,)
corp_codes = (issuer_code,)
reporter   = counterparty
```

Otherwise **AMBIGUOUS is retained**. Queries naming **3+ companies remain
AMBIGUOUS**. Non-holding routes are **unchanged**. Reporter canonicalization
uses **frozen P1-A4.1 behaviour**.

#### 3 · Six target controls

Role resolution is **6/6 correct**. For all six: issuer **에스엠**, reporter
**하이브**, forward relation support **10**, reverse relation support **0**,
exact vector coverage **1.000**.

**Downstream safety is NOT 6/6.**

| question | intended document | note |
|---|---|---|
| H01 | **MISS** | |
| H02 | rank 1 | |
| HX01 | rank 2 | exact / `semantic_unique` |
| HX02 | **MISS** | |
| HX03 | rank 7 | |
| HX04 | rank 1 | acquisition blocker resolved |

#### 4 · HX04 status

**HX04 downstream correctness is resolved.** Under generic bounded P0-D.2
simulation:

| | |
|---|---|
| `requested_fields` | `acquisition_date`, `acquired_shares` |
| matching event count | **1** |
| `acquisition_date` | **`2024-03-07`** |
| `acquired_shares` | **`868,948`** |
| same-row provenance | valid |
| answerable | **true** |
| citations | valid |

`2024-03-14` is **not** `acquisition_date`. `2,967,759` is **not**
`acquired_shares`.

This confirms the HX04 blocker is no longer preventing P0-D.2 activation.

> **Do NOT infer that P0-D.2 itself is safe.** One blocker being cleared is not
> an activation signal.

#### 5 · H01 / HX02 retrieval blocker

> ### ⚠ Corrected by the later Target B first-loss diagnosis
>
> This subsection is retained as the historical Phase 3 record. Its broad
> **retrieval blocker** ownership has been superseded: both questions do have
> ranking/top-k symptoms, but neither standalone question uniquely identifies
> the frozen 2024 filing.
>
> Phase 3 also misstated HX02's intended values. The actual frozen HX02 target
> is the **previous-report** pair **`2,098,811 / 8.81%`** inside
> `holding_20240314001102`, table `t0012`. The pair
> **`2,967,759 / 12.45%`** is the filing's **current** holdings and is **not
> HX02-equivalent**. The original wording below is preserved rather than
> silently rewritten.

**H01 and HX02 remain unresolved.** Both intended chunks are **structurally
present in the retrieval candidate set**. Therefore this is:

- **NOT** filter exclusion;
- **NOT** candidate omission;
- **NOT** vector-coverage failure.

They are **ranking / final-selection losses**.

**H01** — intended evidence is not emitted. Selected events instead include
different dates such as `2023-02-09`, `2023-02-22`, `2023-03-13`, and values
such as `4,392,368 / 18.45%` and `4,626,185 / 19.43%`. The requested intended
values **`2,967,759 / 12.45%` are absent**.

**HX02** — same core failure. Selected evidence carries unrelated/different
event values, and the intended `2,967,759 / 12.45%` evidence **is absent**.

> **These are NOT answer-equivalent exact-Gold misses.** They are genuine
> downstream evidence failures. Current answers are hedged and under-specified
> rather than confidently false — but **activation would replace today's
> clarification with answerable output that does not contain the requested
> event**.

#### 6 · Comparison firewall blocker

An **independent** validator / query-understanding blocker.

Example query:

```
한화오션과 한화에어로스페이스 중 어디가 보유 주식수가 더 많아?
```

Current parsing yields `company_comparison = false`. Because the two-company
holding relation satisfies the proposed role rule, the query **would be
role-resolved as issuer/reporter**.

**This is WRONG.** The query is a company comparison. Activation would
**steal a legitimate comparison query and convert it into a single-issuer
holding lookup**.

Other tested comparison phrasings remained ambiguous, which demonstrates the
issue is an **incomplete comparison detector**, not a desired role-resolution
behaviour. **This blocker is independent of the H01/HX02 retrieval blocker.**

> **⚠ Corrected later — this blocker was UNDER-measured.** The harness behind
> this section read a `QueryPlan` attribute that does not exist, so the
> comparison signal it reported was always false. Re-measured against the real
> `plan.comparison` during Target A diagnosis, **6 of the 10** required matrix
> questions would have been stolen, not the one recorded here, and **17 of 20**
> comparison constructions were unprotected. The scale is corrected in
> *Comparison Intent Firewall* below; this section is kept as it was recorded.

#### 7 · False-positive sweep

**93 questions tested** — Gold60, both orderings of the corpus relation pairs,
and an unrelated pair control.

| | |
|---|---|
| trigger-eligible | **39** |
| role-resolved | **26** |
| left ambiguous | **67** |
| Gold60 resolved | **6** — exactly the six P0-D.2 target controls |
| corpus pair queries | 20 resolved · 12 ambiguous |
| unrelated pair | ambiguous |
| **false role resolution** | **1** |

The single false resolution is the **company-comparison case** above.

> **⚠ `false role resolution = 1` is an artefact of the harness bug.** The
> sweep's comparison column never read the real field, so comparison precedence
> was never exercised across these 93 questions. The corrected comparison-side
> numbers are recorded in *Comparison Intent Firewall*. The remaining counts in
> this table — trigger-eligible, role-resolved, ambiguous, Gold60 — do not
> depend on that column and stand as recorded.

**Word-order independence: PASS.** Issuer/reporter direction did not depend on
mention order.

#### 8 · Corpus relation inventory

Derived from the processed holding corpus. **No Gold information was used to
construct these relationships.**

| | |
|---|---|
| holding documents | **1,083** |
| reporter occurrences | **51,730** |
| distinct master-linked directed pairs | **16** |
| ≥2-document directed pairs | **10** |
| one-document pairs | **6** |
| bidirectional pairs | **0** |

Representative directed support:

| directed pair | support |
|---|---|
| 한화오션 → 한화에어로스페이스 | 15 |
| 현대제철 → 기아 | 11 |
| 에스엠 → 하이브 | 10 |
| 현대자동차 → 현대모비스 | 10 |
| 현대모비스 → 기아 | 8 |
| 레인보우로보틱스 → 삼성전자 | 7 |
| 현대오토에버 → 현대자동차 | 5 |
| 두산퓨얼셀 → 두산에너빌리티 | 4 |
| 삼성바이오로직스 → 삼성전자 | 2 |
| 하이브 → 미래에셋증권 | 2 |

#### 9 · Support threshold

| threshold | eligible pairs | effect |
|---|---|---|
| ≥1 | 16 | admits all six one-document relations |
| **≥2** | **10** | **rejects all six one-document relations** |
| ≥3 | 8 | |

Keep the conservative **≥2 distinct supporting documents** threshold as the best
current corpus-derived candidate — but **do NOT mark it production-active**. The
relation inventory is still small, and the absence of bidirectional relations in
the current corpus **is not a domain guarantee**.

#### 10 · Partial evaluation DB limitation

> **`festival-verify` is not a complete corpus ingest.**

| | |
|---|---|
| total disclosures | **4,204** |
| with zero chunks in the evaluation DB | **3,022** |
| holding disclosures | **1,083** |
| with no reporter projection in the evaluation DB | **703** |
| linked issuer/reporter pairs with no issuer holding documents available | **10 of 16** |

The target **에스엠** pair is complete — **16 / 16** issuer holding documents
available.

Therefore the relation inventory **can** be reconstructed from the processed
corpus artifact, but **full downstream safety across all 16 pairs cannot
currently be demonstrated from `festival-verify`**.

**Do NOT claim all relation pairs were retrieval-tested.**

#### 11 · Route / cardinality firewalls

**Non-holding route firewall: PASS** — `exchange`, `major`, `periodic` and
correction/general lookup are all **not eligible**.

**3+ company firewall: PASS** — no best-pair selection, no company discarded.

#### 12 · Comparison precedence

**FAIL.** The proposed contract depends on `company_comparison == false`, but
current comparison detection is **not complete enough to be a safe firewall**.

P0-D.2 must **not** activate until the comparison-intent gap is repaired and
**independently regression-tested**.

> **RESOLVED by *Comparison Intent Firewall* (`7a0921e`, FINAL FREEZE).** The
> gap is repaired and independently regression-tested, and the precedence
> invariant *comparison firewall before `PRIMARY_COMPANY_WITH_REPORTER`* is now
> pre-armed in the validator. **P0-D.2 activation remains deferred on Target
> B.**

#### 13 · Relation performance

| operation | measured cost |
|---|---|
| per-process DB relation `GROUP BY` construction | **~2,018.8 ms** |
| per-query role lookup once constructed | **~0.013 ms** mean |

Building the relation from SQL per request is **NOT acceptable**. A future
activation should use a small **precomputed corpus-versioned relation artifact**
or an equivalent cached deterministic representation.

#### 14 · Future relation artifact — design finding only

**Do NOT mark implemented.**

```
(issuer_corp_code, reporter_key) -> support
                                 -> supporting document ids
```

Approximate current size: **~16 rows / ~3.2 KB**. It **must** carry
corpus/version identity, with a **fail-closed stale policy**:

> artifact corpus identity != active corpus identity
> → issuer/reporter role resolution unavailable
> → the query remains **AMBIGUOUS**

**Never silently use stale role relationships.**

#### 15 · Activation gate result

| gate | result |
|---|---|
| retrieval residuals have sufficient requested-event evidence | **FAIL** |
| comparison precedence preserved | **FAIL** |
| false-positive sweep clean | **FAIL** |
| downstream event matching fully correct | **FAIL** |
| activation requires no further retrieval/ranking work | **FAIL** |

Other major firewalls passed.

> **P0-D.2 activation is NOT authorized.**

#### 16 · Two independent next targets

**TARGET A — COMPARISON INTENT FIREWALL** · owner: query understanding /
validation. Ensure genuine two-company comparison questions are recognized as
comparison **before** issuer/reporter role resolution can run. This target must
be **generic**, not relation- or company-specific.

**TARGET B — H01 / HX02 RETRIEVAL-RANKING RESIDUAL** · owner: retrieval / final
evidence selection. The facts are structurally present but lost before the
emitted context. This requires a **separate diagnosis**.

> **Do NOT reopen P1-R or P1-B automatically.** Determine the exact first-loss
> stage before changing any frozen retrieval contract.

#### 17 · Next priority

1. Comparison Intent Firewall
2. H01 / HX02 ranking diagnosis
3. P0-D.2 safety recheck again
4. only then, if all gates pass, role-relation artifact implementation /
   activation work

**Do NOT implement the relation artifact while activation remains blocked.**

#### 18 · Negative invariants

Until activation is authorized:

- `PRIMARY_COMPANY_WITH_REPORTER` remains **production-inactive**;
- the two-company ambiguity rule remains **unchanged**;
- **no** relation artifact is used for production role resolution;
- **no** Gold / company / question special cases;
- **no** retrieval change as part of this documentation;
- **no** comparison query may be converted into an issuer/reporter lookup;
- **no** exact-Gold miss may be called answer-equivalent when the requested
  values differ;
- the **HX04 acquisition freeze remains intact**.

#### 19 · Status after this recheck

**P0-D.2 Issuer / Reporter Role Resolution: ACTIVATION DEFERRED — SAFETY
RECHECK REQUIRED.**

Current blockers:

1. **comparison-intent firewall** — **RESOLVED.** See *Comparison Intent
   Firewall* (`7a0921e`, FINAL FREEZE). Target A is closed, and closing it did
   **not** activate P0-D.2.
2. **H01 / HX02 retrieval-ranking residuals** — **STILL OPEN.** This is now the
   sole remaining blocker, and diagnosis must precede any retrieval change.

> **Corrected later by Target B first-loss diagnosis.** Item 2 records the
> status at Phase 3. Diagnosis is now complete and classifies both frozen 2024
> targets as **GOLD_TARGET_CONTEXT_DEPENDENT**. No retrieval change is
> authorized from these questions alone.

**HX04 Acquisition Semantics remains FINAL FREEZE.**

### Target B — H01 / HX02 First-Loss Diagnosis

**Status:** **DIAGNOSIS COMPLETE — BENCHMARK TARGET SEMANTICS REQUIRE
REDEFINITION.**

**Decision:** **BENCHMARK TARGET CONTEXT DEPENDENT.** This is a diagnosis and
documentation result, **not** a retrieval fix. P0-D.2 remains production-inactive.

#### Diagnosis baseline

| | |
|---|---|
| branch | `taeyoon` |
| diagnosis HEAD | `c3d59294dca045ba31cf662f99c66183eb9613e2` |
| full-suite baseline | **1686 passed · 13 skipped · 1139 subtests** |
| tracked worktree | clean |
| embedding provider | `bge_m3_local` |
| model | `BAAI/bge-m3` |
| revision | `6892b95fed65c899a30896eb40d619ae284d0455` |
| dimensions / device | 1024 / `cuda` |
| per-question role-sim vector coverage | **725 / 725 · 1.000** |

#### Correct Phase 3 role simulation and baseline

The corrected harness begins from the validator-normalized two-company plan,
reconstructs the corpus-supported direction, and rewrites the roles together:

```
companies  = ("에스엠",)
corp_codes = ("00260930",)
reporter   = "하이브"
```

The singular `company` / `corp_code` fields are recomputed consistently. The
six controls carry `comparison_frame = None` and
`role_reinterpretation_blocked = false`; the frozen Comparison Intent Firewall
does not block them.

The production-faithful simulation reproduces the Phase 3 document ranks
exactly:

| question | intended document rank |
|---|---:|
| H01 | **MISS** |
| H02 | 1 |
| HX01 | 2 |
| HX02 | **MISS** |
| HX03 | 7 |
| HX04 | 1 |

There is **no retrieval drift from Comparison Intent Firewall**.

#### HX02 Gold correction

Phase 3 misstated HX02's expected values. The actual frozen definition is:

| | |
|---|---|
| query | `에스엠 하이브 직전보고 보유주식 수 비율` |
| document | `holding_20240314001102` |
| table | `t0012` |
| previous shares | **`2,098,811`** |
| previous ratio | **`8.81%`** |

Those are the **previous-report fields** inside the 2024-03-14 filing.
`2,967,759 / 12.45%` are its **current holdings** and are not HX02-equivalent.
Historical text is annotated above rather than erased.

#### H01 semantic finding

H01 asks:

```
에스엠 하이브 이번 보고 보유 주식수와 비율
```

The frozen Gold binds this wording to the 2024-03-14 filing and
`2,967,759 / 12.45%`. Query understanding emits
`period_type = latest_holding`, but the standalone question contains no report
date, receipt number, document id or other filing anchor. The frozen holding
selector does not execute `latest_holding`; labels such as current/latest/first/
last remain inert for event selection.

The same-reporter timeline contains materially different states:

| reference date | current shares / ratio |
|---|---|
| 2023-02-09 | `4,392,368 / 18.45%` |
| 2023-03-13 | `4,626,185 / 19.43%` |
| 2023-04-03 | `2,098,811 / 8.81%` |
| 2024-03-14 | `2,967,759 / 12.45%` |
| 2024-06-04 | `2,212,237 / 9.38%` |
| 2025-06-05 | `0 / 0%` |

The public wording therefore does **not** uniquely identify the frozen filing.

> **H01 classification: `GOLD_TARGET_CONTEXT_DEPENDENT`.** If “이번 보고” is
> defined as the latest same-reporter report in the active corpus, the frozen
> 2024 target is also stale.

#### HX02 semantic finding

HX02 asks:

```
에스엠 하이브 직전보고 보유주식 수 비율
```

Its frozen values are the fields previous to the 2024-03-14 report: previous
date 2023-04-03, `2,098,811 / 8.81%`. The question does not identify which
current report “직전보고” is relative to. Relative to the latest HYBE report,
the applicable previous state would instead be the 2024-06-04 state,
`2,212,237 / 9.38%`.

> **HX02 classification: `GOLD_TARGET_CONTEXT_DEPENDENT`.** The frozen 2024
> filing is supplied by benchmark construction, not by the standalone wording.

#### Technical first loss — recorded separately from semantic ownership

Both intended structured projections remain in the scored candidate set and
fall outside the normal final top 10:

| rank / score field | H01 `holding_report` | HX02 `holding_report` |
|---|---:|---:|
| lexical rank | outside top 50 | 13 |
| vector rank | 12 | 13 |
| fusion rank | 37 | 15 |
| production legacy final rank | 37 | 15 |
| preservation / bounded rank | 23 | 15 |
| normal top 10 | MISS | MISS |

The measurable pipeline first-loss stage for both is **FINAL TOP-K
TRUNCATION**. That is a technical symptom, not the primary correctness owner:
the desired 2024 target is not semantically selected by either question.

#### Query-representation secondary finding

When issuer and reporter are both recognized as companies, both company spans
are removed from the retrieval text. After role simulation:

```
H01 lexical_query  = "이번 보고 보유 주식수와 비율"
HX02 lexical_query = "직전보고 보유주식 수 비율"
```

The known reporter `하이브` remains only in `plan.reporter`. It is absent from
the lexical query, vector query, backend filters and deterministic ranking
score, while the candidate scope contains filings for many SM reporters. HX02
also asks for a ratio, but its current requested-field representation contains
share fields rather than an explicit previous-ratio field.

This is a real **QUERY_REPRESENTATION** asymmetry, but it is not implementation
authorization.

#### Reporter-in-query ablation — diagnostic only

Adding reporter text back to the retrieval query produced:

| question | result |
|---|---|
| H01 | intended document MISS → rank 7, but fully answer-equivalent `2,967,759 / 12.45%` evidence was still not recovered |
| HX02 | intended document MISS → rank 6; `2,098,811 / 8.81%` recovered |
| HX03 | document rank 7 → 3 |
| HX01 | remained successful |
| H02 | remained successful |
| HX04 | remained successful; `2024-03-07 / 868,948` acquisition semantics preserved |

Reporter representation is materially relevant, especially for HX02, but the
ablation does not provide the missing report anchor and is **not** an approved
production fix.

#### Deterministic ranking and P1-R audits

Correct and incorrect `holding_report` projections repeatedly receive the same
deterministic score, approximately **0.382667**. Current deterministic features
do not distinguish reporter or execute `latest_holding` chronology, so generic
holding-state questions depend heavily on lexical/vector relevance.

P1-R behaved according to its FINAL FREEZE contract:

- H01 triggered crowding recovery, which appended the highest-ranked unseen
  document already in the scored tail; the intended target was much deeper.
- HX02 had no crowding trigger.

P1-R is **not reopened**: no cap widening, second append, replacement or score
change is authorized.

Top-k sensitivity remains diagnostic only:

| cutoff | H01 | HX02 |
|---:|---:|---:|
| 5 | miss | miss |
| 10 | miss | miss |
| 11 | miss | miss |
| 15 | miss | hit |
| 20 | miss | hit |
| 30 | miss | hit |
| 50 | hit | hit |

No global `top_k` increase is proposed.

#### Real downstream and forced-evidence validation

The production orchestration path — structured holding coverage,
`EvidenceBuilder`, `HoldingEventResolver`, event selection and composer — was
used, not an EvidenceBuilder-only approximation. HX04 reproduced the frozen
control:

| | |
|---|---|
| matching event count | **1** |
| acquisition date | **`2024-03-07`** |
| acquired shares | **`868,948`** |

Force-including the intended structured projections allowed the real resolver
to construct H01 `2,967,759 / 12.45%` and HX02 previous fields
`2,098,811 / 8.81%`. Both events matched the reporter constraint. The queries
nevertheless remained non-unique because neither supplies an exact report
selector.

Therefore **EVENT_CONSTRUCTION, REPORTER_MATCHING and COMPOSER are not owners**.

#### Owner taxonomy

| question | primary owner | secondary contributors |
|---|---|---|
| H01 | **GOLD_TARGET_CONTEXT_DEPENDENT** | semantic target ambiguity · query representation · lexical/vector/fusion ranking and top-k · P1-R scope |
| HX02 | **GOLD_TARGET_CONTEXT_DEPENDENT** | semantic target ambiguity · reporter and requested-ratio representation · lexical/vector/fusion ranking and top-k · P1-R scope |

Not owners: candidate generation · metadata exclusion · vector coverage ·
structured event construction · reporter matching · composer.

#### Historical “ranking blocker” correction

The Phase 3 / Target B label **H01 / HX02 retrieval-ranking residual** was too
broad. Both questions exhibit measurable ranking and top-k symptoms, but the
frozen 2024 target itself depends on absent document context. Consequently, no
ranking change may be authorized from these questions alone. Historical wording
is retained with correction annotations for comparability.

#### Benchmark redefinition options — design only

**Option A — explicit report anchors.** For example:

```
H01: 2024년 3월 14일 보고서 기준 에스엠에 대한 하이브의 보유 주식수와 비율
HX02: 2024년 3월 14일 보고서의 직전 보유주식 수와 비율
```

**Option B — define a public latest-same-reporter holding-report contract** and
regenerate expected answers from the active corpus.

**Option C — supply explicit document/report context** through evaluation or
API context.

No option is selected or implemented here. Historical Gold60 definitions must
not be silently mutated: a later correction should use a versioned semantic
validation set or Gold revision so historical retrieval baselines remain
comparable.

#### Next production design question and reporter target sequencing

The next question is **not** “How do we force H01/HX02 Gold into top 10?” It is:

> What report-relative semantics should the public agent implement for
> `이번 보고`, `현재`, `직전보고` and `직전 보고 대비` after issuer and
> reporter are resolved?

A bounded reporter-aware query-representation target may exist because a
reporter that is itself a recognized company disappears from lexical/vector
input. It must be evaluated **only after** report-relative semantics are
defined. Reporter propagation remains unauthorized.

#### Negative invariants after diagnosis

- Never force the 2024 Gold merely because the benchmark expects it.
- Never treat `latest_holding` as 2024-specific.
- Never use Gold document ids in ranking.
- Never increase `top_k` to solve a semantically unanchored query.
- Never reopen P1-R or P1-B from these cases alone.
- Never confuse HX02 current and previous holdings.
- Never treat `2,967,759 / 12.45%` as HX02 evidence.
- Never implement reporter propagation before report semantics are defined.
- Never activate P0-D.2 from this diagnosis.
- Never silently rewrite historical Gold60.

#### Status after Target B diagnosis

| | status |
|---|---|
| Target A — Comparison Intent Firewall | **FINAL FREEZE** |
| Target B — H01/HX02 First-Loss | **DIAGNOSIS COMPLETE — BENCHMARK TARGET CONTEXT DEPENDENT** |
| decision | **BENCHMARK TARGET SEMANTICS NEED CORRECTION / REDEFINITION** |
| P0-D.2 | **ACTIVATION DEFERRED** |
| `PRIMARY_COMPANY_WITH_REPORTER` | **production-inactive** |

This diagnosis does **not** establish that P0-D.2 is safe.

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

## P1-A4.1 — Holding Reporter Normalization

**Status:** FINAL FREEZE

**Frozen commit:** `96d3968`

A narrow reopened extension of P1-A4 above. The P1-A4 freeze is **unchanged**:
its D1 exact-date, D2 fusion and D2b reporter-alternative contracts stand
exactly as recorded, and this entry only replaces the string predicate they call.

### Problem solved
Reporter matching handled family suffixes — `국민연금` ↔ `국민연금공단` — but
treated Korean legal-designator variants as different holders:

```
하이브              !=  (주)하이브
한국기업투자홀딩스   !=  (주)한국기업투자홀딩스
한국기업투자홀딩스   !=  주식회사 한국기업투자홀딩스
```

The defect is **generic and embedding-independent**. Measured over the holding
corpus: **1,083 documents**, **51,730 reporter occurrences**, **1,199 unique raw
reporter strings**, and a **~65.2% false-negative rate** on designator-removed
query-like forms under the previous matcher.

### Final architecture / behavior
One shared pure helper, `app/reasoning/holding_reporter.py`, exposing
`canonical_reporter_key` and `reporter_matches`. Both `holding_event_resolver`
and `answer_composer` delegate to it; `holding_event_fusion` inherits it through
the resolver wrapper.

```
raw reporter
→ strip at most one syntactically-recognizable legal designator from each outer edge
→ normalize case / punctuation / whitespace
→ empty key            => no match
→ exact canonical equality
→ existing family-suffix compatibility (공단 / 기금 / 조합 / 법인 / 회사)
→ otherwise false
```

Recognized legal forms, as implemented: `(주)`, `㈜`, `주식회사`, `(유)`,
`유한회사`, `유한책임회사`, `합자회사`, `합명회사`, `사단법인`, `재단법인`.

> ### ⚠ Stripping happens on the **raw** string, never after normalization
>
> Normalizing first deletes the brackets and leaves a bare `주` that is
> indistinguishable from the first syllable of `주성엔지니어링`. A bare Hangul
> `주` or `유` is **never** removed from an ordinary entity name — only a
> syntactically recognizable legal form is, and at most one per edge.

### Production files
- **NEW** `app/reasoning/holding_reporter.py`
- `app/reasoning/holding_event_resolver.py`
- `app/reasoning/answer_composer.py`

Tests: `tests/test_holding_reporter_normalization.py`

No change to query understanding, query validation, `QueryPlan`, retrieval, the
database, or the public API schema.

### Invariants that must remain true
1. The same raw inputs produce deterministic reporter-match results.
2. Legal-designator canonicalization is **comparison-only**; raw display
   reporter text is preserved.
3. **No substring containment.**
4. **No reporter alias table**, ever — as already frozen by P1-A4 D2b.
5. An empty canonical key matches nothing.
6. P1-A4 exact-date semantics are unchanged.
7. `국민연금` family semantics are unchanged.
8. Resolver and composer reporter matching stay **one** shared contract;
   duplicated independent normalizers must not reappear.
9. P1-A5 contracts remain unchanged.
10. The public response exposes exactly five top-level fields.

Also frozen as non-goals: no fuzzy matching, no LLM reporter matching, no
company-master / `corp_code` dependency, no company-specific exception, no
Gold- or question-specific exception.

### Placeholder semantics — an intentional correctness change
Placeholder-like reporters that normalize to empty, such as `-` and `…`, now
**match nothing**. The previous behaviour treated two empty-normalized
placeholders as the same reporter, asserting that two filings with no stated
holder described the same one.

### Verification performed

**Corpus-wide match audit** (1,199 × 1,199 ordered pairs)

| | |
|---|---|
| before | 1,531 |
| after | 1,573 |
| newly true | **46** |
| newly false | **4** |

The 46 newly-true pairs form **15 canonical collision groups**, each reviewed
and all legal-designator or spelling variants of the same entity — for example
`삼성생명보험` ‖ `삼성생명보험(주)` (4,492 occurrences) and `삼성물산` ‖
`삼성물산 주식회사` ‖ `삼성물산(주)` ‖ `삼성물산주식회사`. **No semantically
distinct canonical collision was found.** The four newly-false pairs are
combinations of empty placeholder reporters: `("-", "-")`, `("-", "…")`,
`("…", "-")`, `("…", "…")`.

**Non-Gold generic proof** — a `고려아연` query whose reporter is
`한국기업투자홀딩스`, where corpus reporter forms carry legal designators:

| | events | matches_query | rendered |
|---|---|---|---|
| before | 10 | **0** | **0** |
| after | 10 | **10** | **10** |

No such company literal exists in production matching logic; the names appear
only in diagnostics and tests.

**Frozen holding regression** — HX05, HX09, HX10, HX12, HX13, HX16, HX17, HX20:
**0 / 8 semantic changes**. Exact-date controls unchanged: HX09 → `2022-12-05`,
HX13 → `2023-06-13`, HX17 → `2023-06-30`. P1-A4 exact-date semantics intact.

**P1-A5 regression** — P1-A5-B matching-only rendering unchanged; P1-A5-A
ambiguity semantics unchanged; P1-A5-A.1 HCX semantic-control guard unchanged.
Targeted suites all passed.

**Mutation evidence** — mutations were applied to real source and caught for:
containment; missing empty-key guard; missing prefix stripping; missing suffix
stripping; resolver/composer matcher divergence; arbitrary bare `주`/`유`
stripping. Each is what fixes the semantic boundary in place.

**Test baseline** — **1488 OK, skipped 13** before; **1516 OK, skipped 13**
after. 28 new reporter-normalization tests, zero regressions.

### P0-D.1 downstream compatibility
P1-A4.1 proves **downstream reporter compatibility** for the issuer/reporter
representation diagnosed in P0-D.1 — `company = 에스엠`, `reporter = 하이브`
against corpus reporter `(주)하이브`. Offline downstream controls:

| question | matching events |
|---|---|
| H01 | 0 → 1 |
| H02 | 0 → 1 |
| HX01 | 0 → 1 |
| HX02 | 0 → 4 |
| HX03 | 0 → 1 |
| HX04 | reporter compatibility preserved, matching event available |

> **These six Gold questions are NOT fixed end-to-end.** P0-D still declines
> them, and **P0-D remains independently frozen** until separately reopened.
> HX02 remains an undated multi-event shape, governed by existing P1-A5-A
> ambiguity semantics.

### Known residual issues NOT solved
- The legal-designator vocabulary is intentionally finite and Korean-focused.
- Latin `Inc` / `Ltd` forms remain outside this normalization.
- Widening reporter matching may expose pre-existing **undated multi-event**
  questions, which remain governed by frozen P1-A5-A.
- P1-A4.1 alone does **not** solve the six P0-D-declined Gold questions.
- P0-D role resolution remains a separate future phase.

### Reopen conditions
- a reproducible reporter false positive merging two distinct holders;
- a demonstrated exact-date or 국민연금 family regression;
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

## P1-A5-A — Ambiguity-Safe Holding Presentation

**Status:** FINAL FREEZE

**Frozen commit:** `39574f9`

### Problem solved
A holding answer could present one event as the answer when the question had
never identified one. The count of matching events is a fact about what
retrieval served, not about what was asked: measured on the corpus, a holder
alone leaves more than one event in 84% of cases, a year in 71%, a direction in
69%. A single observed event is routinely produced by which projections happened
to be returned — promoting more of the already-fetched pool took HX12 from one
matching event to nine without the question changing at all.

### Final architecture / behavior

**Core invariant** — `observable_matching_event_count == 1` does **not** imply
semantic uniqueness.

```
semantic_unique = exact query-visible selector AND matching_event_count == 1
```

**Selector taxonomy**

| class | signals |
|---|---|
| **EXACT_SELECTOR** | exact holding reference date only |
| **FILTER_ONLY** | reporter · year · quarter · range · receipt date · direction |
| **FIELD_REQUEST** | 변동일 · 변동전 · 변동후 · 주식수 · 비율 |
| **INERT** | 직전보고 · 현재 · 최근 · 최신 · 최초 · 마지막 |

**Presentation**

| case | result |
|---|---|
| EXACT + 1 | single event, **no notice** |
| non-EXACT + 1 | the event + under-specified **CASE A** |
| non-EXACT + >1 | all matching events + **CASE B** |
| EXACT + >1 | all matching events + **CASE C**, no pick |
| history path (no requested fields) | full timeline, no notice |

A pure helper classifies the mode from the plan alone; the composer consumes it
and records structured flags; the generator selects the deterministic sentence.

### Production files
- `app/reasoning/holding_event_selection.py` (new)
- `app/reasoning/answer_composer.py`
- `app/agent/orchestrator.py`
- `app/generation/answer_generator.py`

Tests: `tests/test_holding_event_ambiguity_presentation.py`

### Invariants that must remain true
- A count of one is never, on its own, treated as semantic uniqueness.
- Only an exact holding reference date qualifies as an EXACT_SELECTOR, read
  through P1-A4's existing helper — no second date parser.
- Filter-only signals narrow events but never establish uniqueness.
- Field requests are never event selectors.
- The inert labels select nothing until a later phase gives them semantics.
  `최초` and `마지막` are opposites that produce the same period label, which is
  the evidence they select nothing.
- The classifier reads the plan only — never rank, evidence, events, or `top_k`.
- No event is chosen for ranking first, for being newest, or for being oldest.
- **Answerability remains independent**: under-specification is presentation,
  not an evidence failure.
- **P0-D clarification remains untouched**; no new clarification flow.
- P1-A5-B matching-only rendering is preserved: non-matching events never return.

### Verification performed
- Full suite: **1450 OK, skipped 13** (confirmed at `39574f9`).
- `tests/test_holding_event_ambiguity_presentation.py`: **33 OK**.
- Mutation checks: forcing every mode to EXACT fails 14; removing the CASE A
  notice fails 4; reusing the old multi-event wording for one observed event
  fails 5; making the notice force verbose rendering fails 1; collapsing
  `matching > 1` to the first event fails 13.
- Retrieval-shape stability: one and nine matching events from the same plan
  carry the same semantic claim.

### Known residual issues NOT solved
- The inert labels still carry no selection semantics; giving them any is a
  separate phase with its own diagnosis.
- Year-month is still reduced to year-only by frozen P0-D.
- Every non-EXACT holding answer now carries a notice, which changes answer
  length broadly and warrants a deliberate Gold60 re-baseline.

### Reopen conditions
Proven regression, a citation/provenance break, or a later architecture that
cannot preserve this contract additively.

---

## P1-A5-A.1 — Lossless Semantic Notice Preservation

**Status:** FINAL FREEZE

**Frozen commit:** `39574f9`

### Problem solved
HCX compact verbalization does not include the deterministic semantic-control
notice, and its reply replaces the whole deterministic answer — so a successful
rewrite silently dropped the notice. Proven, not inferred: `build_compact_claim`
renders only verified factual fields, so the notice is absent from
`claim.deterministic_text`, from the text sent to the model, from
`required_terms`, and from every expectation the lossless validator compares
against. A perfectly compliant stub model reproduced the live HX12 failure
exactly.

### Final architecture / behavior

```
if draft.ambiguity["under_specified"] or draft.ambiguity["exact_multi_match"]:
    the HCX rewrite must not become the final answer
```

A structured boolean guard inside `HcxVerbalizer`, returning the deterministic
answer with status `skipped_semantic_control_notice`. No Korean substring
matching, no prompt-only enforcement, no post-HCX text surgery.

The guard sits as the **last gate before the model call**, so the established
earlier skip statuses retain their precedence and no HCX request is ever made.

### Production files
- `app/generation/hcx_verbalizer.py`

Tests: added to `tests/test_hcx_verbalizer.py`.

### Invariants that must remain true
- The guard reads `under_specified` / `exact_multi_match` structurally. It must
  never match on the notice text: the wording belongs to the generator, and a
  verbalizer holding a copy of a sentence it does not write would drift.
- No HCX request occurs once the guard fires — asserted against the transport's
  own call log, not inferred from output text.
- Established earlier skip statuses (`disabled`, `not_configured`,
  `skipped_not_answerable`, `skipped_no_compact_verified_claim`,
  `skipped_multi_event_compact_claim`) keep precedence.
- A draft without an `ambiguity` mapping behaves exactly as before.
- The deterministic answer is returned byte-identical, so citations and every
  rendered section survive.
- Preservation must not depend on model compliance, nor on how much evidence
  retrieval happened to serve.
- `skipped_semantic_control_notice` is a new **value** for the existing
  `think_trace.hcx_status`; no schema change.

### Verification performed

- Full suite: **1450 OK, skipped 13**.
- `tests/test_hcx_verbalizer.py`: **59 OK**.
- Mutation checks: disabling the predicate fails 6; checking only
  `exact_multi_match` fails 4; checking only `under_specified` fails 3;
  detecting the notice but calling HCX anyway fails 6.

**Live BGE-M3 verification**

| qid | result |
|---|---|
| HX12 | 2024-02-19 · CASE A present · answerable · 2 citations · `skipped_semantic_control_notice` |
| HX10 | CASE B present · 8 citations · answerable, regardless of compact-claim eligibility |
| HX16 | CASE B present · 9 citations · decrease-only |
| HX20 | CASE B present · 6 citations · decrease-only |
| HX13 | 2,202,050 / 7.90% · no ambiguity notice |
| HX17 | 1,092,455 / 6.99% · no ambiguity notice |

HX16/HX20 keep the existing no-compact skip; HX13/HX17 take no semantic-control
skip, so exact-date behaviour is preserved.

**P0 correction regression** — PASS.

### Known residual issues NOT solved
- HCX's holding surface is now narrow: it runs only for exact-date,
  single-event, conflict-free answers. P1-A4 fusion marks fused events
  `field_conflict=True`, so offline no Gold holding question reaches the model.
  This predates the phase but is now structural.
- Two guard patterns coexist for one concern: this skip, and
  `_preserve_multi_document_semantics`'s post-hoc reject-and-fall-back in
  `pipeline.py`. Worth unifying later.

### Reopen conditions
Proven regression, a semantic-control notice reaching production stripped, or a
later architecture that cannot preserve this contract additively.

---

## P1-B — Filter Relaxation / Retrieval Recovery

**Status:** DIAGNOSIS COMPLETE / **IMPLEMENTATION DEFERRED — STRENGTHENED BY
LIVE BGE-M3 EVIDENCE (INFRA-E1)**

This is not a freeze. Nothing was built, so there is no frozen contract to
protect — only a measured decision not to build, and the conditions that would
change it.

> ### Live BGE-M3 reinforcement
>
> Under real BGE-M3, **all 22** evaluable holding questions had the **gold
> document present in the candidate set** and **full expected vector candidate
> coverage** (11,925 of 11,925 rows). The three remaining final misses are
> therefore **neither `FILTER_EXCLUSION` nor evidence-coverage failures** — they
> belong to **P1-R**. MODE A on the complete 17-company evaluation database
> likewise reproduced `FILTER_EXCLUSION = 0` with structural inclusion 54/54.
>
> **P1-B remains implementation-deferred.**

### Question asked
Should retrieval recover from an over-restrictive filter by relaxing filters and
searching again?

### What was measured
Against the complete 4,204-row disclosure table, and a disposable corpus holding
all four doc groups for five companies (222 documents, 85,493 chunks):

- Gold60 evaluable questions whose gold document is already in the **strict**
  candidate set: **54 / 54**.
- Strict filter exclusions (a gold document dropped before anything scored it):
  **0**.
- Holding questions served: **22 / 22**, **Recall@10 = 1.00**, with periodic,
  major and exchange chunks competing.
- `reporter` is **not a retrieval filter** — it is absent from
  `QueryPlan.backend_filters()` and lives only in the resolver.
- Relaxing **company** — Recall@10 1.00 → 0.64, Recall@1 0.50 → 0.23, 160
  wrong-company results served, 5.5× chunks, 3.8× latency.
- Relaxing **doc_group** — Recall@10 1.00 → 0.86, 32× chunks, 15× latency.
  (Note: `doc_group` is enforced twice, in SQL through `backend_filters` and
  again post-SQL through `hard_routes`; removing it from one alone does nothing.)

Every relaxation degraded recall, precision and latency together. There was
nothing for a relaxation ladder to recover.

### Decision
**Do not implement relaxation now.**

The remaining measured headroom is elsewhere and belongs to other work: Recall@1
is 0.50 while Recall@10 is 1.00 (a ranking matter), and six Gold60 questions are
declined by P0-D for company disambiguation, not by any retrieval filter.

### Reopen only if
- a real `STRICT_ZERO` case is demonstrated — the candidate set is empty after
  filtering; or
- hidden/live evaluation proves a document is excluded **before ranking**.

A wrong answer is not evidence of a filter problem. The distinction that matters
is whether the document was excluded before scoring, ranked too low, or lost
downstream; only the first is a filter problem.

### Future fallback, if it is ever needed
- strict-zero trigger only;
- maximum **2** attempts;
- relax **optional inferred metadata only** (`doc_subtype`, `section_path`,
  inferred `is_correction`);
- **company, corp_code, doc_group and date remain hard**;
- union the results, strict candidates keeping their ranks ahead of relaxed ones;
- internal trace only, no public schema change.

---

## P1-R — Bounded Additive Document Recovery

**Status:** **FINAL FREEZE**

**Implementation commit:** `6503c77`

**Regression-test commit:** `283edfb`

> ### Scope of this freeze
>
> This FINAL FREEZE covers **only the bounded additive document recovery
> contract**. It does **not** mean all ranking failures are solved: **HX08 and
> HX16 remain unrecovered**, and **P0-D.2 remains activation-deferred**.

The diagnosis history below is retained deliberately — Phase 1 was hash-based and
diagnostic only, INFRA-E1 later enabled real pinned BGE-M3 validation, and a
replacement-style cap was implemented and rejected before the additive contract
was reached.

> ### ⚠ Do not scope a reranker from the Phase 1 numbers below
>
> **R@1 = 0.50 is not a production measurement.** It was produced with
> `DeterministicHashEmbedder`, whose vector channel is pseudo-random.
> Every Phase 1 ranking figure in this entry inherits that caveat unless marked
> embedding-independent.
>
> The live value is **no longer unknown** — see **Phase 1.5 live BGE-M3 results**
> at the end of this entry. The hash and live figures cover **different question
> subsets** and **must never be combined into one metric**.

### Phase 1 — what was measured

A 22-question holding evaluation (the Gold60 subset whose gold document sits in
the seeded corpus; 6 further questions are declined by P0-D and excluded), run
against a disposable corpus of five companies across all four doc groups
(222 documents, 85,493 chunks), with `DeterministicHashEmbedder`:

| | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| **hash baseline (final served)** | **0.50** | 0.77 | 0.82 | **1.00** | 0.674 |
| lexical only (production-real BM25) | 0.45 | 0.55 | 0.55 | 0.59 | 0.491 |
| + document dedupe, cap 1/doc | 0.50 | 0.82 | **0.91** | 1.00 | 0.682 |

Findings:

- **11 / 11 rank-1 misses were same-company, same-`doc_group`, same-chunk-type
  sibling disclosures** — mean length 376 chars at rank 1 against 363 for the
  gold chunk. No wrong-company and no wrong-doc_group result reached rank 1.
- **Document dedupe raised R@5 from 0.82 to 0.91 and unique-docs@10 from 6.3 to
  9.6, but left R@1 at exactly 0.50.** Caps of 2 or 3 changed nothing.
- **Crowding exists but is not the measured R@1 cause**: 11/22 queries had one
  document holding ≥3 of 10 slots, yet removing that crowding moved no
  rank-1 outcome.
- **No deterministic lightweight method tested improved R@1** — dedupe,
  aggregation, type priors and term-overlap boosts are all blind to the one
  distinction that matters, because the competing documents differ only in
  which event they report.
- The existing `_hybrid_rerank` is a deterministic heuristic scorer
  (0.60 × fusion + 0.40 × metadata), not a model, and has no signal that
  separates sibling filings.

Embedding-independent among these: the sibling-disclosure failure shape, the
chunk-type and length parity, the existence of crowding, and lexical-only
R@10 = 0.59. The exact rank values are not.

### Phase 1.5 — why live validation stopped

*Historical: this records the state at Phase 1.5 diagnosis time. **INFRA-E1 has
since supplied the environment (condition 3 below) and live validation is
complete** — see **Phase 1.5 — LIVE BGE-M3 results** further down.*

BGE-M3 was unreachable by every path the code supports:

- `FlagEmbedding` / `torch` / `transformers` absent from the runtime;
- no local BGE-M3 weights (the HF cache holds only two Korean sentence models);
- `FESTIVAL_EMBEDDING_API_URL` / `FESTIVAL_EMBEDDING_API_KEY` unset, so the HTTP
  and Clova providers cannot be used;
- `EmbeddingConfig.from_env()` therefore resolves to `provider="hash"`.

Substituting hash embeddings, or a different cached model, would have
reproduced the invalid baseline while dressing it as live validation. Nothing
was started; port 8010 was untouched.

Validating ranking also needs the **corpus** embedded with BGE-M3, not only the
queries — mixing BGE-M3 query vectors against hash chunk vectors is worse than
either channel alone. That makes this a data-pipeline prerequisite, not a smoke
test.

### Decision
**Do not scope or implement a reranker from the hash result.**

### Reopen only after one of
1. a live BGE-M3 endpoint becomes available
   (`FESTIVAL_EMBEDDING_API_URL` + `FESTIVAL_EMBEDDING_API_KEY`);
2. a BGE-M3 pre-embedded corpus/database becomes available
   (`chunk_embeddings.embedding_model = BAAI/bge-m3`);
3. a dedicated disposable BGE-M3 evaluation environment is prepared.

**Condition 3 was satisfied by INFRA-E1**, and the 22-question live smoke has been
run exactly as required — unchanged in questions, gold documents, routing,
filters and `top_k`. Its outcome is the **LIGHTWEIGHT TARGET** verdict recorded
below; the decision table that follows is retained as the rule that produced it.

**At reopen, run the same 22-question live smoke first**, unchanged in
questions, gold documents, routing, filters and `top_k`, and compare against
the hash baseline above.

Then decide:

| live smoke result | action |
|---|---|
| ranking already strong | close P1-R with **no implementation** |
| a deterministic ranking defect is found | **lightweight fix** only |
| semantic sibling discrimination still weak | **only then** test a small top-N cross-encoder |

### External design reference
Dart-Agent's published reranker experiment improves ranking metrics but adds
roughly **11 seconds** of average latency. Against Festival's measured retrieval
budget (221 ms mean, 285 ms p95) that is a warning, not a template: a heavy
cross-encoder needs strong Festival-specific evidence before it is considered.

### Ownership firewall
Symptoms are owned by phase, and a weakness in one is not grounds to reopen
another:

| symptom | owner |
|---|---|
| filtered out **before scoring** | **P1-B** (deferred; 0 occurrences measured) |
| eligible but **ranked low** | **P1-R** (this entry) |
| retrieved but **sibling evidence missing** | **P1-C** / coverage |
| evidence correct but **reasoning wrong** | reasoning phases (P1-A4 / P1-A5) |

### Known gaps in the diagnosis
- Correction (P0-A) and corporate/multi-document (P0-B/P0-C) ranking controls
  were **not exercised** — their gold companies sit outside the seeded corpus.
- RRF/fusion ablation, source-disagreement analysis and any cross-encoder trial
  were **not run**: all require a meaningful vector channel.
- The evaluable set is 22 holding questions from five companies; non-holding
  ranking is unmeasured.
- Capping to one chunk per document would remove a second gold chunk in 16 of 22
  queries, which is a downstream risk for P1-A3 sibling anchoring.


### Phase 1.5 — LIVE BGE-M3 results (INFRA-E1)

Measured on the **22 evaluable holding questions** (the 28 holding questions minus
the 6 P0-D declines), under the pinned `BAAI/bge-m3` revision
`6892b95fed65c899a30896eb40d619ae284d0455`, at frozen production retrieval
defaults, with **no tuning**.

| hybrid | value |
|---|---|
| R@1 | **0.3636** |
| R@3 | **0.8636** |
| R@5 | **0.8636** |
| R@10 | **0.8636** |
| MRR | **0.5833** |
| nDCG@5 | **0.6553** |
| nDCG@10 | **0.6553** |

Latency: mean **269.7 ms**, p95 **388.4 ms**, max **515.6 ms**.

> **Do not combine these with the earlier HASH_DIAGNOSTIC numbers** — the prior
> hash subset was a different set of questions.

### Phase 1.5 — the failure shape
Final rank distribution: **rank 1 → 8 · rank 2 → 7 · rank 3 → 4 · miss → 3**.
**No question lands at final ranks 4–10.** A question is either found by rank 3
or not found at all.

Exactly three final misses — **HX08, HX16, HX20** — and all three share one
signature:

- the gold document **is** structurally inside the candidate set;
- a **BGE vector exists** for it;
- **lexical rank is absent**;
- **BGE vector rank is 14–25**;
- fusion fails to lift the document into the final top-10.

### Phase 1.5 — diagnosis

**P1-R = LIGHTWEIGHT TARGET.**

Current evidence supports a **deterministic fusion / candidate-cutoff issue**,
**not** a semantic reranker target — BGE-M3 does retrieve these documents, merely
too deep for fusion to recover when the lexical lane contributes nothing.

*Historical: at Phase 1.5 the solution was not yet known. It was reached in
Phases 2–5.1 below, after a replacement-style cap was tried and rejected.*


### Phase 2–5.1 — the frozen contract

### Problem solved
A filing whose table splits into several similar chunks can occupy most of the
final list, so a document retrieved just below it is never shown. Ranking scores
chunks, not documents, and nothing corrected for that.

> ### ⚠ REJECTED: replacement-style diversification (`cap = 2`)
>
> A per-document cap of **2** was implemented and measured under live pinned
> BGE-M3. It improved document recall and recovered HX20 — and was **REJECTED**,
> because it **replaced existing top-10 chunks and changed answer semantics**:
>
> - **HX07 lost three distinct dated events** (2023-05-18, 2023-06-05, 2023-06-09);
> - further distinct-fact losses on **H03**, **H08** and **HX15**.
>
> **Document/chunk replacement for diversity is NOT part of the frozen
> contract.** This is a negative invariant, not a discarded draft: any future
> diversity work must not reach for it again.

### Final architecture / behavior — bounded additive document recovery
Normal hybrid `top_k` stays **10** and the baseline top-10 is **immutable**.
After normal hybrid retrieval, the existing latest-event and statement-metric
rescues, and correction/event expansion, the recovery may append **at most one**
candidate.

**Trigger** — within the currently emitted evidence, one canonical document
occupies **at least 3** slots.

**Candidate** — the highest-ranked candidate from the existing scored tail whose
canonical `doc_id` is not already represented in the emitted context. It keeps
its own existing scored rank.

No score recomputation, no reranking, no second database query, no model call,
no fabricated candidate.

`_hybrid_rerank` still truncates its emitted output to `top_k`, but retains
diagnostics and scored information for the complete `final_order`. An internal
**`ScoredTail`** lets the post-retrieval rescue materialise one already-scored
candidate **without widening normal `top_k`**. `ScoredTail` is internal
retrieval machinery — not public API, not a new public result schema, and not a
second retrieval pass.

**Budget invariant.** Normal retrieval budget = existing `top_k` = **10**;
additive recovery allowance = **maximum 1**. Base evidence never exceeds the
normal budget, and expanded evidence may carry at most one extra chunk from this
rescue. **This contract must never be re-implemented by setting `plan.top_k` or
`final_top_k` to 11.**

**Document identity** comes only from the canonical `RetrievalResult` /
`FusedCandidate` `doc_id` — never inferred from `chunk_id` parsing, report name,
company or date. A missing `doc_id` neither contributes to crowding nor can be
selected as the rescue candidate.

### Production files
- `app/retrieval/hybrid.py`

Tests: `tests/test_additive_document_recovery.py`

No change to `app/retrieval/bge_m3.py`, query understanding, query validation,
P0-D, the holding event resolver, the answer composer, or the public API schema.

### Verification performed

**Model identity.** `BAAI/bge-m3` revision
`6892b95fed65c899a30896eb40d619ae284d0455`, 1024 dimensions, loaded through the
INFRA-E1 **fail-closed pinned-snapshot loader**. Full candidate-union coverage
was completed *before* global validation.

**Global BGE coverage** — the first complete LIVE pinned-BGE Gold60 retrieval
baseline:

| | |
|---|---|
| accepted Gold60 questions | **54** |
| vector-eligible candidate rows | **157,377** |
| matching stored pinned BGE rows | **157,377** |
| missing | **0** |
| questions with zero BGE candidates | **0** |
| authoritative candidate union | **75,786 chunks** |

**Global LIVE BGE baseline** — official frozen evaluator, `legacy` production
rerank mode. These are **BASE normal retrieval metrics**:

| group | n | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|---|
| exchange | 8 | 0.7500 | 1.0000 | 1.0000 | 1.0000 | 0.8542 |
| major | 10 | 0.8000 | 0.9000 | 0.9000 | 1.0000 | 0.8667 |
| holding | 22 | 0.4091 | 0.7273 | 0.7273 | 0.8636 | 0.5537 |
| periodic | 14 | 0.2857 | 0.3571 | 0.5000 | 0.6429 | 0.3661 |
| **all accepted** | **54** | **0.5000** | **0.7037** | **0.7407** | **0.8519** | **0.6075** |

nDCG@10 across all accepted: **0.6660**.

**Additive recovery impact** across the 54 accepted questions:

| | |
|---|---|
| rescue triggered | **19 / 54** |
| appended | **13 / 54** |
| normal baseline top-10 byte-identical | **54 / 54** |
| Gold document newly recovered | **1 — HX20** |

In the document-level A/B harness, base document R@10 **0.9074** and expanded
evidence recall **@11 0.9259**. For holding, base R@10 remains **0.8636** and
expanded evidence **@11** becomes **0.9091**.

> **Expanded evidence recall is never R@10.** Do not write "R@10 improved to
> 0.9091", and keep the document-level harness figures separate from the
> official chunk-level retrieval metrics above.

**HX20 — validation evidence, not a production case.** Its gold document was
outside the normal emitted top-10; the recovery appends
`holding_20230704000260` from existing scored **rank 11**, yielding the grounded
event **2023-06-30, 1,092,455 shares**. No prior fact disappears. Production
code contains no HX20, document, or date special case.

**Evidence preservation — the decisive distinction from the rejected cap.**
Across all 54 accepted questions: baseline top-10 prefix identical **54/54**,
**lost grounded facts 0**, harmful or contradictory additions **0**. Of the
additions: **2 useful grounded** (HX20's intended event; H03 one additional
grounded event), **11 redundant with no downstream effect**, **0 irrelevant or
contradictory**.

**Holding frozen controls** — HX05, HX09, HX10, HX12, HX13, HX16, HX17, HX20 all
preserve existing grounded facts. The exact-date controls **HX05, HX09, HX13,
HX17** are unchanged in selected event, `semantic_unique`, exact mode and
citations. HX20 gains evidence additively. **P1-A3 / P1-A4 / P1-A4.1 / P1-A5
contracts remain valid.**

**Non-holding global validation** (after completing real BGE coverage) —
exchange **0** appends, major **1**, periodic **3**. For every non-holding
append: baseline evidence preserved, no grounded fact lost, no citation lost, no
uniqueness break, no contradictory evidence, downstream output unchanged.
**Periodic/table baseline chunks are never displaced.** This resolves the Phase 5
global-scope validation concern.

**Existing expansion interaction** — the recovery runs after the existing
retrieval rescues and expansions and checks the actual emitted document set. No
duplicate expansion document was observed. *Caveat: the live Gold60 run contained
no case where a graph/rescue expansion had already added a document, so this
exact overlap path is unit-tested but not live-exercised.*

**Latency and context** — no extra database query or model call. Observed
holding latency delta **≈ +2.9 ms mean**. Across 54 questions: **13 additional
chunks**, **6,421 added characters** total, mean **~494 characters / ~291
tokens** per appended chunk, maximum **1,104**. Context growth from this rescue
is bounded to one chunk.

**Tests** — after the regression test file was committed at `283edfb`, the full
suite gate was re-run: **1555 OK, skipped 13**, with **19 new additive-recovery
tests**. The frozen contract is therefore protected by tracked tests. Mutations
caught: replacement instead of append; baseline reorder; threshold lowered to
2; appending more than one; appending an already represented document; widening
the normal budget; unstable ordering.

### Invariants that must remain true
1. The normal top-10 remains **byte-identical**.
2. `top_k` remains unchanged.
3. Crowding trigger = **≥ 3** chunks from one canonical document.
4. Rescue limit = **1**.
5. The rescue is **additive only**.
6. The rescue candidate must come from an **unseen canonical document**.
7. Existing score and candidate order are **reused**, never recomputed.
8. Baseline grounded evidence is **never removed**.
9. Exact-date semantics remain unchanged.
10. No domain, question or company special casing.
11. Public API unchanged.
12. Deterministic repeatability.
13. `ScoredTail` remains internal-only retrieval machinery.

**Negative invariants — never do any of these:** replace the baseline top-10 for
document diversity · reorder the baseline top-10 · remove same-document evidence
merely to raise document count · widen normal `plan.top_k` to implement this
rescue · append more than one candidate · trigger below crowding count 3 ·
append a document already represented · infer document identity from `chunk_id`
· use Gold, question, company or domain-specific logic · modify scores for this
rescue.

### Known residual issues NOT solved
**P1-R ranking is not globally solved.** Two live BGE misses remain:

- **HX08** — remains unrecovered; the rescue does not trigger.
- **HX16** — the rescue may trigger, but the appended unseen document is not the
  Gold document; remains unrecovered.

Do **not** reopen the frozen additive contract merely because these residuals
remain; future work needs separate evidence.

**P0-D.2 is not unblocked by this freeze.** H01 and HX02 still miss intended
evidence, so P0-D.2 remains **TARGET EXISTS · ACTIVATION DEFERRED**. HX04's
intended document ranks 1 while holding-event construction remains zero in the
simulation — a separate P0-D.2 correctness issue.

**P1-B remains deferred.** Complete BGE validation continues to show the relevant
misses are not caused by metadata/filter exclusion. Do not reopen P1-B because of
P1-R residual misses.

**Periodic retrieval is materially weaker under full LIVE BGE** — R@1 **0.2857**,
R@10 **0.6429**. Recorded here as a **separate follow-up finding**, not part of
this freeze and not to be solved in P1-R.

**Production BGE revision identity** — `app/retrieval/bge_m3.py` passes
`revision` to FlagEmbedding but does not independently verify the resolved
HuggingFace snapshot, and INFRA-E1 proved this may resolve the wrong revision.
Not fixed by P1-R; a separate **Embedding Identity Hardening** follow-up.

### Reopen conditions
- a demonstrated baseline top-10 mutation, reordering or evidence loss;
- a harmful or contradictory addition in any domain;
- a later architecture that cannot preserve this contract additively.

---

## P1-C — Table Sibling / Evidence Neighborhood Expansion

**Status:** **DIAGNOSIS COMPLETE — IMPLEMENTATION DEFERRED · LIVE PERIODIC
TARGET NOT ESTABLISHED**

This is not a freeze. **No production implementation exists**, so there is no
contract to protect — only a measured target, a design that has not been built,
and the conditions for resuming.

> ### ⚠ Header recovery is **not** the target
>
> **0 of 10,318 split raw-table chunks lacked a rendered header.**
> `_render_table_rows` prepends `column_headers` to every chunk and the chunker
> synthesises `열 1 … 열 n` when a splitting table has none. Units are present on
> 97% and period labels on 99.8% of projections. Expanding siblings to recover
> header, unit or period context would solve a problem the corpus does not have,
> and must not be used to justify P1-C.

### The question
An anchor chunk is retrieved; the evidence the answer needs sits in **another
chunk of the same table**; the pipeline never surfaces it. Only that shape is
P1-C.

### Schema already supports it
`table_id` · `row_start` / `row_end` · `prev_chunk_id` / `next_chunk_id` ·
`column_headers` · `unit` · `period_labels` · `source_refs` · projection/source
provenance. Deterministic table-neighborhood reconstruction needs **no schema
change**.

### No stage does this today
No retrieval or evidence stage performs generic same-table sibling expansion.
The in-retrieval rescues work from the fixed pool without table awareness, and
graph expansion adds documents, not rows.

### P1-A3 boundary
P1-A3 is **holding-only** and **projection-only** (`holding_detail_row`,
`holding_report`). It never adds a **raw same-table sibling row**, and
**non-holding tables are outside its ownership**. P1-C must not touch its anchor
semantics or its promotions.

### Measured — table splitting
45,311 sampled tables across all four doc groups:

| doc group | tables | split | rate |
|---|---|---|---|
| periodic | 41,916 | 3,562 | **8.5%** |
| holding | 2,555 | 229 | 9.0% |
| **major** | 616 | 0 | **0.0%** |
| **exchange** | 224 | 0 | **0.0%** |
| **total** | 45,311 | **3,791** | **8.4%** |

Chunks per table: p50 = 1, p90 = 1, p95 = 2, max 1,843.

### Measured — the real target: paired metrics split within one table

| pair | split / total containing both | rate |
|---|---|---|
| **자산총계 / 부채총계** | 217 / 689 | **31.5%** |
| 영업이익 / 당기순이익 | 48 / 577 | 8.3% |
| 매출액 / 영업이익 | 6 / 739 | 0.8% |
| holding and contract pairs (보유주식수/보유비율, 변동전/변동후, 계약금액/계약기간) | 0 | **0.0%** |

Every split instance observed was **periodic**. Since major and exchange never
split and holding pairs never split, **periodic is the only meaningful target
domain**, and P1-A3 already owns the holding side.

### Measured — end-to-end probes
15 generic paired-metric probes (no gold used):

| outcome | count |
|---|---|
| BOTH_SERVED — pipeline already returns both | 6 |
| NEITHER — retrieval found neither metric (P1-R territory) | 4 |
| ONE_ONLY, sibling does **not** contain the missing fact (different table/document) | 4 |
| **genuine SIBLING_RECOVERABLE** | **1** |

The one genuine case: **이마트 영업이익 / 당기순이익** — one requested metric is
absent from the served top-10 while unserved chunks with the same `table_id`
contain it.

### Candidate design, if later validated
- **Trigger** — a served table chunk, an unmatched query term, and unserved
  same-table siblings.
- **Policy** — metric-hint / text-term match first, then distance fallback.
- **Bounds** — max **1** sibling per anchor, max **2** added chunks per query.
- **Isolation** — same document and same `table_id` only.
- **Placement** — after the retrieval rescues, P1-A3 and graph expansion,
  immediately before `EvidenceBuilder`.
- **Score** — **do not fabricate** a lexical/vector/retrieval score.
- **Provenance** — preserve the real `chunk_id`, `doc_id` and `source_refs`;
  record `expansion_source`, `anchor_chunk_id`, `table_distance` and
  `expansion_reason` internally only. No public schema change.

Bounds are not optional: the one recoverable case offered **8** candidate
siblings for a single needed fact, and one table in the corpus emits 1,843
chunks.

### Why implementation was deferred pending live periodic validation
- periodic is the only meaningful target domain;
- **periodic Gold questions were not evaluated** — their companies are outside
  the seeded corpus;
- the retrieval side of the probes used `DeterministicHashEmbedder`, so which
  chunks were served is unreliable;
- correction and multi-document controls were **not exercised**;
- only **1 of 15** probes demonstrated end-to-end recoverable evidence;
- the gain on real evaluation therefore remains unknown.

### Live periodic validation decision
The required validation has now been run on all 14 periodic Gold60 questions
under the pinned live BGE-M3 identity. Gold answers requiring multiple sibling
chunks measured **0/14** overall and **0/7** among table questions. Offline
same-table sibling expansion recovered **0 questions**. The single exploratory
hash-based probe therefore did not establish a production target on real
periodic evaluation questions.

**P1-C REMAINS DEFERRED.** No `TABLE_SIBLING_GAP` target is established.

### Reopen implementation only if
1. real evaluation questions demonstrate that an anchor is retrieved while
   required same-table sibling evidence is absent;
2. one generic bounded rule recovers at least two genuine answer failures
   without removing or replacing existing answer evidence;
3. frozen P0 / P1-A controls are verified against the expansion.

### Ownership firewall
| symptom | owner |
|---|---|
| filtered out **before scoring** | **P1-B** |
| eligible but **ranked low** | **P1-R** |
| anchor retrieved, required **same-table sibling absent** | **P1-C** |
| evidence present but **reasoning wrong** | reasoning phases |

A metric that lives in a *different* table or document is none of these — it is
not a P1-C case.

---

## INFRA-E1 — Reproducible BGE-M3 Evaluation Environment

**Status:** **PHASE 2 COMPLETE — P1-R UNBLOCKED · P0-D.2 ACTIVATION REMAINS
DEFERRED**

This is **not** a production feature freeze. No production behaviour was changed;
this records an evaluation environment and the live measurements it produced.

### Environment
A local, isolated evaluation stack: an **NVIDIA RTX 4060**, a **disposable Docker
BGE-M3 runtime**, and the **isolated pgvector database on port 55433**.
Production and dev services were untouched, and **port 8010 was untouched**.

### Model identity
Evaluation ran on **real BGE-M3**, not a stand-in:

```
model      BAAI/bge-m3
revision   6892b95fed65c899a30896eb40d619ae284d0455
dimensions 1024
```

Resolved runtime: Python 3.11.16 · torch 2.5.1+cu121 · FlagEmbedding 1.4.2 ·
transformers 5.16.1 · sentence-transformers 6.0.0. Dense vectors were **L2
normalized**.

> ### ⚠ FlagEmbedding accepted the revision and did not honour it
>
> **FlagEmbedding 1.4.2 takes a `revision` argument and silently resolves current
> `main` instead.** Left unchecked, every number here would have described an
> unpinned model.
>
> The evaluation therefore introduced a **fail-closed pinned snapshot loader**
> that resolves the requested HuggingFace commit explicitly, **asserts the
> resolved snapshot commit**, requires the pinned **safetensors** checkpoint, and
> passes the verified local snapshot path to FlagEmbedding. **Every figure below
> comes from the verified pinned revision.**
>
> Separately: production `bge_m3_local` **passes `revision` to FlagEmbedding but
> does not independently verify the resolved snapshot**. That is a **production
> identity-hardening concern**, was **not changed** by INFRA-E1, and **must not be
> conflated with P1-R**.

### Structural evaluation baseline
The evaluation database now holds the complete structural corpus for all **17
Gold60 companies**. MODE A on that complete database:

| | |
|---|---|
| `QUERY_UNDERSTANDING_DECLINE` | 6 |
| `FILTER_EXCLUSION` | **0** |
| `COMPLETE` | 54 |
| structural inclusion | **54 / 54** |

This independently reinforces the **P1-B deferral**.

### Authoritative candidate union
After complete structural seeding the Gold60 candidate union is **75,786 unique
chunks**. The earlier **21,832** figure was collected against a 7-company
database, is **incomplete, and is superseded**. P0-D-accepted questions with zero
candidates: **0**.

### Required BGE scope and coverage
The required holding subset is **9,987 unique candidate chunks**. Live BGE vector
coverage for the **22 evaluable holding questions**:

| | |
|---|---|
| expected vector-eligible rows | 11,925 |
| stored matching BGE rows | **11,925** |
| missing | **0** |
| zero-vector questions | **0** |

### Embedding performance
Sustained **~11.95 chunks/sec**, peak GPU VRAM **~1,975 MiB**, **no failed
embeddings**. Pre-existing hash embeddings **remained intact** and coexist with
BGE rows under the `(chunk_id, model, version)` key.

### Fail-closed invariants
Evaluation must remain fail-closed and must reject:

- any **hash** fallback;
- a model other than the exact pinned model;
- a revision other than the exact pinned revision;
- a dimension other than the expected one;
- **incomplete or zero** BGE candidate coverage;
- a **stale manifest**.

---

## Embedding Identity Hardening — Pinned BGE-M3 Snapshot Verification

**Status:** **FINAL FREEZE**

**Frozen commit:** `f592e9d`

> ### Scope of this freeze
>
> This covers **production `bge_m3_local` model snapshot identity verification
> only**. It changes **none** of: retrieval ranking · P1-R · the vector
> availability / lexical-fallback policy · P0-D · P1-A · the database schema ·
> the public API.

### Problem solved
`app/retrieval/bge_m3.py` passed `revision=config.version` to `BGEM3FlagModel`.
**FlagEmbedding 1.4.2 accepted that argument and demonstrably resolved `main`
instead of the configured pinned revision.**

| | |
|---|---|
| configured | `6892b95fed65c899a30896eb40d619ae284d0455` |
| **actually resolved before the fix** | **`5617a9f61b028005a4858fdac845db406aefb181`** (main) |

The crash observed under transformers 5.16.1 / torch 2.5.1 was **accidental
protection** — `main` ships `pytorch_model.bin` and that runtime refuses
`torch.load`. On a compatible runtime the wrong snapshot loads **silently**
while every row is stamped with the configured `embedding_version`.

### Final architecture / behavior
Production loading now follows:

```
configured model + configured revision
→ explicit HuggingFace snapshot resolution
→ exact commit verification
→ local required-file validation
→ BGEM3FlagModel(verified_local_path)
```

FlagEmbedding **no longer receives the repository id** for production BGE
loading, and **receives no `revision` argument** once the snapshot is resolved.
It therefore has no opportunity to re-resolve to `main`.

**Immutable revision.** `bge_m3_local` now rejects mutable revisions — `main`,
branch names, tags, truncated hashes — and requires a full 40-character
hexadecimal commit SHA. `embedding_version` is persisted as model identity and
must describe one immutable snapshot. The current production revision is already
a full SHA, so the known production configuration remains valid. **This is an
intentional fail-closed compatibility break for mutable-revision configuration.**

**Cache / offline contract:**

| case | behaviour |
|---|---|
| A exact pinned usable snapshot cached | load **offline** |
| B pin absent, network available | fetch exact pin, verify, load |
| C pin absent, network unavailable | **fail clearly** |
| D `main` cached, requested pin absent | **fail clearly; never substitute `main`** |
| E requested snapshot missing required files | **fail clearly** |
| F `main` and pin both cached | **pin selected deterministically** |

Resolution is offline-first, so a warm verified cache needs no network.

> ### Pattern-partial caches are valid
>
> An INFRA-E1 finding worth keeping: **a usable BGE dense-inference snapshot is
> not a complete mirror of the HuggingFace repository.** The resolver validates
> the files required for inference rather than demanding README,
> `.gitattributes`, or the sparse/ColBERT heads. A pattern-partial cache matching
> the INFRA-E1 cache shape is explicitly tested and accepted.

**Required files** — `config.json`, `model.safetensors`, and **at least one local
tokenizer payload** (`tokenizer.json` **or** `sentencepiece.bpe.model`). The
tokenizer requirement is **deliberately stricter than the underlying loader's
minimum**: without a local tokenizer payload, tokenizer resolution could occur
from another source and reintroduce identity drift. `README`, `.gitattributes`,
`sparse_linear.pt` and `colbert_linear.pt` are **not** required for dense
inference.

**safetensors is mandatory.** `pytorch_model.bin` does **not** satisfy the
contract and there is **no `.bin` fallback** — the pinned revision already ships
safetensors, loading stays deterministic, the unsafe `torch.load` path is
avoided, and behaviour no longer depends on the runtime's handling of
CVE-2025-32434.

**Internal identity diagnostics.** `BgeM3LocalEmbeddingProvider` exposes
`configured_model`, `configured_revision`, `resolved_model`, `resolved_revision`,
`resolved_snapshot_path`, `requested_device`, `verified`. **Resolved values
describe actual verified identity and must never simply echo the configured
revision.** An injected or test encoder without verified provenance reports
`verified = false` and `resolved_revision = None`. No public API contract
changed.

### Production files
- `app/retrieval/bge_m3.py`

Tests: `tests/test_bge_m3.py`, `tests/test_bge_m3_identity.py`

No change to `app/retrieval/hybrid.py`, query understanding, query validation,
query plan, the holding resolver, the answer composer, schemas/API, or the
database schema.

**Dependency decision** — `requirements-embedding.txt` is **unchanged**.
`huggingface_hub` is already available through the embedding dependency stack,
and this follows the existing `bge_m3.py` convention, which already imports
transitively supplied `torch` directly. *Recorded as the current repository
dependency policy, not as a universal best practice.*

### Invariants that must remain true
1. Revision must be an immutable full 40-character hexadecimal commit SHA.
2. Explicitly resolve configured model + revision through the HuggingFace Hub.
3. Verify the actual resolved snapshot commit equals the configured revision.
4. Validate the required dense-inference files.
5. Require `model.safetensors`.
6. Pass the **verified local snapshot path** to `BGEM3FlagModel`.
7. Do not delegate `revision` to FlagEmbedding after resolution.
8. Never substitute `main` or another revision.
9. Never fall back to `.bin`.
10. Never fall back to hash or another model.
11. Preserve dimensions, normalization and device behaviour.
12. Expose configured and resolved identities **separately** in internal
    diagnostics.

**Negative invariants — never do any of these:** rely on FlagEmbedding's revision
enforcement · pass the repository id directly after resolution · accept an actual
revision differing from the configured one · substitute `main` · use a mutable
revision for `bge_m3_local` · use a `.bin` fallback · claim configured identity
as resolved identity without verification · silently load tokenizer or model
assets from another revision · fall back to hash, another model, or another
revision.

### Verification performed

**Isolated real-model smoke** — `BAAI/bge-m3` @
`6892b95fed65c899a30896eb40d619ae284d0455`, production loader after the fix:

| | |
|---|---|
| configured revision | `6892b95f…` |
| resolved revision | **`6892b95f…`** |
| `model.safetensors` | present |
| `pytorch_model.bin` | absent in the pinned snapshot |
| CUDA smoke | **PASS** |
| output | 1024 dimensions · finite · L2 norms **1.0** |
| repeat cosine | **1.0** |
| different-text cosine | **≈ 0.337051** |

**Production ↔ evaluation equivalence** — production loader vs the INFRA-E1
verified loader over three fixed texts:

| | |
|---|---|
| cosine similarity | **1.000000000000** |
| max absolute difference | **0.000e+00** |

Production and evaluation loaders now agree **exactly** on the frozen model
identity.

**Tests** — implementation gate **1578 OK, skipped 13**, including **23 new
identity tests** covering: exact pin · wrong/`main` revision rejection · mutable
revision rejection · warm-cache offline load · cold-cache network retry · missing
pin · required-file failures · partial-cache acceptance · safetensors requirement
· local-path delegation · resolved diagnostics · no fallback.

> The existing BGE loader test was **intentionally updated** because it asserted
> the old defective behaviour — repository id plus `revision` delegated to
> FlagEmbedding. The replacement freezes the correct contract: **verified local
> path passed, and no revision delegated.**

**Mutations caught** — remove resolved-commit verification · pass repository id
instead of the verified path · accept `main` · remove the safetensors requirement
· fake resolved diagnostics using the configured revision · allow a mutable
revision · delegate `revision` to FlagEmbedding · remove the tokenizer
requirement.

### Database identity
The key remains `(chunk_id, embedding_model, embedding_version)`. **No schema
change.** The **75,786** pinned BGE evaluation rows were produced by INFRA-E1's
verified loader and are considered valid. **No evidence currently shows that any
production database contains rows written by the old unverified loader.** Should
such rows later be discovered, treat them as **suspect and re-embed** — do not
infer their correctness merely from `embedding_version`.

### Known residual issues NOT solved
- Mutable-revision configurations now **fail closed**.
- The broad exception around offline snapshot resolution may cause a network
  retry before the final error is surfaced.
- The tokenizer requirement is deliberately stricter than the underlying
  loader's need.
- Resolved metadata is attached to the encoder object internally.
- Hardening applies **only to `bge_m3_local`**; remote providers resolve model
  identity externally.

> ### Retrieval Vector-Availability Policy — explicitly NOT part of this freeze
>
> Current behaviour is unchanged: when exact model/version/dimension vector
> coverage is zero, diagnostics report `vector_status` and hybrid retrieval may
> degrade to lexical-only. **This freeze does not make retrieval itself
> fail-closed.** Carried forward as a separate follow-up: **Retrieval
> Vector-Availability Policy**.

### Reopen conditions
- a demonstrated resolution of any revision other than the configured commit;
- a snapshot accepted without the required dense-inference files;
- a later architecture that cannot preserve this contract additively.

---

## Retrieval Vector-Availability Policy

**Status:** **FINAL FREEZE**

**Implementation commit:** `b1e31aa`

> ### Scope of this freeze
>
> This freeze covers **strict vector-coverage enforcement for repository-owned
> real-vector evaluation** and **production degradation observability through
> existing `ThinkTrace.warnings`**.
>
> It does **not** make production requests fail closed on missing vectors,
> change retrieval ranking, change P1-R, change Embedding Identity Hardening,
> change readiness, change the database schema, or add a top-level API field.

### Confirmed availability defect
Before this change, exact model/version/dimension vector coverage could be zero,
partial, or unavailable while production continued serving. Zero coverage was
internally marked `no_coverage`, but partial coverage still reported
`vector_status = "ok"` despite asymmetric hybrid ranking: the lexical lane ranks
every candidate, while the vector lane can rank only already-embedded candidates.
Already-embedded candidates therefore receive a structural two-lane advantage.

The defect was reproduced reversibly against the corpus with an isolated
temporary identity:

| state | stored vectors | result |
|---|---:|---|
| full | **75,786** | healthy |
| partial | **37,938** | candidate-scoped coverage **≈ 0.536** |
| zero | **0** | `vector_status = no_coverage`; lexical fallback |

At partial coverage, `vector_status` still read `"ok"`, approximately **91.2%**
of served chunks came from the embedded subset, **H03 degraded from rank 2 to
rank 5**, and **P01 lost the Gold evidence from top-10**. Partial coverage can
therefore be more misleading than zero coverage: it appears healthy while
comparing candidates asymmetrically.

### Final policy — evaluation and production deliberately differ

**Evaluation.** Any repository-owned evaluation claiming real-vector metrics
must have complete candidate-scoped coverage under the exact configured
model/version/dimensions. If coverage is incomplete, it **fails closed before
metrics**.

For every evaluated question:

```
embedded_candidate_count == eligible_candidate_count
```

and, when eligible candidates exist:

```
embedded_candidate_count > 0
```

Zero- or partial-coverage results must never be produced or labelled as
real-vector/BGE metrics.

**Production.** Production retains the existing fallback and availability
semantics. It does not fail a request merely because coverage is incomplete;
instead, degradation is made explicitly observable in `think_trace.warnings`.

### Shared coverage policy
`app/reasoning/vector_coverage_policy.py` is the shared semantic definition of
coverage state used by serving and evaluation. It classifies the existing
vector status and candidate-scoped coverage facts as:

- `healthy`
- `zero_coverage`
- `partial_coverage`
- `empty_vector_result`
- `vector_unavailable`
- `coverage_unknown`

Existing hybrid `vector_status` values are **not renamed**. The shared policy
prevents evaluation and serving from independently defining what complete
coverage means.

### Production trace and warning contract
The public response retains exactly the existing five top-level keys:

```
question_id
question
retrieved_context
think_trace
answer
```

There is no sixth top-level key and no new `ThinkTrace` schema field.
Degradation uses the existing `think_trace.warnings` list and repository warning
token convention. Representative forms, matching implementation formatting,
are:

```
vector_coverage_partial:provider=<provider>,candidates=<n>,embedded=<n>,ratio=<ratio>
vector_coverage_absent:provider=<provider>,candidates=<n>,embedded=0,ratio=0
vector_results_empty:provider=<provider>
vector_unavailable:provider=<provider>,error=<ExceptionType>
vector_coverage_unknown:provider=<provider>,error=<ExceptionType>
```

Errors expose only a sanitized exception type. They must never expose an
exception message, DSN, password, host credential, or stack trace.

### Serving-state contracts

**Healthy full coverage.** When candidate-scoped exact-identity coverage is
complete and vector retrieval succeeds, there is no degradation warning.
Ranking, answer, and retrieval context remain unchanged; healthy full-coverage
behaviour is equivalent to the pre-policy behaviour.

**Partial coverage.** When
`0 < embedded_count < candidate_count`, production continues existing hybrid
retrieval. It does not remove vector results, force lexical-only operation, or
fail the request, but it **must** emit `vector_coverage_partial` with candidate
count, embedded count, and coverage ratio. Partial coverage must never be
observationally indistinguishable from a healthy hybrid request.

**Zero coverage.** When `candidate_count > 0` and `embedded_count == 0`, existing
lexical fallback remains. `think_trace.warnings` must expose
`vector_coverage_absent` with candidate count, embedded count `0`, and ratio
`0`.

**Empty vector search.** Vectors existing while vector search returns no rows is
distinct from stored vectors being absent. Existing lexical fallback remains,
and the warning is `vector_results_empty`; this state must not be classified as
zero coverage.

**Vector error.** Existing configuration remains authoritative:

| `fallback_on_vector_error` | behaviour |
|---|---|
| `True` | serve existing fallback and emit sanitized `vector_unavailable` diagnostics |
| `False` | raise as before |

This freeze does not alter availability semantics.

**Coverage lookup unavailable.** A failed coverage lookup is exposed as
`vector_coverage_unknown`, with provider and sanitized exception type only.

**Hash and diagnostic providers.** Intentional hash/dev retrieval is never
labelled as a BGE failure. Provider identity is reported generically, and any
coverage asymmetry is reported under the actual provider identity. Strict
evaluation enforcement does not prohibit intentionally configured hash or
lexical-only diagnostics. Under the evaluator policy, non-hash providers are
treated as real-vector providers.

### Evaluation enforcement
Before this implementation, `scripts/bge_eval_preflight.py` contained
`assert_vector_coverage`, but no real evaluation entry point called it. The
existence of a helper was insufficient.

Strictness is now mandatory through both:

- shared enforcement inside `QueryPlanHybridEvaluator`;
- a strict vector-executor wrapper for repository-owned direct metric scripts.

Do not revert to optional caller discipline. The tracked direct evaluation
entry points updated to use the strict path are:

- `scripts/diagnose_p1r_additive.py`
- `scripts/validate_p1r_additive_live.py`
- `scripts/validate_p1r_global_domains.py`

Shared evaluation/preflight infrastructure enforces the same policy. This does
**not** mean arbitrary future external scripts are automatically protected;
future repository-owned real-vector evaluation entry points must use the same
strict enforcement path.

### Production and evaluation implementation surface
The exact tracked surface of implementation commit `b1e31aa` is:

- `app/api/pipeline.py`
- `app/reasoning/hybrid_evaluation.py`
- `app/reasoning/vector_coverage_policy.py`
- `scripts/bge_eval_preflight.py`
- `scripts/diagnose_p1r_additive.py`
- `scripts/validate_p1r_additive_live.py`
- `scripts/validate_p1r_global_domains.py`
- `tests/test_vector_availability_policy.py`

No change to `app/retrieval/hybrid.py`, `app/retrieval/bge_m3.py`,
`app/api/schemas.py`, health/readiness, or the database schema.

### Verification performed

**Controlled isolated `festival-verify` smoke:**

| identity | stored rows | strict result | metrics |
|---|---:|---|---|
| full pinned | **75,786** | preflight **PASS** | allowed |
| temporary partial | **37,938** | evaluation **FAIL** | blocked |
| temporary zero | **0** | evaluation **FAIL** | blocked |

The temporary identities were removed. Existing rows remained unchanged:
**75,786 pinned BGE** and **132,768 hash**.

**Production trace smoke:**

| state | serving result | trace result |
|---|---|---|
| full | serves | no degradation warning |
| partial | serves | `vector_coverage_partial` visible |
| zero | serves lexical fallback | `vector_coverage_absent` visible |
| vector error with fallback enabled | serves fallback | `vector_unavailable` visible |

All responses retained exactly the existing five top-level API keys.

**Ranking immutability.** This policy is observability/evaluation enforcement
only. `app/retrieval/hybrid.py` remained **byte-unchanged**. Representative live
comparisons across exchange, holding, major, and periodic produced identical
`retrieved_context`, ordering, answer, and all non-warning trace fields. The
only intended difference under degraded coverage is the warning.

**Tests.** Implementation gate: **1598 OK, skipped 13**, with **20 new tests**.
Coverage includes full-coverage health; partial and zero visibility; empty vector
results as a distinct state; vector-error fallback and the fail-closed switch;
hash/dev behaviour; strict evaluation under full, partial, and zero coverage;
API top-level invariance; ranking immutability; count/ratio visibility; and error
sanitization.

**Mutations caught:** partial coverage reported healthy · zero warning removed ·
evaluation coverage assertion skipped · partial metrics allowed · vector-error
warning suppressed · healthy coverage incorrectly warned · hash incorrectly
rejected · top-level API changed · ranking changed while observability was added
· coverage counts or ratio omitted.

### Relationship to other frozen contracts
**P1-R remains FINAL FREEZE — bounded additive document recovery. Embedding
Identity Hardening remains FINAL FREEZE.** This availability policy must not
alter either contract.

The concerns are complementary and distinct:

| concern | question |
|---|---|
| Embedding Identity | Are these vectors really the configured model/revision? |
| Vector Availability | Do the required vectors for this candidate set exist completely? |

Both are required for trustworthy real-vector evaluation.

**P0-D.2 remains activation-deferred.** This freeze does not change its status.

### Readiness deliberately deferred
The `/healthz` and readiness contracts are unchanged. Requiring complete corpus
coverage at readiness could prevent rolling re-embedding or deployment. This
freeze deliberately chooses **evaluation strictness plus serving
observability**, not readiness enforcement.

### Invariants that must remain true
1. Repository-owned real-vector evaluation fails before metrics on zero or
   partial exact-identity candidate coverage.
2. Production continues its existing serving and fallback behaviour.
3. Partial coverage is explicitly visible with candidate count, embedded count,
   and ratio.
4. Zero coverage, empty vector results, vector errors, and unknown coverage stay
   distinguishable.
5. Healthy full-coverage retrieval remains unchanged and unwarned.
6. Error diagnostics expose exception type only.
7. Existing hybrid `vector_status` values remain unchanged.
8. Ranking, P1-R, Embedding Identity Hardening, readiness, schemas, and database
   schema remain unchanged.

**Negative invariants — never do any of these:** publish real-vector metrics
with partial coverage · publish real-vector metrics with zero coverage · treat
configured model identity as proof that vectors exist · make partial coverage
look healthy · hide production vector degradation from the trace · expose raw
vector exception messages or secrets · change ranking as part of availability
observability · alter P1-R math · alter the Embedding Identity contract · add
readiness gating under this freeze · add a sixth top-level API key · Gold-special
case the availability policy.

### Known residual issues NOT solved
- `filtered_candidates` can mask an empty vector result; the classifier falls
  back to coverage state.
- A future custom evaluation script can bypass enforcement if it does not use
  repository strict infrastructure.
- Non-hash providers are treated as real-vector providers for evaluator policy.
- An incomplete corpus may produce frequent warnings during long re-embedding.
- Multi-document requests currently expose warning information from the primary
  retrieval execution only.
- No production readiness threshold is defined.

### Reopen conditions
- a repository-owned real-vector evaluator produces metrics under incomplete
  exact-identity candidate coverage;
- degraded production vector availability is not visible in the existing trace;
- observability changes ranking, serving fallback, or a separately frozen
  contract;
- a later architecture cannot preserve this evaluation/production split.

---

## Periodic Retrieval — Live BGE Diagnosis

**Status:** **DIAGNOSIS COMPLETE — KEEP DEFERRED**

This is **not** a FINAL FREEZE feature implementation. No production code,
ranking rule, frozen contract, schema, evaluation policy, or database content
was changed. This entry records why the weakest exact-Gold domain does not
currently justify a production retrieval target.

### Live baseline
Evaluation used the pinned real-vector identity:

| field | value |
|---|---|
| model | `BAAI/bge-m3` |
| revision | `6892b95fed65c899a30896eb40d619ae284d0455` |
| periodic questions | **14** |
| exact-identity vector coverage | **complete for all 14** |

Periodic Gold60 exact-Gold chunk metrics:

| metric | value |
|---|---:|
| R@1 | **0.2857** |
| R@3 | **0.3571** |
| R@5 | **0.5000** |
| R@10 | **0.6429** |
| MRR | **0.3661** |

### Structural inclusion

| check | result |
|---|---:|
| Gold document in metadata candidate set | **14/14** |
| Gold mapped chunk in retrieval candidate set | **14/14** |
| required answer fact structurally present | **14/14** |
| filing-period metadata present | **14/14** |

The measured weakness is therefore **not** document candidate exclusion, chunk
candidate exclusion, corpus omission, or filter exclusion.

### Decisive answer-level result
Under the current production pipeline:

| check | result |
|---|---:|
| required answer terms present | **14/14** |
| answerable | **14/14** |
| unresolved required facts | **0** |
| citations retained | **14/14** |

No wrong requested period, unit loss, calculation loss, or citation loss was
observed. The weak exact-Gold metric therefore does **not** currently represent
14 answer-level failures.

### Exact-Gold misses
Primary exact-target owners were:

| owner | count | questions |
|---|---:|---|
| `COMPLETE` | **9** | P02, P03, P04, P05, P07, P08, P09, P12, P14 |
| `VECTOR_RANKING` | **1** | P01 |
| `FUSION_RANKING` | **3** | P06, P11, P13 |
| deterministic rerank / `OTHER` | **1** | P10 |

All five exact-target misses nevertheless have answer-equivalent evidence in
the final evidence set:

- **P01:** equivalent disclosure text already supplies the DX product evidence.
- **P06:** the HUBO / bipedal-robot description repeats across periodic filings.
- **P10:** higher-ranked evidence already contains the requested KOSPI listing
  date; only the exact Gold table is demoted.
- **P11:** later periodic filings repeat the TableOne merger date.
- **P13:** later periodic filings repeat the TC BONDER business description.

Forcing the historical Gold filing or chunk in these cases would be
evaluation-label tuning rather than a demonstrated answer-correctness fix.

### Offline ablations

| ablation | result |
|---|---|
| lexical-only | regressed overall |
| vector-only | regressed overall |
| remove deterministic rerank | recovered P10, lost P02, worsened aggregate ranking |
| stronger existing period relevance | no effect |
| table-representation dedup | no effect |
| same-table sibling expansion | no recovery |
| corrected-version preference | no effect |
| additive table diversity | no recovery |

No deterministic ablation recovered at least two genuine answer failures
without regression.

### P1-C decision
**P1-C REMAINS DEFERRED.** Gold answers requiring multiple sibling chunks
measured **0/14** overall and **0/7** among table questions. Same-table sibling
expansion recovered **0 questions**. No `TABLE_SIBLING_GAP` target is
established.

### P1-B decision
**P1-B REMAINS DEFERRED.** Gold documents and Gold chunks were both present in
their candidate sets **14/14**. Filter relaxation is not responsible for the
periodic metric.

### Period and metric findings
No shared period-aware target and no shared metric/account-alias target were
found. Only P07 is a normalized financial-metric case, and it already ranks
first. The set is heterogeneous: products, sanctions, credit ratings, revenue,
capacity, technology, listing and merger events, audit matters, and business
descriptions. A generic financial-account normalizer is not justified for this
set.

### Correction and version finding
All Gold periodic documents involved in this diagnosis are non-corrections, and
no relevant correction group controls the misses. The answer-equivalent facts
come from ordinary recurring periodic disclosures, not original/corrected
document chains. **P0-A remains unchanged.**

### Latency finding — separate from retrieval correctness
Periodic warm latency measured approximately:

| measurement | value |
|---|---:|
| mean | **2.614 s** |
| median | **2.010 s** |
| p95 / max | **6.829 s** |

Approximate mean stage costs:

| stage | mean |
|---|---:|
| candidate chunk hydration | **2.115 s** |
| query embedding | **184 ms** |
| vector search | **88 ms** |
| existing-vector lookup | **83 ms** |
| chunk preparation | **58 ms** |
| lexical retrieval | **36 ms** |
| hybrid rerank | **7 ms** |

Candidate chunk hydration is a separate performance target. Some periodic
questions have candidate pools around 19,000 chunks, but exact misses were not
systematically associated with larger pools than successful questions. Large
pools explain latency variance better than correctness; this diagnosis does
**not** authorize tighter retrieval filters.

### Decision
**Periodic Retrieval: NO PRODUCTION TARGET — KEEP DEFERRED.**

Reasons:

- all 14 questions are answerable;
- supporting evidence remains cited;
- exact misses mostly contain answer-equivalent recurring-disclosure evidence;
- no shared deterministic failure family recovers at least two genuine
  failures;
- tested interventions either do nothing or regress existing evidence.

### Evaluation-only follow-up
Exact document/chunk metrics appear to understate periodic answer quality when
another valid filing or chunk repeats the same stable fact. A potential future
**semantic-equivalence evidence audit** must be period-sensitive,
value-sensitive, company-sensitive, and citation-sensitive. It must not accept
generic keyword overlap. It is not implemented by this documentation phase.

### Negative invariants
- Do not tune retrieval to force a historical Gold filing when equivalent
  evidence is already served.
- Do not reopen P1-C without an actual sibling-evidence failure.
- Do not reopen P1-B without candidate or filter exclusion.
- Do not remove deterministic reranking merely to recover P10.
- Do not broadly increase `top_k`.
- Do not replace existing answer evidence for diversity.
- Do not optimize exact-Gold metrics at the expense of answer evidence.

### Separate next priority
**P0-D.2 HX04 correctness remains unresolved.** Its intended disclosure ranks
first, but holding event construction is `0` and `matching_event_count = 0`.
This is a downstream correctness issue separate from periodic retrieval. It is
recorded as a follow-up only and is neither diagnosed nor changed here.

---

## HX04 Acquisition Semantics — Same-Row Holding Acquisition Events

**Status:** **FINAL FREEZE**

**Implementation commit:** `7393842`

> ### Scope of this freeze
>
> This freeze covers **additive holding acquisition-event semantics only**: two
> internal event fields, the query language that requests them, a deterministic
> same-row acquisition classifier, and their rendering.
>
> It does **not** activate P0-D.2 globally. **PRIMARY_COMPANY_WITH_REPORTER is
> not production-active.** It changes none of: retrieval ranking · P1-R ·
> Embedding Identity Hardening · the Retrieval Vector-Availability Policy ·
> `QueryValidator` multi-company activation · issuer/reporter role resolution ·
> reporter normalization · P1-A4 exact-date matching · P1-A5 ambiguity and
> citation rules · the database schema · the public API schema.

### The original failure
Question: `에스엠 하이브 풋옵션 행사 주식 취득일과 취득 수량`

Intended evidence — `holding_20240314001102`, table `t0019`, **row 2**:

| field | value |
|---|---|
| 변동일 | `2024-03-07` |
| 취득/처분방법 | `풋옵션권리행사배정에따른주식취득(+)` |
| 변동전 | `2,098,811` |
| 증감 | `868,948` |
| 변동후 | `2,967,759` |
| 취득/처분단가 | `120,000` |

Before this change the intended disclosure was retrieved at **rank 1** and the
`holding_detail_row` projection existed, yet `requested_fields = ()`, so P1-A3
coverage stopped at `no_requested_fields`, the projection was never promoted, the
raw `t0019` chunk remained standalone evidence, holding-event groups were `0`,
constructed and matching events were `0 / 0`, and the answer was
`insufficient_evidence`.

**Retrieval was not the failure.**

### Root cause
The holding-event ontology had no representation for an acquisition date or an
acquired quantity, and requested-field extraction did not recognize `취득일` or
`취득 수량`.

The obvious repair — globally aliasing `취득일 → reference_date` and
`취득 수량 → change_shares` — was **rejected as unsafe**. `reference_date` is
projection-specific: on a detail row it carries the transaction date, on a
`holding_report` projection it carries the report or base date. HX04 makes the
difference concrete — the acquisition happened on `2024-03-07` while the filing's
report and receipt date is `2024-03-14`. Acquisition semantics therefore require
**same-row transaction proof**, not an alias.

### Frozen acquisition contract
Additive internal fields: **`acquisition_date`**, **`acquired_shares`**.

Existing fields keep their exact prior meaning and are **not** reinterpreted:
`reference_date`, `report_date`, `receipt_date`, `before_shares`,
`change_shares`, `after_shares`, `before_ratio`, `after_ratio`, `change_ratio`.

A holding detail row may supply acquisition-specific fields **only** when all of
the following hold together:

1. it is a compatible holding detail/change row;
2. the **same row's** transaction method explicitly proves acquisition;
3. the **same row's** share delta is positive;
4. the **same row's** required date and quantity exist;
5. direct provenance exists.

**Transaction semantics and numeric sign must agree.** Neither alone is proof.

### Query language
Implemented acquisition-specific recognition:

| field | recognized wording |
|---|---|
| `acquisition_date` | `취득일`; `취득 일자` as matched by the implemented pattern |
| `acquired_shares` | `취득 수량` · `취득수량` · `취득 주식수` · `취득주식수` |

Existing phrases retain their prior canonical meaning: `변동일 → reference_date`,
`기준일 →` existing reference-date semantics, `보유주식수 → after_shares`,
`증감 주식수 → change_shares`, `보유비율 → after_ratio`.

**Generic date and quantity language did not become acquisition-specific.**
Wording without the acquisition noun reaches neither field.

### Acquisition classification
Deterministic and vocabulary-driven rather than a list of known strings. A row is
acquisition-compatible only when acquisition semantics such as `취득` / `매수` are
explicitly present, buy-**right** wording alone is not treated as acquisition,
disposal semantics (`처분` · `매도` · `양도` and similar) are absent, the source
direction marker is positive, and the same-row share delta is positive.

| representative accepted | representative rejected traps |
|---|---|
| `풋옵션권리행사배정에따른주식취득(+)` · `무상신주취득(+)` · `유상신주취득(+)` · `신규상장(무상취득)(+)` · `신규상장(유상취득)(+)` · `신규선임(유상취득)(+)` · `장내매수(+)` · `장외매수(+)` · `공개매수(+)` | `기타(+)` · `신규보고(+)` · `주식배당(+)` · `주식매수선택권부여(+)` · `합병(+)` · `상속(+)` · `장내매도(-)` |

**Positive share movement alone is not acquisition proof.** `기타(+)` is an
increase that explains nothing and can never satisfy an acquisition field.

### Acquisition date
`acquisition_date` comes **only** from the same acquisition detail row's 변동일.

HX04: **`2024-03-07`**.

Never derived from the report/base date `2024-03-14`, the receipt date
`2024-03-14`, a prior contract or grant date such as `2023-02-09`, or another
row's date. There is no fallback from a `holding_report` projection.

### Acquired shares
`acquired_shares` comes **only** from the same acquisition row's positive
증감 / 증감주식수.

HX04: **`868,948`**.

Never derived from `after_shares` `2,967,759`, `before_shares` `2,098,811`, the
unit price `120,000`, a negative disposal delta, `abs()` of a negative delta, a
missing quantity, or a fabricated zero.

### Same-row provenance
The transaction method, the acquisition date and the acquired quantity must all
originate from the **same detail row**. No cross-row reconciliation, even when
reporter, document and dates agree.

HX04 provenance: `holding_20240314001102` · table `t0019` · **row 2**. Both the
date and quantity citations trace to that row.

### Storage and identity design — CODE ONLY
No corpus backfill · no rechunk · no re-embedding · no database write · no
database schema change.

The implementation reads metadata the projection chunk **already stores**:
`column_headers`, `table_rows` and `source_refs`. **`app/parsing/` remained
unchanged**, so **`chunk_id` and stored vector identity are unchanged**.

The same-row guarantee is structural rather than reconciled: across all **6,705**
holding detail-row projections, each stores exactly one `table_rows` entry with
`row_start == row_end` and exactly one unambiguous method, date and change
column.

### Latent parsing finding — NOT resolved
The existing normalized-key projection path does not correctly map
`취득/처분방법` to the intended `변동 사유` projection field, because key
normalization collapses whitespace without removing the separator. Measured
across the corpus, **0 of 6,705** holding detail-row projections carry that
projected field.

This was **intentionally not fixed**: changing projection content could alter
deterministic chunk identity and force re-embedding. The acquisition
implementation instead reads already-persisted table metadata.

**This parser issue remains unresolved.**

### Event model
`HoldingEvent` is extended additively with `acquisition_date`,
`acquired_shares` and the transaction-method discriminator, appended with
defaults and serialized only when an acquisition was proven.

Acquisition fields are held **outside the frozen legacy field denominator**, so
legacy completeness and confidence denominators are unchanged and historical
event scoring does not drift. Existing non-acquisition event semantics are
unchanged.

### Coverage
No broad P1-A3 rewrite. For recognized acquisition questions the existing
structured rescue may expose compatible holding detail evidence. Unrelated
behaviour is retained: `requested_fields == ()` still yields
`no_requested_fields`. **No generic raw-table rescue was added.**

### Matching
Existing frozen matching remains authoritative — reporter normalization, reporter
matching, explicit date constraints, event filtering, deduplication, semantic
uniqueness and ambiguity. **Acquisition semantics bypass none of these**; they
add requested-field compatibility and event facts only.

### Ambiguity
Multiple compatible acquisition events are not arbitrarily selected. Selection by
latest, largest, first or nearest is not performed unless an existing frozen
selector authorizes it. **P1-A5 ambiguity behaviour remains authoritative.**

### Composer
Matched acquisition facts may render as `취득일` and `취득 수량`. No generic
raw-table composer fallback was added, and acquisition facts are never rendered
from a non-matching event. Existing P1-A5 citation behaviour is intact.

### HX04 end-to-end result
Measured under the **bounded P0-D.2 role simulation** — issuer `에스엠`, reporter
`하이브` — with production P0-D still declining the question.

| | before | after |
|---|---|---|
| `requested_fields` | `[]` | **`acquisition_date`, `acquired_shares`** |
| intended document rank | 1 | **1** |
| holding_event groups | 0 | **1** |
| constructed / matching events | 0 / 0 | **1 / 1** |
| warnings | `no_holding_event_groups` | **none** |
| answerability | unsupported | **supported** |
| `acquisition_date` | — | **`2024-03-07`** |
| `acquired_shares` | — | **`868,948`** |
| citations | 0 | **2** |

`2024-03-14` remains report/receipt-date information and is **not**
`acquisition_date`. `2,967,759` remains resulting holdings and is **not**
`acquired_shares`. `120,000` remains the unit price and is **not**
`acquired_shares`.

### Disposal safety control
Real-corpus control `holding_20231013000452` · table `t0026` carries the same
reporter and date with both an acquisition of **`+3,314,990`**
(`신규상장(유상취득)(+)`) and a disposal of **`-2,320,493`** (`장내매도(-)`).

The implementation admits the acquisition row and excludes the disposal row, and
does **not** `abs()` negative disposal quantities. **No document- or
table-specific production logic exists.**

### Corpus validation
Read-only validation covered **6,705** holding detail rows across **40** distinct
transaction methods, with **no database write**. Same-row acquisition and
disposal controls produced **zero observed safety violations**; **3,126** rows
classified as acquisitions. One row whose method claimed a purchase while the
signed change did not agree was excluded rather than defaulted — the conjunction
holding, with no fabricated quantity.

The classifier is **intentionally conservative**.

### Frozen control regression
Controls **HX05 · HX09 · HX13 · HX17**, compared before and after on
`requested_fields`, evidence group types and counts, constructed and matching
event counts, the selected event, all fact values, field provenance,
`selection_mode`, `semantic_unique`, ambiguity flags, warnings, citations,
citation source references and the rendered answer:

> **All four UNCHANGED.**

Both sides were measured with the **identical harness** against the
pre-implementation tree, so the comparison reflects behaviour rather than a
change in measurement.

### Test gate
Local reproduced pytest gate — Python **3.13.2**, pytest **9.1.1**, **1676**
collected:

> **1663 passed · 13 skipped · 0 failed**

Two warnings are present and are **not failures**: a Starlette/httpx deprecation
warning and a pytest cache-directory permission warning.

The implementation diagnosis additionally recorded **1060 subtests passed**;
local pytest did not report subtests separately because the current local plugin
set does not include `pytest-subtests`. **The authoritative local regression gate
is 1663 passed · 13 skipped · 0 failed.**

### Invariants that must remain true
1. Acquisition fields are additive; existing field semantics are unchanged.
2. The same-row transaction method proves acquisition; the sign must agree.
3. `acquisition_date` comes only from the proving row's 변동일.
4. `acquired_shares` comes only from the proving row's positive 증감.
5. Method, date and quantity share one row.
6. Direct same-row provenance is required for every acquisition fact.
7. Legacy completeness and confidence denominators are unchanged.
8. Frozen matching and P1-A5 ambiguity remain authoritative.
9. Acquisition behaviour is derived from disclosure structure alone.

**Negative invariants — never do any of these:** globally alias `취득일` to
`reference_date` · globally alias `취득 수량` to `change_shares` · infer
acquisition from a positive delta alone · infer acquisition from a `+` marker
alone · use a report or receipt date as the acquisition date · use resulting
holdings as the acquired quantity · use a unit price as the acquired quantity ·
`abs()` a disposal quantity into an acquisition · combine method, date and
quantity across rows · bypass reporter or date matching · fabricate missing
acquisition facts · drop same-row provenance · add a generic raw holding-table
composer fallback · special-case acquisition behaviour by Gold question,
company, document or table identifier.

### Known residual issues NOT solved
- The classifier **intentionally under-reports** unsupported increase types
  rather than guessing acquisition semantics.
- `기타(+)` cannot answer acquisition-specific questions.
- The latent `변동 사유` projection-normalization bug remains unresolved.
- Generated wording may redundantly repeat the acquisition date.
- Future ambiguous table header shapes fail closed.
- **HX04 is still production-declined because P0-D.2 is not activated.**

### Implementation surface
Files changed by `7393842`:

- `app/reasoning/holding_acquisition.py` *(new)*
- `app/reasoning/holding_event_resolver.py`
- `app/reasoning/evidence_builder.py`
- `app/reasoning/answer_composer.py`
- `app/generation/answer_generator.py`
- `tests/test_holding_acquisition.py`

No changes to `app/retrieval/hybrid.py`, `app/retrieval/bge_m3.py`,
`app/parsing/`, `app/reasoning/query_validation.py`,
`app/reasoning/holding_reporter.py`, the database schema, or the public API
schema.

### P0-D.2 status after this freeze
**DOWNSTREAM HX04 BLOCKER RESOLVED** — the failure was downstream of role
resolution, which already produced the correct issuer and reporter.

**P0-D.2 STILL ACTIVATION-DEFERRED PENDING SAFETY RECHECK.**
`PRIMARY_COMPANY_WITH_REPORTER` is **not** production-active. These statuses are
recorded separately and must not be merged.

That safety recheck has since been performed at `e8851d7` and returned
**KEEP DEFERRED** — see P0-D.2 · *Safety recheck — Phase 3*. It confirmed the
HX04 downstream blocker is cleared and left two unrelated blockers standing.
**HX04 Acquisition Semantics remains FINAL FREEZE.**

### Reopen conditions
- an acquisition fact demonstrably drawn from a row other than the one whose
  method proved the acquisition;
- a disposal, a report projection, or an unexplained increase satisfying an
  acquisition field;
- a later architecture that cannot preserve the same-row contract additively.

---

## Comparison Intent Firewall — Cross-Company Frame Detection

**Status:** FINAL FREEZE

**Frozen commit:** `7a0921e`

> ### ⚠ This freeze resolves **P0-D.2 Target A only**
>
> It does **not** activate P0-D.2. `PRIMARY_COMPANY_WITH_REPORTER` remains
> **NOT production-active**, and P0-D.2 remains **ACTIVATION DEFERRED**. Target B
> first-loss diagnosis is complete and classifies the frozen H01/HX02 targets as
> benchmark-context-dependent; no ranking fix is authorized.

### Reopen justification

This was an **explicit bounded reopen of P0-D Query Understanding**, admitted
under reopen ground (1) — a reproducible defect — not as a convenience feature.

The whole cross-company detector was one condition:

```
len(companies) > 1  AND  (literal "비교" OR literal "대비")
```

It was defective in **both** directions.

**False negatives** — genuine cross-company frames it did not recognize:
`A와 B 중 어디가 …` · `A와 B 중 누가 …` · `A보다 B …` · `A와 B 각각 …` ·
`A와 B의 차이 …` — and `차이` was not in the term list at all.

**False positives** — genuine issuer/reporter temporal constructions it claimed,
using the corpus's own wording `직전 보고 대비` / `직전보고 대비`.

The second class was a **live production defect while P0-D.2 was inactive**: a
holding fact request was being read as a company-versus-company comparison.

### ⚠ Phase 3 measurement correction

The P0-D.2 Phase 3 safety recheck recorded **false role resolution = 1**. That
number was **under-measured**, and the historical record above is annotated
rather than rewritten.

`scripts/diagnose_p0d2_firewalls.py` read:

```python
getattr(plan, "company_comparison", False)
```

`QueryPlan` has **no `company_comparison` attribute** — the real signal is
`plan.comparison`, whose `type` the validator compares against
`"company_comparison"`. The `getattr` therefore always returned `False` and the
comparison-precedence column never measured anything.

Corrected, against the real field:

| | Phase 3 recorded | corrected |
|---|---|---|
| required 10-question matrix | — | **6 / 10** would have been stolen |
| 20-construction inventory | — | **17 / 20** unprotected by the old detector |
| false role resolution | 1 | **6** on the required matrix |

**This was discovered during Target A diagnosis, not during Phase 3.** Phase 3
did not measure these correctly, and this entry does not claim it did.

### Frozen semantic contract

An **additive internal signal** on the channel `operation`, `periodic_intent`
and `correction_intent` already use:

```
plan.evidence["comparison_frame"] ∈ { "cross_company", "uncertain", None }
```

**No new `QueryPlan` dataclass field. No new public API field.** `ThinkTrace`
carries neither `comparison` nor `evidence`, so the response contract is
unchanged.

`comparison_frame` is a **firewall signal**. It is **not execution
authorization** and never asserts that a comparison can be answered.

#### `cross_company`
The query structurally treats two or more canonical company mentions **jointly,
comparatively, selectively, or enumeratively**, such that issuer/reporter
reinterpretation would be unsafe.

It does **not** mean a proven executor exists for that comparison. A newly
detected frame is **not** promoted into `plan.comparison.type ==
"company_comparison"` unless it already had that behaviour before this freeze.

#### `uncertain`
Multi-company comparative-looking language is present, but the parser **cannot
safely determine** whether the question is a cross-company comparison, a
temporal/event comparison, or an issuer/reporter request.

**`uncertain` fails closed** against future P0-D.2 role reinterpretation and
remains **AMBIGUOUS**. It is never forced into either role resolution or
executable comparison semantics.

Representative control:

```
하이브가 에스엠 주식을 더 많이 취득한 시점은 언제야?
  → comparison_frame = uncertain
  → AMBIGUOUS
  → future P0-D.2 blocked
```

#### absent
No comparison firewall evidence.

### Structural detector

Three helpers in `app/reasoning/query_understanding.py`:

| helper | what it establishes |
|---|---|
| `_company_mentions` | where each canonical company is named, **in reading order** |
| `_operator_binds_companies` | whether a comparison operator takes a company as its left operand |
| `_companies_are_coordinated` | whether two mentions are joined into one operand list |

Mention positions are **reconstructed from the text** because canonical
`plan.companies` ordering is sorted, not textual.

`대비` and `보다` are **postpositional** — Korean attaches them to the *left*
operand — so they are resolved **structurally**:

- a **company-bound** operator can indicate a cross-company frame;
- a **temporal- or metric-bound** operator does not.

Coordination is recognized for `와` · `과` · comma · `및` · middle dot · `랑` ·
`이랑` · `하고`. **Bare whitespace is deliberately not coordination**, which is
what preserves keyword-style issuer/reporter questions such as
`에스엠 하이브 …`.

Resolution order: fewer than two companies → absent · no comparative token →
absent · fewer than two positioned mentions → `uncertain` · **structure**
(bound operator, or coordination plus a frame) → `cross_company` · **temporal
anchor** → absent · otherwise → `uncertain`. Structure outranks vocabulary, so
`A와 B 중 어디가 변화가 커?` is a frame while `A에서 B의 지분율 변화` is not.

### Relation independence

Comparison semantics **must not** depend on issuer/reporter corpus relations ·
support counts · relation direction · Gold labels · question IDs · company
allowlists · `disclosure_route` · `task_type`.

Classifier input is **only**: question text · canonical extracted companies ·
textual structure. **Relation direction cannot influence `comparison_frame`.**

### Existing `company_comparison` compatibility

Explicit comparison behaviour using the literal `비교` is unchanged: it still
yields `plan.comparison.type == "company_comparison"` and the prior
`QueryValidator` behaviour.

**This phase did not widen executable comparison.**

| | |
|---|---|
| known cross-company frames protected by `comparison_frame` | **20 / 20** |
| still executable as `company_comparison` | **5** — the same 5 as before |
| recognized but remaining AMBIGUOUS | **15** |

### Temporal false-positive fix

| question | before | after |
|---|---|---|
| 에스엠에서 하이브가 **직전 보고 대비** 늘린 주식수 | `company_comparison` | `None` · frame `None` |
| 에스엠 공시에서 하이브의 **직전보고 대비** 증감 | `company_comparison` | `None` · frame `None` |
| 한화오션에서 한화에어로스페이스의 **직전 보고 대비** 지분율 | `company_comparison` | `None` · frame `None` |

All three remain **eligible for a future issuer/reporter interpretation**
instead of being misread as company-versus-company.

> The fix is **structural** — the operator's left operand decides. It is **not**
> the removal of `대비`.

### Temporal / metric precedence

Existing temporal semantics are preserved and remain authoritative:
`before_after` · `year_over_year` · `period_comparison` · `trend`.

Metric-ratio constructions remain non-company comparisons, including
`자기자본 대비` and `매출액 대비`. No financial-metric dictionary was added.

**Comparison words alone do not establish cross-company semantics.** `더`,
`차이`, `대비` and `보다` are never interpreted without structure or context.

### Cross-company frame coverage

Generic **firewall** patterns protected: explicit `비교` · structurally
company-bound `대비` · company `차이` · `중 어디가` · `중 누가` ·
`중 어느 회사가` · `어느 쪽이` · explicit choice + `더 많다` · explicit choice +
`더 높다` · `더 적다` / `더 낮다` under company structure · `A보다 B` ·
`B보다 A` · `각각` · `각 회사` · `둘 다` · `양사` · reversed company order.

These are **firewall** patterns. They do not imply the comparison planner can
execute every phrase.

### Issuer/reporter negative controls

**10 genuine issuer/reporter questions across 5 corpus-linked pairs** —
에스엠/하이브 · 한화오션/한화에어로스페이스 · 현대제철/기아 ·
현대자동차/현대모비스 · 레인보우로보틱스/삼성전자:

> **10 / 10 `comparison_frame = None`** · **10 / 10
> `role_reinterpretation_blocked = false`**

The **six historical P0-D.2 target questions** (H01, H02, HX01, HX02, HX03,
HX04) likewise carry `comparison_frame = None` and are **not blocked** by
Target A.

**This is the critical result: Target A must not destroy the intended P0-D.2
use case, and it does not.**

### Validator firewall

An additive defensive guard in the multi-company validation path of
`app/reasoning/query_validation.py`. Existing `company_comparison` precedence
remains **first**. For a future P0-D.2:

```
comparison_frame in { cross_company, uncertain }
   → issuer/reporter reinterpretation MUST be blocked
```

P0-D.2 is unimplemented, so **this guard activates no role resolution today**.
It pre-arms the precedence invariant:

> **COMPARISON FIREWALL → BEFORE → PRIMARY_COMPANY_WITH_REPORTER**

### Route independence

`comparison_frame` classification is **route-independent**. The classifier
consumes neither `disclosure_route` nor `task_type`; equivalent language is
classified semantically before route-specific execution. Verified across
**holding · periodic · major · exchange**.

### Reversed order

Company mention order does not change whether a frame exists. `A보다 B` /
`B보다 A`, `A 대비 B` / `B 대비 A`, and `A와 B 중` / `B와 A 중` receive
equivalent firewall semantics. **Corpus relation direction is irrelevant.**

### Three-plus company safety

The classifier may identify a frame for 3+ companies. P0-D.2 remains
**independently ineligible** because its future contract requires **exactly
two** companies. **No pair picking · no strongest-relation selection · no
company dropping.**

### 41-row diagnostic replay

Rows: cross-company **20** · issuer/reporter **8** · temporal **8** ·
single-company **5**. The three axes are recorded separately and **must not be
collapsed into one boolean metric**.

**Axis A — firewall classification**

| rows | result |
|---|---|
| cross-company | **20 / 20** `cross_company` |
| issuer/reporter | **8 / 8** `None` |
| temporal | **7** `None` · **1** `uncertain` |
| single-company | **5 / 5** `None` |

**Axis B — executable `company_comparison`**

| rows | result |
|---|---|
| cross-company | **5** — unchanged from baseline |
| issuer/reporter | **0** |
| temporal | **0** |
| single-company | **0** |

**Axis C — future P0-D.2 eligibility**

| rows | result |
|---|---|
| cross-company | **0 / 20** eligible |
| issuer/reporter | **8 / 8** eligible |
| temporal | **7 / 8** eligible · **1** `uncertain` and blocked |
| single-company | not relevant to a two-company contract |

### Mutation gate

**13 applied behavioural mutations, 13 caught.** Two initially survived and
exposed a real test gap — `중 누가` and `작년보다` were not isolated by any
assertion — which was closed by adding a predicate-free choice-frame control and
a period-anchored-predicate control before the gate was declared passing.

Representative protected mutations: remove `중 어디가` · remove `중 누가` ·
remove `A보다 B` · classify `작년보다` as cross-company · classify every `더` as
cross-company · classify every `차이` as cross-company · classify
`직전 보고 대비` as `company_comparison` · promote every `comparison_frame` to
executable `company_comparison` · treat `uncertain` as safe for issuer/reporter ·
route-dependent classification · relation-direction-dependent classification ·
company or question special-casing.

Two mutations have **no source to mutate** and were gated **structurally**
instead: the executable classifier contains no `relation`, `issuer`, `reporter`,
`support`, `corp_code`, `disclosure_route`, `task_type`, `event_type` or Gold
token, and neither production file contains a corpus company literal, a question
ID, or the known failing sentence.

### Implementation surface

Files changed by `7a0921e`:

- `app/reasoning/query_understanding.py`
- `app/reasoning/query_validation.py`
- `tests/test_query_understanding.py`
- `tests/test_query_validation.py`

No changes to `app/retrieval/`, `app/parsing/`, `app/generation/`,
`app/reasoning/holding_*`, `app/reasoning/holding_reporter.py`, P1-R, BGE, the
database schema, the public API schema, the relation artifact, or P0-C.

### Test gate

| | before | after |
|---|---|---|
| full suite | **1663 passed · 13 skipped · 1060 subtests** | **1686 passed · 13 skipped · 1139 subtests · 0 failed** |

| focused / regression | result |
|---|---|
| `test_query_understanding.py` + `test_query_validation.py` | **80 passed · 99 subtests** |
| planner · router · orchestrator · periodic | **88 passed** |
| API · schema · pipeline | **155 passed** |

Two warnings are present and are **not failures**: a Starlette/httpx deprecation
warning and a pytest cache-directory permission warning.

### Performance

Comparison classification costs approximately **0.214 ms/query** measured over
the alias table in use. **No retrieval call and no database call was added.**

### Invariants that must remain true

1. `comparison_frame` is derived from question text, canonical companies and
   textual structure alone.
2. Comparison operators are resolved by their left operand, not by vocabulary.
3. Structure outranks comparison vocabulary.
4. A temporal or metric anchor keeps a comparative question single-subject.
5. `uncertain` fails closed and stays AMBIGUOUS.
6. The comparison firewall precedes any issuer/reporter reinterpretation.
7. Existing `company_comparison` behaviour and all temporal comparison payloads
   are unchanged.

**Negative invariants — never do any of these:** use relation direction to
determine comparison semantics · use Gold labels or question IDs · use
company-specific allowlists · classify every `더` as cross-company · classify
every `차이` as cross-company · classify every `대비` as company comparison ·
treat a temporal `보다` as company-to-company automatically · promote every
`comparison_frame` into executable `company_comparison` · allow `uncertain`
through a future issuer/reporter reinterpretation · run a future P0-D.2 before
the comparison firewall · select a pair from 3+ companies · change retrieval or
ranking as part of this freeze.

### Known residual issues NOT solved

- `uncertain` intentionally produces a clarification rather than a guess.
- Fixtures are **constructed**; future Korean phrasing may be unseen.
- The temporal-anchor vocabulary may not cover all future phrases.
- Unknown alias surfaces degrade to `uncertain`.
- The **15** newly protected comparison forms remain **AMBIGUOUS rather than
  answered**.
- This is a **correctness firewall, not a cross-company comparison execution
  feature**.
- **Target B — H01 / HX02 first-loss diagnosis is complete: the frozen targets
  are benchmark-context-dependent. P0-D.2 remains deferred pending
  report-relative semantics.**

### Next production design question

Define report-relative semantics for `이번 보고`, `현재`, `직전보고`, and
`직전 보고 대비` before changing reporter retrieval representation. Only after
those semantics are explicit may a bounded reporter-aware query-representation
target be considered. Do not reopen P1-R, P1-B, top-k, or broad ranking work from
the frozen H01/HX02 benchmark targets.

### Reopen conditions

- a genuine cross-company comparison reaching issuer/reporter reinterpretation;
- an `uncertain` frame passing a future role resolution;
- a genuine issuer/reporter question blocked by a comparison frame;
- comparison classification shown to depend on relation data, route, or task
  type.

---

## TARGET B.2 — Reporter-Aware Deterministic Holding Report Index

**Status: FINAL FREEZE** · Frozen implementation commit:
`3f603299b395d11901bb4dbfcde1fb033cc374e6` (`3f60329`).

### Freeze boundary

This freeze covers the deterministic holding-report index and its bounded
selection/execution surface only. It does **not** claim end-to-end
report-relative execution, and it does **not** activate P0-D.2.

| surface | frozen state |
|---|---|
| Target B.2 index | **FINAL FREEZE** |
| `latest` / exact selection engine | **implemented** (`latest`, exact reference date, exact receipt date) |
| correction-free issuer/reporter pair | **executable**, subject to unique selection and an available requested projection |
| correction-bearing issuer/reporter pair | **fail-closed** as `correction_ambiguous` |
| `selected_context` | **intentionally unsupported**; B.2 will not invent a prior report selection |
| P0-D.2 | **still inactive**; `PRIMARY_COMPANY_WITH_REPORTER` is not production-active |

### Frozen corpus and index evidence

| measure | frozen result |
|---|---|
| source completeness | **1,083 / 1,083 holding documents source-complete** |
| report-relative projections | **1,118** |
| indexed records | **1,116** |
| issuer/reporter pairs | **192** |
| correction-bearing pairs | **22 / 192** → `correction_ambiguous` |
| reference date ≠ receipt date | **697 / 1,116** |
| distinct-document latest-date tied pairs | **7** |
| `data/corpus/holding_report_index.json` | **880.1 KiB** |
| artifact load | **≈ 13 ms** |
| indexed lookup | **≈ 1.86 μs** |

The artifact declares the corpus complete and preserves the two date axes.
`latest` is ordered by **reference date**, not receipt date. A same-date tie is
never broken with receipt number, document id, file order, or another technical
field.

### Implementation surface

Files introduced by `3f60329`:

- `app/reasoning/holding_report_index.py`
- `data/corpus/holding_report_index.json`
- `scripts/build_holding_report_index.py`
- `tests/test_holding_report_index.py`

The implementation enumerates records by issuer and the frozen canonical
reporter key, validates corpus identity and completeness, selects one report
deterministically when the contract permits it, and projects only fields stated
by that selected filing. It adds no P0-D.2 production wiring.

### Test gate

| runner | frozen result |
|---|---|
| pytest | **1,775 passed / 13 skipped** |
| unittest | **1,739 OK / 13 skipped** |

### Invariants that must remain true

1. An incomplete, stale, malformed, or schema-incompatible artifact fails
   closed; it cannot answer `latest` from a partial corpus.
2. `latest` uses reference date. Exact reference-date and receipt-date
   selection remain separate operations.
3. Correction finality is checked before any date comparison. With no frozen
   finality source, every correction-bearing pair returns
   `correction_ambiguous` and no selected record.
4. Ties remain ambiguous; technical fields are serialization aids, never
   semantic tie-breakers.
5. `selected_context` remains unsupported until a separate, explicit contract
   supplies the already-selected report.
6. This index freeze does not authorize P0-D.2 activation, retrieval/ranking
   changes, reporter propagation, or benchmark-specific report selection.

### Known residual issues NOT solved

- Correction finality is not available for this corpus.
- A standalone `selected_context` phrase does not identify a report.
- P0-D.2 role resolution and production integration remain inactive.
- The frozen H01/HX02 benchmark targets remain context-dependent; B.2 does not
  manufacture their missing report anchor.

### Reopen conditions

- corpus identity or completeness changes without a regenerated artifact;
- a correction-bearing pair returns a selected report without proven finality;
- a latest-date tie is resolved by a technical tie-breaker;
- `selected_context` is silently promoted to `latest`;
- P0-D.2 activation is inferred from this index freeze alone.

---

## TARGET B.3 — Holding Correction Finality Artifact

**Status: FINAL FREEZE** · Frozen implementation commit:
`da65d232fb44ecec995119d17b8ae562147f3a59` (`da65d23`).

### Freeze boundary

This freeze is limited to the corpus-bound materialization of P0-A holding
correction groups and their document-level attachment to the B.2 report index.
It does not choose among multiple projections inside a final document, provide
a deictic report context, or activate P0-D.2.

| surface | frozen state |
|---|---|
| correction finality artifact groups | **20 resolved / 19 ambiguous groups** |
| `latest` selection | **179/192 latest-resolvable pairs** |
| unresolved correction timelines | **12 correction-bearing pairs remain fail-closed** as `correction_ambiguous` |
| exact reference selection within latest-resolvable pairs | **5 exact-reference points remain projection-ambiguous** |
| `selected_context` | **unsupported** |
| P0-D.2 | **inactive**; `PRIMARY_COMPANY_WITH_REPORTER` is not production-active |

The 192-pair latest result partitions into **179 `resolved` + 12
`correction_ambiguous` + 1 `ambiguous`**. The last pair has a proven final
document but several projections inside that document; B.3 preserves them all
and does not invent a projection-level tie-breaker. The five exact-reference
points are the additional projection-ambiguous points observed within the 179
latest-resolvable pairs.

### Frozen contract

- P0-A remains the sole authority for correction-group status, final document,
  resolution rule and confidence. B.3 does not rebuild or reinterpret the
  correction graph.
- A `resolved` group may remove only document ids P0-A proved superseded. An
  `ambiguous` or `unresolved` group supersedes nothing.
- Collapse is document-level and happens before `latest` or exact-date ordering.
  Every projection belonging to the proven final document survives collapse.
- A missing, incomplete, malformed, schema-incompatible or corpus-mismatched
  finality artifact leaves correction-bearing pairs fail-closed.
- Correction-free pair results are unchanged. Selector meanings, reference and
  receipt date axes, and role projections remain B.2 contracts.
- `selected_context` remains unsupported and this artifact adds no P0-D.2
  production wiring.

### Implementation surface

Files changed by `da65d23`:

- `app/reasoning/holding_correction_finality.py`
- `app/reasoning/holding_report_index.py`
- `data/corpus/holding_correction_finality.json`
- `scripts/build_holding_correction_finality.py`
- `tests/test_holding_correction_finality.py`

### Focused test gate

`tests/test_holding_correction_finality.py` plus
`tests/test_holding_report_index.py`: **123 passed · 696 subtests · 0 failed**.
The pytest cache-directory permission warning is not a test failure.

### Known residual issues NOT solved

- The **19 ambiguous groups** remain unproven.
- The **12 correction-bearing pairs** remain `correction_ambiguous` and return
  no selected report.
- Projection-level ambiguity remains: one latest pair and five exact-reference
  points inside otherwise latest-resolvable pairs still decline.
- `selected_context` is unsupported, and P0-D.2 remains inactive.

### Reopen conditions

- an ambiguous P0-A group is collapsed or made to supersede a document;
- a finality artifact is attached across a corpus-identity mismatch;
- a final document's projections are silently reduced by position, value or
  file order;
- any of the 12 residual correction-bearing pairs stops failing closed without
  new P0-A proof;
- `selected_context` or P0-D.2 activation is inferred from this artifact freeze.

---

## Summary

| Phase | Status | Commit |
|---|---|---|
| P0-A Correction Graph | FINAL FREEZE | `9200ee1` (from `0e6542a`) |
| P0-B Corporate Event Timeline | FINAL FREEZE | `e21ee27` |
| P0-C Multi-Document Planner | FINAL FREEZE | `ba6a7c3` (from `d04c587`) |
| P0-D Query Understanding & Verification | FINAL FREEZE | `7a7da17` |
| P0-D.1 Multi-company Query Understanding | **DIAGNOSIS COMPLETE — KEEP P0-D FROZEN** | — |
| P0-D.2 Issuer / Reporter Role Resolution | **ACTIVATION DEFERRED · TARGET B H01/HX02 FIRST-LOSS DIAGNOSIS COMPLETE — BENCHMARK TARGET CONTEXT DEPENDENT** · Target A comparison-intent firewall **RESOLVED** (`7a0921e`) · no activation work authorized | `c3d5929` |
| P1-A2 Holding Evidence Routing Consistency | FINAL FREEZE | `1b8d08f` |
| P1-A3 Holding Structured Evidence Coverage Rescue | FINAL FREEZE | `53e480f` |
| P1-A4 Exact Holding Event Resolution | FINAL FREEZE | `d39a1b1` |
| P1-A4.1 Holding Reporter Normalization | FINAL FREEZE | `96d3968` |
| P1-A5-B Render Matching Holding Events Only | FINAL FREEZE | `eaec179` |
| P1-A5-A Ambiguity-Safe Holding Presentation | FINAL FREEZE | `39574f9` |
| P1-A5-A.1 Lossless Semantic Notice Preservation | FINAL FREEZE | `39574f9` |
| P1-B Filter Relaxation / Retrieval Recovery | **IMPLEMENTATION DEFERRED — strengthened by live BGE evidence** | — |
| P1-R Bounded Additive Document Recovery | FINAL FREEZE | `6503c77` |
| P1-C Table Sibling / Evidence Neighborhood | **DIAGNOSIS COMPLETE — IMPLEMENTATION DEFERRED · LIVE PERIODIC TARGET NOT ESTABLISHED** | — |
| INFRA-E1 Reproducible BGE-M3 Evaluation Environment | **PHASE 2 COMPLETE — P1-R UNBLOCKED** | — |
| Embedding Identity Hardening | FINAL FREEZE | `f592e9d` |
| Retrieval Vector-Availability Policy | FINAL FREEZE | `b1e31aa` |
| HX04 Acquisition Semantics | FINAL FREEZE | `7393842` |
| Comparison Intent Firewall (P0-D.2 Target A) | FINAL FREEZE | `7a0921e` |
| TARGET B.2 — Reporter-Aware Deterministic Holding Report Index | FINAL FREEZE | `3f60329` |
| TARGET B.3 — Holding Correction Finality Artifact | FINAL FREEZE | `da65d23` |
| Periodic Retrieval — Live BGE Diagnosis | **DIAGNOSIS COMPLETE — KEEP DEFERRED** | — |
