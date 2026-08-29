# Independent Eval v2 — STEP 4–5 Candidate Discovery Report

**Status:** candidates authored, prediction-blind. No agent was executed, no production code touched.
**Corpus:** `data/processed/structural_v2_1_full_4204/` (4,204 disclosures) + `data/corpus/manifest.jsonl`,
`holding_report_index.json` (1,116 records), `holding_correction_finality.json` (39 groups).
**Method:** corpus event → structural fact verification → reasoning requirement → expected behavior →
question. Gold values are pulled programmatically from the scan outputs; none were typed by hand.

---

## 1. Scan results

| Scan | Population | Usable for gold | Notes |
|---|---|---|---|
| A. Date-axis conflicts | 697 `reference_date ≠ receipt_date` | **608 both-axis deterministic** | buckets: 1–7d 389, 8–30d 197, 31–59d 59, 60–179d 47, 180+ 4, inverted 1 |
| B. Same-key collisions | 37 groups (issuer+reporter+reference_date) | 36 across distinct docs | 9 collapsible by B.3, 9 collapsible **with contradictory values**, 18 B.3-ambiguous, 1 same-doc multi-projection |
| C. Correction finality | 39 groups (20 resolved / 19 ambiguous) | **41/41 correction docs have a unique authoritative body value** | 16 also expose a distinct superseded value |
| D. Issuer/reporter collision | 13 records / 3 reporter names | **13/13 clean** | 현대모비스→현대자동차 (10), 삼성전자→레인보우로보틱스 (2), 삼성SDI→삼성E&A (1) |
| E. First report | 48 no-previous records | **48/48 true first reports**, 46 clean | body `직전 보고일` / `직전 보유주식수` are literally `-`; zero data-quality artefacts |
| F. Contract lifecycle | 20 termination filings | **13 chains resolvable in-corpus** | link comes from the filing's own `관련공시` field, not from title similarity |
| F2. Contract amount corrections | 75 exchange corrections with a changed 계약금액 | 29 with exactly one in-corpus original | includes multi-hop chains (현대건설 4-hop and 10+-hop) |
| G. Same-day multi-disclosure | 391 company+day groups | 83 cross-doc_group, 269 same-subtype | e.g. HD현대중공업 3× 공급계약 in one day, KB금융 2× 자기주식취득결정 |
| H. Periodic table+text | 3,199 "split" tables in a 40-doc sample | **≈0 genuine evidence gaps** | see negative finding below |

### Two findings that shaped the set

**1. Correction filings carry their own ground truth.** A holding 정정신고 contains paired
`정정 전` / `정정 후` tables *and* an authoritative body table
(`제1부 보고의 개요 > 3. 보유주식등의 수 및 보유비율`, `aclass=EXTRACTION`). The body always equals
정정 후. All **41** correction documents have a unique body value; **16** also expose a *different*
superseded value in the same document. That superseded value is the perfect distractor: it is
in-document, correctly labelled, and factually wrong.

Example — 셀트리온홀딩스 / 셀트리온, reference date 2025-06-13, `holding_20250804000300`:

| table | section | 보유주식수 |
|---|---|---|
| t0008 | 정정 신고 (주2) 정정 **전** | 69,171,772 |
| t0009 | 정정 신고 (주2) 정정 **후** | 69,160,090 |
| t0033 | **제1부 보고의 개요 → 3. 보유주식등의 수 및 보유비율** | **69,160,090** ← gold |

**2. The P1-C "table sibling gap" hypothesis is not supported by this corpus.** The 3,199 apparently
split tables are `table_id` collisions across the *parts* of a multi-part periodic document, not
genuine within-table splits. Real table chunks carry their own `header_rows`, `column_headers` and
`unit`, so a value row is self-sufficient. The only legitimate cross-chunk dependency found is the
explicit pointer pattern (`※상세 현황은 '상세표-1. …' 참조`), where a summary table names a detail
table elsewhere in the document. **T17 was therefore capped at 1 candidate rather than the 3–7
originally proposed.** Quota was not filled with manufactured cases.

---

## 2. Candidate count

**95 candidates** (`candidates.jsonl`), target range 90–120.

| Quality | n | Meaning |
|---|---|---|
| **A** | 84 | fully corpus-verified, deterministic gold, high novelty and diagnostic value |
| **B** | 10 | corpus-verified, but expected behavior depends on a policy judgment (T3 over-refusal split, T20 citation choice) |
| **C** | 1 | needs evaluator support or further verification (T17) |
| REJECT | 0 | rejected material was dropped during authoring, not recorded as candidates |

Difficulty: **hard 45 / medium 35 / simple 15** (47% / 37% / 16%).
Expected behavior: **answer 79 / insufficient_evidence 15 / clarify 1** (17% non-answer).

### Category distribution vs. target

| Category | n | Target | Note |
|---|---|---|---|
| T1 single-doc unseen | 10 | 8–12 | ✔ |
| T2 resolved correction | 8 | 6–10 | ✔ (5 direct + 3 before/after-pair form) |
| T3 correction chain unprovable | 4 | 5–8 | **reduced** — 4 of 8 were 삼성물산 date-only variants, which the novelty rule forbids |
| T4 event lifecycle | 6 | 5–8 | ✔ |
| T5 multi-doc synthesis | 7 | 6–10 | ✔ |
| T6 two-company comparison | 2 | 3–5 | **reduced** — the 2-per-document limit binds; each comparison consumes two filings already used by T1 |
| T7 3+ company comparison | 1 | 3–5 | **reduced** — same constraint; one 4-company item consumes four filings |
| T8 issuer/reporter collision | 8 | 6–10 | ✔ (8 of the 13 existing records) |
| T9 holding latest | 5 | 4–7 | ✔ |
| T10 exact reference date | 4 | 3–6 | ✔ |
| T11 exact receipt date | 4 | 3–6 | ✔ |
| T12 date-axis conflict | 3 | 4–7 | **reduced** — wide-gap records are dominated by 국민연금공단; more would breach reporter diversity |
| T13 previous/change | 4 | 3–6 | ✔ |
| T14 acquisition | 2 | 3–6 | **reduced** — few unseen issuers have a clean 세부변동내역 row not already used elsewhere |
| T15 first report | 6 | 4–7 | ✔ |
| T16 periodic extraction | 4 | 4–7 | ✔ (2 companies × 2 fields) |
| T17 cross-table reference | 1 | 3–7 | **reduced** — see negative finding above |
| T18 negative / insufficient | 6 | 4–8 | ✔ |
| T19 deictic | 4 | 3–5 | ✔ |
| T20 same-day two filings | 6 | 3–7 | ✔ |

---

## 3. Diversity

* **42 unique companies**, **30 unique reporters**, **92 unique gold documents**, 26 semantic templates.
* Max company concentration: **현대자동차 6 / 95 (6.3%)** — under the 10% limit (≤10 candidates).
* Max template concentration: **`corrected-final-holding-value` 9 / 95 (9.5%)** — under the limit.
* **No document backs more than 2 candidates.** 24 documents back exactly 2, and in every case the two
  differ in reasoning requirement (reference vs receipt selector; current value vs absent previous;
  extraction vs comparison).
* Reporter spread: 국민연금공단 appears in only **3** candidates (Gold60 used it in 26 of 28 holding
  questions). Individual reporters (장병규 · 방시혁 · 양현석 · 조정호 · 최윤범 · 오준호), domestic corporates
  ((주)한화 · (주)LG · 삼성물산 · 에코프로 · 한국조선해양 · 미래에셋캐피탈) and foreign institutions
  (The Capital Group · Fidelity · BlackRock · GIC · Government of Singapore · Invesco · Schroder ·
  Morgan Stanley · Silchester · T. Rowe Price · Tencent Music) are all represented.

---

## 4. Novelty audit

| Overlap with Gold60 | Candidates | Assessment |
|---|---|---|
| Same **document** | **0 / 95** | no Gold60 filing is reused anywhere |
| Same **semantic template** | **0 / 95** *(withdrawn in STEP 6 — see correction below)* | measured only as literal template equality |
| Same **reporter** | 3 / 95 | 국민연금공단 only, all in T12 where the wide date gap is the new requirement |
| Same **company** | 22 / 95 | company reuse only; in every case the disclosure, event and reasoning are new |
| P0-C set company | 16 / 95 | the P0-C set only counts events (`몇 건`, `해지된 것이 있는가`); none of these are counting questions |

> **STEP 6 CORRECTION.** The "same semantic template = 0" figure above was an overstatement: it
> measured only literal template equality and ignored reasoning-family overlap. The three-layer
> audit in `gold_verification_report.md` supersedes it. Measured over the 69 proposed items,
> **26% have `exact_template_overlap`** (T1, T10, T13, T14, T19) and **64% have
> `semantic_family_overlap`**. Document overlap remains genuinely 0. The overlapping items are
> retained on purpose as the baseline-health band; do not cite "semantic overlap 0" anywhere
> downstream.

The 22 company overlaps are deliberate and defensible. 삼성전자 appears as the *issuer being reported
on* (Gold60 uses it only as a periodic subject); 셀트리온 and 하이브 appear as holding **issuers with
correction chains**, a structure Gold60 lacks entirely; 레인보우로보틱스 appears as an issuer in an
issuer/reporter collision and as a contract-amount correction, neither of which Gold60 touches.

---

## 5. Ambiguity and negative candidates (16)

All 15 `insufficient_evidence` and 1 `clarify` items carry `ambiguity_reason`, `forbidden_fallback`
and `expected_behavior_reason`. Grouped:

* **T15 first report (6)** — `직전 보유주식수` / `직전 보유비율` where the body cell is literally `-`.
  Forbidden fallback: answering 0, or restating the current holding as the previous one.
* **T19 deictic (4)** — standalone `이번 보고` / `직전보고` with no report in context. Forbidden
  fallback: silently promoting to latest, or answering from retrieval rank 1. Directly re-tests the
  `b8799e9` fail-closed contract on unseen identities.
* **T18 withheld or absent field (5)** — 계약금액 deferred under 공시유보 (두산퓨얼셀 to 2025-01-31,
  LG에너지솔루션 to 2031-12-31), 취득단가 explicitly 미기재, a 대량보유보고 asked for a purchase price it
  never records, and a correction that discloses the counterparty while the amount stays withheld.
  Every one has a plausible adjacent number to grab — usually 최근매출액 in the neighbouring row.
* **T18 clarify (1)** — 삼성바이오로직스 filed two Pfizer Ireland contracts with the same contract title
  and different amounts (240,993,039,040 vs 922,746,708,000). The right move is to ask which.

---

## 6. Runner gaps — 38 candidates

Inspecting `app/agent/gold60_evaluation.py:366` (`_end_to_end_failure_class`) confirms three
limitations. **No evaluator code was modified.**

1. **Correct refusals are scored as failures (16 candidates).** The classifier returns
   `"answer_not_supported"` whenever `answerable` is False, before any gold comparison. Every
   fail-closed candidate — the entire T15/T19 families and most of T18 — would be recorded as a
   failure even when the behavior is exactly right. Fixing this requires an expected-behavior field
   in the question schema, which is a STEP 8 decision, not a STEP 5 one.
2. **Single-valued gold citation (23 candidates).** `gold_doc_cited` is one boolean over one
   `doc_id`. Multi-document items (T4, T5, T6, T7, T20) cannot express "cite both/all", which is
   precisely the citation-completeness signal those families exist to measure.
3. **Derived answers are literal-matched (10 candidates).** `all_evidence_terms_present` does
   substring matching, so a correctly computed delta expressed in another format scores as missing
   evidence.

Per the brief, no candidate was discarded or forced into a wrong single-document gold because of
this. The affected items keep full multi-document gold and are flagged.

---

## 7. Server verification queue (8 candidates)

All T4 lifecycle items and the two T5 lifecycle deltas. The contract→termination link was taken from
the termination filing's own `관련공시` field (in-document evidence, not title similarity), but the
authoritative relation lives in the DB (`db/006_correction_graph.sql`,
`db/007_corporate_event_timeline.sql`), which is not reachable from this machine. These need a
read-only PostgreSQL confirmation before gold freeze.

Also worth a server check, though not blocking: 7 of the 13 lifecycle chains reference original
contracts filed **before the corpus window**, so those terminations are genuinely un-linkable
in-corpus and were excluded from candidates.

---

## 8. Recommendation for STEP 6

* **Proceed to gold verification: 84 quality-A candidates** — deterministic, corpus-verified,
  no unresolved policy question.
* **Proceed with a decision attached: 10 quality-B.** The 4 T3 items need an explicit ruling on
  whether corpus truth or the system's fail-closed contract is the gold (recommendation: keep corpus
  truth as gold, and score a refusal as `A1 over-refusal` rather than a factual error — this keeps
  the eval independent of the implementation). The 6 T20 items need a ruling on whether citing one
  of two equally valid documents is a pass.
* **Hold 1 quality-C candidate (T17)** until the evaluator can express cross-chunk evidence.
* **Expected final v2 size: 65–75** after STEP 7 filtering, consistent with the original plan.

Reserve material not yet authored, available if a category needs topping up: 25 further clean
first-report records, 5 more issuer/reporter collision records, 604 further date-axis records, 8 more
resolved correction chains, and 21 more contract-amount correction pairs.
