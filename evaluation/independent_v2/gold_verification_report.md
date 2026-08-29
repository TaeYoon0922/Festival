# Independent Eval v2 — STEP 6 Gold Verification Report

**Status:** gold verified and enriched, final set **proposed but not frozen**.
Prediction-blind throughout: the frozen agent was not started, `/answer` was not called, and no
agent prediction was inspected. No production code was modified.

| | count |
|---|---|
| Candidates entering STEP 6 | 95 |
| **Verified** (integrity-clean, corpus-grounded) | **93** |
| **Rejected** during verification | **2** |
| Verified replacements authored | 2 |
| **Held** (category cap / evaluator dependency) | 26 |
| **Proposed final set** | **69** |

Integrity problems in the verified pool: **0**. In the proposed final set: **0**.

---

## 1. Rejections

Both were caught by the STEP 6 discipline rather than by the STEP 5 scans, which is exactly what
this step is for.

**IEV2-C041 — T15, SK하이닉스 / The Capital Group (`holding_20240527000026`).**
The STEP 5 scan classified it a first report because the body's 직전 cells read `-`. The STEP 6
chronology check found an **earlier filing for the same identity** (`holding_20230307000265`).
The filing states no previous holding while the corpus does contain one, so neither `answer` nor
`insufficient_evidence` is defensible as gold. Rejected rather than guessed. This is precisely
the A/B distinction §9 of the brief demanded, and it is the only case out of 48 where the
body-cell heuristic and the corpus chronology disagree.

**IEV2-C083 — T14, LG씨엔에스 / (주)LG (`holding_20250207000688`).**
Same-row verification passed, but the row has 증감주식수 == 보유주식수 == 43,557,218 (a
new-listing re-report where 변동전 is `-`). A system that confuses `acquired_shares` with
`after_shares` still scores correct, so the item has **no discriminating power** for the very
thing T14 exists to test. Rejected on diagnostic grounds, not on gold correctness.

### Replacements (both verified from raw structural JSON)

| New item | Why it is stronger |
|---|---|
| 하나금융지주 / The Capital Group `holding_20230303000669` | Chronology verified — earliest of 16 filings for the identity by both axes. The filing's 변동 사유 field literally reads **"신규보고의무 발생"**, giving in-document proof of first-report status. |
| 한전기술 / Van Eck Associates `holding_20260325000205` | Detail row t0023/3 has 변동전 1,905,209 · 증감 **8,824** · 변동후 1,914,033 — all distinct, so confusion is detectable. The filing also holds a 2026.03.12 row, so the date must select the row. |

---

## 2. T3 re-audit — all four reclassified

The brief's core correction was applied: **artifact ambiguity ≠ corpus ambiguity.**

For each of the four T3 items I checked (a) how many filings share
`(issuer, reporter, reference_date)`, (b) whether the correction is the unique latest by receipt
date, (c) whether any later restatement exists, and (d) whether the 제1부 body table yields one
value.

| Item | B.3 status | Filings for the key | Unique latest correction | Later restatement | Body value | Corpus ambiguity | New expected_behavior |
|---|---|---|---|---|---|---|---|
| 삼성전자 / 삼성물산 `holding_20241025000530` | ambiguous (orphan) | 2 | yes | none | 1,198,889,258 | **no** | **answer** |
| 한화에어로스페이스 / (주)한화 `holding_20240906000499` | ambiguous (orphan) | 2 | yes | none | 15,601,252 / 34.23% | **no** | **answer** |
| 한화솔루션 / (주)한화 `holding_20241008000479` | ambiguous (orphan) | 2 | yes | none | 63,022,658 | **no** | **answer** |
| 에코프로비엠 / 에코프로 `holding_20230417000306` | ambiguous (orphan) | 2 | yes | none | 50,486,899 | **no** | **answer** |

All four are **T3A: artifact-ambiguous, corpus-determinate.** They are folded into
`T2_resolved_correction` carrying `correction_subtype = "T3A_artifact_ambiguous_corpus_determinate"`,
`artifact_ambiguity: true`, `corpus_ambiguity: false`. The eight originally-resolved items carry
`correction_subtype = "T2_chain_linked"`. That preserves the diagnostic split without letting the
implementation set the gold.

Crucially, the T2 and T3A groups are **structurally identical at corpus level** — same filing
count, same uniqueness, same body-table rule. B.3's resolved/ambiguous distinction tracks only
whether the artifact could link the chain. A fail-closed response on a T3A item is therefore an
`A1 over-refusal`, not a factual error, and the STEP 10 taxonomy must score it that way.

### Search for a genuine T3B (corpus-ambiguous) case

I looked for one and **found none in the holding-correction space.** Only two groups have ≥2
correction filings for a single reference date:

- 에스엠 / T. Rowe Price ref 20251201 — three filings, all three body values identical
  (1,195,259). Determinate.
- 에코프로비엠 / 에코프로 ref 20230403 — the closest case: two corrections 11 days apart with
  **different** body values (50,484,612 then 50,469,415). It still resolves, because the later
  correction's 직전 보유주식수 is 50,486,899, which is exactly the *corrected* value of the
  preceding 20230309 filing. The chain is internally consistent and the terminal is provable from
  document content.

**Finding: in this corpus, holding-correction ambiguity is an artifact property, not a corpus
property.** The only genuine corpus ambiguity found anywhere is the twin 삼성바이오로직스 Pfizer
contracts, already carried as the single `clarify` item. The 에코프로비엠 double-correction is
recorded as reserve material — it would make an excellent hard cross-document item.

---

## 3. T20 citation policy

Per-document body values were extracted independently for all six pairs:

| Pair | Doc A body | Doc B body | Identical | Policy |
|---|---|---|---|---|
| 고려아연 / 국민연금공단 | 934,443 / 4.51% | 934,443 / 4.51% | yes | `any_of` |
| HMM / 한국해양진흥공사 | 330,867,712 / 32.28% | 330,867,712 / 32.28% | yes | `any_of` |
| 케이티 / Silchester | 13,084,357 / 5.19% | 13,084,357 / 5.19% | yes | `any_of` |
| 케이티 / T. Rowe Price | 12,898,415 / 5.00% | 12,898,415 / 5.00% | yes | `any_of` |
| 셀트리온 / 셀트리온홀딩스 | 62,296,973 / 28.46% | 62,296,973 / 28.46% | yes | `any_of` |
| LIG / Government of Singapore | 1,043,530 / 4.743% | 1,043,530 / 4.743% | yes | `any_of` |

All six are **Case A**: the fact is unique and each filing independently proves it, so citing
either document is correct. **No item requires both citations merely because two documents
exist.** Case C (differing values plus an unspecified filing) does not occur here — those
divergent-value pairs were already routed to T2/T3A, where the question names the correction.

Each item now carries both filings in `gold_doc_ids` and `gold_chunk_ids`, the union of their
`source_refs`, and a `per_document_body_value` map so a scorer can verify either side.

Final citation-policy distribution: `single` 49 · `all_required` 13 · `any_of` 4 · `none` 3.

---

## 4. Novelty audit — corrected

The STEP 4–5 claim of **"semantic overlap 0" was an overstatement** and is withdrawn. Three
layers, measured over the 69 proposed items:

| Layer | Count | Reading |
|---|---|---|
| `exact_template_overlap` | **18 / 69 (26%)** | near-identical question shape to a Gold60 or prior diagnostic item |
| `semantic_family_overlap` | **44 / 69 (64%)** | same reasoning family as something already tested |
| `same_company` | 12 / 69 | company reuse only |
| `same_reporter` | 3 / 69 | 국민연금공단 only |
| `same_doc` | **0 / 69** | no Gold60 filing reused |

Categories with `exact_template_overlap = true`: **T1** (contract field extraction, like E01–E08),
**T10** (exact reference date, like HX01/HX05), **T13** (previous/change, like HX02/HX03),
**T14** (변동일/변동후 row, like HX04/HX08), **T19** (identical deictic wording to H01/H03/HX02).

These are retained deliberately. T1/T9/T10/T13/T14 are the **baseline-health band** — they exist
to separate "novel structure broke it" from "it was already broken", and that only works if the
template matches. T19 is the sharpest case: the wording is identical to Gold60's, but the
*expected behavior differs*, and the identities have 13–39 filings each, so a latest-promotion
produces a visibly wrong document instead of the plausible-looking one Gold60's single-filing
companies would yield.

What is genuinely novel is carried in `structural_novelty` on every item. **25 of 69 (36%)** have
no semantic-family precedent at all: T2 correction state, T4 lifecycle, T5 multi-doc arithmetic,
T6/T7 comparison, T18 negatives.

---

## 5. Gold integrity

All checks run over both the 93-item verified pool and the 69-item proposed set. **Zero problems
in both.**

| Check | Result |
|---|---|
| Duplicate questions | 0 |
| Missing gold documents | 0 |
| Missing gold chunks (verified against structural JSON) | 0 |
| Answer item with empty evidence | 0 |
| Answer chunk without `source_refs` | 0 (6 T20 items fixed during this step) |
| Refusal item carrying a fixed answer | 0 |
| Refusal without `ambiguity_reason` | 0 |
| Refusal without `forbidden_fallbacks` | 0 |
| Derived values without a `derivation` rule | 0 |
| Invalid derived arithmetic | 0 (all differences and percentages recomputed from `source_values`) |
| Numeric unit consistency | 0 bad units — only `shares`, `percent`, `KRW`, `count` |
| `all_required` with <2 docs / `any_of` with <2 docs | 0 |

**Numeric gold:** 47 of the 57 answer items carry structured `gold_numeric` with `value` + `unit`
(and `secondary` for pair-form answers such as 정정 전/후 and 변동 전/후). The remaining 10 are
legitimately non-numeric — a lifecycle outcome date, counterparty names, a contract end date, an
ordering, and a cross-table pointer.

**Derived arithmetic** was recomputed from source values for all 7 multi-document items, e.g.
`4,150,000,000 → 4,250,000,000` gives `difference = +100,000,000`, `pct_change = +2.4096%`, and
`227,500,000,000 − 114,800,000,000 = 112,700,000,000` remaining.

---

## 6. Proposed final set

**69 items** — within the 65–75 target.

| Category | n | | Category | n |
|---|---|---|---|---|
| T1 single-doc unseen | 7 | | T12 date-axis conflict | 3 |
| T2 resolved correction (incl. 4 T3A) | 8 | | T13 previous/change | 3 |
| T4 event lifecycle | 4 | | T14 acquisition | 2 |
| T5 multi-doc synthesis | 5 | | T15 first report | 4 |
| T6 two-company comparison | 2 | | T16 periodic extraction | 3 |
| T7 multi-company comparison | 1 | | T18 negative/insufficient | 5 |
| T8 issuer/reporter collision | 5 | | T19 deictic | 3 |
| T9 holding latest | 4 | | T20 same-day two filings | 4 |
| T10 exact reference date | 3 | | T17 cross-table | **0 (held)** |
| T11 exact receipt date | 3 | | | |

**Difficulty:** hard 33 (48%) · medium 25 (36%) · simple 11 (16%).
**Behavior:** answer 57 (83%) · insufficient_evidence 11 · clarify 1 (**17% refusal**).
**Quality:** A 65 · B 4.
**Diversity:** 39 companies · 27 reporters · 70 gold documents. Max company 한국항공우주 6/69
(8.7%); max template `corrected-final-holding-value` 5/69 (7.2%).

Against the recommended distribution, refusals land at 17% versus the suggested 10–15%. I kept
the extra items deliberately: they are the only detector for the dangerous failure direction
(answering when the corpus does not support an answer), and the brief puts quality above quota.

**Held: 26.** 25 are category-cap reserve, retained in `candidates.jsonl` with a `hold_reason`.
1 is the T17 cross-table item, held pending evaluator support for cross-chunk evidence.

---

## 7. Refusal cases in the final set (12)

| Kind | n | Gold condition |
|---|---|---|
| `insufficient_evidence` — first report | 4 | 직전 cells are `-`; one filing additionally states 변동 사유 = 신규보고의무 발생 |
| `insufficient_evidence` — deictic | 3 | no report is named by the question; `gold_doc_ids` deliberately empty, `citation_policy: none` |
| `insufficient_evidence` — withheld/absent field | 4 | 계약금액 deferred under 공시유보, 취득단가 미기재, or a price a 대량보유보고 never records |
| `clarify` | 1 | two 삼성바이오로직스 Pfizer contracts share a title; needs a discriminator |

Every one stores behavioural conditions rather than a fixed sentence: `expected_answer` is
`null`, and each carries `ambiguity_reason`, `forbidden_fallbacks` (list) and
`expected_behavior_reason`. The clarify item also carries `clarification_requirements`. Wording
differences cannot cause a false failure.

---

## 8. Server verification — 5 items still held

The 4 T4 lifecycle items and 1 T5 lifecycle delta in the final set carry
`server_verification_required: true`. Their contract→termination links come from each filing's
own `관련공시` field (in-document evidence), but the authoritative relation lives in
`db/006_correction_graph.sql` / `db/007_corporate_event_timeline.sql`. **No PostgreSQL is
reachable from this machine** (port 5432 closed, no `.env`), so no relation was confirmed and
none was guessed.

These 5 must clear a read-only DB check before gold freeze. All other 64 items are verified
entirely from the tracked structural corpus and need no server access.

---

## 9. Decision

**GOLD FREEZE BLOCKED** — on two specific, bounded items:

1. **5 lifecycle items await read-only DB relation verification.** Nothing else blocks them.
2. **The evaluator cannot score 28 of the 69 items** (see `evaluator_requirements.md`). Freezing
   gold that the runner inverts — scoring a correct refusal as a failure — would produce a
   misleading STEP 9 result.

Neither blocker is a gold-quality problem. The 64 non-lifecycle items are corpus-verified,
integrity-clean and ready to freeze the moment the evaluator semantics are agreed.

**Recommended order:** agree the evaluator schema → confirm the 5 relations on the server →
freeze `gold.jsonl` → then run STEP 9.
