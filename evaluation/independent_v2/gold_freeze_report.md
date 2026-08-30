# Independent Eval v2 — Gold Freeze Report (STEP 8)

**Decision: GOLD FREEZE READY.** `gold.jsonl` is written, audited and hash-verified.

**Prediction-blind.** No API was started, no `/answer` call was made, and no frozen-agent output
was inspected at any point during candidate discovery, gold authoring, verification or freeze.
The evaluator has never been fed a live payload — only hand-built synthetic fixtures.

---

## 1. Final gold

| | |
|---|---|
| Items | **69** |
| SHA256 | `d574e696cda10a3b7165dfe20dae8fdfad8742bb5a05dd71217547facb01565f` |
| Bytes / lines | 154,284 / 69 |
| `GOLD_HASH_VERIFIED` | **true** (recomputed twice independently, plus `certutil`) |
| Source | `proposed_gold.jsonl` — 69 selected items extracted and re-audited, never copied wholesale |
| Held / rejected items included | **0** |

## 2. Distributions — all match the STEP 7 expectation exactly

**Behavior:** answer 57 · insufficient_evidence 11 · clarify 1
**Difficulty:** hard 33 · medium 25 · simple 11
**Quality:** A 65 · B 4
**Citation policy:** single 49 · all_required 13 · any_of 4 · none 3
**Diversity:** 39 companies · 27 reporters · 70 gold documents

| Category | n | | Category | n |
|---|---|---|---|---|
| T1 single-doc unseen | 7 | | T12 date-axis conflict | 3 |
| T2 resolved correction | 8 | | T13 previous/change | 3 |
| T4 event lifecycle | 4 | | T14 acquisition | 2 |
| T5 multi-doc synthesis | 5 | | T15 first report | 4 |
| T6 two-company comparison | 2 | | T16 periodic extraction | 3 |
| T7 multi-company comparison | 1 | | T18 negative/insufficient | 5 |
| T8 issuer/reporter collision | 5 | | T19 deictic | 3 |
| T9 holding latest | 4 | | T20 same-day two filings | 4 |
| T10 exact reference date | 3 | | | |
| T11 exact receipt date | 3 | | | |

T2 comprises 5 chain-linked corrections plus **3 of the 4** reclassified T3A items (`correction_subtype = T3A_artifact_ambiguous_corpus_determinate`): C010 한화에어로스페이스, C011 한화솔루션, C012 삼성전자. The fourth (C009 에코프로비엠) sits in the held reserve because the T2 category cap is 8 — it was capped out, not rejected, and its gold is equally valid.

## 3. Integrity checks — all pass

| Check | Result |
|---|---|
| **A. Identity** — 69 unique `question_id`, 69 unique `query`, no held/rejected item | ✔ |
| **B. Provenance** — every answer item has valid gold docs, chunks resolve against the structural JSON, chunks carry `source_refs`; deictic refusals legitimately carry zero docs with `citation_policy = none` | ✔ |
| **C. Citation cardinality** — `single` has exactly 1 required; `any_of` ≥ 2 acceptable; `all_required` ≥ 2 required; `none` only on non-answer items | ✔ |
| **D. Multi-doc integrity** — no T4/T5/T6/T7 item collapsed to one document; all carry `all_required` | ✔ |
| **E. Numeric** — units restricted to `shares`/`percent`/`KRW`/`count`; every secondary value carries a role label; all derived arithmetic recomputed from `source_values` | ✔ |
| **F. Refusals** — all 12 have `expected_answer = null`, `gold_numeric = null`, an `ambiguity_reason` and `forbidden_fallbacks`; the clarify item has `clarification_requirements` | ✔ |
| **G. Lifecycle** — exactly the 5 expected items carry `corpus_verified_db_unconfirmed` with per-item relation evidence, and `corpus_verified` stays `true` | ✔ |
| **Schema consistency** — all 69 items share an identical key set (37 keys), null/empty where not applicable | ✔ |

One defect was found and fixed during this audit: `IEV2-C097` (한전기술 / Van Eck) carried a
secondary numeric value without a role label. It is now `role: "before"`, so the 변동 전/변동 후
pair is role-bound and a role swap is detectable. The audit refused to write `gold.jsonl` until
this was corrected.

The 13 `all_required` items are the 5 T5, 4 T4, 2 T6, 1 T7 — plus `IEV2-C093`, the clarify item,
where establishing that two 삼성바이오로직스 Pfizer contracts match the description genuinely
requires referencing both filings.

## 4. Lifecycle provenance decision

Option (b) adopted. **Gold truth is the canonical provided disclosure corpus.** The PostgreSQL
`correction_relations` / `corporate_events` tables are derived implementation artifacts and are
not a prerequisite for Gold validity where the filings themselves uniquely establish the relation.

The five lifecycle items — C072, C074, C075, C076, C077 — are frozen with
`lifecycle_relation_status: "corpus_verified_db_unconfirmed"`. Each relation rests on three
concurring in-document identifiers: the termination filing's own `관련공시` back-reference
resolving to exactly one corpus filing, an exact 계약상대방 match, and an exact 체결계약명 match
(three of the four also match on amount).

**If the production event graph later fails to reproduce one of these relations, that is a
production relation/resolver finding — not a Gold error.** `server_relation_verification.md`
records the relation as VERIFIED, the DB corroboration as UNCONFIRMED, and Gold impact as NONE.

## 5. Manifest identity

`gold_manifest.json` records: schema version, item count, gold SHA256, UTC generation timestamp,
canonical corpus path `data/processed/structural_v2_1_full_4204/`, corpus snapshot id
`structural_v2_1_full_4204` with the manifest SHA256 and 4,204 disclosure count, production
freeze SHA `b8799e9299a5f795d2bec8ce0fad121b6c67c8db`, documentation freeze SHA `a7a8f99`, the
evaluator contract and implementation paths with their own SHA256s, all five distributions, the
lifecycle item list, the provenance policy, and the prediction-blind statement.

## 6. Evaluator readiness

`python -m pytest -q evaluation/independent_v2/test_evaluator.py` → **28 passed**.

No live agent payload was fed to the evaluator. Production code remains byte-identical to HEAD
(`git diff --quiet HEAD -- app/ tests/ db/ docs/ data/` clean), and the full suite stands at
1,881 passed — the 1,853 production baseline plus these 28 additive evaluation tests.

## 7. Immutability

`gold.jsonl` is now the immutable reference for the blind evaluation. It must not be edited in
response to anything observed in STEP 9. If a genuine gold defect is discovered during
evaluation, the correct procedure is to record it, re-freeze under a **new** hash, and report
both hashes — never to silently amend the artifact.

**STEP 9 has not been run.** No API started, no `/answer` call, no `ask_one_question.py`, no
`evaluate_postgres_agent_gold60.py`, and no evaluation of the frozen agent.
