# Evaluator Requirements for Independent Eval v2

**Status:** design only. No evaluator code was written or modified in STEP 6.
The frozen `AgentGold60Evaluator` remains untouched and continues to serve Gold60.

The frozen evaluator cannot faithfully score **28 of the 69** proposed items. This document
specifies the minimum semantics a v2 evaluator needs. It is a schema proposal, not an
implementation plan.

---

## A. Why the frozen evaluator is insufficient

`app/agent/gold60_evaluation.py:366` — `_end_to_end_failure_class()`:

```
if agent_status != "ok":                       return "agent_pipeline_error"
if not retrieval_hit:                          return "retrieval_miss"
if not answerable:                             return "answer_not_supported"     # <-- (1)
if not comparison.get("gold_doc_cited"):       return "gold_source_not_cited"    # <-- (2)
if not comparison.get("all_evidence_terms_present"): return "gold_evidence_terms_missing"  # <-- (3)
return "success"
```

1. **Refusal is unconditionally a failure.** `answerable == False` short-circuits before any
   gold comparison. All 12 refusal/clarify items would be scored as failures *while behaving
   correctly*, and — worse — a system that answers a first-report question with `0주` scores
   **better** than one that correctly refuses. This inverts the safety signal the set exists to
   measure.
2. **Gold citation is single-valued.** `gold_doc_cited` is one boolean over one `doc_id`. The 13
   `all_required` and 4 `any_of` items cannot express their citation contract.
3. **Answer matching is literal substring containment** over `evidence_terms`. A correctly
   computed `+100,000,000원 (+2.41%)` fails if rendered as `1억원 증가` or `100000000`.

---

## B. Required question-schema fields

The frozen runner requires `question_id, query, doc_id, target_type, target_id`. v2 needs:

| Field | Type | Purpose |
|---|---|---|
| `expected_behavior` | `answer` \| `clarify` \| `insufficient_evidence` | the primary scoring axis |
| `gold_doc_ids` | `list[str]` | replaces the single `doc_id` |
| `gold_chunk_ids` | `list[str]` | evidence-level gold |
| `gold_source_refs` | `list[{table_id,row_start,row_end}]` | row-level provenance |
| `citation_policy` | `single` \| `any_of` \| `all_required` \| `none` | how citations are judged |
| `required_gold_doc_ids` | `list[str]` | used when policy is `single` / `all_required` |
| `acceptable_gold_doc_ids` | `list[str]` | used when policy is `any_of` |
| `gold_numeric` | `{value, unit, secondary?}` | unit-aware numeric comparison |
| `source_values` / `derived_values` / `derivation` | objects + string | derived-answer scoring |
| `forbidden_fallbacks` | `list[str]` | the dangerous wrong answers for refusal items |
| `clarification_requirements` | `list[str]` | what a good clarification must ask for |

Backwards compatibility: keep `doc_id`/`target_id` populated from the first gold document so the
frozen runner can still ingest the file for retrieval-only metrics.

---

## C. Required scoring semantics

### C.1 Behaviour scoring (replaces the `answerable` short-circuit)

```
behavior_correct =
    expected == "answer"               -> produced an answer AND answerable is True
    expected == "insufficient_evidence"-> answerable is False  OR the answer explicitly
                                          states the value is unavailable
    expected == "clarify"              -> the response asks for the missing discriminator
```

For refusal items additionally compute **`forbidden_fallback_triggered`**: true when any string
in `forbidden_fallbacks` (or its numeric equivalent) appears as an asserted value. This is the
P0 signal — a fabricated number is categorically worse than an unhelpful refusal.

### C.2 Citation scoring

```
single        -> the one required doc is cited
all_required  -> every required doc is cited        (report per-doc recall, not just a boolean)
any_of        -> at least one acceptable doc is cited
none          -> no citation expected; citing anything is a leak
```

Report `citation_recall = |cited ∩ required| / |required|` so partial multi-doc citation is
visible rather than collapsing to false.

### C.3 Numeric scoring

Normalise before comparing: strip thousands separators, unify `주`/`shares`, `%`/`percent`,
`원`/`KRW`; compare `gold_numeric.value` with tolerance `0` for share counts and integers and
`1e-4` relative for percentages. Never substring-match a number.

### C.4 Derived-answer scoring

Score `derived_values` against the model's stated figure, and **separately** check whether the
model cited both `source_values` documents. A correct number obtained from a single document
(possible for the 정정 items, whose correction filing restates the prior amount) is a *different*
outcome from one obtained by genuine cross-document reasoning, and the report must distinguish
them.

### C.5 Three independent axes

Do not collapse to one boolean. Report, per item:

| Axis | Meaning |
|---|---|
| `answer_correctness` | the stated fact matches `gold_numeric` / `expected_answer` |
| `evidence_correctness` | the gold chunks/source_refs are in the served context |
| `citation_correctness` | the citation set satisfies `citation_policy` |
| `answerability_correctness` | behaviour matches `expected_behavior` |
| `presentation` | prose quality only — must never affect the four axes above |

The current `end_to_end_success` conflates all of these; the STEP 10 failure taxonomy depends on
them being separable.

### C.6 Prose independence

Refusal scoring must not require particular wording. Match on behaviour (`answerable`,
`answerability.status`, absence of an asserted value), never on a fixed Korean sentence.

---

## D. Distribution of the gap across the proposed set

| Gap | Items affected |
|---|---|
| Refusal scored as failure | 12 |
| Multi-document citation inexpressible | 17 (13 `all_required` + 4 `any_of`) |
| Derived numeric answer literal-matched | 10 |
| **Distinct items affected** | **28 of 69** |

The remaining 41 items can be scored by the frozen runner for retrieval and answer metrics,
though they still benefit from unit-aware numeric comparison.

---

## E. Implementation note

None of this requires touching `app/`. A v2 evaluator can wrap the existing
`AgentGold60Evaluator` — reusing its retrieval and plan capture verbatim — and replace only the
final scoring stage. That keeps the frozen Gold60 numbers byte-identical while giving v2 its own
semantics.
