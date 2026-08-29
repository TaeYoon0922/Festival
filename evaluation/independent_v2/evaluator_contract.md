# Independent Eval v2 — Evaluator Contract

**Frozen before implementation.** This document defines scoring semantics. The implementation in
`evaluation/independent_v2/evaluator.py` follows this contract; where they disagree, this
document is authoritative.

**Scope:** evaluation-only. `AgentGold60Evaluator` and every module under `app/` are untouched.
Gold60 continues to be scored by the frozen evaluator with byte-identical results.

---

## 1. Four independent scoring axes

Each item produces five independent booleans. **None of them gates another.**

| Axis | Meaning |
|---|---|
| `answer_correct` | the stated factual content matches gold (numeric-normalised, role-aware) |
| `evidence_correct` | evidence supporting the requested fact was actually served |
| `citation_correct` | the cited document set satisfies `citation_policy` |
| `answerability_correct` | observed behaviour matches `expected_behavior` |
| `presentation_issue` | prose/verbosity problem **only**; never affects the four axes above |

`overall_pass = answer_correct AND evidence_correct AND citation_correct AND answerability_correct`.
`presentation_issue` is reported separately and never turns a pass into a failure.

For refusal items `answer_correct` means "no unsupported factual value was asserted", not
"a string matched".

---

## 2. `expected_behavior` semantics

### 2.1 `answer`
Correct when `answerable == true` **and** the requested factual content is correct.
A refusal here is `answerability_correct = false` (under-answering — an `A1` failure).

### 2.2 `insufficient_evidence`
Correct when **both** hold:
- `answerable == false`, *or* the answer explicitly states the value cannot be determined from
  the disclosures; **and**
- no `forbidden_fallbacks` value is asserted for the requested field.

A correct refusal here is a **PASS on all four axes**. It must never be recorded as
`answer_not_supported`. This is the single most important divergence from the frozen evaluator,
which returns `"answer_not_supported"` whenever `answerable` is false — inverting the safety
signal so that answering `0주` to a first-report question outscores a correct refusal.

### 2.3 `clarify`
Correct when **all** hold:
- no single factual candidate is asserted as the answer;
- the response signals ambiguity / the need to disambiguate;
- at least one `clarification_requirements` dimension is expressed.

`clarify` and `insufficient_evidence` are **not** collapsed into `answerable == false`. An item
whose gold is `clarify` but which is answered with a flat refusal carrying no ambiguity signal
scores `answerability_correct = false` with sub-reason `refused_without_disambiguation`.

---

## 3. Dangerous false positives — severity policy

The highest-severity failure direction is:

> gold is `insufficient_evidence` / `clarify`, and the agent asserts a factual value.

Concrete instances this set is built to catch:
- a first-report `직전 보유주식수` answered as `0주`;
- a standalone `이번 보고` answered from the latest filing;
- a withheld `계약금액` answered with the adjacent `최근매출액`.

Scoring for these:
```
answerability_correct = false
answer_correct        = false
severity              = "P0_false_positive"
forbidden_fallback_triggered = <matched fallback>
```
Detection is **value-based**, not string-based: each `forbidden_fallbacks` entry is compiled to
(a) a normalised numeric value where one is derivable from `source_values`, and (b) a set of
descriptive markers. A number is "asserted" only when it appears as a value for the requested
field, not merely as recited context.

---

## 4. Answer scoring

### 4.1 Numeric normalisation
Units: `shares`, `percent`, `KRW`, `count`.
Normalisation strips thousands separators and unit suffixes (`주`, `%`, `원`, `회`), accepts
Korean scale words (`억`, `만`) by expansion, and compares numerically. **A number is never
substring-matched.**

`percent` and percentage-point are distinct: `pct_change` and a difference of two ratios are
different quantities and are stored under different keys; the evaluator never treats one as the
other.

### 4.2 Tolerance
| Kind | Tolerance |
|---|---|
| share counts, KRW, counts (exact extraction) | **exact** (0) |
| ratios read directly from a filing | **exact** on the printed precision |
| derived percentages (`pct_change`) | **absolute ≤ 0.01 percentage point** |

No other tolerance exists. Tolerance is never widened per item.

### 4.3 Multiple values with semantic roles
Where gold carries more than one value — 정정 전/정정 후, 변동 전/변동 후 — each value must match
**its own role**. `gold_numeric.value` is the primary role and `gold_numeric.secondary.role`
names the other (`superseded`, `before`). Producing the right *set* of numbers with the roles
swapped is `answer_correct = false`, sub-reason `role_swap`.

### 4.4 Derived answers
Scored against `derived_values` under the stated `derivation`. `answer_correct` depends only on
the derived figure. Whether the agent cited both `source_values` documents is
`citation_correct` — never folded into `answer_correct`. The combination
`answer_correct = true, citation_correct = false` is explicitly representable and expected for
items where a single correction filing restates the prior amount.

### 4.5 Ordering (3+ company ranking)
Both must hold: every company→value mapping is correct, **and** the emitted ordering equals
`derived_values.descending_order`. Merely naming all companies is not a pass.

---

## 5. Evidence scoring

`evidence_correct` asks whether evidence that actually supports the requested fact was served —
checked against `gold_doc_ids`, then `gold_chunk_ids`, then `gold_source_refs`.

A single exact chunk id is required **only** when the fact is unique to that chunk. Where an
equivalent chunk in an acceptable document proves the same fact (the `any_of` case, where two
filings carry identical body values), any of them satisfies the axis. Evidence is judged against
served context, so a right answer reached with unrelated rank-1 evidence is detectable as
`answer_correct = true, evidence_correct = false`.

For `expected_behavior != "answer"` items with empty `gold_doc_ids` (the deictic family),
`evidence_correct` is true when the served context is empty — serving ranked evidence for a
question that names no report is itself the leak this family detects.

---

## 6. Citation policy

| Policy | Pass condition |
|---|---|
| `single` | the one required document is cited |
| `any_of` | **at least one** of `acceptable_gold_doc_ids` is cited; requiring both is a bug |
| `all_required` | **every** document in `required_gold_doc_ids` is cited |
| `none` | nothing is cited (used by the deictic family) |

`citation_recall = |cited ∩ required| / |required|` is always reported, so partial multi-document
citation is visible rather than collapsing to a boolean.

---

## 7. Clarification scoring

No fixed sentence is required. Pass when the response does not assert one candidate, signals
ambiguity, and expresses at least one required dimension. Literal phrase matching is prohibited;
ambiguity is detected from behaviour (`answerable`, absence of an asserted value for the
requested field) plus dimension markers drawn from `clarification_requirements`.

---

## 8. Presentation-only issues

An item with correct facts, evidence, citations and answerability, but verbose or awkward prose,
is `presentation_issue = true` and **passes**.

The Phase 3 ACQ pattern is handled explicitly: when the requested field is correctly reported as
unresolved while surrounding holding facts are recited at length, this is presentation-only —
**provided** the requested field itself is not asserted. If the recitation supplies a value for
the requested field, it is a §3 false positive instead.

---

## 9. Failure taxonomy mapping

Each failing item records `first_failing_stage` from the frozen taxonomy:
`Q1, Q2, R1, R2, R3, S1, M1, E1, F1, A1, C1, P1, ENV, UNKNOWN`.

Mapping is deliberately conservative:

| Observation | Stage |
|---|---|
| gold document absent from candidates | `R1` |
| gold document present but poorly ranked | `R2` |
| wrong filing selected among correct candidates | `S1` |
| multi-doc item served only one document | `M1` |
| answered when gold is refusal, or refused when gold is answerable | `A1` |
| answer and evidence correct, citation set wrong | `C1` |
| all four axes correct, prose problem | `P1` |
| retrieval/vector/runtime error in the trace | `ENV` |

**When attribution is not determined by the recorded trace, the stage is `UNKNOWN`.** The
evaluator never guesses a root cause to avoid an `UNKNOWN`.

---

## 10. Backward compatibility

`AgentGold60Evaluator` is not subclassed, wrapped, or edited. `IndependentV2Evaluator` is a
standalone scorer that consumes an already-produced response payload plus a gold record. It
shares no mutable state with the Gold60 path and imports nothing from it, so Gold60 results
cannot change. Gold60 keeps its own runner and its own report format.

---

## 11. Inputs

The scorer consumes the public `/answer` payload — `answer`, `retrieved_context[]`,
`think_trace` (`answerable`, `stages`, `selected_evidence_count`, `warnings`) — and a gold record
from `gold.jsonl`. It requires no field the current response schema does not already expose, and
it never calls the agent itself.
