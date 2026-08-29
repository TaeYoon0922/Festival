# -*- coding: utf-8 -*-
"""Independent Eval v2 scorer.

Evaluation-only. Imports nothing from ``app`` and shares no state with
``AgentGold60Evaluator``; Gold60 keeps its own runner and its own numbers.

The scorer consumes an already-produced ``/answer`` payload plus one gold record and
returns five independent axes. It never calls the agent.

Semantics are fixed by ``evaluator_contract.md``; that document is authoritative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


DERIVED_PCT_TOLERANCE = 0.01  # absolute percentage points, contract section 4.2

_UNIT_SUFFIX = {
    "shares": ("주", "株"),
    "percent": ("%", "퍼센트"),
    "KRW": ("원",),
    "count": ("회", "건", "개"),
}
_SCALE = (("조", 10 ** 12), ("억", 10 ** 8), ("만", 10 ** 4))

STAGES = ("Q1", "Q2", "R1", "R2", "R3", "S1", "M1", "E1", "F1", "A1", "C1", "P1", "ENV", "UNKNOWN")


# --------------------------------------------------------------------- numbers
def normalize_numbers(text: str) -> set[float]:
    """Every number stated in ``text``, normalised to a plain float.

    Handles thousands separators and Korean scale words so that ``1억 20만``,
    ``120,000,000`` and ``120000000`` all reduce to the same value.
    """

    if not text:
        return set()
    found: set[float] = set()
    for match in re.finditer(r"[-+]?[\d,]+(?:\.\d+)?", text):
        raw = match.group(0).replace(",", "")
        if raw in ("", "-", "+", "."):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        found.add(value)
        tail = text[match.end():match.end() + 1]
        for word, scale in _SCALE:
            if tail == word:
                found.add(value * scale)
    # "1억 2,000만" style compounds: accumulate consecutive descending scale groups
    pattern = re.compile(r"([\d,]+)\s*(조|억|만)")
    run_total, prev_scale, run_end = 0.0, None, None
    for match in pattern.finditer(text):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        scale = dict(_SCALE)[match.group(2)]
        found.add(value * scale)
        contiguous = run_end is not None and text[run_end:match.start()].strip() == ""
        if prev_scale is not None and contiguous and scale < prev_scale:
            run_total += value * scale
            found.add(run_total)
        else:
            run_total = value * scale
        prev_scale, run_end = scale, match.end()
    return found


def numeric_match(gold: Mapping[str, Any], text: str, *, derived: bool = False) -> bool:
    """True when ``text`` states the gold value under the gold's unit."""

    if not gold:
        return False
    target = float(gold["value"])
    stated = normalize_numbers(text)
    if derived and gold.get("unit") == "percent":
        return any(abs(v - target) <= DERIVED_PCT_TOLERANCE for v in stated)
    return any(abs(v - target) < 1e-9 for v in stated)


def _role_window(text: str, markers: Sequence[str], width: int = 60) -> str:
    """Text near a role marker, used to bind a value to its semantic role."""

    spans = []
    for marker in markers:
        for m in re.finditer(re.escape(marker), text):
            spans.append(text[m.start(): m.end() + width])
    return " ".join(spans)


_ROLE_MARKERS = {
    "superseded": ("정정 전", "정정전", "기존", "당초"),
    "before": ("변동 전", "변동전", "직전"),
    "primary_correction": ("정정 후", "정정후", "최종", "확정"),
    "primary_change": ("변동 후", "변동후", "이후"),
}


# ------------------------------------------------------------------- result
@dataclass
class ItemResult:
    question_id: str
    category: str
    expected_behavior: str
    answer_correct: bool = False
    evidence_correct: bool = False
    citation_correct: bool = False
    answerability_correct: bool = False
    presentation_issue: bool = False
    severity: str | None = None
    forbidden_fallback_triggered: str | None = None
    citation_recall: float | None = None
    first_failing_stage: str = "UNKNOWN"
    sub_reasons: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def overall_pass(self) -> bool:
        return (self.answer_correct and self.evidence_correct
                and self.citation_correct and self.answerability_correct)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id, "category": self.category,
            "expected_behavior": self.expected_behavior,
            "answer_correct": self.answer_correct,
            "evidence_correct": self.evidence_correct,
            "citation_correct": self.citation_correct,
            "answerability_correct": self.answerability_correct,
            "presentation_issue": self.presentation_issue,
            "overall_pass": self.overall_pass,
            "severity": self.severity,
            "forbidden_fallback_triggered": self.forbidden_fallback_triggered,
            "citation_recall": self.citation_recall,
            "first_failing_stage": self.first_failing_stage,
            "sub_reasons": list(self.sub_reasons),
            "detail": dict(self.detail),
        }


class IndependentV2Evaluator:
    """Score one ``/answer`` payload against one gold record."""

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _served_docs(payload: Mapping[str, Any]) -> set[str]:
        """Documents placed in the served context (the evidence axis)."""
        return {str(row.get("doc_id")) for row in (payload.get("retrieved_context") or [])
                if row.get("doc_id")}

    @staticmethod
    def _cited_docs(payload: Mapping[str, Any]) -> set[str]:
        """Documents the answer actually references (the citation axis).

        The generator emits ``[n]`` markers keyed to the served context ranks. When no
        marker is present the answer cites nothing, which is itself a citation failure
        for any policy other than ``none``.
        """
        rows = payload.get("retrieved_context") or []
        by_rank = {int(r.get("rank") or i): str(r.get("doc_id"))
                   for i, r in enumerate(rows, 1) if r.get("doc_id")}
        marks = {int(m) for m in re.findall(r"\[(\d{1,2})\]", payload.get("answer") or "")}
        if not marks:
            return set()
        return {by_rank[m] for m in marks if m in by_rank}

    @staticmethod
    def _cited_chunks(payload: Mapping[str, Any]) -> set[str]:
        return {str(row.get("chunk_id")) for row in (payload.get("retrieved_context") or [])
                if row.get("chunk_id")}

    @staticmethod
    def _answerable(payload: Mapping[str, Any]) -> bool:
        return bool((payload.get("think_trace") or {}).get("answerable"))

    @staticmethod
    def _states_unavailable(text: str) -> bool:
        markers = ("확인하기 어렵", "확인할 수 없", "확인되지 않", "알 수 없", "제공된 공시",
                   "근거가 없", "기재되어 있지 않", "공시되지 않", "유보", "미기재",
                   "찾을 수 없", "판단하기 어렵")
        return any(m in (text or "") for m in markers)

    @staticmethod
    def _signals_ambiguity(text: str) -> bool:
        markers = ("어느", "특정해", "구체적으로", "두 건", "복수", "명확히",
                   "어떤 계약", "지정해", "알려주시", "말씀해", "중 어느")
        return any(m in (text or "") for m in markers)

    # --------------------------------------------------------- axis: answer
    def _score_answer(self, gold: Mapping[str, Any], payload: Mapping[str, Any],
                      result: ItemResult) -> None:
        text = payload.get("answer") or ""
        gn = gold.get("gold_numeric")
        derived = bool(gold.get("derived_values"))

        if gold["expected_behavior"] != "answer":
            fired = self._forbidden_fired(gold, text)
            result.forbidden_fallback_triggered = fired
            result.answer_correct = fired is None
            if fired is not None:
                result.sub_reasons.append("asserted_forbidden_value")
            return

        if gn:
            ok = numeric_match(gn, text, derived=derived)
            secondary = gn.get("secondary")
            if ok and secondary:
                # role binding: each value must appear near its own role marker
                role = secondary.get("role")
                if role in ("superseded", "before"):
                    prim_markers = (_ROLE_MARKERS["primary_correction"] if role == "superseded"
                                    else _ROLE_MARKERS["primary_change"])
                    sec_window = _role_window(text, _ROLE_MARKERS[role])
                    prim_window = _role_window(text, prim_markers)
                    sec_ok = numeric_match(secondary, sec_window)
                    prim_ok = numeric_match(gn, prim_window)
                    if not (sec_ok and prim_ok):
                        swapped = (numeric_match(gn, sec_window)
                                   or numeric_match(secondary, prim_window))
                        ok = False
                        result.sub_reasons.append("role_swap" if swapped else "role_unbound")
                else:
                    ok = ok and numeric_match(secondary, text, derived=derived)
            result.answer_correct = ok
        elif gold.get("derived_values", {}).get("descending_order"):
            order = gold["derived_values"]["descending_order"]
            positions = [text.find(name) for name in order]
            result.answer_correct = all(p >= 0 for p in positions) and positions == sorted(positions)
            if not result.answer_correct:
                result.sub_reasons.append("ordering_wrong" if all(p >= 0 for p in positions)
                                          else "ordering_incomplete")
        else:
            expected = str(gold.get("expected_answer") or "")
            tokens = [t for t in re.split(r"[\s/·,()]+", expected) if len(t) >= 2]
            result.answer_correct = bool(tokens) and all(t in text for t in tokens)
            if not result.answer_correct:
                result.sub_reasons.append("text_answer_mismatch")

    def _forbidden_fired(self, gold: Mapping[str, Any], text: str) -> str | None:
        """A forbidden fallback counts only when its value is asserted."""

        stated = normalize_numbers(text)
        for fallback in gold.get("forbidden_fallbacks") or []:
            for value in normalize_numbers(fallback):
                if value in stated:
                    return fallback
            if "0" in re.findall(r"\b0\b", fallback) and 0.0 in stated:
                return fallback
        # a first-report item answered with any share/ratio figure for the requested field
        if gold.get("category") == "T15_first_report_no_previous":
            sv = gold.get("source_values") or {}
            for key in ("보유주식수", "보유비율", "증감주식수"):
                for value in normalize_numbers(str(sv.get(key) or "")):
                    if value in stated and not self._states_unavailable(text):
                        return f"restated {key} as the previous value"
        return None

    # ------------------------------------------------------- axis: evidence
    def _score_evidence(self, gold: Mapping[str, Any], payload: Mapping[str, Any],
                        result: ItemResult) -> None:
        docs, chunks = self._served_docs(payload), self._cited_chunks(payload)
        gold_docs = set(gold.get("gold_doc_ids") or [])
        gold_chunks = set(gold.get("gold_chunk_ids") or [])

        if not gold_docs:
            # deictic family: serving any ranked evidence is itself the leak
            result.evidence_correct = not docs
            if docs:
                result.sub_reasons.append("evidence_served_for_unnamed_report")
            return
        if gold.get("citation_policy") == "any_of":
            result.evidence_correct = bool(docs & gold_docs)
        elif gold.get("citation_policy") == "all_required":
            result.evidence_correct = gold_docs.issubset(docs)
        else:
            result.evidence_correct = bool(docs & gold_docs)
        if gold_chunks and result.evidence_correct and not (chunks & gold_chunks):
            result.detail["gold_chunk_not_served"] = True
        if not result.evidence_correct:
            result.sub_reasons.append("gold_document_not_served")

    # ------------------------------------------------------- axis: citation
    def _score_citation(self, gold: Mapping[str, Any], payload: Mapping[str, Any],
                        result: ItemResult) -> None:
        cited = self._cited_docs(payload)
        policy = gold.get("citation_policy") or "single"
        required = list(gold.get("required_gold_doc_ids") or [])
        acceptable = list(gold.get("acceptable_gold_doc_ids") or [])

        if policy == "none":
            result.citation_correct = not cited
            result.citation_recall = None
            if cited:
                result.sub_reasons.append("cited_for_unnamed_report")
            return
        if policy == "any_of":
            hit = cited & set(acceptable)
            result.citation_correct = bool(hit)
            result.citation_recall = 1.0 if hit else 0.0
        else:
            need = set(required)
            hit = cited & need
            result.citation_recall = (len(hit) / len(need)) if need else None
            result.citation_correct = bool(need) and need.issubset(cited)
        if not result.citation_correct:
            result.sub_reasons.append("citation_incomplete" if result.citation_recall
                                      else "citation_missing")

    # --------------------------------------------------- axis: answerability
    def _score_answerability(self, gold: Mapping[str, Any], payload: Mapping[str, Any],
                             result: ItemResult) -> None:
        expected = gold["expected_behavior"]
        text = payload.get("answer") or ""
        answerable = self._answerable(payload)

        if expected == "answer":
            result.answerability_correct = answerable
            if not answerable:
                result.sub_reasons.append("refused_answerable_question")
                result.severity = "A1_over_refusal"
            return

        asserted = result.forbidden_fallback_triggered is not None
        if expected == "insufficient_evidence":
            result.answerability_correct = (
                (not answerable or self._states_unavailable(text)) and not asserted)
            if asserted:
                result.severity = "P0_false_positive"
                result.sub_reasons.append("answered_unsupported_question")
            elif answerable and not self._states_unavailable(text):
                result.sub_reasons.append("claimed_answerable_without_support")
            return

        # clarify
        no_single_candidate = not asserted
        signals = self._signals_ambiguity(text)
        dims = gold.get("clarification_requirements") or []
        dim_hit = any(any(part in text for part in re.split(r"[\s/]+", d) if len(part) >= 2)
                      for d in dims)
        result.answerability_correct = no_single_candidate and signals and (dim_hit or not dims)
        if asserted:
            result.severity = "P0_false_positive"
            result.sub_reasons.append("chose_one_candidate_arbitrarily")
        elif not signals:
            result.sub_reasons.append("refused_without_disambiguation")

    # ----------------------------------------------------------- attribution
    def _attribute(self, gold: Mapping[str, Any], payload: Mapping[str, Any],
                   result: ItemResult) -> None:
        trace = payload.get("think_trace") or {}
        warnings = [str(w) for w in (trace.get("warnings") or [])]
        if any("vector" in w.lower() or "error" in w.lower() for w in warnings):
            result.first_failing_stage = "ENV"
            return
        if result.overall_pass:
            result.first_failing_stage = "P1" if result.presentation_issue else "UNKNOWN"
            return
        if not result.answerability_correct:
            result.first_failing_stage = "A1"
            return
        if not result.evidence_correct:
            gold_docs = set(gold.get("gold_doc_ids") or [])
            served = self._served_docs(payload)
            if gold_docs and len(gold_docs) > 1 and served & gold_docs:
                result.first_failing_stage = "M1"
            elif not served:
                result.first_failing_stage = "R1"
            else:
                result.first_failing_stage = "S1" if gold_docs & served else "R1"
            return
        if not result.citation_correct:
            result.first_failing_stage = "C1"
            return
        if not result.answer_correct:
            result.first_failing_stage = "F1" if "role_swap" in result.sub_reasons else "UNKNOWN"
            return
        result.first_failing_stage = "UNKNOWN"

    # ------------------------------------------------------------- entrypoint
    def score(self, gold: Mapping[str, Any], payload: Mapping[str, Any]) -> ItemResult:
        result = ItemResult(question_id=gold.get("id") or gold.get("question_id") or "",
                            category=gold.get("category", ""),
                            expected_behavior=gold["expected_behavior"])
        self._score_answer(gold, payload, result)
        self._score_evidence(gold, payload, result)
        self._score_citation(gold, payload, result)
        self._score_answerability(gold, payload, result)

        text = payload.get("answer") or ""
        if (result.answer_correct and result.evidence_correct and result.citation_correct
                and result.answerability_correct and len(text) > 400):
            result.presentation_issue = True
        self._attribute(gold, payload, result)
        return result

    def score_all(self, golds: Iterable[Mapping[str, Any]],
                  payloads: Mapping[str, Mapping[str, Any]]) -> list[ItemResult]:
        out = []
        for gold in golds:
            key = gold.get("id") or gold.get("question_id")
            payload = payloads.get(key)
            if payload is None:
                continue
            out.append(self.score(gold, payload))
        return out


__all__ = ["IndependentV2Evaluator", "ItemResult", "normalize_numbers", "numeric_match",
           "DERIVED_PCT_TOLERANCE", "STAGES"]
