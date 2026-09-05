"""Deterministic state machine for bounded clarification requests."""

from __future__ import annotations

from typing import Any

from app.reasoning.clarification_request import (
    ClarificationDecision,
    ClarificationRequest,
    ClarificationState,
)


class ClarificationResolver:
    def __init__(self, classifier: Any | None = None) -> None:
        self.classifier = classifier

    def resolve(self, request: ClarificationRequest) -> ClarificationDecision:
        candidates = request.candidates
        if not candidates:
            return ClarificationDecision(
                state=request.fallback_state,
                reason="no_bounded_candidates",
                truncated=request.truncated,
            )
        if len(candidates) == 1:
            candidate = candidates[0]
            if request.single_candidate_safe:
                return ClarificationDecision(
                    state=ClarificationState.RESOLVED,
                    reason="single_safe_candidate",
                    candidates=candidates,
                    selected_candidate_id=candidate.id,
                    truncated=request.truncated,
                )
            return ClarificationDecision(
                state=request.fallback_state,
                reason="single_candidate_not_declared_safe",
                candidates=candidates,
                truncated=request.truncated,
            )

        if self.classifier is None:
            return _clarify(
                request,
                reason="multiple_bounded_candidates",
                classifier_status="disabled",
            )
        outcome = self.classifier.classify(request.question, candidates)
        status = str(getattr(outcome, "status", "classifier_error"))
        if not bool(getattr(outcome, "succeeded", False)):
            return _clarify(
                request,
                reason="classifier_safe_fallback",
                classifier_status=status,
            )

        result = outcome.result
        decision = str(getattr(result, "decision", ""))
        candidate_ids = tuple(getattr(result, "candidate_ids", ()) or ())
        if decision == "resolved":
            selected = candidate_ids[0]
            if request.classifier_resolution_safe:
                return ClarificationDecision(
                    state=ClarificationState.RESOLVED,
                    reason="classifier_selected_candidate",
                    candidates=candidates,
                    selected_candidate_id=selected,
                    classifier_status=status,
                    truncated=request.truncated,
                )
            return _clarify(
                request,
                reason="classifier_resolution_not_declared_safe",
                classifier_status=status,
            )
        selected_candidates = tuple(
            candidate for candidate in candidates if candidate.id in set(candidate_ids)
        )
        if request.preserve_candidates:
            # The classifier decided *whether* to ask; what exists is not its
            # call.  Dropping a filing here would offer a choice the corpus
            # does not actually limit the asker to.
            selected_candidates = candidates
        return ClarificationDecision(
            state=ClarificationState.CLARIFY,
            reason="classifier_requested_clarification",
            candidates=selected_candidates or candidates,
            classifier_status=status,
            truncated=request.truncated,
            preserve_candidates=request.preserve_candidates,
        )


def _clarify(
    request: ClarificationRequest,
    *,
    reason: str,
    classifier_status: str,
) -> ClarificationDecision:
    return ClarificationDecision(
        state=ClarificationState.CLARIFY,
        reason=reason,
        candidates=request.candidates,
        classifier_status=classifier_status,
        truncated=request.truncated,
        preserve_candidates=request.preserve_candidates,
    )


__all__ = ["ClarificationResolver"]
