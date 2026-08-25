"""Apply a plan's correction policy through the correction graph.

``QueryPlan.correction_policy`` keeps its four existing values and its existing
meaning.  What changes is the evidence behind them: instead of reading
``is_correction`` alone, a policy now consults the resolved correction group a
document belongs to.

``original_only``
    Keep the document each chain starts from.

``corrected_only``
    Keep correcting documents only.

``latest_preferred``
    Prefer the final valid document of a resolved group.  Superseded documents
    are ranked below it but are never removed, so a question that only a
    superseded filing answers still finds it.

``any``
    Unchanged.

A document that belongs to no correction group, and every document in a group
that stayed ambiguous or unresolved, falls back to the pre-graph behaviour.
That keeps ordinary disclosures and corrections whose original is unknown out
of the way of resolved groups.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Protocol, Sequence

from app.reasoning.correction_graph import (
    CorrectionDocumentState,
    CorrectionGraphUnavailable,
)


logger = logging.getLogger(__name__)

#: Reasons already reported once, so a database without the correction tables
#: warns on the first query instead of on every query.
_REPORTED_UNAVAILABLE: set[str] = set()


POLICY_ANY = "any"
POLICY_CORRECTED_ONLY = "corrected_only"
POLICY_ORIGINAL_ONLY = "original_only"
POLICY_LATEST_PREFERRED = "latest_preferred"


class CorrectionGraphView(Protocol):
    """What retrieval needs from a correction graph.

    ``document_states`` drives policy selection.  ``get_correction_chain`` and
    ``get_latest_report`` are used to report a chain and are looked up
    defensively, so a narrower view still works.
    """

    def document_states(
        self, doc_ids: Iterable[str]
    ) -> Mapping[str, CorrectionDocumentState]: ...


def _report_unavailable(operation: str, error: Exception) -> None:
    """Warn once per reason, then stay quiet, so the cause is never invisible."""

    reason = f"{operation}:{type(error).__name__}"
    if reason in _REPORTED_UNAVAILABLE:
        logger.debug("correction graph unavailable for %s: %s", operation, error)
        return
    _REPORTED_UNAVAILABLE.add(reason)
    logger.warning(
        "correction graph unavailable for %s (%s); falling back to is_correction "
        "only. Apply db/006_correction_graph.sql and run the backfill to enable it.",
        operation,
        error,
    )


def document_states(
    graph: CorrectionGraphView | None, doc_ids: Iterable[str]
) -> dict[str, CorrectionDocumentState]:
    """Look up correction state, tolerating a graph that is not available yet.

    Retrieval must keep working when the correction tables have not been built,
    so only :class:`CorrectionGraphUnavailable` degrades to the pre-graph
    behaviour, and it is logged when it does.  Every other error -- a SQL
    mistake, a schema mismatch, a bug in the graph -- propagates, because
    silently answering as if corrections did not exist would hide it.
    """

    if graph is None:
        return {}
    unique = [str(doc_id) for doc_id in doc_ids if str(doc_id)]
    if not unique:
        return {}
    try:
        states = graph.document_states(unique)
    except CorrectionGraphUnavailable as error:
        _report_unavailable("document_states", error)
        return {}
    return {str(key): value for key, value in states.items()}


def document_allowed(
    policy: str,
    *,
    is_correction: bool,
    state: CorrectionDocumentState | None,
) -> bool:
    """Decide whether one document survives a hard correction policy."""

    if policy == POLICY_ORIGINAL_ONLY:
        if state is not None and state.is_resolved:
            return state.is_resolved_root
        return not is_correction
    if policy == POLICY_CORRECTED_ONLY:
        if state is not None and state.is_resolved:
            return not state.is_resolved_root
        return bool(is_correction)
    return True


def prefers_document(
    policy: str,
    *,
    is_correction: bool,
    state: CorrectionDocumentState | None,
) -> bool:
    """Whether ``latest_preferred`` should boost this document.

    Without a resolved group the answer is the pre-graph one: a correcting
    document is preferred over the original it replaces.  With a resolved group
    the final valid document is preferred and its superseded predecessors are
    not, even though they are also corrections.
    """

    if policy != POLICY_LATEST_PREFERRED:
        return bool(is_correction)
    if state is not None and state.is_resolved:
        # is_resolved_latest, not is_latest: a lone ambiguous or unresolved
        # correction also carries is_latest and must not be promoted by it.
        return state.is_resolved_latest
    return bool(is_correction)


def correction_timeline(
    chain: Sequence[Any],
) -> list[dict[str, Any]]:
    """Render a correction chain as the ordered list a comparison reads.

    Element ``0`` is the document as first filed and the last element is the
    final valid version, so a before/after comparison walks the list directly.
    """

    timeline: list[dict[str, Any]] = []
    for member in chain:
        timeline.append(
            {
                "doc_id": member.doc_id,
                "correction_order": member.correction_order,
                "parent_doc_id": member.parent_doc_id,
                "is_latest": member.is_latest,
                "is_correction": member.is_correction,
                "role": "original" if member.correction_order == 0 else "correction",
                "resolution_status": member.resolution_status,
                "resolution_source": member.resolution_source,
            }
        )
    return timeline


def _chain(graph: Any, doc_id: str) -> Sequence[Any]:
    reader = getattr(graph, "get_correction_chain", None)
    if not callable(reader):
        return ()
    try:
        return tuple(reader(doc_id))
    except CorrectionGraphUnavailable as error:
        _report_unavailable("get_correction_chain", error)
        return ()


def correction_summary(
    graph: CorrectionGraphView | None,
    doc_ids: Iterable[str],
    *,
    policy: str = POLICY_ANY,
) -> dict[str, Any]:
    """Summarize the correction groups a candidate set touches.

    The result carries one time-ordered timeline per group, which is what a
    before/after comparison and a "show me this report's correction history"
    question both read.  It is empty when no graph is wired or when nothing in
    the candidate set belongs to a correction group, so callers that do not use
    corrections see no change.
    """

    states = document_states(graph, doc_ids)
    if not states:
        return {}
    groups: dict[str, dict[str, Any]] = {}
    superseded: list[str] = []
    for doc_id, state in sorted(states.items()):
        if state.is_superseded:
            superseded.append(doc_id)
        if state.correction_group_id in groups:
            continue
        chain = _chain(graph, doc_id)
        timeline = correction_timeline(chain)
        resolved = state.is_resolved and len(timeline) > 1
        latest = next(
            (item["doc_id"] for item in timeline if item["is_latest"]),
            None,
        )
        groups[state.correction_group_id] = {
            "root_doc_id": state.root_doc_id,
            # Only an established chain has a verified final document. A lone
            # ambiguous or unresolved correction reports None so a consumer
            # cannot mistake it for one.
            "latest_doc_id": latest if resolved else None,
            "is_resolved": resolved,
            "resolution_status": state.resolution_status,
            "timeline": timeline,
        }
    return {
        "policy": policy,
        "group_count": len(groups),
        "superseded_doc_ids": superseded,
        "groups": groups,
    }
