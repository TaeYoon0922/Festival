"""Attribute a question's outcome to the stage that actually lost it.

A wrong answer says nothing about *where* the pipeline went wrong, and acting on
that ambiguity is how a filter phase gets reopened for a ranking problem.  This
classifies an already-completed run into one mutually exclusive category and
names the phase that owns it.

The classifier is a pure function over observations gathered *after* production
retrieval has finished.  Gold identifiers appear only in those observations, and
only as things to look for in what production already returned -- nothing here
can steer retrieval, and the module imports nothing from ``app``.

Two modes exist because the evidence has different prerequisites.  MODE A --
query understanding, filter eligibility, table structure -- needs only document
metadata and is embedding-independent.  MODE B -- rank and served evidence --
needs a retrieval run, and its conclusions are only as trustworthy as the
embedding provider that produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------- taxonomy

QUERY_UNDERSTANDING_DECLINE = "QUERY_UNDERSTANDING_DECLINE"
FILTER_EXCLUSION = "FILTER_EXCLUSION"
RANKING_LOW = "RANKING_LOW"
MULTI_DOCUMENT_INCOMPLETE = "MULTI_DOCUMENT_INCOMPLETE"
TABLE_SIBLING_MISS = "TABLE_SIBLING_MISS"
EVIDENCE_COVERAGE_MISS = "EVIDENCE_COVERAGE_MISS"
DOWNSTREAM_LOSS = "DOWNSTREAM_LOSS"
COMPLETE = "COMPLETE"

#: Which phase owns a category.  Metadata only -- the evaluator never acts on it.
OWNER = {
    QUERY_UNDERSTANDING_DECLINE: "P0-D",
    FILTER_EXCLUSION: "P1-B",
    RANKING_LOW: "P1-R",
    MULTI_DOCUMENT_INCOMPLETE: "P0-C / P0-B",
    TABLE_SIBLING_MISS: "P1-C",
    EVIDENCE_COVERAGE_MISS: "evidence coverage",
    DOWNSTREAM_LOSS: "resolver / composer / verbalizer",
    COMPLETE: "none",
}

#: Earliest-loss-first.  A question declined before retrieval has no rank, and a
#: document excluded before scoring has no rank either -- reading its absence
#: from the served list as a ranking problem is exactly the misattribution that
#: sends a ranking phase after a filter bug.  Later stages can only be blamed
#: once every earlier one is known to have succeeded.
PRECEDENCE = (
    QUERY_UNDERSTANDING_DECLINE,
    FILTER_EXCLUSION,
    RANKING_LOW,
    MULTI_DOCUMENT_INCOMPLETE,
    TABLE_SIBLING_MISS,
    EVIDENCE_COVERAGE_MISS,
    DOWNSTREAM_LOSS,
    COMPLETE,
)

# ------------------------------------------------------ embedding confidence

LIVE_BGE_M3 = "LIVE_BGE_M3"
HASH_DIAGNOSTIC = "HASH_DIAGNOSTIC"
LEXICAL_ONLY = "LEXICAL_ONLY"
OTHER_EMBEDDING = "OTHER"

#: Attached to any conclusion that depended on a vector channel the run cannot
#: vouch for.  Ranking measured on hash embeddings is a shape, not a result.
UNVERIFIED_RANKING = "ranking conclusion requires live embedding verification"

#: Categories whose evidence is embedding-independent: they read document
#: metadata and table structure, both of which are the same whoever embedded.
EMBEDDING_INDEPENDENT = frozenset({
    QUERY_UNDERSTANDING_DECLINE, FILTER_EXCLUSION, TABLE_SIBLING_MISS,
})


def confidence_from_provider(provider: str | None) -> str:
    value = str(provider or "").strip().casefold()
    if value in {"bge_m3", "bge-m3", "bgem3"}:
        return LIVE_BGE_M3
    if value == "hash":
        return HASH_DIAGNOSTIC
    if value in {"", "none", "lexical"}:
        return LEXICAL_ONLY
    return OTHER_EMBEDDING


# ------------------------------------------------------------ observation


@dataclass(frozen=True)
class Observation:
    """What a completed run showed, gathered after production finished.

    Every field is something production already produced or something the
    corpus already states.  ``required_docs`` and ``required_fields`` come from
    the evaluation set and are compared against that output -- they are never
    handed to retrieval.
    """

    question_id: str
    question: str = ""
    #: MODE A
    retrieval_allowed: bool = True
    understanding_state: str | None = None
    required_docs: tuple[str, ...] = ()
    eligible_docs: tuple[str, ...] = ()
    excluded_by: Mapping[str, Any] = field(default_factory=dict)
    #: MODE B -- absent when the run was structural only
    ranking_available: bool = False
    served_docs: tuple[str, ...] = ()
    first_required_rank: int | None = None
    graph_added_docs: tuple[str, ...] = ()
    #: evidence
    evidence_available: bool = False
    missing_fields: tuple[str, ...] = ()
    sibling_misses: tuple[Mapping[str, Any], ...] = ()
    coverage_misses: tuple[Mapping[str, Any], ...] = ()
    #: downstream
    downstream_available: bool = False
    downstream_lost_stage: str | None = None
    embedding_provider: str | None = None

    @property
    def mode(self) -> str:
        return "B" if self.ranking_available else "A"


@dataclass(frozen=True)
class Attribution:
    question_id: str
    primary: str
    owner: str
    mode: str
    embedding_confidence: str
    caveats: tuple[str, ...]
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "primary_attribution": self.primary,
            "owner_phase": self.owner,
            "mode": self.mode,
            "embedding_confidence": self.embedding_confidence,
            "caveats": list(self.caveats),
            "details": dict(self.details),
        }


# ------------------------------------------------------------- classifier


def classify(observation: Observation) -> Attribution:
    """Assign exactly one category, earliest loss first.

    Returns ``None`` for no category: a run that cannot see a later stage
    reports what it does know and says so through ``mode``, rather than
    guessing.
    """

    confidence = confidence_from_provider(observation.embedding_provider)
    caveats: list[str] = []
    details: dict[str, Any] = {
        "required_docs": list(observation.required_docs),
        "eligible_docs": list(observation.eligible_docs),
    }

    required = set(observation.required_docs)

    if not observation.retrieval_allowed:
        details["understanding_state"] = observation.understanding_state
        return _make(observation, QUERY_UNDERSTANDING_DECLINE, confidence,
                     caveats, details)

    ineligible = sorted(required - set(observation.eligible_docs))
    if required and ineligible:
        details["ineligible_docs"] = ineligible
        details["excluded_by"] = dict(observation.excluded_by)
        return _make(observation, FILTER_EXCLUSION, confidence, caveats, details)

    if not observation.ranking_available:
        # Structural pass: everything the run could observe succeeded, but the
        # served list was never produced.  Saying COMPLETE here would claim
        # more than was measured.
        details["note"] = "structural pass only; ranking and evidence unmeasured"
        return _make(observation, COMPLETE, confidence,
                     [*caveats, "MODE A: ranking and evidence not evaluated"],
                     details)

    served = set(observation.served_docs) | set(observation.graph_added_docs)
    details["first_required_rank"] = observation.first_required_rank
    details["served_required_docs"] = sorted(required & served)

    unserved = sorted(required - served)
    if required and unserved:
        if len(required) > 1 and (required & served):
            # Some required documents arrived and some did not: the gap is in
            # assembling the set, not in ranking one document.
            details["missing_required_docs"] = unserved
            return _make(observation, MULTI_DOCUMENT_INCOMPLETE, confidence,
                         caveats, details)
        details["missing_required_docs"] = unserved
        if confidence != LIVE_BGE_M3:
            caveats.append(UNVERIFIED_RANKING)
        return _make(observation, RANKING_LOW, confidence, caveats, details)

    if observation.evidence_available:
        if observation.sibling_misses:
            details["sibling_misses"] = [dict(m) for m in observation.sibling_misses]
            return _make(observation, TABLE_SIBLING_MISS, confidence, caveats,
                         details)
        if observation.coverage_misses or observation.missing_fields:
            details["coverage_misses"] = [dict(m) for m in observation.coverage_misses]
            details["missing_fields"] = list(observation.missing_fields)
            return _make(observation, EVIDENCE_COVERAGE_MISS, confidence,
                         caveats, details)

    if observation.downstream_available and observation.downstream_lost_stage:
        details["downstream_lost_stage"] = observation.downstream_lost_stage
        return _make(observation, DOWNSTREAM_LOSS, confidence, caveats, details)

    if not observation.evidence_available:
        caveats.append("evidence coverage not evaluated")
    if not observation.downstream_available:
        caveats.append("downstream retention not evaluated")
    return _make(observation, COMPLETE, confidence, caveats, details)


def _make(observation, primary, confidence, caveats, details):
    caveats = list(caveats)
    if (confidence != LIVE_BGE_M3
            and primary not in EMBEDDING_INDEPENDENT
            and observation.ranking_available
            and UNVERIFIED_RANKING not in caveats):
        caveats.append(UNVERIFIED_RANKING)
    return Attribution(
        question_id=observation.question_id,
        primary=primary,
        owner=OWNER[primary],
        mode=observation.mode,
        embedding_confidence=confidence,
        caveats=tuple(dict.fromkeys(caveats)),
        details=details,
    )


# ------------------------------------------------------------- aggregation


def aggregate(
    attributions: Sequence[Attribution],
    *,
    groups: Mapping[str, str] | None = None,
    tasks: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Counts by category, and by whatever dimensions the caller supplies."""

    from collections import Counter

    counts = Counter(a.primary for a in attributions)
    total = len(attributions)
    by_group: dict[str, Counter] = {}
    by_task: dict[str, Counter] = {}
    for a in attributions:
        if groups and a.question_id in groups:
            by_group.setdefault(groups[a.question_id], Counter())[a.primary] += 1
        if tasks and a.question_id in tasks:
            by_task.setdefault(tasks[a.question_id], Counter())[a.primary] += 1
    ranked = [a for a in attributions if a.mode == "B"]
    ranks = [a.details.get("first_required_rank") for a in ranked]
    found = [r for r in ranks if isinstance(r, int) and r > 0]
    retrieval = {
        "ranking_measured": len(ranked),
        "candidate_inclusion_rate": (
            sum(1 for a in attributions if a.primary != FILTER_EXCLUSION) / total
            if total else 0.0
        ),
        **{f"recall_at_{k}": (sum(1 for r in found if r <= k) / len(ranked)
                              if ranked else 0.0)
           for k in (1, 3, 5, 10)},
        "mrr": (sum(1.0 / r for r in found) / len(ranked)) if ranked else 0.0,
    }
    return {
        "metadata": dict(metadata or {}),
        "total": total,
        "retrieval": retrieval,
        "counts": {name: counts.get(name, 0) for name in PRECEDENCE},
        "rates": {
            name: (counts.get(name, 0) / total if total else 0.0)
            for name in PRECEDENCE
        },
        "owners": {name: OWNER[name] for name in PRECEDENCE},
        "by_doc_group": {g: dict(c) for g, c in sorted(by_group.items())},
        "by_task_type": {t: dict(c) for t, c in sorted(by_task.items())},
        "modes": dict(Counter(a.mode for a in attributions)),
        "embedding_confidence": dict(
            Counter(a.embedding_confidence for a in attributions)
        ),
    }


def markdown_table(attributions: Sequence[Attribution]) -> str:
    lines = ["| question | attribution | owner | mode | confidence |",
             "|---|---|---|---|---|"]
    for a in sorted(attributions, key=lambda x: (x.primary, x.question_id)):
        lines.append(
            f"| {a.question_id} | {a.primary} | {a.owner} | {a.mode} | "
            f"{a.embedding_confidence} |"
        )
    return "\n".join(lines)
