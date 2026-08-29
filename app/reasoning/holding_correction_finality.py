"""Which holding filing in a correction chain is the final one, read from P0-A.

A correction restates an earlier filing, and it arrives later than the filing it
restates -- sometimes much later, and sometimes after other, unrelated reports
have already been filed.  Comparing report dates before the chain is collapsed
therefore ranks a restatement as though it were its own event, and answering
from it states a superseded holding as the current one.

Nothing here decides *which* filing corrects which.  That is settled by the
frozen P0-A correction graph, whose rules, confidences and ambiguity behaviour
this module does not reimplement, reweigh or second-guess.  It reads a
precomputed materialization of P0-A's own output and does one thing with it:
given the reports of one issuer/holder pair, remove the members P0-A proved are
superseded, and refuse when P0-A proved nothing.

Two limits are deliberate and load-bearing.

Collapse is **document-level**, because P0-A's finality is document-level.  A
superseded document's projections all go; a final document's projections all
stay -- including the several a Korean 정정신고 carries side by side, where the
same holder and date appear with different share counts.  Choosing between
*those* would be a projection-level correction semantics that P0-A does not
define, so the selector is left to find them ambiguous.

And a chain is used only when it stays inside one issuer and one holder.  P0-A
resolves relations between *documents*; the timeline this feeds selects within
``(issuer, canonical reporter)``.  A chain crossing that boundary is refused
rather than reconciled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

#: The artifact layout this module can read.  Another version is a different
#: contract and is refused rather than best-effort parsed.
ARTIFACT_SCHEMA_VERSION = "1.0"

#: Where the generator writes, beside the report index it is paired with.
DEFAULT_ARTIFACT_PATH = "data/corpus/holding_correction_finality.json"

# -- P0-A's own statuses, carried through verbatim --------------------------
#: P0-A proved an original-plus-corrections chain with one final member.
STATUS_RESOLVED = "resolved"
#: P0-A found several candidate parents and refused to choose.
STATUS_AMBIGUOUS = "ambiguous"
#: P0-A found no candidate parent at all.
STATUS_UNRESOLVED = "unresolved"

# -- collapse outcomes -------------------------------------------------------
#: Every correction touching these reports is proven; the eligible set stands.
COLLAPSED = "collapsed"
#: Something about these reports' corrections is not proven.  The caller must
#: decline; it must not proceed with the corrections ignored.
UNPROVEN = "unproven"

#: Identity fields that bind an artifact to one corpus.  Spelled exactly as the
#: holding report index spells them, so the two artifacts cannot agree on the
#: corpus while disagreeing on what the words mean.
BINDING_FIELDS = (
    "corpus_snapshot_id",
    "corpus_manifest_sha256",
    "source_holding_disclosure_count",
)


@dataclass(frozen=True)
class CorrectionChain:
    """One P0-A correction group, as P0-A left it.

    ``final_doc_id`` is populated only where P0-A itself proved a final member.
    An ambiguous or unresolved group keeps ``None`` -- P0-A stores such a group
    with ``is_latest`` set on its lone member, and reading that as a verified
    final version is exactly the mistake this field's absence prevents.
    """

    group_id: str
    root_doc_id: str
    members: tuple[str, ...]
    final_doc_id: str | None
    status: str
    resolution_rule: str = ""
    confidence: float = 0.0
    #: Whether the chain's root is an original filing.  False when the root is
    #: itself a correction whose own target lies outside this corpus: the chain
    #: is complete at the tail, which is what finality needs, but it does not
    #: begin at an original and must not be described as though it did.
    head_complete: bool = True
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_resolved(self) -> bool:
        return self.status == STATUS_RESOLVED and bool(self.final_doc_id)

    @property
    def superseded(self) -> tuple[str, ...]:
        """Members the final one replaces.  Empty unless the chain is resolved."""

        if not self.is_resolved:
            return ()
        return tuple(doc for doc in self.members if doc != self.final_doc_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "root_doc_id": self.root_doc_id,
            "members": list(self.members),
            "final_doc_id": self.final_doc_id,
            "status": self.status,
            "resolution_rule": self.resolution_rule,
            "confidence": self.confidence,
            "head_complete": self.head_complete,
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "CorrectionChain":
        members = tuple(str(doc) for doc in (row.get("members") or ()))
        if not members:
            raise ValueError("a correction chain with no members")
        final = row.get("final_doc_id")
        return cls(
            group_id=str(row["group_id"]),
            root_doc_id=str(row["root_doc_id"]),
            members=members,
            final_doc_id=str(final) if final else None,
            status=str(row["status"]),
            resolution_rule=str(row.get("resolution_rule") or ""),
            confidence=float(row.get("confidence") or 0.0),
            head_complete=bool(row.get("head_complete", True)),
            provenance=dict(row.get("provenance") or {}),
        )


@dataclass(frozen=True)
class CollapseResult:
    """The reports that survive correction collapse, or why none may."""

    status: str
    eligible: tuple[Any, ...] = ()
    superseded_doc_ids: tuple[str, ...] = ()
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def proven(self) -> bool:
        return self.status == COLLAPSED


class HoldingCorrectionFinality:
    """P0-A's holding correction groups, ready to collapse one pair's timeline.

    The source is inert about report semantics: it knows documents and chains,
    never reference dates, holders or selectors.  Which report a question then
    names is decided by the report index, over whatever survives here.
    """

    def __init__(
        self,
        chains: Iterable[CorrectionChain],
        *,
        identity: Mapping[str, Any] | None = None,
        complete: bool = False,
    ) -> None:
        self.identity = dict(identity or {})
        self.complete = bool(complete)
        self._chains: dict[str, CorrectionChain] = {}
        self._by_doc: dict[str, CorrectionChain] = {}
        #: Reasons the source refuses to be used at all.  A source that cannot
        #: describe itself consistently is worse than no source: it would
        #: collapse timelines on the strength of a contradiction.
        problems: list[str] = []
        for chain in chains:
            if chain.group_id in self._chains:
                problems.append(f"duplicate group {chain.group_id}")
                continue
            if chain.is_resolved and chain.final_doc_id not in chain.members:
                problems.append(f"group {chain.group_id} names a final member "
                                "that is not in the group")
                continue
            self._chains[chain.group_id] = chain
            for doc in chain.members:
                if doc in self._by_doc:
                    problems.append(f"document {doc} is in two groups")
                    continue
                self._by_doc[doc] = chain
        self.problems = tuple(problems)

    # ------------------------------------------------------------- inspection
    @property
    def usable(self) -> bool:
        """Whether this source may be used to collapse anything at all."""

        return self.complete and not self.problems

    @property
    def chain_count(self) -> int:
        return len(self._chains)

    @property
    def resolved_chain_count(self) -> int:
        return sum(1 for chain in self._chains.values() if chain.is_resolved)

    @property
    def chains(self) -> tuple[CorrectionChain, ...]:
        return tuple(self._chains[key] for key in sorted(self._chains))

    def chain_for(self, doc_id: str) -> CorrectionChain | None:
        return self._by_doc.get(str(doc_id))

    def matches_identity(self, other: Mapping[str, Any]) -> bool:
        """Whether this source describes the same corpus as ``other``.

        Only the fields that bind an artifact to a corpus are compared, and all
        of them must be present on both sides: a missing field is a mismatch,
        never a pass, so an artifact that forgot to record its corpus cannot
        satisfy the check by omission.
        """

        for field_name in BINDING_FIELDS:
            mine = self.identity.get(field_name)
            theirs = other.get(field_name)
            if mine in (None, "") or theirs in (None, ""):
                return False
            if str(mine) != str(theirs):
                return False
        return True

    # -------------------------------------------------------------- collapse
    def collapse(
        self,
        records: Sequence[Any],
        *,
        pair_lookup: Callable[[str], frozenset] | None = None,
    ) -> CollapseResult:
        """Drop the reports of superseded documents, or refuse.

        ``records`` are one issuer/holder pair's reports.  ``pair_lookup`` maps
        a document to the pairs it appears under, and is what keeps a chain from
        collapsing across two holders: P0-A relates *documents*, while this
        timeline is one holder's, and the two are only interchangeable when the
        chain stays inside the pair.
        """

        corrections = [r for r in records if getattr(r, "is_correction", False)]
        if not corrections:
            # Nothing to collapse, and nothing to prove.  A pair with no
            # correction is untouched by this source even when it is unusable.
            return CollapseResult(COLLAPSED, tuple(records))
        if not self.usable:
            return CollapseResult(UNPROVEN, detail={
                "reason": "correction finality source is not usable",
                "problems": list(self.problems),
                "complete": self.complete,
            })

        present = {r.doc_id for r in records}
        superseded: set[str] = set()
        used: list[str] = []
        for doc_id in sorted(present):
            chain = self.chain_for(doc_id)
            if chain is None:
                if doc_id in {r.doc_id for r in corrections}:
                    # Flagged as a correction, yet P0-A knows no group for it.
                    # That is an unproven correction, not an absent one.
                    return CollapseResult(UNPROVEN, detail={
                        "reason": "a correction document is in no P0-A group",
                        "doc_id": doc_id,
                    })
                continue
            if not chain.is_resolved:
                return CollapseResult(UNPROVEN, detail={
                    "reason": "P0-A did not prove a final member for this chain",
                    "group_id": chain.group_id,
                    "status": chain.status,
                    "resolution_rule": chain.resolution_rule,
                })
            if chain.final_doc_id not in present:
                # Collapsing would delete this pair's reports and leave the
                # final document's reports somewhere this timeline cannot see.
                return CollapseResult(UNPROVEN, detail={
                    "reason": "the proven final document has no report in this "
                              "timeline",
                    "group_id": chain.group_id,
                    "final_doc_id": chain.final_doc_id,
                })
            if pair_lookup is not None:
                home = pair_lookup(doc_id) or frozenset()
                if len(home) != 1:
                    # This document reports for more than one holder, so
                    # "the chain stays inside one holder" cannot be established.
                    return CollapseResult(UNPROVEN, detail={
                        "reason": "a chain member reports for more than one "
                                  "issuer or reporter",
                        "group_id": chain.group_id,
                        "doc_id": doc_id,
                    })
                # A member absent from the index contributes no report, so it
                # cannot carry the chain out of this pair.  A member that is
                # present must be present under this pair and no other.
                foreign = sorted(
                    doc for doc in chain.members
                    if (pair_lookup(doc) or frozenset()) not in (frozenset(), home)
                )
                if foreign:
                    # The chain leaves this issuer or this holder.  Merging it
                    # would attribute one holder's correction to another.
                    return CollapseResult(UNPROVEN, detail={
                        "reason": "correction chain crosses issuer or reporter "
                                  "identity",
                        "group_id": chain.group_id,
                        "foreign_members": foreign,
                    })
            superseded.update(chain.superseded)
            used.append(chain.group_id)

        eligible = tuple(r for r in records if r.doc_id not in superseded)
        if not eligible:
            return CollapseResult(UNPROVEN, detail={
                "reason": "collapse would remove every report in the timeline",
            })
        return CollapseResult(
            COLLAPSED, eligible, tuple(sorted(superseded)),
            detail={"groups_applied": sorted(set(used)),
                    "removed_records": len(records) - len(eligible)},
        )


def load_finality(path: str | Path) -> HoldingCorrectionFinality | None:
    """Read a generated finality artifact, or ``None`` when there is none usable.

    A malformed or differently-versioned artifact yields ``None`` rather than a
    partly populated source: half a set of chains collapses some timelines and
    silently leaves others superseded.
    """

    location = Path(path)
    if not location.is_file():
        return None
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, Mapping):
        return None
    header = payload.get("header")
    rows = payload.get("groups")
    if not isinstance(header, Mapping) or not isinstance(rows, Sequence):
        return None
    if str(header.get("artifact_schema_version") or "") != ARTIFACT_SCHEMA_VERSION:
        return None
    try:
        chains = [CorrectionChain.from_dict(row) for row in rows]
    except (KeyError, TypeError, ValueError):
        return None
    return HoldingCorrectionFinality(
        chains, identity=header, complete=bool(payload.get("complete"))
    )
