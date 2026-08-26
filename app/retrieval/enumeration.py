"""Correction-aware collapse for Tier 2 enumeration (P0-C Step 2).

Tier 1 needs nothing from this module: the P0-B event timeline already stores
one row per logical lifecycle, with its correction chain folded in.  Tier 2
enumerates raw disclosures for the families P0-B does not model, so it has to do
that folding itself -- and the only correct way to do it is to read the P0-A
correction graph, never to re-derive which filing corrects which.

The counting rule this module implements:

``resolved``    one correction group is one logical document; the verified
                latest filing represents it.
``ambiguous``   do not fold.  P0-A could not establish what this filing
                corrects, so folding it would silently delete a document that
                may well be distinct.
``unresolved``  do not fold, for the same reason.
no group        a standalone document, counted once.

``is_latest`` is never sufficient on its own.  P0-A stores an ambiguous or
unresolved correction as a one-member group that also carries ``is_latest``, so
using that flag alone would fold exactly the chains it could not verify.
:attr:`CorrectionDocumentState.is_resolved_latest` is the predicate that means
"verified final version", and it is the one used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


#: How a logical document was arrived at.  ``standalone`` is not a P0-A status:
#: it marks a document that takes part in no correction at all, which P0-A
#: deliberately does not store a row for.
ORIGIN_RESOLVED_GROUP = "resolved_group"
ORIGIN_UNVERIFIED_CORRECTION = "unverified_correction"
ORIGIN_STANDALONE = "standalone"

_RESOLVED = "resolved"


@dataclass(frozen=True)
class LogicalDocument:
    """One logical document, and the raw filings that stand behind it."""

    representative_doc_id: str
    member_doc_ids: tuple[str, ...]
    origin: str
    correction_group_id: str | None = None
    resolution_status: str | None = None

    @property
    def is_collapsed(self) -> bool:
        return len(self.member_doc_ids) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "representative_doc_id": self.representative_doc_id,
            "member_doc_ids": list(self.member_doc_ids),
            "origin": self.origin,
            "correction_group_id": self.correction_group_id,
            "resolution_status": self.resolution_status,
            "is_collapsed": self.is_collapsed,
        }


@dataclass(frozen=True)
class LogicalDocumentSet:
    """The result of collapsing an enumerated set of raw disclosures."""

    documents: tuple[LogicalDocument, ...] = ()
    raw_doc_ids: tuple[str, ...] = ()
    unresolved_doc_ids: tuple[str, ...] = ()

    @property
    def logical_count(self) -> int:
        return len(self.documents)

    @property
    def raw_count(self) -> int:
        return len(self.raw_doc_ids)

    @property
    def representative_ids(self) -> tuple[str, ...]:
        """Deterministic; safe to hand straight to a batch state lookup."""

        return tuple(
            sorted(document.representative_doc_id for document in self.documents)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_count": self.raw_count,
            "logical_count": self.logical_count,
            "collapsed_group_count": sum(
                1 for document in self.documents if document.is_collapsed
            ),
            "unresolved_count": len(self.unresolved_doc_ids),
            "documents": [document.to_dict() for document in self.documents],
        }


def collapse_logical_documents(
    doc_ids: Iterable[str],
    states: Mapping[str, Any],
) -> LogicalDocumentSet:
    """Fold an enumerated disclosure set into logical documents.

    ``states`` is the output of P0-A's batch
    :meth:`PostgresCorrectionRepository.document_states` -- one call for the
    whole set, never one per document.  Documents absent from it take part in no
    correction and are counted standalone, which is why a missing key here is
    normal rather than an error.

    Only members present in ``doc_ids`` are folded.  A correction chain that the
    enumeration window covers only partly stays represented by the newest filing
    *inside* the window, so the count describes the set that was asked for and
    never silently reaches outside it.
    """

    ordered = tuple(sorted({str(doc_id) for doc_id in doc_ids if str(doc_id).strip()}))
    present = set(ordered)

    groups: dict[str, list[str]] = {}
    standalone: list[str] = []
    unverified: list[str] = []

    for doc_id in ordered:
        state = states.get(doc_id)
        if state is None:
            standalone.append(doc_id)
            continue
        if _status(state) != _RESOLVED:
            # Ambiguous or unresolved: P0-A never established what this filing
            # corrects, so it stays its own document.
            unverified.append(doc_id)
            continue
        group_id = str(getattr(state, "correction_group_id", "") or "")
        if not group_id:
            standalone.append(doc_id)
            continue
        groups.setdefault(group_id, []).append(doc_id)

    documents: list[LogicalDocument] = []
    for doc_id in standalone:
        documents.append(
            LogicalDocument(
                representative_doc_id=doc_id,
                member_doc_ids=(doc_id,),
                origin=ORIGIN_STANDALONE,
            )
        )
    for doc_id in unverified:
        state = states[doc_id]
        documents.append(
            LogicalDocument(
                representative_doc_id=doc_id,
                member_doc_ids=(doc_id,),
                origin=ORIGIN_UNVERIFIED_CORRECTION,
                correction_group_id=getattr(state, "correction_group_id", None),
                resolution_status=_status(state),
            )
        )
    for group_id, members in groups.items():
        members = sorted(members)
        documents.append(
            LogicalDocument(
                representative_doc_id=_representative(members, states, present),
                member_doc_ids=tuple(members),
                origin=ORIGIN_RESOLVED_GROUP,
                correction_group_id=group_id,
                resolution_status=_RESOLVED,
            )
        )

    documents.sort(key=lambda document: document.representative_doc_id)
    return LogicalDocumentSet(
        documents=tuple(documents),
        raw_doc_ids=ordered,
        unresolved_doc_ids=tuple(sorted(unverified)),
    )


def _status(state: Any) -> str:
    value = getattr(state, "resolution_status", None)
    return str(getattr(value, "value", value) or "")


def _representative(
    members: Sequence[str], states: Mapping[str, Any], present: set[str]
) -> str:
    """The verified latest filing of a resolved group, restricted to the set.

    Falls back to the highest ``correction_order`` present when the group's own
    latest lies outside the enumerated window, so the representative is always a
    document the caller actually asked about.
    """

    for doc_id in members:
        state = states.get(doc_id)
        if state is not None and getattr(state, "is_resolved_latest", False):
            return doc_id
    return max(
        members,
        key=lambda doc_id: (
            int(getattr(states.get(doc_id), "correction_order", 0) or 0),
            doc_id,
        ),
    )


__all__ = [
    "ORIGIN_RESOLVED_GROUP",
    "ORIGIN_STANDALONE",
    "ORIGIN_UNVERIFIED_CORRECTION",
    "LogicalDocument",
    "LogicalDocumentSet",
    "collapse_logical_documents",
]
