"""Which state of a corrected filing one of its own projections states.

A Korean 정정신고 does not replace the report it corrects -- it reprints it.
The filing carries the affected table twice, once under a heading the filer
wrote as ``정정 전`` and once under ``정정 후``, and then states the corrected
report again in its own body.  All three are projections of one filing, one
reporter and one reference date, so the frozen evidence grouping puts them in
one holding event and the frozen field machinery finds two different values for
``after_shares`` and reports a field conflict.

That conflict is real about the *document* and wrong about the *question*.  The
document says both numbers, but it says which one is which: the value under
``정정 전`` is what the report used to state, and the filer printed it precisely
to record that it no longer holds.  A question about the holding as it stands
therefore has one authoritative value, and the two candidates were never equally
authoritative.

Two axes meet here and must not be confused -- the same distinction
:mod:`app.reasoning.correction_pair_roles` draws, one level further in:

``document version``
    Which *filing* states a value: the original, or the correction that
    superseded it.  Resolved by the correction graph, and not by this module.

``correction state``
    Which state *one filing* attributes a value to: what it previously stated,
    or what it now states.  That is what this module reads.

Nothing here reads a number, a rank, a receipt date or a table's position.  The
state is read from the label the filer wrote on the region the projection came
from, in the same spirit as
:func:`app.reasoning.correction_graph.extract_correction_notice`, which reads a
correction's target from the filing's own labelled cells rather than inferring
it.  A region that declares nothing, or that declares both states at once, has
no state here and is treated as unmarked -- which is what the body of the report
is, and what every non-correction filing is.
"""

from __future__ import annotations

import re
from typing import Any, Iterator, Mapping


#: The state a filing prints to record what it used to say.
PRIOR_STATE = "correction_prior"
#: The state a filing prints to record what it now says.
CORRECTED_STATE = "correction_corrected"

#: Written on the evidence set when a filing's own ``정정 전`` region was found
#: to be superseded by that same filing.  Diagnostic: it explains an authority
#: decision and no caller branches on it.
PRIOR_STATE_SUPERSEDED = "correction_prior_state_superseded"

#: The correction intents that ask for the prior state in its own right.  A
#: history question wants both states and an original question wants the first,
#: so neither may have the prior state resolved away.  The vocabulary is the
#: frozen parser's, written onto ``plan.evidence["correction_intent"]``.
_PRIOR_STATE_INTENTS = frozenset({"history", "original"})

#: The frozen retrieval policy that names the original filing.  A question the
#: parser read that way is asking for the superseded text.
_PRIOR_STATE_POLICIES = frozenset({"original_only"})

#: The frozen placeholder a filing writes in a row that a correction left
#: alone.  It is a statement about a row, not a heading over a region, and the
#: chunker already reads it as one -- ``projection_state`` records it.  It is
#: removed before the markers are read so that its own ``정정 전`` cannot be
#: mistaken for a heading.
_UNCHANGED_ROW = re.compile(r"정정\s*전\s*과\s*동일")

#: A correction-state heading.  The marker has to close its own word: a Korean
#: particle may follow it, and nothing else may.  ``정정 전액`` is a different
#: noun and ``정정 전후`` fuses the two markers into a compound that names a
#: comparison, so neither declares that the region under it is one state.  The
#: particle set is the frozen parser's own, spelled the way
#: ``query_understanding`` spells it where it reads the same two markers out of
#: a question.
_MARKER_BOUNDARY = r"(?=[\s\W]|$|[은는이가을를에의로도만와과랑으])"
_PRIOR_MARKER = re.compile(r"정정\s*전" + _MARKER_BOUNDARY)
_CORRECTED_MARKER = re.compile(r"정정\s*후" + _MARKER_BOUNDARY)


def _labels(chunk: Mapping[str, Any]) -> Iterator[str]:
    """Every structural label the chunker attached to this projection.

    The table's own caption first, then the section it sits in and the section
    path above it.  All of them are read, because a filing may put the marker on
    the heading of the region or on the table itself, and neither placement is
    more official than the other.
    """

    yield str(chunk.get("table_title") or "")
    yield str(chunk.get("section_title") or "")
    path = chunk.get("section_path")
    if isinstance(path, str):
        yield path
        return
    for value in path or ():
        yield str(value or "")


def declared_correction_state(chunk: Mapping[str, Any] | None) -> str | None:
    """The correction state this projection's own labels declare, or nothing.

    Fails closed twice.  A chunk whose labels carry no marker has no state --
    that is the body of a report, and every filing that is not a correction.  A
    chunk whose labels carry both markers declares no single state either: a
    heading naming the before and the after describes a comparison, not one side
    of it, and choosing a side would be this module inventing the answer.
    """

    if not isinstance(chunk, Mapping):
        return None
    prior = False
    corrected = False
    for label in _labels(chunk):
        if not label:
            continue
        text = _UNCHANGED_ROW.sub(" ", label)
        if _PRIOR_MARKER.search(text):
            prior = True
        if _CORRECTED_MARKER.search(text):
            corrected = True
    if prior == corrected:
        return None
    return PRIOR_STATE if prior else CORRECTED_STATE


def item_correction_state(item: Any) -> str | None:
    """The state of a served evidence item, read from the metadata it carries.

    The state is computed once, where the item's holding metadata is built, so
    what the item reports and what the evidence output shows cannot drift apart.
    An item built before that key existed is read from its source chunk instead.
    """

    holding = getattr(item, "holding", None)
    if isinstance(holding, Mapping) and "correction_state" in holding:
        value = holding.get("correction_state")
        return str(value) if value else None
    provenance = getattr(item, "provenance", None)
    source = provenance.get("source_chunk") if isinstance(provenance, Mapping) else None
    return declared_correction_state(source if isinstance(source, Mapping) else None)


def current_state_is_authoritative(plan: Mapping[str, Any] | None) -> bool:
    """Whether this question wants the state a corrected filing now states.

    Read from the plan alone, never from the served evidence: what a question
    asks for is settled by query understanding, and the number of states that
    happen to have been retrieved says nothing about it.  A question that names
    the correction history or the original filing is refused here, so the prior
    state keeps every bit of the standing it had for those.
    """

    if not isinstance(plan, Mapping):
        return True
    evidence = plan.get("evidence")
    intent = ""
    if isinstance(evidence, Mapping):
        intent = str(evidence.get("correction_intent") or "")
    if intent in _PRIOR_STATE_INTENTS:
        return False
    return str(plan.get("correction_policy") or "") not in _PRIOR_STATE_POLICIES


__all__ = [
    "CORRECTED_STATE",
    "PRIOR_STATE",
    "PRIOR_STATE_SUPERSEDED",
    "current_state_is_authoritative",
    "declared_correction_state",
    "item_correction_state",
]
