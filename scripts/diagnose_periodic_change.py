"""Say which gate left a growth-rate question unresolved.

``think_trace`` reports ``periodic_metric_change_unresolved`` and nothing more,
because the resolver returns ``None`` rather than a reason. That is the right
shape for production -- a refusal is not a finding -- but it leaves no way to
tell a question the calculator declined on purpose from one it declined because
a gate is mis-set.

So this runs the real pipeline in process, wraps the resolver, and reports each
gate in the order the resolver checks it, against the same resolution the served
answer was built from. It changes nothing: the wrapper delegates to the original
and only reads what it was passed.

    python scripts/diagnose_periodic_change.py
    python scripts/diagnose_periodic_change.py --question "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import orchestrator
from app.api.pipeline import AnswerPipeline
from app.reasoning import periodic_metric_change as change_module


DEFAULT_QUESTIONS = (
    "삼성전자 2025년 1분기 매출액은 전년 동기 대비 몇 퍼센트 증가했어?",
    "삼성전자의 2025년 매출액은 2024년 대비 몇 퍼센트 증가했는가?",
)


def _describe(request: Any, resolution: Any) -> list[str]:
    """Each gate in ``resolve_periodic_metric_change`` order, with its verdict."""

    years = tuple(getattr(request, "years", ()) or ())
    facts = tuple(getattr(resolution, "facts", ()) or ())
    lines = [
        f"metric           {getattr(request, 'metric', None)}",
        f"years            {years}",
        f"comparison_type  {getattr(request, 'comparison_type', None)}",
        "",
        f"1 two ascending years          {'PASS' if len(years) == 2 and years[0] < years[1] else 'FAIL'}",
    ]
    unresolved = tuple(getattr(resolution, "unresolved_requirements", ()) or ())
    ambiguity = getattr(resolution, "temporal_ambiguity", None)
    lines.append(
        f"2 resolution fully resolved    "
        f"{'PASS' if not unresolved and not ambiguity else 'FAIL'}"
        f"   unresolved={list(unresolved)} temporal_ambiguity={ambiguity}"
    )
    lines.append(
        f"3 exactly one fact             "
        f"{'PASS' if len(facts) == 1 else 'FAIL'}   facts={len(facts)}"
    )
    if not facts:
        return lines
    fact = facts[0]
    sources = tuple(getattr(fact, "sources", ()) or ())
    lines.append(
        f"4 one source, no conflict      "
        f"{'PASS' if len(sources) == 1 and not getattr(fact, 'fact_conflict', None) else 'FAIL'}"
        f"   sources={len(sources)} conflict={getattr(fact, 'fact_conflict', None)}"
    )
    if len(sources) == 1:
        text = str(getattr(sources[0], "fact_text", "") or "")
        lines.append("")
        lines.append("source row:")
        lines.extend(f"  {line}" for line in text.splitlines()[:6])
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose growth-rate gates.")
    parser.add_argument("--question", action="append", default=None)
    args = parser.parse_args(argv)

    original = change_module.resolve_periodic_metric_change
    reports: list[list[str]] = []

    def wrapped(request, resolution, *, query_plan):
        outcome = original(request, resolution, query_plan=query_plan)
        report = _describe(request, resolution)
        report.append("")
        report.append(f"RESULT           {'resolved' if outcome else 'unresolved'}")
        reports.append(report)
        return outcome

    orchestrator.resolve_periodic_metric_change = wrapped
    pipeline = AnswerPipeline.from_env()

    for question in (args.question or list(DEFAULT_QUESTIONS)):
        reports.clear()
        print("=" * 78)
        print(f"Q  {question}")
        trace = pipeline.answer("GR", question)["think_trace"]
        stages = [stage for stage in trace.get("stages") or () if "metric_change" in stage]
        print(f"   stage: {stages or 'the calculator did not engage'}")
        if not reports:
            print("   the resolver was never called: no growth request was recognised")
            continue
        for report in reports:
            print()
            for line in report:
                print(f"   {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
