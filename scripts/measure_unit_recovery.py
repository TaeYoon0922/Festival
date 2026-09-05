"""Measure where a missing table unit could be recovered from, before writing any.

The unit gate blocks periodic answers, and the standing conclusion is that
fixing it costs a re-parse: new chunk ids, and 1.36M re-embeddings behind them.
That conclusion is worth re-testing, because it rests on the unit being missing
from the corpus rather than merely missing from one field.

Chunking searches for "(단위: …)" in a local window -- the section path, the
blocks next to the table, and its first five rows. A long table split into many
chunks puts most of those chunks outside that window while the section that
states the unit once, at the top, is still right there in the database. If that
is where the misses are, the fix is an UPDATE of one metadata field, which
touches neither chunk identity nor retrieval text nor embeddings -- exactly the
contract `db/005_table_chunk_provenance_backfill.sql` already established.

This measures only. It writes nothing.

    python scripts/measure_unit_recovery.py
    python scripts/measure_unit_recovery.py --doc-group periodic
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

#: The chunker's own patterns, as SQL. Kept identical on purpose: a measurement
#: that recognises more units than the parser does would promise a recovery the
#: backfill could not repeat.
UNIT_REGEX = r"(\(\s*단위\s*[:：]?\s*[^)\n|]{1,40}\))|(단위\s*[:：]\s*[^\n|\]]{1,40})"

#: ``$1`` is server-side prepared-statement syntax; psycopg substitutes ``%s``.
#: The doc-group filter is composed rather than parameterised so that "all"
#: drops the clause instead of comparing against NULL.
def _group_clause(group: str | None) -> str:
    return "" if group is None else "  AND d.doc_group = %s\n"


def baseline(group: str | None) -> tuple[str, tuple]:
    return (
        f"""
SELECT
  count(*)                                                 AS 표청크,
  count(*) FILTER (WHERE c.metadata->>'unit' IS NOT NULL)   AS 단위있음,
  count(*) FILTER (WHERE c.metadata->>'unit' IS NULL)       AS 단위없음
FROM chunks c
JOIN disclosures d ON d.doc_id = c.doc_id
WHERE c.chunk_type IN ('table', 'table_projection')
{_group_clause(group)};
""",
        () if group is None else (group,),
    )


def sources(group: str | None) -> tuple[str, tuple]:
    """Each source counted against the same denominator: chunks with no unit."""

    sibling = """EXISTS (
      SELECT 1 FROM chunks s
      WHERE s.doc_id = m.doc_id
        AND m.table_id IS NOT NULL
        AND s.metadata->>'source_table_id' = m.table_id
        AND s.metadata->>'unit' IS NOT NULL
    )"""
    section = """EXISTS (
      SELECT 1 FROM sections sec
      WHERE sec.source_part_id = m.source_part_id
        AND sec.section_id = m.section_id
        AND sec.content ~ %s
    )"""
    own = "m.content ~ %s"
    return (
        f"""
WITH missing AS (
  SELECT c.chunk_id, c.doc_id, c.source_part_id, c.section_id,
         c.metadata->>'source_table_id' AS table_id, c.content
  FROM chunks c
  JOIN disclosures d ON d.doc_id = c.doc_id
  WHERE c.chunk_type IN ('table', 'table_projection')
    AND c.metadata->>'unit' IS NULL
{_group_clause(group)})
SELECT
  count(*)                                    AS 단위없음,
  count(*) FILTER (WHERE {sibling})            AS 같은표의다른청크,
  count(*) FILTER (WHERE {section})            AS 섹션본문,
  count(*) FILTER (WHERE {own})                AS 청크본문,
  count(*) FILTER (
    WHERE {sibling} OR {section} OR {own}
  )                                           AS 셋중하나라도
FROM missing m;
""",
        ((group,) if group is not None else ())
        # The regex appears four times: once each for the section and own-content
        # columns, and once each again inside the combined column.
        + (UNIT_REGEX,) * 4,
    )


def ambiguity(group: str | None) -> tuple[str, tuple]:
    """A section stating two units hands one to nobody."""

    return (
        f"""
WITH missing AS (
  SELECT c.chunk_id, c.source_part_id, c.section_id
  FROM chunks c
  JOIN disclosures d ON d.doc_id = c.doc_id
  WHERE c.chunk_type IN ('table', 'table_projection')
    AND c.metadata->>'unit' IS NULL
{_group_clause(group)}), hits AS (
  SELECT m.chunk_id, count(DISTINCT u.match) AS distinct_units
  FROM missing m
  JOIN sections sec
    ON sec.source_part_id = m.source_part_id AND sec.section_id = m.section_id
  CROSS JOIN LATERAL (
    SELECT DISTINCT regexp_matches(sec.content, %s, 'g') AS match
  ) u
  GROUP BY m.chunk_id
)
SELECT
  count(*)                                   AS 섹션에단위있음,
  count(*) FILTER (WHERE distinct_units = 1)  AS 단위가하나,
  count(*) FILTER (WHERE distinct_units > 1)  AS 단위가여럿
FROM hits;
""",
        ((group,) if group is not None else ()) + (UNIT_REGEX,),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure unit recoverability.")
    parser.add_argument("--doc-group", default="periodic", help="or 'all'")
    # An empty DSN is the normal case here, not a missing one: this deployment
    # configures PGHOST/PGUSER/PGPASSWORD and libpq reads them itself, which is
    # also how the backend connects when DATABASE_URL is absent.
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_URL") or os.getenv("FESTIVAL_DATABASE_URL") or "",
    )
    args = parser.parse_args(argv)

    group = None if args.doc_group == "all" else args.doc_group

    import psycopg

    with psycopg.connect(args.dsn) as connection, connection.cursor() as cursor:
        for title, build in (
            ("기준", baseline),
            ("회수 가능한 출처", sources),
            ("섹션 단위의 모호성", ambiguity),
        ):
            sql, params = build(group)
            cursor.execute(sql, params)
            columns = [description[0] for description in cursor.description]
            row = cursor.fetchone()
            print(f"\n{title}")
            print("-" * 46)
            for name, value in zip(columns, row):
                print(f"  {name:<16} {value:>12,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
