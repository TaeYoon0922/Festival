"""Add a company's structural corpus to an isolated evaluation database.

Additive on purpose.  Embeddings are expensive and are keyed by
``(chunk_id, embedding_model, embedding_version)``, so re-seeding a company must
not delete rows that already carry vectors -- including vectors from a different
model, which are meant to coexist.  Nothing here writes an embedding; this loads
only companies, disclosures, sections, tables and chunks.

Reads the frozen processed corpus and the manifest.  Chunk IDs, ``retrieval_text``
and provenance come from the export unchanged, so the evaluation database holds
exactly what production ingestion would hold.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psycopg

PROCESSED = Path("data/processed/structural_v2_1_full_4204")
MANIFEST = Path("data/corpus/manifest.jsonl")


def iso(value: object) -> str | None:
    text = str(value or "")
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return None


def load_index() -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = defaultdict(list)
    with (PROCESSED / "index.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                index[record["doc_id"]].append(record)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corp_codes", nargs="+")
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL"))
    args = parser.parse_args()
    if not args.dsn:
        raise SystemExit("DATABASE_URL or --dsn is required")
    if "55433" not in args.dsn:
        # The evaluation database is the only sanctioned write target.
        raise SystemExit("refusing to write outside the isolated evaluation DB")

    manifest = [json.loads(l) for l in MANIFEST.open(encoding="utf-8") if l.strip()]
    index = load_index()
    targets = set(args.corp_codes)
    docs = [r["doc_id"] for r in manifest if r["corp_code"] in targets]
    print(f"seeding {len(targets)} companies, {len(docs)} documents (structural only)")

    started = time.time()
    with psycopg.connect(args.dsn) as conn:
        with conn.cursor() as cur:
            companies = {
                str(r["corp_code"]): (
                    str(r["corp_code"]), str(r["stock_code"]), str(r["corp_name"]),
                    str(r.get("listed_name") or ""), None,
                    str(r.get("industry") or ""), str(r.get("sector") or ""), "{}",
                )
                for r in manifest
            }
            cur.executemany(
                "INSERT INTO companies VALUES (%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT DO NOTHING", list(companies.values()))
            cur.executemany(
                "INSERT INTO disclosures (doc_id,corp_code,rcept_no,report_nm,rcept_dt,"
                "doc_group,doc_subtype,is_correction,base_year,base_month,file_path,"
                "file_format,metadata) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'{}') "
                "ON CONFLICT DO NOTHING",
                [(str(r["doc_id"]), str(r["corp_code"]), str(r["rcept_no"]),
                  r["report_nm"], iso(r["rcept_dt"]), r["doc_group"], r.get("doc_subtype"),
                  bool(r.get("is_correction")), r.get("base_year"), r.get("base_month"),
                  r["file_path"], r["file_format"]) for r in manifest])

            sections: list[tuple] = []
            tables: list[tuple] = []
            chunks: list[tuple] = []
            seen: set[tuple[str, str]] = set()
            for doc_id in docs:
                for record in index.get(doc_id, []):
                    part = record["part_id"]
                    path = PROCESSED / record["output_path"].replace("\\", "/")
                    if not path.exists():
                        continue
                    data = json.load(gzip.open(path, "rt", encoding="utf-8"))
                    for order, section in enumerate(data.get("sections") or []):
                        key = (part, str(section["section_id"]))
                        if key in seen:
                            continue
                        seen.add(key)
                        sections.append((
                            part, str(section["section_id"]), doc_id, None,
                            str(section.get("title") or section.get("section_title") or "본문"),
                            json.dumps(section.get("section_path") or [], ensure_ascii=False),
                            order,
                            int(section.get("level") or section.get("section_depth") or 0),
                            str(section.get("content") or ""), "{}"))
                    if (part, "s1") not in seen:
                        seen.add((part, "s1"))
                        sections.append((part, "s1", doc_id, None, "본문", "[]", 0, 0, "", "{}"))
                    table_rows: dict[str, int] = {}
                    for order, table in enumerate(data.get("tables") or []):
                        section_id = str(table.get("section_id") or "s1")
                        if (part, section_id) not in seen:
                            section_id = "s1"
                        table_rows[str(table["table_id"])] = len(table.get("rows") or [])
                        tables.append((
                            part, str(table["table_id"]), doc_id, section_id, None, order,
                            "p", len(table.get("rows") or []), "{}",
                            json.dumps(table.get("rows") or [], ensure_ascii=False), "{}"))
                    for chunk in data.get("chunks") or []:
                        section_id = str(chunk.get("section_id") or "s1")
                        if (part, section_id) not in seen:
                            section_id = "s1"
                        table_id = str(chunk.get("table_id") or "") or None
                        if table_id and table_id not in table_rows:
                            table_id = None
                        meta = {k: v for k, v in chunk.items() if k not in {
                            "chunk_id", "doc_id", "section_id", "table_id", "chunk_type",
                            "kind", "chunk_order", "content", "retrieval_text",
                            "char_count"}}
                        chunks.append((
                            str(chunk["chunk_id"]), doc_id, part, section_id, table_id,
                            str(chunk.get("chunk_type") or chunk.get("kind") or "text"),
                            int(chunk.get("chunk_order") or 0),
                            str(chunk.get("content") or ""),
                            str(chunk.get("retrieval_text") or chunk.get("content") or ""),
                            int(chunk.get("char_count") or len(str(chunk.get("content") or ""))),
                            chunk.get("retrieval_priority"),
                            json.dumps(meta, ensure_ascii=False)))
            cur.executemany("INSERT INTO sections VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                            "ON CONFLICT DO NOTHING", sections)
            cur.executemany("INSERT INTO disclosure_tables VALUES "
                            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                            tables)
            cur.executemany("INSERT INTO chunks VALUES "
                            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                            chunks)
            print(f"  sections={len(sections)} tables={len(tables)} chunks={len(chunks)}")
        conn.commit()
    print(f"structural seed took {time.time() - started:.1f}s (no embeddings written)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
