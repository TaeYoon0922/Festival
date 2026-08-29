# STEP 7 — Lifecycle Relation Verification

**Corpus relation: VERIFIED.  PostgreSQL corroboration: UNCONFIRMED.  Gold impact: NONE.**

| Field | Value |
|---|---|
| Corpus relation | **VERIFIED** — established independently from the filings |
| PostgreSQL corroboration | **UNCONFIRMED** |
| Reason | server access unavailable from the authoring workstation |
| Gold impact | **NONE** |
| Future DB comparison | implementation diagnostic only |

The relation itself is **not ambiguous**. Only the database corroboration is outstanding. Under
the STEP 8 decision, Gold truth is determined from the canonical disclosure corpus; the
PostgreSQL correction/event relation tables are derived implementation artifacts and are not a
prerequisite for Gold validity. If the event graph later fails to reproduce one of these
relations, that is a production relation/resolver finding, not a Gold error.

No relation was guessed, and no gold was altered to compensate.

---

## 1. Access attempt

| Step | Result |
|---|---|
| `festival-test-server` hostname | does not resolve (no `~/.ssh/config` entry) |
| Server IPs in the repository | deliberately absent — `TEAM_GUIDE.md` §5 keeps them out because the repo is public |
| Hosts in `~/.ssh/known_hosts` | 4 candidates; only `101.79.20.171` has port 22 open |
| `ssh -o BatchMode=yes root@101.79.20.171` | `Permission denied (publickey,password)` |

The workstation's key is not authorised on the reachable host, and `TEAM_GUIDE.md` §5 notes the
security group only admits an operator's *current* public IP, added manually. I stopped here
rather than probing further hosts or attempting other credentials.

**Nothing was written anywhere.** No `SELECT` was issued because no connection was established.

---

## 2. What was verified instead — corpus-level cross-verification

The DB check was intended to *corroborate* a relation the filings already state. Per the brief
(§13: "whether DB relation agrees with the filing's own 관련공시 field"; §14: "Independent Gold
follows corpus truth"), I performed the corpus-level verification locally and in full.

For each pair I checked **three independent in-document identifiers**:

1. the termination filing's own `관련공시` back-reference (date + disclosure type);
2. whether that reference resolves to exactly **one** corpus filing for that company on that date;
3. whether 계약상대방 and 체결계약명 match between the two filings.

| Item(s) | Company | Original → Termination | 관련공시 resolves uniquely | Counterparty match | Contract-name match | Amount |
|---|---|---|---|---|---|---|
| C072 / C075 | 대우건설 | `exchange_20230228801277` → `exchange_20241115800529` | **yes** (1 candidate) | ✔ 경안리버시티개발 주식회사 | ✔ 광주 경안2지구 도시개발사업 | 351,882,914,000 → 362,307,038,000 (the delta the question asks for) |
| C074 | 효성중공업 | `exchange_20241104800041` → `exchange_20250508800712` | **yes** (1 candidate) | ✔ Orsted Hornsea Project Four Limited | ✔ Hornsea Four Offshore Wind Farm (HOW04) | 291,204,288,000 = 291,204,288,000 |
| C076 | 삼성E&A | `exchange_20240618800188` → `exchange_20241128800504` | **yes** (1 in-corpus candidate; two other referenced dates fall outside the corpus window) | ✔ SONATRACH (알제리 국영석유회사) | ✔ 알제리 Hassi Messaoud 정유 프로젝트 | 1,937,231,667,470 = 1,937,231,667,470 |
| C077 | 두산퓨얼셀 | `exchange_20230331802739` → `exchange_20240603800359` | **yes** (1 in-corpus candidate) | ✔ Zhejiang Beisen Hydrogen Science & Technology | ✔ 연료전지 시스템 공급 계약 | 19,300,000,000 = 19,300,000,000 |

**All four relations are established by the filings themselves**, on three concurring identifiers
each. Three of four additionally show an exact amount match; the 대우건설 pair differs by exactly
the amount its T5 question asks about.

Per-item verdicts under the brief's vocabulary:

| Item | DB result | Corpus agreement | Verdict |
|---|---|---|---|
| IEV2-C072 | not obtained | relation stated in-filing, uniquely resolvable | **CORPUS-VERIFIED / DB-UNCONFIRMED** |
| IEV2-C074 | not obtained | as above, plus exact amount match | **CORPUS-VERIFIED / DB-UNCONFIRMED** |
| IEV2-C075 | not obtained | as above | **CORPUS-VERIFIED / DB-UNCONFIRMED** |
| IEV2-C076 | not obtained | as above, plus exact amount match | **CORPUS-VERIFIED / DB-UNCONFIRMED** |
| IEV2-C077 | not obtained | as above, plus exact amount match | **CORPUS-VERIFIED / DB-UNCONFIRMED** |

No `MISMATCH` was observed, because no DB result exists to conflict with. `NOT_FOUND` is not the
right label either — the relation *is* found in the corpus; only the DB corroboration is missing.

---

## 3. Gold defect found during this verification

The verification surfaced an imprecision in the T4 gold that STEP 6 had missed: the expected
answers named the **공시 접수일** while the filings also carry a distinct **해지일자**.

| Item | Filing receipt date | 해지일자 in the filing |
|---|---|---|
| C074 효성중공업 | 2025-05-08 | **2025-05-07** |
| C075 대우건설 | 2024-11-15 | **2024-11-14** |
| C076 삼성E&A | 2024-11-28 | 2024-11-28 |
| C077 두산퓨얼셀 | 2024-06-03 | **2024-05-31** |

Three of four have a 해지일자 that differs from the disclosure date. The gold has been corrected
to state both explicitly, so a correct answer using either date is not penalised and a wrong date
is still detectable. This is a gold-precision fix derived from the corpus, not from any
prediction.

---

## 4. Decision taken — STEP 8

**Option (b) was adopted:** corpus truth is the Gold source of truth for Independent Eval v2.

All five items are **frozen into `gold.jsonl`** carrying
`lifecycle_relation_status: "corpus_verified_db_unconfirmed"` and their per-item relation
evidence. `corpus_verified` remains `true` for all five — DB non-confirmation does not weaken it.

When server access becomes available, the queries in §5 should still be run, but as an
**implementation diagnostic**: a disagreement would be evidence about the production event
graph, scored under STEP 10's failure taxonomy, never a reason to edit Gold.

---

## 5. Read-only queries prepared for when access is available

Issued verbatim, no writes, no DDL:

```sql
-- 1. correction relations touching the lifecycle documents
SELECT * FROM correction_relations
 WHERE source_doc_id IN (:docs) OR target_doc_id IN (:docs);

-- 2. correction group membership
SELECT * FROM correction_group_members WHERE doc_id IN (:docs);

-- 3. corporate events referencing them
SELECT e.* FROM corporate_events e
  JOIN corporate_event_members m ON m.event_id = e.event_id
 WHERE m.doc_id IN (:docs);

-- 4. event membership and relation direction
SELECT * FROM corporate_event_members WHERE doc_id IN (:docs);
SELECT * FROM corporate_event_relations
 WHERE from_event_id IN (:events) OR to_event_id IN (:events);

-- 5. disclosure metadata for the pairs
SELECT doc_id, corp_code, corp_name, doc_group, report_nm, rcept_dt, is_correction
  FROM disclosures WHERE doc_id IN (:docs);
```

`:docs` = the ten documents in the table in §2. Column names must be confirmed against
`db/006_correction_graph.sql` and `db/007_corporate_event_timeline.sql` at run time; if a column
differs, the query is adjusted, never the schema.
