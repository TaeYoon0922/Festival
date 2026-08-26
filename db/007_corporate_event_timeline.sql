\set ON_ERROR_STOP on

-- Additive corporate event timeline. Every row here is derived from frozen
-- disclosure metadata, from the structured contract tables a filing already
-- carries, and from the correction graph db/006 already resolved. Nothing
-- rewrites chunk identity, content, retrieval_text, ordering, provenance, or
-- embeddings, so applying this migration never requires re-chunking or
-- re-embedding, and db/006 is not touched.
--
-- SCOPE: this file is a FRESH-INSTALL migration, and re-running it against a
-- database it already created is safe (verified: apply #1 and apply #2 both
-- succeed with zero errors). It is NOT an upgrade path from an earlier draft of
-- itself. Every table is created with IF NOT EXISTS, so if a differently shaped
-- corporate_event_* table already exists the CREATE is skipped and the later
-- statements fail against the columns they expect -- verified empirically: an
-- older shape without root_logical_key fails at
-- uq_corporate_events_logical_root with "column root_logical_key does not
-- exist", leaving that pre-existing table untouched rather than damaged.
--
-- This is acceptable because 007 has never been applied to any server: it is
-- introduced by P0-B. Should a future revision change the shape after this file
-- ships, that revision must add its own ALTER-based upgrade steps rather than
-- relying on IF NOT EXISTS.
BEGIN;

-- One contract lifecycle. ``event_id`` is a deterministic hash of corp_code,
-- family, and one immutable root_logical_key. Later contract/update/termination
-- members never change that id. A resolved P0-A opening group contributes its
-- correction_group_id; an ordinary opening contributes its doc_id.
CREATE TABLE IF NOT EXISTS corporate_events (
    event_id text PRIMARY KEY,
    corp_code text NOT NULL REFERENCES companies(corp_code),
    event_family text NOT NULL,
    root_logical_key text NOT NULL,
    lifecycle_status text NOT NULL,
    resolution_status text NOT NULL,
    resolution_source text NOT NULL,
    opened_at date,
    closed_at date,
    member_count integer NOT NULL,
    confidence double precision NOT NULL DEFAULT 0.0,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    -- v1 supports exactly the two families the corpus audit found structured
    -- fields for. Widening this is a schema decision, not a data accident.
    CONSTRAINT corporate_events_family_check
        CHECK (event_family IN ('supply_contract', 'treasury_trust_contract')),
    CONSTRAINT corporate_events_lifecycle_check
        CHECK (lifecycle_status IN ('open', 'terminated')),
    CONSTRAINT corporate_events_status_check
        CHECK (resolution_status IN ('resolved', 'ambiguous', 'unresolved')),
    CONSTRAINT corporate_events_confidence_check
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT corporate_events_member_count_check
        CHECK (member_count > 0),
    CONSTRAINT corporate_events_root_logical_key_check
        CHECK (btrim(root_logical_key) <> ''),
    -- An open lifecycle has not been closed by anything.
    CONSTRAINT corporate_events_open_has_no_close
        CHECK (lifecycle_status <> 'open' OR closed_at IS NULL)
);

CREATE INDEX IF NOT EXISTS idx_corporate_events_corp
    ON corporate_events (corp_code, event_family);
CREATE UNIQUE INDEX IF NOT EXISTS uq_corporate_events_logical_root
    ON corporate_events (corp_code, event_family, root_logical_key);
CREATE INDEX IF NOT EXISTS idx_corporate_events_lifecycle
    ON corporate_events (lifecycle_status, resolution_status);
CREATE INDEX IF NOT EXISTS idx_corporate_events_opened
    ON corporate_events (opened_at);

-- One disclosure's place inside one lifecycle. ``canonical_doc_id`` is the
-- document that represents this filing: for a correction group db/006 resolved
-- it is that group's latest valid filing, otherwise the filing itself.
CREATE TABLE IF NOT EXISTS corporate_event_members (
    event_id text NOT NULL REFERENCES corporate_events(event_id) ON DELETE CASCADE,
    doc_id text NOT NULL REFERENCES disclosures(doc_id),
    canonical_doc_id text NOT NULL REFERENCES disclosures(doc_id),
    member_role text NOT NULL,
    member_order integer NOT NULL,
    event_date date,
    root_doc_id text REFERENCES disclosures(doc_id),
    correction_group_id text,
    correction_resolution_status text,
    correction_chain jsonb NOT NULL DEFAULT '[]'::jsonb,
    is_correction boolean NOT NULL DEFAULT false,
    confidence double precision NOT NULL DEFAULT 0.0,
    provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, doc_id),
    CONSTRAINT corporate_event_members_role_check
        CHECK (member_role IN ('contract', 'contract_update', 'termination')),
    CONSTRAINT corporate_event_members_correction_status_check
        CHECK (
            correction_resolution_status IS NULL
            OR correction_resolution_status IN ('resolved', 'ambiguous', 'unresolved')
        ),
    CONSTRAINT corporate_event_members_confidence_check
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT corporate_event_members_order_check
        CHECK (member_order >= 0),
    CONSTRAINT corporate_event_members_chain_json_check
        CHECK (jsonb_typeof(correction_chain) = 'array'),
    CONSTRAINT corporate_event_members_correction_metadata_check
        CHECK (
            (
                correction_group_id IS NULL
                AND correction_resolution_status IS NULL
                AND jsonb_array_length(correction_chain) = 0
                AND (root_doc_id IS NULL OR root_doc_id = doc_id)
            )
            OR (
                correction_group_id IS NOT NULL
                AND correction_resolution_status IS NOT NULL
                AND (
                    jsonb_array_length(correction_chain) = 0
                    OR (
                        root_doc_id IS NOT NULL
                        AND correction_chain ? root_doc_id
                        AND correction_chain ? doc_id
                    )
                )
            )
        ),
    -- P0-A ambiguous/unresolved groups do have a latest row internally, but it
    -- is not a verified latest and therefore cannot replace the input filing.
    CONSTRAINT corporate_event_members_unverified_canonical_check
        CHECK (
            correction_resolution_status = 'resolved'
            OR canonical_doc_id = doc_id
        )
);

-- A disclosure belongs to at most one lifecycle. This makes the "one follow-up
-- document resolved into several unrelated events" failure unstorable rather
-- than merely absent.
CREATE UNIQUE INDEX IF NOT EXISTS uq_corporate_event_members_doc
    ON corporate_event_members (doc_id);
CREATE INDEX IF NOT EXISTS idx_corporate_event_members_event
    ON corporate_event_members (event_id, member_order);
CREATE INDEX IF NOT EXISTS idx_corporate_event_members_role
    ON corporate_event_members (member_role);
CREATE INDEX IF NOT EXISTS idx_corporate_event_members_canonical
    ON corporate_event_members (canonical_doc_id);
CREATE INDEX IF NOT EXISTS idx_corporate_event_members_correction_group
    ON corporate_event_members (correction_group_id)
    WHERE correction_group_id IS NOT NULL;
-- Membership is stored per logical contract, so a superseded filing has no row
-- of its own. Lookup still has to answer for it, and it does so through the
-- chain and the collapsed id list this row already carries. These two indexes
-- keep that read path from degrading into a sequential scan. They exist for
-- reads only: no raw filing is written back as a second member row.
-- Default ``jsonb_ops`` deliberately: ``jsonb_path_ops`` indexes ``@>`` only and
-- would not serve the ``?`` existence test these lookups use.
CREATE INDEX IF NOT EXISTS idx_corporate_event_members_chain_alias
    ON corporate_event_members USING gin (correction_chain);
CREATE INDEX IF NOT EXISTS idx_corporate_event_members_collapsed_alias
    ON corporate_event_members USING gin ((provenance -> 'collapsed_doc_ids'));
-- ``corporate_event_members`` intentionally does not duplicate corp_code. This
-- trigger compares the company of the event, source disclosure, canonical P0-A
-- representative, and optional correction root before the row is accepted.
CREATE OR REPLACE FUNCTION enforce_corporate_event_member_company()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    event_corp text;
    document_corp text;
    canonical_corp text;
    root_corp text;
BEGIN
    SELECT corp_code INTO event_corp
    FROM corporate_events WHERE event_id = NEW.event_id;
    SELECT corp_code INTO document_corp
    FROM disclosures WHERE doc_id = NEW.doc_id;
    SELECT corp_code INTO canonical_corp
    FROM disclosures WHERE doc_id = NEW.canonical_doc_id;
    IF NEW.root_doc_id IS NOT NULL THEN
        SELECT corp_code INTO root_corp
        FROM disclosures WHERE doc_id = NEW.root_doc_id;
    END IF;
    IF event_corp IS DISTINCT FROM document_corp
       OR event_corp IS DISTINCT FROM canonical_corp
       OR (NEW.root_doc_id IS NOT NULL AND event_corp IS DISTINCT FROM root_corp)
    THEN
        RAISE EXCEPTION 'cross-company corporate event membership is not allowed';
    END IF;
    RETURN NEW;
END
$$;

-- The three columns hashed into event_id are immutable. An unresolved event
-- that later finds its opening contract is inserted under a new id instead of
-- mutating the meaning of an existing primary key.
CREATE OR REPLACE FUNCTION enforce_corporate_event_identity_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.corp_code IS DISTINCT FROM OLD.corp_code
       OR NEW.event_family IS DISTINCT FROM OLD.event_family
       OR NEW.root_logical_key IS DISTINCT FROM OLD.root_logical_key
    THEN
        RAISE EXCEPTION 'corporate event identity columns are immutable';
    END IF;
    RETURN NEW;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = 'corporate_events'::regclass
          AND tgname = 'trg_corporate_event_identity_immutable'
    ) THEN
        CREATE TRIGGER trg_corporate_event_identity_immutable
        BEFORE UPDATE ON corporate_events
        FOR EACH ROW EXECUTE FUNCTION enforce_corporate_event_identity_immutable();
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = 'corporate_event_members'::regclass
          AND tgname = 'trg_corporate_event_member_company'
    ) THEN
        CREATE TRIGGER trg_corporate_event_member_company
        BEFORE INSERT OR UPDATE ON corporate_event_members
        FOR EACH ROW EXECUTE FUNCTION enforce_corporate_event_member_company();
    END IF;
END
$$;

-- One document-to-document edge. ``target_doc_id`` stays NULL while the edge is
-- ambiguous or unresolved, which is how a termination whose contract is not in
-- the corpus keeps its evidence without inventing a target.
CREATE TABLE IF NOT EXISTS corporate_event_relations (
    relation_id text PRIMARY KEY,
    event_id text REFERENCES corporate_events(event_id) ON DELETE CASCADE,
    source_doc_id text NOT NULL REFERENCES disclosures(doc_id),
    target_doc_id text REFERENCES disclosures(doc_id),
    relation_type text NOT NULL,
    resolution_status text NOT NULL,
    resolution_source text NOT NULL,
    confidence double precision NOT NULL DEFAULT 0.0,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT corporate_event_relations_type_check
        CHECK (relation_type IN ('belongs_to_event', 'terminates_event')),
    CONSTRAINT corporate_event_relations_status_check
        CHECK (resolution_status IN ('resolved', 'ambiguous', 'unresolved')),
    CONSTRAINT corporate_event_relations_confidence_check
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    -- A document never relates to itself.
    CONSTRAINT corporate_event_relations_no_self_reference
        CHECK (target_doc_id IS NULL OR target_doc_id <> source_doc_id),
    -- Only a resolved edge names a target.
    CONSTRAINT corporate_event_relations_resolution_check
        CHECK ((resolution_status = 'resolved') = (target_doc_id IS NOT NULL))
);

-- Re-running the build updates this row instead of inserting a second edge.
CREATE UNIQUE INDEX IF NOT EXISTS uq_corporate_event_relations_edge
    ON corporate_event_relations (
        source_doc_id, relation_type, coalesce(target_doc_id, '')
    );
CREATE INDEX IF NOT EXISTS idx_corporate_event_relations_target
    ON corporate_event_relations (target_doc_id)
    WHERE target_doc_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_corporate_event_relations_event
    ON corporate_event_relations (event_id);
CREATE INDEX IF NOT EXISTS idx_corporate_event_relations_status
    ON corporate_event_relations (resolution_status);

CREATE OR REPLACE FUNCTION enforce_corporate_event_relation_company()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    event_corp text;
    source_corp text;
    target_corp text;
BEGIN
    IF NEW.event_id IS NOT NULL THEN
        SELECT corp_code INTO event_corp
        FROM corporate_events WHERE event_id = NEW.event_id;
    END IF;
    SELECT corp_code INTO source_corp
    FROM disclosures WHERE doc_id = NEW.source_doc_id;
    IF NEW.target_doc_id IS NOT NULL THEN
        SELECT corp_code INTO target_corp
        FROM disclosures WHERE doc_id = NEW.target_doc_id;
    END IF;
    IF (NEW.event_id IS NOT NULL AND event_corp IS DISTINCT FROM source_corp)
       OR (NEW.target_doc_id IS NOT NULL AND source_corp IS DISTINCT FROM target_corp)
    THEN
        RAISE EXCEPTION 'cross-company corporate event relation is not allowed';
    END IF;
    RETURN NEW;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = 'corporate_event_relations'::regclass
          AND tgname = 'trg_corporate_event_relation_company'
    ) THEN
        CREATE TRIGGER trg_corporate_event_relation_company
        BEFORE INSERT OR UPDATE ON corporate_event_relations
        FOR EACH ROW EXECUTE FUNCTION enforce_corporate_event_relation_company();
    END IF;
END
$$;

COMMIT;

-- Read-only invariants. A healthy graph reports zero for every count below.
SELECT
    (
        SELECT count(*) FROM corporate_event_relations r
        WHERE r.target_doc_id = r.source_doc_id
    ) AS self_relations,
    (
        SELECT count(*) FROM (
            SELECT doc_id FROM corporate_event_members
            GROUP BY doc_id HAVING count(*) > 1
        ) AS duplicated
    ) AS duplicate_memberships,
    (
        SELECT count(*) FROM (
            SELECT source_doc_id, relation_type, coalesce(target_doc_id, '') AS target
            FROM corporate_event_relations
            GROUP BY 1, 2, 3
            HAVING count(*) > 1
        ) AS duplicated
    ) AS duplicate_relations,
    -- Every rule keys on corp_code, so a member whose company differs from its
    -- event's company means the graph was not produced by this builder.
    (
        SELECT count(*)
        FROM corporate_event_members m
        JOIN corporate_events e ON e.event_id = m.event_id
        JOIN disclosures d ON d.doc_id = m.doc_id
        WHERE d.corp_code <> e.corp_code
    ) AS cross_company_members,
    (
        SELECT count(*)
        FROM corporate_event_relations r
        JOIN disclosures s ON s.doc_id = r.source_doc_id
        JOIN disclosures t ON t.doc_id = r.target_doc_id
        WHERE s.corp_code <> t.corp_code
    ) AS cross_company_relations,
    (
        SELECT count(*)
        FROM corporate_events e
        WHERE e.member_count <> (
            SELECT count(*) FROM corporate_event_members m
            WHERE m.event_id = e.event_id
        )
    ) AS member_count_mismatches,
    (
        SELECT count(*)
        FROM corporate_events e
        WHERE e.lifecycle_status = 'terminated'
          AND NOT EXISTS (
              SELECT 1 FROM corporate_event_members m
              WHERE m.event_id = e.event_id AND m.member_role = 'termination'
          )
    ) AS terminated_events_without_termination;
