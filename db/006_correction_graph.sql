\set ON_ERROR_STOP on

-- Additive correction graph. The two tables below are derived from frozen
-- disclosure metadata and from the correction notice each correcting document
-- already carries, so nothing here rewrites chunk identity, content,
-- retrieval_text, ordering, provenance, or embeddings. Applying this migration
-- never requires re-chunking or re-embedding.
--
-- The same file serves a fresh install and an existing upgrade: every object is
-- created with IF NOT EXISTS and no existing table is altered or dropped.
BEGIN;

-- One row per correcting disclosure. ``target_doc_id`` is the document it
-- corrects; it stays NULL while the relation is ambiguous or unresolved.
CREATE TABLE IF NOT EXISTS correction_relations (
    relation_id text PRIMARY KEY,
    source_doc_id text NOT NULL REFERENCES disclosures(doc_id),
    target_doc_id text REFERENCES disclosures(doc_id),
    relation_type text NOT NULL DEFAULT 'correction_of',
    resolution_status text NOT NULL,
    resolution_source text NOT NULL,
    confidence double precision NOT NULL DEFAULT 0.0,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT correction_relations_type_check
        CHECK (relation_type IN ('correction_of')),
    CONSTRAINT correction_relations_status_check
        CHECK (resolution_status IN ('resolved', 'ambiguous', 'unresolved')),
    CONSTRAINT correction_relations_confidence_check
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    -- A document can never correct itself.
    CONSTRAINT correction_relations_no_self_reference
        CHECK (target_doc_id IS NULL OR target_doc_id <> source_doc_id),
    -- Only a resolved relation names a target.
    CONSTRAINT correction_relations_resolution_check
        CHECK ((resolution_status = 'resolved') = (target_doc_id IS NOT NULL))
);

-- Re-running the backfill updates this row instead of inserting a second edge.
CREATE UNIQUE INDEX IF NOT EXISTS uq_correction_relations_edge
    ON correction_relations (
        source_doc_id, relation_type, coalesce(target_doc_id, '')
    );
CREATE INDEX IF NOT EXISTS idx_correction_relations_target
    ON correction_relations (target_doc_id)
    WHERE target_doc_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_correction_relations_status
    ON correction_relations (resolution_status);

-- One row per disclosure that takes part in a correction. Documents with no
-- correction at all are deliberately absent, so ordinary retrieval is
-- unaffected by this table.
CREATE TABLE IF NOT EXISTS correction_group_members (
    doc_id text PRIMARY KEY REFERENCES disclosures(doc_id),
    correction_group_id text NOT NULL,
    root_doc_id text NOT NULL REFERENCES disclosures(doc_id),
    parent_doc_id text REFERENCES disclosures(doc_id),
    correction_order integer NOT NULL,
    is_latest boolean NOT NULL,
    is_correction boolean NOT NULL,
    resolution_status text NOT NULL,
    resolution_source text NOT NULL,
    confidence double precision NOT NULL DEFAULT 0.0,
    evidence jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT correction_group_members_status_check
        CHECK (resolution_status IN ('resolved', 'ambiguous', 'unresolved')),
    CONSTRAINT correction_group_members_confidence_check
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT correction_group_members_order_check
        CHECK (correction_order >= 0),
    CONSTRAINT correction_group_members_no_self_parent
        CHECK (parent_doc_id IS NULL OR parent_doc_id <> doc_id)
);

-- Chain shape, in both directions: the document at order 0 is the original and
-- has no parent, and every later document in a chain names the document it
-- supersedes. Without the second half a row such as
-- (correction_order = 3, parent_doc_id = NULL, resolution_status = 'resolved')
-- would be storable. Added out of line so a database that already ran an
-- earlier revision of this file gains the constraint too; the table is derived,
-- so a failure here means the graph must be rebuilt rather than patched.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'correction_group_members'::regclass
          AND conname = 'correction_group_members_chain_parent'
    ) THEN
        ALTER TABLE correction_group_members
            ADD CONSTRAINT correction_group_members_chain_parent
            CHECK (
                (correction_order = 0 AND parent_doc_id IS NULL)
                OR (correction_order > 0 AND parent_doc_id IS NOT NULL)
            );
    END IF;
END
$$;

-- A resolved group has exactly one final valid document. This index makes a
-- second one impossible to store.
CREATE UNIQUE INDEX IF NOT EXISTS uq_correction_group_members_latest
    ON correction_group_members (correction_group_id)
    WHERE is_latest;
CREATE INDEX IF NOT EXISTS idx_correction_group_members_chain
    ON correction_group_members (correction_group_id, correction_order);
CREATE INDEX IF NOT EXISTS idx_correction_group_members_root
    ON correction_group_members (root_doc_id, correction_order);
CREATE INDEX IF NOT EXISTS idx_correction_group_members_parent
    ON correction_group_members (parent_doc_id)
    WHERE parent_doc_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_correction_group_members_status
    ON correction_group_members (resolution_status);

COMMIT;

-- Read-only invariants. A healthy graph reports zero for every count below.
SELECT
    (
        SELECT count(*) FROM correction_relations r
        WHERE r.target_doc_id = r.source_doc_id
    ) AS self_reference_relations,
    (
        SELECT count(*) FROM correction_group_members m
        WHERE m.parent_doc_id = m.doc_id
    ) AS self_reference_members,
    (
        SELECT count(*) FROM (
            SELECT correction_group_id
            FROM correction_group_members
            GROUP BY correction_group_id
            HAVING count(*) FILTER (WHERE is_latest) <> 1
        ) AS invalid
    ) AS groups_without_exactly_one_latest,
    (
        SELECT count(*) FROM (
            SELECT source_doc_id, relation_type, coalesce(target_doc_id, '') AS target
            FROM correction_relations
            GROUP BY 1, 2, 3
            HAVING count(*) > 1
        ) AS duplicated
    ) AS duplicate_relations,
    (
        SELECT count(*) FROM correction_group_members m
        WHERE (m.correction_order = 0) <> (m.parent_doc_id IS NULL)
    ) AS broken_chain_parents,
    (
        SELECT count(*) FROM correction_group_members m
        WHERE m.resolution_status <> 'resolved'
          AND EXISTS (
              SELECT 1 FROM correction_group_members other
              WHERE other.correction_group_id = m.correction_group_id
                AND other.doc_id <> m.doc_id
          )
    ) AS unresolved_members_inside_a_group;
