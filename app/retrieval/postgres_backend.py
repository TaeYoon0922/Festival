"""PostgreSQL metadata and lexical retrieval backend.

The backend implements the contracts in :mod:`app.retrieval.interfaces` while
keeping database credentials outside the source tree.  PostgreSQL full-text
search is used as the lexical ranker; the frozen ``content`` field remains the
evidence returned to callers.
"""

from __future__ import annotations

import math
import os
import re
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import psycopg
from psycopg.rows import dict_row

from app.retrieval.interfaces import (
    CandidateChunk,
    CandidateDocument,
    MetadataMatch,
    RetrievalResult,
)
from app.retrieval.vector import VectorRetrievalResult


ConnectionFactory = Callable[..., Any]


def _normalized(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def _values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {_normalized(value)}
    return {_normalized(item) for item in value}


def _integers(value: int | Sequence[int] | None) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


class PostgresBackend:
    """PostgreSQL implementation of metadata, chunk, and retrieval contracts.

    ``DATABASE_URL`` takes precedence when present.  Otherwise psycopg receives
    the standard ``PGHOST``, ``PGPORT``, ``PGDATABASE``, ``PGUSER``, and
    ``PGPASSWORD`` values.  A new connection is opened and closed for each
    public operation; callers can also use :meth:`connection` directly.
    """

    _COMPANY_COLUMNS = """
        co.corp_code,
        co.stock_code,
        co.corp_name,
        co.listed_name,
        co.market,
        co.industry,
        co.sector,
        co.metadata AS company_metadata
    """

    _DISCLOSURE_COLUMNS = """
        d.doc_id,
        d.corp_code,
        d.rcept_no,
        d.report_nm,
        d.rcept_dt,
        d.doc_group,
        d.doc_subtype,
        d.is_correction,
        d.base_year,
        d.base_month,
        d.file_path,
        d.file_format,
        d.metadata AS disclosure_metadata,
        co.stock_code,
        co.corp_name,
        co.listed_name,
        co.market,
        co.industry,
        co.sector,
        co.metadata AS company_metadata
    """

    def __init__(
        self,
        dsn: str | None = None,
        *,
        connection_factory: ConnectionFactory = psycopg.connect,
        connect_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self._dsn = dsn if dsn is not None else os.getenv("DATABASE_URL") or None
        self._connection_factory = connection_factory
        environment = {} if self._dsn else {
            "host": os.getenv("PGHOST"),
            "port": os.getenv("PGPORT"),
            "dbname": os.getenv("PGDATABASE"),
            "user": os.getenv("PGUSER"),
            "password": os.getenv("PGPASSWORD"),
        }
        self._connect_kwargs = {
            key: value for key, value in environment.items() if value not in (None, "")
        }
        if connect_kwargs:
            self._connect_kwargs.update(connect_kwargs)

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """Yield one connection and always close it after the operation."""

        if self._dsn:
            connection = self._connection_factory(self._dsn, **self._connect_kwargs)
        else:
            connection = self._connection_factory(**self._connect_kwargs)
        try:
            yield connection
        finally:
            connection.close()

    def _fetch_all(self, query: str, params: Sequence[Any]) -> list[dict[str, Any]]:
        with self.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(query, tuple(params))
                return [dict(row) for row in cursor.fetchall()]

    def _fetch_one(self, query: str, params: Sequence[Any]) -> dict[str, Any] | None:
        with self.connection() as connection:
            with connection.cursor(row_factory=dict_row) as cursor:
                cursor.execute(query, tuple(params))
                row = cursor.fetchone()
                return dict(row) if row is not None else None

    def _execute_many(
        self, query: str, params: Sequence[Sequence[Any]]
    ) -> int:
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.executemany(query, [tuple(row) for row in params])
                affected = int(cursor.rowcount)
            connection.commit()
        return affected

    @staticmethod
    def _company_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
        metadata = _mapping(row.get("company_metadata"))
        metadata.update(
            {
                key: _json_safe(value)
                for key, value in row.items()
                if key != "company_metadata"
            }
        )
        metadata["metadata"] = _mapping(row.get("company_metadata"))
        return metadata

    @staticmethod
    def _disclosure_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
        metadata = _mapping(row.get("disclosure_metadata"))
        metadata.update(
            {
                key: _json_safe(value)
                for key, value in row.items()
                if key not in {"company_metadata", "disclosure_metadata"}
            }
        )
        metadata["metadata"] = _mapping(row.get("disclosure_metadata"))
        metadata["company_metadata"] = _mapping(row.get("company_metadata"))
        return metadata

    def get_company(self, corp_code: str) -> dict[str, Any] | None:
        """Return one company by exact ``corp_code``."""

        row = self._fetch_one(
            f"""
            SELECT {self._COMPANY_COLUMNS}
            FROM companies co
            WHERE co.corp_code = %s
            """,
            [corp_code],
        )
        return self._company_metadata(row) if row is not None else None

    def resolve_company(self, company: str) -> dict[str, Any] | None:
        """Resolve an exact name/alias, or the longest company name in query text."""

        normalized = _normalized(company)
        if not normalized:
            return None
        corp_name = "lower(regexp_replace(co.corp_name, '[[:space:]]+', '', 'g'))"
        listed_name = (
            "lower(regexp_replace(coalesce(co.listed_name, ''), "
            "'[[:space:]]+', '', 'g'))"
        )
        row = self._fetch_one(
            f"""
            SELECT {self._COMPANY_COLUMNS}
            FROM companies co
            WHERE {corp_name} = %s
               OR {listed_name} = %s
               OR ({corp_name} <> '' AND strpos(%s, {corp_name}) > 0)
               OR ({listed_name} <> '' AND strpos(%s, {listed_name}) > 0)
            ORDER BY
                CASE WHEN {corp_name} = %s OR {listed_name} = %s THEN 0 ELSE 1 END,
                greatest(length({corp_name}), length({listed_name})) DESC,
                co.corp_code
            LIMIT 1
            """,
            [normalized, normalized, normalized, normalized, normalized, normalized],
        )
        return self._company_metadata(row) if row is not None else None

    def filter_disclosures(
        self,
        company: str | Sequence[str] | None = None,
        year: int | Sequence[int] | None = None,
        period: int | tuple[int, int] | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        is_correction: bool | None = None,
        *,
        corp_code: str | Sequence[str] | None = None,
        section_path: str | None = None,
    ) -> list[CandidateDocument]:
        """Apply hard filters in SQL and annotate the existing soft metadata."""

        companies = _values(company)
        corp_codes = _values(corp_code)
        years = _integers(year)
        period_year: int | None = None
        period_month: int | None = None
        if isinstance(period, tuple):
            period_year, period_month = int(period[0]), int(period[1])
        elif isinstance(period, int):
            period_month = period

        conditions: list[str] = []
        params: list[Any] = []
        if companies:
            normalized_names = sorted(companies)
            conditions.append(
                "(" 
                "lower(regexp_replace(co.corp_name, '[[:space:]]+', '', 'g')) = ANY(%s) "
                "OR lower(regexp_replace(coalesce(co.listed_name, ''), "
                "'[[:space:]]+', '', 'g')) = ANY(%s)"
                ")"
            )
            params.extend([normalized_names, normalized_names])
        if corp_codes:
            conditions.append(
                "lower(regexp_replace(d.corp_code, '[[:space:]]+', '', 'g')) = ANY(%s)"
            )
            params.append(sorted(corp_codes))
        if years:
            conditions.append("d.base_year = ANY(%s)")
            params.append(sorted(years))
        if period_year is not None:
            conditions.append("d.base_year = %s")
            params.append(period_year)
        if period_month is not None:
            conditions.append("d.base_month = %s")
            params.append(period_month)

        where = " AND ".join(conditions) if conditions else "TRUE"
        rows = self._fetch_all(
            f"""
            SELECT {self._DISCLOSURE_COLUMNS}
            FROM disclosures d
            JOIN companies co ON co.corp_code = d.corp_code
            WHERE {where}
            ORDER BY d.doc_id
            """,
            params,
        )

        hard = {
            "company": sorted(companies),
            "corp_code": sorted(corp_codes),
            "year": sorted(years),
            "period": period,
        }
        soft_inputs = {
            key: value
            for key, value in {
                "doc_group": doc_group,
                "doc_subtype": doc_subtype,
                "is_correction": is_correction,
                "section_path": section_path,
            }.items()
            if value is not None
        }
        candidates: list[CandidateDocument] = []
        for row in rows:
            soft = {
                "doc_group": doc_group is not None and row.get("doc_group") == doc_group,
                "doc_subtype": doc_subtype is not None
                and row.get("doc_subtype") == doc_subtype,
                "is_correction": is_correction is not None
                and bool(row.get("is_correction")) is is_correction,
                "section_path": False,
            }
            active_soft = {
                key: value for key, value in soft.items() if key in soft_inputs
            }
            candidates.append(
                CandidateDocument(
                    doc_id=str(row["doc_id"]),
                    metadata=self._disclosure_metadata(row),
                    metadata_match=MetadataMatch(
                        hard_filters={key: value for key, value in hard.items() if value},
                        soft_boosts=active_soft,
                        soft_inputs=soft_inputs,
                        soft_score=float(sum(active_soft.values())),
                    ),
                )
            )
        return candidates

    def fetch_documents(self, doc_ids: Sequence[str]) -> list[CandidateDocument]:
        """Fetch disclosures by identity, ignoring every query-level filter.

        Correction expansion needs the other documents of a chain even though
        they fall outside the question's own metadata window: a question anchored
        on the original's receipt date must still reach a correction filed years
        later.  Selection stays deterministic because the identifiers come from
        the correction graph, not from the query.
        """

        unique = sorted({str(doc_id) for doc_id in doc_ids if str(doc_id)})
        if not unique:
            return []
        rows = self._fetch_all(
            f"""
            SELECT {self._DISCLOSURE_COLUMNS}
            FROM disclosures d
            JOIN companies co ON co.corp_code = d.corp_code
            WHERE d.doc_id = ANY(%s)
            ORDER BY d.doc_id
            """,
            [unique],
        )
        return [
            CandidateDocument(
                doc_id=str(row["doc_id"]),
                metadata=self._disclosure_metadata(row),
                metadata_match=MetadataMatch(),
            )
            for row in rows
        ]

    def enumerate_disclosures(
        self,
        *,
        corp_code: str | Sequence[str],
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> tuple[CandidateDocument, ...]:
        """List every disclosure matching an exact metadata definition.

        This is the Tier 2 enumeration primitive, for the document families the
        P0-B event timeline does not model.  It differs from
        :meth:`filter_disclosures` in the one way that matters for counting:
        ``doc_group``, ``doc_subtype``, and the receipt-date range are applied as
        SQL ``WHERE`` predicates, so the result *is* the set.  In
        ``filter_disclosures`` those three are soft boosts that only influence
        ranking, and ``year`` there means ``base_year`` -- a column that is NULL
        for every exchange, major, and holding filing in this corpus, so a year
        filter cannot select them at all.

        That method keeps its behaviour untouched: the frozen Gold60 ranking
        depends on those soft-boost semantics, and this is an additive sibling
        rather than a change to it.

        ``date_from``/``date_to`` are half-open (``>= from``, ``< to``) and apply
        to ``rcept_dt``, which is the only date this table carries.  A caller
        asking when a *contract* was signed wants
        :meth:`PostgresCorporateEventRepository.enumerate_events` instead; the
        two dates disagree often enough that substituting one for the other
        changes the answer.

        One statement, ordered by ``doc_id`` so repeated calls agree.
        """

        # ``_values`` normalizes but keeps blanks; an empty identifier must not
        # become a predicate that matches the empty string.
        corp_codes = {value for value in _values(corp_code) if value}
        if not corp_codes:
            raise ValueError("corp_code is required for enumeration")
        if date_from and date_to and str(date_from) > str(date_to):
            raise ValueError("date_from must not be after date_to")

        # ``_values`` already normalized the input, and every stored corp_code is
        # verifiably in that same form (0 of 4,204 disclosures and 0 of 70
        # companies differ).  Comparing the column directly rather than wrapping
        # it in ``regexp_replace`` keeps ``idx_disclosures_company_date``
        # (corp_code, rcept_dt) usable as an index condition instead of
        # degrading it to a post-scan filter.
        conditions = ["d.corp_code = ANY(%s)"]
        params: list[Any] = [sorted(corp_codes)]
        if doc_group is not None:
            conditions.append("d.doc_group = %s")
            params.append(str(doc_group))
        if doc_subtype is not None:
            conditions.append("d.doc_subtype = %s")
            params.append(str(doc_subtype))
        if date_from:
            conditions.append("d.rcept_dt >= %s::date")
            params.append(str(date_from))
        if date_to:
            conditions.append("d.rcept_dt < %s::date")
            params.append(str(date_to))

        rows = self._fetch_all(
            f"""
            SELECT {self._DISCLOSURE_COLUMNS}
            FROM disclosures d
            JOIN companies co ON co.corp_code = d.corp_code
            WHERE {" AND ".join(conditions)}
            ORDER BY d.doc_id
            """,
            params,
        )
        # Every predicate above was a hard filter, so there is no soft score to
        # report: an enumerated document matched exactly or is absent.
        return tuple(
            CandidateDocument(
                doc_id=str(row["doc_id"]),
                metadata=self._disclosure_metadata(row),
                metadata_match=MetadataMatch(),
            )
            for row in rows
        )

    def get_candidate_documents(
        self,
        company: str | Sequence[str] | None = None,
        year: int | Sequence[int] | None = None,
        period: int | tuple[int, int] | None = None,
        doc_group: str | None = None,
        doc_subtype: str | None = None,
        is_correction: bool | None = None,
        *,
        corp_code: str | Sequence[str] | None = None,
        section_path: str | None = None,
    ) -> list[CandidateDocument]:
        return self.filter_disclosures(
            company,
            year,
            period,
            doc_group,
            doc_subtype,
            is_correction,
            corp_code=corp_code,
            section_path=section_path,
        )

    def fetch_chunks(
        self, documents: Iterable[CandidateDocument]
    ) -> list[CandidateChunk]:
        document_list = list(documents)
        if not document_list:
            return []
        documents_by_id = {document.doc_id: document for document in document_list}
        rows = self._fetch_all(
            """
            SELECT
                c.chunk_id,
                c.doc_id,
                c.source_part_id,
                c.section_id,
                c.table_id,
                c.chunk_type,
                c.chunk_order,
                c.content,
                c.retrieval_text,
                c.char_count,
                c.retrieval_priority,
                c.metadata AS chunk_metadata,
                s.section_title,
                s.section_path,
                d.doc_group,
                d.doc_subtype,
                d.report_nm,
                d.rcept_dt,
                d.is_correction,
                co.corp_code,
                co.corp_name,
                co.listed_name
            FROM chunks c
            JOIN disclosures d ON d.doc_id = c.doc_id
            JOIN companies co ON co.corp_code = d.corp_code
            JOIN sections s
              ON s.source_part_id = c.source_part_id
             AND s.section_id = c.section_id
            WHERE c.doc_id = ANY(%s)
              AND coalesce(c.metadata ->> 'is_indexable', 'true') = 'true'
            ORDER BY c.doc_id, c.chunk_order, c.chunk_id
            """,
            [sorted(documents_by_id)],
        )

        candidates: list[CandidateChunk] = []
        for row in rows:
            document = documents_by_id.get(str(row["doc_id"]))
            if document is None:
                continue
            chunk = _mapping(row.get("chunk_metadata"))
            chunk.update(
                {
                    key: _json_safe(value)
                    for key, value in row.items()
                    if key != "chunk_metadata"
                }
            )
            section_path_value = chunk.get("section_path") or []
            if not isinstance(section_path_value, list):
                section_path_value = list(section_path_value)
            chunk["section_path"] = section_path_value
            chunk["evaluation_text"] = "\n".join(
                [
                    f"[기업명] {chunk.get('corp_name') or ''}",
                    f"[공시명] {chunk.get('report_nm') or ''}",
                    f"[Section Path] {' > '.join(str(item) for item in section_path_value)}",
                    "",
                    str(chunk.get("content") or ""),
                ]
            )

            soft = dict(document.metadata_match.soft_boosts)
            requested_path = document.metadata_match.soft_inputs.get("section_path")
            if requested_path:
                path = " > ".join(str(item) for item in section_path_value)
                soft["section_path"] = _normalized(requested_path) in _normalized(path)
            match = MetadataMatch(
                hard_filters=document.metadata_match.hard_filters,
                soft_boosts=soft,
                soft_inputs=document.metadata_match.soft_inputs,
                soft_score=float(sum(soft.values())),
            )
            candidates.append(
                CandidateChunk(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(row["doc_id"]),
                    chunk=chunk,
                    metadata_match=match,
                )
            )
        return candidates

    def get_candidate_chunks(
        self, documents: Iterable[CandidateDocument]
    ) -> list[CandidateChunk]:
        return self.fetch_chunks(documents)

    def fetch_embedding_source_chunks(
        self, chunk_ids: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Fetch only frozen retrieval text and identifying metadata for embedding."""

        unique_ids = sorted({str(chunk_id) for chunk_id in chunk_ids if str(chunk_id)})
        if not unique_ids:
            return []
        return self._fetch_all(
            """
            SELECT
                c.chunk_id,
                c.doc_id,
                d.corp_code,
                co.corp_name,
                d.doc_group,
                c.chunk_type,
                c.retrieval_text
            FROM chunks c
            JOIN disclosures d ON d.doc_id = c.doc_id
            JOIN companies co ON co.corp_code = d.corp_code
            WHERE c.chunk_id = ANY(%s)
              AND coalesce(c.metadata ->> 'is_indexable', 'true') = 'true'
            ORDER BY c.chunk_id
            """,
            [unique_ids],
        )

    def existing_embedding_chunk_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        embedding_model: str,
        embedding_version: str,
        embedding_dimensions: int,
    ) -> set[str]:
        """Return scoped rows already persisted for resume and coverage checks."""

        unique_ids = sorted({str(chunk_id) for chunk_id in chunk_ids if str(chunk_id)})
        if not unique_ids:
            return set()
        if not embedding_model.strip() or not embedding_version.strip():
            raise ValueError("embedding model and version must not be empty")
        if not 1 <= int(embedding_dimensions) <= 2000:
            raise ValueError("embedding dimensions must be between 1 and 2000")
        rows = self._fetch_all(
            """
            SELECT chunk_id, embedding_dimensions
            FROM chunk_embeddings
            WHERE chunk_id = ANY(%s)
              AND embedding_model = %s
              AND embedding_version = %s
            """,
            [
                unique_ids,
                embedding_model,
                embedding_version,
            ],
        )
        mismatches = [
            str(row["chunk_id"])
            for row in rows
            if int(row["embedding_dimensions"]) != int(embedding_dimensions)
        ]
        if mismatches:
            raise ValueError(
                "existing embedding dimension mismatch for model/version: "
                + ", ".join(mismatches[:5])
            )
        return {str(row["chunk_id"]) for row in rows}

    def upsert_embeddings(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        embedding_model: str,
        embedding_version: str,
        force: bool = False,
    ) -> int:
        """Insert one committed batch; overwrite only when explicitly forced."""

        if not records:
            return 0
        if not embedding_model.strip() or not embedding_version.strip():
            raise ValueError("embedding model and version must not be empty")
        params: list[tuple[Any, ...]] = []
        for record in records:
            chunk_id = str(record.get("chunk_id") or "").strip()
            embedding = [float(value) for value in record.get("embedding") or []]
            dimensions = int(record.get("embedding_dimensions") or len(embedding))
            if not chunk_id:
                raise ValueError("embedding record chunk_id must not be empty")
            if not 1 <= dimensions <= 2000 or len(embedding) != dimensions:
                raise ValueError("embedding record dimension mismatch")
            if any(not math.isfinite(value) for value in embedding):
                raise ValueError("embedding values must be finite")
            vector_literal = (
                "[" + ",".join(format(value, ".17g") for value in embedding) + "]"
            )
            params.append(
                (
                    chunk_id,
                    embedding_model,
                    embedding_version,
                    dimensions,
                    vector_literal,
                )
            )
        conflict = (
            "DO UPDATE SET embedding_dimensions = EXCLUDED.embedding_dimensions, "
            "embedding = EXCLUDED.embedding, created_at = now()"
            if force
            else "DO NOTHING"
        )
        return self._execute_many(
            f"""
            INSERT INTO chunk_embeddings (
                chunk_id,
                embedding_model,
                embedding_version,
                embedding_dimensions,
                embedding
            ) VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (chunk_id, embedding_model, embedding_version)
            {conflict}
            """,
            params,
        )

    def lexical_search(
        self,
        query: str,
        candidates: Sequence[CandidateChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Rank candidate chunks with PostgreSQL FTS over ``retrieval_text``."""

        if not candidates or not query.strip() or (top_k is not None and top_k <= 0):
            return []
        candidates_by_id = {candidate.chunk_id: candidate for candidate in candidates}
        params: list[Any] = [query, sorted(candidates_by_id)]
        limit = ""
        if top_k is not None:
            limit = "LIMIT %s"
            params.append(int(top_k))
        rows = self._fetch_all(
            f"""
            WITH lexical_query AS (
                SELECT websearch_to_tsquery('simple', %s) AS value
            )
            SELECT
                c.chunk_id,
                c.doc_id,
                ts_rank_cd(
                    to_tsvector('simple', c.retrieval_text),
                    lexical_query.value
                ) AS bm25_score
            FROM chunks c
            CROSS JOIN lexical_query
            WHERE c.chunk_id = ANY(%s)
              AND lexical_query.value @@ to_tsvector('simple', c.retrieval_text)
            ORDER BY bm25_score DESC, c.chunk_id
            {limit}
            """,
            params,
        )
        results: list[RetrievalResult] = []
        for rank, row in enumerate(rows, start=1):
            candidate = candidates_by_id.get(str(row["chunk_id"]))
            if candidate is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=candidate.chunk_id,
                    doc_id=candidate.doc_id,
                    bm25_score=float(row["bm25_score"]),
                    rank=rank,
                    metadata_match=candidate.metadata_match.to_dict(),
                )
            )
        return results

    def retrieve(
        self,
        query: str,
        candidates: Sequence[CandidateChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        return self.lexical_search(query, candidates, top_k=top_k)

    def vector_search(
        self,
        query_embedding: Sequence[float],
        candidates: Sequence[CandidateChunk],
        *,
        embedding_model: str,
        embedding_version: str,
        top_k: int = 50,
    ) -> list[VectorRetrievalResult]:
        """Cosine-rank embeddings inside the already-routed candidate universe."""

        if not candidates or top_k <= 0:
            return []
        if not embedding_model.strip() or not embedding_version.strip():
            raise ValueError("embedding model and version must not be empty")
        embedding = [float(value) for value in query_embedding]
        if not embedding or len(embedding) > 2000:
            raise ValueError("query embedding dimensions must be between 1 and 2000")
        if any(not math.isfinite(value) for value in embedding):
            raise ValueError("query embedding values must be finite")

        candidates_by_id = {candidate.chunk_id: candidate for candidate in candidates}
        dimensions = len(embedding)
        # pgvector dimensions are SQL type modifiers, not bindable values. The
        # integer is derived only from the validated vector length; all external
        # text and vector values remain bound parameters.
        vector_literal = "[" + ",".join(format(value, ".17g") for value in embedding) + "]"
        rows = self._fetch_all(
            f"""
            WITH query_vector AS (
                SELECT %s::vector({dimensions}) AS value
            )
            SELECT
                ce.chunk_id,
                c.doc_id,
                1 - (ce.embedding::vector({dimensions}) <=> query_vector.value)
                    AS vector_score
            FROM chunk_embeddings ce
            JOIN chunks c ON c.chunk_id = ce.chunk_id
            CROSS JOIN query_vector
            WHERE ce.chunk_id = ANY(%s)
              AND ce.embedding_model = %s
              AND ce.embedding_version = %s
              AND ce.embedding_dimensions = {dimensions}
            ORDER BY ce.embedding::vector({dimensions}) <=> query_vector.value,
                     ce.chunk_id
            LIMIT %s
            """,
            [
                vector_literal,
                sorted(candidates_by_id),
                embedding_model,
                embedding_version,
                int(top_k),
            ],
        )
        results: list[VectorRetrievalResult] = []
        for rank, row in enumerate(rows, start=1):
            candidate = candidates_by_id.get(str(row["chunk_id"]))
            if candidate is None:
                continue
            results.append(
                VectorRetrievalResult(
                    chunk_id=candidate.chunk_id,
                    doc_id=candidate.doc_id,
                    vector_score=float(row["vector_score"]),
                    rank=rank,
                )
            )
        return results
