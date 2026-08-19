import os
import unittest
from datetime import date
from unittest.mock import patch

from app.retrieval.interfaces import CandidateChunk, CandidateDocument, MetadataMatch
from app.retrieval.local_backend import LocalBM25Retriever, LocalManifestBackend
from app.retrieval.postgres_backend import PostgresBackend


class FakeCursor:
    def __init__(self, database: "FakeDatabase") -> None:
        self.database = database
        self.rows: list[dict] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: tuple) -> None:
        self.database.calls.append((query, params))
        self.rows = list(self.database.handler(query, params))

    def fetchall(self) -> list[dict]:
        return self.rows

    def fetchone(self) -> dict | None:
        return self.rows[0] if self.rows else None


class FakeConnection:
    def __init__(self, database: "FakeDatabase") -> None:
        self.database = database
        self.closed = False

    def cursor(self, **_kwargs: object) -> FakeCursor:
        return FakeCursor(self.database)

    def close(self) -> None:
        self.closed = True


class FakeDatabase:
    def __init__(self, handler=None) -> None:
        self.handler = handler or (lambda _query, _params: [])
        self.calls: list[tuple[str, tuple]] = []
        self.connect_calls: list[tuple[tuple, dict]] = []
        self.connections: list[FakeConnection] = []

    def connect(self, *args: object, **kwargs: object) -> FakeConnection:
        self.connect_calls.append((args, kwargs))
        connection = FakeConnection(self)
        self.connections.append(connection)
        return connection


def disclosure_row(doc_id: str, *, group: str, correction: bool) -> dict:
    return {
        "doc_id": doc_id,
        "corp_code": "001",
        "rcept_no": f"r-{doc_id}",
        "report_nm": "분기보고서",
        "rcept_dt": date(2024, 5, 1),
        "doc_group": group,
        "doc_subtype": "quarter" if group == "periodic" else "holding",
        "is_correction": correction,
        "base_year": 2024,
        "base_month": 3 if doc_id == "d1" else 6,
        "file_path": f"{doc_id}.xml",
        "file_format": "xml",
        "disclosure_metadata": {},
        "stock_code": "000001",
        "corp_name": "테스트",
        "listed_name": "TEST",
        "market": None,
        "industry": "software",
        "sector": "IT",
        "company_metadata": {},
    }


class PostgresBackendTests(unittest.TestCase):
    def test_environment_connection_settings_and_lifecycle(self) -> None:
        database = FakeDatabase(
            lambda _query, _params: [
                {
                    "corp_code": "001",
                    "stock_code": "000001",
                    "corp_name": "테스트",
                    "listed_name": "TEST",
                    "market": None,
                    "industry": None,
                    "sector": None,
                    "company_metadata": {},
                }
            ]
        )
        environment = {
            "PGHOST": "db.internal",
            "PGPORT": "5432",
            "PGDATABASE": "festival",
            "PGUSER": "reader",
            "PGPASSWORD": "secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            backend = PostgresBackend(connection_factory=database.connect)
            company = backend.get_company("001")

        self.assertEqual(company["corp_name"], "테스트")
        self.assertEqual(database.connect_calls[0][0], ())
        self.assertEqual(
            database.connect_calls[0][1],
            {
                "host": "db.internal",
                "port": "5432",
                "dbname": "festival",
                "user": "reader",
                "password": "secret",
            },
        )
        self.assertTrue(database.connections[0].closed)

    def test_database_url_takes_precedence(self) -> None:
        database = FakeDatabase()
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://configured-by-environment/db",
                "PGHOST": "must-not-override-url",
            },
            clear=True,
        ):
            backend = PostgresBackend(connection_factory=database.connect)
            backend.get_company("missing")
        self.assertEqual(
            database.connect_calls[0][0],
            ("postgresql://configured-by-environment/db",),
        )
        self.assertEqual(database.connect_calls[0][1], {})

    def test_hard_filters_and_soft_matches_are_local_compatible(self) -> None:
        rows = [
            disclosure_row("d1", group="periodic", correction=False),
            disclosure_row("d2", group="holding", correction=True),
        ]
        database = FakeDatabase(lambda _query, _params: rows)
        postgres = PostgresBackend(dsn="test", connection_factory=database.connect)
        local = LocalManifestBackend(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"disclosure_metadata", "company_metadata"}
                }
                for row in rows
            ]
        )

        pg_candidates = postgres.get_candidate_documents(
            corp_code="001", doc_group="holding", is_correction=True
        )
        local_candidates = local.get_candidate_documents(
            corp_code="001", doc_group="holding", is_correction=True
        )

        self.assertEqual(
            [candidate.doc_id for candidate in pg_candidates],
            [candidate.doc_id for candidate in local_candidates],
        )
        self.assertEqual(
            [candidate.metadata_match.to_dict() for candidate in pg_candidates],
            [candidate.metadata_match.to_dict() for candidate in local_candidates],
        )
        sql, params = database.calls[0]
        self.assertIn("d.corp_code", sql)
        self.assertNotIn("d.doc_group =", sql)
        self.assertNotIn("d.is_correction =", sql)
        self.assertEqual(params, (["001"],))

    def test_filter_values_are_bound_not_interpolated(self) -> None:
        database = FakeDatabase()
        backend = PostgresBackend(dsn="test", connection_factory=database.connect)
        malicious = "x' OR TRUE --"
        backend.filter_disclosures(company=malicious, year=[2023, 2024], period=3)
        sql, params = database.calls[0]
        self.assertNotIn(malicious, sql)
        self.assertIn("x'ortrue--", params[0])
        self.assertEqual(params[-2:], ([2023, 2024], 3))

    def test_fetch_chunks_preserves_evidence_and_section_soft_match(self) -> None:
        chunk_row = {
            "chunk_id": "c1",
            "doc_id": "d1",
            "source_part_id": "p1",
            "section_id": "s1",
            "table_id": None,
            "chunk_type": "text",
            "chunk_order": 0,
            "content": "원문 증거",
            "retrieval_text": "테스트 매출액 원문 증거",
            "char_count": 5,
            "retrieval_priority": "normal",
            "chunk_metadata": {"is_indexable": True},
            "section_title": "매출",
            "section_path": ["사업의 내용", "매출"],
            "doc_group": "periodic",
            "doc_subtype": "quarter",
            "report_nm": "분기보고서",
            "rcept_dt": date(2024, 5, 1),
            "is_correction": False,
            "corp_code": "001",
            "corp_name": "테스트",
            "listed_name": "TEST",
        }
        database = FakeDatabase(lambda _query, _params: [chunk_row])
        backend = PostgresBackend(dsn="test", connection_factory=database.connect)
        document = CandidateDocument(
            doc_id="d1",
            metadata={},
            metadata_match=MetadataMatch(
                soft_boosts={"doc_group": True, "section_path": False},
                soft_inputs={"doc_group": "periodic", "section_path": "사업의 내용"},
                soft_score=1.0,
            ),
        )

        chunk = backend.get_candidate_chunks([document])[0]
        self.assertEqual(chunk.chunk["content"], "원문 증거")
        self.assertEqual(chunk.chunk["retrieval_text"], "테스트 매출액 원문 증거")
        self.assertEqual(chunk.chunk["rcept_dt"], "2024-05-01")
        self.assertTrue(chunk.metadata_match.soft_boosts["section_path"])
        self.assertEqual(chunk.metadata_match.soft_score, 2.0)

    def test_lexical_result_structure_matches_local_backend(self) -> None:
        candidates = [
            CandidateChunk(
                chunk_id="c1",
                doc_id="d1",
                chunk={"chunk_id": "c1", "retrieval_text": "alpha revenue"},
                metadata_match=MetadataMatch(
                    soft_boosts={"doc_group": True}, soft_score=1.0
                ),
            ),
            CandidateChunk(
                chunk_id="c2",
                doc_id="d2",
                chunk={"chunk_id": "c2", "retrieval_text": "beta assets"},
                metadata_match=MetadataMatch(),
            ),
        ]

        def handler(query: str, _params: tuple) -> list[dict]:
            self.assertIn("to_tsvector('simple', c.retrieval_text)", query)
            return [{"chunk_id": "c1", "doc_id": "d1", "bm25_score": 0.75}]

        database = FakeDatabase(handler)
        postgres = PostgresBackend(dsn="test", connection_factory=database.connect)
        pg_result = postgres.retrieve("alpha", candidates, top_k=1)[0].to_dict()
        local_result = LocalBM25Retriever().retrieve(
            "alpha", candidates, top_k=1
        )[0].to_dict()

        self.assertEqual(set(pg_result), set(local_result))
        for key in ("chunk_id", "doc_id", "rank", "metadata_match"):
            self.assertEqual(pg_result[key], local_result[key])
        sql, params = database.calls[0]
        self.assertIn("websearch_to_tsquery('simple', %s)", sql)
        self.assertEqual(params, ("alpha", ["c1", "c2"], 1))

    def test_vector_search_is_scoped_and_parameter_bound(self) -> None:
        candidates = [
            CandidateChunk("c1", "d1", {"retrieval_text": "alpha"}, MetadataMatch()),
            CandidateChunk("c2", "d2", {"retrieval_text": "beta"}, MetadataMatch()),
        ]
        malicious_model = "model' OR TRUE --"

        def handler(query: str, params: tuple) -> list[dict]:
            self.assertIn("JOIN chunks c ON c.chunk_id = ce.chunk_id", query)
            self.assertIn("ce.chunk_id = ANY(%s)", query)
            self.assertIn("<=>", query)
            self.assertIn("vector(3)", query)
            self.assertNotIn(malicious_model, query)
            return [{"chunk_id": "c1", "doc_id": "d1", "vector_score": 0.75}]

        database = FakeDatabase(handler)
        backend = PostgresBackend(dsn="test", connection_factory=database.connect)
        result = backend.vector_search(
            [0.1, 0.2, 0.3],
            candidates,
            embedding_model=malicious_model,
            embedding_version="v1",
            top_k=1,
        )[0]

        self.assertEqual(result.chunk_id, "c1")
        self.assertEqual(result.vector_score, 0.75)
        _sql, params = database.calls[0]
        self.assertEqual(params[0], "[0.10000000000000001,0.20000000000000001,0.29999999999999999]")
        self.assertEqual(params[1:], (["c1", "c2"], malicious_model, "v1", 1))

    def test_vector_search_rejects_invalid_embeddings_without_querying(self) -> None:
        database = FakeDatabase()
        backend = PostgresBackend(dsn="test", connection_factory=database.connect)
        candidate = CandidateChunk("c1", "d1", {}, MetadataMatch())
        with self.assertRaises(ValueError):
            backend.vector_search(
                [float("nan")],
                [candidate],
                embedding_model="model",
                embedding_version="v1",
            )
        self.assertEqual(database.calls, [])


if __name__ == "__main__":
    unittest.main()
