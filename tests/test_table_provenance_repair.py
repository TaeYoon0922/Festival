from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from app.parsing.table_provenance_repair import repair_table_provenance


def _write_payload(root: Path, payload: dict) -> Path:
    path = root / "documents" / "periodic" / "document.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as target:
        json.dump(payload, target, ensure_ascii=False)
    return path


def _read_payload(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def _valid_payload() -> dict:
    return {
        "tables": [
            {
                "table_id": "t0001",
                "rows": [["구분", "금액"], ["제품", "100"], ["서비스", "50"]],
            }
        ],
        "chunks": [
            {
                "chunk_id": "periodic_1:ch_table",
                "chunk_type": "table",
                "table_id": "t0001",
                "row_start": 1,
                "row_end": 2,
                "content": "| 제품 | 100 |\n| 서비스 | 50 |",
                "retrieval_text": "[Table]\n\n| 제품 | 100 |\n| 서비스 | 50 |",
                "embedding_model": "unchanged-model",
                "embedding_dimensions": 1024,
            },
            {
                "chunk_id": "periodic_1:ch_existing",
                "chunk_type": "table",
                "table_id": "t0001",
                "row_start": 1,
                "row_end": 1,
                "content": "existing",
                "retrieval_text": "existing",
                "source_table_id": "t0001",
                "source_table_ids": ["t0001"],
                "source_refs": [
                    {"table_id": "t0001", "row_start": 1, "row_end": 1}
                ],
            },
            {
                "chunk_id": "periodic_1:ch_projection",
                "chunk_type": "table_projection",
                "table_id": "t0001",
                "row_start": 2,
                "row_end": 2,
                "content": "projection",
                "retrieval_text": "projection",
            },
        ],
    }


def test_repairs_only_missing_general_table_provenance_atomically() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = _write_payload(root, _valid_payload())
        before = _read_payload(path)
        before_ids = {chunk["chunk_id"] for chunk in before["chunks"]}
        existing_before = dict(before["chunks"][1])
        projection_before = dict(before["chunks"][2])

        with mock.patch(
            "app.parsing.table_provenance_repair.os.replace", wraps=os.replace
        ) as replace:
            report = repair_table_provenance(root)

        after = _read_payload(path)
        repaired = after["chunks"][0]
        assert repaired["source_table_id"] == "t0001"
        assert repaired["source_table_ids"] == ["t0001"]
        assert repaired["source_refs"] == [
            {"table_id": "t0001", "row_start": 1, "row_end": 2}
        ]
        assert {chunk["chunk_id"] for chunk in after["chunks"]} == before_ids
        assert repaired["content"] == before["chunks"][0]["content"]
        assert repaired["retrieval_text"] == before["chunks"][0]["retrieval_text"]
        assert repaired["embedding_model"] == "unchanged-model"
        assert repaired["embedding_dimensions"] == 1024
        assert after["chunks"][1] == existing_before
        assert after["chunks"][2] == projection_before
        assert any(call.args[1] == path for call in replace.call_args_list)
        assert report["modified_chunk_count"] == 1
        assert report["modified_source_count"] == 1
        assert report["already_provenanced_chunk_count"] == 1
        assert report["chunk_id_set_unchanged"] is True
        assert report["protected_fields_unchanged"] is True
        assert report["valid"] is True
        assert json.loads(
            (root / "table_provenance_repair" / "failed_chunks.json").read_text(
                encoding="utf-8"
            )
        ) == []
        assert not list(path.parent.glob("*.tmp"))


def test_dry_run_reports_candidates_without_mutating_payload() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = _write_payload(root, _valid_payload())
        before = path.read_bytes()

        report = repair_table_provenance(root, dry_run=True)

        assert path.read_bytes() == before
        assert report["repairable_chunk_count"] == 1
        assert report["modified_chunk_count"] == 0
        assert report["modified_source_count"] == 0
        assert report["valid"] is True


def test_invalid_table_range_is_not_modified_and_is_reported() -> None:
    payload = _valid_payload()
    payload["chunks"] = [
        {
            "chunk_id": "periodic_1:ch_invalid",
            "chunk_type": "table",
            "table_id": "t0001",
            "row_start": 1,
            "row_end": 99,
            "content": "invalid range",
            "retrieval_text": "invalid range",
        }
    ]
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = _write_payload(root, payload)
        before = path.read_bytes()

        report = repair_table_provenance(root)

        assert path.read_bytes() == before
        assert report["modified_chunk_count"] == 0
        assert report["failed_chunk_count"] == 1
        assert report["valid"] is False
        failures = json.loads(
            (root / "table_provenance_repair" / "failed_chunks.json").read_text(
                encoding="utf-8"
            )
        )
        assert failures[0]["chunk_id"] == "periodic_1:ch_invalid"
        assert failures[0]["reason"] == "invalid_row_range"


def test_repair_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        path = _write_payload(root, _valid_payload())

        first = repair_table_provenance(root)
        after_first = path.read_bytes()
        second = repair_table_provenance(root)

        assert first["modified_chunk_count"] == 1
        assert second["repair_candidate_count"] == 0
        assert second["modified_chunk_count"] == 0
        assert path.read_bytes() == after_first

