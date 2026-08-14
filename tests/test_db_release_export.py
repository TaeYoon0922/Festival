import unittest

from app.exporting.db_release_export import _source_ref_rows


class DBReleaseExportTests(unittest.TestCase):
    def test_projection_field_refs_are_exported_without_inference(self) -> None:
        chunk = {
            "chunk_id": "doc:chunk",
            "chunk_type": "table_projection",
            "projection_fields": {"보유 목적": "단순투자"},
            "projection_field_refs": {
                "보유 목적": [{"table_id": "t0002", "row_start": 11, "row_end": 11}]
            },
            "source_refs": [
                {"table_id": "t0002", "row_start": 11, "row_end": 11}
            ],
        }
        rows, errors = _source_ref_rows(chunk, "part-1", {"t0002": 12})
        self.assertEqual(errors, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["table_id"], "t0002")
        self.assertEqual(rows[0]["field_name"], "보유 목적")
        self.assertEqual(rows[0]["source_type"], "projection_field_ref")

    def test_projection_field_without_actual_ref_is_an_error(self) -> None:
        chunk = {
            "chunk_id": "doc:chunk",
            "chunk_type": "table_projection",
            "projection_fields": {"보유 목적": "단순투자"},
            "projection_field_refs": {},
            "source_refs": [],
        }
        rows, errors = _source_ref_rows(chunk, "part-1", {})
        self.assertEqual(rows, [])
        self.assertEqual(
            {category for category, _ in errors},
            {"missing_source_ref", "missing_projection_field_ref"},
        )


if __name__ == "__main__":
    unittest.main()
