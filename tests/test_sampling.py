import unittest
from collections import Counter

from app.config import CORPUS_DIR, MANIFEST_PATH, RAW_DIR
from app.parsing.sampling import select_sample_documents


@unittest.skipUnless(MANIFEST_PATH.exists() and RAW_DIR.exists(), "local corpus is required")
class SampleSelectionTests(unittest.TestCase):
    def test_selects_five_documents_per_group(self) -> None:
        selected = select_sample_documents(MANIFEST_PATH, CORPUS_DIR, per_group=5)
        counts = Counter(row["doc_group"] for row in selected)

        self.assertEqual(len(selected), 20)
        self.assertEqual(
            counts,
            {"periodic": 5, "exchange": 5, "major": 5, "holding": 5},
        )
        self.assertTrue(all(not row["is_correction"] for row in selected))
        self.assertTrue(all(row["file_format"] == "xml" for row in selected))
        self.assertTrue(
            any(
                row["corp_name"] == "삼성전자"
                and row["doc_subtype"] == "annual"
                for row in selected
            )
        )


if __name__ == "__main__":
    unittest.main()
