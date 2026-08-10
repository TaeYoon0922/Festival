import unittest

from app.config import CORPUS_DIR, MANIFEST_PATH, RAW_DIR, UNIVERSE_PATH


class CorpusPathTests(unittest.TestCase):
    def test_corpus_children_share_the_configured_root(self) -> None:
        self.assertEqual(UNIVERSE_PATH.parent, CORPUS_DIR)
        self.assertEqual(MANIFEST_PATH.parent, CORPUS_DIR)
        self.assertEqual(RAW_DIR.parent, CORPUS_DIR)


if __name__ == "__main__":
    unittest.main()
