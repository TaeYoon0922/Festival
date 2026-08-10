"""Project paths and local corpus configuration."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = Path(
    os.getenv("DISCLOSURE_DATA_DIR", str(PROJECT_ROOT / "data" / "corpus"))
).expanduser()

UNIVERSE_PATH = CORPUS_DIR / "universe.csv"
MANIFEST_PATH = CORPUS_DIR / "manifest.jsonl"
RAW_DIR = CORPUS_DIR / "raw"
