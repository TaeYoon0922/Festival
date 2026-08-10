"""Run a lightweight corpus connection check."""

from app.config import MANIFEST_PATH, RAW_DIR, UNIVERSE_PATH


def main() -> None:
    paths = {
        "universe": UNIVERSE_PATH,
        "manifest": MANIFEST_PATH,
        "raw": RAW_DIR,
    }

    missing = [name for name, path in paths.items() if not path.exists()]
    for name, path in paths.items():
        state = "OK" if path.exists() else "MISSING"
        print(f"[{state}] {name}: {path}")

    if missing:
        raise SystemExit(f"Missing corpus components: {', '.join(missing)}")


if __name__ == "__main__":
    main()
