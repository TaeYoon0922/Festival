"""Run the answer API with uvicorn: ``python -m app.api``."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    uvicorn.run(
        "app.api.app:app",
        host=os.getenv("FESTIVAL_API_HOST", "0.0.0.0"),  # noqa: S104 - container port
        port=int(os.getenv("FESTIVAL_API_PORT", "8000")),
        workers=int(os.getenv("FESTIVAL_API_WORKERS", "1")),
        log_level=os.getenv("FESTIVAL_API_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
