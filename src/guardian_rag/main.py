"""Production ASGI entry point: `uvicorn guardian_rag.main:app`."""

from __future__ import annotations

from guardian_rag.api.app import create_app

app = create_app()
