"""FastAPI startup for DeepSearch Agent."""

from __future__ import annotations

import logging
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from app.api.routes import router as chat_router
from app.config import settings
from app.rag.auto_index import auto_index_if_needed

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="DeepSearch Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=settings.cors_allow_methods,
        allow_headers=settings.cors_allow_headers,
    )
    app.include_router(chat_router)

    @app.get("/", include_in_schema=False)
    def frontend() -> FileResponse:
        """Serve only the UI document, never the project directory."""
        return FileResponse(Path(__file__).resolve().parents[1] / "frontend" / "index.html")

    @app.get("/config.js", include_in_schema=False)
    def public_config() -> Response:
        """Expose the explicit browser whitelist; keep all credentials server-side."""
        config = {
            "API_BASE": settings.api_base_url,
            "defaultUseRag": settings.frontend_use_rag,
        }
        return Response(
            content="window.DEEPSEARCH_CONFIG = Object.freeze(" + json.dumps(config) + ");\n",
            media_type="application/javascript",
            headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"},
        )

    @app.on_event("startup")
    async def _startup_auto_index() -> None:
        """Auto-detect new/changed docs and rebuild index if needed."""
        if not settings.startup_auto_index:
            logger.info("Startup index check disabled.")
            return
        logger.info("Checking for document changes...")
        auto_index_if_needed()
        logger.info("Startup index check complete.")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
        log_level=settings.log_level.lower(),
    )

