"""FastAPI startup for DeepSearch Agent."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as chat_router
from app.rag.auto_index import auto_index_if_needed

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="DeepSearch Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(chat_router)

    @app.on_event("startup")
    async def _startup_auto_index() -> None:
        """Auto-detect new/changed docs and rebuild index if needed."""
        logger.info("Checking for document changes...")
        auto_index_if_needed()
        logger.info("Startup index check complete.")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

