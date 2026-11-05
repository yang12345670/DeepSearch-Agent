"""Root entrypoint for DeepSearch Agent.

This file forwards to `app.main:app` so both:
- `uvicorn main:app --reload`
- `uvicorn app.main:app --reload`
work.
"""

from app.main import app
from app.config import settings


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
        log_level=settings.log_level.lower(),
    )

