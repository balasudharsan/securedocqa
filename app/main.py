from fastapi import FastAPI

from app.api.errors import install_exception_handlers
from app.api.routes import router

app = FastAPI(title="SecureDocQA", version="0.1.0")

app.include_router(router)
install_exception_handlers(app)


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict[str, str]:
    """Liveness probe. Deliberately dependency-free.

    It performs no authentication and makes no external calls (no Qdrant,
    no LLM), so it stays 200 even when those dependencies are down. That
    lets monitoring distinguish "app alive" from "dependency down".

    Registered for both GET and HEAD: a GET-only route returns 405 to HEAD
    requests, which breaks uptime checkers like UptimeRobot.
    """
    return {"status": "ok"}