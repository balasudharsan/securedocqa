"""Typed-exception -> clean HTTP mapping (Decision 13: fail closed, no detail).

Every handler returns a SAFE message with the right status code. Internal
detail — stack traces, model names, file paths, raw exceptions — is logged
server-side only and never reaches the response body.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.session.state import (
    DailyLimitExceeded,
    RateLimitExceeded,
    SessionQuestionLimitExceeded,
    SessionTokenLimitExceeded,
    UnknownSession,
)

logger = logging.getLogger(__name__)


def _error(status_code: int, message: str) -> JSONResponse:
    """Build a clean error body — the only shape any handler returns."""
    return JSONResponse(status_code=status_code, content={"detail": message})


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers mapping our typed exceptions to safe HTTP responses."""

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        return _error(429, "Too many requests. Please wait and try again.")

    @app.exception_handler(SessionQuestionLimitExceeded)
    async def _session_questions(
        request: Request, exc: SessionQuestionLimitExceeded
    ) -> JSONResponse:
        return _error(429, "This session has reached its limit. Start a new session.")

    @app.exception_handler(SessionTokenLimitExceeded)
    async def _session_tokens(
        request: Request, exc: SessionTokenLimitExceeded
    ) -> JSONResponse:
        return _error(429, "This session has reached its limit. Start a new session.")

    @app.exception_handler(DailyLimitExceeded)
    async def _daily_limit(request: Request, exc: DailyLimitExceeded) -> JSONResponse:
        return _error(503, "Service is at capacity. Please try later.")

    @app.exception_handler(UnknownSession)
    async def _unknown_session(request: Request, exc: UnknownSession) -> JSONResponse:
        return _error(404, "Session not found.")

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        # Full detail server-side only; the client sees nothing internal.
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error(500, "An unexpected error occurred.")
