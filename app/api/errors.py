"""Typed-exception -> clean HTTP mapping (Decision 13: fail closed, no detail).

Every handler returns a SAFE message with the right status code. Internal
detail — stack traces, model names, file paths, raw exceptions — is logged
server-side only and never reaches the response body.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.llm.answer import LLMError
from app.pipeline.extract import (
    FileTooLargeError,
    InvalidFileTypeError,
    NoTextFoundError,
    TooManyPagesError,
    UnreadablePDFError,
)
from app.retrieval.store import RetrievalError
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


# Typed exception -> (status code, SAFE client message). Data-driven so adding a
# mapping is a one-line change and install_exception_handlers stays trivial.
_ERROR_MAP: list[tuple[type[Exception], int, str]] = [
    (RateLimitExceeded, 429, "Too many requests. Please wait and try again."),
    (SessionQuestionLimitExceeded, 429, "This session has reached its limit. Start a new session."),
    (SessionTokenLimitExceeded, 429, "This session has reached its limit. Start a new session."),
    (DailyLimitExceeded, 503, "Service is at capacity. Please try later."),
    (UnknownSession, 404, "Session not found."),
    (FileTooLargeError, 413, "The uploaded file is too large."),
    (TooManyPagesError, 422, "The document has too many pages."),
    (InvalidFileTypeError, 422, "The file must be a valid PDF."),
    (UnreadablePDFError, 422, "The PDF could not be read."),
    (NoTextFoundError, 422, "No extractable text was found (is this a scanned PDF?)."),
    (RetrievalError, 503, "The document service is temporarily unavailable."),
    (LLMError, 503, "The answer service is temporarily unavailable."),
]


def _make_handler(status_code: int, message: str):
    """Build a handler that returns a fixed safe response for one exception type."""
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        return _error(status_code, message)
    return handler


async def _unexpected_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all: log full detail server-side, return nothing internal."""
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "An unexpected error occurred.")


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers mapping our typed exceptions to safe HTTP responses."""
    for exc_type, status_code, message in _ERROR_MAP:
        app.add_exception_handler(exc_type, _make_handler(status_code, message))
    app.add_exception_handler(Exception, _unexpected_handler)
