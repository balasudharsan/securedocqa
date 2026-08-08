"""API routes — thin orchestration only (no business logic in endpoints).

Endpoints call existing modules in order and let the typed exceptions bubble up
to the handlers in errors.py. One module-level SessionStore holds the app's
in-memory state (Decision 4). Nothing here logs document text, chunk content,
questions, or answers — only the modules' own count/category/model logging.
"""

from __future__ import annotations

import secrets

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Security,
    UploadFile,
    status,
)
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.answer import answer as generate_answer
from app.pipeline.chunk import chunk_text
from app.pipeline.extract import (
    FileTooLargeError,
    assert_has_text,
    extract_pages,
    validate_file,
)
from app.retrieval.store import (
    create_collection,
    delete_collection,
    filter_by_threshold,
    index_chunks,
    search,
)
from app.security.redaction import log_redaction, redact
from app.security.screening import log_screening, screen_text
from app.session.state import SessionStore

# The app's single in-memory state instance.
store = SessionStore()

router = APIRouter()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Hard request-layer upload cap (D3) and the streaming read size.
MAX_UPLOAD_BYTES = settings.MAX_FILE_MB * 1024 * 1024
READ_CHUNK_BYTES = 1024 * 1024

# The exact grounded-refusal string (matches the LLM system prompt).
REFUSAL = "I could not find that in this document."


def verify_key(api_key: str = Security(_api_key_header)) -> None:
    """Reject any request without the exact demo key. Applied per-route.

    Uses secrets.compare_digest for a constant-time comparison, so the check
    cannot be used as a timing oracle to guess the key character by character.
    """
    if not api_key or not secrets.compare_digest(api_key, settings.DEMO_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


class AskRequest(BaseModel):
    """Validated /ask body. Question length is bounded to cap input cost."""

    session_id: str = Field(min_length=1, max_length=64)
    question: str = Field(min_length=1, max_length=2000)


# --- simple session routes --------------------------------------------------

@router.post("/session", status_code=status.HTTP_201_CREATED, dependencies=[Depends(verify_key)])
def create_session() -> dict[str, str]:
    """Create a new in-memory session and return its id."""
    return {"session_id": store.create_session()}


@router.delete("/session/{session_id}", dependencies=[Depends(verify_key)])
async def delete_session(session_id: str) -> dict[str, bool]:
    """Best-effort session teardown: drop the collection, then the state.

    Idempotent — an unknown id still returns ok (delete_collection swallows its
    own errors; store.delete_session is a no-op on a missing id).
    """
    await delete_collection(session_id)
    store.delete_session(session_id)
    return {"deleted": True}


# --- upload: validate -> extract -> screen -> chunk -> redact -> index ------

async def _read_capped(file: UploadFile) -> bytes:
    """Read an UploadFile in chunks, aborting BEFORE a full RAM load if it
    exceeds the cap (D3: reject oversized uploads at read time, not after)."""
    buffer = bytearray()
    while True:
        chunk = await file.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        buffer.extend(chunk)
        if len(buffer) > MAX_UPLOAD_BYTES:
            raise FileTooLargeError("Upload exceeds the size limit.")
    return bytes(buffer)


def _screen_pages(pages: list[tuple[int, str]], session_id: str) -> None:
    """Detect-and-log injection phrasing per page. Never blocks."""
    for _page, text in pages:
        categories = screen_text(text)
        if categories:
            log_screening(categories, session_id)


def _redact_chunks(chunks: list[dict], session_id: str) -> None:
    """Redact each chunk's text in place BEFORE indexing; log total counts."""
    total: dict[str, int] = {}
    for chunk in chunks:
        redacted, counts = redact(chunk["text"])
        chunk["text"] = redacted
        for label, count in counts.items():
            total[label] = total.get(label, 0) + count
    log_redaction(total, session_id)


@router.post("/upload", dependencies=[Depends(verify_key)])
async def upload(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> dict[str, object]:
    """Ingest a PDF into the session's collection. Thin orchestration only."""
    data = await _read_capped(file)
    validate_file(data, file.filename or "")
    pages = extract_pages(data)
    assert_has_text(pages)
    _screen_pages(pages, session_id)
    chunks = chunk_text(pages)
    _redact_chunks(chunks, session_id)
    await create_collection(session_id)
    indexed = await index_chunks(session_id, chunks)
    store.touch(session_id)
    return {"status": "indexed", "pages": len(pages), "chunks": indexed}


# --- ask: budget gate -> search -> threshold -> (LLM or refuse) -------------

def _estimate_tokens(kept: list[dict], answer_text: str) -> int:
    """Rough token estimate (~4 chars/token) for usage accounting."""
    context_chars = sum(len(hit["text"]) for hit in kept)
    return context_chars // 4 + len(answer_text) // 4


@router.post("/ask", dependencies=[Depends(verify_key)])
async def ask(payload: AskRequest) -> dict[str, object]:
    """Answer a question against the session's indexed document. Thin."""
    store.check_budget(payload.session_id)
    results = await search(payload.session_id, payload.question)
    kept = filter_by_threshold(results)
    if not kept:
        # Threshold cost-filter: nothing relevant, so make NO paid LLM call.
        store.record_usage(payload.session_id, tokens_used=0)
        return {"answer": REFUSAL, "citations": [], "model": None, "grounded": False}

    result = await generate_answer(payload.question, kept)
    citations = sorted({hit["page"] for hit in kept})
    store.record_usage(
        payload.session_id,
        tokens_used=_estimate_tokens(kept, result["answer"]),
    )
    return {
        "answer": result["answer"],
        "citations": citations,
        "model": result["model"],
        "grounded": True,
    }
