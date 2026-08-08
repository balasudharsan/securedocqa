"""API routes — thin orchestration only (no business logic in endpoints).

Endpoints call existing modules and let the typed exceptions bubble up to the
handlers in errors.py. One module-level SessionStore holds the app's in-memory
state (Decision 4).
"""
import secrets
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.config import settings
from app.retrieval.store import delete_collection
from app.session.state import SessionStore

# The app's single in-memory state instance.
store = SessionStore()

router = APIRouter()

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)



def verify_key(api_key: str = Security(_api_key_header)) -> None:
    """Reject any request without the exact demo key. Applied per-route. Uses secrets.compare_digest for a constant-time comparison, so the check
    cannot be used as a timing oracle to guess the key character by character.
    """
    if not api_key or not secrets.compare_digest(api_key, settings.DEMO_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


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
