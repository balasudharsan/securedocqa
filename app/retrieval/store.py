"""Qdrant Cloud Inference wrapper: per-session collections (Decisions 5/6/9).

Network I/O is isolated from pure logic so the pure parts (collection_name,
filter_by_threshold) are unit-testable without a network, and the async parts
are testable by monkeypatching get_client(). Every Qdrant call is wrapped and
raises a typed RetrievalError — except delete_collection, which is best-effort
cleanup and only logs on failure. Document text/chunk content is never logged.
"""

from __future__ import annotations

import logging
import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, Document, PointStruct, VectorParams

from app.config import settings
from app.pipeline.chunk import to_passage, to_query

logger = logging.getLogger(__name__)

# e5-small embedding dimensionality and a network timeout for every call.
VECTOR_SIZE = 384
REQUEST_TIMEOUT_S = 30


class RetrievalError(Exception):
    """Typed wrapper — no raw Qdrant exception may reach a caller."""


def get_client() -> AsyncQdrantClient:
    """Build the Qdrant client. A function (not a global) so tests can patch it."""
    return AsyncQdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY,
        cloud_inference=True,
        timeout=REQUEST_TIMEOUT_S,
    )


def collection_name(session_id: str) -> str:
    """Per-session collection name (Decision 5). Pure."""
    return f"sdq_{session_id}"


async def create_collection(session_id: str) -> None:
    """Create the per-session collection (size=384, cosine). Wrapped + timed."""
    client = get_client()
    try:
        await client.create_collection(
            collection_name=collection_name(session_id),
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    except Exception as exc:
        logger.error("create_collection failed session_id=%s", session_id)
        raise RetrievalError("Could not create collection.") from exc


async def index_chunks(session_id: str, chunks: list[dict]) -> int:
    """Upsert chunks as points; store CLEAN text, embed the passage-prefixed text.

    Returns the number of points indexed. The Qdrant call is wrapped so a
    failure surfaces as RetrievalError, never a raw library exception.
    """
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=Document(text=to_passage(chunk["text"]), model=settings.EMBED_MODEL),
            payload={"page": chunk["page"], "text": chunk["text"]},
        )
        for chunk in chunks
    ]
    client = get_client()
    try:
        await client.upsert(collection_name=collection_name(session_id), points=points)
    except Exception as exc:
        logger.error("index_chunks failed session_id=%s", session_id)
        raise RetrievalError("Could not index chunks.") from exc
    return len(points)


async def search(session_id: str, question: str) -> list[dict]:
    """Query the collection; return shaped {page, text, score} dicts. Wrapped."""
    client = get_client()
    try:
        response = await client.query_points(
            collection_name=collection_name(session_id),
            query=Document(text=to_query(question), model=settings.EMBED_MODEL),
            limit=settings.TOP_K,
            with_payload=True,
        )
    except Exception as exc:
        logger.error("search failed session_id=%s", session_id)
        raise RetrievalError("Could not search collection.") from exc
    return [
        {
            "page": (point.payload or {}).get("page"),
            "text": (point.payload or {}).get("text"),
            "score": point.score,
        }
        for point in response.points
    ]


def filter_by_threshold(results: list[dict], threshold: float | None = None) -> list[dict]:
    """Keep only results scoring >= threshold (Decision 9 cost-filter). Pure.

    Defaults to settings.SCORE_THRESHOLD. This decides whether it is even worth
    calling the LLM, so it runs with no network of its own.
    """
    if threshold is None:
        threshold = settings.SCORE_THRESHOLD
    return [result for result in results if result["score"] >= threshold]


async def delete_collection(session_id: str) -> None:
    """Best-effort session-end cleanup (Decision 5/6): log on failure, never raise.

    A failed delete must not crash a user's session-end, so the exception is
    swallowed after logging (no document content is logged).
    """
    client = get_client()
    try:
        await client.delete_collection(collection_name=collection_name(session_id))
    except Exception:
        logger.warning("delete_collection failed session_id=%s", session_id)
