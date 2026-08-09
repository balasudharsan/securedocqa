from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.config import settings
from app.retrieval import store
from app.retrieval.store import (
    RetrievalError,
    collection_name,
    create_collection,
    delete_collection,
    filter_by_threshold,
    index_chunks,
    search,
)
from qdrant_client.models import Distance


# --- Pure functions (no mocking needed) -------------------------------------

def test_collection_name():
    assert collection_name("abc123") == "sdq_abc123"


def test_filter_keeps_at_or_above_threshold():
    results = [{"score": 0.9}, {"score": 0.75}, {"score": 0.5}]
    kept = filter_by_threshold(results, threshold=0.75)
    assert kept == [{"score": 0.9}, {"score": 0.75}]


def test_filter_uses_default_threshold():
    results = [{"score": settings.SCORE_THRESHOLD}, {"score": settings.SCORE_THRESHOLD - 0.01}]
    kept = filter_by_threshold(results)
    assert kept == [{"score": settings.SCORE_THRESHOLD}]


def test_filter_empty_list():
    assert filter_by_threshold([]) == []


# --- Async network functions (monkeypatched client) -------------------------

def _patch_client(monkeypatch) -> AsyncMock:
    client = AsyncMock()
    monkeypatch.setattr(store, "get_client", lambda: client)
    return client


@pytest.mark.asyncio
async def test_create_collection_uses_384_cosine(monkeypatch):
    client = _patch_client(monkeypatch)
    await create_collection("s1")
    kwargs = client.create_collection.call_args.kwargs
    assert kwargs["collection_name"] == "sdq_s1"
    assert kwargs["vectors_config"].size == 384
    assert kwargs["vectors_config"].distance == Distance.COSINE


@pytest.mark.asyncio
async def test_index_chunks_upserts_and_returns_count(monkeypatch):
    client = _patch_client(monkeypatch)
    chunks = [{"page": 1, "text": "alpha"}, {"page": 2, "text": "beta"}]
    count = await index_chunks("s1", chunks)
    assert count == 2
    points = client.upsert.call_args.kwargs["points"]
    assert len(points) == 2
    # Clean text is stored in the payload (no "passage: " prefix).
    assert points[0].payload == {"page": 1, "text": "alpha"}


@pytest.mark.asyncio
async def test_search_shapes_results(monkeypatch):
    client = _patch_client(monkeypatch)
    client.query_points.return_value = SimpleNamespace(points=[
        SimpleNamespace(score=0.91, payload={"page": 3, "text": "hit"}),
        SimpleNamespace(score=0.42, payload=None),
    ])
    results = await search("s1", "what is x?")
    assert results[0] == {"page": 3, "text": "hit", "score": 0.91}
    assert results[1] == {"page": None, "text": None, "score": 0.42}


@pytest.mark.asyncio
async def test_create_collection_raises_retrieval_error(monkeypatch):
    client = _patch_client(monkeypatch)
    client.create_collection.side_effect = RuntimeError("boom")
    with pytest.raises(RetrievalError):
        await create_collection("s1")


@pytest.mark.asyncio
async def test_index_chunks_raises_retrieval_error(monkeypatch):
    client = _patch_client(monkeypatch)
    client.upsert.side_effect = RuntimeError("boom")
    with pytest.raises(RetrievalError):
        await index_chunks("s1", [{"page": 1, "text": "alpha"}])


@pytest.mark.asyncio
async def test_search_raises_retrieval_error(monkeypatch):
    client = _patch_client(monkeypatch)
    client.query_points.side_effect = RuntimeError("boom")
    with pytest.raises(RetrievalError):
        await search("s1", "q")


@pytest.mark.asyncio
async def test_delete_collection_swallows_errors(monkeypatch):
    client = _patch_client(monkeypatch)
    client.delete_collection.side_effect = RuntimeError("boom")
    # Best-effort cleanup: must NOT raise.
    await delete_collection("s1")
