from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from app.api import routes
from app.config import settings
from app.llm.answer import LLMError
from app.main import app

client = TestClient(app)
KEY = {"X-API-Key": settings.DEMO_API_KEY}

PDF_BYTES = b"%PDF-1.4\n%mock pdf body\n"


def _new_session() -> str:
    resp = client.post("/session", headers=KEY)
    assert resp.status_code == 201
    return resp.json()["session_id"]


# --- /upload ----------------------------------------------------------------

def test_upload_happy_path(monkeypatch):
    sid = _new_session()
    # extract_pages is monkeypatched (pypdf can't make text-bearing PDFs in a
    # test); the rest of the pipeline (validate/assert_has_text/chunk/redact)
    # runs for real, network is mocked.
    monkeypatch.setattr(routes, "extract_pages",
                        lambda data: [(1, "clause " * 40), (2, "more text " * 30)])
    monkeypatch.setattr(routes, "create_collection", AsyncMock())
    monkeypatch.setattr(routes, "index_chunks", AsyncMock(return_value=3))

    resp = client.post(
        "/upload",
        headers=KEY,
        data={"session_id": sid},
        files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "indexed"
    assert body["pages"] == 2
    assert body["chunks"] == 3


def test_upload_oversized_is_413(monkeypatch):
    sid = _new_session()
    # Shrink the cap so we don't have to stream 10 MB in a test.
    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 100)
    oversized = b"%PDF-" + b"0" * 500
    resp = client.post(
        "/upload",
        headers=KEY,
        data={"session_id": sid},
        files={"file": ("big.pdf", oversized, "application/pdf")},
    )
    assert resp.status_code == 413


def test_upload_wrong_type_is_422():
    sid = _new_session()
    resp = client.post(
        "/upload",
        headers=KEY,
        data={"session_id": sid},
        files={"file": ("notes.txt", b"just some text, not a pdf", "text/plain")},
    )
    assert resp.status_code == 422


def test_upload_requires_key():
    resp = client.post(
        "/upload",
        data={"session_id": "x"},
        files={"file": ("doc.pdf", PDF_BYTES, "application/pdf")},
    )
    assert resp.status_code == 401


# --- /ask -------------------------------------------------------------------

def test_ask_below_threshold_makes_no_llm_call(monkeypatch):
    sid = _new_session()
    monkeypatch.setattr(routes, "search",
                        AsyncMock(return_value=[{"page": 1, "text": "x", "score": 0.10}]))
    never = AsyncMock(side_effect=AssertionError("LLM must not be called"))
    monkeypatch.setattr(routes, "generate_answer", never)

    resp = client.post("/ask", headers=KEY, json={"session_id": sid, "question": "hi?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is False
    assert body["answer"] == "I could not find that in this document."
    assert body["citations"] == []
    assert body["model"] is None
    never.assert_not_called()


def test_ask_above_threshold_calls_llm_and_records(monkeypatch):
    sid = _new_session()
    monkeypatch.setattr(routes, "search", AsyncMock(return_value=[
        {"page": 3, "text": "the fee is five dollars", "score": 0.95},
        {"page": 1, "text": "intro text", "score": 0.88},
    ]))
    monkeypatch.setattr(routes, "generate_answer", AsyncMock(
        return_value={"answer": "The fee is $5 [p.3]", "model": "primary"}))
    record_spy = MagicMock()
    monkeypatch.setattr(routes.store, "record_usage", record_spy)

    resp = client.post("/ask", headers=KEY, json={"session_id": sid, "question": "fee?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["answer"] == "The fee is $5 [p.3]"
    assert body["citations"] == [1, 3]
    assert body["model"] == "primary"
    record_spy.assert_called_once()


def test_ask_llm_error_is_503_no_fabrication(monkeypatch):
    sid = _new_session()
    monkeypatch.setattr(routes, "search",
                        AsyncMock(return_value=[{"page": 1, "text": "t", "score": 0.95}]))
    monkeypatch.setattr(routes, "generate_answer", AsyncMock(side_effect=LLMError("down")))

    resp = client.post("/ask", headers=KEY, json={"session_id": sid, "question": "q?"})
    assert resp.status_code == 503
    assert "answer" not in resp.json()


def test_ask_overlong_question_is_422():
    sid = _new_session()
    resp = client.post(
        "/ask",
        headers=KEY,
        json={"session_id": sid, "question": "x" * 2001},
    )
    assert resp.status_code == 422


def test_ask_requires_key():
    resp = client.post("/ask", json={"session_id": "x", "question": "hi?"})
    assert resp.status_code == 401
