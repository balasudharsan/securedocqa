from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.api import routes
from app.config import settings
from app.main import app
from app.session.state import UnknownSession

client = TestClient(app)
KEY = {"X-API-Key": settings.DEMO_API_KEY}


def test_health_needs_no_key():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_session_with_key():
    resp = client.post("/session", headers=KEY)
    assert resp.status_code == 201
    assert isinstance(resp.json()["session_id"], str)
    assert resp.json()["session_id"]


def test_create_session_without_key_is_401():
    resp = client.post("/session")
    assert resp.status_code == 401


def test_create_session_with_wrong_key_is_401():
    resp = client.post("/session", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_delete_session_returns_deleted(monkeypatch):
    monkeypatch.setattr(routes, "delete_collection", AsyncMock())
    resp = client.delete("/session/anything", headers=KEY)
    assert resp.status_code == 200
    assert resp.json() == {"deleted": True}


def test_handler_maps_unknown_session_to_404_without_detail(monkeypatch):
    # Force a typed exception from the endpoint and assert the safe mapping.
    def _raise() -> str:
        raise UnknownSession

    monkeypatch.setattr(routes.store, "create_session", _raise)
    resp = client.post("/session", headers=KEY)
    assert resp.status_code == 404
    assert resp.json() == {"detail": "Session not found."}


def test_catch_all_returns_500_without_stack_trace(monkeypatch):
    def _boom() -> str:
        raise RuntimeError("internal secret detail")

    monkeypatch.setattr(routes.store, "create_session", _boom)
    safe_client = TestClient(app, raise_server_exceptions=False)
    resp = safe_client.post("/session", headers=KEY)
    assert resp.status_code == 500
    assert resp.json() == {"detail": "An unexpected error occurred."}
    # Internal detail must never leak into the body.
    assert "internal secret detail" not in resp.text
    assert "RuntimeError" not in resp.text
