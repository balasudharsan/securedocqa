from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_get_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_head_not_405():
    # A GET-only route returns 405 to HEAD, which breaks UptimeRobot.
    response = client.head("/health")
    assert response.status_code == 200
