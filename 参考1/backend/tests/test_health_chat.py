from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_mock_chat() -> None:
    response = client.post("/api/chat", json={"message": "What is Transformer?", "mode": "qa"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["agent"] == "EduAgentOrchestrator"
    assert payload["intent"] == "qa"
    assert payload["model_provider"] == "mock"
    assert payload["reply"]
    assert payload["agent_trace"]
