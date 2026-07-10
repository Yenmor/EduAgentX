from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app
from backend.config.settings import get_settings


def test_model_status_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("ENABLE_MOCK_LLM", "true")
    monkeypatch.setenv("ENABLE_LLM_FALLBACK", "true")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret-value")
    get_settings.cache_clear()

    response = TestClient(app).get("/api/model/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["llm_provider"] == "openai-compatible"
    assert payload["model"] == "gpt-4o-mini"
    assert payload["mock_llm_enabled"] is True
    assert payload["openai_api_key_configured"] is True
    assert "sk-test-secret-value" not in response.text

    get_settings.cache_clear()
