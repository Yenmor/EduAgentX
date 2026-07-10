from __future__ import annotations

from backend.config.settings import Settings
from backend.services.spark_wrapper import SparkWrapper


def test_spark_wrapper_mock_mode() -> None:
    wrapper = SparkWrapper(settings=Settings(enable_mock_llm=True, spark_api_password=""))
    response = wrapper.generate([{"role": "user", "content": "请解释 RAG"}])
    assert response["model_provider"] == "mock"
    assert response["model_name"] == "mock"
    assert "RAG" in response["content"]


def test_spark_wrapper_missing_config() -> None:
    wrapper = SparkWrapper(
        settings=Settings(
            enable_mock_llm=False,
            enable_llm_fallback=True,
            spark_app_id="",
            spark_api_key="",
            spark_api_secret="",
            spark_api_password="",
        )
    )
    health = wrapper.health_check()
    assert health["ok"] is False
    assert health["message"] == "Spark config incomplete"


def test_generate_json_extracts_json() -> None:
    wrapper = SparkWrapper(settings=Settings(enable_mock_llm=True))

    wrapper.generate = lambda *args, **kwargs: {  # type: ignore[method-assign]
        "content": '{"pass": true, "score": 91}',
        "model_provider": "mock",
        "model_name": "mock",
        "fallback_used": False,
        "raw": {},
    }
    parsed = wrapper.generate_json([{"role": "user", "content": "json"}])
    assert parsed["pass"] is True
    assert parsed["score"] == 91

    wrapper.generate = lambda *args, **kwargs: {  # type: ignore[method-assign]
        "content": "```json\n{\"pass\": false, \"score\": 70}\n```",
        "model_provider": "mock",
        "model_name": "mock",
        "fallback_used": True,
        "raw": {},
    }
    parsed_block = wrapper.generate_json([{"role": "user", "content": "json"}])
    assert parsed_block["pass"] is False
    assert parsed_block["score"] == 70
    assert parsed_block["fallback_used"] is True


def test_openai_compatible_wrapper_calls_generic_chat_completions() -> None:
    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "OpenAI-compatible says hi"}}]}

    class DummyClient:
        def __init__(self) -> None:
            self.called_with: dict | None = None

        def post(self, url: str, headers: dict[str, str], json: dict) -> DummyResponse:
            self.called_with = {"url": url, "headers": headers, "json": json}
            return DummyResponse()

    client = DummyClient()
    wrapper = SparkWrapper(
        settings=Settings(
            enable_mock_llm=False,
            enable_llm_fallback=False,
            llm_provider="openai-compatible",
            openai_api_key="sk-test-key",
            openai_model="gpt-4o-mini",
            openai_api_base="https://example.com/v1/chat/completions",
        ),
        client=client,  # type: ignore[arg-type]
    )

    response = wrapper.generate([{"role": "user", "content": "hello"}])
    assert response["model_provider"] == "openai-compatible"
    assert response["model_name"] == "gpt-4o-mini"
    assert response["content"] == "OpenAI-compatible says hi"
    assert client.called_with is not None
    assert client.called_with["url"] == "https://example.com/v1/chat/completions"
    assert client.called_with["headers"]["Authorization"] == "Bearer sk-test-key"
