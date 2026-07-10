from __future__ import annotations

from backend.agents.profile_agent import ProfileAgent
from backend.config.settings import Settings
from backend.schemas.chat import ChatRequest
from backend.services.spark_wrapper import SparkWrapper


def test_agents_use_spark_wrapper() -> None:
    agent = ProfileAgent(llm=SparkWrapper(settings=Settings(enable_mock_llm=True)))
    response = agent.run(ChatRequest(message="请更新我的学习画像，我想学 RAG", student_id="agent-spark-test"))
    assert response.model_provider == "mock"
    assert response.model_name == "mock"
    assert response.fallback_used is False
