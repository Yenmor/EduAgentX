from __future__ import annotations

from fastapi.testclient import TestClient

from backend.agents.judge_agent import JudgeAgent
from backend.agents.orchestrator import EduAgentOrchestrator
from backend.app import app
from backend.rag.ingest import build_sample_course_index
from backend.schemas.chat import ChatRequest


client = TestClient(app)


def test_chat_generates_five_resources() -> None:
    build_sample_course_index()
    response = client.post("/api/chat", json={"message": "我不懂 RAG，帮我生成学习资料", "student_id": "orch-resource"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "generate_resource"
    assert len(payload["result"]) == 5
    assert {item["type"] for item in payload["result"]} == {
        "explanation_doc",
        "mindmap",
        "quiz",
        "reading_material",
        "code_lab",
    }
    assert all(item["sources"] for item in payload["result"])
    assert payload["agent_trace"]


def test_chat_answers_question_with_sources() -> None:
    build_sample_course_index()
    response = client.post("/api/chat", json={"message": "Self-Attention 是什么", "student_id": "orch-qa"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "qa"
    assert payload["result"]["answer"]
    assert payload["result"]["sources"]
    assert "confidence" in payload["result"]


class OneFailJudge(JudgeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def judge_content(self, content, sources=None, profile=None, intent="qa"):
        self.calls += 1
        if self.calls == 1:
            return {
                "pass": False,
                "score": 50,
                "issues": ["forced low score"],
                "rewrite_instruction": "Rewrite with stronger grounding.",
            }
        return {
            "pass": True,
            "score": 88,
            "issues": [],
            "rewrite_instruction": "",
        }


def test_orchestrator_rewrites_once_when_judge_fails() -> None:
    build_sample_course_index()
    orchestrator = EduAgentOrchestrator(judge_agent=OneFailJudge())
    payload = orchestrator.run(ChatRequest(message="什么是 RAG", student_id="rewrite-test", mode="qa"))
    assert payload["rewritten"] is True
    assert payload["judge"]["pass"] is True
    assert any(step["action"] == "reviewed_after_rewrite" for step in payload["agent_trace"])
