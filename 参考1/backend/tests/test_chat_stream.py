from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app
from backend.rag.ingest import build_sample_course_index


client = TestClient(app)


def test_chat_stream_emits_workflow_events() -> None:
    build_sample_course_index()
    with client.stream(
        "POST",
        "/api/chat/stream",
        json={"message": "I do not understand RAG, generate learning resources.", "student_id": "stream-test"},
    ) as response:
        assert response.status_code == 200
        body = response.read().decode("utf-8")

    for event_name in [
        "profile_update",
        "intent_detected",
        "retrieve_start",
        "retrieve_done",
        "generate_start",
        "token",
        "judge_start",
        "judge_done",
        "final",
    ]:
        assert f"event: {event_name}" in body
    assert '"intent": "generate_resource"' in body
