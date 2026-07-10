from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app import app


client = TestClient(app)


def test_profile_update_and_get() -> None:
    response = client.post(
        "/api/profile/update",
        json={
            "student_id": "api-test-student",
            "name": "API Test",
            "text": "我的目标是学习 LLM Agent，希望通过代码实践掌握 Transformer。",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["student_id"] == "api-test-student"
    assert payload["profile"]["preferred_style"] == "code_lab"

    get_response = client.get("/api/profile/api-test-student")
    assert get_response.status_code == 200
    assert get_response.json()["name"] == "API Test"


def test_resource_generation_returns_five_types() -> None:
    response = client.post(
        "/api/resources/generate",
        json={"student_id": "resource-test-student", "topic": "Transformer", "knowledge_point": "Transformer"},
    )
    assert response.status_code == 200
    payload = response.json()
    resource_types = {item["type"] for item in payload["resources"]}
    assert {"explanation_doc", "mindmap", "quiz", "reading_material", "code_lab"} <= resource_types
    assert isinstance(payload["judge_score"], int)


def test_path_generation_returns_steps() -> None:
    response = client.post(
        "/api/path/generate",
        json={"student_id": "path-test-student", "goal": "Build an LLM Agent", "horizon_days": 21},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["steps"]) >= 3
    assert payload["steps"][0]["order"] == 1


def test_assessment_updates_mastery() -> None:
    student_id = f"assessment-test-{uuid4().hex}"
    response = client.post(
        "/api/assessment/submit",
        json={
            "student_id": student_id,
            "knowledge_point": "Transformer",
            "is_correct": True,
            "score": 0.9,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mastery_probability"] != payload["previous_mastery_probability"]
    assert payload["mastery_probability"] > 0
