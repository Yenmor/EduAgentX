from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.rag.ingest import DEFAULT_RAW_ZIP, ingest_knowledge


client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def full_knowledge_index() -> None:
    summary = ingest_knowledge(zip_path=DEFAULT_RAW_ZIP)
    assert summary["document_count"] == 15
    assert summary["chunk_count"] > 100


def _search_sources(query: str, top_k: int = 8) -> set[str]:
    response = client.post("/api/knowledge/search", json={"query": query, "top_k": top_k})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == query
    assert payload["retrieval_backend"] in {"faiss", "fallback"}
    assert payload["chunk_count"] > 100
    assert payload["results"]
    return {item["source_file"] for item in payload["results"]}


def test_status_reports_active_full_knowledge_index() -> None:
    response = client.get("/api/knowledge/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["active_index_source"] == "knowledge.zip"
    assert payload["document_count"] == 15
    assert payload["chunk_count"] > 100
    assert payload["retrieval_backend"] in {"faiss", "fallback"}
    assert isinstance(payload["faiss_available"], bool)
    assert payload["manifest_loaded"] is True
    assert payload["tree_loaded"] is True


def test_search_transformer_returns_transformer_document() -> None:
    assert "foundations-transformers-architecture.md" in _search_sources("Transformer")


def test_search_llm_agent_returns_agent_document() -> None:
    assert "intro-llm-agents.md" in _search_sources("LLM Agent")


def test_search_rlhf_returns_rlhf_document() -> None:
    assert "rlhf-reinforcement-learning-human-feedback.md" in _search_sources("RLHF")


def test_knowledge_tree_has_course_module_chapter_chunk_layers() -> None:
    response = client.get("/api/knowledge/tree")
    assert response.status_code == 200
    tree = response.json()
    assert tree["schema_version"] == "raptor_v2"
    assert tree["course"]["type"] == "course"
    assert tree["modules"]
    assert tree["modules"][0]["type"] == "module"
    assert tree["modules"][0]["chapter_count"] >= 1
    assert tree["modules"][0]["chunk_count"] >= 1
    assert tree["chapters"]
    assert tree["chapters"][0]["type"] == "chapter"
    assert tree["chapters"][0]["heading_path"]
    assert tree["chapters"][0]["chunk_ids"]
    assert tree["chunks"]
    assert tree["chunks"][0]["type"] == "chunk"


def test_sample_build_does_not_overwrite_active_knowledge_index() -> None:
    before = client.get("/api/knowledge/status").json()
    response = client.post("/api/knowledge/build_index", json={})
    assert response.status_code == 200
    sample_payload = response.json()
    assert sample_payload["active_index_source"] == "sample_course_demo"
    assert "sample_index" in sample_payload["index_dir"]
    assert Path(sample_payload["chunks_path"]).name == "chunks.jsonl"

    after = client.get("/api/knowledge/status").json()
    assert after["active_index_source"] == "knowledge.zip"
    assert after["document_count"] == before["document_count"] == 15
    assert after["chunk_count"] == before["chunk_count"]
