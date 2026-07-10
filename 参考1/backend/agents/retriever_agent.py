from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from backend.rag.retriever import RagRetriever


class RetrieverAgent:
    """Agent-facing wrapper around the course RAG retriever."""

    def __init__(self, index_dir: str | Path | None = None) -> None:
        self.retriever = RagRetriever(index_dir=index_dir) if index_dir else RagRetriever()

    def retrieve(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not self.retriever.is_ready():
            return {
                "query": query,
                "results": [],
                "error": "Knowledge index is missing. Call /api/knowledge/build_index or run python scripts/build_sample_index.py first.",
            }
        return self.retriever.search(query, top_k=top_k)

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        return self.retrieve(query, top_k=top_k)
