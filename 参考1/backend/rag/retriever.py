from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .document_loader import DEFAULT_DATA_DIR
from .vector_store import VectorStore


DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "processed" / "faiss_index"


class RagRetriever:
    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR, snippet_chars: int = 800) -> None:
        self.index_dir = Path(index_dir)
        self.snippet_chars = snippet_chars
        self.store = VectorStore()
        self._loaded = False

    def load(self) -> None:
        self.store.load_index(self.index_dir)
        self._loaded = True

    def is_ready(self) -> bool:
        return (self.index_dir / "metadata.json").exists()

    def metadata(self) -> dict[str, Any]:
        metadata_path = self.index_dir / "metadata.json"
        if not metadata_path.exists():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        if not query or not query.strip():
            metadata = self.metadata()
            return {
                "query": query,
                "retrieval_backend": metadata.get("backend", self.store.backend),
                "chunk_count": int(metadata.get("chunk_count", 0) or 0),
                "results": [],
            }
        if not self._loaded:
            self.load()
        raw_results = self.store.search(query.strip(), top_k=top_k)
        results = []
        for chunk in raw_results:
            content = str(chunk.get("content", "")).strip()
            if len(content) > self.snippet_chars:
                content = content[: self.snippet_chars].rstrip() + "..."
            results.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "content": content,
                    "source_file": chunk.get("source_file"),
                    "module_name": chunk.get("module_name"),
                    "heading_path": chunk.get("heading_path", []),
                    "score": chunk.get("score", 0.0),
                }
            )
        return {
            "query": query,
            "retrieval_backend": self.store.backend,
            "chunk_count": len(self.store.chunks),
            "results": results,
        }
