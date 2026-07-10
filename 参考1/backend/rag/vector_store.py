from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from .embedder import MockEmbedder

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional native package
    faiss = None

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - pure Python fallback is supported
    np = None


class VectorStore:
    """FAISS-backed vector store with a deterministic local fallback."""

    def __init__(self, embedder: MockEmbedder | None = None) -> None:
        self.embedder = embedder or MockEmbedder()
        self.chunks: list[dict[str, Any]] = []
        self.embeddings: list[list[float]] = []
        self.index: Any = None
        self.backend = "fallback"
        self.metadata: dict[str, Any] = {}

    @property
    def faiss_available(self) -> bool:
        return faiss is not None and np is not None

    def _index_text(self, chunk: dict[str, Any]) -> str:
        heading = " ".join(str(item) for item in chunk.get("heading_path", []))
        tags = " ".join(str(item) for item in chunk.get("tags", []))
        return " ".join(
            [
                str(chunk.get("source_file", "")),
                str(chunk.get("title", "")),
                str(chunk.get("module_name", "")),
                heading,
                tags,
                str(chunk.get("content", "")),
            ]
        )

    def build_index(self, chunks: Iterable[dict[str, Any]]) -> None:
        self.chunks = list(chunks)
        texts = [self._index_text(chunk) for chunk in self.chunks]
        self.embeddings = self.embedder.embed_documents(texts)
        if self.faiss_available and self.embeddings:
            matrix = np.array(self.embeddings, dtype="float32")
            self.index = faiss.IndexFlatIP(matrix.shape[1])
            self.index.add(matrix)
            self.backend = "faiss"
        else:
            self.index = None
            self.backend = "fallback"

    def save_index(self, path: str | Path, extra_metadata: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend": self.backend,
            "chunk_count": len(self.chunks),
            "faiss_available": self.faiss_available,
            "chunks": self.chunks,
        }
        if extra_metadata:
            payload.update(extra_metadata)
        self.metadata = payload
        (path / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.backend == "faiss" and faiss is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
            if np is not None:
                np.save(path / "embeddings.npy", np.array(self.embeddings, dtype="float32"))
        else:
            (path / "embeddings.json").write_text(json.dumps(self.embeddings), encoding="utf-8")
        return path

    def load_index(self, path: str | Path) -> None:
        path = Path(path)
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Vector index metadata not found: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.metadata = dict(metadata)
        self.chunks = list(metadata.get("chunks", []))
        faiss_path = path / "index.faiss"
        if self.faiss_available and faiss_path.exists():
            self.index = faiss.read_index(str(faiss_path))
            self.backend = "faiss"
            if (path / "embeddings.npy").exists():
                self.embeddings = np.load(path / "embeddings.npy").astype("float32").tolist()
            else:
                self.embeddings = [self.embedder.embed_text(self._index_text(chunk)) for chunk in self.chunks]
            return
        embeddings_path = path / "embeddings.json"
        if embeddings_path.exists():
            self.embeddings = json.loads(embeddings_path.read_text(encoding="utf-8"))
        elif np is not None and (path / "embeddings.npy").exists():
            self.embeddings = np.load(path / "embeddings.npy").astype("float32").tolist()
        else:
            self.embeddings = [self.embedder.embed_text(self._index_text(chunk)) for chunk in self.chunks]
        self.index = None
        self.backend = "fallback"

    def _cosine_scores(self, query_vector: list[float]) -> list[float]:
        scores: list[float] = []
        for vector in self.embeddings:
            dot = sum(a * b for a, b in zip(query_vector, vector))
            denom = math.sqrt(sum(a * a for a in query_vector)) * math.sqrt(sum(b * b for b in vector)) or 1.0
            scores.append(dot / denom)
        return scores

    def _keyword_score(self, query: str, chunk: dict[str, Any]) -> float:
        query_lower = query.lower().strip()
        haystack = self._index_text(chunk).lower()
        tokens = [token for token in self.embedder._tokens(query) if len(token) > 1]
        unique_tokens = list(dict.fromkeys(tokens))
        score = 0.0
        if query_lower and query_lower in haystack:
            score += 0.8
        source = str(chunk.get("source_file", "")).lower()
        heading = " ".join(str(item) for item in chunk.get("heading_path", [])).lower()
        tags = " ".join(str(item) for item in chunk.get("tags", [])).lower()
        for token in unique_tokens:
            if token in source:
                score += 0.45
            if token in tags:
                score += 0.35
            if token in heading:
                score += 0.3
            if token in haystack:
                score += 0.08
        domain_pairs = {
            "transformer": ["transformer", "attention", "encoder", "decoder"],
            "llm agent": ["llm-agent", "agent", "tool-use", "planning"],
            "agent": ["agent", "workflow", "tool"],
            "rlhf": ["rlhf", "human-feedback", "alignment", "reinforcement"],
            "pytorch": ["pytorch", "torch", "tensor"],
            "machine learning": ["machine-learning", "supervised", "unsupervised"],
            "rag": ["rag", "retrieval", "augmented", "generation"],
        }
        for trigger, needles in domain_pairs.items():
            if trigger in query_lower:
                score += sum(0.12 for needle in needles if needle in haystack)
        return score

    def _keyword_candidates(self, query: str, limit: int) -> list[tuple[int, float]]:
        scored = [(idx, self._keyword_score(query, chunk)) for idx, chunk in enumerate(self.chunks)]
        scored = [(idx, score) for idx, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:limit]

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.chunks:
            return []
        top_k = max(1, min(top_k, len(self.chunks)))
        query_vector = self.embedder.embed_text(query)
        vector_limit = min(len(self.chunks), max(top_k * 8, 40))

        if self.backend == "faiss" and self.index is not None and np is not None:
            q = np.array([query_vector], dtype="float32")
            scores, indices = self.index.search(q, vector_limit)
            candidates = [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]
        else:
            scores = self._cosine_scores(query_vector)
            candidates = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:vector_limit]

        by_idx: dict[int, float] = {idx: float(score) for idx, score in candidates}
        for idx, keyword_score in self._keyword_candidates(query, max(top_k * 12, 80)):
            by_idx[idx] = max(by_idx.get(idx, 0.0), 0.0) + keyword_score

        ranked = []
        for idx, base_score in by_idx.items():
            keyword_score = self._keyword_score(query, self.chunks[idx])
            ranked.append((idx, float(base_score) + keyword_score))
        ranked.sort(key=lambda item: item[1], reverse=True)

        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for idx, score in ranked:
            chunk = self.chunks[idx]
            chunk_id = str(chunk.get("chunk_id"))
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            result = dict(chunk)
            result["score"] = round(float(score), 4)
            results.append(result)
            if len(results) >= top_k:
                break
        return results
