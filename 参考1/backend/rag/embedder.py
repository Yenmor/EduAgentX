from __future__ import annotations

import hashlib
import math
import re
from typing import List


DOMAIN_ALIASES = {
    "rag": ["rag", "retrieval", "augmented", "generation", "context"],
    "检索增强生成": ["rag", "retrieval", "augmented", "generation"],
    "向量检索": ["embedding", "vector", "retrieval", "similarity"],
    "embedding": ["embedding", "vector", "semantic", "similarity"],
    "prompt": ["prompt", "instruction", "context", "example"],
    "function calling": ["function", "calling", "tool", "schema"],
    "agent": ["agent", "tool", "planning", "memory", "workflow"],
    "智能体": ["agent", "tool", "planning", "memory", "workflow"],
    "langgraph": ["langgraph", "graph", "state", "workflow"],
    "幻觉": ["hallucination", "evaluation", "grounding", "safety"],
    "评估": ["evaluation", "judge", "rubric", "metrics"],
    "transformer": ["transformer", "attention", "encoder", "decoder"],
    "llm": ["llm", "large", "language", "model"],
    "大模型": ["llm", "large", "language", "model"],
}


class MockEmbedder:
    """Deterministic local embedding with domain-aware token expansion."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _tokens(self, text: str) -> list[str]:
        lowered = text.lower()
        tokens = re.findall(r"[a-z0-9][a-z0-9_\-+.]*|[\u4e00-\u9fff]+", lowered)
        expanded: list[str] = []
        for token in tokens:
            expanded.append(token)
            if len(token) > 1 and re.search(r"[\u4e00-\u9fff]", token):
                expanded.extend(token[i : i + 2] for i in range(max(0, len(token) - 1)))
        for key, values in DOMAIN_ALIASES.items():
            if key in lowered:
                expanded.extend(values)
        return expanded

    def embed_text(self, text: str) -> List[float]:
        vector = [0.0] * self.dim
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            weight = 1.0 + min(len(token), 12) / 24.0
            vector[slot] += sign * weight
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]


def embed_text(text: str) -> List[float]:
    return MockEmbedder().embed_text(text)


def embed_documents(texts: List[str]) -> List[List[float]]:
    return MockEmbedder().embed_documents(texts)
