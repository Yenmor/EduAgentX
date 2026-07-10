from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from .document_loader import MarkdownDocument


COURSE_ID = "ai-large-model-app-dev-practice"
COURSE_NAME = "人工智能与大模型应用开发实践"

COURSE_MODULES: dict[str, list[str]] = {
    "编程与工程基础": [
        "python_for_beginners.md",
        "getting-started-with-git.md",
        "essential-numpy-pandas.md",
    ],
    "机器学习与深度学习基础": [
        "introduction-to-machine-learning.md",
        "introduction-to-neural-networks.md",
        "getting-started-with-pytorch.md",
    ],
    "计算机视觉与图学习": [
        "introduction-to-computer-vision.md",
        "cnns-for-computer-vision.md",
        "introduction-to-graph-neural-networks.md",
    ],
    "Transformer 与大语言模型": [
        "foundations-transformers-architecture.md",
        "how-to-build-a-large-language-model.md",
    ],
    "LLM 应用开发与智能体": [
        "langchain-production-llm.md",
        "intro-llm-agents.md",
    ],
    "强化学习与对齐": [
        "advanced-reinforcement-learning.md",
        "rlhf-reinforcement-learning-human-feedback.md",
    ],
}

FILENAME_TO_MODULE = {
    filename: module_name
    for module_name, filenames in COURSE_MODULES.items()
    for filename in filenames
}

MODULE_LEVEL = {
    "编程与工程基础": "beginner",
    "机器学习与深度学习基础": "beginner",
    "计算机视觉与图学习": "intermediate",
    "Transformer 与大语言模型": "intermediate",
    "LLM 应用开发与智能体": "intermediate",
    "强化学习与对齐": "advanced",
}

EXTRA_TAGS = {
    "python_for_beginners.md": ["python", "programming"],
    "getting-started-with-git.md": ["git", "engineering"],
    "essential-numpy-pandas.md": ["numpy", "pandas", "data"],
    "introduction-to-machine-learning.md": ["machine-learning", "ml"],
    "introduction-to-neural-networks.md": ["neural-network", "deep-learning"],
    "getting-started-with-pytorch.md": ["pytorch", "deep-learning"],
    "introduction-to-computer-vision.md": ["computer-vision", "cv"],
    "cnns-for-computer-vision.md": ["cnn", "computer-vision"],
    "introduction-to-graph-neural-networks.md": ["gnn", "graph"],
    "foundations-transformers-architecture.md": ["transformer", "attention", "llm"],
    "how-to-build-a-large-language-model.md": ["llm", "large-language-model"],
    "langchain-production-llm.md": ["langchain", "llmops", "llm"],
    "intro-llm-agents.md": ["llm-agent", "agent", "tool-use"],
    "advanced-reinforcement-learning.md": ["reinforcement-learning", "rl"],
    "rlhf-reinforcement-learning-human-feedback.md": ["rlhf", "alignment"],
}


def title_from_markdown(content: str, source_file: str) -> str:
    for line in content.splitlines():
        match = re.match(r"^\s*#\s+(.+?)\s*#*\s*$", line)
        if match:
            return match.group(1).strip()
    return Path(source_file).stem.replace("-", " ").replace("_", " ").title()


def tags_for_file(source_file: str, module_name: str) -> list[str]:
    filename = Path(source_file).name
    stem_tags = [part for part in re.split(r"[-_\s]+", Path(filename).stem.lower()) if part]
    module_tags = [part.lower() for part in re.split(r"\s+|与|及", module_name) if part]
    return sorted(set(stem_tags + module_tags + EXTRA_TAGS.get(filename, [])))


def build_manifest(documents: Iterable[MarkdownDocument]) -> List[Dict[str, object]]:
    entries: list[dict[str, object]] = []
    for doc in documents:
        filename = Path(doc.source_file).name
        module_name = FILENAME_TO_MODULE.get(filename, "未分类资料")
        entries.append(
            {
                "course_id": COURSE_ID,
                "course_name": COURSE_NAME,
                "module_name": module_name,
                "source_file": doc.source_file,
                "title": title_from_markdown(doc.content, doc.source_file),
                "estimated_level": MODULE_LEVEL.get(module_name, "intermediate"),
                "tags": tags_for_file(doc.source_file, module_name),
            }
        )
    return entries


def manifest_by_source(manifest: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(entry["source_file"]): dict(entry) for entry in manifest}


def save_manifest(manifest: List[Dict[str, object]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def load_manifest(path: str | Path) -> List[Dict[str, object]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
