from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable


DEFAULT_COURSE_NAME = "AI Large Model Application Development Practice"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def rule_summary(text: str, limit: int = 220) -> str:
    compact = _clean_text(text)
    if len(compact) <= limit:
        return compact
    boundary = compact.rfind(".", 0, limit)
    if boundary < 80:
        boundary = compact.rfind("。", 0, limit)
    if boundary < 80:
        boundary = limit
    return compact[:boundary].rstrip() + "..."


def summarize_with_llm(text: str, spark_wrapper: Any | None = None, max_chars: int = 240) -> str:
    """Optional LLM summarization hook; falls back to a deterministic rule summary."""
    if spark_wrapper is None:
        return rule_summary(text, max_chars)
    prompt = (
        "Summarize the following learning content as a concise knowledge-tree node. "
        "Keep the summary factual and under 80 words.\n\n"
        f"{text[:4000]}"
    )
    try:
        if hasattr(spark_wrapper, "generate_text"):
            summary = str(spark_wrapper.generate_text(prompt, temperature=0.2, max_tokens=180))
        else:
            response = spark_wrapper.generate(prompt, temperature=0.2, max_tokens=180)
            summary = str(response.get("content", ""))
        return rule_summary(summary, max_chars) if summary.strip() else rule_summary(text, max_chars)
    except Exception:
        return rule_summary(text, max_chars)


def _node_id(prefix: str, value: str, ordinal: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = slug[:48] or f"{prefix}-{ordinal}"
    return f"{prefix}-{ordinal:03d}-{slug}"


def _heading_key(chunk: dict[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    heading_path = tuple(str(item) for item in chunk.get("heading_path", []) if str(item).strip())
    if not heading_path:
        heading_path = (str(chunk.get("title") or chunk.get("source_file") or "Untitled"),)
    return (
        str(chunk.get("module_name") or "Unassigned Module"),
        str(chunk.get("source_file") or "unknown.md"),
        heading_path,
    )


def build_raptor_tree(
    chunks: Iterable[dict[str, Any]],
    course_name: str = DEFAULT_COURSE_NAME,
    spark_wrapper: Any | None = None,
) -> Dict[str, Any]:
    """Build a RAPTOR-style hierarchy: course -> module -> chapter/heading -> chunk."""
    chunk_list = list(chunks)
    modules: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    chapters: "OrderedDict[tuple[str, str, tuple[str, ...]], dict[str, Any]]" = OrderedDict()
    chunk_nodes: list[dict[str, Any]] = []

    for chunk in chunk_list:
        module_name = str(chunk.get("module_name") or "Unassigned Module")
        module = modules.setdefault(
            module_name,
            {
                "node_id": _node_id("module", module_name, len(modules) + 1),
                "type": "module",
                "module_name": module_name,
                "summary": "",
                "chapter_count": 0,
                "chunk_count": 0,
                "child_ids": [],
                "_summary_parts": [],
            },
        )

        heading_key = _heading_key(chunk)
        heading_path = list(heading_key[2])
        chapter_title = heading_path[-1] if heading_path else str(chunk.get("title") or "Untitled")
        chapter = chapters.setdefault(
            heading_key,
            {
                "node_id": _node_id("chapter", f"{module_name}-{chapter_title}", len(chapters) + 1),
                "type": "chapter",
                "module_id": module["node_id"],
                "module_name": module_name,
                "title": chapter_title,
                "source_file": heading_key[1],
                "heading_path": heading_path,
                "summary": "",
                "chunk_ids": [],
                "child_ids": [],
                "_summary_parts": [],
            },
        )
        if chapter["node_id"] not in module["child_ids"]:
            module["child_ids"].append(chapter["node_id"])

        chunk_id = str(chunk.get("chunk_id"))
        chunk_summary = rule_summary(str(chunk.get("content", "")), 140)
        chunk_node = {
            "node_id": chunk_id,
            "type": "chunk",
            "chunk_id": chunk_id,
            "source_file": chunk.get("source_file"),
            "module_name": module_name,
            "heading_path": heading_path,
            "summary": chunk_summary,
            "start_char": chunk.get("start_char"),
            "end_char": chunk.get("end_char"),
        }
        chunk_nodes.append(chunk_node)
        chapter["chunk_ids"].append(chunk_id)
        chapter["child_ids"].append(chunk_id)
        chapter["_summary_parts"].append(chunk_summary)
        module["_summary_parts"].append(chunk_summary)
        module["chunk_count"] += 1

    chapter_nodes: list[dict[str, Any]] = []
    for chapter in chapters.values():
        chapter["summary"] = summarize_with_llm(" ".join(chapter["_summary_parts"][:12]), spark_wrapper)
        chapter.pop("_summary_parts", None)
        chapter_nodes.append(chapter)

    module_nodes: list[dict[str, Any]] = []
    for module in modules.values():
        module["chapter_count"] = len(module["child_ids"])
        module["summary"] = summarize_with_llm(" ".join(module["_summary_parts"][:16]), spark_wrapper)
        module.pop("_summary_parts", None)
        module_nodes.append(module)

    course_id = "course-root"
    course_summary = summarize_with_llm(
        " ".join(module["module_name"] + ": " + module["summary"] for module in module_nodes),
        spark_wrapper,
        max_chars=320,
    )
    course_node = {
        "node_id": course_id,
        "type": "course",
        "course_name": course_name,
        "summary": course_summary,
        "module_count": len(module_nodes),
        "chapter_count": len(chapter_nodes),
        "chunk_count": len(chunk_nodes),
        "child_ids": [module["node_id"] for module in module_nodes],
    }

    nodes = [course_node, *module_nodes, *chapter_nodes, *chunk_nodes]
    return {
        "schema_version": "raptor_v2",
        "course_name": course_name,
        "course": course_node,
        "modules": module_nodes,
        "chapters": chapter_nodes,
        "chunks": chunk_nodes,
        "nodes": nodes,
        "course_summary": {
            "title": course_name,
            "summary": course_summary,
            "module_count": len(module_nodes),
            "chapter_count": len(chapter_nodes),
            "chunk_count": len(chunk_nodes),
        },
        "chapter_summaries": chapter_nodes,
    }


def save_raptor_tree(tree: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tree, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def load_raptor_tree(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
