from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .chunker import chunk_documents, save_chunks_jsonl
from .course_manifest import build_manifest, manifest_by_source, save_manifest
from .document_loader import (
    DEFAULT_DATA_DIR,
    DEFAULT_RAW_ZIP,
    DEFAULT_SAMPLE_COURSE_DIR,
    copy_to_default_raw_zip,
    extract_zip,
    load_markdown_documents,
    load_sample_course_documents,
)
from .raptor_builder import build_raptor_tree, save_raptor_tree
from .vector_store import VectorStore


DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"
DEFAULT_CHUNKS_PATH = DEFAULT_PROCESSED_DIR / "chunks.jsonl"
DEFAULT_MANIFEST_PATH = DEFAULT_PROCESSED_DIR / "course_manifest.json"
DEFAULT_RAPTOR_TREE_PATH = DEFAULT_PROCESSED_DIR / "raptor_tree.json"
DEFAULT_INDEX_DIR = DEFAULT_PROCESSED_DIR / "faiss_index"
DEFAULT_SAMPLE_INDEX_DIR = DEFAULT_PROCESSED_DIR / "sample_index"
SAMPLE_COURSE_NAME = "LLM Agent Sample Course"


def _course_name_from_manifest(manifest: list[dict[str, object]], fallback: str) -> str:
    for entry in manifest:
        course_name = str(entry.get("course_name", "")).strip()
        if course_name:
            return course_name
    return fallback


def ingest_knowledge(
    zip_path: str | Path = DEFAULT_RAW_ZIP,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    copy_zip: bool = True,
) -> Dict[str, Any]:
    """Build the active production knowledge index from a Markdown zip."""
    data_dir = Path(data_dir)
    zip_path = Path(zip_path).expanduser()
    raw_zip_path = data_dir / "raw" / "knowledge.zip"
    extract_dir = data_dir / "sample_course" / "raw"
    processed_dir = data_dir / "processed"
    chunks_path = processed_dir / "chunks.jsonl"
    manifest_path = processed_dir / "course_manifest.json"
    tree_path = processed_dir / "raptor_tree.json"
    index_dir = processed_dir / "faiss_index"

    if copy_zip:
        zip_path = copy_to_default_raw_zip(zip_path, raw_zip_path)

    extract_zip(zip_path, extract_dir)
    documents = load_markdown_documents(extract_dir)
    manifest = build_manifest(documents)
    manifest_lookup = manifest_by_source(manifest)
    chunks = chunk_documents(documents, manifest_lookup)
    tree = build_raptor_tree(chunks, course_name=_course_name_from_manifest(manifest, "Knowledge Zip"))

    processed_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest, manifest_path)
    save_chunks_jsonl(chunks, chunks_path)
    save_raptor_tree(tree, tree_path)

    store = VectorStore()
    store.build_index(chunks)
    store.save_index(
        index_dir,
        extra_metadata={
            "index_source": "knowledge.zip",
            "active_index_source": "knowledge.zip",
            "document_count": len(documents),
            "source_zip_path": str(Path(zip_path).resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "chunks_path": str(chunks_path.resolve()),
            "raptor_tree_path": str(tree_path.resolve()),
        },
    )

    summary = {
        "zip_path": str(Path(zip_path).resolve()),
        "raw_dir": str(extract_dir.resolve()),
        "processed_dir": str(processed_dir.resolve()),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "manifest_path": str(manifest_path.resolve()),
        "chunks_path": str(chunks_path.resolve()),
        "raptor_tree_path": str(tree_path.resolve()),
        "index_dir": str(index_dir.resolve()),
        "vector_backend": store.backend,
        "retrieval_backend": store.backend,
        "active_index_source": "knowledge.zip",
    }
    (processed_dir / "ingest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (processed_dir / "build_index_summary.json").write_text(
        json.dumps(
            {
                "deprecated": True,
                "message": "Sample demo indexes are now written to data/processed/sample_index and do not replace the active knowledge index.",
                "active_index_source": "knowledge.zip",
                "sample_index_summary": str((processed_dir / "sample_index" / "build_index_summary.json").resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def _sample_manifest_for_documents(documents: list) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for doc in documents:
        if doc.source_file.lower() == "readme.md":
            continue
        first_heading = SAMPLE_COURSE_NAME
        for line in doc.content.splitlines():
            if line.startswith("# "):
                first_heading = line[2:].strip()
                break
        manifest.append(
            {
                "course_id": "llm-agent-practice-sample",
                "course_name": SAMPLE_COURSE_NAME,
                "module_name": "Demo Sample",
                "source_file": doc.source_file,
                "title": first_heading,
                "estimated_level": "intermediate",
                "tags": ["llm", "rag", "agent", "langgraph", "evaluation", "sample"],
            }
        )
    return manifest


def build_sample_course_index(
    sample_dir: str | Path = DEFAULT_SAMPLE_COURSE_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Build a demo-only sample index without replacing the active knowledge index."""
    data_dir = Path(data_dir)
    processed_dir = Path(output_dir) if output_dir is not None else data_dir / "processed" / "sample_index"
    chunks_path = processed_dir / "chunks.jsonl"
    manifest_path = processed_dir / "course_manifest.json"
    tree_path = processed_dir / "raptor_tree.json"
    index_dir = processed_dir / "faiss_index"

    documents = [doc for doc in load_sample_course_documents(sample_dir) if doc.source_file.lower() != "readme.md"]
    if not documents:
        raise FileNotFoundError(f"No sample-course Markdown files found in {Path(sample_dir).resolve()}")

    manifest = _sample_manifest_for_documents(documents)
    manifest_lookup = manifest_by_source(manifest)
    chunks = chunk_documents(documents, manifest_lookup)
    tree = build_raptor_tree(chunks, course_name=SAMPLE_COURSE_NAME)

    processed_dir.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest, manifest_path)
    save_chunks_jsonl(chunks, chunks_path)
    save_raptor_tree(tree, tree_path)

    store = VectorStore()
    store.build_index(chunks)
    store.save_index(
        index_dir,
        extra_metadata={
            "index_source": "sample_course",
            "active_index_source": "sample_course_demo",
            "document_count": len(documents),
            "manifest_path": str(manifest_path.resolve()),
            "chunks_path": str(chunks_path.resolve()),
            "raptor_tree_path": str(tree_path.resolve()),
        },
    )

    summary = {
        "sample_dir": str(Path(sample_dir).resolve()),
        "processed_dir": str(processed_dir.resolve()),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "manifest_path": str(manifest_path.resolve()),
        "chunks_path": str(chunks_path.resolve()),
        "raptor_tree_path": str(tree_path.resolve()),
        "index_dir": str(index_dir.resolve()),
        "vector_backend": store.backend,
        "retrieval_backend": store.backend,
        "active_index_source": "sample_course_demo",
        "note": "Sample indexes are demo-only and do not replace data/processed/faiss_index.",
    }
    (processed_dir / "build_index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
