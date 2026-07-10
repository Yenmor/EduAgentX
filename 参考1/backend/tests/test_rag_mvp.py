from __future__ import annotations

import zipfile
from pathlib import Path
import shutil

from backend.rag.chunker import chunk_documents
from backend.rag.course_manifest import build_manifest, manifest_by_source
from backend.rag.document_loader import extract_zip, load_markdown_documents
from backend.rag.vector_store import VectorStore


def test_zip_manifest_chunk_and_search() -> None:
    tmp_path = Path("data/test_tmp/rag_mvp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    raw_md = tmp_path / "python_for_beginners.md"
    raw_md.write_text("# Python Basics\n\nTransformer and LLM Agent examples for testing.", encoding="utf-8")
    zip_path = tmp_path / "knowledge.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(raw_md, arcname=raw_md.name)

    extract_dir = extract_zip(zip_path, tmp_path / "raw")
    docs = load_markdown_documents(extract_dir)
    manifest = build_manifest(docs)
    chunks = chunk_documents(docs, manifest_by_source(manifest))

    assert len(docs) == 1
    assert manifest[0]["course_name"]
    assert chunks[0]["heading_path"] == ["Python Basics"]

    store = VectorStore()
    store.build_index(chunks)
    results = store.search("What is LLM Agent?", top_k=1)
    assert results
    assert results[0]["source_file"] == "python_for_beginners.md"
