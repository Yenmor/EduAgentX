from __future__ import annotations

import zipfile
from pathlib import Path

from backend.rag.document_loader import build_default_raw_zip_from_sample_course


def test_build_default_raw_zip_from_sample_course_creates_expected_archive() -> None:
    tmp_path = Path("data/test_tmp/knowledge_bootstrap")
    tmp_path.mkdir(parents=True, exist_ok=True)
    zip_path = tmp_path / "knowledge.zip"

    build_default_raw_zip_from_sample_course(target_path=zip_path)

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(Path(name).name for name in archive.namelist() if name.endswith(".md"))

    assert len(names) == 15
    assert "foundations-transformers-architecture.md" in names
    assert "intro-llm-agents.md" in names
