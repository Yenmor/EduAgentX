from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_RAW_ZIP = DEFAULT_DATA_DIR / "raw" / "knowledge.zip"
DEFAULT_EXTRACT_DIR = DEFAULT_DATA_DIR / "sample_course" / "raw"
DEFAULT_SAMPLE_COURSE_DIR = DEFAULT_DATA_DIR / "sample_course"


@dataclass(frozen=True)
class MarkdownDocument:
    source_file: str
    path: Path
    content: str


def _safe_target(base_dir: Path, member_name: str) -> Path:
    target = (base_dir / member_name).resolve()
    base = base_dir.resolve()
    if target != base and base not in target.parents:
        raise ValueError(f"Unsafe zip entry path: {member_name}")
    return target


def extract_zip(zip_path: str | Path, output_dir: str | Path = DEFAULT_EXTRACT_DIR, clear: bool = True) -> Path:
    """Extract a knowledge zip into the sample course raw directory."""
    zip_path = Path(zip_path).expanduser().resolve()
    output_dir = Path(output_dir).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Knowledge zip not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip file: {zip_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if clear:
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                _safe_target(output_dir, info.filename).mkdir(parents=True, exist_ok=True)
                continue
            target = _safe_target(output_dir, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
    return output_dir


def copy_to_default_raw_zip(zip_path: str | Path, target_path: str | Path = DEFAULT_RAW_ZIP) -> Path:
    """Copy an arbitrary source zip to data/raw/knowledge.zip for repeatable ingestion."""
    source = Path(zip_path).expanduser().resolve()
    target = Path(target_path).resolve()
    if not source.exists():
        if source == DEFAULT_RAW_ZIP.resolve():
            return build_default_raw_zip_from_sample_course(target_path=target)
        raise FileNotFoundError(f"Knowledge zip not found: {source}")
    if source == target and not zipfile.is_zipfile(source):
        return build_default_raw_zip_from_sample_course(target_path=target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source != target:
        shutil.copy2(source, target)
    return target


def build_default_raw_zip_from_sample_course(
    target_path: str | Path = DEFAULT_RAW_ZIP,
    sample_dir: str | Path = DEFAULT_SAMPLE_COURSE_DIR,
) -> Path:
    """Package the built-in sample Markdown set into a default knowledge zip."""
    target = Path(target_path).resolve()
    source_dir = Path(sample_dir).resolve() / "raw"
    markdown_files = sorted(source_dir.glob("*.md"))
    if not markdown_files:
        raise FileNotFoundError(f"No sample-course Markdown files found in {source_dir}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(target.suffix + ".tmp")
    with zipfile.ZipFile(temp_target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in markdown_files:
            archive.write(path, arcname=path.name)
    temp_target.replace(target)
    return target


def load_markdown_documents(
    raw_dir: str | Path = DEFAULT_EXTRACT_DIR,
    exclude_dirs: set[str] | None = None,
) -> List[MarkdownDocument]:
    raw_dir = Path(raw_dir).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw course directory not found: {raw_dir}")

    exclude_dirs = exclude_dirs or set()
    documents: list[MarkdownDocument] = []
    for path in sorted(raw_dir.rglob("*.md")):
        if any(part in exclude_dirs for part in path.relative_to(raw_dir).parts[:-1]):
            continue
        relative = path.relative_to(raw_dir).as_posix()
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            content = path.read_text(encoding="utf-8", errors="replace")
        documents.append(MarkdownDocument(source_file=relative, path=path, content=content))
    return documents


def iter_markdown_files(raw_dir: str | Path = DEFAULT_EXTRACT_DIR) -> Iterable[Path]:
    raw_dir = Path(raw_dir)
    return sorted(raw_dir.rglob("*.md"))


def load_sample_course_documents(sample_dir: str | Path = DEFAULT_SAMPLE_COURSE_DIR) -> List[MarkdownDocument]:
    """Load first-party sample-course Markdown, excluding extracted raw zip content."""
    return load_markdown_documents(sample_dir, exclude_dirs={"raw", "processed", "test_tmp"})
