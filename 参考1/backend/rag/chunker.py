from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

from .document_loader import MarkdownDocument


DEFAULT_COURSE_NAME = "大模型应用开发与智能体实践"
MIN_CHARS = 350
MAX_CHARS = 900
HARD_MAX_CHARS = 1100


@dataclass
class MarkdownSection:
    heading_path: list[str]
    text: str
    start_char: int
    end_char: int


def _is_heading(line: str) -> re.Match[str] | None:
    return re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line.strip())


def _clean_heading(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip()


def split_markdown_sections(content: str) -> list[MarkdownSection]:
    lines = content.splitlines(keepends=True)
    sections: list[MarkdownSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_heading: list[str] = []
    current_start = 0
    offset = 0

    def flush(end_offset: int) -> None:
        nonlocal current_lines, current_heading, current_start
        text = "".join(current_lines).strip()
        if text:
            sections.append(
                MarkdownSection(
                    heading_path=list(current_heading),
                    text=text,
                    start_char=current_start,
                    end_char=end_offset,
                )
            )
        current_lines = []

    for line in lines:
        line_start = offset
        offset += len(line)
        match = _is_heading(line)
        if match:
            flush(line_start)
            level = len(match.group(1))
            title = _clean_heading(match.group(2))
            heading_stack = [(h_level, h_title) for h_level, h_title in heading_stack if h_level < level]
            heading_stack.append((level, title))
            current_heading = [h_title for _, h_title in heading_stack]
            current_start = line_start
            current_lines = [line]
        else:
            if not current_lines:
                current_start = line_start
                current_heading = [h_title for _, h_title in heading_stack]
            current_lines.append(line)
    flush(len(content))
    return sections


def _split_long_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"(\n\s*\n)", text)
    pieces: list[str] = []
    current = ""
    for part in paragraphs:
        if not part:
            continue
        if len(part) > HARD_MAX_CHARS:
            if current.strip():
                pieces.append(current.strip())
                current = ""
            for start in range(0, len(part), max_chars):
                piece = part[start : start + max_chars].strip()
                if piece:
                    pieces.append(piece)
            continue
        if len(current) + len(part) > max_chars and current.strip():
            pieces.append(current.strip())
            current = part
        else:
            current += part
    if current.strip():
        pieces.append(current.strip())
    return [piece for piece in pieces if piece]


def _chunk_sections(sections: list[MarkdownSection]) -> list[MarkdownSection]:
    chunks: list[MarkdownSection] = []
    pending: MarkdownSection | None = None

    for section in sections:
        parts = _split_long_text(section.text)
        if len(parts) > 1:
            if pending:
                chunks.append(pending)
                pending = None
            cursor = section.start_char
            for part in parts:
                start = section.text.find(part, max(0, cursor - section.start_char))
                absolute_start = section.start_char + (start if start >= 0 else 0)
                chunks.append(MarkdownSection(section.heading_path, part, absolute_start, absolute_start + len(part)))
                cursor = absolute_start + len(part)
            continue

        if pending is None:
            pending = section
            continue

        same_heading = pending.heading_path == section.heading_path
        merged_text = f"{pending.text}\n\n{section.text}"
        if same_heading and len(merged_text) <= MAX_CHARS:
            pending = MarkdownSection(pending.heading_path, merged_text, pending.start_char, section.end_char)
        elif len(pending.text) < MIN_CHARS and same_heading and len(merged_text) <= HARD_MAX_CHARS:
            pending = MarkdownSection(pending.heading_path, merged_text, pending.start_char, section.end_char)
        else:
            chunks.append(pending)
            pending = section

    if pending:
        chunks.append(pending)
    return chunks


def _chunk_id(source_file: str, heading_path: list[str], ordinal: int) -> str:
    basis = f"{source_file}|{'/'.join(heading_path)}|{ordinal}"
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]
    return f"{Path(source_file).stem}-{ordinal:04d}-{digest}"


def chunk_document(doc: MarkdownDocument, manifest_entry: Dict[str, object] | None = None) -> list[dict[str, object]]:
    manifest_entry = manifest_entry or {}
    sections = _chunk_sections(split_markdown_sections(doc.content))
    chunks: list[dict[str, object]] = []
    for ordinal, section in enumerate(sections, start=1):
        content = section.text.strip()
        chunks.append(
            {
                "chunk_id": _chunk_id(doc.source_file, section.heading_path, ordinal),
                "course_id": str(manifest_entry.get("course_id", "")),
                "content": content,
                "source_file": doc.source_file,
                "course_name": str(manifest_entry.get("course_name", DEFAULT_COURSE_NAME)),
                "module_name": str(manifest_entry.get("module_name", "Sample Course")),
                "chapter": section.heading_path[0] if section.heading_path else "",
                "section": section.heading_path[-1] if section.heading_path else "",
                "heading_path": section.heading_path,
                "start_char": section.start_char,
                "end_char": section.end_char,
                "char_count": len(content),
                "title": str(manifest_entry.get("title", section.heading_path[0] if section.heading_path else "")),
                "tags": list(manifest_entry.get("tags", [])),
            }
        )
    return chunks


def chunk_documents(
    documents: Iterable[MarkdownDocument],
    manifest_lookup: Dict[str, Dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    manifest_lookup = manifest_lookup or {}
    all_chunks: list[dict[str, object]] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc, manifest_lookup.get(doc.source_file, {})))
    return all_chunks


def save_chunks_jsonl(chunks: Iterable[dict[str, object]], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return output_path


def load_chunks_jsonl(path: str | Path) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks
