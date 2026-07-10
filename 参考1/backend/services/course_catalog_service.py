from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CATALOG_PATH = DATA_DIR / "course_catalog" / "catalog.json"
COURSES_DIR = DATA_DIR / "courses"


@lru_cache(maxsize=1)
def load_course_catalog() -> dict[str, Any]:
    """Load the AI major course-group catalog from backend/data."""
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        catalog = json.load(file)
    catalog.setdefault("courses", [])
    return catalog


def get_all_courses() -> list[dict[str, Any]]:
    return list(load_course_catalog().get("courses", []))


def get_course_by_id(course_id: str) -> dict[str, Any] | None:
    return next((course for course in get_all_courses() if course.get("id") == course_id), None)


def get_courses_by_level(level: str) -> list[dict[str, Any]]:
    normalized = level.casefold()
    return [course for course in get_all_courses() if str(course.get("level", "")).casefold() == normalized]


def get_course_source_file(course_id: str) -> Path | None:
    course = get_course_by_id(course_id)
    if not course or not course.get("file"):
        return None
    return COURSES_DIR / str(course["file"])


def get_course_prerequisites(course_id: str) -> list[str]:
    course = get_course_by_id(course_id)
    return list(course.get("prerequisites", [])) if course else []


def get_course_next_courses(course_id: str) -> list[str]:
    course = get_course_by_id(course_id)
    return list(course.get("nextCourses", [])) if course else []


RAG_CHUNK_METADATA_FIELDS = (
    "chunk_id",
    "course_id",
    "course_name",
    "source_file",
    "chapter",
    "section",
    "heading_path",
    "content",
    "char_count",
)
