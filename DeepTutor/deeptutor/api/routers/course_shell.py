"""Course shell endpoints for curated course entry, onboarding, and path builds."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, ValidationError

from deeptutor.learning.models import LearningModule, LearningProgress, LearningStage
from deeptutor.learning.service import LearningService
from deeptutor.learning.storage import LearningStore
from deeptutor.runtime.home import PACKAGE_ROOT
from deeptutor.services.path_service import get_path_service
from deeptutor.utils.json_parser import parse_json_response

router = APIRouter()

COURSES_FILE = PACKAGE_ROOT / "deeptutor" / "course_data" / "courses.json"
PROFILE_FIELDS = ("prerequisite", "goal", "pace", "weak_points")
ONBOARDING_QUESTIONS = [
    "我们先确认先修基础：你现在对函数、三角函数、方程变形和高中导数大概是什么状态？",
    "这门高等数学你最想达成什么目标？比如应付考试、补齐基础、刷题提分，还是准备竞赛/考研。",
    "接下来你的学习节奏怎样？每周大概能学几次、每次多久，是否有考试截止时间？",
    "最后说说薄弱点：极限、导数、积分、应用题、级数里，哪些最容易卡住你？",
]


class CourseMessage(BaseModel):
    role: Literal["assistant", "user"]
    content: str


class OnboardingRequest(BaseModel):
    session_id: str
    history: list[CourseMessage] = Field(default_factory=list)
    answer: str | None = None


class CourseProfile(BaseModel):
    prerequisite: str = ""
    goal: str = ""
    pace: str = ""
    weak_points: list[str] = Field(default_factory=list)


class BuildPathRequest(BaseModel):
    session_id: str
    profile: CourseProfile


def _validate_book_id(book_id: str) -> None:
    if not book_id or ".." in book_id or "/" in book_id or "\\" in book_id or ":" in book_id:
        raise HTTPException(status_code=400, detail="Invalid session_id")


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_\-]", "_", session_id.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid session_id")
    _validate_book_id(cleaned)
    return cleaned[:120]


def _load_courses_raw() -> list[dict[str, Any]]:
    if not COURSES_FILE.exists():
        raise HTTPException(status_code=500, detail="courses.json not found")
    data = json.loads(COURSES_FILE.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise HTTPException(status_code=500, detail="courses.json must contain a list")
    return data


def _course_public(course: dict[str, Any]) -> dict[str, Any]:
    hidden = {"template_modules", "optional_module_ids"}
    return {key: value for key, value in course.items() if key not in hidden}


def _get_course(course_id: str) -> dict[str, Any]:
    for course in _load_courses_raw():
        if course.get("id") == course_id:
            return course
    raise HTTPException(status_code=404, detail="Course not found")


def _ordered_user_answers(history: list[CourseMessage], answer: str | None) -> list[str]:
    values = [item.content.strip() for item in history if item.role == "user" and item.content.strip()]
    if answer and answer.strip():
        values.append(answer.strip())
    return values[: len(PROFILE_FIELDS)]


def _fallback_profile(answers: list[str]) -> CourseProfile:
    weak_text = answers[3] if len(answers) > 3 else ""
    weak_points = [
        item.strip(" ，,;；。")
        for item in re.split(r"[、,，;；\s]+", weak_text)
        if item.strip(" ，,;；。")
    ]
    return CourseProfile(
        prerequisite=answers[0] if len(answers) > 0 else "",
        goal=answers[1] if len(answers) > 1 else "",
        pace=answers[2] if len(answers) > 2 else "",
        weak_points=weak_points[:8],
    )


async def _extract_profile_with_llm(answers: list[str]) -> CourseProfile:
    fallback = _fallback_profile(answers)
    if not _llm_configured():
        return fallback
    try:
        from deeptutor.services.llm import complete

        prompt = (
            "请把以下高等数学 onboarding 回答提炼为严格 JSON，不要输出 Markdown。\n"
            "JSON schema: {\"prerequisite\": string, \"goal\": string, "
            "\"pace\": string, \"weak_points\": string[]}。\n"
            f"先修基础：{answers[0]}\n"
            f"学习目标：{answers[1]}\n"
            f"时间节奏：{answers[2]}\n"
            f"薄弱点：{answers[3]}\n"
        )
        raw = await complete(
            prompt=prompt,
            system_prompt="你是课程规划助手，只输出可解析的 JSON。",
        )
        data = parse_json_response(raw, fallback=None)
        if isinstance(data, dict):
            return CourseProfile.model_validate(data)
    except Exception:
        return fallback
    return fallback


def _llm_configured() -> bool:
    try:
        from deeptutor.services.config import load_model_settings

        settings = load_model_settings()
        return bool((settings.get("llm") or {}).get("api_key"))
    except Exception:
        return False


def _modules_from_progress(progress: LearningProgress) -> list[LearningModule]:
    return [LearningModule.model_validate(module.model_dump()) for module in progress.modules]


def _modules_from_course(course: dict[str, Any]) -> list[LearningModule]:
    modules_raw = course.get("template_modules") or []
    if not isinstance(modules_raw, list) or not modules_raw:
        raise HTTPException(status_code=500, detail="Course template is empty")
    try:
        return [LearningModule.model_validate(module) for module in modules_raw]
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid course template: {exc}") from exc


def _load_template_modules(course: dict[str, Any]) -> tuple[list[LearningModule], str]:
    template_book_id = str(course.get("template_book_id") or "").strip()
    if template_book_id:
        template = LearningStore().load(template_book_id)
        if template and template.modules:
            return _modules_from_progress(template), "template"
    return _modules_from_course(course), "starter"


def _matches_weakness(module: LearningModule, weak_points: list[str]) -> bool:
    if not weak_points:
        return False
    haystack = " ".join(
        [module.name, module.id, *[kp.name for kp in module.knowledge_points]]
    ).lower()
    aliases = {
        "极限": ["极限", "limit"],
        "导数": ["导数", "微分", "derivative"],
        "积分": ["积分", "integral"],
        "级数": ["级数", "series"],
        "应用": ["应用", "建模", "体积", "面积"],
    }
    needles: list[str] = []
    for point in weak_points:
        point = point.strip().lower()
        needles.extend(aliases.get(point, [point]))
    return any(needle and needle.lower() in haystack for needle in needles)


def _is_fast_pace(pace: str) -> bool:
    pace = pace.lower()
    return any(token in pace for token in ("冲刺", "考试", "两周", "2周", "快", "short", "exam"))


def _has_strong_prerequisite(prerequisite: str) -> bool:
    prerequisite = prerequisite.lower()
    return any(token in prerequisite for token in ("较好", "很好", "扎实", "熟", "强", "advanced", "good"))


def _personalize_modules(
    modules: list[LearningModule], course: dict[str, Any], profile: CourseProfile
) -> tuple[list[LearningModule], dict[str, Any]]:
    optional_ids = set(course.get("optional_module_ids") or [])
    skipped: list[str] = []
    focused: list[str] = []
    personalized = [LearningModule.model_validate(deepcopy(module.model_dump())) for module in modules]

    if _is_fast_pace(profile.pace) and optional_ids:
        kept = [module for module in personalized if module.id not in optional_ids]
        skipped = [module.name for module in personalized if module.id in optional_ids]
        personalized = kept or personalized

    for module in personalized:
        if _matches_weakness(module, profile.weak_points):
            module.pass_threshold = max(module.pass_threshold, 0.85)
            focused.append(module.name)

    if focused:
        personalized.sort(key=lambda module: (0 if module.name in focused else 1, module.order))

    for index, module in enumerate(personalized):
        module.order = index

    start_module_id = personalized[0].id if personalized else ""
    if _has_strong_prerequisite(profile.prerequisite) and len(personalized) > 1:
        skipped.append(personalized[0].name)
        start_module_id = personalized[1].id

    plan = {
        "skipped": list(dict.fromkeys(skipped)),
        "focused": list(dict.fromkeys(focused)),
        "start_module_id": start_module_id,
        "template_source": "template",
    }
    return personalized, plan


def _plan_path(book_id: str) -> Path:
    root = get_path_service().get_workspace_dir() / "course_shell"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{book_id}.json"


def _save_plan(book_id: str, payload: dict[str, Any]) -> None:
    _plan_path(book_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_plan(book_id: str) -> dict[str, Any] | None:
    path = _plan_path(book_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/courses")
async def list_courses():
    return {"courses": [_course_public(course) for course in _load_courses_raw()]}


@router.post("/courses/{course_id}/onboarding")
async def onboarding(course_id: str, body: OnboardingRequest):
    _get_course(course_id)
    _safe_session_id(body.session_id)
    answers = _ordered_user_answers(body.history, body.answer)
    if len(answers) >= len(PROFILE_FIELDS):
        profile = await _extract_profile_with_llm(answers)
        return {
            "message": "我已经整理好你的学习画像，可以生成专属精通之路了。",
            "done": True,
            "profile": profile.model_dump(),
        }
    return {
        "message": ONBOARDING_QUESTIONS[len(answers)],
        "done": False,
        "profile": None,
    }


@router.post("/courses/{course_id}/build-path")
async def build_path(course_id: str, body: BuildPathRequest):
    course = _get_course(course_id)
    book_id = _safe_session_id(body.session_id)
    modules, template_source = _load_template_modules(course)
    modules, plan = _personalize_modules(modules, course, body.profile)
    plan["template_source"] = template_source

    if not modules:
        raise HTTPException(status_code=500, detail="No runnable modules after personalization")

    store = LearningStore()
    service = LearningService(store)
    progress = LearningProgress(book_id=book_id)
    service.init_modules(progress, modules)
    progress.current_module_id = str(plan.get("start_module_id") or modules[0].id)
    progress.current_stage = LearningStage.DIAGNOSTIC
    progress.current_kp_index = 0
    service.save(progress)

    payload = {
        "course": _course_public(course),
        "profile": body.profile.model_dump(),
        "plan": plan,
        "book_id": book_id,
    }
    _save_plan(book_id, payload)
    return {"book_id": book_id, **payload}


@router.get("/courses/{course_id}/path-plan/{book_id}")
async def get_path_plan(course_id: str, book_id: str):
    _get_course(course_id)
    _validate_book_id(book_id)
    payload = _load_plan(book_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Path plan not found")
    return payload
