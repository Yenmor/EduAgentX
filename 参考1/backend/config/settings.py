from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_dotenv(path: Path = ENV_PATH) -> None:
    """Load a small .env file without overriding existing process env values."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "spark"
    enable_mock_llm: bool = True
    enable_llm_fallback: bool = True
    spark_app_id: str = ""
    spark_api_key: str = ""
    spark_api_secret: str = ""
    spark_api_password: str = ""
    spark_model: str = "x1"
    spark_api_base: str = "https://spark-api-open.xf-yun.com/x2/chat/completions"
    spark_timeout_seconds: int = 60
    spark_temperature: float = 0.7
    spark_max_tokens: int = 2048
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_api_base: str = "https://api.openai.com/v1/chat/completions"
    openai_timeout_seconds: int = 60
    openai_temperature: float = 0.7
    openai_max_tokens: int = 2048


def mask_secret(value: str) -> str:
    """Mask secrets for status pages and logs."""
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}****{value[-4:]}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "spark"),
        enable_mock_llm=_get_bool("ENABLE_MOCK_LLM", True),
        enable_llm_fallback=_get_bool("ENABLE_LLM_FALLBACK", _get_bool("ENABLE_SPARK_FALLBACK", True)),
        spark_app_id=os.getenv("SPARK_APP_ID", ""),
        spark_api_key=os.getenv("SPARK_API_KEY", ""),
        spark_api_secret=os.getenv("SPARK_API_SECRET", ""),
        spark_api_password=os.getenv("SPARK_API_PASSWORD", ""),
        spark_model=os.getenv("SPARK_MODEL", "x1"),
        spark_api_base=os.getenv("SPARK_API_BASE", "https://spark-api-open.xf-yun.com/x2/chat/completions"),
        spark_timeout_seconds=_get_int("SPARK_TIMEOUT_SECONDS", 60),
        spark_temperature=_get_float("SPARK_TEMPERATURE", 0.7),
        spark_max_tokens=_get_int("SPARK_MAX_TOKENS", 2048),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1/chat/completions"),
        openai_timeout_seconds=_get_int("OPENAI_TIMEOUT_SECONDS", 60),
        openai_temperature=_get_float("OPENAI_TEMPERATURE", 0.7),
        openai_max_tokens=_get_int("OPENAI_MAX_TOKENS", 2048),
    )


def spark_config_loaded() -> bool:
    settings = get_settings()
    has_http_auth = bool(settings.spark_api_password)
    has_legacy_auth = bool(settings.spark_app_id and settings.spark_api_key and settings.spark_api_secret)
    return bool(settings.spark_api_base and settings.spark_model and (has_http_auth or has_legacy_auth))


def openai_config_loaded() -> bool:
    settings = get_settings()
    return bool(settings.openai_api_base and settings.openai_model and settings.openai_api_key)
