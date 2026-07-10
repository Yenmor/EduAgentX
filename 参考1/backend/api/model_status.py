from __future__ import annotations

from fastapi import APIRouter

from backend.config.settings import get_settings, openai_config_loaded, spark_config_loaded
from backend.services.spark_wrapper import SparkWrapper


router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/status")
def model_status() -> dict:
    settings = get_settings()
    health = SparkWrapper(settings=settings).health_check()
    return {
        "llm_provider": settings.llm_provider,
        "model": settings.spark_model if settings.llm_provider == "spark" else settings.openai_model,
        "mock_llm_enabled": settings.enable_mock_llm,
        "fallback_enabled": settings.enable_llm_fallback,
        "spark_config_loaded": spark_config_loaded(),
        "openai_config_loaded": openai_config_loaded(),
        "api_base": settings.spark_api_base if settings.llm_provider == "spark" else settings.openai_api_base,
        "app_id_configured": bool(settings.spark_app_id),
        "api_key_configured": bool(settings.spark_api_key),
        "api_secret_configured": bool(settings.spark_api_secret),
        "api_password_configured": bool(settings.spark_api_password),
        "openai_api_key_configured": bool(settings.openai_api_key),
        "providers": {
            "spark": {
                "model": settings.spark_model,
                "api_base": settings.spark_api_base,
                "config_loaded": spark_config_loaded(),
            },
            "openai-compatible": {
                "model": settings.openai_model,
                "api_base": settings.openai_api_base,
                "config_loaded": openai_config_loaded(),
            },
        },
        "health": health,
    }
