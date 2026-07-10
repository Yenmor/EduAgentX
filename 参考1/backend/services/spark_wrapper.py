from __future__ import annotations

import json
import re
from typing import Any, Iterable

import httpx

from backend.config.settings import Settings, get_settings, mask_secret


class SparkAPIError(RuntimeError):
    """Raised when Spark API calls fail and fallback is disabled."""


class SparkWrapper:
    """Unified gateway for all LLM calls in EduAgentX."""

    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None, mock: bool | None = None) -> None:
        self.settings = settings or get_settings()
        if mock is not None:
            self.settings = Settings(**{**self.settings.__dict__, "enable_mock_llm": mock})
        self.client = client

    def generate(
        self,
        messages: str | list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        role: str | None = None,
    ) -> dict:
        normalized = self._normalize_messages(messages, role=role)
        if self.settings.enable_mock_llm:
            return self.mock_generate(normalized)
        provider = self._normalized_provider()
        if provider == "spark":
            return self._generate_with_spark(normalized, temperature=temperature, max_tokens=max_tokens, stream=stream)
        if provider in {"openai", "openai-compatible", "openai_compatible"}:
            return self._generate_with_openai(normalized, temperature=temperature, max_tokens=max_tokens, stream=stream)
        error = f"Unsupported llm_provider: {self.settings.llm_provider}"
        if self.settings.enable_llm_fallback:
            fallback = self.mock_generate(normalized)
            fallback["fallback_used"] = True
            fallback["error"] = error
            return fallback
        raise SparkAPIError(error)

    def _generate_with_spark(
        self,
        normalized: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict:
        if not self._spark_config_loaded():
            error = "Spark config incomplete"
            if self.settings.enable_llm_fallback:
                fallback = self.mock_generate(normalized)
                fallback["fallback_used"] = True
                fallback["error"] = error
                return fallback
            raise SparkAPIError(error)
        if stream:
            return {
                "content": "".join(part.get("content", "") for part in self.stream_generate(normalized, temperature, max_tokens)),
                "model_provider": "spark",
                "model_name": self.settings.spark_model,
                "fallback_used": False,
                "raw": {},
            }
        try:
            return self._call_spark(normalized, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            error = self._safe_error(exc)
            if self.settings.enable_llm_fallback:
                fallback = self.mock_generate(normalized)
                fallback["fallback_used"] = True
                fallback["error"] = error
                return fallback
            raise SparkAPIError(error) from exc

    def _generate_with_openai(
        self,
        normalized: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict:
        if not self._openai_config_loaded():
            error = "OpenAI-compatible config incomplete"
            if self.settings.enable_llm_fallback:
                fallback = self.mock_generate(normalized)
                fallback["fallback_used"] = True
                fallback["error"] = error
                return fallback
            raise SparkAPIError(error)
        if stream:
            return {
                "content": "".join(part.get("content", "") for part in self.stream_generate(normalized, temperature, max_tokens)),
                "model_provider": "openai-compatible",
                "model_name": self.settings.openai_model,
                "fallback_used": False,
                "raw": {},
            }
        try:
            return self._call_openai_compatible(normalized, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            error = self._safe_error(exc)
            if self.settings.enable_llm_fallback:
                fallback = self.mock_generate(normalized)
                fallback["fallback_used"] = True
                fallback["error"] = error
                return fallback
            raise SparkAPIError(error) from exc

    def generate_text(
        self,
        messages: str | list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        role: str | None = None,
    ) -> str:
        return str(self.generate(messages, temperature=temperature, max_tokens=max_tokens, role=role).get("content", ""))

    def generate_json(self, messages: str | list[dict[str, str]], temperature: float = 0.2) -> dict:
        response = self.generate(messages, temperature=temperature)
        text = str(response.get("content", ""))
        extracted = self._extract_json_text(text)
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                parsed.setdefault("model_provider", response.get("model_provider"))
                parsed.setdefault("model_name", response.get("model_name"))
                parsed.setdefault("fallback_used", response.get("fallback_used", False))
                return parsed
            return {
                "value": parsed,
                "model_provider": response.get("model_provider"),
                "model_name": response.get("model_name"),
                "fallback_used": response.get("fallback_used", False),
            }
        except json.JSONDecodeError:
            return {
                "parse_error": True,
                "raw_text": text,
                "model_provider": response.get("model_provider"),
                "model_name": response.get("model_name"),
                "fallback_used": response.get("fallback_used", False),
            }

    def stream_generate(
        self,
        messages: str | list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        normalized = self._normalize_messages(messages)
        provider = self._normalized_provider()
        if self.settings.enable_mock_llm:
            yield self.mock_generate(normalized)
            return
        if provider == "spark":
            if not self._spark_config_loaded():
                yield self.mock_generate(normalized)
                return
            payload = self._build_spark_payload(normalized, temperature=temperature, max_tokens=max_tokens, stream=True)
            try:
                with httpx.stream(
                    "POST",
                    self.settings.spark_api_base,
                    headers=self._build_spark_headers(),
                    json=payload,
                    timeout=self.settings.spark_timeout_seconds,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        content = self._parse_stream_line(line)
                        if content:
                            yield {
                                "content": content,
                                "model_provider": "spark",
                                "model_name": self.settings.spark_model,
                                "fallback_used": False,
                                "raw": {},
                            }
                return
            except Exception as exc:
                if self.settings.enable_llm_fallback:
                    fallback = self.mock_generate(normalized)
                    fallback["fallback_used"] = True
                    fallback["error"] = self._safe_error(exc)
                    yield fallback
                    return
                raise SparkAPIError(self._safe_error(exc)) from exc
        if provider in {"openai", "openai-compatible", "openai_compatible"}:
            if not self._openai_config_loaded():
                yield self.mock_generate(normalized)
                return
            payload = self._build_openai_payload(normalized, temperature=temperature, max_tokens=max_tokens, stream=True)
            try:
                with httpx.stream(
                    "POST",
                    self.settings.openai_api_base,
                    headers=self._build_openai_headers(),
                    json=payload,
                    timeout=self.settings.openai_timeout_seconds,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        content = self._parse_stream_line(line)
                        if content:
                            yield {
                                "content": content,
                                "model_provider": "openai-compatible",
                                "model_name": self.settings.openai_model,
                                "fallback_used": False,
                                "raw": {},
                            }
                return
            except Exception as exc:
                if self.settings.enable_llm_fallback:
                    fallback = self.mock_generate(normalized)
                    fallback["fallback_used"] = True
                    fallback["error"] = self._safe_error(exc)
                    yield fallback
                    return
                raise SparkAPIError(self._safe_error(exc)) from exc
        yield self.mock_generate(normalized)

    def mock_generate(self, messages: str | list[dict[str, str]]) -> dict:
        normalized = self._normalize_messages(messages)
        last_user = self._last_user_content(normalized)
        lowered = "\n".join(message.get("content", "") for message in normalized).lower()
        content = self._mock_content(last_user, lowered)
        return {
            "content": content,
            "model_provider": "mock",
            "model_name": "mock",
            "fallback_used": False,
            "raw": {},
        }

    def mock_generate_text(self, messages: str | list[dict[str, str]]) -> str:
        return str(self.mock_generate(messages).get("content", ""))

    def health_check(self) -> dict:
        if self.settings.enable_mock_llm:
            return {"ok": True, "mode": "mock", "message": "Mock LLM is enabled"}
        provider = self._normalized_provider()
        if provider == "spark":
            if not self._spark_config_loaded():
                return {"ok": False, "mode": "spark", "message": "Spark config incomplete"}
            try:
                response = self._call_spark(
                    [
                        {"role": "system", "content": "你是一个连通性检测助手。"},
                        {"role": "user", "content": "请回复 ok"},
                    ],
                    max_tokens=8,
                    temperature=0.1,
                )
                return {
                    "ok": True,
                    "mode": "spark",
                    "message": "Spark API reachable",
                    "model_provider": response.get("model_provider"),
                    "fallback_used": response.get("fallback_used", False),
                }
            except SparkAPIError as exc:
                return {"ok": False, "mode": "spark", "message": str(exc)}
            except Exception as exc:
                return {"ok": False, "mode": "spark", "message": self._safe_error(exc)}
        if provider in {"openai", "openai-compatible", "openai_compatible"}:
            if not self._openai_config_loaded():
                return {"ok": False, "mode": "openai-compatible", "message": "OpenAI-compatible config incomplete"}
            try:
                response = self._call_openai_compatible(
                    [
                        {"role": "system", "content": "You are a connectivity check assistant."},
                        {"role": "user", "content": "Reply with ok"},
                    ],
                    max_tokens=8,
                    temperature=0.1,
                )
                return {
                    "ok": True,
                    "mode": "openai-compatible",
                    "message": "OpenAI-compatible API reachable",
                    "model_provider": response.get("model_provider"),
                    "fallback_used": response.get("fallback_used", False),
                }
            except SparkAPIError as exc:
                return {"ok": False, "mode": "openai-compatible", "message": str(exc)}
            except Exception as exc:
                return {"ok": False, "mode": "openai-compatible", "message": self._safe_error(exc)}
        return {"ok": False, "mode": provider, "message": f"Unsupported llm_provider: {self.settings.llm_provider}"}

    def _call_spark(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        payload = self._build_spark_payload(messages, temperature=temperature, max_tokens=max_tokens)
        client = self.client or httpx.Client(timeout=self.settings.spark_timeout_seconds)
        close_client = self.client is None
        try:
            response = client.post(self.settings.spark_api_base, headers=self._build_spark_headers(), json=payload)
            response.raise_for_status()
            raw = response.json()
            return {
                "content": self._extract_content(raw),
                "model_provider": "spark",
                "model_name": self.settings.spark_model,
                "fallback_used": False,
                "raw": raw,
            }
        finally:
            if close_client:
                client.close()

    def _call_openai_compatible(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        payload = self._build_openai_payload(messages, temperature=temperature, max_tokens=max_tokens)
        client = self.client or httpx.Client(timeout=self.settings.openai_timeout_seconds)
        close_client = self.client is None
        try:
            response = client.post(self.settings.openai_api_base, headers=self._build_openai_headers(), json=payload)
            response.raise_for_status()
            raw = response.json()
            return {
                "content": self._extract_content(raw),
                "model_provider": "openai-compatible",
                "model_name": self.settings.openai_model,
                "fallback_used": False,
                "raw": raw,
            }
        finally:
            if close_client:
                client.close()

    def _build_spark_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.spark_api_password:
            headers["Authorization"] = f"Bearer {self.settings.spark_api_password}"
        elif self.settings.spark_api_key:
            headers["X-API-Key"] = self.settings.spark_api_key
            if self.settings.spark_app_id:
                headers["X-Appid"] = self.settings.spark_app_id
            if self.settings.spark_api_secret:
                headers["X-API-Secret"] = self.settings.spark_api_secret
        return headers

    def _build_openai_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.settings.openai_api_key}",
        }

    def _spark_config_loaded(self) -> bool:
        has_http_auth = bool(self.settings.spark_api_password)
        has_legacy_auth = bool(self.settings.spark_app_id and self.settings.spark_api_key and self.settings.spark_api_secret)
        return bool(self.settings.spark_api_base and self.settings.spark_model and (has_http_auth or has_legacy_auth))

    def _openai_config_loaded(self) -> bool:
        return bool(self.settings.openai_api_base and self.settings.openai_model and self.settings.openai_api_key)

    def _build_spark_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict:
        return {
            "model": self.settings.spark_model,
            "messages": messages,
            "temperature": self.settings.spark_temperature if temperature is None else temperature,
            "max_tokens": self.settings.spark_max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    def _build_openai_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ) -> dict:
        return {
            "model": self.settings.openai_model,
            "messages": messages,
            "temperature": self.settings.openai_temperature if temperature is None else temperature,
            "max_tokens": self.settings.openai_max_tokens if max_tokens is None else max_tokens,
            "stream": stream,
        }

    def _normalize_messages(self, messages: str | list[dict[str, str]], role: str | None = None) -> list[dict[str, str]]:
        if isinstance(messages, str):
            prefix = f"[{role}] " if role else ""
            return [{"role": "user", "content": prefix + messages}]
        normalized = []
        for message in messages:
            normalized.append({"role": str(message.get("role", "user")), "content": str(message.get("content", ""))})
        return normalized

    def _last_user_content(self, messages: Iterable[dict[str, str]]) -> str:
        last = ""
        for message in messages:
            if message.get("role") == "user":
                last = message.get("content", "")
        return last

    def _mock_content(self, last_user: str, lowered: str) -> str:
        if "内容质量评审智能体" in lowered or ("rewrite_instruction" in lowered and "score" in lowered):
            return json.dumps({"pass": True, "score": 88, "issues": [], "rewrite_instruction": ""}, ensure_ascii=False)
        if any(key in lowered for key in ["画像", "profile", "学生"]):
            return json.dumps(
                {
                    "专业背景": "计算机相关或对人工智能应用感兴趣",
                    "当前课程": "大模型应用开发与智能体实践",
                    "知识基础": "具备基础编程能力，正在补齐 RAG 与 Agent 概念",
                    "学习目标": "能够构建基于课程知识库的个性化学习智能体",
                    "薄弱知识点": ["RAG", "Function Calling", "多智能体协同"],
                    "认知风格": "偏实践和结构化解释",
                    "资源偏好": ["示例代码", "思维导图", "小测验"],
                    "学习节奏": "每次 30-45 分钟，先概念后实践",
                },
                ensure_ascii=False,
            )
        if any(key in lowered for key in ["学习路径", "learning path", "规划", "计划"]):
            return json.dumps(
                {
                    "steps": [
                        {"title": "Prompt Engineering", "activity": "掌握结构化提示词", "estimated_minutes": 30},
                        {"title": "RAG", "activity": "理解检索增强生成链路", "estimated_minutes": 45},
                        {"title": "Agent Workflow", "activity": "实现工具调用与审查闭环", "estimated_minutes": 60},
                    ]
                },
                ensure_ascii=False,
            )
        if "rag" in lowered or "检索增强生成" in lowered:
            return "RAG 是检索增强生成：先从课程知识库检索相关片段，再把片段作为上下文交给大模型生成答案，从而提升可追溯性并降低幻觉风险。"
        if "transformer" in lowered or "self-attention" in lowered or "attention" in lowered:
            return "Self-Attention 会让每个 token 根据 Query、Key、Value 与其他 token 建立关联，得到包含上下文信息的表示，是 Transformer 的核心机制。"
        if "agent" in lowered or "智能体" in lowered:
            return "LLM Agent 会围绕目标进行规划、调用工具、读取反馈并继续执行。多智能体系统会把画像、检索、规划、资源生成和审查拆给不同 Agent 协同完成。"
        return f"这是统一 LLM 网关的 mock 模式回答：我会围绕“{last_user[:80]}”给出可演示的学习建议，并在真实密钥配置后切换到你选择的 LLM API。"

    def _extract_content(self, raw: dict[str, Any]) -> str:
        try:
            return str(raw["choices"][0]["message"]["content"])
        except Exception:
            pass
        if "content" in raw:
            return str(raw["content"])
        if "message" in raw:
            return str(raw["message"])
        return json.dumps(raw, ensure_ascii=False)

    def _extract_json_text(self, text: str) -> str:
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence:
            return fence.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return text.strip()

    def _parse_stream_line(self, line: str) -> str:
        line = line.strip()
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            return ""
        try:
            raw = json.loads(line)
            return self._extract_content(raw)
        except json.JSONDecodeError:
            return ""

    def _safe_error(self, exc: Exception) -> str:
        text = str(exc)
        for secret in [
            self.settings.spark_api_password,
            self.settings.spark_api_key,
            self.settings.spark_api_secret,
            self.settings.spark_app_id,
            self.settings.openai_api_key,
        ]:
            if secret:
                text = text.replace(secret, mask_secret(secret))
        return text[:240]

    def _normalized_provider(self) -> str:
        return self.settings.llm_provider.strip().lower()
