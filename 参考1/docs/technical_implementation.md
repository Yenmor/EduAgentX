# Technical Implementation

## 科大讯飞星火 API 接入说明

EduAgentX 在模型底座层接入科大讯飞星火大模型 HTTP Chat Completions API。后端统一通过 `backend/services/spark_wrapper.py` 发起模型调用，业务代码和 Agent 不直接读取密钥，也不直接调用外部 HTTP 接口。

ProfileAgent、PlannerAgent、ResourceAgent、TutorAgent、JudgeAgent 均通过 `SparkWrapper` 调用星火能力。`SparkWrapper` 接收 OpenAI 风格 `messages`，内部转换为 HTTP Chat Completions 请求，并统一返回 `content`、`model_provider`、`model_name`、`fallback_used` 和 `raw`。

系统提供 mock fallback：本地无网络、未配置密钥或真实接口异常时，只要 `ENABLE_SPARK_FALLBACK=true`，后端会自动回退到 mock 生成，保证比赛演示不中断。`GET /api/model/status` 可查看当前 provider、模型、mock 状态、fallback 状态和配置完整性，但不会返回真实密钥或 Authorization。

API Key、APISecret、APIPassword 等敏感信息仅通过 `.env` 环境变量管理，不进入代码仓库。仓库只提供 `.env.example`，并在 `.gitignore` 中排除 `.env`、`.env.local` 和 `*.env`。

当前 RAG 检索仍使用 mock embedding、fallback vector 或 keyword 检索，不依赖星火 Embedding。后续可在 `backend/services/spark_embedding.py` 中接入科大讯飞 Embedding 服务，并替换现有 embedding 实现。
