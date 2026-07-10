# EduAgentX Code Review Summary

本审查包用于交给 ChatGPT 或评审助手核对项目是否符合比赛方案。包内不包含 `.env`、真实密钥、`node_modules`、`.venv`、`__pycache__`、`.next`、`dist` 或 `build`。

## 项目技术栈

- 后端：Python、FastAPI、Pydantic schemas、SQLAlchemy、SQLite、httpx、uvicorn。
- 前端：Next.js 14、React 18、TypeScript、Tailwind CSS、Mermaid。
- RAG：Markdown 文档加载、chunk 切分、课程 manifest、RAPTOR 风格树结构、MockEmbedder、可选 FAISS、fallback 向量/关键词检索。
- 测试：pytest，测试目录为 `backend/tests`。
- LLM 接入：统一通过 `backend/services/spark_wrapper.py` 调用讯飞星火 HTTP Chat Completions API；默认可运行 mock 模式。

## 当前已实现功能

- FastAPI 应用初始化、CORS、健康检查和主要业务 API。
- 多智能体聊天主流程：画像抽取、意图识别、RAG 检索、内容生成、Judge 审核、必要时重写。
- 普通聊天接口 `/api/chat` 和 SSE 流式接口 `/api/chat/stream`。
- 学生画像读取与更新。
- 学习资源生成，并将资源写入 SQLite。
- 学习路径生成，并记录学习事件。
- 测评提交与简化 BKT/mastery 更新。
- 知识库 ingest、sample index build、search、tree、manifest API。
- 讯飞 Spark wrapper、mock 输出、真实 API 配置检测、fallback 机制和模型状态接口。
- Next.js 前端页面：dashboard、chat、resources、learning path、profile、skill tree 等。
- pytest 覆盖健康检查、聊天、流式聊天、模型状态、RAG、知识库 API、orchestrator 和 Spark wrapper 等基础行为。

## 当前未实现功能

- 没有完整登录、权限、班级/教师/学生多角色体系。
- SQLite 只作为 MVP 持久化脚手架，尚未实现完整生产级用户数据、会话历史和审计链路。
- 当前 embedding 为 `MockEmbedder`；真实向量模型、在线 embedding 服务和高质量语义召回尚未接入。
- FAISS 为可选依赖；当前打包状态下索引元数据为 fallback 后端，不是原生 `index.faiss`。
- 多智能体主要是规则和 wrapper 编排，尚未接入复杂工具调用、长期记忆、队列化任务或可视化工作流引擎。
- 前端以 MVP 演示交互为主，尚未实现完整错误恢复、权限控制、生产部署配置和 CI/CD。
- 文档中部分中文内容存在编码显示异常，建议审查时重点关注源文件编码一致性。

## 如何启动后端

```bash
pip install -r requirements.txt
uvicorn backend.app:app --reload
```

兼容入口：

```bash
uvicorn backend.main:app --reload
```

默认后端地址：`http://127.0.0.1:8000`。

## 如何启动前端

```bash
cd frontend
npm install
npm run dev
```

默认前端地址：`http://localhost:3000`。前端 API 地址由 `NEXT_PUBLIC_API_BASE_URL` 控制，默认值在 `frontend/lib/api.ts` 中为 `http://127.0.0.1:8000`。

## 如何运行测试

```bash
pip install -r requirements.txt
pytest backend/tests
```

前端可运行：

```bash
cd frontend
npm run lint
npm run build
```

## 如何切换 mock 模式和真实讯飞 API 模式

本审查包只包含 `.env.example`，不包含 `.env`。本地运行时复制示例文件：

```bash
cp .env.example .env
```

mock 模式：

```text
LLM_PROVIDER=spark
ENABLE_MOCK_LLM=true
ENABLE_SPARK_FALLBACK=true
```

真实讯飞 Spark API 模式：

```text
LLM_PROVIDER=spark
ENABLE_MOCK_LLM=false
ENABLE_SPARK_FALLBACK=true
SPARK_API_BASE=https://spark-api-open.xf-yun.com/x2/chat/completions
SPARK_MODEL=x1
SPARK_APP_ID=
SPARK_API_KEY=
SPARK_API_SECRET=
SPARK_API_PASSWORD=
```

请只在本地 `.env` 中填入自己的真实凭证，不要提交或打包。

如使用 X1.5：

```text
SPARK_API_BASE=https://spark-api-open.xf-yun.com/v2/chat/completions
SPARK_MODEL=x1.5
```

检查模型状态：

```bash
curl http://127.0.0.1:8000/api/model/status
```

## RAG 当前使用 FAISS 还是 fallback

当前 `data/processed/faiss_index/metadata.json` 中记录：

```json
{
  "backend": "fallback"
}
```

因此当前审查包中的 RAG 索引使用 fallback 检索，不是原生 FAISS `index.faiss`。代码层面 `backend/rag/vector_store.py` 支持在安装 `faiss` 和 `numpy` 且构建索引时生成 FAISS 后端；Windows 环境下 requirements 将 `faiss-cpu` 标为非 Windows 平台依赖。

## 多智能体调用流程说明

`/api/chat` 和 `/api/chat/stream` 通过 `ChatService` 调用 `EduAgentOrchestrator`。主流程如下：

1. `ProfileAgent` 从用户消息中抽取或更新学生画像。
2. `PlannerAgent` 根据显式 mode 或关键词规则识别意图。
3. `RetrieverAgent` 调用 `RagRetriever` 检索课程知识片段。
4. 根据意图分流：
   - `TutorAgent`：问答解释。
   - `ResourceAgent`：生成学习资源。
   - `PlannerAgent`：生成学习路径。
   - `ProfileAgent`：返回画像更新结果。
5. `JudgeAgent` 对生成内容进行审核评分。
6. 如果审核未通过，orchestrator 带 rewrite instruction 再生成一次。
7. 返回 `reply`、`intent`、`result`、`judge`、`rewritten`、`agent_trace`、`model_provider`、`model_name` 和 `fallback_used`。

SSE 事件包括：`profile_update`、`intent_detected`、`retrieve_start`、`retrieve_done`、`generate_start`、`token`、`judge_start`、`judge_done`、`final`。

## 主要 API 列表

- `GET /health`
- `GET /api/health`
- `POST /api/chat`
- `POST /api/chat/stream`
- `GET /api/profile/{student_id}`
- `POST /api/profile/update`
- `POST /api/resources/generate`
- `POST /api/path/generate`
- `POST /api/assessment/submit`
- `GET /api/model/status`
- `POST /api/knowledge/build_index`
- `POST /api/knowledge/ingest`
- `POST /api/knowledge/search`
- `GET /api/knowledge/tree`
- `GET /api/knowledge/manifest`

## 打包范围说明

- 已保留：`.env.example`、`README.md`、`docs/`、`backend/`、`frontend/`、`scripts/`、`requirements.txt`、`pytest.ini`、`data/processed/` 小型处理产物、`data/sample_course/` 的 3 个 Markdown 示例。
- 已排除：`.env`、真实密钥、`node_modules/`、`.venv/`、`__pycache__/`、`.next/`、`dist/`、`build/`、`.git/`、日志文件、pytest 临时缓存、`data/raw/knowledge.zip`、大批量 sample course raw Markdown。
