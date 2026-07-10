# EduAgentX

EduAgentX is a higher-education personalized learning agent MVP. The current positioning is: 面向高校人工智能专业课程群的个性化资源生成与多智能体学习系统. The first stage includes a FastAPI backend, mock multi-agent chat, RAG knowledge hooks, SQLite scaffolding, and a dark Next.js frontend.

## Course Knowledge Base

The project now includes an AI-major course-group knowledge base under `backend/data`:

```text
backend/data/
  course_catalog/catalog.json
  courses/*.md
  metadata/processing_report.json
  README.md
```

The catalog covers Git, Python, NumPy/Pandas, machine learning, neural networks, PyTorch, computer vision, Transformer, LLM, RAG, LangChain, Agent systems, reinforcement learning, RLHF, and GNN. The competition demo focuses on RAG 检索增强生成, RAG 与 LangChain 应用开发, and AI Agent 与多智能体系统.

Backend course catalog helpers live in `backend/services/course_catalog_service.py`. Frontend classroom types live in `frontend/lib/classroom-types.ts`, and API/mock fallback lives in `frontend/lib/classroom-api.ts`.

## Project Structure

```text
backend/        FastAPI app, agents, RAG, services, schemas, database, tests
frontend/       Next.js + TypeScript + Tailwind UI
data/           Raw zip, extracted sample course, processed RAG artifacts
docs/           Architecture notes
scripts/        Knowledge ingestion and index rebuild scripts
```

## Backend

Install dependencies:

```bash
pip install -r requirements.txt
```

If `data/raw/knowledge.zip` is missing on a fresh clone, EduAgentX now auto-packages the built-in sample course into that zip the first time you run ingestion or the knowledge-index tests.

Start the API:

```bash
uvicorn backend.app:app --reload
```

The compatibility entrypoint also works:

```bash
uvicorn backend.main:app --reload
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Or start both frontend and backend together from the project root:

```bash
python scripts/dev.py
```

To install dependencies first and then launch both services:

```bash
python scripts/dev.py --install-deps
```

Main demo entries:

- `/studio`: AI course-group classroom generation entry.
- `/classroom?sessionId=demo`: interactive classroom demo.
- `/demo`: competition demonstration route.

## Mock and Real API Modes

The frontend first tries the classroom API configured by `NEXT_PUBLIC_API_BASE_URL`:

- `POST /api/classroom/sessions`
- `GET /api/classroom/sessions/{sessionId}`
- `POST /api/classroom/sessions/{sessionId}/quiz`
- `POST /api/classroom/sessions/{sessionId}/replan`

These classroom endpoints are implemented with an in-memory session store. If the backend is unavailable, `frontend/lib/classroom-api.ts` automatically falls back to mock classroom state so `/studio`, `/classroom`, and `/demo` remain usable. Sessions are lost after backend restart.

## Classroom API

Create a classroom session:

```bash
curl -X POST http://localhost:8000/api/classroom/sessions \
  -H "Content-Type: application/json" \
  -d "{\"goal\":\"我想学习 RAG 和 AI Agent\",\"selected_course_ids\":[\"rag\",\"langchain\",\"agent\"]}"
```

Fetch classroom state:

```bash
curl http://localhost:8000/api/classroom/sessions/demo
```

Subscribe to SSE:

```bash
curl http://localhost:8000/api/classroom/sessions/demo/stream
```

Submit quiz answers:

```bash
curl -X POST http://localhost:8000/api/classroom/sessions/demo/quiz \
  -H "Content-Type: application/json" \
  -d "{\"answers\":{\"q1\":0,\"q2\":1,\"q3\":1}}"
```

Replan:

```bash
curl -X POST http://localhost:8000/api/classroom/sessions/demo/replan \
  -H "Content-Type: application/json" \
  -d "{\"reason\":\"quiz_result\",\"weak_concepts\":[\"Chunking\",\"Retriever\"]}"
```

Verify frontend fallback:

```bash
cd frontend
npm run dev
```

Open `/classroom?sessionId=demo`. With backend running, the page prefers the real classroom API; with backend stopped, it falls back to mock data.

Additional docs:

- `docs/course_knowledge_base.md`
- `docs/frontend_classroom_experience.md`

Useful endpoints:

- `GET /health`
- `POST /api/chat`
- `POST /api/chat/stream`
- `GET /api/profile/{student_id}`
- `POST /api/profile/update`
- `POST /api/resources/generate`
- `POST /api/path/generate`
- `POST /api/assessment/submit`
- `POST /api/knowledge/ingest`
- `POST /api/knowledge/build_index`
- `POST /api/knowledge/search`
- `GET /api/knowledge/status`
- `GET /api/knowledge/tree`
- `GET /api/knowledge/manifest`

Mock chat example:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"What is Transformer?\",\"mode\":\"qa\"}"
```

SSE streaming chat example:

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d "{\"student_id\":\"demo\",\"message\":\"I do not understand RAG, generate learning resources.\",\"mode\":\"auto\"}"
```

The stream emits named events: `profile_update`, `intent_detected`, `retrieve_start`, `retrieve_done`, `generate_start`, `token`, `judge_start`, `judge_done`, and `final`.

`/api/chat` now runs the multi-agent workflow:

```text
ProfileAgent -> PlannerAgent -> RetrieverAgent -> ResourceAgent/TutorAgent -> JudgeAgent -> optional rewrite
```

The JSON response includes `intent`, `result`, `judge`, `rewritten`, and `agent_trace`. For example:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"student_id\":\"demo\",\"message\":\"我不懂 RAG，帮我生成学习资料\"}"
```

## Knowledge Indexes

EduAgentX has two index builders with different purposes:

- `scripts/ingest_knowledge.py` is the production knowledge-base entrypoint. It reads `data/raw/knowledge.zip` and writes the active RAG/RAPTOR artifacts under `data/processed/`.
- `scripts/build_sample_index.py` is demo-only. It indexes the small built-in sample course under `data/processed/sample_index/` and does not replace the active production index.

If a real `knowledge.zip` has already been ingested, running the sample builder will not silently overwrite `data/processed/chunks.jsonl`, `data/processed/raptor_tree.json`, or `data/processed/faiss_index/`.

Place the course package at:

```text
data/raw/knowledge.zip
```

Ingest it:

```bash
python scripts/ingest_knowledge.py --zip data/raw/knowledge.zip
```

Build the built-in sample course index:

```bash
python scripts/build_sample_index.py
```

This creates a demo index in `data/processed/sample_index/`. It is useful for UI demos and smoke tests, but `/api/knowledge/search`, `/api/knowledge/tree`, and `/api/knowledge/status` continue to use the active production index in `data/processed/`.

Or call the API to build the demo sample index:

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/build_index \
  -H "Content-Type: application/json" \
  -d "{}"
```

Search example:

```bash
curl -X POST http://127.0.0.1:8000/api/knowledge/search \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"Transformer\",\"top_k\":3}"
```

Status example:

```bash
curl http://127.0.0.1:8000/api/knowledge/status
```

Production ingest generates the active artifacts:

- `data/processed/chunks.jsonl`
- `data/processed/course_manifest.json`
- `data/processed/raptor_tree.json`
- `data/processed/faiss_index/`

If FAISS is unavailable, EduAgentX automatically uses the local fallback vector/keyword search.

The RAPTOR-style tree is exposed as:

```text
course -> module -> chapter/heading -> chunk
```

## Frontend

Install dependencies:

```bash
cd frontend
npm install
```

Start the UI:

```bash
npm run dev
```

Open `http://localhost:3000`.

## Tests

After installing backend dependencies:

```bash
pytest backend/tests
```

Run the full automated check suite from the project root:

```bash
python scripts/run_checks.py
```

To install dependencies before running checks:

```bash
python scripts/run_checks.py --install-deps
```

The current tests cover health/chat, profile update, resource generation, path generation, assessment/BKT update, and the basic RAG zip-manifest-chunk-search loop.

Example API checks:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/api/resources/generate \
  -H "Content-Type: application/json" \
  -d "{\"student_id\":\"demo\",\"topic\":\"Transformer\",\"knowledge_point\":\"Transformer\"}"

curl -X POST http://127.0.0.1:8000/api/path/generate \
  -H "Content-Type: application/json" \
  -d "{\"student_id\":\"demo\",\"goal\":\"Build an LLM Agent\",\"horizon_days\":14}"

curl -X POST http://127.0.0.1:8000/api/assessment/submit \
  -H "Content-Type: application/json" \
  -d "{\"student_id\":\"demo\",\"knowledge_point\":\"Transformer\",\"is_correct\":true,\"score\":0.9}"
```

## LLM API 切换

Create local environment config:

```bash
cp .env.example .env
```

Fill these fields in `.env` with your own credentials:

```text
LLM_PROVIDER=spark
ENABLE_MOCK_LLM=false
ENABLE_LLM_FALLBACK=true

SPARK_API_BASE=https://spark-api-open.xf-yun.com/x2/chat/completions
SPARK_MODEL=x1
SPARK_APP_ID=
SPARK_API_KEY=
SPARK_API_SECRET=
SPARK_API_PASSWORD=
```

Spark mock mode for local demos without credentials:

```text
ENABLE_MOCK_LLM=true
LLM_PROVIDER=spark
ENABLE_LLM_FALLBACK=true
```

Real Spark mode:

```text
ENABLE_MOCK_LLM=false
LLM_PROVIDER=spark
ENABLE_LLM_FALLBACK=true
```

For X1.5, switch both endpoint and model:

```text
SPARK_API_BASE=https://spark-api-open.xf-yun.com/v2/chat/completions
SPARK_MODEL=x1.5
```

OpenAI-compatible mode:

```text
ENABLE_MOCK_LLM=false
LLM_PROVIDER=openai-compatible
ENABLE_LLM_FALLBACK=true

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_BASE=https://api.openai.com/v1/chat/completions
```

`OPENAI_API_BASE` can also point to other OpenAI-compatible providers or self-hosted gateways, as long as they expose a `/chat/completions` style interface.

Check backend model status:

```bash
curl http://127.0.0.1:8000/api/model/status
```

Security notes:

- Do not commit `.env`.
- Do not expose real keys in screenshots, docs, logs, or GitHub.
- The frontend never calls LLM providers directly; all model calls go through the backend `SparkWrapper` gateway.
- If the real API fails and `ENABLE_LLM_FALLBACK=true`, EduAgentX automatically falls back to mock output for demos.
