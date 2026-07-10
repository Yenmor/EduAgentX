# EduAgentX MVP Architecture

EduAgentX starts with a small but runnable loop:

- FastAPI exposes health, chat, SSE chat, and knowledge endpoints.
- Mock agents call `SparkWrapper.mock_generate()` until the real Spark API is configured.
- RAG components can ingest Markdown course zips into processed chunks, a manifest, a knowledge tree, and a vector index.
- SQLite and SQLAlchemy are scaffolded for future chat history, profiles, and evaluation records.
- Next.js renders the dashboard, chat, resources, learning path, and profile screens.

The first stage intentionally favors stable interfaces over complex intelligence.
