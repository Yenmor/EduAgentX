from __future__ import annotations

from backend.agents.retriever_agent import RetrieverAgent
from backend.schemas.chat import ChatRequest, ChatResponse
from backend.services.spark_wrapper import SparkWrapper


TUTOR_SYSTEM_PROMPT = (
    "你是智能辅导智能体。请根据学生问题、学生画像和课程检索结果，"
    "提供清晰、分层、适合学生水平的解释。"
)


class TutorAgent:
    """Answers learning questions and attaches RAG citations when available."""

    def __init__(self, llm: SparkWrapper | None = None) -> None:
        self.llm = llm or SparkWrapper()
        self.retriever = RetrieverAgent()
        self.last_model_info = {"model_provider": "mock", "model_name": "mock", "fallback_used": False}

    def run(self, request: ChatRequest) -> ChatResponse:
        rag = self.retriever.retrieve(request.message, top_k=3)
        citations = rag.get("results", [])
        answer = self.answer_question(request.message, citations)
        return ChatResponse(
            reply=str(answer["answer"]),
            agent="TutorAgent",
            citations=citations,
            **self.last_model_info,
        )

    def answer_question(
        self,
        question: str,
        sources: list[dict],
        profile: dict | None = None,
        rewrite_instruction: str | None = None,
    ) -> dict:
        """Generate a structured tutoring answer grounded in retrieved snippets."""
        profile = profile or {}
        source_text = "\n".join(str(item.get("content", ""))[:360] for item in sources[:3])
        response = self.llm.generate(
            [
                {"role": "system", "content": TUTOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"学生问题：{question}\n学生画像：{profile}\n课程检索结果：{source_text}\n"
                        f"重写要求：{rewrite_instruction or '无'}"
                    ),
                },
            ]
        )
        self.last_model_info = self._model_info(response)
        answer = str(response.get("content", ""))
        if sources:
            answer += f"\n\n来源：{sources[0].get('source_file')} / {' > '.join(sources[0].get('heading_path', []))}"
        mermaid_graph = None
        if "attention" in question.lower() or "self-attention" in question.lower():
            mermaid_graph = "graph LR\n  Token[Input tokens] --> QKV[Q/K/V projections]\n  QKV --> Score[Attention scores]\n  Score --> Context[Contextual representation]"
        return {
            "answer": answer,
            "mermaid_graph": mermaid_graph,
            "code_example": None,
            "sources": sources,
            "confidence": 0.86 if sources else 0.45,
            **self.last_model_info,
        }

    def _model_info(self, result: dict) -> dict:
        return {
            "model_provider": str(result.get("model_provider", "mock")),
            "model_name": str(result.get("model_name", "mock")),
            "fallback_used": bool(result.get("fallback_used", False)),
        }
