from __future__ import annotations

import json
from typing import Iterator

from backend.agents.orchestrator import EduAgentOrchestrator
from backend.schemas.chat import ChatRequest


class ChatService:
    """Coordinates the multi-agent workflow behind /api/chat."""

    def __init__(self, orchestrator: EduAgentOrchestrator | None = None) -> None:
        self.orchestrator = orchestrator or EduAgentOrchestrator()

    def chat(self, request: ChatRequest) -> dict:
        return self.orchestrator.run(request)

    def stream_chat(self, request: ChatRequest) -> Iterator[str]:
        try:
            for event in self.orchestrator.stream(request):
                yield self._format_sse(event["event"], event["data"])
        except Exception as exc:
            yield self._format_sse("error", {"message": str(exc)[:240]})

    def _format_sse(self, event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
