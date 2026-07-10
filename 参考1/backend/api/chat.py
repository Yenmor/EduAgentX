from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.schemas.chat import ChatRequest
from backend.services.chat_service import ChatService


router = APIRouter(prefix="/api", tags=["chat"])
chat_service = ChatService()


@router.post("/chat")
def chat_endpoint(request: ChatRequest) -> dict:
    return chat_service.chat(request)


@router.post("/chat/stream")
def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        chat_service.stream_chat(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
