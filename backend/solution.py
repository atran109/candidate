from __future__ import annotations

import logging

from backend.chat_schema import ChatMessage, ChatRequest, ChatResponse
from backend.helpers.agent import run_chat

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
# Keep our own logs readable; the OpenAI/httpx clients are chatty at INFO.
logging.getLogger("httpx").setLevel(logging.WARNING)


def chat(request: ChatRequest) -> ChatResponse:
    """Route the latest request to a small tool subset, run the tool loop,
    and return the assistant reply plus an ordered log of every tool call."""
    reply, tool_calls = run_chat(request.messages)
    return ChatResponse(
        messages=[*request.messages, ChatMessage(role="assistant", content=reply)],
        tool_calls=tool_calls,
    )
