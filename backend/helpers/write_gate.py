"""Out-of-band ambiguity check before the first write of a request.

A separate fast model call reviews the conversation, this request's lookup
results, and the proposed write(s), then answers PROCEED or ASK. The executor
never sees the check, so its behavior can't drift (an in-band "please verify"
message made it start asking permission for explicit requests).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from backend.chat_schema import ChatMessage, ToolCallLog

from .config import ROUTER_MODEL

log = logging.getLogger("chat.write_gate")

_client: OpenAI | None = None

GATE_INSTRUCTIONS = """You review a workplace assistant's proposed write action (create/send/update/delete) before it runs.

Answer PROCEED when the user asked for this action and its targets are determined: named by the user, or uniquely identified by a lookup, or clear from earlier turns. Explicit requests with a clear target ("reply to the VIP client's renewal email", "move ENG-4 to In Progress", "DM Morgan") are PROCEED — wording, message text, titles and times the assistant composed from the user's instructions are fine.

Answer ASK only when the user left the target unspecified and it could be several things — e.g. "the project" / "the meeting" / "the repo" when lookups show more than one plausible match and nothing in the conversation says which — or when the action uses people, emails, or IDs that appear nowhere in the conversation or lookup results (invented).

Reply with exactly `PROCEED`, or `ASK: <one concise question for the user, listing the options found>`."""


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(timeout=8, max_retries=1)
    return _client


def _render(
    messages: list[ChatMessage], lookups: list[ToolCallLog], proposed: list[tuple[str, dict[str, Any]]]
) -> str:
    convo = "\n".join(
        f"{m.role.upper()}: {m.content[:1500]}" for m in messages if m.role in ("user", "assistant")
    )
    seen = "\n".join(
        f"- {c.name}({json.dumps(c.arguments, default=str)[:200]}) -> "
        + (f"ERROR {c.error}" if c.error else json.dumps(c.result, default=str)[:1200])
        for c in lookups
    ) or "(none)"
    writes = "\n".join(
        f"- {name}({json.dumps({k: v for k, v in args.items() if v is not None}, default=str)[:800]})"
        for name, args in proposed
    )
    return f"Conversation:\n{convo}\n\nLookups this request:\n{seen}\n\nProposed write(s):\n{writes}"


def review(
    messages: list[ChatMessage],
    lookups: list[ToolCallLog],
    proposed: list[tuple[str, dict[str, Any]]],
) -> str | None:
    """Return None to proceed, or a clarifying question to ask the user.

    Fails open: if the check itself errors, the write proceeds (the executor's
    own prompt rules still apply).
    """
    started = time.perf_counter()
    try:
        resp = _openai().responses.create(
            model=ROUTER_MODEL,
            instructions=GATE_INSTRUCTIONS,
            input=_render(messages, lookups, proposed),
            temperature=0,
            max_output_tokens=150,
        )
        verdict = resp.output_text.strip()
    except Exception:
        log.exception("write gate failed; proceeding")
        return None
    log.info("write gate %.0fms: %s", (time.perf_counter() - started) * 1000, verdict[:120])
    if verdict.upper().startswith("ASK"):
        return verdict.split(":", 1)[1].strip() if ":" in verdict else verdict
    return None
