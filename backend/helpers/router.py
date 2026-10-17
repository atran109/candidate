"""Tool routing: pick a small tool subset per request.

1. LLM router over a compact catalog (one line per tool). The catalog is a
   static prompt prefix, so OpenAI prompt caching makes repeat calls cheap.
   Structured output with an enum of real tool names means it can't
   hallucinate a tool.
2. Deterministic companion expansion: add lookup tools that writes depend on
   (the router reliably picks the "verb" but often forgets the prerequisite).
3. `search_tools` meta-tool lets the executor pull in more tools mid-loop.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from functools import lru_cache
from typing import Any

from openai import OpenAI

from backend.chat_schema import ChatMessage

from . import tools
from .config import MAX_ROUTED_TOOLS, ROUTER_MODEL
from .prompts import ROUTER_INSTRUCTIONS

log = logging.getLogger("chat.router")

SERVICE_LABELS = {
    "gmail": "Gmail",
    "googlecalendar": "Google Calendar",
    "googledrive": "Google Drive",
    "slack": "Slack",
    "linear": "Linear",
    "github": "GitHub",
    "perplexity": "Perplexity (web search)",
}

# tool → tools it usually needs alongside it (adds schemas only, not calls).
COMPANIONS: dict[str, list[str]] = {
    "slack_send_message": ["slack_list_conversations"],
    "slack_conversations_history": ["slack_list_conversations"],
    "slack_get_full_conversation": ["slack_list_conversations"],
    "slack_get_thread": ["slack_list_conversations"],
    "GOOGLECALENDAR_CREATE_EVENT": ["GMAIL_GET_CONTACTS", "GOOGLECALENDAR_FIND_FREE_SLOTS"],
    "GOOGLECALENDAR_UPDATE_EVENT": ["GOOGLECALENDAR_FIND_EVENT"],
    "GOOGLECALENDAR_PATCH_EVENT": ["GOOGLECALENDAR_FIND_EVENT"],
    "GOOGLECALENDAR_DELETE_EVENT": ["GOOGLECALENDAR_FIND_EVENT"],
    "GOOGLECALENDAR_REMOVE_ATTENDEE": ["GOOGLECALENDAR_FIND_EVENT"],
    "GOOGLECALENDAR_EVENTS_MOVE": ["GOOGLECALENDAR_FIND_EVENT", "GOOGLECALENDAR_LIST_CALENDARS"],
    "GMAIL_REPLY_TO_THREAD": ["GMAIL_FETCH_EMAILS"],
    "GMAIL_CREATE_EMAIL_DRAFT": ["GMAIL_GET_CONTACTS"],
    "GMAIL_MOVE_TO_TRASH": ["GMAIL_FETCH_EMAILS"],
    "GMAIL_DELETE_MESSAGE": ["GMAIL_FETCH_EMAILS"],
    "GMAIL_ADD_LABEL_TO_EMAIL": ["GMAIL_FETCH_EMAILS", "GMAIL_LIST_LABELS"],
    "GOOGLEDRIVE_EDIT_FILE": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_COPY_FILE": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_DOWNLOAD_FILE": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_CREATE_COMMENT": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_ADD_FILE_SHARING_PREFERENCE": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_DELETE_PERMISSION": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_FILES_MODIFY_LABELS": ["GOOGLEDRIVE_FIND_FILE"],
    "GOOGLEDRIVE_CREATE_SHORTCUT_TO_FILE": ["GOOGLEDRIVE_FIND_FILE"],
    "linear_create_issue": ["linear_list_teams"],
    "linear_update_issue": ["linear_get_issue", "linear_list_issue_statuses"],
    "linear_create_comment": ["linear_get_issue"],
    "github_issue_write": ["github_search_issues"],
    "github_add_issue_comment": ["github_search_issues"],
    "github_merge_pull_request": ["github_list_pull_requests"],
    "github_update_pull_request": ["github_list_pull_requests"],
}

# Cheap discovery tools added whenever the router touches a service, so the
# executor can always resolve names / broaden a search within that service.
SERVICE_CORE: dict[str, list[str]] = {
    "gmail": ["GMAIL_FETCH_EMAILS"],
    "googlecalendar": ["GOOGLECALENDAR_EVENTS_LIST", "GOOGLECALENDAR_LIST_CALENDARS"],
    "googledrive": ["GOOGLEDRIVE_FIND_FILE"],
    "slack": ["slack_list_conversations", "slack_conversations_history"],
    "linear": ["linear_list_issues", "linear_list_projects", "linear_list_teams"],
    "github": ["github_search_issues", "github_list_pull_requests"],
}

SEARCH_TOOLS_NAME = "search_tools"
SEARCH_TOOLS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "name": SEARCH_TOOLS_NAME,
    "description": (
        "Find additional tools when none of your current tools can do what's "
        "needed. Describe the capability, e.g. 'delete a calendar event' or "
        "'list github branches'. Matching tools become available on your next step."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Capability needed."}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(timeout=ROUTER_DEADLINE_S, max_retries=0)
    return _client


@lru_cache(maxsize=1)
def catalog_text() -> str:
    by_service: dict[str, list[str]] = defaultdict(list)
    for name, spec in sorted(tools.all_specs().items()):
        by_service[spec.service].append(f"{name}: {spec.description}")
    sections = [
        f"## {SERVICE_LABELS.get(svc, svc)}\n" + "\n".join(lines)
        for svc, lines in by_service.items()
    ]
    return ROUTER_INSTRUCTIONS + "\n\n".join(sections)


def _router_input(messages: list[ChatMessage]) -> list[dict[str, str]]:
    recent = [m for m in messages if m.role in ("user", "assistant")][-6:]
    return [{"role": m.role, "content": m.content[:2000]} for m in recent]


def expand_companions(selected: list[str]) -> list[str]:
    specs = tools.all_specs()
    out = list(dict.fromkeys(selected))
    extras = [e for name in selected for e in COMPANIONS.get(name, [])]
    services = dict.fromkeys(specs[name].service for name in selected)
    extras += [e for svc in services for e in SERVICE_CORE.get(svc, [])]
    for extra in extras:
        if extra not in out and extra in specs:
            out.append(extra)
    return out


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
HEDGE_AFTER_S = 2.5  # fire a duplicate request if the first is this slow
ROUTER_DEADLINE_S = 9.0


def _ask_router(messages: list[ChatMessage]) -> list[str]:
    # Plain-text output instead of a JSON-schema enum: gpt-4.1-mini's
    # structured output occasionally emits runaway whitespace (hundreds of
    # junk tokens → multi-second stalls). Names are validated below anyway.
    resp = _openai().responses.create(
        model=ROUTER_MODEL,
        instructions=catalog_text(),
        input=_router_input(messages),
        temperature=0,
        max_output_tokens=200,
    )
    specs = tools.all_specs()
    text = resp.output_text.strip()
    names = [n for n in dict.fromkeys(_NAME.findall(text)) if n in specs]
    if not names and text.upper() != "NONE":
        # Model replied conversationally instead of selecting: treat as a miss.
        raise ValueError(f"router returned no tool names: {text[:120]!r}")
    return names


def _hedged(messages: list[ChatMessage]) -> list[str]:
    """Tail-latency hedge: if the first router call is slow, race a second."""
    pool = ThreadPoolExecutor(max_workers=2)
    try:
        futures = [pool.submit(_ask_router, messages)]
        done, _ = wait(futures, timeout=HEDGE_AFTER_S)
        if not done:
            log.info("router slow; hedging with a second request")
            futures.append(pool.submit(_ask_router, messages))
        remaining = ROUTER_DEADLINE_S - HEDGE_AFTER_S
        while remaining > 0:
            t0 = time.perf_counter()
            done, pending = wait(futures, timeout=remaining, return_when=FIRST_COMPLETED)
            for f in done:
                if f.exception() is None:
                    return f.result()
            futures = list(pending)
            if not futures:
                break
            remaining -= time.perf_counter() - t0
        raise TimeoutError("router did not answer in time")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _fallback(messages: list[ChatMessage]) -> list[str]:
    last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    return [name for name, _ in search_tools(last_user, exclude=[], limit=8)]


def select_tools(messages: list[ChatMessage]) -> list[str]:
    started = time.perf_counter()
    try:
        picked = _hedged(messages)
    except Exception:
        # Router failure shouldn't fail the request: seed tools by keyword
        # match on the user's message; search_tools remains available too.
        log.exception("router failed; using keyword fallback")
        picked = _fallback(messages)
    picked = picked[:MAX_ROUTED_TOOLS]
    expanded = expand_companions(picked)
    log.info(
        "router %.0fms picked=%s companions=%s",
        (time.perf_counter() - started) * 1000,
        picked,
        expanded[len(picked):],
    )
    return expanded


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    words = set(_WORD.findall(text.lower().replace("_", " ")))
    # crude stemming so "events"/"event", "issues"/"issue" match
    return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3}


def search_tools(query: str, exclude: list[str], limit: int = 6) -> list[tuple[str, str]]:
    q = _tokens(query)
    scored = []
    for name, spec in tools.all_specs().items():
        if name in exclude:
            continue
        name_hits = len(q & _tokens(name))
        desc_hits = len(q & _tokens(spec.description))
        score = 2 * name_hits + desc_hits
        if score:
            scored.append((score, name, spec.description))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(name, desc) for _, name, desc in scored[:limit]]
