"""Tool registry access and execution with logging.

`backend.main` imports `backend.solution` at module load, so anything from
`backend.main` is imported lazily here to avoid a circular import.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any

from backend.chat_schema import ToolCallLog

from .config import TOOL_OUTPUT_CHAR_LIMIT

log = logging.getLogger("chat.tools")

# Name fragments that mark a tool as read-only. Read-only batches run in
# parallel; anything that may write runs sequentially so mock-state ID
# counters can't race.
_READ_MARKERS = (
    "list", "get", "fetch", "find", "search", "read", "query", "status",
    "health", "info", "history", "conversation", "thread", "download",
    "instances", "free_slots", "free_busy",
)
_WRITE_MARKERS = (
    "create", "update", "delete", "send", "write", "reply", "patch", "move",
    "add", "remove", "trash", "merge", "insert", "clear", "empty", "edit",
    "star", "fork", "push", "trigger", "dismiss", "mark", "assign", "label",
    "modify", "copy", "duplicate", "watch", "triage", "request", "manage",
    "run_", "share", "generate",
)


@lru_cache(maxsize=1)
def all_specs() -> dict[str, Any]:
    from backend.main import get_all_tool_specs

    return get_all_tool_specs()


def openai_schemas(names: list[str]) -> list[dict[str, Any]]:
    from backend.main import get_openai_tools

    return get_openai_tools(names) if names else []


def is_read_only(name: str) -> bool:
    lowered = name.lower()
    if any(marker in lowered for marker in _WRITE_MARKERS):
        return False
    return any(marker in lowered for marker in _READ_MARKERS)


def _strip_nulls(args: dict[str, Any]) -> dict[str, Any]:
    # Strict mode makes the model send every optional param, usually as null.
    return {k: v for k, v in args.items() if v is not None}


def execute(name: str, raw_args: dict[str, Any]) -> ToolCallLog:
    args = _strip_nulls(raw_args)
    started = time.perf_counter()
    try:
        spec = all_specs()[name]
    except KeyError:
        return ToolCallLog(name=name, arguments=args, error=f"Unknown tool: {name}")
    try:
        result = spec.invoke(**args).model_dump(mode="json")["result"]
        entry = ToolCallLog(name=name, arguments=args, result=result)
    except Exception as exc:  # tool errors are data for the model, not crashes
        entry = ToolCallLog(
            name=name, arguments=args, error=f"{type(exc).__name__}: {exc}"
        )
    log.info(
        "tool %s %.0fms %s",
        name,
        (time.perf_counter() - started) * 1000,
        "ERROR " + entry.error if entry.error else "ok",
    )
    return entry


def execute_batch(calls: list[tuple[str, dict[str, Any]]]) -> list[ToolCallLog]:
    """Run one model turn's calls; results come back in the order requested."""
    if len(calls) > 1 and all(is_read_only(name) for name, _ in calls):
        with ThreadPoolExecutor(max_workers=min(8, len(calls))) as pool:
            return list(pool.map(lambda c: execute(*c), calls))
    return [execute(name, args) for name, args in calls]


def output_for_model(entry: ToolCallLog) -> str:
    """Serialize a tool result for the model, truncating very large payloads."""
    if entry.error:
        payload: Any = {"error": entry.error}
    else:
        payload = entry.result
    text = json.dumps(payload, default=str, separators=(",", ":"))
    if len(text) > TOOL_OUTPUT_CHAR_LIMIT:
        text = (
            text[:TOOL_OUTPUT_CHAR_LIMIT]
            + f"... [truncated {len(text) - TOOL_OUTPUT_CHAR_LIMIT} chars; "
            "narrow the query if you need more]"
        )
    return text
