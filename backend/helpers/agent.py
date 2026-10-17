"""Executor loop: route → call model → run tools → repeat until a text answer."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI, OpenAIError

from backend.chat_schema import ChatMessage, ToolCallLog

from . import router, tools
from .config import AGENT_MODEL, MAX_ITERATIONS, ROUTING_MODE, WRITE_GATE
from .prompts import agent_instructions

log = logging.getLogger("chat.agent")

_client: OpenAI | None = None

WRITE_GATE_MESSAGE = (
    "[Write check — not executed yet.] Before changing anything, verify: "
    "(1) every target, recipient, and value comes from the user's words or a "
    "tool result (nothing invented); (2) if the user referred to something "
    "unspecified (\"the project\", \"the meeting\") or several items matched, "
    "the user chose which one — you didn't pick. If both hold (the usual case "
    "for an explicit request), immediately call the same tool again with the "
    "same arguments — do not reply with text first. Only if one fails: don't "
    "write; ask the user one concise question instead."
)


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(timeout=30, max_retries=2)
    return _client


def _to_input(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system":
            items.append({"role": "developer", "content": m.content})
        elif m.role == "tool":
            # Prior-turn tool output passed back by the client: keep as context.
            items.append(
                {"role": "assistant", "content": f"[earlier tool output]\n{m.content}"}
            )
        else:
            items.append({"role": m.role, "content": m.content})
    return items


def _call_key(name: str, args: dict[str, Any]) -> str:
    return name + json.dumps(args, sort_keys=True, default=str)


def run_chat(messages: list[ChatMessage]) -> tuple[str, list[ToolCallLog]]:
    """Never raises for LLM failures: returns an honest reply plus whatever
    tool calls already ran, so the log stays complete."""
    call_log: list[ToolCallLog] = []
    try:
        reply = _run(messages, call_log)
    except OpenAIError:
        log.exception("LLM call failed")
        done = [c.name for c in call_log if not c.error]
        reply = "I ran into an error reaching the language model and couldn't finish this request. " + (
            f"Actions completed before the error: {', '.join(done)}." if done else "No actions were taken."
        )
    return reply, call_log


def _run(messages: list[ChatMessage], call_log: list[ToolCallLog]) -> str:
    started = time.perf_counter()
    if ROUTING_MODE == "all":
        active = sorted(tools.all_specs())
        use_meta = False
    else:
        active = router.select_tools(messages)
        use_meta = True
    log.info("executor sees %d tools: %s", len(active), active if use_meta else "ALL")

    input_items = _to_input(messages)
    seen: dict[str, str] = {}  # identical successful call → cached model output
    usage = {"input": 0, "output": 0, "cached": 0}

    def create(**extra: Any):
        schemas = tools.openai_schemas(active)
        if use_meta:
            schemas = schemas + [router.SEARCH_TOOLS_SCHEMA]
        specs = tools.all_specs()
        services = None if not use_meta else sorted({specs[n].service for n in active})
        kwargs: dict[str, Any] = dict(
            model=AGENT_MODEL,
            instructions=agent_instructions(services),
            input=input_items,
            temperature=0,
        )
        if schemas:
            kwargs.update(tools=schemas, parallel_tool_calls=True)
        kwargs.update(extra)
        resp = _openai().responses.create(**kwargs)
        if resp.usage:
            usage["input"] += resp.usage.input_tokens
            usage["output"] += resp.usage.output_tokens
            details = resp.usage.input_tokens_details
            usage["cached"] += getattr(details, "cached_tokens", 0) or 0
        return resp

    reply = ""
    gate_passed = not WRITE_GATE
    for iteration in range(MAX_ITERATIONS):
        held_writes = False
        resp = create()
        calls = [o for o in resp.output if o.type == "function_call"]
        if not calls:
            reply = resp.output_text
            break

        input_items.extend(o.model_dump(exclude_none=True) for o in resp.output)

        outputs: dict[str, str] = {}
        to_run: list[tuple[str, str, dict[str, Any]]] = []
        for c in calls:
            try:
                args = json.loads(c.arguments or "{}")
            except json.JSONDecodeError as exc:
                outputs[c.call_id] = json.dumps({"error": f"Invalid JSON arguments: {exc}"})
                continue
            if c.name == router.SEARCH_TOOLS_NAME:
                found = router.search_tools(args.get("query") or "", exclude=active)
                active = active + [name for name, _ in found]
                outputs[c.call_id] = json.dumps(
                    {"added_tools": [f"{n}: {d}" for n, d in found]}
                )
                continue
            key = _call_key(c.name, tools._strip_nulls(args))
            if key in seen:
                outputs[c.call_id] = seen[key]
                continue
            if not gate_passed and not tools.is_read_only(c.name):
                # Hold the first write of the request for a self-check (not
                # executed, so not logged). Same call re-issued ⇒ it runs.
                outputs[c.call_id] = WRITE_GATE_MESSAGE
                held_writes = True
                continue
            to_run.append((c.call_id, c.name, args))
        if held_writes:
            gate_passed = True

        results = tools.execute_batch([(name, args) for _, name, args in to_run])
        for (call_id, name, args), entry in zip(to_run, results):
            call_log.append(entry)
            text = tools.output_for_model(entry)
            outputs[call_id] = text
            if not entry.error:
                seen[_call_key(name, entry.arguments)] = text

        for c in calls:
            input_items.append(
                {"type": "function_call_output", "call_id": c.call_id, "output": outputs[c.call_id]}
            )
    else:
        # Iteration cap hit: force a text answer from what we have.
        resp = create(tool_choice="none")
        reply = resp.output_text

    log.info(
        "done in %.0fms, %d tool calls, tokens in=%d (cached %d) out=%d",
        (time.perf_counter() - started) * 1000,
        len(call_log),
        usage["input"],
        usage["cached"],
        usage["output"],
    )
    return reply or "Sorry, I wasn't able to produce an answer."
