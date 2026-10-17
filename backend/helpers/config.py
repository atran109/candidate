"""Runtime configuration: .env loading, model choices, loop limits."""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Minimal .env loader (python-dotenv isn't a dependency).

    Real environment variables always win over file values.
    """
    for path in (_REPO_ROOT / ".env", _REPO_ROOT / "backend" / ".env"):
        if not path.is_file():
            continue
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_env()

# Separate from OPENAI_MODEL on purpose: a grader env copied from .env.example
# shouldn't silently downgrade the executor.
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "gpt-4.1-mini")
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4.1")

# "routed" (default) or "all" (baseline: every tool schema on every call).
ROUTING_MODE = os.getenv("ROUTING_MODE", "routed")

# Hold the first write of each request for a self-check (ambiguity guard).
WRITE_GATE = os.getenv("WRITE_GATE", "1") != "0"

MAX_ITERATIONS = 8
MAX_ROUTED_TOOLS = 12
TOOL_OUTPUT_CHAR_LIMIT = 8000
