"""Scenario suite for POST /chat, modeled on the grader's scorecard.

Each scenario resets mock state, posts a conversation, then checks:
  - required tools were called (and succeeded)
  - forbidden tools were not called
  - optional end-state predicate over GET /state
  - optional substrings in the final assistant reply

Calls the real LLM, so it is skipped unless OPENAI_API_KEY is set.
Run a subset with: SCENARIOS=s01,s05 pnpm --filter backend test
"""

from __future__ import annotations

import os
import time
import unittest
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi.testclient import TestClient

from backend.main import app

State = dict[str, Any]


def _dump(obj: Any) -> Any:
    return obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj


def _slack_texts(state: State, channel: str) -> list[str]:
    for entry in state["slack"]["channel_messages"]:
        if entry["channel"] == channel:
            return [m["text"] for m in entry["messages"]]
    return []


def _calendar_events(state: State, calendar_id: str = "primary") -> list[dict]:
    for entry in state["googlecalendar"]["events"]:
        if entry["calendar_id"] == calendar_id:
            return entry["events"]
    return []


def _linear_issues(state: State) -> list[dict]:
    issues = state["linear"]["issues"]
    return list(issues.values()) if isinstance(issues, dict) else issues


def _as_list(x: Any) -> list:
    return list(x.values()) if isinstance(x, dict) else list(x)


@dataclass
class Scenario:
    id: str
    messages: list[dict[str, str]]
    required: list[str] = field(default_factory=list)
    # any one of these groups must be satisfied (e.g. list vs search variants)
    required_any: list[list[str]] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    min_calls: int = 0
    state_check: Callable[[State], bool] | None = None
    reply_contains: list[str] = field(default_factory=list)
    expect_question: bool = False  # ambiguous prompt → should ask, not act


def user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


WRITE_TOOLS = [
    "slack_send_message",
    "GOOGLECALENDAR_CREATE_EVENT",
    "GOOGLECALENDAR_QUICK_ADD",
    "linear_create_issue",
    "GMAIL_REPLY_TO_THREAD",
    "GOOGLEDRIVE_EMPTY_TRASH",
]

SCENARIOS: list[Scenario] = [
    # --- single-service lookups -------------------------------------------
    Scenario(
        "s01_greeting_no_tools",
        user("Hello!"),
        min_calls=0,
        forbidden=WRITE_TOOLS,
        state_check=None,
    ),
    Scenario(
        "s02_slack_list_conversations",
        user("What conversations do I have in Slack?"),
        required=["slack_list_conversations"],
        min_calls=1,
        reply_contains=["engineering", "launch-war-room"],
    ),
    Scenario(
        "s03_calendar_tomorrow",
        user("What's on my calendar tomorrow?"),
        required_any=[["GOOGLECALENDAR_EVENTS_LIST"], ["GOOGLECALENDAR_FIND_EVENT"]],
        min_calls=1,
        reply_contains=["Deep work"],
    ),
    Scenario(
        "s04_github_open_prs",
        user("What open pull requests are there in instalily/orchestrator?"),
        required_any=[["github_list_pull_requests"], ["github_search_pull_requests"]],
        min_calls=1,
        reply_contains=["router prototype"],
    ),
    Scenario(
        "s05_linear_high_priority",
        user("Which Linear issues are urgent priority?"),
        required_any=[["linear_list_issues"]],
        min_calls=1,
        reply_contains=["ENG-1", "ENG-3"],
    ),
    Scenario(
        "s06_slack_channel_history",
        user("What's the latest in the launch war room channel on Slack?"),
        required_any=[
            ["slack_conversations_history"],
            ["slack_get_full_conversation"],
            ["slack_search_messages"],
        ],
        min_calls=1,
        reply_contains=["webinar"],
    ),
    Scenario(
        "s07_vip_emails",
        user("Do I have any emails from the VIP client?"),
        required_any=[["GMAIL_FETCH_EMAILS"], ["GMAIL_LIST_THREADS"]],
        min_calls=1,
        reply_contains=["renewal"],
    ),
    # --- cross-service joins ----------------------------------------------
    Scenario(
        "s08_gmail_to_slack_summary",
        user(
            "Find the most recent email about the Q1 revenue review and post a "
            "summary to the engineering Slack channel."
        ),
        required=["GMAIL_FETCH_EMAILS", "slack_send_message"],
        min_calls=2,
        state_check=lambda s: len(_slack_texts(s, "C001")) == 5,
    ),
    Scenario(
        "s09_perplexity_to_linear",
        user(
            "Look up Q3 budget benchmarks for SaaS companies on the web and create "
            "a Linear issue in the Engineering team summarizing them."
        ),
        required=["perplexity_search", "linear_create_issue"],
        min_calls=2,
        state_check=lambda s: len(_linear_issues(s)) == 8,
    ),
    Scenario(
        "s10_slack_dm_to_calendar",
        user(
            "Morgan asked me on Slack to review something. Block 30 minutes on my "
            "calendar tomorrow at 2pm for it."
        ),
        required=["GOOGLECALENDAR_CREATE_EVENT"],
        min_calls=2,
        state_check=lambda s: any(
            "2026-04-09T14:00" in (e["start"].get("dateTime") or "")
            for e in _calendar_events(s)
        ),
    ),
    # --- writes with explicit targets -------------------------------------
    Scenario(
        "s11_schedule_with_person",
        user(
            "Schedule a 30-minute meeting with Morgan Lee next Tuesday at 2pm to go "
            "over revenue."
        ),
        required=["GOOGLECALENDAR_CREATE_EVENT"],
        min_calls=1,
        state_check=lambda s: any(
            "2026-04-14T14:00" in (e["start"].get("dateTime") or "")
            and any(
                a.get("email") == "finance@corp.com" for a in e.get("attendees") or []
            )
            for e in _calendar_events(s)
        ),
    ),
    Scenario(
        "s12_reply_vip_thread",
        user(
            "Reply to the VIP client's renewal email and tell them we'll send "
            "revised pricing by Friday."
        ),
        required=["GMAIL_REPLY_TO_THREAD"],
        min_calls=2,
        state_check=lambda s: len(_as_list(s["gmail"]["messages"])) == 7,
    ),
    Scenario(
        "s13_linear_update_status",
        user("Move ENG-4 to In Progress."),
        required=["linear_update_issue"],
        min_calls=1,
    ),
    Scenario(
        "s14_draft_email",
        user("Draft an email to Nadia Patel asking for Priya's onsite schedule."),
        required=["GMAIL_CREATE_EMAIL_DRAFT"],
        forbidden=["GMAIL_REPLY_TO_THREAD"],
        min_calls=1,
        state_check=lambda s: len(_as_list(s["gmail"]["drafts"])) == 2,
    ),
    # --- ambiguity: should ask, not act -----------------------------------
    Scenario(
        "s15_ambiguous_project_meeting",
        user("Schedule a 30-minute meeting with everyone on the project next week."),
        forbidden=["GOOGLECALENDAR_CREATE_EVENT", "GOOGLECALENDAR_QUICK_ADD"],
        expect_question=True,
    ),
    Scenario(
        "s16_ambiguous_repo_issue",
        user("Close the tool routing strategy issue on GitHub."),
        forbidden=["github_issue_write"],
        expect_question=True,  # exists in 3 repos
    ),
    # --- error / not-found paths: explain, don't fabricate ----------------
    Scenario(
        "s17_delete_missing_drive_file",
        user("Delete the file 'budget_2025.xlsx' from my Drive."),
        required_any=[["GOOGLEDRIVE_FIND_FILE"]],
        forbidden=["GOOGLEDRIVE_EMPTY_TRASH"],
        state_check=lambda s: len(_as_list(s["googledrive"]["files"])) == 6,
        reply_contains=["budget_2025.xlsx"],
    ),
    Scenario(
        "s18_post_missing_channel",
        user("Post 'deploy is done' in the #marketing Slack channel."),
        forbidden=["slack_send_message"],
        required_any=[["slack_list_conversations"]],
    ),
    # --- multi-turn: clarification answered ------------------------------
    Scenario(
        "s19_followup_after_clarification",
        [
            {
                "role": "user",
                "content": "Schedule a 30-minute meeting with everyone on the project next week.",
            },
            {
                "role": "assistant",
                "content": "Which project do you mean? I see Tool Orchestrator, "
                "Onboarding Refresh, and Renewals Risk in Linear.",
            },
            {"role": "user", "content": "Tool Orchestrator. Monday at 10am works."},
        ],
        required=["GOOGLECALENDAR_CREATE_EVENT"],
        required_any=[["linear_get_project"], ["linear_list_projects"], ["linear_list_users"]],
        min_calls=2,
        state_check=lambda s: any(
            "2026-04-13T10:00" in (e["start"].get("dateTime") or "")
            for e in _calendar_events(s)
        ),
    ),
]


@unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "needs OPENAI_API_KEY (live LLM)")
class ScenarioTests(unittest.TestCase):
    scorecard: list[str] = []

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        print("\n" + "\n".join(cls.scorecard))

    def _run(self, sc: Scenario) -> None:
        self.client.post("/reset")
        t0 = time.perf_counter()
        resp = self.client.post("/chat", json={"messages": sc.messages})
        ms = (time.perf_counter() - t0) * 1000
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        calls = body["tool_calls"]
        ok_names = [c["name"] for c in calls if not c.get("error")]
        names = [c["name"] for c in calls]
        reply = body["messages"][-1]["content"]

        failures: list[str] = []
        for name in sc.required:
            if name not in ok_names:
                failures.append(f"missing required tool: {name}")
        if sc.required_any and not any(
            all(n in ok_names for n in group) for group in sc.required_any
        ):
            failures.append(f"missing one of: {sc.required_any}")
        for name in sc.forbidden:
            if name in ok_names:
                failures.append(f"called forbidden tool: {name}")
        if sc.state_check:
            state = {k: _dump(v) for k, v in self.client.get("/state").json().items()}
            if not sc.state_check(state):
                failures.append("state check failed")
        for needle in sc.reply_contains:
            if needle.lower() not in reply.lower():
                failures.append(f"reply missing {needle!r}")
        if sc.expect_question and "?" not in reply:
            failures.append("expected a clarifying question")

        mark = "✓ PASS" if not failures else "✗ FAIL"
        line = f"  {mark}  [{sc.id}] {ms:7.0f} ms  ({len(calls)}/{sc.min_calls} calls) {names}"
        if failures:
            line += " — " + "; ".join(failures)
        self.scorecard.append(line)
        self.assertFalse(failures, f"{sc.id}: {failures}\nreply: {reply}")


def _make_test(sc: Scenario):
    def test(self: ScenarioTests) -> None:
        self._run(sc)

    return test


_only = {s.strip() for s in os.getenv("SCENARIOS", "").split(",") if s.strip()}
for _sc in SCENARIOS:
    if _only and not any(_sc.id.startswith(o) for o in _only):
        continue
    setattr(ScenarioTests, f"test_{_sc.id}", _make_test(_sc))


if __name__ == "__main__":
    unittest.main()
