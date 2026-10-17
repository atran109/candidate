from __future__ import annotations

import unittest

from backend.main import app
from fastapi.testclient import TestClient


class ChatEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_chat_endpoint_exists_and_accepts_messages(self) -> None:
        response = self.client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        self.assertNotEqual(
            response.status_code, 404, "POST /chat should be registered"
        )
        self.assertNotEqual(
            response.status_code, 422, "POST /chat should accept the documented schema"
        )

    def test_chat_endpoint_rejects_malformed_payload(self) -> None:
        response = self.client.post("/chat", json={"messages": "nope"})
        self.assertEqual(response.status_code, 422)

    def test_chat_endpoint_returns_documented_shape(self) -> None:
        request_messages = [{"role": "user", "content": "hi"}]
        response = self.client.post("/chat", json={"messages": request_messages})
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        # Conversation echoed back with exactly one assistant turn appended.
        messages = payload["messages"]
        self.assertEqual(messages[: len(request_messages)], request_messages)
        self.assertEqual(len(messages), len(request_messages) + 1)
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertIsInstance(messages[-1]["content"], str)
        self.assertTrue(messages[-1]["content"])

        # Tool-call log entries match ToolCallLog.
        self.assertIsInstance(payload["tool_calls"], list)
        for call in payload["tool_calls"]:
            self.assertEqual(set(call), {"name", "arguments", "result", "error"})
            self.assertIsInstance(call["name"], str)
            self.assertIsInstance(call["arguments"], dict)


if __name__ == "__main__":
    unittest.main()
