import json
import unittest
from unittest.mock import patch

import tools


class FakeResponse:
    def __init__(self, *, status_code=200, text="", payload=None, headers=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class SourceToolTests(unittest.TestCase):
    def setUp(self):
        tools._last_arxiv_call = 0.0
        tools.begin_source_evidence_capture()

    def test_with_retry_retries_retryable_errors_without_sleeping_after_last_attempt(self):
        calls = 0

        def flaky():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise tools.RetryableError("busy", retry_after=0)
            return "ok"

        with patch("tools.time.sleep") as sleep:
            self.assertEqual(tools.with_retry(flaky, attempts=3), "ok")
        self.assertEqual(calls, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_exa_call_has_a_short_bounded_retry_window(self):
        retry_kwargs = {}

        def capture_retry(_operation, **kwargs):
            retry_kwargs.update(kwargs)
            return {"content": []}

        with patch("tools.with_retry", side_effect=capture_retry):
            tools._exa_call("web_search_exa", {"query": "world model"})

        self.assertEqual(retry_kwargs["attempts"], 3)
        self.assertEqual(retry_kwargs["cap"], 15.0)

    def test_arxiv_search_normalizes_ids_and_whitespace(self):
        atom = """<?xml version='1.0'?>
        <feed xmlns='http://www.w3.org/2005/Atom'>
          <entry>
            <id>http://arxiv.org/abs/2501.00001v2</id>
            <published>2025-01-02T00:00:00Z</published>
            <title> A\n title </title>
            <summary> A\n long summary </summary>
          </entry>
        </feed>"""
        with patch("tools._request_once", return_value=FakeResponse(text=atom)), patch(
            "tools._wait_for_arxiv_slot"
        ):
            result = json.loads(tools.arxiv_search.invoke({"query": "world:model", "max_results": 99}))
        self.assertEqual(result[0]["id"], "2501.00001")
        self.assertEqual(result[0]["url"], "https://arxiv.org/abs/2501.00001")
        self.assertEqual(result[0]["title"], "A title")

    def test_arxiv_search_records_only_urls_returned_by_the_live_tool(self):
        atom = """<?xml version='1.0'?>
        <feed xmlns='http://www.w3.org/2005/Atom'>
          <entry><id>http://arxiv.org/abs/2501.00001</id><title>Paper</title></entry>
        </feed>"""
        with patch("tools._request_once", return_value=FakeResponse(text=atom)), patch(
            "tools._wait_for_arxiv_slot"
        ):
            tools.arxiv_search.invoke({"query": "world model"})

        self.assertEqual(
            tools.collected_source_evidence(),
            {("arxiv", "https://arxiv.org/abs/2501.00001")},
        )

    def test_hugging_face_search_prefers_ai_summary_and_sorts_by_upvotes(self):
        payload = [
            {"paper": {"id": "a", "title": "A", "summary": "old", "ai_summary": "new", "upvotes": 2}},
            {"paper": {"id": "b", "title": "B", "summary": "second", "upvotes": 5}},
        ]
        with patch("tools._request", return_value=FakeResponse(payload=payload)):
            result = json.loads(tools.hf_search_papers.invoke({"query": "agents", "limit": 10}))
        self.assertEqual([entry["id"] for entry in result], ["b", "a"])
        self.assertEqual(result[1]["summary"], "new")

    def test_web_search_records_urls_present_in_a_successful_tool_response(self):
        payload = {"content": [{"type": "text", "text": "Read https://example.org/research."}]}
        with patch("tools._exa_call", return_value=payload):
            tools.web_search.invoke({"query": "world model"})

        self.assertEqual(
            tools.collected_source_evidence(),
            {("web", "https://example.org/research")},
        )

    def test_exa_rate_limit_metadata_is_retryable_and_errors_redact_keys(self):
        rate_limited = 'data: {"result":{"_meta":{"rateLimitReached":true},"content":[]}}\n'
        with self.assertRaises(tools.RetryableError):
            tools._parse_exa_response(rate_limited)
        self.assertEqual(tools._redact("failed key-123", "key-123"), "failed [REDACTED]")


if __name__ == "__main__":
    unittest.main()
