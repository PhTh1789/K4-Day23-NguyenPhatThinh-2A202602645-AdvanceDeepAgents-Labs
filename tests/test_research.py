import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import research


class FakeBackend:
    def __init__(self, files):
        self.files = files


class FakeModel:
    max_tokens = None


class FakeClient:
    def __init__(self):
        self.max_retries = 2


class RetryingFakeModel(FakeModel):
    def __init__(self):
        self.max_retries = 2
        self.root_client = FakeClient()
        self.root_async_client = FakeClient()


class DiagnosticBackend:
    def execute(self, command):
        self.command = command
        return type(
            "Response",
            (),
            {
                "exit_code": 0,
                "output": "/tmp/work/research/notes/01-overview.md\n"
                "/tmp/work/research/sources.json\n",
            },
        )()


class ResearchPersistenceTests(unittest.TestCase):
    def test_report_contract_rejects_missing_level_one_title(self):
        report = """## TL;DR\n- Finding [1]\n\n## Background\nClaim [1].\n\n## Theme A\nClaim [2].\n\n## Theme B\nClaim [3].\n\n## Theme C\nClaim [1].\n\n## Trends and open problems\nClaim [2].\n\n## References\n[1] x\n"""
        sources = [
            {"n": 1, "source": "arxiv", "title": "Topic foundations", "url": "https://a.test"},
            {"n": 2, "source": "hf-daily", "title": "Topic methods", "url": "https://b.test"},
            {"n": 3, "source": "hf-search", "title": "Topic applications", "url": "https://c.test"},
        ]

        with self.assertRaisesRegex(RuntimeError, "level-one title"):
            research.validate_report_contract(report, sources, "survey about topic")

    def test_report_contract_rejects_missing_required_trends_section(self):
        report = """# Topic\n\n## TL;DR\n- Finding [1]\n\n## Background\nClaim [1].\n\n## Theme A\nClaim [2].\n\n## Theme B\nClaim [3].\n\n## Theme C\nClaim [1].\n\n## References\n[1] x\n"""
        sources = [
            {"n": 1, "source": "arxiv", "title": "Topic foundations", "url": "https://a.test"},
            {"n": 2, "source": "hf-daily", "title": "Topic methods", "url": "https://b.test"},
            {"n": 3, "source": "hf-search", "title": "Topic applications", "url": "https://c.test"},
        ]

        with self.assertRaisesRegex(RuntimeError, "trends and open problems"):
            research.validate_report_contract(report, sources, "survey about topic")

    def test_report_contract_allows_an_indirectly_named_source_with_broad_topic_coverage(self):
        report = """# Topic survey\n\n## TL;DR\n- Finding [1].\n\n## Background\nClaim [1].\n\n## Theme A\nClaim [2].\n\n## Theme B\nClaim [3].\n\n## Theme C\nClaim [4].\n\n## Trends and open problems\nOpen question [4].\n\n## References\n[1] x\n"""
        sources = [
            {"n": 1, "source": "arxiv", "title": "Topic foundations", "url": "https://a.test"},
            {"n": 2, "source": "hf-daily", "title": "Topic methods", "url": "https://b.test"},
            {"n": 3, "source": "hf-search", "title": "Topic applications", "url": "https://c.test"},
            {"n": 4, "source": "web", "title": "Model-based recursive self-improvement", "url": "https://d.test"},
        ]

        research.validate_report_contract(report, sources, "survey about topic")

    def test_apply_model_limit_uses_a_bounded_configured_output_budget(self):
        model = FakeModel()
        previous = os.environ.get("LAB_MAX_TOKENS")
        previous_base_url = os.environ.get("LAB_BASE_URL")
        try:
            os.environ.pop("LAB_BASE_URL", None)
            os.environ["LAB_MAX_TOKENS"] = "1200"
            self.assertIs(research.apply_model_limit(model), model)
        finally:
            if previous is None:
                os.environ.pop("LAB_MAX_TOKENS", None)
            else:
                os.environ["LAB_MAX_TOKENS"] = previous
            if previous_base_url is None:
                os.environ.pop("LAB_BASE_URL", None)
            else:
                os.environ["LAB_BASE_URL"] = previous_base_url
        self.assertEqual(model.max_tokens, 1200)

    def test_apply_model_limit_throttles_groq_and_clamps_its_output(self):
        model = FakeModel()
        previous = {name: os.environ.get(name) for name in ("LAB_BASE_URL", "LAB_MAX_TOKENS")}
        try:
            os.environ["LAB_BASE_URL"] = "https://api.groq.com/openai/v1"
            os.environ["LAB_MAX_TOKENS"] = "2000"
            research.apply_model_limit(model)
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.assertEqual(model.max_tokens, 1000)
        self.assertIsNotNone(model.rate_limiter)

    def test_apply_model_limit_bounds_hidden_provider_retries(self):
        model = RetryingFakeModel()

        research.apply_model_limit(model)

        self.assertEqual(model.max_retries, 1)
        self.assertEqual(model.root_client.max_retries, 1)
        self.assertEqual(model.root_async_client.max_retries, 1)

    def test_apply_model_limit_rejects_invalid_or_unsupported_configuration(self):
        previous = os.environ.get("LAB_MAX_TOKENS")
        try:
            os.environ["LAB_MAX_TOKENS"] = "zero"
            with self.assertRaisesRegex(RuntimeError, "LAB_MAX_TOKENS"):
                research.apply_model_limit(FakeModel())
        finally:
            if previous is None:
                os.environ.pop("LAB_MAX_TOKENS", None)
            else:
                os.environ["LAB_MAX_TOKENS"] = previous
    def test_slugify_is_bounded_and_cannot_escape_reports(self):
        self.assertEqual(research.slugify("../../World Model!!"), "world-model")
        self.assertEqual(research.slugify(""), "topic")
        self.assertLessEqual(len(research.slugify("a" * 100)), 60)

    def test_summarize_counts_lead_tool_calls_and_usage(self):
        messages = [
            {
                "tool_calls": [{"name": "task"}, {"name": "arxiv_search"}],
                "usage_metadata": {"input_tokens": 10, "output_tokens": 4},
            },
            {"tool_calls": [{"name": "task"}], "usage_metadata": {"input_tokens": 3, "output_tokens": 2}},
        ]
        summary = research.summarize(messages, 1.26, "test-model")
        self.assertEqual(summary["subagent_calls"], 2)
        self.assertEqual(summary["tool_calls"], {"arxiv_search": 1, "task": 2})
        self.assertEqual(summary["tokens"], {"input": 13, "output": 6})
        self.assertEqual(summary["elapsed_s"], 1.3)

    def test_run_diagnostics_reports_agent_counts_and_sandbox_file_names_only(self):
        backend = DiagnosticBackend()
        messages = [
            {"tool_calls": [{"name": "task"}, {"name": "task"}, {"name": "arxiv_search"}]},
            {"tool_calls": [{"name": "task"}]},
        ]

        diagnostic = research.run_diagnostics(backend, messages)

        self.assertIn("lead_messages=2", diagnostic)
        self.assertIn("subagent_calls=3", diagnostic)
        self.assertIn("arxiv_search=1", diagnostic)
        self.assertIn("/tmp/work/research/sources.json", diagnostic)
        self.assertNotIn("content", diagnostic.lower())

    def test_source_evidence_rejects_a_url_not_returned_by_a_host_tool(self):
        sources = [{"n": 1, "url": "https://example.com/fabricated", "source": "web"}]

        with self.assertRaisesRegex(RuntimeError, "not returned by a host research tool"):
            research.validate_source_evidence(sources, {("web", "https://example.com/real")})

    def test_canonical_source_url_requires_host_observed_id(self):
        evidence = {("arxiv", "https://arxiv.org/abs/2501.00001")}

        self.assertEqual(
            research.canonical_source_url("arxiv", "2501.00001v2", evidence),
            "https://arxiv.org/abs/2501.00001",
        )
        self.assertIsNone(research.canonical_source_url("arxiv", "9999.99999", evidence))

    def test_provenance_repair_prompt_lists_only_permitted_source_pairs(self):
        prompt = research.provenance_repair_prompt(
            {("hf-search", "https://huggingface.co/papers/2501.00001")}
        )

        self.assertIn("hf-search | https://huggingface.co/papers/2501.00001", prompt)
        self.assertIn("must not invent", " ".join(prompt.split()))
        self.assertIn(research.SOURCES_PATH, prompt)
        self.assertIn(research.PROVENANCE_PATH, prompt)

    def test_save_outputs_writes_an_atomic_three_file_artifact_set(self):
        sources = [
            {"n": 1, "url": "https://example.org/a", "source": "arxiv", "title": "Topic foundation"},
            {"n": 2, "url": "https://example.org/b", "source": "hf-daily", "title": "Topic methods"},
            {"n": 3, "url": "https://example.org/c", "source": "hf-search", "title": "Topic applications"},
        ]
        backend = FakeBackend(
            {
                research.REPORT_PATH: b"""# Topic\n\n## TL;DR\n- Finding [1].\n\n## Background\nClaim [1].\n\n## Theme A\nClaim [1].\n\n## Theme B\nClaim [2].\n\n## Theme C\nClaim [3].\n\n## Trends and open problems\nOpen question [3].\n\n## References\n[1] Topic foundation. arxiv. https://example.org/a\n[2] Topic methods. hf-daily. https://example.org/b\n[3] Topic applications. hf-search. https://example.org/c\n""",
                research.SOURCES_PATH: json.dumps(sources).encode(),
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory) / "reports"
            with patch("research.download", return_value=backend.files):
                path = research.save_outputs(
                    backend,
                    "Topic",
                    [],
                    1.0,
                    "model",
                    reports,
                    source_evidence={
                        ("arxiv", "https://example.org/a"),
                        ("hf-daily", "https://example.org/b"),
                        ("hf-search", "https://example.org/c"),
                    },
                )
            self.assertEqual(path, reports / "topic.md")
            self.assertEqual(json.loads((reports / "topic.sources.json").read_text()), sources)
            meta = json.loads((reports / "topic.meta.json").read_text())
            self.assertEqual(meta["source_families"], ["arxiv", "hf-daily", "hf-search"])

    def test_save_outputs_leaves_no_report_when_download_is_invalid(self):
        backend = FakeBackend({research.REPORT_PATH: b"", research.SOURCES_PATH: b"not json"})
        with tempfile.TemporaryDirectory() as directory:
            reports = Path(directory) / "reports"
            with patch("research.download", return_value=backend.files):
                with self.assertRaises(RuntimeError):
                    research.save_outputs(backend, "Topic", [], 1.0, "model", reports)
            self.assertFalse(reports.exists())


if __name__ == "__main__":
    unittest.main()
