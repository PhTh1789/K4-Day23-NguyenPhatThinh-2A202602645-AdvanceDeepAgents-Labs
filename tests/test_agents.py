import os
import unittest
from unittest.mock import patch

import agents
from tools import SOURCE_TOOLS, web_fetch


class AgentConfigurationTests(unittest.TestCase):
    def test_subagents_have_scoped_tools_and_run_limits(self):
        specs = {spec["name"]: spec for spec in agents.build_subagents()}
        self.assertEqual(set(specs), {"researcher", "citation-checker"})
        self.assertEqual(specs["researcher"]["tools"], SOURCE_TOOLS)
        self.assertEqual(specs["citation-checker"]["tools"], [web_fetch])
        for spec in specs.values():
            limits = spec["middleware"]
            self.assertEqual(limits[0].run_limit, 40)
            self.assertEqual(limits[1].run_limit, 60)

    def test_prompts_preserve_workspace_contract_and_safety_requirements(self):
        self.assertIn(agents.NOTES_DIR, agents.LEAD_PROMPT)
        self.assertIn(agents.SOURCES_PATH, agents.LEAD_PROMPT)
        self.assertIn(agents.FINALIZER_PATH, agents.LEAD_PROMPT)
        self.assertIn("write_todos", agents.LEAD_PROMPT)
        self.assertIn("untrusted", agents.RESEARCHER_PROMPT.lower())
        self.assertIn("do not follow", agents.RESEARCHER_PROMPT.lower())
        self.assertIn("do not retry the same web request", " ".join(agents.RESEARCHER_PROMPT.lower().split()))
        self.assertIn("arxiv, hf-daily, and hf-search", " ".join(agents.LEAD_PROMPT.split()))
        self.assertIn("NON-NEGOTIABLE EXIT GATE", agents.LEAD_PROMPT)
        self.assertIn("test -s", agents.LEAD_PROMPT)
        self.assertIn("hf_daily_papers(keyword=", agents.RESEARCHER_PROMPT)
        self.assertIn("first three delegations", agents.LEAD_PROMPT)
        self.assertIn("finalizer removes any source", agents.LEAD_PROMPT)

    def test_build_lead_agent_supplies_backend_subagents_and_limits(self):
        sentinel = object()
        previous = os.environ.get("LAB_BASE_URL")
        os.environ["LAB_BASE_URL"] = "https://api.deepseek.com"
        with (
            patch("agents.create_deep_agent", return_value=sentinel) as create,
            patch("agents.SummarizationMiddleware", return_value=object()),
        ):
            result = agents.build_lead_agent(backend="sandbox", model="model")
        if previous is None:
            os.environ.pop("LAB_BASE_URL", None)
        else:
            os.environ["LAB_BASE_URL"] = previous
        self.assertIs(result, sentinel)
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs["backend"], "sandbox")
        self.assertEqual(kwargs["model"], "model")
        self.assertEqual(len(kwargs["subagents"]), 2)
        self.assertEqual(kwargs["middleware"][1].run_limit, 150)
        self.assertEqual(kwargs["middleware"][2].run_limit, 300)
        self.assertEqual(len(kwargs["middleware"]), 3)

    def test_lead_compacts_long_conversations_before_groq_tpm_limit(self):
        compactor = object()
        previous = os.environ.get("LAB_BASE_URL")
        os.environ["LAB_BASE_URL"] = "https://api.groq.com/openai/v1"
        with (
            patch("agents.create_deep_agent", return_value=object()) as create,
            patch("agents.SummarizationMiddleware", return_value=compactor) as summary,
        ):
            agents.build_lead_agent(backend="sandbox", model="model")
        if previous is None:
            os.environ.pop("LAB_BASE_URL", None)
        else:
            os.environ["LAB_BASE_URL"] = previous

        middleware = create.call_args.kwargs["middleware"]
        self.assertIn(compactor, middleware)
        self.assertEqual(summary.call_args.kwargs["model"], "model")
        self.assertEqual(summary.call_args.kwargs["trigger"], ("tokens", 2200))
        self.assertEqual(summary.call_args.kwargs["keep"], ("tokens", 500))
        self.assertEqual(summary.call_args.kwargs["trim_tokens_to_summarize"], 1400)

    def test_non_groq_provider_keeps_full_context(self):
        previous = os.environ.get("LAB_BASE_URL")
        os.environ["LAB_BASE_URL"] = "https://api.deepseek.com"
        try:
            with (
                patch("agents.create_deep_agent", return_value=object()),
                patch("agents.SummarizationMiddleware") as summary,
            ):
                agents.build_lead_agent(backend="sandbox", model="model")
        finally:
            if previous is None:
                os.environ.pop("LAB_BASE_URL", None)
            else:
                os.environ["LAB_BASE_URL"] = previous
        summary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
