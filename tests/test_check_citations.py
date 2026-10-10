import unittest

from check_citations import check


SOURCES = [
    {
        "n": 1,
        "id": "2501.00001",
        "url": "https://arxiv.org/abs/2501.00001",
        "title": "First paper",
        "date": "2025-01-02",
        "source": "arxiv",
    },
    {
        "n": 2,
        "id": "second",
        "url": "https://example.org/second",
        "title": "Second source",
        "date": "2025-02-03",
        "source": "web",
    },
]


def valid_report(body="A supported claim [1][2]."):
    return f"""# Survey

{body}

## References
[1] First paper. arxiv. https://arxiv.org/abs/2501.00001 (2025-01-02)
[2] Second source. web. https://example.org/second (2025-02-03)
"""


class CitationCheckerTests(unittest.TestCase):
    def test_accepts_resolved_citations_and_exact_reference_urls(self):
        self.assertEqual(check(valid_report(), SOURCES), [])

    def test_expands_grouped_citations_and_ignores_code_and_markdown_links(self):
        report = valid_report(
            "Grouped evidence [1, 2] and [1-2]. `Example [99]`. "
            "A Markdown link [88](https://example.org) is not a citation."
        )
        self.assertEqual(check(report, SOURCES), [])

    def test_rejects_empty_or_duplicate_sources(self):
        self.assertIn("no sources in sources.json", check(valid_report(), []))
        duplicated = [dict(SOURCES[0]), dict(SOURCES[0], n=2)]
        problems = check(valid_report(), duplicated)
        self.assertTrue(any("duplicate url" in problem for problem in problems))

    def test_rejects_citations_missing_from_sources_and_uncited_sources(self):
        report = valid_report("Known claim [1] and unknown claim [3].")
        problems = check(report, SOURCES)
        self.assertTrue(any("[3] cited but missing" in problem for problem in problems))
        self.assertTrue(any("source [2] never cited" in problem for problem in problems))

    def test_rejects_missing_or_invalid_reference_lines(self):
        missing = "# Survey\n\nClaim [1][2]."
        self.assertTrue(any("missing ## References" in p for p in check(missing, SOURCES)))

        invalid = valid_report().replace(
            "https://example.org/second", "https://wrong.example/second", 1
        )
        self.assertTrue(any("does not match" in p for p in check(invalid, SOURCES)))


if __name__ == "__main__":
    unittest.main()
