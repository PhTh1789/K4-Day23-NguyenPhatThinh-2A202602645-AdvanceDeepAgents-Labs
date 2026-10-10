"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"

_REFERENCE_HEADING = re.compile(r"(?m)^##[ \t]+References[ \t]*$")
_REFERENCE_LINE = re.compile(r"(?m)^\s*\[(\d+)\]\s*(.*)$")
_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*(?:,|–|-)\s*\d+)*)\](?!\s*\()")
_CODE = re.compile(r"(```.*?```|`[^`\n]*`)", re.DOTALL)
_URL = re.compile(r"https?://[^\s<>]+")


def _citation_numbers(body):
    """Yield citation numbers while ignoring code and Markdown links."""
    for index, segment in enumerate(_CODE.split(body)):
        if index % 2:
            continue
        for match in _CITATION_GROUP.finditer(segment):
            for part in re.split(r"\s*,\s*", match.group(1)):
                span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
                if span:
                    start, end = (int(value) for value in span.groups())
                    if 0 <= end - start <= 200:
                        yield from range(start, end + 1)
                    else:
                        yield start
                        yield end
                else:
                    yield int(part)


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    source_by_number = {}
    seen_urls = set()
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            problems.append(f"source #{index} is not an object")
            continue
        number = source.get("n")
        url = source.get("url")
        if not isinstance(number, int) or isinstance(number, bool):
            problems.append(f"source #{index} has non-integer n={number!r}")
        elif number in source_by_number:
            problems.append(f"duplicate source number [{number}]")
        else:
            source_by_number[number] = source
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            problems.append(f"source #{index} has invalid url={url!r}")
        elif url in seen_urls:
            problems.append(f"duplicate url: {url}")
        else:
            seen_urls.add(url)

    matches = list(_REFERENCE_HEADING.finditer(report_text))
    if not matches:
        return problems + ["missing ## References heading"]
    body = report_text[: matches[-1].start()]
    references = report_text[matches[-1].end() :]

    cited = set(_citation_numbers(body))
    for number in sorted(cited - set(source_by_number)):
        problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(set(source_by_number) - cited):
        problems.append(f"source [{number}] never cited")

    reference_by_number = {}
    for line_match in _REFERENCE_LINE.finditer(references):
        number = int(line_match.group(1))
        line = line_match.group(0)
        if number in reference_by_number:
            problems.append(f"duplicate reference line [{number}]")
            continue
        reference_by_number[number] = line
        if number not in source_by_number:
            problems.append(f"reference [{number}] has no matching source")
            continue
        urls = _URL.findall(line)
        if len(urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one http(s) URL")
        elif urls[0] != source_by_number[number].get("url"):
            problems.append(f"reference [{number}] URL does not match sources.json")

    for number in sorted(set(source_by_number) - set(reference_by_number)):
        problems.append(f"source [{number}] has no reference line")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
