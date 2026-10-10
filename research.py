"""Run one complete sandboxed deep-research job."""
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import begin_source_evidence_capture, collected_source_evidence

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
PROVENANCE_PATH = f"{WORKDIR}/research/host_source_evidence.json"
PROVENANCE_NORMALIZER_PATH = f"{WORKDIR}/research/normalize_source_evidence.py"

PROVENANCE_NORMALIZER_SOURCE = r'''"""Normalize only source URLs whose source/id was observed by a host tool."""
import json
import re
import sys


def canonical_url(source, identifier):
    value = re.sub(r"v\d+$", "", str(identifier or "").strip())
    if not value:
        return None
    if source == "arxiv":
        return f"https://arxiv.org/abs/{value}"
    if source in {"hf-daily", "hf-search"}:
        return f"https://huggingface.co/papers/{value}"
    return None


def main():
    sources_path, evidence_path = sys.argv[1:3]
    with open(sources_path, encoding="utf-8") as handle:
        sources = json.load(handle)
    with open(evidence_path, encoding="utf-8") as handle:
        evidence = json.load(handle)
    allowed = {(item.get("source"), item.get("url")) for item in evidence if isinstance(item, dict)}
    unknown = []
    for entry in sources:
        if not isinstance(entry, dict):
            unknown.append("non-object source")
            continue
        pair = (entry.get("source"), entry.get("url"))
        if pair in allowed:
            continue
        canonical = canonical_url(entry.get("source"), entry.get("id"))
        if canonical and (entry.get("source"), canonical) in allowed:
            entry["url"] = canonical
        else:
            unknown.append("unobserved source")
    if unknown:
        print(f"NOT NORMALIZED: {len(unknown)} unobserved source record(s)")
        return 1
    with open(sources_path, "w", encoding="utf-8") as handle:
        json.dump(sources, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"NORMALIZED: {len(sources)} source record(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def slugify(topic):
    """Return a bounded filename-safe slug that cannot escape ``reports``."""
    value = re.sub(r"[^\w]+", "-", str(topic or "").strip().lower(), flags=re.UNICODE)
    value = value.strip("-._")[:60]
    return value or "topic"


def build_prompt(topic):
    """Build the fully-scoped task passed to the lead agent."""
    return (
        f"Produce a rigorous English survey about: {topic.strip()}. "
        "Follow your system workflow from planning through final citation validation. "
        "The deliverable is the sandbox report and sources registry; do not return a report in chat."
    )


def _field(message, name, default=None):
    if isinstance(message, dict):
        return message.get(name, default)
    return getattr(message, name, default)


def _token_count(usage, *names):
    if not isinstance(usage, dict):
        return 0
    for name in names:
        value = usage.get(name)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def summarize(messages, elapsed, model_name):
    """Summarize lead-only execution evidence for the report metadata."""
    tool_calls = Counter()
    input_tokens = output_tokens = 0
    for message in messages or []:
        for call in _field(message, "tool_calls", []) or []:
            name = _field(call, "name")
            if name:
                tool_calls[str(name)] += 1
        usage = _field(message, "usage_metadata", {}) or {}
        input_tokens += _token_count(usage, "input_tokens", "input_token_count")
        output_tokens += _token_count(usage, "output_tokens", "output_token_count")
    return {
        "model": str(model_name),
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_calls.get("task", 0),
        "tool_calls": dict(sorted(tool_calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def run_diagnostics(backend, messages):
    """Return a redacted snapshot of agent progress and sandbox files after a failed run.

    The sandbox is deleted when its context exits, so collect only execution counts and
    path names while it is still alive.  Never include model messages, tool responses, or
    environment values in this diagnostic.
    """
    summary = summarize(messages, 0, "")
    tool_counts = ", ".join(
        f"{name}={count}" for name, count in summary["tool_calls"].items()
    ) or "none"
    parts = [
        f"lead_messages={len(messages or [])}",
        f"subagent_calls={summary['subagent_calls']}",
        f"lead_tool_calls={tool_counts}",
    ]
    try:
        response = backend.execute(f"find {WORKDIR} -type f -print 2>/dev/null | sort | head -n 100")
        if getattr(response, "exit_code", 0) == 0:
            paths = str(getattr(response, "output", "")).strip()
            parts.append(f"sandbox_files={paths or 'none'}")
        else:
            parts.append(f"sandbox_inventory_exit={getattr(response, 'exit_code', 'unknown')}")
    except Exception as exc:
        parts.append(f"sandbox_inventory_error={type(exc).__name__}")
    return "; ".join(parts)


def _decode_file(files, path, label):
    content = files.get(path)
    if not content:
        raise RuntimeError(f"sandbox did not create {label}")
    try:
        return bytes(content).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"sandbox {label} is not UTF-8") from exc


def _write_atomically(path, content):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def validate_source_evidence(sources, source_evidence):
    """Reject sources that were not returned by a host-side research tool this run."""
    supported = set(source_evidence or ())
    unsupported = []
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        pair = (entry.get("source"), entry.get("url"))
        if pair not in supported:
            unsupported.append(f"{pair[0]}:{pair[1]}")
    if unsupported:
        details = ", ".join(unsupported[:5])
        raise RuntimeError(f"sources not returned by a host research tool: {details}")


def canonical_source_url(source, identifier, source_evidence):
    """Return a host-observed canonical URL for a known structured source identifier."""
    value = re.sub(r"v\d+$", "", str(identifier or "").strip())
    if source == "arxiv" and value:
        candidate = f"https://arxiv.org/abs/{value}"
    elif source in {"hf-daily", "hf-search"} and value:
        candidate = f"https://huggingface.co/papers/{value}"
    else:
        return None
    return candidate if (source, candidate) in set(source_evidence or ()) else None


def provenance_repair_prompt(source_evidence):
    """Tell a fresh repair pass which exact source-family/URL pairs it may keep."""
    pairs = sorted(set(source_evidence or ()))
    permitted = "\n".join(f"- {source} | {url}" for source, url in pairs[:80]) or "- none"
    return f"""The previous attempt created sandbox files but failed the host provenance check.
Do not return a chat summary. Read {SOURCES_PATH} and {REPORT_PATH}, then repair them inside
the sandbox. {PROVENANCE_PATH} is an uploaded, host-generated manifest of every permitted
source-family/URL pair. A source is permitted ONLY when its source-family and URL exactly
match one manifest record. For quick review, the same records are:\n{permitted}

Remove unsupported sources and their claims, or replace them with matching permitted records.
Do not convert an arXiv abs URL to a pdf URL; do not relabel a URL's source family; you must
not invent URLs, titles, dates, IDs, or citations. Preserve at least three source families;
use only matching source records from the existing researcher notes. Do not delegate more
researchers in this repair pass. Run the finalizer and validator, then satisfy the
non-negotiable exit gate.
"""


def upload_source_evidence(backend, source_evidence):
    """Expose only host-observed source URL pairs to a sandbox repair pass."""
    pairs = [
        {"source": source, "url": url}
        for source, url in sorted(set(source_evidence or ()))
    ]
    upload(backend, {PROVENANCE_PATH: json.dumps(pairs, ensure_ascii=False, indent=2).encode("utf-8")})


def reconcile_sandbox_source_evidence(backend, source_evidence):
    """Normalize only host-observed canonical URLs, then regenerate sandbox references."""
    upload_source_evidence(backend, source_evidence)
    upload(backend, {PROVENANCE_NORMALIZER_PATH: PROVENANCE_NORMALIZER_SOURCE.encode("utf-8")})
    normalized = backend.execute(
        f"python3 {PROVENANCE_NORMALIZER_PATH} {SOURCES_PATH} {PROVENANCE_PATH}"
    )
    if getattr(normalized, "exit_code", 0) != 0:
        return False
    finalized = backend.execute(f"python3 {FINALIZER_PATH}")
    if getattr(finalized, "exit_code", 0) != 0:
        raise RuntimeError("sandbox finalizer failed after provenance normalization")
    validated = backend.execute(f"python3 {VALIDATOR_PATH}")
    if getattr(validated, "exit_code", 0) != 0:
        raise RuntimeError("sandbox citation validator failed after provenance normalization")
    return True


def validate_report_contract(report, sources, topic):
    """Enforce the rubric's minimum structure and topic relevance before persistence."""
    body = re.split(r"(?im)^##\s+references\s*$", str(report), maxsplit=1)[0]
    if not re.search(r"(?m)^#\s+\S", body):
        raise RuntimeError("report contract missing a level-one title")
    headings = re.findall(r"(?im)^##\s+(.+?)\s*$", body)
    normalized = {heading.strip().lower() for heading in headings}
    required = {"tl;dr", "background", "trends and open problems"}
    missing = sorted(required - normalized)
    if missing:
        raise RuntimeError(f"report contract missing required section(s): {', '.join(missing)}")
    thematic = [heading for heading in headings if heading.strip().lower() not in required]
    if not 3 <= len(thematic) <= 6:
        raise RuntimeError("report contract requires 3-6 thematic synthesis sections")

    families = {entry.get("source") for entry in sources if isinstance(entry, dict)}
    if len(families) < 3:
        raise RuntimeError("report contract requires at least three source families")

    ignored = {"a", "an", "about", "and", "for", "in", "of", "on", "survey", "the", "to"}
    terms = {term.lower() for term in re.findall(r"[A-Za-z0-9]+", str(topic)) if term.lower() not in ignored}
    related_count = 0
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").lower()
        if not terms or any(re.search(rf"\b{re.escape(term)}\b", title) for term in terms):
            related_count += 1
    required_related = min(3, len(sources))
    if related_count < required_related:
        raise RuntimeError(
            f"report contract needs at least {required_related} source titles directly related to the topic"
        )


def quality_repair_prompt(topic):
    """Direct a bounded second pass to repair rubric-visible report defects in the sandbox."""
    return f"""The existing sandbox report for {topic!r} failed the report-quality contract.
Do not return a chat summary. Read {SOURCES_PATH} and {REPORT_PATH}, then repair them inside
the sandbox. The body must begin with a level-one Markdown title exactly like `# Descriptive
topic title`, followed by `## TL;DR`, `## Background`, three to six `##` thematic synthesis
sections, and `## Trends and open problems`; do not replace the last section with a conclusion.
Every bullet and non-obvious claim must be supported by an inline citation from the notes.
Remove speculative applications and any source whose title is not directly about the topic.
Topic relevance always beats source-family coverage; call fresh researchers if a relevant third
family is missing. If finalizer dropped a family, first read {SOURCES_PATH}, then add supported
body sentences citing a source from each of any three available source families and rerun the
finalizer; references alone do not preserve a source. Use execute to count distinct `source`
values after finalization. Run the validator, then satisfy the non-negotiable exit gate.
"""


def save_outputs(
    backend, topic, messages, elapsed, model_name, reports_dir=REPORTS, source_evidence=None
):
    """Download and persist a complete report/source/meta artifact set."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report = _decode_file(files, REPORT_PATH, "report.md")
    if not report.strip():
        raise RuntimeError("sandbox report.md is empty")
    source_text = _decode_file(files, SOURCES_PATH, "sources.json")
    try:
        sources = json.loads(source_text)
    except ValueError as exc:
        raise RuntimeError("sandbox sources.json is invalid JSON") from exc
    if not isinstance(sources, list):
        raise RuntimeError("sandbox sources.json must be a JSON list")
    validate_source_evidence(sources, source_evidence)
    validate_report_contract(report, sources, topic)

    families = sorted(
        {entry.get("source") for entry in sources if isinstance(entry, dict) and entry.get("source")}
    )
    metadata = summarize(messages, elapsed, model_name)
    metadata.update(
        {
            "topic": topic,
            "n_sources": len(sources),
            "source_families": families,
        }
    )

    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(topic)
    report_path = reports_dir / f"{stem}.md"
    sources_path = reports_dir / f"{stem}.sources.json"
    metadata_path = reports_dir / f"{stem}.meta.json"
    _write_atomically(sources_path, json.dumps(sources, ensure_ascii=False, indent=2) + "\n")
    _write_atomically(metadata_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    _write_atomically(report_path, report)
    return report_path


def _model_name(model):
    return getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__


def apply_model_limit(model):
    """Set a bounded per-call output budget without modifying ``model.py``.

    OpenAI-compatible gateways may otherwise choose a large completion default,
    which can reject a request before the agent can use its own call limits.
    Groq's free tier also has a small tokens-per-minute bucket.  Its requests
    are therefore serialized and given a smaller completion allowance.
    """
    raw_limit = (os.getenv("LAB_MAX_TOKENS") or "1200").strip()
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise RuntimeError("LAB_MAX_TOKENS must be a positive integer") from exc
    if limit < 1:
        raise RuntimeError("LAB_MAX_TOKENS must be a positive integer")
    if not hasattr(model, "max_tokens"):
        raise RuntimeError("configured model does not support a max_tokens output limit")
    base_url = (os.getenv("LAB_BASE_URL") or "").strip().lower()
    if base_url.startswith("https://api.groq.com/"):
        from langchain_core.rate_limiters import InMemoryRateLimiter

        limit = min(limit, 1000)
        model.rate_limiter = InMemoryRateLimiter(
            requests_per_second=1 / 60,
            check_every_n_seconds=0.5,
            max_bucket_size=1,
        )
    # Permit one recovery attempt for transient provider 5xx responses, without
    # multiplying a slow gateway failure across every agent turn.
    for client in (model, getattr(model, "root_client", None), getattr(model, "root_async_client", None)):
        if client is not None and hasattr(client, "max_retries"):
            client.max_retries = 1
    model.max_tokens = limit
    return model


def main(topic):
    """Run research and return 0 on success, 1 on failure, or 2 for missing input."""
    topic = str(topic or "").strip()
    if not topic:
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    try:
        model = apply_model_limit(make_model())
        model_name = _model_name(model)
        begin_source_evidence_capture()
        started = time.monotonic()
        with open_sandbox() as backend:
            setup = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if getattr(setup, "exit_code", 0) != 0:
                raise RuntimeError(f"could not initialize sandbox workspace: {setup.output}")
            upload(
                backend,
                {
                    VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                    FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
                },
            )
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000, "max_concurrency": 1},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else []
            source_evidence = collected_source_evidence()
            for repair_attempt in range(3):
                try:
                    reconcile_sandbox_source_evidence(backend, source_evidence)
                    report_path = save_outputs(
                        backend,
                        topic,
                        messages,
                        time.monotonic() - started,
                        model_name,
                        source_evidence=source_evidence,
                    )
                    break
                except RuntimeError as exc:
                    error = str(exc)
                    if "sources not returned by a host research tool" in error and source_evidence:
                        repair_prompt = provenance_repair_prompt(source_evidence)
                    elif error.startswith("report contract"):
                        repair_prompt = quality_repair_prompt(topic)
                    else:
                        raise RuntimeError(f"{exc}; {run_diagnostics(backend, messages)}") from exc
                    if repair_attempt == 2:
                        raise RuntimeError(f"{exc}; {run_diagnostics(backend, messages)}") from exc
                    repair = agent.invoke(
                        {"messages": [{"role": "user", "content": repair_prompt}]},
                        config={"recursion_limit": 150, "max_concurrency": 1},
                    )
                    repair_messages = repair.get("messages", []) if isinstance(repair, dict) else []
                    messages = [*messages, *repair_messages]
                    source_evidence = collected_source_evidence()
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Saved report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
