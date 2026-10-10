"""Host-side source tools used by the research subagents."""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlencode

import httpx
from langchain_core.tools import tool

ARXIV_URL = "https://export.arxiv.org/api/query"
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_ARXIV_NAMESPACE = "{http://www.w3.org/2005/Atom}"
_ARXIV_LOCK = threading.Lock()
_last_arxiv_call = 0.0
_EVIDENCE_LOCK = threading.Lock()
_SOURCE_EVIDENCE = set()


class RetryableError(Exception):
    """An upstream failure that may succeed after a bounded delay."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def begin_source_evidence_capture():
    """Clear the per-run URL provenance set before invoking a research agent."""
    with _EVIDENCE_LOCK:
        _SOURCE_EVIDENCE.clear()


def collected_source_evidence():
    """Return the source-family/URL pairs returned by successful host tools."""
    with _EVIDENCE_LOCK:
        return set(_SOURCE_EVIDENCE)


def _record_source_records(source, records):
    """Record only normalized URLs, never tool payloads or credentials."""
    pairs = {
        (source, str(record.get("url")).strip())
        for record in records
        if isinstance(record, dict) and str(record.get("url") or "").strip()
    }
    with _EVIDENCE_LOCK:
        _SOURCE_EVIDENCE.update(pairs)


def _record_source_url(source, url):
    value = str(url or "").strip()
    if value:
        with _EVIDENCE_LOCK:
            _SOURCE_EVIDENCE.add((source, value))


def _record_urls_in_text(source, text):
    """Record absolute URLs a successful text-returning tool actually exposed."""
    for url in re.findall(r"https?://[^\s<>()\[\]{}\"']+", str(text or "")):
        _record_source_url(source, url.rstrip(".,;:!?"))


def _compact(value, limit=None):
    text = " ".join(str(value or "").split())
    return text[:limit] if limit is not None else text


def _clamp(value, lower, upper, default):
    try:
        return max(lower, min(upper, int(value)))
    except (TypeError, ValueError):
        return default


def _retry_after(headers):
    value = headers.get("Retry-After") if headers else None
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call ``fn`` until it succeeds or its retry budget is exhausted."""
    attempts = max(1, int(attempts))
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(cap, max(0.0, float(exc.retry_after)))
            else:
                delay = min(cap, base * (2**attempt) + random.uniform(0.0, base))
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _request_once(method, url, *, params=None, json_body=None, headers=None, timeout=30.0):
    try:
        response = httpx.request(
            method,
            url,
            params=params,
            json=json_body,
            headers=headers,
            timeout=timeout,
        )
    except httpx.TransportError as exc:
        raise RetryableError(f"network error: {exc}") from exc
    if response.status_code in RETRYABLE_STATUS:
        raise RetryableError(
            f"HTTP {response.status_code}", retry_after=_retry_after(response.headers)
        )
    response.raise_for_status()
    return response


def _request(method, url, *, params=None, json_body=None, headers=None, attempts=5, cap=30.0):
    return with_retry(
        lambda: _request_once(
            method, url, params=params, json_body=json_body, headers=headers
        ),
        attempts=attempts,
        cap=cap,
    )


def _redact(message, secret=""):
    text = str(message)
    if secret:
        for variant in {secret, quote(secret, safe="")}:
            if variant:
                text = text.replace(variant, "[REDACTED]")
    return text


def _error(exc, secret=""):
    return f"ERROR: {type(exc).__name__}: {_redact(exc, secret)}"


def _wait_for_arxiv_slot():
    global _last_arxiv_call
    with _ARXIV_LOCK:
        now = time.monotonic()
        remaining = 3.0 - (now - _last_arxiv_call)
        if remaining > 0:
            time.sleep(remaining)
        _last_arxiv_call = time.monotonic()


def _arxiv_request(params):
    def fetch():
        _wait_for_arxiv_slot()
        return _request_once("GET", ARXIV_URL, params=params)

    return with_retry(fetch, attempts=6, cap=60.0)


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first.

    Returns compact JSON records with id, url, published, title, and summary.
    Use for primary paper abstracts; empty results return ``NO RESULTS``.
    """
    try:
        terms = re.findall(r"[\w-]+", str(query or ""), flags=re.UNICODE)
        if not terms:
            return "NO RESULTS"
        params = {
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": _clamp(max_results, 1, 30, 10),
        }
        root = ET.fromstring(_arxiv_request(params).text)
        records = []
        for entry in root.findall(f"{_ARXIV_NAMESPACE}entry"):
            raw_id = _compact(entry.findtext(f"{_ARXIV_NAMESPACE}id"))
            identifier = raw_id.rsplit("/abs/", 1)[-1]
            identifier = re.sub(r"v\d+$", "", identifier)
            if not identifier:
                continue
            records.append(
                {
                    "id": identifier,
                    "url": f"https://arxiv.org/abs/{identifier}",
                    "published": _compact(entry.findtext(f"{_ARXIV_NAMESPACE}published"))[:10],
                    "title": _compact(entry.findtext(f"{_ARXIV_NAMESPACE}title")),
                    "summary": _compact(entry.findtext(f"{_ARXIV_NAMESPACE}summary"), 600),
                }
            )
        if not records:
            return "NO RESULTS"
        _record_source_records("arxiv", records)
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _error(exc)


def _number(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _hf_records(payload, *, prefer_ai_summary=False):
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("papers", [])
    else:
        items = []
    records = []
    for item in items:
        if not isinstance(item, dict):
            continue
        paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
        identifier = paper.get("id")
        if not identifier:
            continue
        if prefer_ai_summary:
            summary = paper.get("ai_summary") or item.get("ai_summary") or paper.get("summary") or item.get("summary")
        else:
            summary = paper.get("summary") or item.get("summary")
        published = paper.get("publishedAt") or item.get("publishedAt") or ""
        records.append(
            {
                "id": str(identifier),
                "url": f"https://huggingface.co/papers/{identifier}",
                "published": _compact(published)[:10],
                "title": _compact(paper.get("title") or item.get("title")),
                "summary": _compact(summary, 600),
                "upvotes": _number(paper.get("upvotes") or item.get("upvotes")),
                "github": paper.get("githubRepo") or item.get("githubRepo") or "",
                "stars": _number(paper.get("githubStars") or item.get("githubStars")),
            }
        )
    return records


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Return trending Hugging Face Daily Papers as compact JSON records.

    ``keyword`` filters a daily list locally; use ``hf_search_papers`` for topic
    search. Results are sorted by upvotes.
    """
    try:
        params = {"limit": _clamp(limit, 1, 100, 30)}
        if str(date).strip():
            params["date"] = str(date).strip()
        records = _hf_records(_request("GET", HF_DAILY_URL, params=params).json())
        needle = str(keyword or "").strip().lower()
        if needle:
            records = [
                record
                for record in records
                if needle in f"{record['title']} {record['summary']}".lower()
            ]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        if not records:
            return "NO RESULTS"
        _record_source_records("hf-daily", records)
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic and return compact JSON records."""
    try:
        query = str(query or "").strip()
        if not query:
            return "NO RESULTS"
        records = _hf_records(
            _request(
                "GET",
                HF_SEARCH_URL,
                params={"q": query, "limit": _clamp(limit, 1, 50, 10)},
            ).json(),
            prefer_ai_summary=True,
        )
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        if not records:
            return "NO RESULTS"
        _record_source_records("hf-search", records)
        return json.dumps(records, ensure_ascii=False)
    except Exception as exc:
        return _error(exc)


def _rate_limited(meta):
    if not meta:
        return False
    text = json.dumps(meta, ensure_ascii=False).lower()
    return ("rate" in text and "limit" in text) or "too many requests" in text or "429" in text


def _parse_exa_response(text):
    """Extract the final JSON-RPC result from Exa's SSE response."""
    events = []
    for line in str(text).splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                events.append(json.loads(data))
    if not events:
        events = [json.loads(text)]
    result = None
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("error"):
            raise RuntimeError(f"Exa JSON-RPC error: {event['error']}")
        if "result" in event:
            result = event["result"]
    if not isinstance(result, dict):
        raise RuntimeError("Exa response did not contain a result")
    if _rate_limited(result.get("_meta")):
        raise RetryableError("Exa rate limit signalled in response metadata")
    return result


def _exa_call(name, arguments):
    secret = (os.getenv("EXA_API_KEY") or "").strip()
    endpoint = EXA_URL
    if secret:
        endpoint = f"{EXA_URL}?{urlencode({'exaApiKey': secret})}"
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    try:
        return with_retry(
            lambda: _parse_exa_response(
                _request_once("POST", endpoint, json_body=payload, headers=headers).text
            ),
            attempts=3,
            cap=15.0,
        )
    except Exception as exc:
        raise RuntimeError(_redact(exc, secret)) from exc


def _exa_text(result):
    blocks = result.get("content", []) if isinstance(result, dict) else []
    texts = [
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n\n".join(text.strip() for text in texts if text and text.strip())


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web through Exa and return the text of relevant result snippets."""
    secret = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        query = str(query or "").strip()
        if not query:
            return "NO RESULTS"
        result = _exa_call(
            "web_search_exa",
            {
                "query": query,
                "objective": str(objective or "").strip() or f"Find reliable sources about {query}",
                "numResults": _clamp(num_results, 1, 10, 5),
            },
        )
        text = _exa_text(result)
        _record_urls_in_text("web", text)
        return text if text else "NO RESULTS"
    except Exception as exc:
        return _error(exc, secret)


@tool
def web_fetch(url: str) -> str:
    """Fetch one URL through Exa and return up to 12,000 characters of text."""
    secret = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        url = str(url or "").strip()
        if not url:
            return "NO RESULTS"
        text = _exa_text(_exa_call("web_fetch_exa", {"urls": [url]}))
        if not text:
            return "NO RESULTS"
        _record_source_url("web", url)
        return text[:12000]
    except Exception as exc:
        return _error(exc, secret)


SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
