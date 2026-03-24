"""
Web search module for grounding LLM responses with real-world knowledge.

Supports Brave Search (Korean queries) and Tavily Search (English queries)
running in parallel. Uses only Python stdlib — no external packages.
"""

import gzip
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def brave_search(query: str, api_key: str | None = None, count: int = 5) -> list[dict]:
    """Search via Brave Search API. Returns list of {title, url, snippet}."""
    api_key = api_key or os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return []

    try:
        params = urllib.parse.urlencode({
            "q": query,
            "count": count,
            "extra_snippets": "true",
        })
        url = f"https://api.search.brave.com/res/v1/web/search?{params}"

        req = urllib.request.Request(url, headers={
            "X-Subscription-Token": api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        })

        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            data = json.loads(raw)

        results = []
        for item in data.get("web", {}).get("results", []):
            snippets = [item.get("description", "")]
            snippets.extend(item.get("extra_snippets", []))
            snippet = " ".join(s for s in snippets if s)
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": snippet,
            })
        return results

    except Exception:
        return []


def tavily_search(query: str, api_key: str | None = None, max_results: int = 5) -> list[dict]:
    """Search via Tavily API. Returns list of {title, url, snippet, score}."""
    api_key = api_key or os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []

    try:
        payload = json.dumps({
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        }).encode()

        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())

        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "score": item.get("score", 0),
            })
        return results

    except Exception:
        return []


def search_both(ko_query: str, en_query: str) -> tuple[list[dict], list[dict]]:
    """Run Brave (Korean) and Tavily (English) searches in parallel."""
    with ThreadPoolExecutor(max_workers=2) as pool:
        ko_future = pool.submit(brave_search, ko_query)
        en_future = pool.submit(tavily_search, en_query)
        return ko_future.result(), en_future.result()


def format_search_context(ko_results: list[dict], en_results: list[dict]) -> str:
    """Format search results for injection into an LLM prompt."""
    if not ko_results and not en_results:
        return ""

    lines = []

    if ko_results:
        lines.append("Korean sources:")
        for i, r in enumerate(ko_results, 1):
            lines.append(f"  {i}. [{r['title']}] {r['snippet']}")

    if ko_results and en_results:
        lines.append("")

    if en_results:
        lines.append("English sources:")
        for i, r in enumerate(en_results, 1):
            lines.append(f"  {i}. [{r['title']}] {r['snippet']}")

    return "\n".join(lines)
