"""Internet fallback used when the knowledge base cannot support an answer.

Provider selection (WEB_SEARCH_PROVIDER):
  auto       -> Tavily if TAVILY_API_KEY is set, else DuckDuckGo
  tavily     -> Tavily (free tier key from https://app.tavily.com), falls back to DuckDuckGo on error
  duckduckgo -> DuckDuckGo via the `ddgs` package, no key
  none       -> disabled
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from .config import settings

log = logging.getLogger(__name__)


@dataclass
class WebResult:
    title: str
    url: str
    content: str


def provider_name() -> str:
    p = settings.web_search_provider
    if p == "auto":
        return "tavily" if settings.tavily_api_key else "duckduckgo"
    return p


def web_enabled() -> bool:
    return provider_name() in {"tavily", "duckduckgo"}


def _tavily(query: str, n: int) -> list[WebResult]:
    resp = httpx.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        json={"query": query, "max_results": n, "search_depth": "basic", "include_answer": False},
        timeout=25,
    )
    resp.raise_for_status()
    return [
        WebResult(title=r.get("title") or r.get("url", ""), url=r.get("url", ""), content=r.get("content") or "")
        for r in resp.json().get("results", [])
        if r.get("url")
    ]


def _duckduckgo(query: str, n: int) -> list[WebResult]:
    from ddgs import DDGS  # imported lazily: optional at runtime

    rows = DDGS().text(query, max_results=n) or []
    return [
        WebResult(title=r.get("title", ""), url=r.get("href") or r.get("url", ""), content=r.get("body", ""))
        for r in rows
        if r.get("href") or r.get("url")
    ]


def search_web(query: str, n: int | None = None) -> tuple[list[WebResult], str | None]:
    """Returns (results, error_message). Never raises."""
    n = n or settings.web_results
    provider = provider_name()
    if provider not in {"tavily", "duckduckgo"}:
        return [], "web search disabled"
    errors: list[str] = []
    order = ["tavily", "duckduckgo"] if provider == "tavily" else ["duckduckgo"]
    for name in order:
        try:
            results = _tavily(query, n) if name == "tavily" else _duckduckgo(query, n)
            results = [r for r in results if r.content.strip()]
            if results:
                return results, None
            errors.append(f"{name}: no results")
        except Exception as exc:  # network errors, rate limits, package issues
            log.warning("web search via %s failed: %s", name, exc)
            errors.append(f"{name}: {type(exc).__name__}")
    return [], "; ".join(errors)
