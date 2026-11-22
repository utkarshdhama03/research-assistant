"""Web search behind a small provider interface. SearXNG first; ddgs only as a fallback."""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import httpx

from research_assistant.config import Settings
from research_assistant.tools.url import normalize_url

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    url: str  # normalized
    title: str
    snippet: str
    engine: str
    rank: int  # 1 = best


class SearchError(Exception):
    """A provider failed (network, HTTP status, bad payload)."""


class SearchProvider(Protocol):
    name: str

    def search(self, query: str, max_results: int) -> list[SearchResult]: ...


def dedupe_results(results: Sequence[SearchResult]) -> list[SearchResult]:
    """Keep one result per normalized URL (the best-ranked one), then renumber ranks from 1."""
    best: dict[str, SearchResult] = {}
    for r in results:
        current = best.get(r.url)
        if current is None or r.rank < current.rank:
            best[r.url] = r
    ordered = sorted(best.values(), key=lambda r: r.rank)
    return [replace(r, rank=i) for i, r in enumerate(ordered, start=1)]


def _to_results(items: Sequence[tuple[Any, Any, Any, str]]) -> list[SearchResult]:
    """Build ranked results from (url, title, snippet, engine), dropping non-http(s) URLs."""
    out: list[SearchResult] = []
    for raw_url, title, snippet, engine in items:
        try:
            url = normalize_url(str(raw_url))
        except ValueError:
            continue
        out.append(
            SearchResult(
                url=url, title=str(title or ""), snippet=str(snippet or ""), engine=engine, rank=len(out) + 1
            )
        )
    return out


class SearxngProvider:
    """SearXNG JSON API with a timeout and bounded retries (network errors, 5xx, 429)."""

    name = "searxng"

    def __init__(
        self,
        base_url: str,
        client: httpx.Client,
        *,
        retries: int = 2,
        backoff_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._url = base_url.rstrip("/") + "/search"
        self._client = client
        self._retries = retries
        self._backoff_s = backoff_s
        self._sleep = sleep

    def _get(self, query: str) -> httpx.Response:
        last_error = ""
        for attempt in range(self._retries + 1):
            if attempt:
                self._sleep(self._backoff_s * 2 ** (attempt - 1))
            try:
                resp = self._client.get(self._url, params={"q": query, "format": "json"})
            except httpx.TransportError as e:
                last_error = f"{type(e).__name__}: {e}"
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                continue
            if resp.status_code != 200:
                raise SearchError(f"searxng returned HTTP {resp.status_code}")
            return resp
        raise SearchError(f"searxng failed after {self._retries + 1} attempts: {last_error}")

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        resp = self._get(query)
        try:
            data = resp.json()
        except ValueError as e:
            raise SearchError("searxng returned invalid JSON") from e
        items = data.get("results") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise SearchError("searxng response has no results list")
        rows = [
            (i.get("url"), i.get("title"), i.get("content"), str(i.get("engine") or self.name))
            for i in items
            if isinstance(i, dict)
        ]
        return dedupe_results(_to_results(rows))[:max_results]


DdgsSearchFn = Callable[[str, int], list[dict[str, Any]]]


def _ddgs_search_fn(timeout_s: float) -> DdgsSearchFn:
    def run(query: str, max_results: int) -> list[dict[str, Any]]:
        from ddgs import DDGS

        return DDGS(timeout=max(1, int(timeout_s))).text(query, max_results=max_results)

    return run


class DdgsProvider:
    """Fallback search through the ddgs package.

    ddgs does its own HTTP (via primp), not through httpx, so tests inject `search_fn`.
    """

    name = "ddgs"

    def __init__(self, search_fn: DdgsSearchFn | None = None, *, timeout_s: float = 15.0) -> None:
        self._search_fn = search_fn or _ddgs_search_fn(timeout_s)

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            items = self._search_fn(query, max_results)
        except Exception as e:  # any failure inside the third-party library becomes a SearchError
            raise SearchError(f"ddgs failed: {type(e).__name__}: {e}") from e
        rows = [
            (i.get("href"), i.get("title"), i.get("body"), self.name) for i in items if isinstance(i, dict)
        ]
        return dedupe_results(_to_results(rows))[:max_results]


class FallbackSearch:
    """Use `fallback` only when `primary` raises SearchError or returns no results, and only if enabled."""

    name = "fallback"

    def __init__(self, primary: SearchProvider, fallback: SearchProvider, *, enabled: bool = True) -> None:
        self.primary = primary
        self.fallback = fallback
        self.enabled = enabled

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        trace = traced_search(self, query, max_results)
        if trace.error is not None:
            raise trace.error
        return trace.results


@dataclass(frozen=True, slots=True)
class SearchCall:
    """One provider call: logged as a search_done event."""

    query: str
    provider: str
    role: str  # "primary", "fallback" or "only"
    result_count: int
    duration_s: float
    error: str | None


@dataclass(frozen=True, slots=True)
class SearchTrace:
    results: list[SearchResult]
    calls: list[SearchCall]
    error: SearchError | None  # set when the search as a whole failed


def _timed(
    provider: SearchProvider, query: str, max_results: int, role: str, clock: Callable[[], float]
) -> tuple[list[SearchResult], SearchCall, SearchError | None]:
    started = clock()
    try:
        results = provider.search(query, max_results)
        error: SearchError | None = None
    except SearchError as e:
        results, error = [], e
    call = SearchCall(
        query, provider.name, role, len(results), round(clock() - started, 3), str(error) if error else None
    )
    return results, call, error


def traced_search(
    provider: SearchProvider, query: str, max_results: int, clock: Callable[[], float] = time.monotonic
) -> SearchTrace:
    """Search and record every provider call, the fallback included. Never raises SearchError: a failed
    search comes back with `error` set (the fallback's error if the fallback ran)."""
    if not isinstance(provider, FallbackSearch):
        results, call, error = _timed(provider, query, max_results, "only", clock)
        return SearchTrace(results, [call], error)
    results, call, error = _timed(provider.primary, query, max_results, "primary", clock)
    calls = [call]
    if not provider.enabled or (error is None and results):
        return SearchTrace(results, calls, error)
    if error is not None:
        log.warning(
            "%s failed (%s); falling back to %s", provider.primary.name, error, provider.fallback.name
        )
    else:
        log.warning(
            "%s returned no results; falling back to %s", provider.primary.name, provider.fallback.name
        )
    results, call, error = _timed(provider.fallback, query, max_results, "fallback", clock)
    calls.append(call)
    return SearchTrace(results, calls, error)


def build_search_provider(settings: Settings, client: httpx.Client) -> FallbackSearch:
    """SearXNG with the ddgs fallback, as configured. The caller owns `client`."""
    primary = SearxngProvider(
        settings.searxng_url, client, retries=settings.search_retries, backoff_s=settings.search_backoff_s
    )
    return FallbackSearch(
        primary, DdgsProvider(timeout_s=settings.search_timeout_s), enabled=settings.search_fallback_enabled
    )
