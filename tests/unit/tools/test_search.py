import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from research_assistant.config import Settings
from research_assistant.tools.search import (
    DdgsProvider,
    FallbackSearch,
    SearchError,
    SearchResult,
    SearxngProvider,
    build_search_provider,
    dedupe_results,
    traced_search,
)

FIXTURE = Path(__file__).parents[2] / "fixtures" / "searxng_python_asyncio.json"
SEARXNG = "http://searxng.test"


def searxng(handler: Any, sleeps: list[float] | None = None, retries: int = 2) -> SearxngProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    sleep = sleeps.append if sleeps is not None else (lambda _s: None)
    return SearxngProvider(SEARXNG, client, retries=retries, backoff_s=0.5, sleep=sleep)


def fixture_response(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=FIXTURE.read_bytes(), headers={"content-type": "application/json"})


class FakeProvider:
    def __init__(self, name: str, results: list[SearchResult] | None = None, error: bool = False) -> None:
        self.name = name
        self.results = results or []
        self.error = error
        self.calls = 0

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls += 1
        if self.error:
            raise SearchError(f"{self.name} down")
        return self.results[:max_results]


def result(url: str, rank: int, engine: str = "x") -> SearchResult:
    return SearchResult(url=url, title=f"t{rank}", snippet="s", engine=engine, rank=rank)


# -- SearXNG ---------------------------------------------------------------------------------


def test_searxng_parses_recorded_response() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return fixture_response(request)

    results = searxng(handler).search("python asyncio event loop", max_results=10)

    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))["results"]
    assert len(results) == 10
    assert [r.rank for r in results] == list(range(1, 11))
    assert results[0].url == raw[0]["url"] == "https://docs.python.org/3/library/asyncio-eventloop.html"
    assert results[0].title == raw[0]["title"]
    assert results[0].snippet == raw[0]["content"]
    assert results[0].engine == raw[0]["engine"]
    assert seen[0].url.path == "/search"
    assert seen[0].url.params["format"] == "json"
    assert seen[0].url.params["q"] == "python asyncio event loop"


def test_searxng_respects_max_results() -> None:
    assert len(searxng(fixture_response).search("q", max_results=3)) == 3


def test_searxng_drops_non_http_and_dedupes() -> None:
    body = {
        "results": [
            {"url": "https://a.com/x?utm_source=s", "title": "A", "content": "", "engine": "e"},
            {"url": "ftp://files.example.com/", "title": "F", "content": "", "engine": "e"},
            {"url": "HTTPS://A.com/x#top", "title": "A again", "content": "", "engine": "e"},
            {"url": "https://b.com/", "title": "B", "content": "", "engine": "e"},
            "not-a-dict",
        ]
    }
    results = searxng(lambda _r: httpx.Response(200, json=body)).search("q", 10)
    assert [(r.url, r.title, r.rank) for r in results] == [
        ("https://a.com/x", "A", 1),
        ("https://b.com/", "B", 2),
    ]


def test_searxng_retries_5xx_with_backoff() -> None:
    statuses = iter([503, 429, 200])
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return fixture_response(request) if status == 200 else httpx.Response(status)

    assert searxng(handler, sleeps).search("q", 5)
    assert sleeps == [0.5, 1.0]


def test_searxng_retries_network_errors_then_gives_up() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(SearchError, match="after 3 attempts"):
        searxng(handler).search("q", 5)
    assert calls == 3


def test_searxng_4xx_is_not_retried() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(403)

    with pytest.raises(SearchError, match="403"):
        searxng(handler).search("q", 5)
    assert calls == 1


@pytest.mark.parametrize("content", [b"<html>not json</html>", b'{"no_results": []}', b"[]"])
def test_searxng_bad_payload(content: bytes) -> None:
    with pytest.raises(SearchError):
        searxng(lambda _r: httpx.Response(200, content=content)).search("q", 5)


# -- dedupe ----------------------------------------------------------------------------------


def test_dedupe_keeps_best_rank_and_renumbers() -> None:
    results = [result("https://x.com/", 5), result("https://y.com/", 2), result("https://x.com/", 1)]
    deduped = dedupe_results(results)
    assert [(r.url, r.rank, r.title) for r in deduped] == [
        ("https://x.com/", 1, "t1"),
        ("https://y.com/", 2, "t2"),
    ]


# -- ddgs ------------------------------------------------------------------------------------


def test_ddgs_maps_fields() -> None:
    def fake(query: str, max_results: int) -> list[dict[str, Any]]:
        assert (query, max_results) == ("q", 5)
        return [
            {"href": "https://a.com/?fbclid=1", "title": "A", "body": "snippet a"},
            {"href": "not a url", "title": "bad", "body": ""},
            {"href": "https://b.com/p", "title": "B", "body": "snippet b"},
        ]

    results = DdgsProvider(fake).search("q", 5)
    assert [(r.url, r.title, r.snippet, r.engine, r.rank) for r in results] == [
        ("https://a.com/", "A", "snippet a", "ddgs", 1),
        ("https://b.com/p", "B", "snippet b", "ddgs", 2),
    ]


def test_ddgs_errors_become_search_error() -> None:
    def boom(_q: str, _n: int) -> list[dict[str, Any]]:
        raise RuntimeError("rate limited")

    with pytest.raises(SearchError, match="rate limited"):
        DdgsProvider(boom).search("q", 5)


# -- fallback --------------------------------------------------------------------------------


def test_fallback_not_used_when_primary_has_results() -> None:
    primary = FakeProvider("p", [result("https://a.com/", 1)])
    fallback = FakeProvider("f", [result("https://b.com/", 1)])
    assert FallbackSearch(primary, fallback).search("q", 5)[0].url == "https://a.com/"
    assert fallback.calls == 0


@pytest.mark.parametrize(
    "primary", [FakeProvider("p"), FakeProvider("p", error=True)], ids=["empty", "error"]
)
def test_fallback_used_when_primary_empty_or_fails(primary: FakeProvider) -> None:
    fallback = FakeProvider("f", [result("https://b.com/", 1)])
    assert FallbackSearch(primary, fallback).search("q", 5)[0].url == "https://b.com/"
    assert fallback.calls == 1


def test_fallback_disabled_returns_empty_without_calling_ddgs() -> None:
    fallback = FakeProvider("f", [result("https://b.com/", 1)])
    assert FallbackSearch(FakeProvider("p"), fallback, enabled=False).search("q", 5) == []
    assert fallback.calls == 0


def test_fallback_disabled_reraises_primary_error() -> None:
    fallback = FakeProvider("f", [result("https://b.com/", 1)])
    with pytest.raises(SearchError, match="p down"):
        FallbackSearch(FakeProvider("p", error=True), fallback, enabled=False).search("q", 5)
    assert fallback.calls == 0


def test_fallback_error_propagates() -> None:
    with pytest.raises(SearchError, match="f down"):
        FallbackSearch(FakeProvider("p", error=True), FakeProvider("f", error=True)).search("q", 5)


@pytest.mark.parametrize("enabled", [True, False])
def test_build_search_provider_uses_config_flag(enabled: bool) -> None:
    settings = Settings(_env_file=None, search_fallback_enabled=enabled)
    with httpx.Client(transport=httpx.MockTransport(fixture_response)) as client:
        provider = build_search_provider(settings, client)
    assert provider.enabled is enabled
    assert isinstance(provider.primary, SearxngProvider)
    assert isinstance(provider.fallback, DdgsProvider)


# -- traced search ---------------------------------------------------------------------------


class _Scripted:
    def __init__(self, name: str, outcome: list[SearchResult] | SearchError) -> None:
        self.name = name
        self.outcome = outcome
        self.calls = 0

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls += 1
        if isinstance(self.outcome, SearchError):
            raise self.outcome
        return list(self.outcome)


def _hit(url: str = "https://a.example.org/") -> SearchResult:
    return SearchResult(url=url, title="t", snippet="s", engine="e", rank=1)


def test_traced_search_records_primary_and_fallback_calls() -> None:
    clock = iter([0.0, 0.4, 1.0, 1.25])
    empty, ddgs = _Scripted("searxng", []), _Scripted("ddgs", [_hit()])
    trace = traced_search(FallbackSearch(empty, ddgs), "q", 5, clock=lambda: next(clock))
    assert [(c.provider, c.role, c.result_count, c.duration_s, c.error) for c in trace.calls] == [
        ("searxng", "primary", 0, 0.4, None),
        ("ddgs", "fallback", 1, 0.25, None),
    ]
    assert trace.results == [_hit()] and trace.error is None


def test_traced_search_primary_hit_skips_fallback() -> None:
    primary, ddgs = _Scripted("searxng", [_hit()]), _Scripted("ddgs", [])
    trace = traced_search(FallbackSearch(primary, ddgs), "q", 5)
    assert [c.role for c in trace.calls] == ["primary"] and ddgs.calls == 0


def test_traced_search_both_fail_keeps_both_calls_and_the_fallback_error() -> None:
    primary = _Scripted("searxng", SearchError("searxng down"))
    ddgs = _Scripted("ddgs", SearchError("ddgs failed: no results"))
    trace = traced_search(FallbackSearch(primary, ddgs), "q", 5)
    assert [(c.provider, c.error) for c in trace.calls] == [
        ("searxng", "searxng down"),
        ("ddgs", "ddgs failed: no results"),
    ]
    assert str(trace.error) == "ddgs failed: no results" and trace.results == []
    with pytest.raises(SearchError, match="ddgs failed"):
        FallbackSearch(primary, ddgs).search("q", 5)  # search() still raises, as before


def test_traced_search_disabled_fallback_and_plain_provider() -> None:
    empty, ddgs = _Scripted("searxng", []), _Scripted("ddgs", [_hit()])
    assert [c.role for c in traced_search(FallbackSearch(empty, ddgs, enabled=False), "q", 5).calls] == [
        "primary"
    ]
    assert [c.role for c in traced_search(_Scripted("fake", [_hit()]), "q", 5).calls] == ["only"]
