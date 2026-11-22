"""Fetcher tests. No network: httpx.MockTransport serves responses, DNS is a dict, time is fake."""

import gzip
import socket
from collections.abc import Callable, Iterator

import httpx
import pytest

from research_assistant.config import Settings
from research_assistant.tools.fetch import Fetcher, FetchError, FetchErrorReason

CONTACT = "https://contact.test/research-assistant"
UA = f"ResearchAssistant/0.1 (+{CONTACT})"
HTML = {"content-type": "text/html; charset=utf-8"}
PAGE = b"<html><body><p>hello</p></body></html>"

# Public names map to real global addresses; the resolver never touches the network.
DNS: dict[str, list[str]] = {
    "public.test": ["93.184.215.14"],
    "other.test": ["1.1.1.1"],
    "v6.test": ["2606:4700::1111"],
    "localhost": ["127.0.0.1", "::1"],
    "internal.test": ["192.168.0.10"],
    "mixed.test": ["93.184.215.14", "10.0.0.1"],
    "nat64.test": ["64:ff9b::a00:1"],
    "metadata.test": ["169.254.169.254"],
}

Factory = Callable[[httpx.Request], httpx.Response]


class Site:
    """Routes requests by host + path. Unknown robots.txt → 404 (allow all); unknown page → 404."""

    def __init__(self, routes: dict[str, Factory] | None = None) -> None:
        self.routes = routes or {}
        self.log: list[str] = []
        self.user_agents: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        key = request.url.host + request.url.raw_path.decode()
        self.log.append(key)
        self.user_agents.append(request.headers.get("user-agent", ""))
        factory = self.routes.get(key)
        return factory(request) if factory else httpx.Response(404)

    def count(self, key: str) -> int:
        return self.log.count(key)

    def pages(self) -> list[str]:
        return [k for k in self.log if not k.endswith("/robots.txt")]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def resolver(host: str, port: int) -> list[str]:
    if host not in DNS:
        raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")
    return DNS[host]


def make_fetcher(site: Site, clock: FakeClock | None = None, **overrides: object) -> Fetcher:
    values: dict[str, object] = {
        "contact_url": CONTACT,
        "fetch_timeout_s": 20.0,
        "fetch_host_min_delay_s": 1.0,
        "max_fetch_bytes": 1000,
        "robots_max_bytes": 512_000,
    }
    values.update(overrides)
    settings = Settings(_env_file=None, **values)  # type: ignore[arg-type]
    clock = clock or FakeClock()
    client = httpx.Client(transport=httpx.MockTransport(site), trust_env=False)
    return Fetcher(settings, client=client, resolver=resolver, clock=clock, sleep=clock.sleep)


def ok(body: bytes = PAGE, headers: dict[str, str] | None = None) -> Factory:
    return lambda _r: httpx.Response(200, content=body, headers=HTML if headers is None else headers)


def redirect(location: str, status: int = 302) -> Factory:
    return lambda _r: httpx.Response(status, headers={"location": location})


def robots(text: str) -> Factory:
    return lambda _r: httpx.Response(200, text=text, headers={"content-type": "text/plain"})


def reason_of(fetcher: Fetcher, url: str) -> FetchErrorReason:
    with pytest.raises(FetchError) as info:
        fetcher.fetch(url)
    return info.value.reason


# -- success ---------------------------------------------------------------------------------


def test_fetch_success() -> None:
    site = Site({"public.test/page": ok()})
    result = make_fetcher(site).fetch("http://public.test/page")
    assert result.url == result.final_url == "http://public.test/page"
    assert result.status_code == 200
    assert result.content_type == "text/html; charset=utf-8"
    assert result.body == PAGE
    assert site.log == ["public.test/robots.txt", "public.test/page"]
    assert set(site.user_agents) == {UA}


def test_fetch_ipv6_public_host() -> None:
    site = Site({"v6.test/": ok()})
    assert make_fetcher(site).fetch("https://v6.test/").status_code == 200


# -- SSRF: rejected before any request is sent ------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.1.2.3/",
        "http://172.16.0.1/",
        "http://172.31.255.254/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.test/latest/meta-data/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[::ffff:7f00:1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://[64:ff9b::7f00:1]/",  # NAT64 of 127.0.0.1
        "http://[::7f00:1]/",  # IPv4-compatible 127.0.0.1
        "http://[2002:7f00:1::1]/",  # 6to4 of 127.0.0.1
        "http://[fe80::1]/",
        "http://[fc00::1]/",
        "http://0.0.0.0/",
        "http://100.64.0.1/",
        "http://224.0.0.1/",
        "http://internal.test/",
        "http://mixed.test/",  # one private address among public ones
        "http://nat64.test/",
    ],
)
def test_blocked_ip(url: str) -> None:
    site = Site()
    assert reason_of(make_fetcher(site), url) is FetchErrorReason.BLOCKED_IP
    assert site.log == []


@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/",  # decimal 127.0.0.1
        "http://0x7f000001/",
        "http://0x7f.1/",
        "http://0177.0.0.1/",
        "http://127.1/",
        "http://user@public.test/",
        "http://user:pass@public.test/",
        "ftp://public.test/file",
        "file:///etc/passwd",
        "gopher://public.test/",
        "http://public.test:99999/",
        "not a url",
    ],
)
def test_invalid_url(url: str) -> None:
    site = Site()
    assert reason_of(make_fetcher(site), url) is FetchErrorReason.INVALID_URL
    assert site.log == []


def test_dns_failure_is_network_error() -> None:
    assert reason_of(make_fetcher(Site()), "http://nonexistent.test/") is FetchErrorReason.NETWORK_ERROR


# -- redirects -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "location",
    [
        "http://10.0.0.1/admin",
        "http://internal.test/",
        "http://[::ffff:169.254.169.254]/latest/meta-data/",
        "http://localhost:8080/",
    ],
)
def test_redirect_from_public_to_private_is_blocked(location: str) -> None:
    site = Site({"public.test/go": redirect(location)})
    assert reason_of(make_fetcher(site), "http://public.test/go") is FetchErrorReason.BLOCKED_IP
    assert site.pages() == ["public.test/go"]  # the private target was never requested


def test_redirect_to_other_scheme_is_invalid() -> None:
    site = Site({"public.test/go": redirect("ftp://public.test/file")})
    assert reason_of(make_fetcher(site), "http://public.test/go") is FetchErrorReason.INVALID_URL


def test_redirect_loop_stops_after_five_hops() -> None:
    site = Site({"public.test/a": redirect("/b"), "public.test/b": redirect("/a")})
    assert reason_of(make_fetcher(site), "http://public.test/a") is FetchErrorReason.TOO_MANY_REDIRECTS
    assert len(site.pages()) == 6  # the first request plus five followed redirects


def test_five_redirects_are_followed() -> None:
    routes: dict[str, Factory] = {f"public.test/r{i}": redirect(f"/r{i + 1}", 301 + i % 3) for i in range(5)}
    routes["public.test/r5"] = ok()
    site = Site(routes)
    result = make_fetcher(site).fetch("http://public.test/r0")
    assert result.url == "http://public.test/r0"
    assert result.final_url == "http://public.test/r5"
    assert len(site.pages()) == 6


def test_cross_host_redirect_checks_target_robots() -> None:
    site = Site(
        {
            "public.test/go": redirect("https://other.test/doc"),
            "other.test/robots.txt": robots("User-agent: *\nDisallow: /doc\n"),
        }
    )
    assert reason_of(make_fetcher(site), "http://public.test/go") is FetchErrorReason.ROBOTS_DISALLOWED
    assert "other.test/doc" not in site.log


def test_redirect_without_location_is_http_error() -> None:
    site = Site({"public.test/x": lambda _r: httpx.Response(302)})
    with pytest.raises(FetchError) as info:
        make_fetcher(site).fetch("http://public.test/x")
    assert (info.value.reason, info.value.status_code) == (FetchErrorReason.HTTP_ERROR, 302)


# -- status, content type, size, time --------------------------------------------------------


@pytest.mark.parametrize("status", [404, 410, 500, 503])
def test_http_error(status: int) -> None:
    site = Site({"public.test/x": lambda _r: httpx.Response(status)})
    with pytest.raises(FetchError) as info:
        make_fetcher(site).fetch("http://public.test/x")
    assert (info.value.reason, info.value.status_code) == (FetchErrorReason.HTTP_ERROR, status)


@pytest.mark.parametrize(
    "content_type",
    ["text/html", "Text/HTML; charset=UTF-8", "application/xhtml+xml", "text/plain", "application/pdf"],
)
def test_allowed_content_types(content_type: str) -> None:
    site = Site({"public.test/x": ok(headers={"content-type": content_type})})
    assert make_fetcher(site).fetch("http://public.test/x").content_type == content_type


@pytest.mark.parametrize("content_type", ["image/png", "application/json", "application/octet-stream", None])
def test_bad_content_type(content_type: str | None) -> None:
    headers = {} if content_type is None else {"content-type": content_type}
    site = Site({"public.test/x": ok(headers=headers)})
    assert reason_of(make_fetcher(site), "http://public.test/x") is FetchErrorReason.BAD_CONTENT_TYPE


def chunks(
    n: int, size: int, counter: list[int], on_chunk: Callable[[], None] | None = None
) -> Iterator[bytes]:
    for _ in range(n):
        counter.append(1)
        if on_chunk:
            on_chunk()
        yield b"x" * size


def test_too_large_by_content_length_reads_nothing() -> None:
    pulled: list[int] = []
    site = Site(
        {
            "public.test/x": lambda _r: httpx.Response(
                200, headers={**HTML, "content-length": "5000"}, content=chunks(25, 200, pulled)
            )
        }
    )
    assert reason_of(make_fetcher(site), "http://public.test/x") is FetchErrorReason.TOO_LARGE
    assert pulled == []


def test_too_large_while_streaming_stops_early() -> None:
    pulled: list[int] = []
    site = Site(
        {"public.test/x": lambda _r: httpx.Response(200, headers=HTML, content=chunks(50, 200, pulled))}
    )
    assert reason_of(make_fetcher(site), "http://public.test/x") is FetchErrorReason.TOO_LARGE
    assert len(pulled) == 6  # 1200 bytes > 1000, then stop


def test_body_exactly_at_cap_is_accepted() -> None:
    site = Site({"public.test/x": ok(b"x" * 1000)})
    assert len(make_fetcher(site).fetch("http://public.test/x").body) == 1000


def test_gzip_bomb_is_capped_on_decompressed_size() -> None:
    bomb = gzip.compress(b"\0" * 1_000_000)
    assert len(bomb) < 10_000
    site = Site({"public.test/x": ok(bomb, headers={**HTML, "content-encoding": "gzip"})})
    fetcher = make_fetcher(site, max_fetch_bytes=10_000)
    assert reason_of(fetcher, "http://public.test/x") is FetchErrorReason.TOO_LARGE


def test_transport_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    site = Site({"public.test/x": handler})
    assert reason_of(make_fetcher(site), "http://public.test/x") is FetchErrorReason.TIMEOUT


def test_total_deadline_while_streaming() -> None:
    clock = FakeClock()
    pulled: list[int] = []

    def advance() -> None:
        clock.now += 15

    site = Site(
        {
            "public.test/x": lambda _r: httpx.Response(
                200, headers=HTML, content=chunks(5, 10, pulled, advance)
            )
        }
    )
    fetcher = make_fetcher(site, clock, fetch_timeout_s=20.0)
    assert reason_of(fetcher, "http://public.test/x") is FetchErrorReason.TIMEOUT
    assert len(pulled) == 2


def test_connect_error_is_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    site = Site({"public.test/x": handler})
    assert reason_of(make_fetcher(site), "http://public.test/x") is FetchErrorReason.NETWORK_ERROR


def test_client_that_trusts_env_proxies_is_rejected() -> None:
    client = httpx.Client(transport=httpx.MockTransport(Site()), trust_env=True)
    with pytest.raises(ValueError, match="trust_env"):
        Fetcher(Settings(_env_file=None), client=client, resolver=resolver)


# -- robots.txt ------------------------------------------------------------------------------


def test_robots_disallow_and_cache() -> None:
    site = Site(
        {
            "public.test/robots.txt": robots("User-agent: *\nDisallow: /private\n"),
            "public.test/public": ok(),
        }
    )
    fetcher = make_fetcher(site)
    assert reason_of(fetcher, "http://public.test/private/x") is FetchErrorReason.ROBOTS_DISALLOWED
    assert fetcher.fetch("http://public.test/public").status_code == 200
    assert site.count("public.test/robots.txt") == 1
    assert "public.test/private/x" not in site.log


def test_robots_rule_for_our_user_agent() -> None:
    text = "User-agent: ResearchAssistant\nDisallow: /\n\nUser-agent: *\nAllow: /\n"
    site = Site({"public.test/robots.txt": robots(text), "public.test/x": ok()})
    assert reason_of(make_fetcher(site), "http://public.test/x") is FetchErrorReason.ROBOTS_DISALLOWED


def test_robots_rule_for_other_agent_does_not_apply() -> None:
    site = Site(
        {"public.test/robots.txt": robots("User-agent: OtherBot\nDisallow: /\n"), "public.test/x": ok()}
    )
    assert make_fetcher(site).fetch("http://public.test/x").status_code == 200


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_robots_4xx_allows_and_is_cached(status: int) -> None:
    site = Site({"public.test/robots.txt": lambda _r: httpx.Response(status), "public.test/x": ok()})
    fetcher = make_fetcher(site)
    fetcher.fetch("http://public.test/x")
    fetcher.fetch("http://public.test/x")
    assert site.count("public.test/robots.txt") == 1
    assert site.count("public.test/x") == 2


def _robots_unavailable(kind: str) -> Factory:
    def handler(request: httpx.Request) -> httpx.Response:
        if kind == "5xx":
            return httpx.Response(503)
        if kind == "timeout":
            raise httpx.ConnectTimeout("slow", request=request)
        raise httpx.ConnectError("refused", request=request)

    return handler


@pytest.mark.parametrize("kind", ["5xx", "connect_error", "timeout"])
def test_robots_unavailable_retries_once_disallows_and_is_not_cached(kind: str) -> None:
    site = Site({"public.test/robots.txt": _robots_unavailable(kind), "public.test/x": ok()})
    fetcher = make_fetcher(site)
    assert reason_of(fetcher, "http://public.test/x") is FetchErrorReason.ROBOTS_DISALLOWED
    assert site.count("public.test/robots.txt") == 2
    assert reason_of(fetcher, "http://public.test/x") is FetchErrorReason.ROBOTS_DISALLOWED
    assert site.count("public.test/robots.txt") == 4  # not cached: tried again
    assert site.pages() == []


def test_robots_recovers_on_retry_and_is_cached() -> None:
    answers = iter([httpx.Response(500), httpx.Response(200, text="User-agent: *\nAllow: /\n")])
    site = Site({"public.test/robots.txt": lambda _r: next(answers), "public.test/x": ok()})
    fetcher = make_fetcher(site)
    fetcher.fetch("http://public.test/x")
    fetcher.fetch("http://public.test/x")
    assert site.count("public.test/robots.txt") == 2
    assert site.count("public.test/x") == 2


def test_robots_redirect_to_private_ip_disallows_without_retry() -> None:
    site = Site({"public.test/robots.txt": redirect("http://10.0.0.1/robots.txt"), "public.test/x": ok()})
    fetcher = make_fetcher(site)
    assert reason_of(fetcher, "http://public.test/x") is FetchErrorReason.ROBOTS_DISALLOWED
    assert site.count("public.test/robots.txt") == 1
    assert site.pages() == []


def test_robots_over_size_cap_is_truncated_not_rejected() -> None:
    text = "User-agent: *\nDisallow: /private\n" + "# padding\n" * 200
    site = Site({"public.test/robots.txt": robots(text), "public.test/x": ok()})
    fetcher = make_fetcher(site, robots_max_bytes=100)
    assert fetcher.fetch("http://public.test/x").status_code == 200
    assert reason_of(fetcher, "http://public.test/private") is FetchErrorReason.ROBOTS_DISALLOWED


# -- per-host delay --------------------------------------------------------------------------


def test_per_host_min_delay() -> None:
    clock = FakeClock()
    site = Site({"public.test/a": ok(), "public.test/b": ok()})
    fetcher = make_fetcher(site, clock)
    fetcher.fetch("http://public.test/a")  # robots.txt at t=0, page after 1 s
    fetcher.fetch("http://public.test/b")
    assert clock.sleeps == [1.0, 1.0]


def test_no_wait_when_delay_already_elapsed_or_other_host() -> None:
    clock = FakeClock()
    site = Site({"public.test/a": ok(), "other.test/a": ok()})
    fetcher = make_fetcher(site, clock, fetch_host_min_delay_s=0.0)
    fetcher.fetch("http://public.test/a")
    fetcher.fetch("http://other.test/a")
    fetcher.fetch("http://public.test/a")
    assert clock.sleeps == []


@pytest.mark.parametrize(("crawl_delay", "expected"), [(3, 3.0), (60, 10.0)])
def test_crawl_delay_is_honored_and_capped(crawl_delay: int, expected: float) -> None:
    clock = FakeClock()
    site = Site(
        {
            "public.test/robots.txt": robots(f"User-agent: *\nCrawl-delay: {crawl_delay}\n"),
            "public.test/x": ok(),
        }
    )
    make_fetcher(site, clock).fetch("http://public.test/x")
    assert clock.sleeps == [expected]


def test_delay_beyond_deadline_is_timeout() -> None:
    site = Site({"public.test/robots.txt": robots("User-agent: *\nCrawl-delay: 10\n"), "public.test/x": ok()})
    fetcher = make_fetcher(site, fetch_timeout_s=5.0)
    assert reason_of(fetcher, "http://public.test/x") is FetchErrorReason.TIMEOUT
