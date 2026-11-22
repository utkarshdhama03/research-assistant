"""SSRF-safe HTTP fetcher on httpx.

Every hop (the first request and each redirect) is checked before it is sent:
- http/https only, no userinfo, no non-canonical numeric hosts (2130706433, 0x7f.1, 127.1);
- the host is resolved and EVERY address must be globally routable, including IPv4 addresses
  embedded in IPv6 (IPv4-mapped, IPv4-compatible, NAT64 64:ff9b::/96, 6to4);
- robots.txt allows the URL for our User-Agent, and the per-host minimum delay has passed.

Known gap (accepted, see the decision log): httpx resolves the host again when it connects, so a
DNS-rebinding server could answer our check with a public IP and httpx with a private one.
"""

import ipaddress
import re
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Self
from urllib.robotparser import RobotFileParser

import httpx

from research_assistant.config import Settings, get_settings

MAX_REDIRECTS = 5
MAX_CRAWL_DELAY_S = 10.0
ALLOWED_CONTENT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain", "application/pdf"})
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
DEFAULT_PORTS = {"http": 80, "https": 443}

_NUMERIC_LABEL = re.compile(r"0x[0-9a-f]*|[0-9]+")
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
_IPV4_COMPATIBLE = ipaddress.ip_network("::/96")

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str, int], list[str]]


class FetchErrorReason(StrEnum):
    INVALID_URL = "invalid_url"
    BLOCKED_IP = "blocked_ip"
    ROBOTS_DISALLOWED = "robots_disallowed"
    TOO_LARGE = "too_large"
    BAD_CONTENT_TYPE = "bad_content_type"
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    TOO_MANY_REDIRECTS = "too_many_redirects"
    NETWORK_ERROR = "network_error"
    REPLAY_MISS = "replay_miss"  # eval replay: the URL was not fetched in the recorded runs


class FetchError(Exception):
    def __init__(
        self, reason: FetchErrorReason, message: str, *, url: str, status_code: int | None = None
    ) -> None:
        super().__init__(f"{reason}: {message} ({url})")
        self.reason = reason
        self.url = url
        self.status_code = status_code


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str  # as requested
    final_url: str  # after redirects
    status_code: int
    content_type: str  # raw header value, e.g. "text/html; charset=utf-8"
    body: bytes


@dataclass(frozen=True, slots=True)
class _Target:
    url: httpx.URL
    host: str  # ASCII (IDNA) host, lowercased; also the per-host delay key
    origin: str  # scheme://host:port, the robots.txt cache key


@dataclass(frozen=True, slots=True)
class _Response:
    status_code: int
    content_type: str
    body: bytes
    location: str | None  # set for redirects


def system_resolver(host: str, port: int) -> list[str]:
    return [str(info[4][0]) for info in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def is_public_ip(ip: IPAddress) -> bool:
    """True only for globally routable unicast addresses, checking IPv4 embedded in IPv6."""
    if ip.is_multicast or not ip.is_global:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip in _IPV4_COMPATIBLE:  # deprecated ::a.b.c.d form; Python reports it as global
            return False
        embedded = ip.ipv4_mapped or ip.sixtofour
        if embedded is None and ip in _NAT64:
            embedded = ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF)
        if embedded is not None and not is_public_ip(embedded):
            return False
    return True


def _media_type(content_type: str) -> str:
    return content_type.split(";", 1)[0].strip().lower()


class Fetcher:
    """Fetch web pages safely. Synchronous; create one instance per research run.

    Not thread-safe: the robots.txt cache and the per-host request times are plain dicts.
    The httpx client is created with trust_env=False so that proxy environment variables can't
    route requests around the IP checks; an injected client must do the same.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        resolver: Resolver = system_resolver,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        s = settings or get_settings()
        self._user_agent = s.fetch_user_agent
        self._timeout_s = s.fetch_timeout_s
        self._min_delay_s = s.fetch_host_min_delay_s
        self._max_bytes = s.max_fetch_bytes
        self._robots_max_bytes = s.robots_max_bytes
        if client is not None and client.trust_env:
            raise ValueError("the fetcher's httpx client must be created with trust_env=False")
        self._owns_client = client is None
        self._client = client or httpx.Client(trust_env=False, follow_redirects=False)
        self._resolver = resolver
        self._clock = clock
        self._sleep = sleep
        self._robots: dict[str, RobotFileParser] = {}
        self._last_request: dict[str, float] = {}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def fetch(self, url: str) -> FetchResult:
        """Return the page, or raise FetchError with a reason code."""
        deadline = self._clock() + self._timeout_s
        target, resp = self._follow(url, deadline, for_robots=False)
        return FetchResult(
            url=url,
            final_url=str(target.url),
            status_code=resp.status_code,
            content_type=resp.content_type,
            body=resp.body,
        )

    # -- one request chain ---------------------------------------------------------------------

    def _follow(self, url: str, deadline: float, *, for_robots: bool) -> tuple[_Target, _Response]:
        """Follow up to MAX_REDIRECTS redirects by hand, re-validating every hop."""
        current: str | httpx.URL = url
        for _ in range(MAX_REDIRECTS + 1):
            target = self._check_url(current)
            if not for_robots:
                self._check_robots(target, deadline)
            self._wait_for_host(target, deadline)
            resp = self._send(target, deadline, for_robots=for_robots)
            if resp.location is None:
                return target, resp
            try:
                current = target.url.join(resp.location)
            except httpx.InvalidURL as e:
                raise self._error(FetchErrorReason.INVALID_URL, f"bad redirect: {e}", target.url) from e
        raise self._error(FetchErrorReason.TOO_MANY_REDIRECTS, f"more than {MAX_REDIRECTS} redirects", url)

    def _check_url(self, url: str | httpx.URL) -> _Target:
        try:
            parsed = httpx.URL(url)
            host = parsed.raw_host.decode("ascii").lower()
        except (httpx.InvalidURL, UnicodeError) as e:
            raise self._error(FetchErrorReason.INVALID_URL, str(e), url) from e
        if parsed.scheme not in DEFAULT_PORTS:
            raise self._error(FetchErrorReason.INVALID_URL, f"scheme {parsed.scheme!r} not allowed", url)
        if parsed.userinfo:
            raise self._error(FetchErrorReason.INVALID_URL, "credentials in URL not allowed", url)
        if not host:
            raise self._error(FetchErrorReason.INVALID_URL, "no host", url)
        port = parsed.port or DEFAULT_PORTS[parsed.scheme]
        if not 0 < port < 65536:
            raise self._error(FetchErrorReason.INVALID_URL, f"invalid port {port}", url)

        try:
            addresses: list[IPAddress] = [ipaddress.ip_address(host)]
        except ValueError:
            if _NUMERIC_LABEL.fullmatch(host.rstrip(".").rsplit(".", 1)[-1]):
                # getaddrinfo would turn 2130706433 / 0x7f.1 / 127.1 into an IP; refuse to guess
                raise self._error(
                    FetchErrorReason.INVALID_URL, f"non-canonical numeric host {host!r}", url
                ) from None
            addresses = self._resolve(host, port, url)
        for address in addresses:
            if not is_public_ip(address):
                raise self._error(
                    FetchErrorReason.BLOCKED_IP, f"{host} resolves to non-public address {address}", url
                )

        url_host = f"[{host}]" if ":" in host else host
        return _Target(url=parsed, host=host, origin=f"{parsed.scheme}://{url_host}:{port}")

    def _resolve(self, host: str, port: int, url: str | httpx.URL) -> list[IPAddress]:
        try:
            raw = self._resolver(host, port)
        except (OSError, UnicodeError) as e:  # socket.gaierror is an OSError
            raise self._error(FetchErrorReason.NETWORK_ERROR, f"cannot resolve {host}: {e}", url) from e
        if not raw:
            raise self._error(FetchErrorReason.NETWORK_ERROR, f"{host} resolved to no addresses", url)
        try:
            return [ipaddress.ip_address(a) for a in raw]
        except ValueError as e:
            raise self._error(FetchErrorReason.BLOCKED_IP, f"unparseable address for {host}: {e}", url) from e

    def _send(self, target: _Target, deadline: float, *, for_robots: bool) -> _Response:
        """One GET without redirects. The body is streamed and capped; robots.txt is truncated instead."""
        max_bytes = self._robots_max_bytes if for_robots else self._max_bytes
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise self._error(FetchErrorReason.TIMEOUT, "deadline passed", target.url)
        try:
            with self._client.stream(
                "GET",
                target.url,
                headers={"User-Agent": self._user_agent},
                follow_redirects=False,
                timeout=remaining,
            ) as resp:
                status = resp.status_code
                if status in REDIRECT_STATUSES and "location" in resp.headers:
                    return _Response(status, "", b"", resp.headers["location"])
                if not 200 <= status < 300:
                    raise self._error(
                        FetchErrorReason.HTTP_ERROR, f"HTTP {status}", target.url, status_code=status
                    )
                content_type = resp.headers.get("content-type", "").strip()
                if not for_robots and _media_type(content_type) not in ALLOWED_CONTENT_TYPES:
                    raise self._error(
                        FetchErrorReason.BAD_CONTENT_TYPE, f"content type {content_type!r}", target.url
                    )
                declared = resp.headers.get("content-length", "")
                if not for_robots and declared.isdigit() and int(declared) > max_bytes:
                    raise self._error(
                        FetchErrorReason.TOO_LARGE, f"Content-Length {declared} > {max_bytes}", target.url
                    )
                body = self._read_capped(resp, target, deadline, max_bytes, truncate=for_robots)
                return _Response(status, content_type, body, None)
        except httpx.TimeoutException as e:
            raise self._error(FetchErrorReason.TIMEOUT, type(e).__name__, target.url) from e
        except httpx.RequestError as e:
            raise self._error(FetchErrorReason.NETWORK_ERROR, f"{type(e).__name__}: {e}", target.url) from e

    def _read_capped(
        self, resp: httpx.Response, target: _Target, deadline: float, max_bytes: int, *, truncate: bool
    ) -> bytes:
        """Stream the body. Over max_bytes: too_large, or keep the first max_bytes if truncate."""
        body = bytearray()
        for chunk in resp.iter_bytes():  # decompressed bytes, so the cap also limits gzip bombs
            body += chunk
            if len(body) > max_bytes:
                if truncate:
                    del body[max_bytes:]
                    break
                raise self._error(FetchErrorReason.TOO_LARGE, f"body over {max_bytes} bytes", target.url)
            if self._clock() > deadline:
                raise self._error(FetchErrorReason.TIMEOUT, "deadline passed while reading", target.url)
        return bytes(body)

    # -- politeness ----------------------------------------------------------------------------

    def _wait_for_host(self, target: _Target, deadline: float) -> None:
        delay = self._min_delay_s
        parser = self._robots.get(target.origin)
        crawl_delay = parser.crawl_delay(self._user_agent) if parser is not None else None
        if crawl_delay is not None:
            delay = max(delay, min(float(crawl_delay), MAX_CRAWL_DELAY_S))
        last = self._last_request.get(target.host)
        if last is not None:
            wait = last + delay - self._clock()
            if wait > 0:
                if self._clock() + wait > deadline:
                    raise self._error(FetchErrorReason.TIMEOUT, "per-host delay exceeds deadline", target.url)
                self._sleep(wait)
        self._last_request[target.host] = self._clock()

    def _check_robots(self, target: _Target, deadline: float) -> None:
        parser = self._robots.get(target.origin) or self._load_robots(target, deadline)
        if parser is None:
            raise self._error(
                FetchErrorReason.ROBOTS_DISALLOWED, "robots.txt unreachable; disallowed for now", target.url
            )
        if not parser.can_fetch(self._user_agent, str(target.url)):
            raise self._error(FetchErrorReason.ROBOTS_DISALLOWED, "disallowed by robots.txt", target.url)

    def _load_robots(self, target: _Target, deadline: float) -> RobotFileParser | None:
        """Fetch and cache robots.txt (RFC 9309).

        2xx: parse and cache. 4xx: allow everything and cache. 5xx or network/timeout: retry once;
        if it still fails, return None (disallow this request) and cache nothing, so the next
        request to this host tries again. Any other failure (e.g. a redirect to a blocked IP) is
        also "disallow, not cached".
        """
        robots_url = f"{target.origin}/robots.txt"
        for _attempt in range(2):
            try:
                _, resp = self._follow(robots_url, deadline, for_robots=True)
            except FetchError as e:
                status = e.status_code
                if e.reason is FetchErrorReason.HTTP_ERROR and status is not None and 400 <= status < 500:
                    parser = RobotFileParser()
                    parser.parse([])  # an empty robots.txt allows everything
                    self._robots[target.origin] = parser
                    return parser
                retryable = e.reason in (FetchErrorReason.TIMEOUT, FetchErrorReason.NETWORK_ERROR) or (
                    e.reason is FetchErrorReason.HTTP_ERROR and status is not None and status >= 500
                )
                if retryable:
                    continue
                return None
            parser = RobotFileParser()
            parser.parse(resp.body.decode("utf-8", errors="replace").splitlines())
            self._robots[target.origin] = parser
            return parser
        return None

    @staticmethod
    def _error(
        reason: FetchErrorReason, message: str, url: str | httpx.URL, *, status_code: int | None = None
    ) -> FetchError:
        return FetchError(reason, message, url=str(url), status_code=status_code)
