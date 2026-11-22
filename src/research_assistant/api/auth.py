"""Single-user login: POST /auth/login, POST /auth/logout, GET /auth/session, and `require_session`.

- The key is compared in constant time (hmac.compare_digest over SHA-256 digests, so the lengths match).
- A session is a random id kept in memory with its expiry. The cookie carries `<id>.<expiry>` plus an
  HMAC-SHA256 signature under a key derived from ACCESS_KEY (a new ACCESS_KEY voids every cookie).
  Logout forgets the id, so a copied cookie stops working at logout; an API restart logs everyone out.
- Cookie: HttpOnly, SameSite=Strict, Path=/, Max-Age=SESSION_TTL_H, Secure when the request came over
  HTTPS.
- Failed logins are counted in memory for all clients together (one user; behind Docker every client
  has the same address): after LOGIN_MAX_FAILURES within LOGIN_WINDOW_S, every login attempt gets 429
  until the oldest failure leaves the window. A successful login does not reset the count.
- With AUTH_ENABLED false (local development) nothing is required and login answers 404.
"""

import base64
import hashlib
import hmac
import math
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from research_assistant.api.errors import ApiError
from research_assistant.api.schemas import LoginRequest, SessionInfo
from research_assistant.config import Settings

COOKIE_NAME = "ra_session"
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_S = 60.0
_SIGNING_CONTEXT = b"ra-session-v1"

Clock = Callable[[], float]


class Sessions:
    """Live session ids, the cookie signature, and the failed-login counter. Thread-safe."""

    def __init__(self, settings: Settings, clock: Clock = time.time) -> None:
        self.enabled = settings.auth_enabled
        self.ttl_s = settings.session_ttl_h * 3600
        self.clock = clock
        key = settings.access_key.get_secret_value().strip() if settings.access_key is not None else ""
        self._key_digest = hashlib.sha256(key.encode()).digest()
        self._signing_key = hmac.new(key.encode(), _SIGNING_CONTEXT, hashlib.sha256).digest()
        self._live: dict[str, float] = {}  # session id -> expiry (unix seconds)
        self._failures: deque[float] = deque()
        self._lock = threading.Lock()

    # -- login -------------------------------------------------------------------------------------

    def key_matches(self, given: str) -> bool:
        return hmac.compare_digest(hashlib.sha256(given.encode()).digest(), self._key_digest)

    def retry_after(self) -> int | None:
        """Seconds until a login may be tried again, or None if it may be tried now."""
        with self._lock:
            now = self.clock()
            while self._failures and self._failures[0] <= now - LOGIN_WINDOW_S:
                self._failures.popleft()
            if len(self._failures) < LOGIN_MAX_FAILURES:
                return None
            return max(1, math.ceil(self._failures[0] + LOGIN_WINDOW_S - now))

    def record_failure(self) -> None:
        with self._lock:
            self._failures.append(self.clock())

    def create(self) -> tuple[str, int]:
        """A new session: (cookie value, max-age seconds)."""
        session_id = secrets.token_urlsafe(32)
        now = self.clock()
        expiry = int(now + self.ttl_s)
        with self._lock:
            self._live = {sid: exp for sid, exp in self._live.items() if exp > now}  # drop expired
            self._live[session_id] = expiry
        payload = f"{session_id}.{expiry}"
        return f"{payload}.{self._sign(payload)}", int(self.ttl_s)

    # -- check and logout --------------------------------------------------------------------------

    def _sign(self, payload: str) -> str:
        mac = hmac.new(self._signing_key, payload.encode(), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(mac).rstrip(b"=").decode()

    def _session_id(self, cookie: str | None) -> str | None:
        """The session id of a validly signed, unexpired, live cookie; else None."""
        if not cookie or cookie.count(".") != 2:
            return None
        session_id, expiry_text, signature = cookie.split(".")
        expected = self._sign(f"{session_id}.{expiry_text}").encode()
        if not hmac.compare_digest(signature.encode(), expected):  # bytes: any text compares safely
            return None
        try:
            expiry = int(expiry_text)
        except ValueError:
            return None
        now = self.clock()
        with self._lock:
            live = self._live.get(session_id)
            if live is None or live != expiry:
                return None
            if expiry <= now:
                del self._live[session_id]
                return None
        return session_id

    def valid(self, cookie: str | None) -> bool:
        return self._session_id(cookie) is not None

    def end(self, cookie: str | None) -> None:
        session_id = self._session_id(cookie)
        if session_id is not None:
            with self._lock:
                self._live.pop(session_id, None)


def sessions(request: Request) -> Sessions:
    value = request.app.state.sessions
    assert isinstance(value, Sessions)
    return value


SessionsDep = Annotated[Sessions, Depends(sessions)]


def authenticated(request: Request, store: Sessions) -> bool:
    """True when auth is off or the request carries a live session cookie."""
    return not store.enabled or store.valid(request.cookies.get(COOKIE_NAME))


def require_session(request: Request, store: SessionsDep) -> None:
    if not authenticated(request, store):
        raise ApiError(401, "log in first", code="unauthorized")


router = APIRouter(prefix="/auth")


def _set_cookie(response: Response, request: Request, value: str, max_age: int) -> None:
    response.set_cookie(
        COOKIE_NAME,
        value,
        max_age=max_age,
        path="/",
        httponly=True,
        samesite="strict",
        secure=request.url.scheme == "https",
    )


@router.post("/login", response_model=SessionInfo)
def login(body: LoginRequest, request: Request, response: Response, store: SessionsDep) -> SessionInfo:
    if not store.enabled:
        raise ApiError(404, "login is not enabled (AUTH_ENABLED is false)")
    wait = store.retry_after()
    if wait is not None:
        message = f"too many failed logins; try again in {wait} s"
        raise ApiError(429, message, code="rate_limited", headers={"Retry-After": str(wait)})
    if not store.key_matches(body.key):
        store.record_failure()
        raise ApiError(401, "wrong access key", code="unauthorized")
    value, max_age = store.create()
    _set_cookie(response, request, value, max_age)
    return SessionInfo(auth_enabled=True, authenticated=True)


@router.post("/logout", response_model=SessionInfo)
def logout(request: Request, response: Response, store: SessionsDep) -> SessionInfo:
    store.end(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(
        COOKIE_NAME, path="/", httponly=True, samesite="strict", secure=request.url.scheme == "https"
    )
    return SessionInfo(auth_enabled=store.enabled, authenticated=False)


@router.get("/session", response_model=SessionInfo)
def session_info(request: Request, store: SessionsDep) -> SessionInfo:
    return SessionInfo(auth_enabled=store.enabled, authenticated=authenticated(request, store))
