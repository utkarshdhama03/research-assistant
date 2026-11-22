"""Phase 9: login, sessions, rate limit, protected routes, /health detail, the served UI shell."""

import re
import uuid
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import SecretStr

from research_assistant.api.app import create_app
from research_assistant.api.auth import COOKIE_NAME, LOGIN_MAX_FAILURES
from research_assistant.api.ui import CONTENT_SECURITY_POLICY, UI_ROUTES
from research_assistant.db.repositories.queue import beat
from research_assistant.db.session import session_scope
from tests.integration.conftest import TrackingSessionFactory
from tests.integration.phase7_support import API_PORT, health_transport
from tests.integration.test_graph import Harness

KEY = "k" * 40
PUBLIC = {("POST", "/auth/login"), ("POST", "/auth/logout"), ("GET", "/auth/session"), ("GET", "/health")}
PUBLIC |= {("GET", path) for path in UI_ROUTES}


class Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def h(committing_factory: TrackingSessionFactory) -> Harness:
    return Harness(committing_factory)


@pytest.fixture
def ui_dist(tmp_path: Path) -> Path:
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><div id=root></div>", encoding="utf-8")
    (tmp_path / "assets" / "index-abc.js").write_text("console.log(1)", encoding="utf-8")
    return tmp_path


def auth_app(h: Harness, clock: Clock, ui_dist: Path | None = None, **settings: Any) -> FastAPI:
    values = {"api_port": API_PORT, "auth_enabled": True, "access_key": SecretStr(KEY)}
    values["ui_dist_dir"] = ui_dist
    s = h.settings.model_copy(update={**values, **settings})
    return create_app(
        s,
        h.factory,
        http_client=httpx.Client(transport=health_transport()),
        delete_checkpoint=lambda _run_id: None,
        extra_hosts=("testserver",),
        clock=clock,
    )


def login(client: TestClient, key: str = KEY) -> httpx.Response:
    return client.post("/auth/login", json={"key": key})


def set_cookie(response: httpx.Response) -> SimpleCookie:
    cookie: SimpleCookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    return cookie


def routes(app: FastAPI) -> list[tuple[str, str]]:
    """Every API route (from the OpenAPI document, which app.openapi() builds even when /openapi.json is
    off) plus the UI shell routes, with a random UUID for each path parameter."""
    found = [
        (method.upper(), path)
        for path, item in app.openapi()["paths"].items()
        for method in item
        if method in ("get", "post", "put", "patch", "delete")
    ]
    found += [("GET", path) for path in UI_ROUTES]
    return [(m, re.sub(r"\{[^}]+\}", str(uuid.uuid4()), path)) for m, path in found]


def test_every_route_but_the_public_ones_needs_a_session(h: Harness, ui_dist: Path) -> None:
    app = auth_app(h, Clock(), ui_dist)
    client = TestClient(app)
    public = set()
    for method, path in routes(app):
        response = client.request(method, path, json={} if method == "POST" else None)
        if response.status_code == 401:
            assert response.json()["error"]["code"] == "unauthorized"
        else:
            public.add((method, path))
    # Every route that answered without a session is one of the deliberate public ones.
    run_ui = [p for m, p in public if p.startswith("/run/")]
    normalised = {(m, "/run/{run_id}" if p.startswith("/run/") else p) for m, p in public}
    assert normalised == PUBLIC and len(run_ui) == 1
    assert len(routes(app)) - len(public) >= 9  # the run routes really were walked
    assert client.get("/openapi.json").status_code == 404 and client.get("/docs").status_code == 404
    assert client.get("/runs").status_code == 401  # the list, the detail and the stream are covered above
    assert client.get("/assets/index-abc.js").status_code == 200  # the bundle holds no data


def test_login_wrong_key_then_rate_limit(h: Harness) -> None:
    clock = Clock()
    client = TestClient(auth_app(h, clock))
    for _ in range(LOGIN_MAX_FAILURES):
        response = login(client, "wrong" * 8)
        assert (response.status_code, response.json()["error"]["code"]) == (401, "unauthorized")
        assert "set-cookie" not in response.headers
        clock.now += 1
    limited = login(client)  # the right key is refused too while limited
    assert (limited.status_code, limited.json()["error"]["code"]) == (429, "rate_limited")
    assert limited.headers["retry-after"] == "55"  # the oldest failure leaves the 60 s window in 55 s
    assert "set-cookie" not in limited.headers
    clock.now += 55
    assert login(client).status_code == 200


def test_login_sets_a_strict_http_only_cookie_and_opens_protected_routes(h: Harness) -> None:
    client = TestClient(auth_app(h, Clock()))
    assert client.get("/runs").status_code == 401
    response = login(client)
    assert response.status_code == 200 and response.json() == {"auth_enabled": True, "authenticated": True}
    morsel = set_cookie(response)[COOKIE_NAME]
    assert morsel["httponly"] is True and morsel["samesite"].lower() == "strict"
    assert morsel["path"] == "/" and morsel["max-age"] == str(12 * 3600)
    assert not morsel["secure"]  # plain http (local); see the https test
    assert client.get("/runs").status_code == 200
    assert client.get("/auth/session").json() == {"auth_enabled": True, "authenticated": True}


def test_cookie_is_secure_over_https(h: Harness) -> None:
    client = TestClient(auth_app(h, Clock()), base_url="https://testserver")
    morsel = set_cookie(login(client))[COOKIE_NAME]
    assert morsel["secure"] is True
    assert client.get("/runs").status_code == 200


def test_expired_session_is_refused(h: Harness) -> None:
    clock = Clock()
    client = TestClient(auth_app(h, clock, session_ttl_h=1.0))
    login(client)
    clock.now += 3599
    assert client.get("/runs").status_code == 200
    clock.now += 1
    assert client.get("/runs").status_code == 401
    assert client.get("/auth/session").json()["authenticated"] is False


def test_tampered_cookie_and_logged_out_cookie_are_refused(h: Harness) -> None:
    client = TestClient(auth_app(h, Clock()))
    value = set_cookie(login(client))[COOKIE_NAME].value
    session_id, expiry, signature = value.split(".")
    forged = [
        f"{session_id}.{int(expiry) + 3600}.{signature}",  # longer expiry, old signature
        f"{session_id}.{expiry}.{signature[:-2]}AA",
        "garbage",
    ]
    for bad in forged:
        other = TestClient(client.app)
        other.cookies.set(COOKIE_NAME, bad)
        assert other.get("/runs").status_code == 401, bad
    assert not client.app.state.sessions.valid(f"{session_id}.{expiry}.é{signature[1:]}")  # no crash
    assert client.post("/auth/logout", json={}).status_code == 200
    replay = TestClient(client.app)
    replay.cookies.set(COOKIE_NAME, value)  # a copy of the cookie taken before logout
    assert replay.get("/runs").status_code == 401


def test_a_new_access_key_voids_old_cookies(h: Harness) -> None:
    client = TestClient(auth_app(h, Clock()))
    value = set_cookie(login(client))[COOKIE_NAME].value
    other = TestClient(auth_app(h, Clock(), access_key=SecretStr("n" * 40)))
    other.cookies.set(COOKIE_NAME, value)
    assert other.get("/runs").status_code == 401


def test_health_without_a_session_is_the_status_only(h: Harness) -> None:
    with session_scope(h.factory) as session:
        beat(session, uuid.uuid4(), None)
    client = TestClient(auth_app(h, Clock()))
    response = client.get("/health")
    assert (response.status_code, response.json()) == (200, {"status": "ok"})
    login(client)
    detailed = client.get("/health").json()
    assert detailed["status"] == "ok" and detailed["db"]["ok"] and detailed["worker"]["ok"]


def test_health_without_a_session_keeps_the_503(h: Harness) -> None:
    app = auth_app(h, Clock())
    app.state.api.http_client = httpx.Client(transport=health_transport(ollama=False))
    response = TestClient(app).get("/health")
    assert (response.status_code, response.json()) == (503, {"status": "degraded"})


def test_login_is_off_without_auth(h: Harness) -> None:
    client = TestClient(auth_app(h, Clock(), auth_enabled=False, access_key=None))
    assert login(client).status_code == 404
    assert client.get("/auth/session").json() == {"auth_enabled": False, "authenticated": True}
    assert client.get("/runs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_cors_lets_the_dev_ui_send_the_cookie(h: Harness) -> None:
    client = TestClient(auth_app(h, Clock()))
    response = client.options(
        "/auth/login",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert response.headers["access-control-allow-credentials"] == "true"
    foreign = client.options(
        "/auth/login",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in foreign.headers


def test_ui_shell_routes_serve_index_html_with_a_csp(h: Harness, ui_dist: Path) -> None:
    client = TestClient(auth_app(h, Clock(), ui_dist))
    for path in ("/", "/login", f"/run/{uuid.uuid4()}"):
        response = client.get(path)
        assert response.status_code == 200 and response.text.startswith("<!doctype html>")
        assert response.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
        assert response.headers["x-content-type-options"] == "nosniff"
    assert client.get("/nowhere").status_code == 404
    assert client.get("/runs/x").status_code == 401  # API paths are not swallowed by the UI


def test_extra_allowed_hosts_come_from_settings(h: Harness) -> None:
    app = auth_app(h, Clock(), api_allowed_hosts=["research.lan:8000"])
    assert TestClient(app, base_url="http://research.lan:8000").get("/auth/session").status_code == 200
    assert TestClient(app, base_url="http://other.lan:8000").get("/auth/session").status_code == 400
