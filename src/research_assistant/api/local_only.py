"""Local-only request guard (pure ASGI middleware).

Binding to 127.0.0.1 is not enough on its own:
- DNS rebinding: a web page on evil.example can make its name resolve to 127.0.0.1 and then talk to
  this server from the victim's browser; the browser sends `Host: evil.example:<port>`. Only the exact
  `127.0.0.1:<port>` and `localhost:<port>` are accepted. (Starlette's TrustedHostMiddleware compares
  host names without the port, so it cannot enforce the port.)
- Simple cross-site POSTs: a form or `fetch(..., {mode: "no-cors"})` with `text/plain` is sent without
  a CORS preflight. Requiring `Content-Type: application/json` on every POST forces a preflight,
  which CORS refuses for any origin but the local UI.
"""

import json
from collections.abc import Iterable

from starlette.types import ASGIApp, Receive, Scope, Send


class LocalOnlyMiddleware:
    def __init__(self, app: ASGIApp, allowed_hosts: Iterable[str]) -> None:
        self.app = app
        self.allowed_hosts = frozenset(h.lower() for h in allowed_hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan; the app has no websocket routes
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        host = headers.get("host", "").strip().lower()
        if host not in self.allowed_hosts:
            await _reject(send, 400, "bad_request", "host not allowed")
            return
        if scope.get("method") == "POST":
            media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                await _reject(send, 415, "unsupported_media_type", "POST requests must use application/json")
                return
        await self.app(scope, receive, send)


async def _reject(send: Send, status: int, code: str, message: str) -> None:
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
