"""The built UI (frontend/dist), served by the API when UI_DIST_DIR is set: same origin, no CORS.

/assets/* are the hashed bundle files. The UI's own routes (/, /login, /run/<id>) return index.html,
which holds no data; everything the UI shows comes from the API routes, which need a session.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

UI_ROUTES = ("/", "/login", "/run/{run_id}")
# The bundle loads only its own script and stylesheet and talks only to this origin.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
SHELL_HEADERS = {
    "Content-Security-Policy": CONTENT_SECURITY_POLICY,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-cache",
}


def mount(app: FastAPI, dist: Path) -> None:
    index = dist / "index.html"
    if not index.is_file() or not (dist / "assets").is_dir():
        raise RuntimeError(f"UI_DIST_DIR {dist} has no index.html and assets/ (build the UI first)")

    def shell() -> FileResponse:
        return FileResponse(index, media_type="text/html", headers=SHELL_HEADERS)

    for path in UI_ROUTES:
        app.add_api_route(path, shell, methods=["GET"], include_in_schema=False)
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
