# syntax=docker/dockerfile:1
# One image for the API (which also serves the built UI), the worker and migrations.
# Base images are pinned by tag and digest so a rebuild uses exactly the same images.

# -- 1. Build the UI ------------------------------------------------------------------------------
FROM node:26.13.0-bookworm-slim@sha256:add5c17c14752c54d2f3d519ec46a3815684d41c33f198bcb09fb998f7dcbf68 AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY frontend/ ./
# Same origin as the API: no VITE_API_BASE_URL, so the UI calls /runs, /auth, ... on its own host.
RUN npm run build

# -- 2. Python dependencies (no dev tools) ---------------------------------------------------------
FROM python:3.13.18-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72 AS deps
COPY --from=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src
# Dependencies first: this layer is rebuilt only when pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

# -- 3. Runtime ------------------------------------------------------------------------------------
FROM python:3.13.18-slim-bookworm@sha256:a5a7017efc5969e00ae2b91c8eb0631049a569652aa4ffc4501769049ae45c72 AS runtime
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --home-dir /app --no-create-home --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=deps /app/.venv /app/.venv
COPY alembic.ini ./
COPY migrations/ ./migrations/
COPY --from=ui /ui/dist /app/ui
# data/: where `ra research` writes report files (the only path the app writes to)
RUN mkdir -p /app/data && chown app:app /app/data
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UI_DIST_DIR=/app/ui
USER app
EXPOSE 8000
CMD ["ra", "serve"]
