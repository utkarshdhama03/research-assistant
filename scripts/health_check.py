"""Phase 0 health check: Postgres + pgvector, SearXNG JSON, Ollama, embeddings.

Usage: uv run python scripts/health_check.py
Exits non-zero if any check fails.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import httpx
import psycopg

from research_assistant.config import get_settings


def _pg_dsn(sqlalchemy_url: str) -> str:
    # psycopg wants a plain libpq URL, not the SQLAlchemy dialect form.
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


def check_postgres(url: str) -> str:
    with psycopg.connect(_pg_dsn(url), connect_timeout=5) as conn:
        server = conn.execute("SHOW server_version").fetchone()[0]
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        ext = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0]
        dist = conn.execute("SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector").fetchone()[0]
        conn.commit()
    return f"server {server}, vector {ext}, test distance {dist}"


def check_searxng() -> str:
    s = get_settings()
    r = httpx.get(
        f"{s.searxng_url}/search",
        params={"q": "pgvector postgres extension", "format": "json"},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if not results:
        raise RuntimeError(
            f"JSON returned but 0 results; unresponsive engines: {data.get('unresponsive_engines')}"
        )
    return f"{len(results)} results, first: {results[0].get('url')}"


def check_ollama() -> str:
    s = get_settings()
    v = httpx.get(f"{s.ollama_url}/api/version", timeout=10).json()["version"]
    tags = httpx.get(f"{s.ollama_url}/api/tags", timeout=10).json()["models"]
    names = sorted(m["name"] for m in tags)
    missing = [
        m
        for m in {s.chat_model, s.verify_model, s.embed_model}
        if m not in names and f"{m}:latest" not in names
    ]
    if missing:
        raise RuntimeError(f"Ollama {v} up but models missing: {missing}")
    return f"version {v}, models: {', '.join(names)}"


def check_embeddings() -> str:
    s = get_settings()
    r = httpx.post(
        f"{s.ollama_url}/api/embed",
        json={"model": s.embed_model, "input": "health check"},
        timeout=120,
    )
    r.raise_for_status()
    dim = len(r.json()["embeddings"][0])
    if dim != s.embed_dim:
        raise RuntimeError(f"expected {s.embed_dim} dims, got {dim}")
    return f"{s.embed_model} -> {dim} dims"


def main() -> int:
    s = get_settings()
    checks: list[tuple[str, Callable[[], str]]] = [
        ("postgres (main)", lambda: check_postgres(s.database_url)),
        ("postgres (test)", lambda: check_postgres(s.test_database_url)),
        ("searxng json", check_searxng),
        ("ollama", check_ollama),
        ("embeddings", check_embeddings),
    ]
    failed = 0
    for name, fn in checks:
        try:
            print(f"[ OK ] {name}: {fn()}")
        except Exception as e:  # report every check, don't stop at the first failure
            failed += 1
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
