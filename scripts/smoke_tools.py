"""Manual smoke test of the Phase 2 tools against the live network. Not part of pytest.

One real query -> top N results -> fetch -> extract -> chunk -> embed, with a per-step summary
and timings. Needs SearXNG (docker compose) and Ollama running.

    uv run python scripts/smoke_tools.py ["your query"] [--top 3] [--force-fallback]

--force-fallback skips SearXNG and searches through ddgs only.
"""

import argparse
import time

import httpx

from research_assistant.config import get_settings
from research_assistant.tools.chunk import chunk_text
from research_assistant.tools.embed import Embedder, EmbedError
from research_assistant.tools.extract import ExtractStatus, extract
from research_assistant.tools.fetch import Fetcher, FetchError
from research_assistant.tools.search import DdgsProvider, SearchError, SearchProvider, build_search_provider


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("query", nargs="?", default="how does the robots exclusion protocol work")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--force-fallback", action="store_true", help="search with ddgs only")
    args = parser.parse_args()
    s = get_settings()
    print(f"query: {args.query!r}")
    print(
        f"config: max_fetch_bytes={s.max_fetch_bytes} fetch_timeout_s={s.fetch_timeout_s} "
        f"min_chars={s.extract_min_chars} chunk={s.chunk_target_tokens}tok/{s.chunk_overlap:.0%} "
        f"fallback={s.search_fallback_enabled} force_fallback={args.force_fallback}"
    )

    t = time.perf_counter()
    with httpx.Client(timeout=s.search_timeout_s, trust_env=False) as client:
        provider: SearchProvider = (
            DdgsProvider(timeout_s=s.search_timeout_s)
            if args.force_fallback
            else build_search_provider(s, client)
        )
        try:
            results = provider.search(args.query, s.max_sources_per_query)
        except SearchError as e:
            print(f"[search] FAILED in {time.perf_counter() - t:.2f}s: {e}")
            return 1
    print(f"[search] {len(results)} results in {time.perf_counter() - t:.2f}s")
    for r in results:
        print(f"   {r.rank}. [{r.engine}] {r.url}  {r.title[:60]!r}")

    total_chunks = 0
    with Fetcher(s) as fetcher:
        embedder = Embedder(s)
        try:
            for r in results[: args.top]:
                print(f"\n== {r.rank}. {r.url}")
                t = time.perf_counter()
                try:
                    page = fetcher.fetch(r.url)
                except FetchError as e:
                    print(f"[fetch]   FAILED {e.reason} in {time.perf_counter() - t:.2f}s: {e}")
                    continue
                print(
                    f"[fetch]   {page.status_code} {page.content_type!r} {len(page.body)} bytes "
                    f"in {time.perf_counter() - t:.2f}s; final_url={page.final_url}"
                )

                t = time.perf_counter()
                ex = extract(page.body, page.content_type, min_chars=s.extract_min_chars)
                print(
                    f"[extract] {ex.status} reason={ex.reason} title={ex.title!r} chars={len(ex.text)} "
                    f"sha256={ex.content_hash[:12]} in {time.perf_counter() - t:.2f}s"
                )
                if ex.status is not ExtractStatus.OK:
                    continue

                t = time.perf_counter()
                chunks = chunk_text(
                    ex.text,
                    target_tokens=s.chunk_target_tokens,
                    overlap=s.chunk_overlap,
                    chars_per_token=s.chars_per_token,
                )
                assert all(c.text == ex.text[c.char_start : c.char_end] for c in chunks)
                sizes = [len(c.text) for c in chunks]
                print(
                    f"[chunk]   {len(chunks)} chunks, chars min/max {min(sizes)}/{max(sizes)}, "
                    f"slice invariant ok, in {time.perf_counter() - t:.3f}s"
                )
                total_chunks += len(chunks)

                t = time.perf_counter()
                try:
                    vectors = embedder.embed_documents([c.text for c in chunks])
                except EmbedError as e:
                    print(f"[embed]   FAILED in {time.perf_counter() - t:.2f}s: {e}")
                    continue
                print(
                    f"[embed]   {len(vectors)} vectors x {len(vectors[0])} dims "
                    f"in {time.perf_counter() - t:.2f}s"
                )

            t = time.perf_counter()
            q = embedder.embed_query(args.query)
            print(f"\n[embed_query] {len(q)} dims in {time.perf_counter() - t:.2f}s")
        finally:
            embedder.close()
    print(f"total chunks embedded: {total_chunks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
