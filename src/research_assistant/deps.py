"""Production dependencies for one run (search, fetcher, embedder, LLM), closed when the run ends."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

import httpx

from research_assistant.config import get_settings
from research_assistant.db.session import default_session_factory
from research_assistant.graph.context import Deps
from research_assistant.llm.client import OllamaClient, OnCall
from research_assistant.tools.embed import Embedder
from research_assistant.tools.fetch import Fetcher
from research_assistant.tools.search import build_search_provider


@contextmanager
def production_deps(
    on_event: Callable[[str, str, dict[str, Any]], None] | None,
    stop_requested: Callable[[], bool] | None = None,
) -> Iterator[Deps]:
    settings = get_settings()
    with ExitStack() as stack:
        search_client = stack.enter_context(httpx.Client(timeout=settings.search_timeout_s, trust_env=False))
        fetcher = stack.enter_context(Fetcher(settings))
        embedder = Embedder(settings)
        stack.callback(embedder.close)

        def make_llm(on_call: OnCall) -> OllamaClient:
            client = OllamaClient(settings, on_call=on_call)
            stack.callback(client.close)
            return client

        yield Deps(
            settings=settings,
            session_factory=default_session_factory(),
            make_llm=make_llm,
            search=build_search_provider(settings, search_client),
            fetcher=fetcher,
            embedder=embedder,
            on_event=on_event,
            stop_requested=stop_requested,
        )
