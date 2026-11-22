"""Embeddings through Ollama /api/embed.

nomic-embed-text (v1.5) requires a task prefix: "search_document: " for texts being indexed and
"search_query: " for queries. The prefix is added here, to the embedding input only; stored chunk
text never contains it. Use embed_documents() for chunks and embed_query() for queries.
"""

import time
from collections.abc import Callable, Sequence
from itertools import batched

import httpx

from research_assistant.config import Settings, get_settings

DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
# Model card: "clustering" groups texts and can "remove semantic duplicates"; used for claim dedupe.
CLUSTERING_PREFIX = "clustering: "


class EmbedError(Exception):
    """Ollama failed or returned vectors of the wrong count or dimension."""


class Embedder:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        s = settings or get_settings()
        self._url = s.ollama_url.rstrip("/") + "/api/embed"
        self._model = s.embed_model
        self._dim = s.embed_dim
        self._batch_size = s.embed_batch_size
        self._retries = s.embed_retries
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=s.llm_timeout_s)
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, in order. For chunks being stored."""
        return self._embed([DOCUMENT_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """One vector for a search query."""
        return self._embed([QUERY_PREFIX + text])[0]

    def embed_for_clustering(self, texts: Sequence[str]) -> list[list[float]]:
        """One vector per text, for symmetric comparison of texts with each other (claim dedupe)."""
        return self._embed([CLUSTERING_PREFIX + t for t in texts])

    def warm_up(self) -> float:
        """Load the embedding model with one tiny query. Returns the wall time in seconds."""
        start = time.perf_counter()
        self.embed_query("warm-up")
        return time.perf_counter() - start

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in batched(inputs, self._batch_size, strict=False):
            vectors.extend(self._embed_batch(list(batch)))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        # truncate=False: an over-long input is an error, not a silently shortened embedding
        payload = {"model": self._model, "input": batch, "truncate": False}
        last_error = ""
        for attempt in range(self._retries + 1):
            if attempt:
                self._sleep(0.5 * 2 ** (attempt - 1))
            try:
                resp = self._client.post(self._url, json=payload)
            except httpx.TransportError as e:
                last_error = f"{type(e).__name__}: {e}"
                continue
            if resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                continue
            if resp.status_code != 200:
                raise EmbedError(f"ollama embed returned HTTP {resp.status_code}: {resp.text[:200]}")
            return self._validate(resp, len(batch))
        raise EmbedError(f"ollama embed failed after {self._retries + 1} attempts: {last_error}")

    def _validate(self, resp: httpx.Response, expected: int) -> list[list[float]]:
        try:
            vectors = resp.json().get("embeddings")
        except (ValueError, AttributeError) as e:
            raise EmbedError("ollama embed returned invalid JSON") from e
        if not isinstance(vectors, list) or len(vectors) != expected:
            got = len(vectors) if isinstance(vectors, list) else type(vectors).__name__
            raise EmbedError(f"expected {expected} embeddings, got {got}")
        out: list[list[float]] = []
        for v in vectors:
            if not isinstance(v, list) or len(v) != self._dim:
                got = len(v) if isinstance(v, list) else type(v).__name__
                raise EmbedError(f"expected dimension {self._dim}, got {got}")
            if not all(isinstance(x, int | float) and not isinstance(x, bool) for x in v):
                raise EmbedError("embedding contains non-numeric values")
            out.append([float(x) for x in v])
        return out
