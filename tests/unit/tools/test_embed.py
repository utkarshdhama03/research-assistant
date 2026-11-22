import json
from typing import Any

import httpx
import pytest

from research_assistant.config import Settings
from research_assistant.tools.embed import DOCUMENT_PREFIX, QUERY_PREFIX, Embedder, EmbedError

DIM = 768


class FakeOllama:
    """Answers /api/embed with one vector per input; vector[0] is the input's index in the request."""

    def __init__(self, statuses: list[int] | None = None, dim: int = DIM, drop_one: bool = False) -> None:
        self.statuses = statuses or []
        self.dim = dim
        self.drop_one = drop_one
        self.payloads: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/embed"
        payload = json.loads(request.content)
        self.payloads.append(payload)
        if self.statuses:
            status = self.statuses.pop(0)
            if status != 200:
                return httpx.Response(status, text="error")
        vectors = [[float(i)] + [0.0] * (self.dim - 1) for i in range(len(payload["input"]))]
        if self.drop_one:
            vectors = vectors[:-1]
        return httpx.Response(200, json={"model": payload["model"], "embeddings": vectors})


def make_embedder(ollama: FakeOllama, batch_size: int = 32, sleeps: list[float] | None = None) -> Embedder:
    settings = Settings(
        _env_file=None, embed_batch_size=batch_size, embed_model="nomic-embed-text", embed_dim=DIM
    )
    client = httpx.Client(transport=httpx.MockTransport(ollama))
    return Embedder(settings, client=client, sleep=(sleeps.append if sleeps is not None else lambda _s: None))


def test_embed_documents_adds_document_prefix_and_batches() -> None:
    ollama = FakeOllama()
    texts = [f"chunk {i}" for i in range(70)]
    vectors = make_embedder(ollama, batch_size=32).embed_documents(texts)

    assert len(vectors) == 70 and all(len(v) == DIM for v in vectors)
    assert [len(p["input"]) for p in ollama.payloads] == [32, 32, 6]
    sent = [s for p in ollama.payloads for s in p["input"]]
    assert sent == [DOCUMENT_PREFIX + t for t in texts]
    assert [v[0] for v in vectors[30:34]] == [30.0, 31.0, 0.0, 1.0]  # order kept across batches
    assert all(p["model"] == "nomic-embed-text" and p["truncate"] is False for p in ollama.payloads)


def test_embed_documents_does_not_change_caller_texts() -> None:
    texts = ["keep me as is"]
    make_embedder(FakeOllama()).embed_documents(texts)
    assert texts == ["keep me as is"]


def test_embed_query_adds_query_prefix() -> None:
    ollama = FakeOllama()
    vector = make_embedder(ollama).embed_query("what is an okapi")
    assert len(vector) == DIM
    assert ollama.payloads[0]["input"] == ["search_query: what is an okapi"]
    assert QUERY_PREFIX == "search_query: " and DOCUMENT_PREFIX == "search_document: "


def test_empty_input_makes_no_request() -> None:
    ollama = FakeOllama()
    assert make_embedder(ollama).embed_documents([]) == []
    assert ollama.payloads == []


def test_wrong_dimension_is_rejected() -> None:
    with pytest.raises(EmbedError, match="dimension 768, got 384"):
        make_embedder(FakeOllama(dim=384)).embed_documents(["a"])


def test_wrong_count_is_rejected() -> None:
    with pytest.raises(EmbedError, match="expected 2 embeddings, got 1"):
        make_embedder(FakeOllama(drop_one=True)).embed_documents(["a", "b"])


def test_5xx_is_retried() -> None:
    sleeps: list[float] = []
    ollama = FakeOllama(statuses=[500, 503, 200])
    assert len(make_embedder(ollama, sleeps=sleeps).embed_documents(["a"])) == 1
    assert len(ollama.payloads) == 3
    assert sleeps == [0.5, 1.0]


def test_5xx_retries_are_bounded() -> None:
    ollama = FakeOllama(statuses=[500, 500, 500, 500])
    with pytest.raises(EmbedError, match="after 3 attempts"):
        make_embedder(ollama).embed_documents(["a"])
    assert len(ollama.payloads) == 3


def test_4xx_is_not_retried() -> None:
    ollama = FakeOllama(statuses=[400])
    with pytest.raises(EmbedError, match="HTTP 400"):
        make_embedder(ollama).embed_documents(["a"])
    assert len(ollama.payloads) == 1


def test_connection_error_is_retried_then_fails() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused", request=request)

    embedder = Embedder(
        Settings(_env_file=None),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _s: None,
    )
    with pytest.raises(EmbedError, match="ConnectError"):
        embedder.embed_query("q")
    assert calls == 3
