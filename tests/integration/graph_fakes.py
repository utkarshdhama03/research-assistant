"""Fakes for graph tests: search, fetcher (serving tests/fixtures), embedder, and a scripted model.

Every fake asserts that no database session is open when it is called.
"""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from research_assistant.llm.fake import FakeCall, Scripted
from research_assistant.tools.embed import EmbedError
from research_assistant.tools.fetch import FetchError, FetchResult
from research_assistant.tools.search import SearchError, SearchResult
from tests.integration.conftest import TrackingSessionFactory

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

OKAPI = "https://simple.wikipedia.org/wiki/Okapi"
SECRETS = "https://docs.python.org/3/library/secrets.html"
RFC2606 = "https://www.rfc-editor.org/rfc/rfc2606.txt"
RFC9309 = "https://www.rfc-editor.org/rfc/rfc9309.pdf"
EXAMPLE = "https://example.com/"

PAGES: dict[str, tuple[str, str]] = {
    OKAPI: ("simplewiki_okapi.html", "text/html; charset=UTF-8"),
    SECRETS: ("python_docs_secrets.html", "text/html"),
    RFC2606: ("rfc2606.txt", "text/plain; charset=utf-8"),
    RFC9309: ("rfc9309.pdf", "application/pdf"),
    EXAMPLE: ("example_com.html", "text/html; charset=utf-8"),
}

PLAN = {
    "sub_questions": [
        {"question": "What is an okapi and where does it live?", "search_queries": ["okapi animal"]},
        {"question": "Which DNS names are reserved for testing?", "search_queries": ["reserved dns names"]},
        {"question": "How does the robots exclusion protocol work?", "search_queries": ["robots exclusion"]},
    ]
}
SEARCH = {"okapi animal": [OKAPI], "reserved dns names": [RFC2606], "robots exclusion": [RFC9309]}

# A q16-style false premise: the first sub-question checks the asserted fact.
PREMISE_QUESTION = "Why was the Eiffel Tower moved to Lyon in 1999?"
PREMISE_PLAN = {
    "sub_questions": [
        {
            "question": "Was the Eiffel Tower moved to Lyon in 1999?",
            "search_queries": ["eiffel tower lyon move"],
            "premise_check": True,
        },
        {"question": "What is an okapi and where does it live?", "search_queries": ["okapi animal"]},
        {"question": "Which DNS names are reserved for testing?", "search_queries": ["reserved dns names"]},
    ]
}
PREMISE_SEARCH = {**SEARCH, "eiffel tower lyon move": [SECRETS]}

_BLOCK = re.compile(r"<<<UNTRUSTED_DATA label=([A-Z]\d+)>>>\n(.*?)\n<<<END_UNTRUSTED_DATA>>>", re.DOTALL)


class FakeSearch:
    name = "fake"

    def __init__(self, results: dict[str, list[str]], failing: set[str] | None = None) -> None:
        self.results = results
        self.failing = failing or set()
        self.titles: dict[str, str] = {}  # url -> result title, instead of the generated one
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.queries.append(query)
        if query in self.failing:
            raise SearchError(f"scripted failure for {query!r}")
        urls = self.results.get(query, [])[:max_results]
        return [
            SearchResult(
                url=u,
                title=self.titles.get(u, f"Result {i} for {query}"),
                snippet=f"about {query}",
                engine="fake",
                rank=i,
            )
            for i, u in enumerate(urls, start=1)
        ]


class FakeFetcher:
    def __init__(
        self,
        tracker: TrackingSessionFactory,
        pages: dict[str, tuple[str, str] | FetchError] | None = None,
        on_fetch: Callable[[str], None] | None = None,
    ) -> None:
        self.tracker = tracker
        self.pages: dict[str, tuple[str, str] | FetchError] = dict(PAGES) if pages is None else pages
        self.on_fetch = on_fetch
        self.calls: list[str] = []

    def fetch(self, url: str) -> FetchResult:
        self.tracker.assert_idle(f"fetch({url})")
        self.calls.append(url)
        if self.on_fetch:
            self.on_fetch(url)
        entry = self.pages.get(url)
        if entry is None:
            raise AssertionError(f"FakeFetcher has no page for {url}")
        if isinstance(entry, FetchError):
            raise entry
        name, content_type = entry
        return FetchResult(url, url, 200, content_type, (FIXTURES / name).read_bytes())


class FakeEmbedder:
    """Hashing bag-of-words vectors: deterministic, and similar texts get similar vectors."""

    dim = 768

    def __init__(self, tracker: TrackingSessionFactory, fail_when: str | None = None) -> None:
        self.tracker = tracker
        self.fail_when = fail_when  # embed_documents raises EmbedError if any text contains this
        self.documents = 0
        self.queries: list[str] = []
        self.clustering_calls = 0

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        out = [v / norm for v in vec]
        if not any(out):
            out[0] = 1.0
        return out

    def embed_documents(self, texts: Any) -> list[list[float]]:
        self.tracker.assert_idle("embed_documents")
        if self.fail_when and any(self.fail_when in t for t in texts):
            raise EmbedError("scripted embed failure")
        self.documents += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.tracker.assert_idle("embed_query")
        self.queries.append(text)
        return self._vector(text)

    def embed_for_clustering(self, texts: Any) -> list[list[float]]:
        self.tracker.assert_idle("embed_for_clustering")
        self.clustering_calls += 1
        return [self._vector(t) for t in texts]

    def warm_up(self) -> float:
        return 0.0


def labels_in(call: FakeCall, prefix: str) -> list[str]:
    return re.findall(rf"label=({prefix}\d+)>>>", call.prompt.user)


def blocks_in(call: FakeCall) -> dict[str, str]:
    """label -> text of each untrusted block in the prompt."""
    return dict(_BLOCK.findall(call.prompt.user))


def sub_question_of(call: FakeCall) -> str:
    match = re.search(r"Sub-question(?: for this section)?: (.+)", call.prompt.user)
    return match.group(1).strip() if match else ""


def claim_of(call: FakeCall) -> str:
    match = re.search(r"^Claim: (.+)$", call.prompt.user, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def supported(call: FakeCall, quote: str | None = None, verdict: str = "supported") -> str:
    """A verify reply quoting the first 80 characters of C1 (exact substring of the chunk)."""
    text = blocks_in(call)["C1"]
    return json.dumps(
        {
            "verdict": verdict,
            "evidence_alias": "C1",
            "quote": text[:80] if quote is None else quote,
            "rationale": "scripted",
        }
    )


# Claims say numbers in words: the verify number check requires every digit in a claim to occur in its quote.
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def faithful_section(call: FakeCall) -> str:
    """Each claim's own text with its marker: passes the write step's sentence content check."""
    return " ".join(f"{text.strip().rstrip('.')} [{label}]." for label, text in blocks_in(call).items())


def default_section(labels: list[str]) -> str:
    return " ".join(
        f"Finding number {i} is stated here [{label}]." for i, label in enumerate(labels, start=1)
    )


class ScriptedModel:
    """Responder for FakeLLM, answering by prompt name. Overrides are callables taking the call."""

    def __init__(
        self,
        tracker: TrackingSessionFactory,
        plan: dict[str, Any] | Exception = PLAN,
        select: Callable[[FakeCall], Scripted] | None = None,
        analyze: Callable[[FakeCall], Scripted] | None = None,
        verify: Callable[[FakeCall], Scripted] | None = None,
        gap: Callable[[FakeCall], Scripted] | None = None,
        write: Callable[[FakeCall], Scripted] | None = None,
    ) -> None:
        self.tracker = tracker
        self.plan = plan
        self.select = select
        self.analyze = analyze
        self.verify = verify
        self.gap = gap
        self.write = write
        self.calls: list[FakeCall] = []
        self.analyze_rounds: Counter[str] = Counter()

    def __call__(self, call: FakeCall) -> Scripted:
        self.tracker.assert_idle(f"LLM {call.prompt.name}")
        self.calls.append(call)
        name = call.prompt.name
        if name == "plan":
            return self.plan if isinstance(self.plan, Exception) else json.dumps(self.plan)
        override = getattr(self, {"write_section": "write"}.get(name, name), None)
        if override is not None and not isinstance(override, dict):
            return override(call)  # type: ignore[no-any-return]
        if name == "select":
            assert call.schema is not None
            max_picks = call.schema.model_json_schema()["properties"]["selected"]["maxItems"]
            return json.dumps({"selected": labels_in(call, "R")[:max_picks]})
        if name == "analyze":
            return json.dumps({"claims": self.default_claims(call)})
        if name == "verify":
            return supported(call)
        if name == "gap":
            return json.dumps({"queries": [f"another angle on {sub_question_of(call)[:30]}"]})
        if name == "write_section":
            return faithful_section(call)
        raise AssertionError(f"unexpected prompt {name}")

    def default_claims(self, call: FakeCall) -> list[dict[str, Any]]:
        """Up to two claims, each citing one shown chunk, with texts unique per sub-question and round."""
        sq = sub_question_of(call)
        self.analyze_rounds[sq] += 1
        topic = " ".join(re.findall(r"[a-z]+", sq.lower())[-3:])
        return [
            {
                "text": f"Round {WORDS[self.analyze_rounds[sq]]} finding {WORDS[i]} about {topic} "
                f"from {label}.",
                "evidence": [label],
            }
            for i, label in enumerate(labels_in(call, "C")[:2], start=1)
        ]

    def count(self, name: str) -> int:
        return sum(1 for c in self.calls if c.prompt.name == name)

    def prompts(self, name: str) -> list[FakeCall]:
        return [c for c in self.calls if c.prompt.name == name]
