"""Property-style chunking tests over every fixture and over seeded random documents."""

import random
from pathlib import Path

import pytest

from research_assistant.tools.chunk import TextChunk, chunk_text
from research_assistant.tools.extract import extract

FIXTURES = Path(__file__).parents[2] / "fixtures"
FIXTURE_TYPES = {
    "simplewiki_okapi.html": "text/html",
    "python_docs_secrets.html": "text/html",
    "example_com.html": "text/html",
    "rfc2606.txt": "text/plain; charset=utf-8",
    "rfc9309.pdf": "application/pdf",
}
# (target_tokens, overlap, chars_per_token)
PARAMS = [(400, 0.15, 4.0), (100, 0.15, 4.0), (50, 0.3, 3.5), (20, 0.0, 4.0), (5, 0.15, 4.0)]


def check_invariants(
    doc: str, chunks: list[TextChunk], target_tokens: int, overlap: float, cpt: float
) -> None:
    max_chars = max(1, round(target_tokens * cpt))
    covered = [False] * len(doc)
    for n, c in enumerate(chunks):
        assert c.ordinal == n
        assert c.text == doc[c.char_start : c.char_end]  # the core invariant
        assert 0 < len(c.text) <= max_chars
        assert not c.text[0].isspace() and not c.text[-1].isspace()
        for i in range(c.char_start, c.char_end):
            covered[i] = True
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert cur.char_start > prev.char_start  # always moves forward
        assert cur.char_end > prev.char_end
        assert prev.char_end - cur.char_start <= int(max_chars * overlap)  # overlap within budget
    missing = [i for i, ch in enumerate(doc) if not ch.isspace() and not covered[i]]
    assert missing == [], f"non-whitespace characters not in any chunk: {missing[:10]}"


def fixture_text(name: str) -> str:
    return extract((FIXTURES / name).read_bytes(), FIXTURE_TYPES[name], min_chars=0).text


@pytest.mark.parametrize("name", sorted(FIXTURE_TYPES))
@pytest.mark.parametrize(("target_tokens", "overlap", "cpt"), PARAMS)
def test_invariants_on_fixtures(name: str, target_tokens: int, overlap: float, cpt: float) -> None:
    doc = fixture_text(name)
    chunks = chunk_text(doc, target_tokens=target_tokens, overlap=overlap, chars_per_token=cpt)
    assert chunks
    check_invariants(doc, chunks, target_tokens, overlap, cpt)


def test_invariants_on_raw_unnormalized_fixture() -> None:
    doc = (FIXTURES / "rfc2606.txt").read_text(encoding="utf-8")  # original spacing and form feeds
    check_invariants(doc, chunk_text(doc), 400, 0.15, 4.0)


ALPHABET = [
    "word",
    "Wörter",
    "数据",
    "x",
    "a.",
    "b!",
    "c?",
    "(d).",
    '"e."',
    "-",
    "…",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
]
SEPARATORS = [" ", " ", " ", "  ", "\n", "\n\n", "\n \n", "\t", " ", ""]


def random_doc(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(0, 400)):
        parts.append(rng.choice(ALPHABET))
        parts.append(rng.choice(SEPARATORS))
    return rng.choice(["", " ", "\n"]) + "".join(parts)


@pytest.mark.parametrize("seed", range(150))
def test_invariants_on_random_documents(seed: int) -> None:
    rng = random.Random(seed)
    doc = random_doc(rng)
    target_tokens, overlap, cpt = rng.choice(PARAMS)
    check_invariants(
        doc,
        chunk_text(doc, target_tokens=target_tokens, overlap=overlap, chars_per_token=cpt),
        target_tokens,
        overlap,
        cpt,
    )


@pytest.mark.parametrize("doc", ["", "   ", "\n\n\t \n"])
def test_empty_documents(doc: str) -> None:
    assert chunk_text(doc) == []


def test_short_document_is_one_chunk() -> None:
    doc = "  One short paragraph.\n\nAnd another.  "
    assert [c.text for c in chunk_text(doc)] == ["One short paragraph.\n\nAnd another."]


def test_breaks_on_paragraph_boundaries() -> None:
    p1 = "First paragraph sentence. " * 5
    p2 = "Second paragraph sentence. " * 5
    doc = f"{p1.strip()}\n\n{p2.strip()}"
    chunks = chunk_text(doc, target_tokens=40, overlap=0.0, chars_per_token=4.0)  # 160 chars
    assert [c.text for c in chunks] == [p1.strip(), p2.strip()]


def test_breaks_on_sentence_boundaries_with_overlap() -> None:
    sentences = [f"Sentence number {i:03d} is here." for i in range(100)]  # 30 chars each
    doc = " ".join(sentences)
    chunks = chunk_text(doc, target_tokens=100, overlap=0.15, chars_per_token=4.0)  # 400 chars, 60 overlap
    assert len(chunks) > 5
    for c in chunks:
        assert c.text.startswith("Sentence number") and c.text.endswith("is here.")
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert 0 < prev.char_end - cur.char_start <= 60  # consecutive chunks share whole sentences


def test_unbreakable_text_is_cut_at_max_chars() -> None:
    doc = "x" * 1000
    chunks = chunk_text(doc, target_tokens=100, overlap=0.15, chars_per_token=4.0)
    assert [len(c.text) for c in chunks] == [400, 400, 200]
    check_invariants(doc, chunks, 100, 0.15, 4.0)


def test_long_sentence_is_cut_at_whitespace() -> None:
    doc = " ".join(["word"] * 300)  # 1499 chars, no sentence ends
    chunks = chunk_text(doc, target_tokens=100, overlap=0.0, chars_per_token=4.0)
    assert all(c.text.startswith("word") and c.text.endswith("word") for c in chunks)
    check_invariants(doc, chunks, 100, 0.0, 4.0)
