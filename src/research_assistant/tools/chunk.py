"""Split a document into overlapping chunks of about `target_tokens` tokens.

Invariant: for every chunk, chunk.text == doc[chunk.char_start:chunk.char_end]. Chunks are trimmed
by moving offsets, never by editing text, so quotes found in a chunk map back to the document.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

_PARAGRAPH_BREAK = re.compile(r"\n[^\S\n]*\n\s*")
# end of a sentence (punctuation, optional closing quotes/brackets, then whitespace) or a line break
_SENTENCE_END = re.compile(r"[.!?…][\"'”’»)\]]*\s|\n")


@dataclass(frozen=True, slots=True)
class TextChunk:
    ordinal: int
    text: str
    char_start: int
    char_end: int


def _trim(doc: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and doc[start].isspace():
        start += 1
    while end > start and doc[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _pieces(doc: str, start: int, end: int, pattern: re.Pattern[str]) -> Iterator[tuple[int, int]]:
    """Trimmed pieces of doc[start:end], cut after each match of pattern."""
    pos = start
    for m in pattern.finditer(doc, start, end):
        if (span := _trim(doc, pos, m.end())) is not None:
            yield span
        pos = m.end()
    if (span := _trim(doc, pos, end)) is not None:
        yield span


def _hard_split(doc: str, start: int, end: int, max_chars: int) -> Iterator[tuple[int, int]]:
    """Cut an over-long sentence at the last whitespace within max_chars, or exactly at max_chars."""
    while end - start > max_chars:
        cut = next((i for i in range(start + max_chars, start, -1) if doc[i].isspace()), None)
        if cut is None:
            yield start, start + max_chars
            start += max_chars
        else:
            if (span := _trim(doc, start, cut)) is not None:
                yield span
            start = cut
        if (rest := _trim(doc, start, end)) is None:
            return
        start, end = rest
    yield start, end


def _spans(doc: str, max_chars: int) -> list[tuple[int, int]]:
    """Units no longer than max_chars: paragraphs, else sentences, else whitespace/hard cuts."""
    spans: list[tuple[int, int]] = []
    for p_start, p_end in _pieces(doc, 0, len(doc), _PARAGRAPH_BREAK):
        if p_end - p_start <= max_chars:
            spans.append((p_start, p_end))
            continue
        for s_start, s_end in _pieces(doc, p_start, p_end, _SENTENCE_END):
            spans.extend(_hard_split(doc, s_start, s_end, max_chars))
    return spans


def chunk_text(
    doc: str, *, target_tokens: int = 400, overlap: float = 0.15, chars_per_token: float = 4.0
) -> list[TextChunk]:
    """Pack spans greedily up to target_tokens * chars_per_token characters.

    The next chunk starts at the earliest span of the previous chunk that lies within the last
    `overlap` fraction of it, so consecutive chunks share whole sentences when they fit. Each chunk
    ends strictly after the previous one.
    """
    max_chars = max(1, round(target_tokens * chars_per_token))
    overlap_chars = int(max_chars * overlap)
    spans = _spans(doc, max_chars)
    bounds: list[tuple[int, int]] = []
    i = 0
    while i < len(spans):
        start, end = spans[i]
        j = i
        while j + 1 < len(spans) and spans[j + 1][1] - start <= max_chars:
            j += 1
            end = spans[j][1]
        bounds.append((start, end))
        if j == len(spans) - 1:
            break
        # overlap only if the next chunk can still take the following span (no chunk inside another)
        next_end = spans[j + 1][1]
        i = next(
            (
                k
                for k in range(i + 1, j + 1)
                if spans[k][0] >= end - overlap_chars and next_end - spans[k][0] <= max_chars
            ),
            j + 1,
        )
    return [TextChunk(n, doc[s:e], s, e) for n, (s, e) in enumerate(bounds)]
