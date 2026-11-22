"""Quote location, verdict rules and claim dedupe (pure)."""

import difflib
import math
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass

from research_assistant.db.models import Verdict

_WS = re.compile(r"\s+")

QUOTE_NOT_FOUND = "quote_not_found"
NUMBER_NOT_IN_QUOTE = "number_not_in_quote"


@dataclass(frozen=True)
class QuoteMatch:
    start: int
    end: int
    how: str  # "exact", "whitespace", "list" or "fuzzy" (the match method, logged with every verdict)


# List markers at line starts, ignored by the list-normalized match: -, •, *, ·, "1." and "1)".
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-•*·]|\d{1,3}[.)])[ \t]+", re.MULTILINE)
# Bounded fuzzy match (typos the model silently corrected): only for longer quotes, at most this many
# character edits and at most this share of the quote's length.
FUZZY_MIN_CHARS = 40
FUZZY_MAX_EDITS = 2
FUZZY_MAX_SHARE = 0.03


def _normalize(text: str, *, list_mode: bool) -> tuple[str, list[int]]:
    """Text with whitespace runs collapsed to one space (and, in list mode, list markers at line starts
    removed and commas/semicolons before a space dropped), plus each kept character's index in text."""
    removed = [False] * len(text)
    if list_mode:
        for m in _LIST_MARKER.finditer(text):
            for i in range(m.start(), m.end()):
                removed[i] = text[i] != "\n"  # the marker goes; the line break stays as a space
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text):
        if removed[i]:
            continue
        if ch.isspace():
            if chars and chars[-1] == " ":
                continue
            chars.append(" ")
        else:
            chars.append(ch)
        index.append(i)
    if list_mode:  # a joined list: "A, B" in the quote for "A" and "B" on separate lines
        keep = [not (c in ",;" and k + 1 < len(chars) and chars[k + 1] == " ") for k, c in enumerate(chars)]
        chars = [c for c, ok in zip(chars, keep, strict=True) if ok]
        index = [i for i, ok in zip(index, keep, strict=True) if ok]
    # Leading/trailing spaces and spaces that became adjacent after dropping separators.
    out_chars: list[str] = []
    out_index: list[int] = []
    for c, i in zip(chars, index, strict=True):
        if c == " " and (not out_chars or out_chars[-1] == " "):
            continue
        out_chars.append(c)
        out_index.append(i)
    while out_chars and out_chars[-1] == " ":
        out_chars.pop()
        out_index.pop()
    return "".join(out_chars), out_index


def _edits(a: str, b: str) -> int:
    """Character edits between a and b from difflib's opcodes (a replace counts its longer side)."""
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in ops if tag != "equal")


def _fuzzy(normalized: str, needle: str) -> tuple[int, int] | None:
    """Best window of normalized within the edit bound around the longest common block of needle."""
    if len(needle) < FUZZY_MIN_CHARS:
        return None
    limit = min(FUZZY_MAX_EDITS, int(len(needle) * FUZZY_MAX_SHARE))
    block = difflib.SequenceMatcher(None, normalized, needle, autojunk=False).find_longest_match(
        0, len(normalized), 0, len(needle)
    )
    if block.size == 0:
        return None
    anchor = block.a - block.b
    best: tuple[int, int, int] | None = None
    for start in range(max(0, anchor - limit), min(len(normalized), anchor + limit) + 1):
        for length in range(len(needle) - limit, len(needle) + limit + 1):
            end = start + length
            if length <= 0 or end > len(normalized):
                continue
            edits = _edits(normalized[start:end], needle)
            if edits <= limit and (best is None or edits < best[0]):
                best = (edits, start, end)
    return (best[1], best[2]) if best is not None else None


def locate_quote(text: str, quote: str) -> QuoteMatch | None:
    """Find quote in text, in this order: exact; whitespace runs collapsed; list-normalized (list markers
    at line starts and the commas joining items ignored); bounded fuzzy on the list-normalized text
    (quote >= FUZZY_MIN_CHARS, <= FUZZY_MAX_EDITS edits and <= FUZZY_MAX_SHARE of its length). Offsets
    are mapped back to the original text, so text[start:end] is always a real substring of text and
    that, never the model's wording, is what gets stored."""
    quote = quote.strip()
    if not quote:
        return None
    exact = text.find(quote)
    if exact >= 0:
        return QuoteMatch(exact, exact + len(quote), "exact")
    for how, list_mode in (("whitespace", False), ("list", True)):
        normalized, index = _normalize(text, list_mode=list_mode)
        needle, _ = _normalize(quote, list_mode=list_mode)
        if not needle:
            return None
        pos = normalized.find(needle)
        if pos >= 0:
            return QuoteMatch(index[pos], index[pos + len(needle) - 1] + 1, how)
    window = _fuzzy(normalized, needle)
    if window is None:
        return None
    start, end = window
    return QuoteMatch(index[start], index[end - 1] + 1, "fuzzy")


def final_verdict(verdict: Verdict, match: QuoteMatch | None) -> tuple[Verdict, str | None]:
    """supported/partial need a located quote, otherwise they become unsupported (quote_not_found).
    unsupported and contradicted are kept as given."""
    if verdict in (Verdict.SUPPORTED, Verdict.PARTIAL) and match is None:
        return Verdict.UNSUPPORTED, QUOTE_NOT_FOUND
    return verdict, None


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass(frozen=True)
class DroppedClaim:
    claim_id: uuid.UUID
    duplicate_of: uuid.UUID
    similarity: float


def dedupe_claims(
    claim_ids: Sequence[uuid.UUID], vectors: Sequence[Sequence[float]], threshold: float
) -> tuple[list[uuid.UUID], list[DroppedClaim]]:
    """Walk claims in the given (ordinal) order; drop a claim whose cosine similarity with an earlier
    kept claim is >= threshold. Returns (kept ids, dropped)."""
    kept: list[tuple[uuid.UUID, Sequence[float]]] = []
    dropped: list[DroppedClaim] = []
    for claim_id, vec in zip(claim_ids, vectors, strict=True):
        best_id, best = None, -1.0
        for other_id, other_vec in kept:
            sim = cosine(vec, other_vec)
            if sim > best:
                best_id, best = other_id, sim
        if best_id is not None and best >= threshold:
            dropped.append(DroppedClaim(claim_id, best_id, round(best, 4)))
        else:
            kept.append((claim_id, vec))
    return [k for k, _ in kept], dropped


def most_similar_pairs(
    claim_ids: Sequence[uuid.UUID], vectors: Sequence[Sequence[float]], top: int
) -> list[tuple[uuid.UUID, uuid.UUID, float]]:
    pairs = [
        (claim_ids[i], claim_ids[j], cosine(vectors[i], vectors[j]))
        for i in range(len(claim_ids))
        for j in range(i + 1, len(claim_ids))
    ]
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs[:top]
