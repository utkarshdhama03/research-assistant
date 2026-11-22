"""Citation handling for report sections (pure, no I/O).

The writer marks claims with aliases like [K3] or [K1][K4] (a comma list inside one bracket is
accepted too). Every bracket whose content looks like a label or a number must resolve exactly
through the section's AliasMap; any URL in the text is also rejected. Sentences without a marker
are deleted. Each claim maps to the chunk holding its verified quote, and so to a source;
assemble_report() numbers sources [1]..[n] in order of first citation, builds the reference list,
the citations record and the evidence appendix.
"""

import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from research_assistant.graph.faithfulness import sentence_gaps
from research_assistant.llm.aliases import AliasMap
from research_assistant.llm.errors import UnknownAliasError

_BRACKET = re.compile(r"\[([^\[\]\n]{1,40})\]")
# Label-like tokens: K3, k3, K03, C2, also bare numbers such as 1 (a citation the model made up).
_LABEL_LIKE = re.compile(r"[A-Za-z]{0,3}\s*\d{1,4}")
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# Sentence end: . ! ? (plus closing quotes/brackets), whitespace, then a capital, digit, quote,
# opening bracket or the end of the text. "e.g. pgvector" is not split.
_BOUNDARY = re.compile(r"[.!?][\"'”’)\]]*(\s+)(?=[A-Z0-9\"“‘(]|$)")
# Text after a sentence's last citation: (anything)(final punctuation with closers)?(whitespace)
_TAIL = re.compile(r"(.*?)([.!?][\"'”’)\]]*)?\s*", re.DOTALL)


@dataclass(frozen=True)
class Cite:
    ids: tuple[uuid.UUID, ...]  # claim ids


Segment = str | Cite


class CitationError(Exception):
    def __init__(self, unknown: list[str], urls: list[str]) -> None:
        parts = []
        if unknown:
            parts.append(f"unknown labels {unknown}")
        if urls:
            parts.append(f"URLs {urls}")
        super().__init__("; ".join(parts))
        self.unknown = unknown
        self.urls = urls


def parse_section(text: str, aliases: AliasMap[uuid.UUID]) -> list[Segment]:
    """Split text into prose and resolved citations. Raises CitationError on any unknown label or URL."""
    urls = _URL.findall(text)
    unknown: list[str] = []
    segments: list[Segment] = []
    pos = 0
    for m in _BRACKET.finditer(text):
        labels = [part.strip() for part in m.group(1).split(",")]
        if not all(_LABEL_LIKE.fullmatch(label) for label in labels):
            continue  # ordinary bracketed prose, e.g. "[sic]"
        ids: list[uuid.UUID] = []
        for label in labels:
            try:
                ids.append(aliases.resolve(label))
            except UnknownAliasError:
                unknown.append(label)
        segments.append(text[pos : m.start()])
        segments.append(Cite(tuple(ids)))
        pos = m.end()
    segments.append(text[pos:])
    if unknown or urls:
        raise CitationError(unknown, urls)
    return _merge_adjacent(segments)


def _merge_adjacent(segments: Sequence[Segment]) -> list[Segment]:
    """[K1][K2] -> one Cite; adjacent strings joined; empty strings dropped."""
    out: list[Segment] = []
    for seg in segments:
        if isinstance(seg, str):
            if not seg:
                continue
            if out and isinstance(out[-1], str):
                out[-1] += seg
            else:
                out.append(seg)
            continue
        if out and isinstance(out[-1], Cite):
            out[-1] = Cite(out[-1].ids + seg.ids)
        else:
            out.append(seg)
    return out


def _has_text(parts: list[Segment]) -> bool:
    return any(isinstance(p, str) and p.strip() for p in parts)


def split_sentences(segments: Sequence[Segment]) -> list[list[Segment]]:
    """Group segments into sentences. A citation that follows a finished sentence ("Fact. [K1]")
    belongs to that sentence."""
    sentences: list[list[Segment]] = [[]]
    # Sentence boundaries are found inside strings, so a "." and the following " Next" must be one string.
    for seg in _merge_adjacent(segments):
        if isinstance(seg, Cite):
            if _has_text(sentences[-1]) or len(sentences) == 1:
                sentences[-1].append(seg)
            else:  # "Fact. [K1]": the whitespace and the citation belong to the finished sentence
                whitespace = sentences.pop()
                sentences[-1].extend(whitespace)
                sentences[-1].append(seg)
                sentences.append([])
            continue
        pos = 0
        for m in _BOUNDARY.finditer(seg):
            sentences[-1].append(seg[pos : m.start(1)])
            sentences.append([seg[m.start(1) : m.end(1)]])
            pos = m.end(1)
        sentences[-1].append(seg[pos:])
    return sentences


@dataclass(frozen=True)
class CleanedSection:
    segments: list[Segment]
    deleted: list[str]  # whole sentences without a citation
    trimmed: list[str]  # text that followed the last citation inside a cited sentence


def trim_after_last_cite(sentence: list[Segment]) -> tuple[list[Segment], str | None]:
    """Cut everything after the sentence's last citation, keeping its final punctuation.
    "A [K1], which may matter." -> "A [K1]." and the removed fragment ", which may matter"."""
    last = max(i for i, p in enumerate(sentence) if isinstance(p, Cite))
    tail = "".join(p for p in sentence[last + 1 :] if isinstance(p, str))
    m = _TAIL.fullmatch(tail)
    assert m is not None  # the pattern matches any string
    fragment, punctuation = m.group(1), m.group(2) or ""
    if not fragment.strip():
        return sentence, None
    return [*sentence[: last + 1], punctuation], fragment.strip()


def remove_uncited(segments: Sequence[Segment]) -> CleanedSection:
    """Delete every sentence that has text but no citation, and cut text after the last citation
    of every remaining sentence."""
    kept: list[Segment] = []
    deleted: list[str] = []
    trimmed: list[str] = []
    for sentence in split_sentences(segments):
        if not _has_text(sentence):
            continue
        if not any(isinstance(p, Cite) for p in sentence):
            deleted.append("".join(p for p in sentence if isinstance(p, str)).strip())
            continue
        sentence, fragment = trim_after_last_cite(sentence)
        if fragment is not None:
            trimmed.append(fragment)
        kept.extend(sentence)
    return CleanedSection(_merge_adjacent(kept), deleted, trimmed)


SECTION_KINDS = ("normal", "insufficient", "covered", "no_sources", "failed")


@dataclass
class SectionContent:
    heading: str
    segments: list[Segment] = field(default_factory=list)  # empty if the section has no written text
    note: str | None = None  # shown in italics above the text
    kind: str = "normal"  # one of SECTION_KINDS (the UI's label for the section)


@dataclass(frozen=True)
class SourceRef:
    title: str
    url: str
    retrieved: date | None = None  # fetch date (UTC); a reused source keeps the original date
    reused: bool = False  # copied from an earlier run (memory)


@dataclass(frozen=True)
class ClaimRef:
    """A written claim: the chunk and source of its verified quote."""

    ordinal: int
    text: str
    quote: str
    chunk_id: uuid.UUID
    source_id: uuid.UUID
    verdict: str = "supported"  # write receives supported claims only; the caller passes the stored verdict


@dataclass(frozen=True)
class AssembledReport:
    markdown: str
    citations: list[dict[str, Any]]
    cited_claims: list[uuid.UUID]
    # The report as data (reports.structure); `markdown` is rendered from it by render_markdown().
    structure: dict[str, Any] = field(default_factory=dict)


def render_segments(
    segments: Sequence[Segment], claims: Mapping[uuid.UUID, ClaimRef], numbers: dict[uuid.UUID, int]
) -> str:
    """Replace Cites with [n] per source, assigning new numbers in order of first appearance."""
    parts: list[str] = []
    for seg in segments:
        if isinstance(seg, str):
            parts.append(seg)
            continue
        cited: list[int] = []
        for claim_id in seg.ids:
            source_id = claims[claim_id].source_id
            if source_id not in numbers:
                numbers[source_id] = len(numbers) + 1
            if numbers[source_id] not in cited:
                cited.append(numbers[source_id])
        parts.append("".join(f"[{n}]" for n in cited))
    return "".join(parts).strip()


_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")


def _section_sentences(
    segments: Sequence[Segment], claims: Mapping[uuid.UUID, ClaimRef], numbers: dict[uuid.UUID, int]
) -> list[dict[str, Any]]:
    """A section's sentences as data. `parts` keep the exact text and marker positions (joined, they
    render to what render_segments() gives); each marker lists its sources' numbers and the claims cited
    there. Sources are numbered in order of first citation, exactly like render_segments()."""
    sentences: list[dict[str, Any]] = []
    pending = ""  # whitespace-only text between sentences, carried into the next sentence
    for sentence in split_sentences(segments):
        parts: list[dict[str, Any]] = []

        def add_text(text: str, parts: list[dict[str, Any]] = parts) -> None:
            if not text:
                return
            if parts and "text" in parts[-1]:
                parts[-1]["text"] += text
            else:
                parts.append({"text": text})

        add_text(pending)
        pending = ""
        for seg in sentence:
            if isinstance(seg, str):
                add_text(seg)
                continue
            groups: dict[int, list[str]] = {}
            for claim_id in seg.ids:
                source_id = claims[claim_id].source_id
                if source_id not in numbers:
                    numbers[source_id] = len(numbers) + 1
                groups.setdefault(numbers[source_id], []).append(str(claim_id))
            parts.append({"cite": [{"n": n, "claim_ids": ids} for n, ids in groups.items()]})
        prose = "".join(p.get("text", "") for p in parts)
        if not any("cite" in p for p in parts) and not prose.strip():
            pending = prose
            continue
        cited_ns = list(dict.fromkeys(c["n"] for p in parts if "cite" in p for c in p["cite"]))
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(prose.split()))
        sentences.append({"text": text, "citations": cited_ns, "parts": parts})
    if pending and sentences:
        sentences[-1]["parts"].append({"text": pending})
    return sentences


def _render_parts(sentences: Sequence[Mapping[str, Any]]) -> str:
    out: list[str] = []
    for sentence in sentences:
        for part in sentence["parts"]:
            out.append(part["text"] if "text" in part else "".join(f"[{c['n']}]" for c in part["cite"]))
    return "".join(out).strip()


def render_markdown(question: str, structure: Mapping[str, Any]) -> str:
    """The report's Markdown, from its structure (reports.structure). assemble_report() uses this, so the
    stored Markdown and the structure always agree."""
    lines = ["# Research report", "", f"**Question:** {question}", ""]
    if structure.get("premise_note"):
        lines += [f"_{structure['premise_note']}_", ""]
    for section in structure["sections"]:
        lines += [f"## {section['ordinal']}. {section['heading']}", ""]
        if section.get("note"):
            lines += [f"_{section['note']}_", ""]
        body = _render_parts(section["sentences"])
        if body:
            lines += [body, ""]

    lines += ["## References", ""]
    if not structure["references"]:
        lines.append("No sources were cited.")
    for ref in structure["references"]:
        retrieved = f" (retrieved {ref['retrieved_at']})" if ref.get("retrieved_at") else ""
        lines.append(f"- [{ref['n']}] {ref['title']} — <{ref['url']}>{retrieved}")

    lines += ["", "## Evidence", ""]
    evidence = sorted(
        ((claim, int(n)) for n, items in structure["claims_by_n"].items() for claim in items),
        key=lambda item: item[0]["ordinal"],
    )
    if not evidence:
        lines.append("No claims were cited.")
    for claim, n in evidence:
        lines.append(f"- **Claim {claim['ordinal']}** [{n}]: {claim['text']}")
        lines += [f"  > {line}" for line in claim["quote"].split("\n")]
    return "\n".join(lines).rstrip() + "\n"


def assemble_report(
    question: str,
    sections: Sequence[SectionContent],
    claims: Mapping[uuid.UUID, ClaimRef],
    sources: Mapping[uuid.UUID, SourceRef],
    *,
    premise_note: str | None = None,
) -> AssembledReport:
    numbers: dict[uuid.UUID, int] = {}
    cited: list[uuid.UUID] = []
    section_data: list[dict[str, Any]] = []
    for i, section in enumerate(sections, start=1):
        for seg in section.segments:
            if isinstance(seg, Cite):
                cited.extend(c for c in seg.ids if c not in cited)
        section_data.append(
            {
                "ordinal": i,
                "heading": section.heading,
                "kind": section.kind,
                "note": section.note,
                "sentences": _section_sentences(section.segments, claims, numbers),
            }
        )
    ordered = sorted(numbers.items(), key=lambda item: item[1])
    references = []
    for source_id, n in ordered:
        ref = sources[source_id]
        references.append(
            {
                "n": n,
                "title": ref.title,
                "url": ref.url,
                "retrieved_at": ref.retrieved.isoformat() if ref.retrieved is not None else None,
                "reused": ref.reused,
            }
        )
    claims_by_n: dict[str, list[dict[str, Any]]] = {}
    for claim_id in sorted(cited, key=lambda c: claims[c].ordinal):
        claim = claims[claim_id]
        claims_by_n.setdefault(str(numbers[claim.source_id]), []).append(
            {
                "claim_id": str(claim_id),
                "ordinal": claim.ordinal,
                "text": claim.text,
                "verdict": claim.verdict,
                "quote": claim.quote,
            }
        )
    structure: dict[str, Any] = {
        "premise_note": premise_note or None,
        "sections": section_data,
        "references": references,
        "claims_by_n": claims_by_n,
    }

    citations = []
    for source_id, n in ordered:
        claim_ids = [c for c in cited if claims[c].source_id == source_id]
        chunk_ids = list(dict.fromkeys(claims[c].chunk_id for c in claim_ids))
        citations.append(
            {
                "n": n,
                "source_id": str(source_id),
                "chunk_ids": [str(c) for c in chunk_ids],
                "claim_ids": [str(c) for c in claim_ids],
            }
        )
    return AssembledReport(render_markdown(question, structure), citations, cited, structure)


# -- speculation guard -----------------------------------------------------------------------


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whole-word, case-insensitive match; spaces in the phrase match any whitespace."""
    words = [re.escape(w) for w in phrase.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(words) + r"(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class SpeculationReplacement:
    original: str  # the sentence as written, markers shown with their labels
    replacement: str
    phrases: list[str]  # speculative phrases the sentence had and its claims did not


def _render(parts: Sequence[Segment], label: Callable[[uuid.UUID], str]) -> str:
    return "".join(p if isinstance(p, str) else "".join(f"[{label(i)}]" for i in p.ids) for p in parts)


@dataclass(frozen=True)
class SentenceReplacement:
    kind: str  # "speculation" or "content"
    original: str  # the sentence as written, markers shown with their labels
    replacement: str
    phrases: list[str]  # speculation: phrases the sentence had and its claims did not
    words: list[str]  # content: stemmed words no cited claim contains
    numbers: list[str]  # content: numbers no cited claim contains
    deduplicated: bool = False  # the replacement repeated a sentence already in the section: dropped


def _sentence_key(parts: Sequence[Segment], label: Callable[[uuid.UUID], str]) -> str:
    return " ".join(_render(parts, label).split())


def check_sentences(
    segments: Sequence[Segment],
    claim_texts: Mapping[uuid.UUID, str],
    *,
    phrases: Sequence[str],
    ignored: Sequence[str] | None,
    label: Callable[[uuid.UUID], str],
) -> tuple[list[Segment], list[SentenceReplacement], int]:
    """Check every cited sentence against the claims it cites; a sentence that fails is replaced by those
    claims' own texts, each followed by its marker. First the speculation guard (a speculative phrase none
    of the cited claims contains), then, unless `ignored` is None, the content check (faithfulness: every
    content word and number must be in the cited claims). A replacement that exactly repeats a sentence
    already in the section (whitespace normalized, markers included) is dropped and flagged `deduplicated`:
    several sentences citing one claim would otherwise all become that claim's text. Returns the segments,
    the replacements and the number of cited sentences checked."""
    patterns = [(p, phrase_pattern(p)) for p in phrases]
    out: list[Segment] = []
    replaced: list[SentenceReplacement] = []
    seen: set[str] = set()
    checked = 0
    for sentence in split_sentences(segments):
        text = "".join(p for p in sentence if isinstance(p, str))
        cited = list(dict.fromkeys(i for p in sentence if isinstance(p, Cite) for i in p.ids))
        if not cited:  # nothing to fall back to (uncited sentences were already deleted)
            out.extend(sentence)
            seen.add(_sentence_key(sentence, label))
            continue
        checked += 1
        claims = [claim_texts[c] for c in cited]
        uncovered = [
            phrase
            for phrase, pattern in patterns
            if pattern.search(text) and not any(pattern.search(claim) for claim in claims)
        ]
        gaps = sentence_gaps(text, claims, ignored) if ignored is not None and not uncovered else None
        if not uncovered and not gaps:
            out.extend(sentence)
            seen.add(_sentence_key(sentence, label))
            continue
        leading = text[: len(text) - len(text.lstrip())]
        new: list[Segment] = [leading]
        for n, claim_id in enumerate(cited):
            claim = claim_texts[claim_id].strip().rstrip(".!?").rstrip()
            new += [(" " if n else "") + claim + " ", Cite((claim_id,)), "."]
        key = _sentence_key(new, label)
        duplicate = key in seen
        seen.add(key)
        replaced.append(
            SentenceReplacement(
                kind="speculation" if uncovered else "content",
                original=_render(sentence, label).strip(),
                replacement=_render(new, label).strip(),
                phrases=uncovered,
                words=gaps.words if gaps else [],
                numbers=gaps.numbers if gaps else [],
                deduplicated=duplicate,
            )
        )
        if not duplicate:
            out.extend(new)
    return _merge_adjacent(out), replaced, checked


def replace_speculation(
    segments: Sequence[Segment],
    claim_texts: Mapping[uuid.UUID, str],
    phrases: Sequence[str],
    label: Callable[[uuid.UUID], str],
) -> tuple[list[Segment], list[SpeculationReplacement]]:
    """The speculation guard alone (check_sentences without the content check)."""
    out, replaced, _ = check_sentences(segments, claim_texts, phrases=phrases, ignored=None, label=label)
    return out, [SpeculationReplacement(r.original, r.replacement, r.phrases) for r in replaced]
