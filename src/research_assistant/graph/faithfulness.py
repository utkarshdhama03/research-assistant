"""Sentence content check (pure): a written sentence may say only what the claims it cites say.

Content words are the sentence's words, lower-cased, minus stopwords and connectives (lists in config),
with a crude stem: the longest of "ing", "ed", "es", "ly", "s" is stripped when at least 3 letters
remain, then a final "e" ("lives", "lived", "living", "live" -> "liv"). Every content word must occur,
stemmed the same way, in the cited claims; every number (numbers.extract_numbers, identifiers excluded)
must occur among the claims' numbers. Synonyms are not recognized, by design: a reworded sentence is
replaced by its claims' own text.
"""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from research_assistant.graph.numbers import missing_numbers

_WORD = re.compile(r"[a-z]+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'"})
_SUFFIXES = ("ing", "ed", "es", "ly", "s")


def stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 3 else word


def _words(text: str) -> list[str]:
    text = text.lower().translate(_APOSTROPHES).replace("'s", " ")
    return _WORD.findall(text)


def content_words(text: str, ignored: Iterable[str]) -> list[str]:
    """Stemmed content words of text, in order, without repeats."""
    skip = {w.lower() for w in ignored}
    out: list[str] = []
    for word in _words(text):
        if word in skip:
            continue
        stemmed = stem(word)
        if stemmed not in out:
            out.append(stemmed)
    return out


@dataclass(frozen=True)
class Gaps:
    words: list[str]  # stemmed content words of the sentence that no cited claim contains
    numbers: list[str]

    def __bool__(self) -> bool:
        return bool(self.words or self.numbers)


def sentence_gaps(sentence: str, claims: Sequence[str], ignored: Iterable[str]) -> Gaps:
    ignored = list(ignored)
    claim_text = " ".join(claims)
    claim_words = set(content_words(claim_text, ()))
    words = [w for w in content_words(sentence, ignored) if w not in claim_words]
    return Gaps(words, missing_numbers(sentence, claim_text))
