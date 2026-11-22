"""Sentence content check: a written sentence may say only what the claims it cites say."""

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from research_assistant.config import Settings
from research_assistant.graph.citations import Cite, check_sentences
from research_assistant.graph.faithfulness import content_words, sentence_gaps, stem

SETTINGS = Settings(_env_file=None, contact_url="mailto:test@example.org")
IGNORED = [*SETTINGS.sentence_stopwords, *SETTINGS.sentence_connectives]
CASES: list[dict[str, Any]] = json.loads(
    (Path(__file__).parents[2] / "fixtures" / "sentence_check_cases.json").read_text(encoding="utf-8")
)["cases"]
RTF_CLAIM = (
    "The Eiffel Tower in Lyon was bought by RTF, the national society of French radio and television, "
    "for 15 million francs."
)


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["sentence"][:40])
def test_live_q16_sentences(case: dict[str, Any]) -> None:
    gaps = sentence_gaps(case["sentence"], case["claims"], IGNORED)
    assert bool(gaps) is case["replace"]
    assert (gaps.words, gaps.numbers) == (case["unmatched_words"], case["unmatched_numbers"])


def test_live_q16_named_cases() -> None:
    by_start = {c["sentence"][:30]: c for c in CASES}
    exposition = next(c for c in CASES if "Exposition Universelle" in c["sentence"])
    assert {"significant", "global", "gather"} <= set(exposition["unmatched_words"])
    initiative = next(c for c in CASES if "broader cultural initiative" in c["sentence"])
    assert "initiativ" in initiative["unmatched_words"] and initiative["unmatched_numbers"] == ["1999"]
    rtf = next(c for c in CASES if "bought by RTF" in c["sentence"])
    assert rtf["unmatched_numbers"] == ["1999"] and rtf["unmatched_words"] == []
    assert sum(c["replace"] for c in CASES) == 6 and len(by_start) == len(CASES)


def test_faithful_reordering_passes() -> None:
    sentence = (
        "For 15 million francs, RTF, the national society of French radio and television, "
        "bought the Eiffel Tower in Lyon."
    )
    assert not sentence_gaps(sentence, [RTF_CLAIM], IGNORED)


def test_connectives_and_word_forms_pass() -> None:
    claims = ["The tower closed to the public in 1953.", "The tower serves as a radio antenna."]
    assert not sentence_gaps("However, the tower also closes to the public in 1953.", claims, IGNORED)
    assert not sentence_gaps("Both towers served as radio antennas.", claims, IGNORED)
    assert sentence_gaps("The tower therefore closed in 1953.", claims, IGNORED).words == ["therefor"]


def test_negation_and_quantity_words_are_content() -> None:
    claims = ["The tower is open to the public."]
    assert sentence_gaps("The tower is not open to the public.", claims, IGNORED).words == ["not"]
    assert sentence_gaps("Only the tower is open to the public.", claims, IGNORED).words == ["only"]


def test_numbers_must_be_in_the_claims() -> None:
    claims = ["The tower closed to the public in 1953."]
    assert sentence_gaps("The tower closed to the public in 1999.", claims, IGNORED).numbers == ["1999"]
    assert not sentence_gaps("In 1953 the tower closed to the public.", claims, IGNORED)


def test_stemming() -> None:
    assert {stem(w) for w in ("lives", "lived", "living", "live")} == {"liv"}
    assert stem("is") == "is" and stem("its") == "its"
    assert content_words("The towers' antennas", IGNORED) == ["tower", "antenna"]


def test_check_sentences_replaces_and_reports() -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    claims = {a: RTF_CLAIM, b: "The tower closed to the public in 1953."}
    segments = [
        "The tower closed to the public in 1953 ",
        Cite((b,)),
        ". In 1999, RTF bought the tower for 15 million francs ",
        Cite((a,)),
        ".",
    ]
    out, replaced, checked = check_sentences(
        segments, claims, phrases=SETTINGS.speculation_phrases, ignored=IGNORED, label=lambda i: "K1"
    )
    assert checked == 2 and len(replaced) == 1
    [r] = replaced
    assert (r.kind, r.numbers) == ("content", ["1999"])
    text = "".join(s for s in out if isinstance(s, str))
    assert "In 1999" not in text and "for 15 million francs" in text
    assert [s.ids for s in out if isinstance(s, Cite)] == [(b,), (a,)]


VETO_CLAIM = (
    "The plan to relocate the Eiffel Tower to Lyon was allegedly vetoed by the company operating the tower "
    "due to fears that the French government might refuse permission for the tower to be restored in its "
    "original location"
)


def test_replacements_that_repeat_a_sentence_are_dropped() -> None:
    # Final eval q16, section 2: the first sentence is the claim itself; the other three each cite the same
    # claim, fail the content check, and became the claim's text, so the section said it four times.
    a, other = uuid.uuid4(), uuid.uuid4()
    segments: list[str | Cite] = [VETO_CLAIM + " ", Cite((a,))]
    for original in [
        ". The purpose of moving the Eiffel Tower to Lyon in 1999 was to create a temporary replica as part "
        "of a cultural event ",
        ". This decision reflected concerns about the feasibility of relocating the actual structure and the "
        "potential complications involved in obtaining governmental approval for its restoration ",
        ". The move to Lyon was intended to showcase the iconic landmark in a new setting, but the company's "
        "reluctance highlighted the challenges of such a relocation ",
    ]:
        segments += [original, Cite((a,))]
    segments += [". The tower closed in 1953 ", Cite((other,)), "."]
    out, replaced, checked = check_sentences(
        segments,
        {a: VETO_CLAIM + ".", other: "The tower closed in 1953."},
        phrases=SETTINGS.speculation_phrases,
        ignored=IGNORED,
        label=lambda i: "K1" if i == a else "K2",
    )
    assert checked == 5 and [r.kind for r in replaced] == ["content"] * 3
    assert [r.deduplicated for r in replaced] == [True, True, True]  # still reported, not written
    text = "".join(s if isinstance(s, str) else "[K]" for s in out)
    assert text.count("allegedly vetoed") == 1
    assert text == VETO_CLAIM + " [K]. The tower closed in 1953 [K]."
    assert [s.ids for s in out if isinstance(s, Cite)] == [(a,), (other,)]


def test_first_replacement_is_kept_and_whitespace_is_normalized() -> None:
    a = uuid.uuid4()
    claim = "The tower closed to the public in 1953."
    segments = [
        "The tower shut in 1999 ",  # replaced: the first copy of the claim, kept
        Cite((a,)),
        ". It closed in 1999 ",  # replaced again: a repeat, dropped
        Cite((a,)),
        ". The tower\nclosed to the  public in 1953 ",  # passes as written: kept, not a replacement
        Cite((a,)),
        ". The tower closed to the public in 1954 ",  # repeats the first after whitespace normalization
        Cite((a,)),
        ".",
    ]
    out, replaced, _ = check_sentences(
        segments, {a: claim}, phrases=[], ignored=IGNORED, label=lambda i: "K1"
    )
    assert [r.deduplicated for r in replaced] == [False, True, True]
    assert "".join(s for s in out if isinstance(s, str)) == (
        "The tower closed to the public in 1953 . The tower\nclosed to the  public in 1953 ."
    )


def test_speculation_is_reported_as_speculation() -> None:
    a = uuid.uuid4()
    segments = ["The tower likely closed to the public in 1953 ", Cite((a,)), "."]
    _, replaced, _ = check_sentences(
        segments,
        {a: "The tower closed to the public in 1953."},
        phrases=SETTINGS.speculation_phrases,
        ignored=IGNORED,
        label=lambda i: "K1",
    )
    assert [r.kind for r in replaced] == ["speculation"] and replaced[0].phrases == ["likely"]
