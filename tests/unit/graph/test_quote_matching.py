"""Quote location beyond exact matching: list-normalized and bounded fuzzy, on real cases."""

import json
from pathlib import Path
from typing import Any

import pytest

from research_assistant.graph.numbers import missing_numbers
from research_assistant.graph.verification import FUZZY_MAX_EDITS, locate_quote

FIXTURE = json.loads(
    (Path(__file__).parents[2] / "fixtures" / "vacuum_quote_cases.json").read_text(encoding="utf-8")
)
CASES: list[dict[str, Any]] = FIXTURE["cases"]
LIST_TEXT: str = FIXTURE["excerpts"]["list"]["text"]
TYPO_TEXT: str = FIXTURE["excerpts"]["typo"]["text"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c['run']}-claim{c['claim_ordinal']}")
def test_live_vacuum_cases(case: dict[str, Any]) -> None:
    text = FIXTURE["excerpts"][case["excerpt"]]["text"]
    match = locate_quote(text, case["model_quote"])
    assert (match.how if match else None) == case["expected_method"]
    if match is not None:
        stored = text[match.start : match.end]
        assert stored in text  # the chunk's own text is stored, never the model's wording
        assert not stored[0].isspace() and not stored[-1].isspace()


def test_live_case_counts() -> None:
    methods = [c["expected_method"] for c in CASES]
    # 11 bullet cases: 8 list, 1 fuzzy (two items lower-cased), 2 not contiguous; 2 corrected typos: fuzzy.
    assert (methods.count("list"), methods.count("fuzzy"), methods.count(None), len(CASES)) == (8, 3, 2, 13)


def test_list_match_keeps_the_source_text_with_its_markers() -> None:
    match = locate_quote(LIST_TEXT, "VACUUM: Removes dead tuples, Updates visibility map")
    assert match is not None and match.how == "list"
    assert LIST_TEXT[match.start : match.end] == "VACUUM:\n- Removes dead tuples\n- Updates visibility map"


def test_typo_match_stores_the_source_spelling() -> None:
    quote = (
        "This type of vacuuming rewrites the entire table "
        "which compact it and reclaims the disk space for OS."
    )
    match = locate_quote(TYPO_TEXT, quote)
    assert match is not None and match.how == "fuzzy"
    assert "recalims" in TYPO_TEXT[match.start : match.end]  # the page's typo, not the model's correction


@pytest.mark.parametrize(
    "text",
    [
        "Steps:\n• Stop the server\n• Copy the data directory\nDone.",
        "Steps:\n* Stop the server\n* Copy the data directory\nDone.",
        "Steps:\n· Stop the server\n· Copy the data directory\nDone.",
        "Steps:\n1. Stop the server\n2. Copy the data directory\nDone.",
        "Steps:\n1) Stop the server\n2) Copy the data directory\nDone.",
        "Steps:\n  - Stop the server\n  - Copy the data directory\nDone.",
    ],
)
def test_list_marker_styles(text: str) -> None:
    for quote in (
        "Steps: Stop the server, Copy the data directory",
        "Stop the server; Copy the data directory",
    ):
        match = locate_quote(text, quote)
        assert match is not None and match.how == "list", (text, quote)


def test_paraphrase_beyond_the_bound_fails() -> None:
    paraphrase = (
        "This kind of vacuuming rewrites the whole table which compacts it and reclaims disk space for OS."
    )
    assert locate_quote(TYPO_TEXT, paraphrase) is None
    # One character beyond the bound: the typo (2 edits) plus "OS" -> "the OS" (4 more).
    assert (
        locate_quote(
            TYPO_TEXT, "rewrites the entire table which compact it and reclaims the disk space for the OS"
        )
        is None
    )


def test_short_quotes_are_never_fuzzy() -> None:
    text = "The checker recalims space."
    assert locate_quote(text, "recalims space") is not None  # exact
    assert locate_quote(text, "reclaims space") is None  # 14 chars < FUZZY_MIN_CHARS


def test_share_bound_on_mid_length_quotes() -> None:
    # 45 characters: 3% is 1.35, so only one edit is allowed even though FUZZY_MAX_EDITS is 2.
    text = "Autovacuum runs in the background on each table."
    assert FUZZY_MAX_EDITS == 2
    assert locate_quote(text, "Autovacuum runs in the backgrund on each table.") is not None  # 1 edit
    assert locate_quote(text, "Autovacum runs in the backgrund on each table.") is None  # 2 edits


def test_fuzzy_match_with_a_different_number_still_fails_the_number_check() -> None:
    text = "Field tests showed that the pumps reached a COP between 1.8 and 2.8 during the coldest weeks."
    quote = "the pumps reached a COP between 1.8 and 2.9 during the coldest weeks"  # one digit changed
    match = locate_quote(text, quote)
    assert match is not None and match.how == "fuzzy"
    stored = text[match.start : match.end]
    assert "2.8" in stored
    assert missing_numbers("The pumps reached a COP between 1.8 and 2.9.", stored) == ["2.9"]
