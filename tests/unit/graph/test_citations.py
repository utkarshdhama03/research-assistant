import json
import uuid
from datetime import date
from typing import Any

import pytest

from research_assistant.graph.citations import (
    AssembledReport,
    CitationError,
    Cite,
    ClaimRef,
    SectionContent,
    SourceRef,
    _merge_adjacent,
    _render_parts,
    assemble_report,
    parse_section,
    remove_uncited,
    render_markdown,
    render_segments,
    replace_speculation,
    split_sentences,
)
from research_assistant.llm.aliases import AliasMap

CLAIMS = [uuid.uuid4() for _ in range(4)]
CHUNKS = [uuid.uuid4() for _ in range(4)]
SRC_A, SRC_B = uuid.uuid4(), uuid.uuid4()
# K1 (ordinal 0), K2 (ordinal 1) quote source A; K3 (ordinal 2), K4 (ordinal 3) quote source B
REFS = {
    CLAIMS[0]: ClaimRef(0, "Claim zero.", "quote zero", CHUNKS[0], SRC_A),
    CLAIMS[1]: ClaimRef(1, "Claim one.", "quote one", CHUNKS[1], SRC_A),
    CLAIMS[2]: ClaimRef(2, "Claim two.", "line one\nline two", CHUNKS[2], SRC_B),
    CLAIMS[3]: ClaimRef(3, "Claim three.", "quote three", CHUNKS[3], SRC_B),
}
SOURCES = {
    SRC_A: SourceRef("Source A", "https://a.example.org/x"),
    SRC_B: SourceRef("Source B", "https://b.example.org/"),
}


def aliases() -> AliasMap[uuid.UUID]:
    return AliasMap(CLAIMS, prefix="K")


def test_parse_resolves_markers() -> None:
    segments = parse_section("Fact one [K1]. Fact two [K3][K2]. Fact three [K4, K1].", aliases())
    assert segments == [
        "Fact one ",
        Cite((CLAIMS[0],)),
        ". Fact two ",
        Cite((CLAIMS[2], CLAIMS[1])),
        ". Fact three ",
        Cite((CLAIMS[3], CLAIMS[0])),
        ".",
    ]


@pytest.mark.parametrize("bad", ["[K9]", "[K0]", "[k1]", "[K01]", "[1]", "[C1]", "[K1, K9]", "[ K 1 ]"])
def test_unknown_or_malformed_labels_are_errors(bad: str) -> None:
    with pytest.raises(CitationError) as info:
        parse_section(f"A fact {bad}.", aliases())
    assert info.value.unknown


@pytest.mark.parametrize("text", ["See https://example.org/page [K1].", "Visit www.example.org [K1]."])
def test_urls_are_errors(text: str) -> None:
    with pytest.raises(CitationError) as info:
        parse_section(text, aliases())
    assert info.value.urls


def test_ordinary_brackets_are_left_alone() -> None:
    assert (
        parse_section("He wrote [sic] that it works [K1].", aliases())[0] == "He wrote [sic] that it works "
    )


# -- deleting uncited sentences --------------------------------------------------------------


def kept_and_deleted(text: str) -> tuple[str, list[str]]:
    cleaned = remove_uncited(parse_section(text, aliases()))
    rendered = "".join(s if isinstance(s, str) else "[*]" for s in cleaned.segments)
    return rendered.strip(), cleaned.deleted


def kept_and_trimmed(text: str) -> tuple[str, list[str]]:
    cleaned = remove_uncited(parse_section(text, aliases()))
    rendered = "".join(s if isinstance(s, str) else "[*]" for s in cleaned.segments)
    return rendered.strip(), cleaned.trimmed


@pytest.mark.parametrize(
    ("text", "kept", "deleted"),
    [
        ("Fact one [K1]. Fact two [K2].", "Fact one [*]. Fact two [*].", []),
        ("Fact one [K1]. No label here. Fact two [K2].", "Fact one [*]. Fact two [*].", ["No label here."]),
        ("Intro without a label. Fact [K1].", "Fact [*].", ["Intro without a label."]),
        ("Fact one. [K1] Fact two. [K2]", "Fact one. [*] Fact two. [*]", []),
        ("Postgres (e.g. pgvector) stores vectors [K1].", "Postgres (e.g. pgvector) stores vectors [*].", []),
        (
            "Fact [K1].\n\nA closing summary without labels.",
            "Fact [*].",
            ["A closing summary without labels."],
        ),
        ("Is it fast? Yes [K1]! Really fast.", "Yes [*]!", ["Is it fast?", "Really fast."]),
        ("Nothing cited at all. Still nothing.", "", ["Nothing cited at all.", "Still nothing."]),
    ],
)
def test_remove_uncited(text: str, kept: str, deleted: list[str]) -> None:
    assert kept_and_deleted(text) == (kept, deleted)


# -- assembly --------------------------------------------------------------------------------


def test_assemble_numbers_sources_and_builds_evidence_appendix() -> None:
    first = SectionContent("First", parse_section("B fact [K3]. A fact [K1]. Both [K2][K4].", aliases()))
    second = SectionContent("Second", parse_section("A again [K2].", aliases()))
    report = assemble_report("Q?", [first, second], REFS, SOURCES)
    assert "B fact [1]. A fact [2]. Both [2][1]." in report.markdown
    assert "A again [2]." in report.markdown
    assert report.citations == [
        {
            "n": 1,
            "source_id": str(SRC_B),
            "chunk_ids": [str(CHUNKS[2]), str(CHUNKS[3])],
            "claim_ids": [str(CLAIMS[2]), str(CLAIMS[3])],
        },
        {
            "n": 2,
            "source_id": str(SRC_A),
            "chunk_ids": [str(CHUNKS[0]), str(CHUNKS[1])],
            "claim_ids": [str(CLAIMS[0]), str(CLAIMS[1])],
        },
    ]
    references, evidence = report.markdown.split("## References")[1].split("## Evidence")
    assert (
        "- [1] Source B — <https://b.example.org/>\n- [2] Source A — <https://a.example.org/x>" in references
    )
    # appendix in claim-ordinal order, quotes verbatim (multi-line kept line by line)
    assert evidence.strip().splitlines() == [
        "- **Claim 0** [2]: Claim zero.",
        "  > quote zero",
        "- **Claim 1** [2]: Claim one.",
        "  > quote one",
        "- **Claim 2** [1]: Claim two.",
        "  > line one",
        "  > line two",
        "- **Claim 3** [1]: Claim three.",
        "  > quote three",
    ]
    assert report.cited_claims == [CLAIMS[2], CLAIMS[0], CLAIMS[1], CLAIMS[3]]


def test_uncited_claims_are_not_in_the_appendix() -> None:
    report = assemble_report(
        "Q?", [SectionContent("S", parse_section("Fact [K2].", aliases()))], REFS, SOURCES
    )
    assert "Claim 1" in report.markdown and "Claim 0" not in report.markdown


def test_notes_and_empty_report() -> None:
    sections = [SectionContent("A", note="There is insufficient verified evidence."), SectionContent("B")]
    report = assemble_report("Q?", sections, REFS, SOURCES)
    assert "## 1. A\n\n_There is insufficient verified evidence._" in report.markdown
    assert "No sources were cited." in report.markdown and "No claims were cited." in report.markdown
    assert report.citations == []


# -- trailing text after the last marker -----------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kept", "trimmed"),
    [
        (
            "A fact [K2], which also influences its memory footprint.",
            "A fact [*].",
            [", which also influences its memory footprint"],
        ),
        (
            "Glitch [K1] [K3], potentially leading to errors. Next fact [K2], ensuring reliability.",
            "Glitch [*] [*]. Next fact [*].",
            [", potentially leading to errors", ", ensuring reliability"],
        ),
        ("Fact [K1] in some detail [K2].", "Fact [*] in some detail [*].", []),
        ("Fact [K1].", "Fact [*].", []),
        ("Fact. [K1] Next fact [K2]!", "Fact. [*] Next fact [*]!", []),
        ("Is it so [K1], or not?", "Is it so [*]?", [", or not"]),
        ("Ends without a stop [K1] and then more", "Ends without a stop [*]", ["and then more"]),
        ("Quoted fact [K1] (see the table).", "Quoted fact [*].", ["(see the table)"]),
    ],
)
def test_text_after_last_marker_is_trimmed(text: str, kept: str, trimmed: list[str]) -> None:
    assert kept_and_trimmed(text) == (kept, trimmed)


def test_trimming_and_deletion_together() -> None:
    cleaned = remove_uncited(
        parse_section("Fact [K1], likely important. No label here. Other [K2].", aliases())
    )
    assert cleaned.deleted == ["No label here."]
    assert cleaned.trimmed == [", likely important"]


# -- speculation guard -----------------------------------------------------------------------

PHRASES = ["may have", "likely", "could have", "contributing to", "suggests that"]
TEXTS = {
    CLAIMS[0]: "The lander had a higher vertical velocity component.",
    CLAIMS[1]: "The orbiter was omitted.",
    CLAIMS[2]: "Analysts said the glitch likely caused the crash.",
    CLAIMS[3]: "Fuel margins were increased.",
}


def label(claim_id: uuid.UUID) -> str:
    return aliases().alias(claim_id)


def guard(text: str) -> tuple[str, list[str], list[str]]:
    segments, replaced = replace_speculation(parse_section(text, aliases()), TEXTS, PHRASES, label)
    rendered = "".join(s if isinstance(s, str) else "".join(f"[{label(i)}]" for i in s.ids) for s in segments)
    return rendered, [r.original for r in replaced], [p for r in replaced for p in r.phrases]


def test_speculative_sentence_is_replaced_by_its_claims() -> None:
    text = "The lander had a higher velocity, which may have improved control [K1]. Fuel margins grew [K4]."
    rendered, originals, phrases = guard(text)
    assert rendered == "The lander had a higher vertical velocity component [K1]. Fuel margins grew [K4]."
    assert originals == ["The lander had a higher velocity, which may have improved control [K1]."]
    assert phrases == ["may have"]


def test_multiple_cited_claims_each_get_their_marker() -> None:
    rendered, _, _ = guard("Both changes could have reduced risk [K1][K2].")
    assert (
        rendered == "The lander had a higher vertical velocity component [K1]. The orbiter was omitted [K2]."
    )


def test_phrase_present_in_a_cited_claim_is_allowed() -> None:
    text = "The glitch likely caused the crash [K3]."
    assert guard(text) == (text, [], [])


def test_phrase_only_in_an_uncited_claim_does_not_help() -> None:
    rendered, originals, _ = guard("Higher velocity likely helped [K1].")
    assert rendered == "The lander had a higher vertical velocity component [K1]." and originals


@pytest.mark.parametrize(
    "text",
    [
        "The landing was unlikely to fail [K1].",  # "unlikely" is not the whole word "likely"
        "It may be faster [K1].",  # "may" alone is not "may have"
        "Fuel margins were increased [K4].",
    ],
)
def test_non_speculative_sentences_are_kept(text: str) -> None:
    assert guard(text) == (text, [], [])


@pytest.mark.parametrize(
    "text",
    [
        "Higher velocity, LIKELY helpful [K1].",
        "The change, Contributing  To success [K1].",
        "Data suggests that it\nworked [K1].",
    ],
)
def test_matching_is_case_insensitive_and_whitespace_tolerant(text: str) -> None:
    rendered, originals, _ = guard(text)
    assert rendered == "The lander had a higher vertical velocity component [K1]." and len(originals) == 1


def test_only_the_speculative_sentence_changes() -> None:
    text = "Fuel margins grew [K4]. The orbiter's absence possibly mattered [K2]. More fuel [K4]."
    rendered, _, _ = guard(text.replace("possibly", "likely"))
    assert rendered == "Fuel margins grew [K4]. The orbiter was omitted [K2]. More fuel [K4]."


def test_guard_after_trimming_splits_sentences_at_punctuation_segments() -> None:
    # remove_uncited leaves "." and " Next" as separate strings; the guard must still see two sentences
    cleaned = remove_uncited(parse_section("First fact [K4]. Velocity likely helped [K1].", aliases()))
    segments, replaced = replace_speculation(cleaned.segments, TEXTS, PHRASES, label)
    assert [r.original for r in replaced] == ["Velocity likely helped [K1]."]
    rendered = "".join(s if isinstance(s, str) else "".join(f"[{label(i)}]" for i in s.ids) for s in segments)
    assert rendered == "First fact [K4]. The lander had a higher vertical velocity component [K1]."


def test_split_sentences_joins_separate_string_segments() -> None:
    cite1, cite2 = Cite((CLAIMS[0],)), Cite((CLAIMS[1],))
    sentences = split_sentences(["First fact ", cite1, ".", " Next fact ", cite2, "."])
    assert len([s for s in sentences if any(isinstance(p, Cite) for p in s)]) == 2


def test_merge_adjacent_joins_strings_and_cites() -> None:
    cite1, cite2 = Cite((CLAIMS[0],)), Cite((CLAIMS[1],))
    assert _merge_adjacent(["a", "", "b", cite1, cite2, "c"]) == ["ab", Cite((CLAIMS[0], CLAIMS[1])), "c"]


# -- report structure ------------------------------------------------------------------------


def structured(texts: list[str], **kwargs: Any) -> AssembledReport:
    sections = [SectionContent(f"S{i}", parse_section(t, aliases())) for i, t in enumerate(texts)]
    return assemble_report("Q?", sections, REFS, SOURCES, **kwargs)


def test_structure_keeps_sentences_markers_and_claims() -> None:
    report = structured(["B fact [K3]. A fact [K1]. Both [K2][K4].", "A again [K2]."])
    first = report.structure["sections"][0]
    assert (first["ordinal"], first["heading"], first["kind"], first["note"]) == (1, "S0", "normal", None)
    assert [s["text"] for s in first["sentences"]] == ["B fact.", "A fact.", "Both."]
    assert [s["citations"] for s in first["sentences"]] == [[1], [2], [2, 1]]
    assert first["sentences"][2]["parts"] == [
        {"text": " Both "},
        {"cite": [{"n": 2, "claim_ids": [str(CLAIMS[1])]}, {"n": 1, "claim_ids": [str(CLAIMS[3])]}]},
        {"text": "."},
    ]
    assert report.structure["references"] == [
        {"n": 1, "title": "Source B", "url": "https://b.example.org/", "retrieved_at": None, "reused": False},
        {
            "n": 2,
            "title": "Source A",
            "url": "https://a.example.org/x",
            "retrieved_at": None,
            "reused": False,
        },
    ]
    assert report.structure["claims_by_n"]["1"] == [
        {
            "claim_id": str(CLAIMS[2]),
            "ordinal": 2,
            "text": "Claim two.",
            "verdict": "supported",
            "quote": "line one\nline two",
        },
        {
            "claim_id": str(CLAIMS[3]),
            "ordinal": 3,
            "text": "Claim three.",
            "verdict": "supported",
            "quote": "quote three",
        },
    ]


def test_mid_sentence_markers_stay_in_place() -> None:
    [sentence] = structured(["Fact [K1] in some detail [K3]."]).structure["sections"][0]["sentences"]
    assert [("cite" if "cite" in p else p["text"]) for p in sentence["parts"]] == [
        "Fact ",
        "cite",
        " in some detail ",
        "cite",
        ".",
    ]
    assert sentence["citations"] == [1, 2] and sentence["text"] == "Fact in some detail."


def test_structure_notes_kinds_premise_and_reused_reference() -> None:
    sources = {
        SRC_A: SourceRef("Source A", "https://a.example.org/x", date(2026, 11, 20), reused=True),
        SRC_B: SOURCES[SRC_B],
    }
    sections = [
        SectionContent("P", note="No verified source supports the premise.", kind="insufficient"),
        SectionContent("C", note="Covered above.", kind="covered"),
        SectionContent(
            "W", parse_section("A fact [K1].", aliases()), note="No own sources.", kind="no_sources"
        ),
        SectionContent("F", note="Could not be written.", kind="failed"),
    ]
    report = assemble_report("Q?", sections, REFS, sources, premise_note="No verified source supports it.")
    s = report.structure
    assert s["premise_note"] == "No verified source supports it."
    assert [(x["kind"], x["note"], len(x["sentences"])) for x in s["sections"]] == [
        ("insufficient", "No verified source supports the premise.", 0),
        ("covered", "Covered above.", 0),
        ("no_sources", "No own sources.", 1),
        ("failed", "Could not be written.", 0),
    ]
    assert s["references"] == [
        {
            "n": 1,
            "title": "Source A",
            "url": "https://a.example.org/x",
            "retrieved_at": "2026-11-20",
            "reused": True,
        }
    ]
    assert "(retrieved 2026-11-20)" in report.markdown


AGREEMENT_CASES = [
    ["B fact [K3]. A fact [K1]. Both [K2][K4].", "A again [K2]."],
    ["Fact [K1] in some detail [K2]."],
    ["Fact. [K1] Next fact [K2]!"],
    ["Fact one [K1].\n\nFact two [K3].   Fact three [K4]."],
    ["  Leading space [K1]. Trailing space [K2].  "],
    ["He wrote [sic] that it works [K1]. Is it fast? Yes [K3]!"],
    [],
]


@pytest.mark.parametrize("texts", AGREEMENT_CASES)
def test_markdown_rendered_from_the_stored_structure_is_the_markdown(texts: list[str]) -> None:
    report = structured(texts, premise_note="A note." if texts else None)
    stored = json.loads(json.dumps(report.structure))  # as it comes back from JSONB
    assert render_markdown("Q?", stored) == report.markdown


@pytest.mark.parametrize("texts", AGREEMENT_CASES)
def test_section_bodies_match_the_independent_renderer(texts: list[str]) -> None:
    sections = [parse_section(t, aliases()) for t in texts]
    numbers: dict[uuid.UUID, int] = {}
    expected = [render_segments(segments, REFS, numbers) for segments in sections]
    report = structured(texts)
    got = [_render_parts(s["sentences"]) for s in report.structure["sections"]]
    assert got == expected
