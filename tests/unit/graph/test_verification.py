import uuid

import pytest

from research_assistant.db.models import Verdict
from research_assistant.graph.planning import filter_new_queries, finalize_plan, jaccard, query_tokens
from research_assistant.graph.verification import (
    QUOTE_NOT_FOUND,
    dedupe_claims,
    final_verdict,
    locate_quote,
    most_similar_pairs,
)
from research_assistant.llm.schemas import ResearchPlan

CHUNK = (
    "HNSW is a multilayer graph index;\nit has better query   speed-to-recall than IVFFlat.\n\n"
    "IVFFlat builds faster."
)

# -- quote location --------------------------------------------------------------------------


def test_exact_match() -> None:
    m = locate_quote(CHUNK, "IVFFlat builds faster.")
    assert m is not None and m.how == "exact" and CHUNK[m.start : m.end] == "IVFFlat builds faster."


def test_whitespace_normalized_match_maps_to_real_offsets() -> None:
    quote = "graph index; it has better query speed-to-recall"
    m = locate_quote(CHUNK, quote)
    assert m is not None and m.how == "whitespace"
    found = CHUNK[m.start : m.end]
    assert found == "graph index;\nit has better query   speed-to-recall"
    assert " ".join(found.split()) == quote


def test_whitespace_match_across_paragraph_break() -> None:
    m = locate_quote(CHUNK, "than IVFFlat. IVFFlat builds")
    assert m is not None and CHUNK[m.start : m.end] == "than IVFFlat.\n\nIVFFlat builds"


@pytest.mark.parametrize(
    "quote", ["", "   ", "IVFFlat builds slower.", "graph  index ;", "hnsw is a multilayer"]
)
def test_not_found(quote: str) -> None:
    assert locate_quote(CHUNK, quote) is None


def test_surrounding_whitespace_in_quote_is_ignored() -> None:
    m = locate_quote(CHUNK, "  IVFFlat builds faster.\n")
    assert m is not None and m.how == "exact"


# -- verdict mapping -------------------------------------------------------------------------

FOUND = locate_quote(CHUNK, "IVFFlat builds faster.")


@pytest.mark.parametrize(
    ("verdict", "found", "expected"),
    [
        (Verdict.SUPPORTED, True, (Verdict.SUPPORTED, None)),
        (Verdict.PARTIAL, True, (Verdict.PARTIAL, None)),
        (Verdict.SUPPORTED, False, (Verdict.UNSUPPORTED, QUOTE_NOT_FOUND)),
        (Verdict.PARTIAL, False, (Verdict.UNSUPPORTED, QUOTE_NOT_FOUND)),
        (Verdict.UNSUPPORTED, False, (Verdict.UNSUPPORTED, None)),
        (Verdict.UNSUPPORTED, True, (Verdict.UNSUPPORTED, None)),
        (Verdict.CONTRADICTED, True, (Verdict.CONTRADICTED, None)),
        (Verdict.CONTRADICTED, False, (Verdict.CONTRADICTED, None)),
    ],
)
def test_final_verdict(verdict: Verdict, found: bool, expected: tuple[Verdict, str | None]) -> None:
    assert final_verdict(verdict, FOUND if found else None) == expected


# -- claim dedupe ----------------------------------------------------------------------------

IDS = [uuid.uuid4() for _ in range(4)]


def test_dedupe_drops_later_near_duplicates() -> None:
    vectors = [[1.0, 0.0, 0.0], [0.99, 0.1, 0.0], [0.0, 1.0, 0.0], [0.0, 0.95, 0.31]]
    kept, dropped = dedupe_claims(IDS, vectors, 0.9)
    assert kept == [IDS[0], IDS[2]]
    assert [(d.claim_id, d.duplicate_of) for d in dropped] == [(IDS[1], IDS[0]), (IDS[3], IDS[2])]
    assert all(d.similarity >= 0.9 for d in dropped)


def test_dedupe_threshold_is_inclusive_and_order_matters() -> None:
    vectors = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    kept, dropped = dedupe_claims(IDS[:3], vectors, 1.0)
    assert kept == IDS[:2] and dropped[0].duplicate_of == IDS[1]


def test_most_similar_pairs() -> None:
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    pairs = most_similar_pairs(IDS[:3], vectors, 2)
    assert [(a, b) for a, b, _ in pairs] == [(IDS[0], IDS[2]), (IDS[1], IDS[2])]


# -- query dedupe (Jaccard) ------------------------------------------------------------------


def test_tokens_and_jaccard() -> None:
    assert query_tokens("Chandrayaan-2 lander FAILURE") == {"chandrayaan", "2", "lander", "failure"}
    a = query_tokens("HNSW IVFFlat memory usage pgvector")
    b = query_tokens("HNSW vs IVFFlat memory pgvector")
    assert jaccard(a, b) == pytest.approx(4 / 6)


def test_plan_drops_near_duplicate_queries_from_live_run() -> None:
    # the queries of run 8e206510's second sub-question, which only reorder the same words
    plan = ResearchPlan.model_validate(
        {
            "sub_questions": [
                {"question": "Performance?", "search_queries": ["HNSW vs IVFFlat performance pgvector"]},
                {
                    "question": "Memory usage?",
                    "search_queries": [
                        "HNSW IVFFlat memory usage pgvector",
                        "HNSW vs IVFFlat memory pgvector",
                        "pgvector HNSW IVFFlat memory comparison",
                    ],
                },
                {"question": "Accuracy?", "search_queries": ["ivfflat probes recall tuning"]},
            ]
        }
    )
    planned, dropped = finalize_plan(plan, 5, 0.6)
    assert [p.queries for p in planned] == [
        ["HNSW vs IVFFlat performance pgvector"],
        ["HNSW IVFFlat memory usage pgvector"],
        ["ivfflat probes recall tuning"],
    ]
    assert dropped == ["HNSW vs IVFFlat memory pgvector", "pgvector HNSW IVFFlat memory comparison"]


def test_filter_new_queries_against_earlier_and_each_other() -> None:
    kept, dropped = filter_new_queries(
        [
            "chandrayaan 2 lander failure causes",
            "vikram lander software error",
            "vikram lander software errors",
        ],
        ["Chandrayaan-2 lander failure causes"],
        0.6,
    )
    assert kept == ["vikram lander software error"]
    assert dropped == ["chandrayaan 2 lander failure causes", "vikram lander software errors"]
