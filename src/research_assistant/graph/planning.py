"""Turn a validated ResearchPlan into stored sub-questions, and filter near-duplicate queries (pure)."""

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from research_assistant.llm.schemas import ResearchPlan

_TOKEN = re.compile(r"\w+")


@dataclass(frozen=True)
class PlannedSubQuestion:
    question: str
    queries: list[str]
    premise_check: bool = False


def query_tokens(query: str) -> frozenset[str]:
    """Lower-cased word tokens; punctuation splits words (chandrayaan-2 -> chandrayaan, 2)."""
    return frozenset(_TOKEN.findall(query.casefold()))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def is_near_duplicate(query: str, earlier: Iterable[frozenset[str]], threshold: float) -> bool:
    tokens = query_tokens(query)
    return any(jaccard(tokens, other) >= threshold for other in earlier)


def filter_new_queries(
    candidates: Sequence[str], earlier: Sequence[str], threshold: float
) -> tuple[list[str], list[str]]:
    """Keep candidates that are not near-duplicates of `earlier` or of each other. Returns (kept, dropped)."""
    seen = [query_tokens(q) for q in earlier]
    kept: list[str] = []
    dropped: list[str] = []
    for query in candidates:
        if is_near_duplicate(query, seen, threshold):
            dropped.append(query)
            continue
        seen.append(query_tokens(query))
        kept.append(query)
    return kept, dropped


def finalize_plan(
    plan: ResearchPlan, max_sub_questions: int, threshold: float = 0.6
) -> tuple[list[PlannedSubQuestion], list[str]]:
    """Trim to max_sub_questions, then drop every query whose token set overlaps an earlier query
    anywhere in the plan with Jaccard >= threshold. A sub-question left without queries is searched
    by its own text. Returns the sub-questions and the dropped queries."""
    seen: list[frozenset[str]] = []
    dropped: list[str] = []
    out: list[PlannedSubQuestion] = []
    for sub in plan.sub_questions[:max_sub_questions]:
        queries: list[str] = []
        for query in sub.search_queries:
            if is_near_duplicate(query, seen, threshold):
                dropped.append(query)
                continue
            seen.append(query_tokens(query))
            queries.append(query)
        if not queries:
            queries = [sub.question]
            seen.append(query_tokens(sub.question))
        out.append(PlannedSubQuestion(sub.question, queries, sub.premise_check))
    return out, dropped
