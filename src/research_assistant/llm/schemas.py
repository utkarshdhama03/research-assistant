"""Structured-output schemas. Bounds live in the schema, so Pydantic enforces them on every reply."""

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, create_model, field_validator

MIN_SUB_QUESTIONS = 3
MAX_SUB_QUESTIONS = 5
MIN_QUERIES = 1
MAX_QUERIES = 3

SearchQuery = Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=200)]


def _require_unique(values: list[str], what: str) -> list[str]:
    seen: set[str] = set()
    for value in values:
        key = " ".join(value.casefold().split())
        if key in seen:
            raise ValueError(f"duplicate {what}: {value!r}")
        seen.add(key)
    return values


class SubQuestionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=300)]
    search_queries: list[SearchQuery] = Field(min_length=MIN_QUERIES, max_length=MAX_QUERIES)
    # True only for a first sub-question that checks a specific fact the question asserts.
    premise_check: bool = False

    @field_validator("search_queries")
    @classmethod
    def _unique_queries(cls, value: list[str]) -> list[str]:
        return _require_unique(value, "search query")


class ResearchPlan(BaseModel):
    """plan role: question -> 3-5 sub-questions, each with 1-3 search queries."""

    model_config = ConfigDict(extra="forbid")

    sub_questions: list[SubQuestionPlan] = Field(min_length=MIN_SUB_QUESTIONS, max_length=MAX_SUB_QUESTIONS)

    @field_validator("sub_questions")
    @classmethod
    def _unique_sub_questions(cls, value: list[SubQuestionPlan]) -> list[SubQuestionPlan]:
        _require_unique([s.question for s in value], "sub-question")
        return value

    @field_validator("sub_questions")
    @classmethod
    def _premise_check_first_only(cls, value: list[SubQuestionPlan]) -> list[SubQuestionPlan]:
        flagged = [i for i, s in enumerate(value) if s.premise_check]
        if len(flagged) > 1 or flagged not in ([], [0]):
            raise ValueError("premise_check may be true for the first sub-question only")
        return value


class SourceSelection(BaseModel):
    """select role: which search results (by alias) to read. Built per call by selection_schema()."""

    model_config = ConfigDict(extra="forbid")

    selected: list[str]


def selection_schema(aliases: Sequence[str], max_picks: int) -> type[SourceSelection]:
    """A SourceSelection subclass whose items are exactly `aliases` (a JSON-schema enum, so Ollama's
    grammar can only produce valid labels) with 1..max_picks unique picks."""
    if not aliases:
        raise ValueError("selection_schema needs at least one alias")
    if not 1 <= max_picks <= len(aliases):
        raise ValueError(f"max_picks must be in 1..{len(aliases)}, got {max_picks}")
    label = Literal.__getitem__(tuple(aliases))  # dynamic Literal["R1", "R2", ...]

    def unique(value: list[str]) -> list[str]:
        return _require_unique(value, "pick")

    model: type[SourceSelection] = create_model(
        "SourceSelection",
        __base__=SourceSelection,
        selected=(list[label], Field(min_length=1, max_length=max_picks)),  # type: ignore[valid-type]
        __validators__={"unique_picks": field_validator("selected")(unique)},
    )
    return model


ClaimText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=10, max_length=400)]


class AnalyzedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    evidence: list[str]


class ClaimSet(BaseModel):
    """analyze role: claims with the evidence labels that support them. Built by claims_schema()."""

    model_config = ConfigDict(extra="forbid")

    claims: list[AnalyzedClaim]


def _label_type(aliases: Sequence[str]) -> Any:
    if not aliases:
        raise ValueError("at least one alias is required")
    return Literal.__getitem__(tuple(aliases))  # dynamic Literal["C1", "C2", ...]


def claims_schema(aliases: Sequence[str], max_claims: int) -> type[ClaimSet]:
    """0..max_claims claims; each one fact with 1-2 unique evidence labels from `aliases`."""
    label = _label_type(aliases)

    def unique(value: list[str]) -> list[str]:
        return _require_unique(value, "evidence label")

    claim: type[AnalyzedClaim] = create_model(
        "AnalyzedClaim",
        __base__=AnalyzedClaim,
        text=(ClaimText, ...),
        evidence=(list[label], Field(min_length=1, max_length=2)),  # type: ignore[valid-type]
        __validators__={"unique_evidence": field_validator("evidence")(unique)},
    )
    model: type[ClaimSet] = create_model(
        "ClaimSet",
        __base__=ClaimSet,
        claims=(list[claim], Field(min_length=0, max_length=max_claims)),
    )
    return model


VerdictLabel = Literal["supported", "partial", "unsupported", "contradicted"]


class Verification(BaseModel):
    """verify role: verdict for one claim against its cited passages. Built by verification_schema()."""

    model_config = ConfigDict(extra="forbid")

    verdict: VerdictLabel
    evidence_alias: str
    quote: Annotated[str, StringConstraints(max_length=500)]
    rationale: Annotated[str, StringConstraints(max_length=300)]


def verification_schema(aliases: Sequence[str]) -> type[Verification]:
    model: type[Verification] = create_model(
        "Verification",
        __base__=Verification,
        evidence_alias=(_label_type(aliases), ...),
    )
    return model


class GapQueries(BaseModel):
    """gap role: 1-2 new search queries for a sub-question with too little verified evidence."""

    model_config = ConfigDict(extra="forbid")

    queries: list[SearchQuery] = Field(min_length=1, max_length=2)

    @field_validator("queries")
    @classmethod
    def _unique(cls, value: list[str]) -> list[str]:
        return _require_unique(value, "search query")
