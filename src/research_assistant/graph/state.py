"""Graph state: IDs (and two counters) only. Bulk data lives in Postgres."""

import uuid

from pydantic import BaseModel, ConfigDict, Field


class GraphState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: uuid.UUID
    sub_question_ids: list[uuid.UUID] = Field(default_factory=list)
    # Sub-questions that research and analyze work on: all of them, then only gap targets.
    active_sub_question_ids: list[uuid.UUID] = Field(default_factory=list)
    # How many of each sub-question's stored search queries research has already run.
    queries_searched: dict[uuid.UUID, int] = Field(default_factory=dict)
    # Sources with chunks found by each sub-question's own searches (status only; retrieval is run-wide).
    sources_by_sub_question: dict[uuid.UUID, list[uuid.UUID]] = Field(default_factory=dict)
    # Chunks already shown to analyze for each sub-question (a gap round shows new ones).
    analyzed_chunk_ids: dict[uuid.UUID, list[uuid.UUID]] = Field(default_factory=dict)
    # Chunks each claim cited in analyze; verify shows only these.
    claim_chunks: dict[uuid.UUID, list[uuid.UUID]] = Field(default_factory=dict)
    # Hosts that answered 403 or 429; not fetched again for the rest of the run.
    blocked_hosts: list[str] = Field(default_factory=list)
    gap_targets: list[uuid.UUID] = Field(default_factory=list)
    gap_rounds: int = 0
    report_id: uuid.UUID | None = None
