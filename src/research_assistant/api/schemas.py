"""Request and response models. Responses are built from ORM rows explicitly, never returned as ORM."""

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateRunRequest(_Strict):
    question: str  # stripped and length-checked against API_QUESTION_MAX_CHARS in the route
    auto_approve: bool = False


class EditedSubQuestion(_Strict):
    question: str
    search_queries: list[str]
    premise_check: bool = False


class ApproveRequest(_Strict):
    # None: approve the plan as it is. Otherwise the full edited plan, validated like a model plan.
    sub_questions: list[EditedSubQuestion] | None = None


class EmptyRequest(_Strict):
    """Bodyless actions still send `{}` with Content-Type: application/json."""


class RunCreated(BaseModel):
    id: uuid.UUID
    status: str


class RunSummary(BaseModel):
    id: uuid.UUID
    question: str
    status: str
    reason: str | None
    auto_approve: bool
    created_at: datetime
    finished_at: datetime | None


class RunPage(BaseModel):
    items: list[RunSummary]
    next_cursor: str | None


class SubQuestionOut(BaseModel):
    ordinal: int
    question: str
    search_queries: list[str]
    premise_check: bool


class SourceOut(BaseModel):
    id: uuid.UUID
    url: str
    final_url: str | None
    title: str | None
    status: str
    fetched_at: datetime | None
    reused: bool  # copied from an earlier run (memory)
    reused_from_source_id: uuid.UUID | None
    reused_from_run_id: uuid.UUID | None  # None if the original's run was deleted


class SourceRow(BaseModel):
    """GET /runs/{id}/sources: every source of the run, failed and skipped ones included."""

    id: uuid.UUID
    url: str
    final_url: str | None
    title: str | None
    status: str  # fetched, failed or skipped
    reason: str | None  # why a source failed or was skipped (sources.error)
    retrieved_at: datetime | None
    reused: bool
    reused_from_run_id: uuid.UUID | None  # None if not reused, or the original's run was deleted
    cited: bool  # cited in the run's latest report


class StructuredReport(BaseModel):
    """GET /runs/{id}/report: reports.structure (see graph/citations.py assemble_report)."""

    report_id: uuid.UUID
    created_at: datetime
    premise_note: str | None
    sections: list[dict[str, Any]]
    references: list[dict[str, Any]]
    claims_by_n: dict[str, list[dict[str, Any]]]


class LegacyReport(BaseModel):
    """A report written before reports.structure existed: its Markdown only, to show as plain text."""

    report_id: uuid.UUID
    created_at: datetime
    legacy_markdown: str


class NodeTiming(BaseModel):
    node: str
    event: str  # node_end, node_error or node_stopped
    duration_s: float


class RunDetail(RunSummary):
    cancel_requested: bool
    sub_questions: list[SubQuestionOut]
    sources: list[SourceOut]
    node_timings: list[NodeTiming]
    claims_by_verdict: dict[str, int]
    report_markdown: str | None


class RunAction(BaseModel):
    id: uuid.UUID
    status: str
    cancel_requested: bool


class ComponentHealth(BaseModel):
    ok: bool
    detail: str | None = None


class WorkerHealth(ComponentHealth):
    last_seen: datetime | None = None
    age_s: float | None = None
    current_run_id: uuid.UUID | None = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    db: ComponentHealth
    ollama: ComponentHealth
    searxng: ComponentHealth
    worker: WorkerHealth


class LoginRequest(_Strict):
    key: str = Field(min_length=1, max_length=1024)


class SessionInfo(BaseModel):
    auth_enabled: bool
    authenticated: bool


class HealthStatus(BaseModel):
    """/health without a session: the overall status only."""

    status: str


class ErrorDetail(BaseModel):
    loc: list[str | int]
    msg: str


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorBody(BaseModel):
    error: ErrorInfo
