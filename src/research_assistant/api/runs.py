"""Run endpoints: create, list, detail, report, sources, approve, reject, cancel."""

import base64
import binascii
import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError
from sqlalchemy import func, select, tuple_
from sqlalchemy.orm import Session

from research_assistant.api.errors import ApiError, validation_details
from research_assistant.api.schemas import (
    ApproveRequest,
    CreateRunRequest,
    EmptyRequest,
    LegacyReport,
    NodeTiming,
    RunAction,
    RunCreated,
    RunDetail,
    RunPage,
    RunSummary,
    SourceOut,
    SourceRow,
    StructuredReport,
    SubQuestionOut,
)
from research_assistant.api.state import ApiState, api_state
from research_assistant.db.models import Claim, ResearchRun, RunStatus, Source
from research_assistant.db.repositories.events import append_event, list_events
from research_assistant.db.repositories.queue import lock_run
from research_assistant.db.repositories.reports import latest_report
from research_assistant.db.repositories.runs import (
    create_run,
    list_sub_questions,
    replace_sub_questions,
    update_run_status,
)
from research_assistant.db.session import session_scope
from research_assistant.graph.planning import finalize_plan
from research_assistant.graph.runner import CANCELLED_REASON, run_config
from research_assistant.llm.schemas import ResearchPlan

router = APIRouter()
State = Annotated[ApiState, Depends(api_state)]
REJECTED_REASON = "plan rejected"
TIMING_EVENTS = ("node_end", "node_error", "node_stopped")


def _summary(run: ResearchRun) -> RunSummary:
    return RunSummary(
        id=run.id,
        question=run.question,
        status=str(run.status),
        reason=run.error,
        auto_approve=run.auto_approve,
        created_at=run.created_at,
        finished_at=run.finished_at,
    )


def _get_run(session: Session, run_id: uuid.UUID, *, lock: bool = False) -> ResearchRun:
    run = lock_run(session, run_id) if lock else session.get(ResearchRun, run_id)
    if run is None:
        raise ApiError(404, f"run {run_id} not found")
    return run


def _conflict(run: ResearchRun, action: str, allowed: str) -> ApiError:
    return ApiError(409, f"cannot {action} a run that is {run.status}; only {allowed} runs can be")


def escape_like(text: str) -> str:
    """Escape LIKE wildcards (and the escape character) so they match literally."""
    return text.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


# -- cursor ----------------------------------------------------------------------------------


def encode_cursor(created_at: datetime, run_id: uuid.UUID) -> str:
    raw = f"{created_at.isoformat()}|{run_id}".encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        created, run_id = raw.split("|", 1)
        return datetime.fromisoformat(created), uuid.UUID(run_id)
    except (ValueError, binascii.Error, UnicodeDecodeError) as e:
        raise ApiError(422, "invalid cursor") from e


def _sources_with_origins(
    session: Session, run_id: uuid.UUID
) -> tuple[list[Source], dict[uuid.UUID, uuid.UUID]]:
    """The run's sources, and for reused ones the run their original belongs to (if it still exists)."""
    rows = list(
        session.scalars(select(Source).where(Source.run_id == run_id).order_by(Source.fetched_at, Source.url))
    )
    originals = [s.reused_from_source_id for s in rows if s.reused_from_source_id is not None]
    original_runs: dict[uuid.UUID, uuid.UUID] = {}
    if originals:
        stmt = select(Source.id, Source.run_id).where(Source.id.in_(originals))
        original_runs = {source_id: run for source_id, run in session.execute(stmt)}
    return rows, original_runs


# -- create / list / detail / report ---------------------------------------------------------


@router.post("/runs", status_code=status.HTTP_202_ACCEPTED, response_model=RunCreated)
def create(body: CreateRunRequest, state: State) -> RunCreated:
    question = body.question.strip()
    limit = state.settings.api_question_max_chars
    if not 5 <= len(question) <= limit:
        raise ApiError(422, f"question must be 5 to {limit} characters after stripping whitespace")
    with session_scope(state.session_factory) as session:
        run = create_run(
            session,
            question,
            run_config(state.settings),
            status=RunStatus.QUEUED,
            auto_approve=body.auto_approve,
        )
        append_event(session, run.id, "api", "run_queued", {"auto_approve": body.auto_approve})
        return RunCreated(id=run.id, status=str(run.status))


@router.get("/runs", response_model=RunPage)
def list_runs(
    state: State,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
    q: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
) -> RunPage:
    stmt = select(ResearchRun).order_by(ResearchRun.created_at.desc(), ResearchRun.id.desc()).limit(limit + 1)
    if q is not None:
        # Bound parameter, LIKE wildcards escaped: the text matches literally, case-insensitively.
        stmt = stmt.where(ResearchRun.question.ilike(f"%{escape_like(q)}%", escape="\\"))
    if cursor is not None:
        created_at, run_id = decode_cursor(cursor)
        stmt = stmt.where(tuple_(ResearchRun.created_at, ResearchRun.id) < tuple_(created_at, run_id))
    with session_scope(state.session_factory) as session:
        rows = list(session.scalars(stmt))
    page = rows[:limit]
    next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if len(rows) > limit else None
    return RunPage(items=[_summary(r) for r in page], next_cursor=next_cursor)


@router.get("/runs/{run_id}", response_model=RunDetail)
def detail(run_id: uuid.UUID, state: State) -> RunDetail:
    with session_scope(state.session_factory) as session:
        run = _get_run(session, run_id)
        sub_questions = [
            SubQuestionOut(
                ordinal=sq.ordinal,
                question=sq.text,
                search_queries=list(sq.search_queries),
                premise_check=sq.premise_check,
            )
            for sq in list_sub_questions(session, run_id)
        ]
        source_rows, original_runs = _sources_with_origins(session, run_id)
        sources = [
            SourceOut(
                id=s.id,
                url=s.url,
                final_url=s.final_url,
                title=s.title,
                status=str(s.status),
                fetched_at=s.fetched_at,
                reused=s.reused_from_source_id is not None,
                reused_from_source_id=s.reused_from_source_id,
                reused_from_run_id=original_runs.get(s.reused_from_source_id)
                if s.reused_from_source_id
                else None,
            )
            for s in source_rows
        ]
        timings = [
            NodeTiming(node=e.node, event=e.event, duration_s=float(e.payload.get("duration_s", 0.0)))
            for e in list_events(session, run_id)
            if e.event in TIMING_EVENTS
        ]
        counts = {
            str(verdict) if verdict is not None else "unverified": int(n)
            for verdict, n in session.execute(
                select(Claim.verdict, func.count()).where(Claim.run_id == run_id).group_by(Claim.verdict)
            )
        }
        report = latest_report(session, run_id)
        base = _summary(run)
        return RunDetail(
            **base.model_dump(),
            cancel_requested=run.cancel_requested,
            sub_questions=sub_questions,
            sources=sources,
            node_timings=timings,
            claims_by_verdict=counts,
            report_markdown=report.markdown if report is not None else None,
        )


@router.get("/runs/{run_id}/report.md", response_class=Response)
def report_markdown(run_id: uuid.UUID, state: State) -> Response:
    with session_scope(state.session_factory) as session:
        _get_run(session, run_id)
        report = latest_report(session, run_id)
        if report is None:
            raise ApiError(404, f"run {run_id} has no report")
        return Response(report.markdown, media_type="text/markdown; charset=utf-8")


@router.get("/runs/{run_id}/report", response_model=StructuredReport | LegacyReport)
def report(run_id: uuid.UUID, state: State) -> StructuredReport | LegacyReport:
    """The latest report as data; a report written before reports.structure existed gives its Markdown."""
    with session_scope(state.session_factory) as session:
        _get_run(session, run_id)
        row = latest_report(session, run_id)
        if row is None:
            raise ApiError(404, f"run {run_id} has no report")
        if row.structure is None:
            return LegacyReport(report_id=row.id, created_at=row.created_at, legacy_markdown=row.markdown)
        return StructuredReport(report_id=row.id, created_at=row.created_at, **row.structure)


@router.get("/runs/{run_id}/sources", response_model=list[SourceRow])
def sources(run_id: uuid.UUID, state: State) -> list[SourceRow]:
    with session_scope(state.session_factory) as session:
        _get_run(session, run_id)
        rows, original_runs = _sources_with_origins(session, run_id)
        row = latest_report(session, run_id)
        cited = {c.get("source_id") for c in row.citations} if row is not None else set()
        return [
            SourceRow(
                id=s.id,
                url=s.url,
                final_url=s.final_url,
                title=s.title,
                status=str(s.status),
                reason=s.error,
                retrieved_at=s.fetched_at,
                reused=s.reused_from_source_id is not None,
                reused_from_run_id=original_runs.get(s.reused_from_source_id)
                if s.reused_from_source_id
                else None,
                cited=str(s.id) in cited,
            )
            for s in rows
        ]


# -- approve / reject / cancel ---------------------------------------------------------------


def _validated_plan(
    body: ApproveRequest, state: ApiState
) -> tuple[list[str], list[list[str]], list[bool], list[str]]:
    """The edited plan with the model plan's bounds and the same Jaccard query dedupe."""
    assert body.sub_questions is not None
    raw = {"sub_questions": [s.model_dump() for s in body.sub_questions]}
    try:
        plan = ResearchPlan.model_validate(raw)
    except ValidationError as e:
        raise ApiError(422, "invalid plan", details=validation_details(e.errors())) from e
    limit = state.settings.max_sub_questions
    if len(plan.sub_questions) > limit:
        raise ApiError(422, f"the plan may have at most {limit} sub-questions (MAX_SUB_QUESTIONS)")
    planned, dropped = finalize_plan(plan, limit, state.settings.query_jaccard_threshold)
    return (
        [p.question for p in planned],
        [p.queries for p in planned],
        [p.premise_check for p in planned],
        dropped,
    )


@router.post("/runs/{run_id}/approve", response_model=RunAction)
def approve(run_id: uuid.UUID, state: State, body: ApproveRequest) -> RunAction:
    edited = _validated_plan(body, state) if body.sub_questions is not None else None
    with session_scope(state.session_factory) as session:
        run = _get_run(session, run_id, lock=True)
        if run.status is not RunStatus.AWAITING_APPROVAL:
            raise _conflict(run, "approve", "awaiting_approval")
        if edited is not None:
            texts, queries, flags, dropped = edited
            before = [
                {
                    "question": sq.text,
                    "search_queries": list(sq.search_queries),
                    "premise_check": sq.premise_check,
                }
                for sq in list_sub_questions(session, run_id)
            ]
            replace_sub_questions(session, run_id, texts, queries, flags)
            after = [
                {"question": t, "search_queries": q, "premise_check": f}
                for t, q, f in zip(texts, queries, flags, strict=True)
            ]
            append_event(
                session,
                run_id,
                "api",
                "plan_edited",
                {"before": before, "after": after, "dropped_queries": dropped},
            )
        append_event(session, run_id, "api", "plan_approved", {"edited": edited is not None})
        update_run_status(session, run_id, RunStatus.QUEUED)  # the worker resumes it from the checkpoint
        return RunAction(id=run_id, status=str(RunStatus.QUEUED), cancel_requested=run.cancel_requested)


def _end_before_running(session: Session, run_id: uuid.UUID, reason: str, event: str) -> None:
    """Cancel a run no worker is executing: status, then its closing events, in one transaction."""
    update_run_status(session, run_id, RunStatus.CANCELLED, error=reason)
    append_event(session, run_id, "api", event, {"reason": reason})
    append_event(
        session, run_id, "run", "run_finished", {"status": str(RunStatus.CANCELLED), "reason": reason}
    )


@router.post("/runs/{run_id}/reject", response_model=RunAction)
def reject(run_id: uuid.UUID, state: State, _body: EmptyRequest) -> RunAction:
    with session_scope(state.session_factory) as session:
        run = _get_run(session, run_id, lock=True)
        if run.status is not RunStatus.AWAITING_APPROVAL:
            raise _conflict(run, "reject", "awaiting_approval")
        _end_before_running(session, run_id, REJECTED_REASON, "plan_rejected")
    state.delete_checkpoint(run_id)
    return RunAction(id=run_id, status=str(RunStatus.CANCELLED), cancel_requested=False)


@router.post(
    "/runs/{run_id}/cancel",
    response_model=RunAction,
    responses={202: {"model": RunAction, "description": "cancel requested; the worker stops the run"}},
)
def cancel(run_id: uuid.UUID, state: State, _body: EmptyRequest) -> RunAction | JSONResponse:
    with session_scope(state.session_factory) as session:
        run = _get_run(session, run_id, lock=True)
        current = run.status
        if current is RunStatus.RUNNING:
            run.cancel_requested = True
            append_event(session, run_id, "api", "cancel_requested", {})
        elif current in (RunStatus.QUEUED, RunStatus.AWAITING_APPROVAL):
            _end_before_running(session, run_id, CANCELLED_REASON, "run_cancelled")
        else:
            raise _conflict(run, "cancel", "queued, awaiting_approval or running")
    if current is RunStatus.RUNNING:
        action = RunAction(id=run_id, status=str(RunStatus.RUNNING), cancel_requested=True)
        return JSONResponse(action.model_dump(mode="json"), status_code=status.HTTP_202_ACCEPTED)
    state.delete_checkpoint(run_id)  # an approved-then-queued run may have one
    return RunAction(id=run_id, status=str(RunStatus.CANCELLED), cancel_requested=False)
