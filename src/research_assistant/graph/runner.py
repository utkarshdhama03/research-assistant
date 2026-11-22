"""Run one research question and record a terminal status (or a pause for plan approval).

run_research: inline (CLI, eval): the run is inserted as running with auto_approve, so no worker can
claim it, and executed without a checkpointer.
execute_run: the worker's entry point for a claimed run; with a checkpointer it can pause after the
plan (status awaiting_approval) and resume from the checkpoint once the plan is approved.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Command
from sqlalchemy import func, select

from research_assistant.config import Settings
from research_assistant.db.models import Chunk, ResearchRun, RunStatus
from research_assistant.db.repositories.events import append_event, list_events
from research_assistant.db.repositories.reports import latest_report
from research_assistant.db.repositories.runs import create_run, update_run_status
from research_assistant.db.session import session_scope
from research_assistant.graph.build import build_graph
from research_assistant.graph.context import Deps, RunCancelled, RunContext, WorkerStopping
from research_assistant.graph.nodes.write import verdict_counts
from research_assistant.graph.state import GraphState
from research_assistant.graph.status import decide_status

log = logging.getLogger(__name__)

RUN_CONFIG_KEYS = (
    "chat_model",
    "embed_model",
    "ollama_num_ctx",
    "max_sub_questions",
    "max_sources_per_query",
    "max_sources_per_sub_question",
    "max_sources_per_run",
    "max_chunks_per_source",
    "run_deadline_s",
    "research_budget_share",
    "write_top_k",
    "analyze_top_k",
    "max_claims_per_sub_question",
    "min_supported_claims",
    "claim_dedupe_threshold",
    "max_gap_loops",
    "max_backfills_per_sub_question",
    "query_jaccard_threshold",
    "verify_budget_share",
    "chunk_target_tokens",
    "chunk_overlap",
    "memory_enabled",
    "memory_min_similarity",
)
CANCELLED_REASON = "cancelled by user"
WORKER_STOPPED_REASON = "worker stopped"


@dataclass(frozen=True)
class RunOutcome:
    run_id: uuid.UUID
    status: RunStatus
    reason: str | None
    report_markdown: str | None


def run_config(settings: Settings) -> dict[str, Any]:
    return {key: getattr(settings, key) for key in RUN_CONFIG_KEYS}


def run_research(question: str, deps: Deps) -> RunOutcome:
    """Run inline (CLI, eval). Inserted directly as running, so a worker never claims it."""
    with session_scope(deps.session_factory) as session:
        run_id = create_run(
            session, question, run_config(deps.settings), status=RunStatus.RUNNING, auto_approve=True
        ).id
    return execute_run(run_id, deps)


def _thread(run_id: uuid.UUID) -> RunnableConfig:
    return {"configurable": {"thread_id": str(run_id)}}


def _pending_interrupts(graph: Any, run_id: uuid.UUID) -> tuple[bool, bool]:
    """(has a checkpoint, has a pending interrupt) for this run's thread."""
    snapshot = graph.get_state(_thread(run_id))
    pending = any(t.interrupts for t in snapshot.tasks)
    return bool(snapshot.values), pending


def _resume_point(ctx: RunContext) -> tuple[float, list[str]]:
    """Run time spent before the pause and the problems recorded so far (they live in events)."""
    with ctx.db() as session:
        events = list_events(session, ctx.run_id)
    elapsed = next(
        (float(e.payload.get("elapsed_s", 0.0)) for e in reversed(events) if e.event == "awaiting_approval"),
        0.0,
    )
    issues: list[str] = []
    for e in events:
        message = e.payload.get("message")
        if e.event == "problem" and isinstance(message, str) and message not in issues:
            issues.append(message)
    return elapsed, issues


def execute_run(
    run_id: uuid.UUID, deps: Deps, checkpointer: BaseCheckpointSaver[Any] | None = None
) -> RunOutcome:
    """Execute a run whose status is already running. Without a checkpointer the plan is never paused
    (the run must have auto_approve). Returns status awaiting_approval when it paused."""
    with session_scope(deps.session_factory) as session:
        run = session.get(ResearchRun, run_id)
        if run is None:
            raise ValueError(f"run {run_id} not found")
        question, auto_approve = run.question, run.auto_approve
    if checkpointer is None and not auto_approve:
        raise ValueError("a run that waits for plan approval needs a checkpointer")

    ctx = RunContext(
        deps=deps, run_id=run_id, question=question, started_at=deps.clock(), auto_approve=auto_approve
    )
    graph = build_graph(ctx, checkpointer)
    status: RunStatus
    reason: str | None
    try:
        has_checkpoint, resume = _pending_interrupts(graph, run_id) if checkpointer else (False, False)
        if has_checkpoint and not resume:
            raise RuntimeError("unexpected checkpoint state: a checkpoint without a pending approval")
        if resume:
            elapsed, ctx.issues = _resume_point(ctx)
            ctx.started_at = deps.clock() - elapsed  # the approval wait does not count against the deadline
            ctx.event("run_resumed", {"elapsed_s": elapsed})
            graph_input: Any = Command(resume=True)
        else:
            ctx.event("run_start", {"question": question})
            graph_input = GraphState(run_id=run_id)
        ctx.llm.warm_up()  # logged as an llm_call event
        ctx.event("embedder_warm_up", {"duration_s": round(deps.embedder.warm_up(), 3)})
        if checkpointer is not None:  # each checkpoint written before the next step starts
            result = graph.invoke(graph_input, _thread(run_id), durability="sync")
        else:  # durability needs a checkpointer (langgraph 1.2.22 fails on it without one)
            result = graph.invoke(graph_input)
        if "__interrupt__" in result:
            return _pause(ctx, checkpointer)
        final = GraphState.model_validate(result)
        with ctx.db() as session:
            total_chunks = session.scalar(
                select(func.count()).select_from(Chunk).where(Chunk.run_id == run_id)
            )
        status, reason = decide_status(
            plan_failure=ctx.plan_failure,
            total_chunks=int(total_chunks or 0),
            has_report=final.report_id is not None,
            issues=ctx.issues,
        )
    except RunCancelled:
        status, reason = RunStatus.CANCELLED, CANCELLED_REASON
    except WorkerStopping:
        status, reason = RunStatus.FAILED, WORKER_STOPPED_REASON
    except Exception as e:  # any unexpected error ends the run as failed, with the reason recorded
        log.exception("run %s failed", run_id)
        status, reason = RunStatus.FAILED, f"unexpected error: {type(e).__name__}: {e}"
    return _finish(ctx, status, reason, checkpointer)


def _pause(ctx: RunContext, checkpointer: BaseCheckpointSaver[Any] | None) -> RunOutcome:
    """The graph stopped at the approval interrupt: wait for approval, unless a cancel came in."""
    elapsed = round(ctx.now() - ctx.started_at, 3)
    ctx.current_node = "run"
    with ctx.db() as session:
        run = session.get(ResearchRun, ctx.run_id, with_for_update=True)
        cancelled = run is not None and run.cancel_requested
        if not cancelled:
            update_run_status(session, ctx.run_id, RunStatus.AWAITING_APPROVAL)
    if cancelled:
        return _finish(ctx, RunStatus.CANCELLED, CANCELLED_REASON, checkpointer)
    ctx.event("awaiting_approval", {"elapsed_s": elapsed})
    return RunOutcome(ctx.run_id, RunStatus.AWAITING_APPROVAL, None, None)


def _finish(
    ctx: RunContext, status: RunStatus, reason: str | None, checkpointer: BaseCheckpointSaver[Any] | None
) -> RunOutcome:
    ctx.current_node = "run"
    payload = {
        "status": str(status),
        "reason": reason,
        "duration_s": round(ctx.deps.clock() - ctx.started_at, 3),
        "claims": verdict_counts(ctx, ctx.run_id),
    }
    # Status and run_finished in one transaction: a reader that sees the terminal status (the event
    # stream) has every event of the run, including run_finished.
    with ctx.db() as session:
        update_run_status(session, ctx.run_id, status, error=reason[:4000] if reason else None)
        append_event(session, ctx.run_id, "run", "run_finished", payload)
        report = latest_report(session, ctx.run_id)
        markdown = report.markdown if report is not None else None
    if ctx.deps.on_event is not None:
        ctx.deps.on_event("run", "run_finished", payload)
    if checkpointer is not None:
        checkpointer.delete_thread(str(ctx.run_id))  # a finished run is never resumed
    return RunOutcome(ctx.run_id, status, reason, markdown)
