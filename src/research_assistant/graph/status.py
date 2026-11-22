"""Terminal status rules (pure)."""

from collections.abc import Sequence

from research_assistant.db.models import RunStatus


def decide_status(
    *, plan_failure: str | None, total_chunks: int, has_report: bool, issues: Sequence[str]
) -> tuple[RunStatus, str | None]:
    """failed: the plan failed, or the run stored no chunks, or no report was written.
    partial: a report exists but problems were recorded (each listed in the reason).
    completed: a report exists and nothing went wrong."""
    if plan_failure:
        return RunStatus.FAILED, plan_failure
    if total_chunks == 0:
        detail = f" ({'; '.join(issues)})" if issues else ""
        return RunStatus.FAILED, f"no chunks were stored in this run{detail}"
    if not has_report:
        return RunStatus.FAILED, "no report was written"
    if issues:
        return RunStatus.PARTIAL, "; ".join(issues)
    return RunStatus.COMPLETED, None
