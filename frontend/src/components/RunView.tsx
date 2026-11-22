import { useCallback, useEffect, useRef, useState } from "react";

import { cancelRun, getRun, getSources } from "../api/runs";
import type { RunDetail, SourceRow } from "../api/types";
import { useRunEvents } from "../hooks/useRunEvents";
import { claimLines, steps as stepsOf } from "../lib/runProgress";
import { formatElapsed, isActive } from "../lib/status";
import { ErrorMessage } from "./ErrorMessage";
import { EventLog } from "./EventLog";
import { PlanEditor } from "./PlanEditor";
import { ReportView } from "./ReportView";
import { RunLog } from "./RunLog";
import { SourcesTab } from "./SourcesTab";
import { StatusDot } from "./StatusDot";
import { StepLine } from "./StepLine";

// Events after which the run's status, plan or report may have changed: reload GET /runs/{id}.
const REFRESH_EVENTS = new Set([
  "worker_claimed",
  "run_start",
  "run_resumed",
  "awaiting_approval",
  "plan_approved",
  "plan_edited",
  "plan_rejected",
  "run_cancelled",
  "cancel_requested",
  "run_aborted",
  "run_finished",
]);

type Tab = "report" | "sources" | "log";
const TABS: Tab[] = ["report", "sources", "log"];

function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return undefined;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [active]);
  return now;
}

export function RunView({ runId, onStatus }: { runId: string; onStatus: (status: string) => void }) {
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [tab, setTab] = useState<Tab>("report");
  const [sources, setSources] = useState<SourceRow[] | null>(null);
  const [sourcesError, setSourcesError] = useState<unknown>(null);
  const [cancelError, setCancelError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const { events, endStatus } = useRunEvents(runId);
  const processed = useRef(0);
  const lastStatus = useRef<string | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    try {
      setRun(await getRun(runId));
      setError(null);
    } catch (e) {
      setError(e);
    }
  }, [runId]);

  useEffect(() => {
    processed.current = 0;
    void refresh();
  }, [refresh]);

  useEffect(() => {
    const fresh = events.slice(processed.current);
    processed.current = events.length;
    if (fresh.some((e) => REFRESH_EVENTS.has(e.event))) void refresh();
  }, [events, refresh]);

  useEffect(() => {
    if (endStatus !== null) void refresh();
  }, [endStatus, refresh]);

  const status = run?.status ?? null;
  useEffect(() => {
    if (status !== null && status !== lastStatus.current) {
      lastStatus.current = status;
      onStatus(status);
    }
  }, [status, onStatus]);

  const finished = run !== null && !isActive(run.status);
  useEffect(() => {
    if (!finished) return undefined;
    let active = true;
    getSources(runId)
      .then((rows) => active && setSources(rows))
      .catch((e: unknown) => active && setSourcesError(e));
    return () => {
      active = false;
    };
  }, [finished, runId]);

  const now = useNow(run !== null && isActive(run.status));

  if (error && !run) return <ErrorMessage error={error} />;
  if (!run) return <p className="dim">loading…</p>;

  const steps = stepsOf(events, run.status);
  const supported = finished
    ? (run.claims_by_verdict.supported ?? 0)
    : claimLines(events).filter((c) => c.ok).length;
  const end = run.finished_at ? Date.parse(run.finished_at) : now;
  const elapsed = (end - Date.parse(run.created_at)) / 1000;
  const canCancel = isActive(run.status) && !run.cancel_requested;

  const cancel = async (): Promise<void> => {
    if (!window.confirm("Cancel this run? A running run stops at its next step.")) return;
    setBusy(true);
    setCancelError(null);
    try {
      await cancelRun(run.id);
      void refresh();
    } catch (e) {
      setCancelError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <article className="run-view">
      <h1 className="question">{run.question}</h1>
      <div className="status-line">
        <StatusDot status={run.status} word />
        <span className="dim"> · </span>
        <span>{formatElapsed(elapsed)}</span>
        <span className="dim"> · </span>
        <span>
          {supported} {supported === 1 ? "claim" : "claims"} verified
        </span>
        {run.auto_approve && <span className="dim"> · auto-approve</span>}
        {run.cancel_requested && isActive(run.status) && <span className="dim"> · cancel requested</span>}
        {canCancel && (
          <button type="button" className="danger" onClick={() => void cancel()} disabled={busy}>
            cancel
          </button>
        )}
      </div>
      {run.reason && <p className="reason">reason: {run.reason}</p>}
      <ErrorMessage error={cancelError ?? error} />
      <StepLine steps={steps} />

      {run.status === "awaiting_approval" && (
        <PlanEditor
          key={JSON.stringify(run.sub_questions)}
          runId={run.id}
          subQuestions={run.sub_questions}
          onDone={() => void refresh()}
        />
      )}

      {isActive(run.status) && run.status !== "awaiting_approval" && (
        <RunLog events={events} steps={steps} subQuestions={run.sub_questions.length} />
      )}

      {finished && (
        <>
          <div className="tabs" role="tablist" aria-label="run">
            {TABS.map((t) => (
              <button
                key={t}
                type="button"
                role="tab"
                id={`tab-${t}`}
                aria-selected={tab === t}
                aria-controls="run-tabpanel"
                onClick={() => setTab(t)}
              >
                {t === "sources" ? `sources (${sources ? sources.length : "…"})` : t}
              </button>
            ))}
          </div>
          <div role="tabpanel" id="run-tabpanel" aria-labelledby={`tab-${tab}`}>
            {tab === "report" &&
              (run.report_markdown !== null ? (
                <ReportView key={run.finished_at ?? "report"} runId={run.id} />
              ) : (
                <p className="dim">no report.</p>
              ))}
            {tab === "sources" && <SourcesTab rows={sources} error={sourcesError} />}
            {tab === "log" && <EventLog events={events} />}
          </div>
        </>
      )}
    </article>
  );
}
