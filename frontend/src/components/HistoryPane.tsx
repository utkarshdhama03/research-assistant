import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { listRuns } from "../api/runs";
import type { RunSummary } from "../api/types";
import { runPath } from "../lib/paths";
import { isActive, statusWord } from "../lib/status";
import { ErrorMessage } from "./ErrorMessage";
import { StatusDot } from "./StatusDot";

export const FILTER_DEBOUNCE_MS = 300;
export const HISTORY_POLL_MS = 10_000;
const FILTER_MAX = 200; // GET /runs?q= accepts 1-200 characters

/** Every run, newest first: status dot and the question on one line; a server-side filter. */
export function HistoryPane({
  selectedId,
  refreshKey,
  onPick,
}: {
  selectedId: string | null;
  refreshKey: number;
  onPick: () => void;
}) {
  const [filter, setFilter] = useState("");
  const [q, setQ] = useState("");
  const [items, setItems] = useState<RunSummary[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    const timer = window.setTimeout(() => setQ(filter.trim()), FILTER_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [filter]);

  const load = useCallback(
    async (next: string | null): Promise<void> => {
      setLoading(true);
      try {
        const page = await listRuns({ cursor: next, q: q || null });
        setItems((previous) => (next ? [...previous, ...page.items] : page.items));
        setCursor(page.next_cursor);
        setError(null);
      } catch (e) {
        setError(e);
      } finally {
        setLoading(false);
      }
    },
    [q],
  );

  useEffect(() => {
    void load(null);
  }, [load, refreshKey]);

  const anyActive = items.some((r) => isActive(r.status));
  useEffect(() => {
    if (!anyActive) return undefined;
    const timer = window.setInterval(() => void load(null), HISTORY_POLL_MS);
    return () => window.clearInterval(timer);
  }, [anyActive, load]);

  return (
    <nav className="history" aria-label="history">
      <div className="pane-title">history</div>
      <input
        type="search"
        className="filter"
        placeholder="filter"
        aria-label="filter history"
        value={filter}
        maxLength={FILTER_MAX}
        onChange={(e) => setFilter(e.target.value)}
      />
      <ErrorMessage error={error} />
      <ul className="history-list">
        {items.map((run) => (
          <li key={run.id}>
            <Link
              to={runPath(run.id)}
              className={run.id === selectedId ? "selected" : undefined}
              aria-current={run.id === selectedId ? "page" : undefined}
              title={`${statusWord(run.status)} · ${run.question}`}
              onClick={onPick}
            >
              <StatusDot status={run.status} /> <span className="question">{run.question}</span>
            </Link>
          </li>
        ))}
      </ul>
      {!loading && items.length === 0 && !error && <p className="dim">{q ? "no match" : "no runs yet"}</p>}
      {cursor && (
        <button type="button" className="load-more" onClick={() => void load(cursor)} disabled={loading}>
          load more
        </button>
      )}
    </nav>
  );
}
