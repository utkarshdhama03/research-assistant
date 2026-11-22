import { useEffect, useState } from "react";

import { ApiError } from "../api/client";
import { getHealth } from "../api/runs";
import type { Health } from "../api/types";
import type { Tone } from "../lib/status";

export const HEALTH_POLL_MS = 15_000;
const COMPONENTS = ["db", "ollama", "searxng", "worker"] as const;

interface Summary {
  tone: Tone;
  text: string;
  details: string[];
}

/** One line for the top bar: green "all systems ok", amber naming what is down, red when the database is
 * down or the API cannot be reached. */
export function summarize(health: Health | null, error: string | null): Summary {
  if (error) return { tone: "bad", text: "api unreachable", details: [error] };
  if (!health) return { tone: "idle", text: "checking…", details: [] };
  const known = COMPONENTS.filter((name) => health[name] !== undefined);
  const details = known.map((name) => {
    const part = health[name];
    return `${name}: ${part?.ok ? "ok" : "down"}${part?.detail ? ` (${part.detail})` : ""}`;
  });
  if (health.status === "ok") return { tone: "ok", text: known.length ? "all systems ok" : "ok", details };
  const down = known.filter((name) => health[name]?.ok === false);
  if (down.length === 0) return { tone: "warn", text: "degraded", details }; // no session: no details
  const tone: Tone = down.includes("db") ? "bad" : "warn";
  const text = down.length === 1 ? `${down[0]} down` : `${down.length} down: ${down.join(", ")}`;
  return { tone, text, details };
}

/** GET /health now and every 15 s; click for the per-component details. */
export function HealthIndicator({ intervalMs = HEALTH_POLL_MS }: { intervalMs?: number }) {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let active = true;
    const poll = async (): Promise<void> => {
      try {
        const result = await getHealth();
        if (active) {
          setHealth(result);
          setError(null);
        }
      } catch (e) {
        if (active) setError(e instanceof ApiError ? e.message : "health check failed");
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), intervalMs);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [intervalMs]);

  const s = summarize(health, error);
  return (
    <div className="health">
      <button
        type="button"
        className={`health-button tone-${s.tone}`}
        aria-expanded={open}
        aria-controls="health-details"
        title={s.details.join("\n") || undefined}
        onClick={() => setOpen((v) => !v)}
        disabled={s.details.length === 0}
      >
        <span className="dot" aria-hidden="true">
          ●
        </span>{" "}
        <span role="status">{s.text}</span>
      </button>
      {open && s.details.length > 0 && (
        <ul id="health-details" className="health-details">
          {s.details.map((d) => (
            <li key={d}>{d}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
