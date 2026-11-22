// One readable line per event for the log tab. Event types in EVENT_NAMES are known; anything else
// (an event type this UI version has not heard of) is marked unknown and shown dimmed with its payload.

import { EVENT_NAMES } from "../api/eventNames";
import type { RunEvent } from "../api/types";
import { formatDuration } from "./format";

const KNOWN: ReadonlySet<string> = new Set<string>(EVENT_NAMES);

function str(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

function num(value: unknown): number | null {
  return typeof value === "number" && !Number.isNaN(value) ? value : null;
}

type Formatter = (p: Record<string, unknown>) => string;

const FORMATTERS: Record<string, Formatter> = {
  node_end: (p) => formatDuration(num(p.duration_s)),
  node_error: (p) => [str(p.error) ?? str(p.message), num(p.duration_s) !== null ? formatDuration(num(p.duration_s)) : null].filter(Boolean).join(" · "),
  llm_call: (p) =>
    [str(p.prompt_name), str(p.outcome), num(p.total_duration_s) !== null ? formatDuration(num(p.total_duration_s)) : null]
      .filter(Boolean)
      .join(" · "),
  search_done: (p) =>
    `${str(p.query) ?? ""} · ${str(p.provider) ?? "?"} · ${num(p.result_count) ?? 0} results${str(p.error) ? ` · ${str(p.error)}` : ""}`,
  source_fetched: (p) => `${str(p.url) ?? ""} · ${num(p.chunks) ?? 0} chunks`,
  fetch_failed: (p) => `${str(p.url) ?? ""} · ${str(p.reason) ?? "failed"}${num(p.status_code) ? ` ${num(p.status_code)}` : ""}`,
  extract_skipped: (p) => `${str(p.url) ?? ""} · ${str(p.reason) ?? ""}`,
  host_blocked: (p) => `${str(p.host) ?? ""} · HTTP ${num(p.status_code) ?? "?"}`,
  claims_created: (p) => `${Array.isArray(p.claims) ? p.claims.length : 0} claims`,
  claim_verified: (p) => `claim ${num(p.ordinal) ?? "?"} · ${str(p.verdict) ?? "?"}${str(p.claim) ? ` · ${str(p.claim)}` : ""}`,
  number_mismatch: (p) =>
    `claim ${num(p.ordinal) ?? "?"} · missing ${Array.isArray(p.missing) ? p.missing.map(String).join(", ") : "?"}`,
  quote_not_found: (p) => `claim ${num(p.ordinal) ?? "?"}`,
  run_finished: (p) => str(p.status) ?? "",
};

const GENERIC_KEYS = ["message", "reason", "query", "url", "verdict", "status", "host"];

function generic(p: Record<string, unknown>): string {
  for (const key of GENERIC_KEYS) {
    const value = str(p[key]);
    if (value) return value;
  }
  const d = num(p.duration_s);
  return d !== null ? formatDuration(d) : "";
}

export interface LogLine {
  id: string;
  time: string;
  node: string;
  event: string;
  text: string;
  known: boolean;
}

function clock(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "--:--:--" : date.toLocaleTimeString([], { hour12: false });
}

const RAW_MAX = 160;

export function formatEvent(e: RunEvent): LogLine {
  const known = KNOWN.has(e.event);
  let text: string;
  if (known) text = (FORMATTERS[e.event] ?? generic)(e.payload);
  else {
    const raw = JSON.stringify(e.payload);
    text = raw.length > RAW_MAX ? `${raw.slice(0, RAW_MAX)}…` : raw;
  }
  return { id: e.id, time: clock(e.created_at), node: e.node, event: e.event, text, known };
}
