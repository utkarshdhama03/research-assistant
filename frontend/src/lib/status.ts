// A run status as a colour tone (for the dot) and a word (always shown or announced with it).

export type Tone = "ok" | "warn" | "bad" | "idle";

const TONES: Record<string, Tone> = {
  completed: "ok",
  running: "warn",
  partial: "warn",
  awaiting_approval: "warn",
  failed: "bad",
  cancelled: "idle",
  queued: "idle",
};

const ACTIVE = new Set(["queued", "running", "awaiting_approval"]);

export function statusTone(status: string): Tone {
  return TONES[status] ?? "idle";
}

export function statusWord(status: string): string {
  return status.replace(/_/g, " ");
}

/** Not finished yet: the run can still change (and can be cancelled). */
export function isActive(status: string): boolean {
  return ACTIVE.has(status);
}

/** "45 s", "5 min 31 s", "1 h 2 min": elapsed time in whole seconds. */
export function formatElapsed(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${s % 60} s`;
  return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
}
