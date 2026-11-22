import type { Step, StepState } from "../lib/runProgress";

const SYMBOL: Record<StepState, string> = { done: "✓", running: "●", pending: "○", failed: "✗", paused: "⏸" };
const WORD: Record<StepState, string> = {
  done: "done",
  running: "running",
  pending: "not started",
  failed: "failed",
  paused: "waiting for you",
};

/** ✓ plan  ✓ research  ● verify  ○ write: a symbol and a name per step (each with a spoken state). */
export function StepLine({ steps }: { steps: Step[] }) {
  return (
    <ol className="step-line" aria-label="steps">
      {steps.map((s) => (
        <li key={s.name} className={`step step-${s.state}`} aria-label={`${s.name} ${WORD[s.state]}`}>
          <span aria-hidden="true">
            {SYMBOL[s.state]} {s.name}
          </span>
        </li>
      ))}
    </ol>
  );
}
