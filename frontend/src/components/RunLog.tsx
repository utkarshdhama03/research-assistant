import { useLayoutEffect, useRef, type ReactNode } from "react";

import type { RunEvent } from "../api/types";
import {
  checkingLine,
  claimLines,
  plural,
  problems,
  researchSummary,
  sectionsWritten,
  type Step,
  type StepName,
} from "../lib/runProgress";

// Stay at the bottom while new lines arrive, unless the reader scrolled up more than this.
export const STICK_PX = 24;

function Block({ name, summary, children }: { name: string; summary: string; children?: ReactNode }) {
  return (
    <div className="log-block">
      <div className="log-head">
        <span className="log-step">── {name}</span> <span className="dim">{summary}</span>
      </div>
      {children}
    </div>
  );
}

/** The live log of a running run: one block per started step, claim lines under verify. */
export function RunLog({ events, steps, subQuestions }: { events: RunEvent[]; steps: Step[]; subQuestions: number }) {
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useLayoutEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [events.length]);

  const onScroll = (): void => {
    const el = box.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight <= STICK_PX;
  };

  const started = new Set(steps.filter((s) => s.state !== "pending").map((s) => s.name));
  const verifyRunning = steps.some((s) => s.name === "verify" && s.state === "running");
  const claims = claimLines(events);
  const ok = claims.filter((c) => c.ok).length;
  const checking = checkingLine(events, verifyRunning);
  const found = problems(events);
  const problemLines = (step: StepName) =>
    found
      .filter((p) => p.step === step)
      .map((p) => (
        <div key={p.id} className="log-line problem">
          ! {p.message}
        </div>
      ));

  return (
    <div className="run-log" role="log" aria-label="progress" ref={box} onScroll={onScroll} tabIndex={0}>
      {started.has("plan") && (
        <Block name="plan" summary={subQuestions ? plural(subQuestions, "sub-question", "sub-questions") : "planning…"}>
          {problemLines("plan")}
        </Block>
      )}
      {started.has("research") && (
        <Block name="research" summary={researchSummary(events)}>
          {problemLines("research")}
        </Block>
      )}
      {started.has("verify") && (
        <Block name="verify" summary={`${ok} ✓ · ${claims.length - ok} ✗`}>
          {claims.map((c) => (
            <div key={c.claimId} className={`log-line claim ${c.ok ? "claim-ok" : "claim-rejected"}`}>
              <span className="mark" aria-hidden="true">
                {c.ok ? "✓" : "✗"}
              </span>
              <span className="sr-only">{c.ok ? "supported:" : "rejected:"}</span> {c.text}
              {c.reason && <span className="dim"> · {c.reason}</span>}
            </div>
          ))}
          {problemLines("verify")}
          {checking && <div className="log-line dim">{checking}</div>}
        </Block>
      )}
      {started.has("write") && (
        <Block name="write" summary={plural(sectionsWritten(events), "section written", "sections written")}>
          {problemLines("write")}
        </Block>
      )}
      {started.size === 0 && <div className="log-line dim">waiting for the worker…</div>}
    </div>
  );
}
