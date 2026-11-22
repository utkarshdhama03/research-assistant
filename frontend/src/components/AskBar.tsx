import { useState, type FormEvent } from "react";

import { createRun } from "../api/runs";
import { ErrorMessage } from "./ErrorMessage";

export const QUESTION_MIN = 5;
export const QUESTION_MAX = 500; // the server's default API_QUESTION_MAX_CHARS; its 422 is shown if it differs

/** "> ask a new question": Enter starts a run (it waits for plan approval unless auto-approve is on). */
export function AskBar({ onCreated }: { onCreated: (id: string) => void }) {
  const [question, setQuestion] = useState("");
  const [autoApprove, setAutoApprove] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (event: FormEvent): Promise<void> => {
    event.preventDefault();
    if (busy) return;
    setError(null);
    const trimmed = question.trim();
    if (trimmed.length < QUESTION_MIN || trimmed.length > QUESTION_MAX) {
      setProblem(`The question needs ${QUESTION_MIN} to ${QUESTION_MAX} characters (now ${trimmed.length}).`);
      return;
    }
    setProblem(null);
    setBusy(true);
    try {
      const created = await createRun(trimmed, autoApprove);
      setQuestion("");
      onCreated(created.id);
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className="ask-bar" onSubmit={(e) => void submit(e)} noValidate>
      <div className="ask-row">
        <span className="prompt" aria-hidden="true">
          &gt;
        </span>
        <input
          className="ask-input"
          aria-label="ask a new question"
          placeholder="ask a new question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          aria-invalid={problem !== null}
          aria-describedby={problem ? "ask-problem" : undefined}
          disabled={busy}
        />
        <label className="toggle">
          <input type="checkbox" checked={autoApprove} onChange={(e) => setAutoApprove(e.target.checked)} />
          auto-approve
        </label>
        <button type="submit" disabled={busy}>
          {busy ? "starting…" : "run"}
        </button>
      </div>
      {problem && (
        <div id="ask-problem" className="error" role="alert">
          {problem}
        </div>
      )}
      <ErrorMessage error={error} />
    </form>
  );
}
