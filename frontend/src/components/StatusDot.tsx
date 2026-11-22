import { statusTone, statusWord } from "../lib/status";

/** A coloured dot for a run status. `word`: also show the status as text; otherwise it is announced to
 * screen readers only, so the status never rests on colour alone. */
export function StatusDot({ status, word = false }: { status: string; word?: boolean }) {
  const tone = statusTone(status);
  return (
    <span className={`status tone-${tone}`}>
      <span className="dot" aria-hidden="true">
        ●
      </span>
      {word ? <span className="status-word">{statusWord(status)}</span> : <span className="sr-only">{statusWord(status)}</span>}
    </span>
  );
}
