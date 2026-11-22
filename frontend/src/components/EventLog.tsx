import type { RunEvent } from "../api/types";
import { formatEvent } from "../lib/eventFormat";

/** Every event of the run, one line each; event types this UI does not know are dimmed. */
export function EventLog({ events }: { events: RunEvent[] }) {
  if (events.length === 0) return <p className="dim">No events.</p>;
  return (
    <ol className="event-log" aria-label="event log">
      {events.map((e) => {
        const line = formatEvent(e);
        return (
          <li key={line.id} className={line.known ? "event" : "event unknown"} title={line.known ? undefined : "unknown event type"}>
            <span className="dim">{line.time}</span> <span className="node">{line.node}</span>{" "}
            <span className="name">{line.event}</span> <span className="text">{line.text}</span>
          </li>
        );
      })}
    </ol>
  );
}
