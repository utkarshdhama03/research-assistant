import { useEffect, useState } from "react";

import { EVENT_NAMES } from "../api/eventNames";
import { eventsUrl } from "../api/runs";
import type { RunEvent } from "../api/types";

export interface RunEvents {
  events: RunEvent[];
  endStatus: string | null; // the run's final status, from the stream's `end` event
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * The run's events from GET /runs/{id}/events (server-sent events). The server replays all events, then
 * streams new ones; after a reconnect the browser sends Last-Event-ID, and any event seen before is
 * dropped by id. The stream is closed on `end` and when the component unmounts.
 */
export function useRunEvents(runId: string | undefined): RunEvents {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [endStatus, setEndStatus] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) return undefined;
    setEvents([]);
    setEndStatus(null);
    const seen = new Set<string>();
    const source = new EventSource(eventsUrl(runId), { withCredentials: true }); // cookie, also from Vite

    const onEvent = (raw: Event): void => {
      const message = raw as MessageEvent<string>;
      const id = message.lastEventId;
      if (!id || seen.has(id)) return;
      let data: unknown;
      try {
        data = JSON.parse(message.data);
      } catch {
        return;
      }
      if (!isRecord(data)) return;
      seen.add(id);
      const event: RunEvent = {
        id,
        node: typeof data.node === "string" ? data.node : "",
        event: typeof data.event === "string" ? data.event : message.type,
        payload: isRecord(data.payload) ? data.payload : {},
        created_at: typeof data.created_at === "string" ? data.created_at : "",
      };
      setEvents((previous) => [...previous, event]);
    };

    const onEnd = (raw: Event): void => {
      let status = "ended";
      try {
        const data: unknown = JSON.parse((raw as MessageEvent<string>).data);
        if (isRecord(data) && typeof data.status === "string") status = data.status;
      } catch {
        // keep "ended"
      }
      setEndStatus(status);
      source.close();
    };

    for (const name of EVENT_NAMES) source.addEventListener(name, onEvent);
    source.addEventListener("end", onEnd);
    return () => {
      source.close();
    };
  }, [runId]);

  return { events, endStatus };
}
