import { vi } from "vitest";

import { API_BASE } from "../api/client";

export interface Call {
  method: string;
  path: string; // without the API base, with the query string
  headers: Record<string, string>;
  body: unknown;
  credentials: RequestCredentials | undefined;
}

type Reply = { status?: number; body?: unknown } | ((call: Call) => { status?: number; body?: unknown });

/** fetch stub: routes are "METHOD /path" (exact, query included) or "METHOD /path*" (prefix). */
export function mockFetch(routes: Record<string, Reply>): { calls: Call[] } {
  const calls: Call[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    const path = url.startsWith(API_BASE) ? url.slice(API_BASE.length) : url;
    const method = init?.method ?? "GET";
    const headers = (init?.headers ?? {}) as Record<string, string>;
    const body = typeof init?.body === "string" ? (JSON.parse(init.body) as unknown) : undefined;
    const call: Call = { method, path, headers, body, credentials: init?.credentials };
    calls.push(call);
    const key = `${method} ${path}`;
    const match =
      routes[key] ??
      Object.entries(routes).find(([k]) => k.endsWith("*") && key.startsWith(k.slice(0, -1)))?.[1];
    if (match === undefined) {
      return new Response(JSON.stringify({ error: { code: "not_found", message: `no mock for ${key}` } }), {
        status: 404,
      });
    }
    const reply = typeof match === "function" ? match(call) : match;
    return new Response(reply.body === undefined ? "" : JSON.stringify(reply.body), {
      status: reply.status ?? 200,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fn);
  return { calls };
}

/** EventSource stub: tests push named events with ids; `closed` tells whether the page closed it. */
export class MockEventSource {
  static instances: MockEventSource[] = [];
  readonly url: string;
  closed = false;
  private listeners = new Map<string, ((e: Event) => void)[]>();

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  addEventListener(name: string, listener: (e: Event) => void): void {
    this.listeners.set(name, [...(this.listeners.get(name) ?? []), listener]);
  }

  removeEventListener(): void {}

  close(): void {
    this.closed = true;
  }

  emit(name: string, data: unknown, id = ""): void {
    const event = new MessageEvent(name, { data: JSON.stringify(data), lastEventId: id });
    for (const listener of this.listeners.get(name) ?? []) listener(event);
  }

  static install(): void {
    MockEventSource.instances = [];
    vi.stubGlobal("EventSource", MockEventSource);
  }

  static latest(): MockEventSource {
    const last = MockEventSource.instances.at(-1);
    if (!last) throw new Error("no EventSource was opened");
    return last;
  }
}

export function runEvent(event: string, node: string, payload: Record<string, unknown> = {}) {
  return { node, event, payload, created_at: "2026-11-22T10:00:00+00:00" };
}
