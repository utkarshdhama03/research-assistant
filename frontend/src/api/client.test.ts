import { describe, expect, it } from "vitest";

import { mockFetch } from "../test/helpers";
import { ApiError, apiFetch } from "./client";
import { cancelRun, getHealth } from "./runs";

describe("apiFetch", () => {
  it("sends JSON with Content-Type on every POST, {} when there is no body", async () => {
    const { calls } = mockFetch({ "POST /runs/r1/cancel": { body: { id: "r1", status: "cancelled" } } });
    await cancelRun("r1");
    expect(calls[0]?.headers["Content-Type"]).toBe("application/json");
    expect(calls[0]?.body).toEqual({});
  });

  it("turns the API error shape into ApiError with its details", async () => {
    mockFetch({
      "POST /runs": {
        status: 422,
        body: { error: { code: "validation_error", message: "invalid", details: [{ loc: ["body", "question"], msg: "too short" }] } },
      },
    });
    const error = await apiFetch("/runs", { method: "POST", body: { question: "x" } }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    const api = error as ApiError;
    expect([api.status, api.code, api.message]).toEqual([422, "validation_error", "invalid"]);
    expect(api.details).toEqual([{ loc: ["body", "question"], msg: "too short" }]);
  });

  it("falls back to a generic message for a body that is not the error shape", async () => {
    mockFetch({ "GET /runs/x": { status: 502, body: "gateway" } });
    const error = (await apiFetch("/runs/x").catch((e: unknown) => e)) as ApiError;
    expect([error.status, error.message]).toEqual([502, "The API answered HTTP 502."]);
  });

  it("returns /health's 503 body as data", async () => {
    const degraded = {
      status: "degraded",
      db: { ok: true },
      ollama: { ok: false, detail: "down" },
      searxng: { ok: true },
      worker: { ok: true },
    };
    mockFetch({ "GET /health": { status: 503, body: degraded } });
    expect(await getHealth()).toEqual(degraded);
  });
});
