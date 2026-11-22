import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { mockFetch } from "../test/helpers";
import { HealthIndicator, summarize } from "./HealthIndicator";

const up = { ok: true };
const down = (detail: string) => ({ ok: false, detail });

describe("health summary", () => {
  it("is green when all is ok, amber naming what is down, red when the database is down", () => {
    expect(summarize({ status: "ok", db: up, ollama: up, searxng: up, worker: up }, null)).toMatchObject({
      tone: "ok",
      text: "all systems ok",
    });
    expect(summarize({ status: "degraded", db: up, ollama: down("x"), searxng: up, worker: up }, null)).toMatchObject({
      tone: "warn",
      text: "ollama down",
    });
    expect(
      summarize({ status: "degraded", db: up, ollama: down("x"), searxng: up, worker: down("y") }, null).text,
    ).toBe("2 down: ollama, worker");
    expect(summarize({ status: "degraded", db: down("x"), ollama: up, searxng: up, worker: up }, null).tone).toBe("bad");
    expect(summarize(null, "Cannot reach the API.")).toMatchObject({ tone: "bad", text: "api unreachable" });
    expect(summarize({ status: "degraded" }, null)).toMatchObject({ tone: "warn", text: "degraded" }); // no session
  });
});

describe("HealthIndicator", () => {
  it("shows the summary and the per-component details on click", async () => {
    mockFetch({
      "GET /health": { status: 503, body: { status: "degraded", db: up, ollama: down("unreachable (ConnectError)"), searxng: up, worker: up } },
    });
    render(<HealthIndicator />);
    const button = await screen.findByRole("button", { name: /ollama down/ });
    expect(button.className).toContain("tone-warn");
    expect(button.getAttribute("aria-expanded")).toBe("false");
    await userEvent.click(button);
    expect(button.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("ollama: down (unreachable (ConnectError))")).toBeTruthy();
    expect(screen.getByText("db: ok")).toBeTruthy();
  });
});
