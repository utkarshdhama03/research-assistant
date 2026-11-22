import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, useLocation } from "react-router-dom";
import { describe, expect, it } from "vitest";

import { App } from "./App";
import { apiFetch } from "./api/client";
import { mockFetch } from "./test/helpers";

const LOGGED_OUT = { auth_enabled: true, authenticated: false };
const LOGGED_IN = { auth_enabled: true, authenticated: true };
const EMPTY_PAGE = { items: [], next_cursor: null };

function Where() {
  const location = useLocation();
  return <div data-testid="where">{location.pathname + location.search}</div>;
}

function renderApp(initial: string) {
  return render(
    <MemoryRouter initialEntries={[initial]}>
      <App />
      <Where />
    </MemoryRouter>,
  );
}

const where = () => screen.getByTestId("where").textContent;

describe("login", () => {
  it("sends a logged-out user to /login, remembering where they were going", async () => {
    mockFetch({ "GET /auth/session": { body: LOGGED_OUT }, "GET /health": { body: { status: "ok" } } });
    renderApp("/run/r1");
    await waitFor(() => expect(where()).toBe("/login?next=%2Frun%2Fr1"));
    expect(screen.getByLabelText("access key")).toHaveProperty("type", "password");
    expect(screen.queryByRole("button", { name: "log out" })).toBeNull();
  });

  it("shows a wrong key without leaving the page, then logs in and returns to next", async () => {
    let attempts = 0;
    const { calls } = mockFetch({
      "GET /auth/session": { body: LOGGED_OUT },
      "GET /health": { body: { status: "ok" } },
      "POST /auth/login": () =>
        ++attempts === 1
          ? { status: 401, body: { error: { code: "unauthorized", message: "wrong access key" } } }
          : { body: LOGGED_IN },
      "GET /runs?limit=20": { body: EMPTY_PAGE },
    });
    renderApp("/login?next=%2F%3Fq%3Dbees");
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("access key"), "not-the-key");
    await user.click(screen.getByRole("button", { name: "log in" }));
    expect((await screen.findByRole("alert")).textContent).toBe("Wrong access key.");
    expect(where()).toBe("/login?next=%2F%3Fq%3Dbees");
    await user.clear(screen.getByLabelText("access key"));
    await user.type(screen.getByLabelText("access key"), "the-right-key");
    await user.click(screen.getByRole("button", { name: "log in" }));
    await waitFor(() => expect(where()).toBe("/?q=bees"));
    const posts = calls.filter((c) => c.method === "POST");
    expect(posts.map((c) => c.body)).toEqual([{ key: "not-the-key" }, { key: "the-right-key" }]);
    expect(posts.every((c) => c.credentials === "include" && c.headers["Content-Type"] === "application/json")).toBe(true);
    expect(await screen.findByRole("button", { name: "log out" })).toBeTruthy();
  });

  it("shows the rate limit message", async () => {
    mockFetch({
      "GET /auth/session": { body: LOGGED_OUT },
      "GET /health": { body: { status: "ok" } },
      "POST /auth/login": { status: 429, body: { error: { code: "rate_limited", message: "too many failed logins; try again in 42 s" } } },
    });
    renderApp("/login");
    const user = userEvent.setup();
    await user.type(await screen.findByLabelText("access key"), "x");
    await user.click(screen.getByRole("button", { name: "log in" }));
    expect((await screen.findByRole("alert")).textContent).toBe("too many failed logins; try again in 42 s");
  });

  it("logs out and returns to the login page", async () => {
    const { calls } = mockFetch({
      "GET /auth/session": { body: LOGGED_IN },
      "GET /health": { body: { status: "ok" } },
      "GET /runs?limit=20": { body: EMPTY_PAGE },
      "POST /auth/logout": { body: { auth_enabled: true, authenticated: false } },
    });
    renderApp("/");
    await userEvent.setup().click(await screen.findByRole("button", { name: "log out" }));
    await waitFor(() => expect(where()).toBe("/login?next=%2F"));
    expect(calls.some((c) => c.method === "POST" && c.path === "/auth/logout")).toBe(true);
    expect(screen.getByLabelText("access key")).toBeTruthy();
  });

  it("goes to the login page when any API call answers 401 (session expired)", async () => {
    mockFetch({
      "GET /auth/session": { body: LOGGED_IN },
      "GET /health": { body: { status: "ok" } },
      "GET /runs?limit=20": { status: 401, body: { error: { code: "unauthorized", message: "log in first" } } },
    });
    renderApp("/?q=x");
    await waitFor(() => expect(where()).toBe("/login?next=%2F%3Fq%3Dx"));
  });

  it("has no login page and no logout button when login is off", async () => {
    mockFetch({
      "GET /auth/session": { body: { auth_enabled: false, authenticated: true } },
      "GET /health": { body: { status: "ok" } },
      "GET /runs?limit=20": { body: EMPTY_PAGE },
    });
    renderApp("/login");
    await waitFor(() => expect(where()).toBe("/"));
    expect(screen.queryByRole("button", { name: "log out" })).toBeNull();
  });

  it("a 401 from the login call itself does not count as a lost session", async () => {
    mockFetch({ "POST /auth/login": { status: 401, body: { error: { code: "unauthorized", message: "wrong access key" } } } });
    const error = await apiFetch("/auth/login", { method: "POST", body: { key: "x" }, quiet401: true }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(Error);
  });
});
