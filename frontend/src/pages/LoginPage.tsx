import { useState, type FormEvent } from "react";

import { login } from "../api/auth";
import { ApiError } from "../api/client";
import { ErrorMessage } from "../components/ErrorMessage";

/** The access key (ACCESS_KEY in .env) as a password field. After login the app returns to ?next=. */
export function LoginPage({ onLoggedIn }: { onLoggedIn: () => void }) {
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  const submit = async (e: FormEvent): Promise<void> => {
    e.preventDefault();
    if (!key) return;
    setBusy(true);
    setError(null);
    try {
      await login(key);
      setKey("");
      onLoggedIn();
    } catch (err) {
      setError(err instanceof ApiError && err.status === 401 ? new Error("Wrong access key.") : err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="login">
      <h1>
        <span className="hash" aria-hidden="true">
          ##{" "}
        </span>
        log in
      </h1>
      <form onSubmit={(e) => void submit(e)}>
        <label htmlFor="access-key">access key</label>
        <input
          id="access-key"
          type="password"
          autoComplete="current-password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          autoFocus
        />
        <button type="submit" className="primary" disabled={busy || !key}>
          {busy ? "logging in…" : "log in"}
        </button>
      </form>
      <ErrorMessage error={error} />
      <p className="dim">The key is ACCESS_KEY in the server's .env file.</p>
    </section>
  );
}
