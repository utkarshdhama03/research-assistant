import { useCallback, useEffect, useState } from "react";
import { Navigate, Route, Routes, useLocation, useMatch, useNavigate, useSearchParams } from "react-router-dom";

import { getSession, logout } from "./api/auth";
import { setUnauthorizedHandler } from "./api/client";
import type { SessionInfo } from "./api/types";
import { AskBar } from "./components/AskBar";
import { ErrorMessage } from "./components/ErrorMessage";
import { HistoryPane } from "./components/HistoryPane";
import { RunView } from "./components/RunView";
import { TopBar } from "./components/TopBar";
import { useMediaQuery } from "./hooks/useMediaQuery";
import { runPath, safeNext } from "./lib/paths";
import { LoginPage } from "./pages/LoginPage";

// Below this width the history pane collapses behind the "history" button.
export const NARROW_QUERY = "(max-width: 799px)";

function RunRoute({ onStatus }: { onStatus: (status: string) => void }) {
  const match = useMatch("/run/:runId");
  const runId = match?.params.runId ?? "";
  return <RunView key={runId} runId={runId} onStatus={onStatus} />;
}

function Welcome() {
  return (
    <div className="welcome dim">
      <p>Pick a run from the history, or ask a new question below.</p>
      <p>A new run plans 3–5 sub-questions and waits for your approval, unless auto-approve is on.</p>
    </div>
  );
}

export function App() {
  const location = useLocation();
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const selected = useMatch("/run/:runId")?.params.runId ?? null;
  const narrow = useMediaQuery(NARROW_QUERY);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [session, setSession] = useState<SessionInfo | null>(null);
  const [logoutError, setLogoutError] = useState<unknown>(null);

  useEffect(() => {
    // Any 401 (expired session, API restarted): back to the login page, then return here.
    setUnauthorizedHandler(() =>
      setSession((s) => ({ auth_enabled: s?.auth_enabled ?? true, authenticated: false })),
    );
    let active = true;
    getSession()
      .then((s) => active && setSession(s))
      .catch(() => active && setSession({ auth_enabled: false, authenticated: true })); // pages show the error
    return () => {
      active = false;
      setUnauthorizedHandler(null);
    };
  }, []);

  const refreshHistory = useCallback(() => setRefreshKey((k) => k + 1), []);

  const onLogout = async (): Promise<void> => {
    setLogoutError(null);
    try {
      await logout();
      setSession({ auth_enabled: true, authenticated: false }); // the redirect below goes to /login
    } catch (e) {
      setLogoutError(e);
    }
  };

  const needsLogin = session !== null && session.auth_enabled && !session.authenticated;
  const onLoginPage = location.pathname === "/login";
  if (needsLogin && !onLoginPage) {
    const next = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?next=${next}`} replace />;
  }

  const loggedIn = session?.auth_enabled === true && session.authenticated;
  const healthKey = session?.authenticated ? "in" : "out";

  if (onLoginPage || session === null) {
    return (
      <div className="app app-login">
        <TopBar historyToggle={null} onLogout={null} healthKey={healthKey} />
        <main className="main">
          {session === null ? (
            <p className="dim">loading…</p>
          ) : needsLogin ? (
            <LoginPage onLoggedIn={() => setSession({ auth_enabled: true, authenticated: true })} />
          ) : (
            <Navigate to={safeNext(params.get("next"))} replace /> // logged in, or login is off
          )}
        </main>
      </div>
    );
  }

  const showHistory = !narrow || historyOpen;
  return (
    <div className={`app${narrow ? " narrow" : ""}`}>
      <TopBar
        historyToggle={narrow ? { open: historyOpen, onToggle: () => setHistoryOpen((v) => !v) } : null}
        onLogout={loggedIn ? () => void onLogout() : null}
        healthKey={healthKey}
      />
      <div className="body">
        {showHistory && (
          <div id="history-pane" className={narrow ? "history-pane overlay" : "history-pane"}>
            <HistoryPane
              selectedId={selected}
              refreshKey={refreshKey}
              onPick={() => {
                if (narrow) setHistoryOpen(false);
              }}
            />
          </div>
        )}
        <main className="main">
          <ErrorMessage error={logoutError} />
          <Routes>
            <Route path="/" element={<Welcome />} />
            <Route path="/run/:runId" element={<RunRoute onStatus={refreshHistory} />} />
            <Route path="*" element={<p className="dim">page not found.</p>} />
          </Routes>
        </main>
      </div>
      <AskBar
        onCreated={(id) => {
          refreshHistory();
          navigate(runPath(id));
        }}
      />
    </div>
  );
}
