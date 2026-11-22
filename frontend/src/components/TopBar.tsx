import { Link } from "react-router-dom";

import { HealthIndicator } from "./HealthIndicator";

export function TopBar({
  historyToggle,
  onLogout,
  healthKey,
}: {
  historyToggle: { open: boolean; onToggle: () => void } | null; // shown below 800 px
  onLogout: (() => void) | null;
  healthKey: string;
}) {
  return (
    <header className="topbar">
      <Link to="/" className="brand">
        research-assistant
      </Link>
      <div className="topbar-right">
        {historyToggle && (
          <button
            type="button"
            aria-expanded={historyToggle.open}
            aria-controls="history-pane"
            onClick={historyToggle.onToggle}
          >
            history
          </button>
        )}
        <HealthIndicator key={healthKey} />
        {onLogout && (
          <button type="button" onClick={onLogout}>
            log out
          </button>
        )}
      </div>
    </header>
  );
}
