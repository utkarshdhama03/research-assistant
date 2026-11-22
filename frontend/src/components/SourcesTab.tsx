import { Link } from "react-router-dom";

import type { SourceRow } from "../api/types";
import { formatTime } from "../lib/format";
import { runPath } from "../lib/paths";
import { safeHttpUrl } from "../lib/safeUrl";
import { ErrorMessage } from "./ErrorMessage";
import { ExternalLink } from "./ExternalLink";

export function hostOf(url: string): string {
  const safe = safeHttpUrl(url);
  return safe ? new URL(safe).host : url;
}

/** The failure reason without the URL the server appends ("http_error: HTTP 403 (https://…)"): the host
 * column already says where; the full reason stays in the cell's title. */
export function shortReason(reason: string): string {
  return reason.replace(/\s*\(https?:\/\/[^\s)]*\)\s*$/, "");
}

/** Every source of the run: status (and why it failed), host, retrieved, reused, cited. */
export function SourcesTab({ rows, error }: { rows: SourceRow[] | null; error: unknown }) {
  if (error) return <ErrorMessage error={error} />;
  if (!rows) return <p className="dim">loading sources…</p>;
  if (rows.length === 0) return <p className="dim">no sources.</p>;
  return (
    <table className="sources">
      <thead>
        <tr>
          <th scope="col">status</th>
          <th scope="col">host</th>
          <th scope="col">retrieved</th>
          <th scope="col">reused</th>
          <th scope="col">cited</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((s) => {
          const url = s.final_url ?? s.url;
          return (
            <tr key={s.id} className={`source-${s.status}`}>
              <td className="source-status" title={s.reason ?? undefined}>
                {s.status}
                {s.reason && <div className="dim small">{shortReason(s.reason)}</div>}
              </td>
              <td title={s.title ? `${s.title}\n${url}` : url}>
                <ExternalLink href={url}>{hostOf(url)}</ExternalLink>
                {s.title && <div className="dim small one-line">{s.title}</div>}
              </td>
              <td>{formatTime(s.retrieved_at)}</td>
              <td>
                {s.reused ? (
                  s.reused_from_run_id ? (
                    <Link to={runPath(s.reused_from_run_id)}>yes (run)</Link>
                  ) : (
                    "yes"
                  )
                ) : (
                  "no"
                )}
              </td>
              <td>{s.cited ? "yes" : "no"}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
