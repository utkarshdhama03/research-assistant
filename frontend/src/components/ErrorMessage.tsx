import { ApiError } from "../api/client";

/** An API error as the server reported it: message, then each detail (location: message). */
export function ErrorMessage({ error }: { error: unknown }) {
  if (error === null || error === undefined) return null;
  if (error instanceof ApiError) {
    return (
      <div className="error" role="alert">
        <span>{error.message}</span>
        {error.details.length > 0 && (
          <ul>
            {error.details.map((d, i) => (
              <li key={i}>
                {d.loc.length > 0 && <code>{d.loc.join(".")}</code>} {d.msg}
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }
  return (
    <div className="error" role="alert">
      {error instanceof Error ? error.message : String(error)}
    </div>
  );
}
