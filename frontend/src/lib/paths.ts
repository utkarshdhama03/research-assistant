// UI paths. A run's page is /run/<id>, not /runs/<id>: the built UI is served by the API on the same
// origin, where /runs/<id> is the API's JSON route (a reload there would show JSON).
export function runPath(id: string): string {
  return `/run/${encodeURIComponent(id)}`;
}

/** Where to go after login: a path on this site only (no //host or scheme), else "/". */
export function safeNext(value: string | null): string {
  if (!value || !value.startsWith("/") || value.startsWith("//") || value.startsWith("/\\")) return "/";
  if (value === "/login" || value.startsWith("/login?")) return "/";
  return value;
}
