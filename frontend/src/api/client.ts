// The one fetch wrapper. POSTs always send JSON (the API answers 415 otherwise); errors become ApiError,
// built from the API's shape {"error": {"code", "message", "details"?}}. The session cookie goes with
// every request; a 401 calls the unauthorized handler (the app sends the user to /login).

// The built UI is served by the API itself (same origin); the Vite dev server talks to the API on :8000.
const DEFAULT_BASE = import.meta.env.DEV ? "http://127.0.0.1:8000" : "";
const configured = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
export const API_BASE: string = (configured || DEFAULT_BASE).replace(/\/+$/, "");

let unauthorizedHandler: (() => void) | null = null;

/** Called on any 401 except from the login call itself; null removes it. */
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  unauthorizedHandler = handler;
}

export interface ErrorDetail {
  loc: (string | number)[];
  msg: string;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: ErrorDetail[];

  constructor(status: number, code: string, message: string, details: ErrorDetail[] = []) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

export interface ApiResponse<T> {
  status: number;
  data: T;
}

interface Options {
  method?: "GET" | "POST";
  body?: unknown; // POST only; {} when omitted
  okStatuses?: number[]; // non-2xx statuses whose body is data, not an error (/health's 503)
  quiet401?: boolean; // a 401 is an answer here (wrong key at login), not a lost session
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function toDetails(value: unknown): ErrorDetail[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((d) => ({
    loc: Array.isArray(d.loc)
      ? d.loc.filter((p): p is string | number => typeof p === "string" || typeof p === "number")
      : [],
    msg: typeof d.msg === "string" ? d.msg : "",
  }));
}

export function toApiError(status: number, body: unknown): ApiError {
  const error = isRecord(body) && isRecord(body.error) ? body.error : null;
  if (error && typeof error.message === "string") {
    const code = typeof error.code === "string" ? error.code : "error";
    return new ApiError(status, code, error.message, toDetails(error.details));
  }
  return new ApiError(status, "error", `The API answered HTTP ${status}.`);
}

export async function apiFetch<T>(path: string, options: Options = {}): Promise<ApiResponse<T>> {
  const method = options.method ?? "GET";
  const headers: Record<string, string> = { Accept: "application/json" };
  const init: RequestInit = { method, headers, credentials: "include" };
  if (method === "POST") {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body ?? {});
  }
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, init);
  } catch {
    throw new ApiError(0, "network_error", `Cannot reach the API at ${API_BASE || window.location.origin}.`);
  }
  const text = await response.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }
  if (response.ok || options.okStatuses?.includes(response.status)) {
    return { status: response.status, data: body as T };
  }
  if (response.status === 401 && !options.quiet401) unauthorizedHandler?.();
  throw toApiError(response.status, body);
}
