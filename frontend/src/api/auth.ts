import { apiFetch } from "./client";
import type { SessionInfo } from "./types";

export async function getSession(): Promise<SessionInfo> {
  return (await apiFetch<SessionInfo>("/auth/session")).data;
}

/** Wrong key: ApiError 401 (no redirect). Too many failures: ApiError 429. */
export async function login(key: string): Promise<SessionInfo> {
  return (await apiFetch<SessionInfo>("/auth/login", { method: "POST", body: { key }, quiet401: true })).data;
}

export async function logout(): Promise<SessionInfo> {
  return (await apiFetch<SessionInfo>("/auth/logout", { method: "POST" })).data;
}
