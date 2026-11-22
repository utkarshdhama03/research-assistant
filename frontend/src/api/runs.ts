import { API_BASE, apiFetch } from "./client";
import type {
  EditedSubQuestion,
  Health,
  Report,
  RunAction,
  RunCreated,
  RunDetail,
  RunPage,
  SourceRow,
} from "./types";

const enc = encodeURIComponent;
export const PAGE_SIZE = 20;

export async function listRuns(params: { cursor?: string | null; q?: string | null } = {}): Promise<RunPage> {
  const query = new URLSearchParams({ limit: String(PAGE_SIZE) });
  if (params.cursor) query.set("cursor", params.cursor);
  if (params.q) query.set("q", params.q);
  return (await apiFetch<RunPage>(`/runs?${query.toString()}`)).data;
}

export async function getRun(id: string): Promise<RunDetail> {
  return (await apiFetch<RunDetail>(`/runs/${enc(id)}`)).data;
}

export async function createRun(question: string, autoApprove: boolean): Promise<RunCreated> {
  const body = { question, auto_approve: autoApprove };
  return (await apiFetch<RunCreated>("/runs", { method: "POST", body })).data;
}

/** subQuestions null: approve the plan as it is. */
export async function approveRun(id: string, subQuestions: EditedSubQuestion[] | null): Promise<RunAction> {
  const body = subQuestions === null ? {} : { sub_questions: subQuestions };
  return (await apiFetch<RunAction>(`/runs/${enc(id)}/approve`, { method: "POST", body })).data;
}

export async function rejectRun(id: string): Promise<RunAction> {
  return (await apiFetch<RunAction>(`/runs/${enc(id)}/reject`, { method: "POST" })).data;
}

export async function cancelRun(id: string): Promise<RunAction> {
  return (await apiFetch<RunAction>(`/runs/${enc(id)}/cancel`, { method: "POST" })).data;
}

export async function getReport(id: string): Promise<Report> {
  return (await apiFetch<Report>(`/runs/${enc(id)}/report`)).data;
}

export async function getSources(id: string): Promise<SourceRow[]> {
  return (await apiFetch<SourceRow[]>(`/runs/${enc(id)}/sources`)).data;
}

/** /health answers 503 with the same body when degraded: data, not an error. */
export async function getHealth(): Promise<Health> {
  return (await apiFetch<Health>("/health", { okStatuses: [503] })).data;
}

export function eventsUrl(id: string): string {
  return `${API_BASE}/runs/${enc(id)}/events`;
}
