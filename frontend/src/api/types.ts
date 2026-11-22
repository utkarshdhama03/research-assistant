// Mirrors src/research_assistant/api/schemas.py.

export interface RunSummary {
  id: string;
  question: string;
  status: string; // queued, running, awaiting_approval, completed, partial, failed, cancelled
  reason: string | null;
  auto_approve: boolean;
  created_at: string;
  finished_at: string | null;
}

export interface RunPage {
  items: RunSummary[];
  next_cursor: string | null;
}

export interface SubQuestion {
  ordinal: number;
  question: string;
  search_queries: string[];
  premise_check: boolean;
}

export interface NodeTiming {
  node: string;
  event: string;
  duration_s: number;
}

export interface RunDetail extends RunSummary {
  cancel_requested: boolean;
  sub_questions: SubQuestion[];
  node_timings: NodeTiming[];
  claims_by_verdict: Record<string, number>;
  report_markdown: string | null;
}

export interface RunCreated {
  id: string;
  status: string;
}

export interface RunAction {
  id: string;
  status: string;
  cancel_requested: boolean;
}

export interface EditedSubQuestion {
  question: string;
  search_queries: string[];
  premise_check: boolean;
}

export interface CiteRef {
  n: number;
  claim_ids: string[];
}

export type SentencePart = { text: string } | { cite: CiteRef[] };

export interface Sentence {
  text: string;
  citations: number[];
  parts: SentencePart[];
}

export type SectionKind = "normal" | "insufficient" | "covered" | "no_sources" | "failed";

export interface ReportSection {
  ordinal: number;
  heading: string;
  kind: SectionKind;
  note: string | null;
  sentences: Sentence[];
}

export interface Reference {
  n: number;
  title: string;
  url: string;
  retrieved_at: string | null;
  reused: boolean;
}

export interface CitedClaim {
  claim_id: string;
  ordinal: number;
  text: string;
  verdict: string;
  quote: string;
}

export interface StructuredReport {
  report_id: string;
  created_at: string;
  premise_note: string | null;
  sections: ReportSection[];
  references: Reference[];
  claims_by_n: Record<string, CitedClaim[]>;
}

export interface LegacyReport {
  report_id: string;
  created_at: string;
  legacy_markdown: string;
}

export type Report = StructuredReport | LegacyReport;

export function isLegacyReport(report: Report): report is LegacyReport {
  return "legacy_markdown" in report;
}

export interface SourceRow {
  id: string;
  url: string;
  final_url: string | null;
  title: string | null;
  status: string; // fetched, failed, skipped
  reason: string | null;
  retrieved_at: string | null;
  reused: boolean;
  reused_from_run_id: string | null;
  cited: boolean;
}

export interface ComponentHealth {
  ok: boolean;
  detail?: string | null;
}

/** Without a session (login on) /health has the status only; the components need a session. */
export interface Health {
  status: "ok" | "degraded";
  db?: ComponentHealth;
  ollama?: ComponentHealth;
  searxng?: ComponentHealth;
  worker?: ComponentHealth;
}

export interface SessionInfo {
  auth_enabled: boolean;
  authenticated: boolean;
}

export interface RunEvent {
  id: string;
  node: string;
  event: string;
  payload: Record<string, unknown>;
  created_at: string;
}
