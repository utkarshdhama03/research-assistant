import { useCallback, useEffect, useState } from "react";

import { getReport } from "../api/runs";
import { isLegacyReport, type CiteRef, type Report, type StructuredReport } from "../api/types";
import { CitationPopover } from "./CitationPopover";
import { ErrorMessage } from "./ErrorMessage";
import { ExternalLink } from "./ExternalLink";

const KIND_LABELS: Record<string, string> = {
  insufficient: "insufficient evidence",
  covered: "covered above",
  no_sources: "no own sources",
  failed: "not written",
};

function Marker({ cite, report, id }: { cite: CiteRef; report: StructuredReport; id: string }) {
  const [open, setOpen] = useState(false);
  const close = useCallback(() => setOpen(false), []);
  const claims = (report.claims_by_n[String(cite.n)] ?? []).filter((c) => cite.claim_ids.includes(c.claim_id));
  const reference = report.references.find((r) => r.n === cite.n);
  return (
    <span className="marker-wrap">
      <button
        type="button"
        className="marker"
        aria-label={`citation ${cite.n}`}
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        onClick={() => setOpen((v) => !v)}
      >
        {cite.n}
      </button>
      {open && (
        <span id={id}>
          <CitationPopover n={cite.n} claims={claims} reference={reference} onClose={close} />
        </span>
      )}
    </span>
  );
}

export function StructuredReportView({ report }: { report: StructuredReport }) {
  const evidence = Object.entries(report.claims_by_n)
    .flatMap(([n, claims]) => claims.map((c) => ({ ...c, n: Number(n) })))
    .sort((a, b) => a.ordinal - b.ordinal);
  return (
    <article className="report">
      {report.premise_note && <p className="premise-note">{report.premise_note}</p>}
      {report.sections.map((section) => (
        <section key={section.ordinal} className={`report-section kind-${section.kind}`}>
          <h2>
            <span className="hash" aria-hidden="true">
              ##{" "}
            </span>
            {section.ordinal}. {section.heading}
            {section.kind !== "normal" && <span className="tag">{KIND_LABELS[section.kind] ?? section.kind}</span>}
          </h2>
          {section.note && <p className="note">{section.note}</p>}
          {section.sentences.length > 0 && (
            <div className="section-body">
              {section.sentences.map((sentence, si) =>
                sentence.parts.map((part, pi) =>
                  "text" in part ? (
                    <span key={`${si}-${pi}`}>{part.text}</span>
                  ) : (
                    part.cite.map((cite) => (
                      <Marker
                        key={`${si}-${pi}-${cite.n}`}
                        cite={cite}
                        report={report}
                        id={`cite-${section.ordinal}-${si}-${pi}-${cite.n}`}
                      />
                    ))
                  ),
                ),
              )}
            </div>
          )}
        </section>
      ))}
      <section className="references">
        <h2>
          <span className="hash" aria-hidden="true">
            ##{" "}
          </span>
          references
        </h2>
        {report.references.length === 0 && <p className="dim">No sources were cited.</p>}
        <ol>
          {report.references.map((ref) => (
            <li key={ref.n} value={ref.n}>
              <ExternalLink href={ref.url}>{ref.title}</ExternalLink>
              {ref.retrieved_at && <span className="dim"> · retrieved {ref.retrieved_at}</span>}
              {ref.reused && <span className="tag">reused</span>}
            </li>
          ))}
        </ol>
      </section>
      <section className="evidence">
        <h2>
          <span className="hash" aria-hidden="true">
            ##{" "}
          </span>
          evidence
        </h2>
        {evidence.length === 0 && <p className="dim">No claims were cited.</p>}
        <ul>
          {evidence.map((c) => (
            <li key={c.claim_id}>
              <strong>Claim {c.ordinal}</strong> [{c.n}]: {c.text}
              <blockquote>{c.quote}</blockquote>
            </li>
          ))}
        </ul>
      </section>
    </article>
  );
}

export function ReportView({ runId }: { runId: string }) {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    let active = true;
    getReport(runId)
      .then((r) => active && setReport(r))
      .catch((e: unknown) => active && setError(e));
    return () => {
      active = false;
    };
  }, [runId]);

  if (error) return <ErrorMessage error={error} />;
  if (!report) return <p className="dim">loading report…</p>;
  if (isLegacyReport(report)) {
    return (
      <div className="legacy-report">
        <p className="dim">This report was written before structured reports and is shown as plain text.</p>
        <pre>{report.legacy_markdown}</pre>
      </div>
    );
  }
  return <StructuredReportView report={report} />;
}
