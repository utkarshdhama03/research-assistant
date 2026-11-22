import { useMemo, useState } from "react";

import { ApiError } from "../api/client";
import { approveRun, rejectRun } from "../api/runs";
import type { EditedSubQuestion, SubQuestion } from "../api/types";
import {
  MAX_QUERIES,
  MAX_SUB_QUESTIONS,
  MIN_QUERIES,
  MIN_SUB_QUESTIONS,
  QUERY_MAX,
  QUESTION_MAX,
  hasErrors,
  queriesKey,
  queryKey,
  questionKey,
  serverErrors,
  validatePlan,
  type PlanErrors,
} from "../lib/planValidation";
import { ErrorMessage } from "./ErrorMessage";

const NO_ERRORS: PlanErrors = { fields: {}, form: [] };

function toEdited(subQuestions: SubQuestion[]): EditedSubQuestion[] {
  return [...subQuestions]
    .sort((a, b) => a.ordinal - b.ordinal)
    .map((s) => ({ question: s.question, search_queries: [...s.search_queries], premise_check: s.premise_check }));
}

function knownFields(plan: EditedSubQuestion[]): Set<string> {
  const keys = new Set<string>();
  plan.forEach((sq, i) => {
    keys.add(questionKey(i));
    keys.add(queriesKey(i));
    sq.search_queries.forEach((_q, j) => keys.add(queryKey(i, j)));
  });
  return keys;
}

export function PlanEditor({
  runId,
  subQuestions,
  onDone,
}: {
  runId: string;
  subQuestions: SubQuestion[];
  onDone: () => void;
}) {
  const initial = useMemo(() => toEdited(subQuestions), [subQuestions]);
  const [plan, setPlan] = useState<EditedSubQuestion[]>(initial);
  const [errors, setErrors] = useState<PlanErrors>(NO_ERRORS);
  const [actionError, setActionError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const edited = JSON.stringify(plan) !== JSON.stringify(initial);

  const update = (next: EditedSubQuestion[]): void => {
    // premise_check stays only on a sub-question that is still first
    setPlan(next.map((sq, i) => ({ ...sq, premise_check: i === 0 && sq.premise_check })));
    setErrors(NO_ERRORS);
  };
  const setQuestion = (i: number, value: string): void =>
    update(plan.map((sq, k) => (k === i ? { ...sq, question: value } : sq)));
  const setQuery = (i: number, j: number, value: string): void =>
    update(
      plan.map((sq, k) =>
        k === i ? { ...sq, search_queries: sq.search_queries.map((q, m) => (m === j ? value : q)) } : sq,
      ),
    );
  const addQuery = (i: number): void =>
    update(plan.map((sq, k) => (k === i ? { ...sq, search_queries: [...sq.search_queries, ""] } : sq)));
  const removeQuery = (i: number, j: number): void =>
    update(
      plan.map((sq, k) => (k === i ? { ...sq, search_queries: sq.search_queries.filter((_q, m) => m !== j) } : sq)),
    );
  const addSubQuestion = (): void =>
    update([...plan, { question: "", search_queries: [""], premise_check: false }]);
  const removeSubQuestion = (i: number): void => update(plan.filter((_sq, k) => k !== i));

  const approve = async (): Promise<void> => {
    setActionError(null);
    let body: EditedSubQuestion[] | null = null;
    if (edited) {
      const found = validatePlan(plan);
      if (hasErrors(found)) {
        setErrors(found);
        return;
      }
      body = plan.map((sq) => ({
        question: sq.question.trim(),
        search_queries: sq.search_queries.map((q) => q.trim()),
        premise_check: sq.premise_check,
      }));
    }
    setBusy(true);
    try {
      await approveRun(runId, body);
      onDone();
    } catch (e) {
      if (e instanceof ApiError && e.status === 422) setErrors(serverErrors(e.details, e.message, knownFields(plan)));
      else setActionError(e);
    } finally {
      setBusy(false);
    }
  };

  const reject = async (): Promise<void> => {
    if (!window.confirm("Reject this plan? The run will be cancelled.")) return;
    setActionError(null);
    setBusy(true);
    try {
      await rejectRun(runId);
      onDone();
    } catch (e) {
      setActionError(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="plan-editor" aria-label="Plan approval">
      <h2>
        <span className="hash" aria-hidden="true">
          ##{" "}
        </span>
        plan
      </h2>
      <p className="dim">
        Edit the sub-questions ({MIN_SUB_QUESTIONS}–{MAX_SUB_QUESTIONS}) and their search queries ({MIN_QUERIES}–
        {MAX_QUERIES} each), then approve. An unchanged plan is approved as it is.
      </p>
      {errors.form.length > 0 && (
        <ul className="error" role="alert">
          {errors.form.map((m, i) => (
            <li key={i}>{m}</li>
          ))}
        </ul>
      )}
      <ol className="sub-questions">
        {plan.map((sq, i) => (
          <li key={i} className="sub-question">
            <label htmlFor={`sq-${i}`} className="sq-label" aria-hidden="true">
              {i + 1}.{sq.premise_check && <span className="tag">premise check</span>}
            </label>
            <div className="row">
              <input
                id={`sq-${i}`}
                aria-label={`Sub-question ${i + 1}${sq.premise_check ? " (premise check)" : ""}`}
                value={sq.question}
                maxLength={QUESTION_MAX + 50}
                onChange={(e) => setQuestion(i, e.target.value)}
                aria-invalid={questionKey(i) in errors.fields}
              />
              <button
                type="button"
                onClick={() => removeSubQuestion(i)}
                disabled={plan.length <= MIN_SUB_QUESTIONS}
                aria-label={`Remove sub-question ${i + 1}`}
              >
                ×
              </button>
            </div>
            {errors.fields[questionKey(i)] && <div className="field-error">{errors.fields[questionKey(i)]}</div>}
            <ul className="queries">
              {sq.search_queries.map((q, j) => (
                <li key={j}>
                  <div className="row query-row">
                    <span className="dim" aria-hidden="true">
                      q:
                    </span>
                    <input
                      value={q}
                      maxLength={QUERY_MAX + 50}
                      aria-label={`Sub-question ${i + 1} query ${j + 1}`}
                      onChange={(e) => setQuery(i, j, e.target.value)}
                      aria-invalid={queryKey(i, j) in errors.fields}
                    />
                    <button
                      type="button"
                      onClick={() => removeQuery(i, j)}
                      disabled={sq.search_queries.length <= MIN_QUERIES}
                      aria-label={`Remove query ${j + 1} of sub-question ${i + 1}`}
                    >
                      ×
                    </button>
                  </div>
                  {errors.fields[queryKey(i, j)] && (
                    <div className="field-error">{errors.fields[queryKey(i, j)]}</div>
                  )}
                </li>
              ))}
            </ul>
            {errors.fields[queriesKey(i)] && <div className="field-error">{errors.fields[queriesKey(i)]}</div>}
            <button
              type="button"
              onClick={() => addQuery(i)}
              disabled={sq.search_queries.length >= MAX_QUERIES}
              aria-label={`Add query to sub-question ${i + 1}`}
              className="add"
            >
              + query
            </button>
          </li>
        ))}
      </ol>
      <button type="button" className="add" onClick={addSubQuestion} disabled={plan.length >= MAX_SUB_QUESTIONS}>
        + sub-question
      </button>
      <ErrorMessage error={actionError} />
      <div className="actions">
        <button type="button" className="primary" onClick={() => void approve()} disabled={busy}>
          {edited ? "approve edited plan" : "approve"}
        </button>
        <button type="button" className="danger" onClick={() => void reject()} disabled={busy}>
          reject
        </button>
      </div>
    </section>
  );
}
