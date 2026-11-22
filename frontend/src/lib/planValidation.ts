// Client-side plan validation, mirroring the server (llm/schemas.py ResearchPlan, api/runs.py approve):
// 3-5 sub-questions of 5-300 characters, each with 1-3 search queries of 2-200 characters (after trimming),
// no duplicate sub-questions or duplicate queries within a sub-question (case and spacing ignored),
// premise_check on the first sub-question only.

import type { EditedSubQuestion } from "../api/types";

export const MIN_SUB_QUESTIONS = 3;
export const MAX_SUB_QUESTIONS = 5;
export const MIN_QUERIES = 1;
export const MAX_QUERIES = 3;
export const QUESTION_MIN = 5;
export const QUESTION_MAX = 300;
export const QUERY_MIN = 2;
export const QUERY_MAX = 200;

/** Field keys follow the server's error locations: sub_questions.0.question, sub_questions.0.search_queries.1 */
export type FieldErrors = Record<string, string>;

export interface PlanErrors {
  fields: FieldErrors;
  form: string[];
}

const key = (value: string): string => value.toLowerCase().split(/\s+/).filter(Boolean).join(" ");

export function questionKey(i: number): string {
  return `sub_questions.${i}.question`;
}

export function queriesKey(i: number): string {
  return `sub_questions.${i}.search_queries`;
}

export function queryKey(i: number, j: number): string {
  return `sub_questions.${i}.search_queries.${j}`;
}

export function validatePlan(plan: EditedSubQuestion[]): PlanErrors {
  const fields: FieldErrors = {};
  const form: string[] = [];
  if (plan.length < MIN_SUB_QUESTIONS || plan.length > MAX_SUB_QUESTIONS) {
    form.push(`The plan needs ${MIN_SUB_QUESTIONS} to ${MAX_SUB_QUESTIONS} sub-questions.`);
  }
  const seenQuestions = new Set<string>();
  plan.forEach((sq, i) => {
    const question = sq.question.trim();
    if (question.length < QUESTION_MIN || question.length > QUESTION_MAX) {
      fields[questionKey(i)] = `A sub-question needs ${QUESTION_MIN} to ${QUESTION_MAX} characters.`;
    } else if (seenQuestions.has(key(question))) {
      fields[questionKey(i)] = "This sub-question repeats an earlier one.";
    }
    seenQuestions.add(key(question));
    if (sq.search_queries.length < MIN_QUERIES || sq.search_queries.length > MAX_QUERIES) {
      fields[queriesKey(i)] = `Each sub-question needs ${MIN_QUERIES} to ${MAX_QUERIES} search queries.`;
    }
    const seenQueries = new Set<string>();
    sq.search_queries.forEach((raw, j) => {
      const query = raw.trim();
      if (query.length < QUERY_MIN || query.length > QUERY_MAX) {
        fields[queryKey(i, j)] = `A search query needs ${QUERY_MIN} to ${QUERY_MAX} characters.`;
      } else if (seenQueries.has(key(query))) {
        fields[queryKey(i, j)] = "This query repeats another query of this sub-question.";
      }
      seenQueries.add(key(query));
    });
    if (sq.premise_check && i !== 0) {
      form.push("Only the first sub-question may check the question's premise.");
    }
  });
  return { fields, form };
}

export function hasErrors(errors: PlanErrors): boolean {
  return errors.form.length > 0 || Object.keys(errors.fields).length > 0;
}

/** The server's 422 details as field errors (location without the leading "body") and form messages. */
export function serverErrors(
  details: { loc: (string | number)[]; msg: string }[],
  message: string,
  knownFields: Set<string>,
): PlanErrors {
  const fields: FieldErrors = {};
  const form: string[] = [];
  for (const detail of details) {
    const loc = detail.loc[0] === "body" ? detail.loc.slice(1) : detail.loc;
    const field = loc.join(".");
    if (knownFields.has(field)) {
      fields[field] = detail.msg;
    } else {
      form.push(loc.length ? `${field}: ${detail.msg}` : detail.msg);
    }
  }
  if (details.length === 0) form.push(message);
  return { fields, form };
}
