"""Phase 0 model benchmark: structured-output reliability and speed of local chat models.

Every number comes from real Ollama calls. Usage:
    uv run python scripts/benchmark_models.py [--models qwen3:8b qwen3:4b llama3.1:8b]

Writes raw per-call results to data/benchmark/<timestamp>.json and prints a Markdown table.
The passages below are hand-written test inputs, not retrieved sources.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field, ValidationError

from research_assistant.config import get_settings


# ---------- schemas ----------
class SearchQueries(BaseModel):
    queries: list[str] = Field(min_length=4, max_length=4)


class Verification(BaseModel):
    verdict: Literal["supported", "partial", "unsupported", "contradicted"]
    quote: str
    rationale: str


# ---------- tasks ----------
QUERY_QUESTIONS = [
    "How does pgvector's HNSW index compare to IVFFlat for approximate nearest neighbour search?",
    "What are the main health effects of long-term exposure to PM2.5 air pollution?",
    "Why did the Bronze Age collapse happen around 1200 BCE?",
    "How effective are heat pumps in cold climates below -15 C?",
    "What evidence links gut microbiota to depression?",
]

VERIFY_CASES: list[tuple[str, str, str]] = [  # (passage, claim, expected verdict)
    (
        "The Eiffel Tower was completed in March 1889 as the entrance arch to the 1889 World's Fair. "
        "It is 330 metres tall and was the tallest man-made structure in the world until the Chrysler "
        "Building in New York was finished in 1930.",
        "The Eiffel Tower was the world's tallest man-made structure until 1930.",
        "supported",
    ),
    (
        "Honey bees communicate the location of food sources through the waggle dance. The angle of the "
        "dance relative to vertical indicates the direction of the source relative to the sun, and the "
        "duration of the waggle run correlates with distance.",
        "Honey bees use the waggle dance to signal the presence of predators near the hive.",
        "unsupported",
    ),
    (
        "In the trial, 412 participants received the drug and 405 received placebo. After 12 weeks, "
        "systolic blood pressure fell by an average of 4.1 mmHg in the drug group and 3.8 mmHg in the "
        "placebo group, a difference that was not statistically significant (p = 0.62).",
        "The drug significantly lowered systolic blood pressure compared with placebo.",
        "contradicted",
    ),
    (
        "Python 3.13 introduced an experimental free-threaded build that can run with the global "
        "interpreter lock disabled. It also added an experimental just-in-time compiler, which is "
        "disabled by default.",
        "Python 3.13 removed the global interpreter lock and enabled a JIT compiler by default.",
        "contradicted",
    ),
    (
        "Lithium iron phosphate (LFP) batteries have lower energy density than nickel manganese cobalt "
        "(NMC) cells, but they tolerate more charge cycles and are less prone to thermal runaway.",
        "LFP batteries are safer and longer-lasting than NMC batteries, and also cheaper.",
        "partial",
    ),
]

SYSTEM = "You are a careful research assistant. Answer only with JSON matching the given schema."


def query_prompt(q: str) -> str:
    return (
        f"Research question: {q}\n\n"
        "Generate exactly 4 distinct web search queries that together would find good sources "
        "for this question. Return JSON with a 'queries' list."
    )


def verify_prompt(passage: str, claim: str) -> str:
    return (
        "Decide whether the PASSAGE supports the CLAIM.\n"
        "verdict: supported | partial | unsupported | contradicted.\n"
        "quote: copy the single most relevant sentence or phrase from the PASSAGE exactly, "
        "character for character. Do not paraphrase.\n"
        "rationale: one short sentence.\n\n"
        f"PASSAGE:\n<<<\n{passage}\n>>>\n\nCLAIM: {claim}"
    )


@dataclass
class CallResult:
    model: str
    task: str
    idx: int
    latency_s: float
    eval_count: int | None
    eval_duration_ns: int | None
    json_valid: bool
    schema_valid: bool
    quote_verbatim: bool | None = None
    verdict: str | None = None
    expected: str | None = None
    raw: str = ""
    error: str | None = None


@dataclass
class ModelSummary:
    model: str
    num_ctx: int
    label: str
    n: int
    json_valid_pct: float
    schema_valid_pct: float
    quote_verbatim_pct: float
    verdict_acc_pct: float
    avg_latency_s: float
    tokens_per_s: float
    size_gb: float
    vram_gb: float
    gpu_pct: float
    calls: list[CallResult] = field(default_factory=list)


def chat(client: httpx.Client, model: str, prompt: str, schema: dict, num_ctx: int) -> tuple[dict, float]:
    body: dict = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    if model.startswith("qwen3"):
        body["think"] = False
    t0 = time.perf_counter()
    r = client.post("/api/chat", json=body)
    dt = time.perf_counter() - t0
    r.raise_for_status()
    return r.json(), dt


def run_model(client: httpx.Client, model: str, num_ctx: int, verify_only: bool, label: str) -> ModelSummary:
    q_schema = SearchQueries.model_json_schema()

    # Warm-up (loads the model; excluded from metrics).
    chat(client, model, query_prompt("warm-up question"), q_schema, num_ctx)
    ps = client.get("/api/ps").json()["models"]
    loaded = next((m for m in ps if m["name"] == model or m["model"] == model), None)
    size = loaded["size"] if loaded else 0
    size_vram = loaded["size_vram"] if loaded else 0

    calls: list[CallResult] = []
    tasks: list[tuple[str, int, str, type[BaseModel], str | None, str | None]] = []
    if not verify_only:
        tasks += [
            ("queries", i, query_prompt(q), SearchQueries, None, None) for i, q in enumerate(QUERY_QUESTIONS)
        ]
    tasks += [
        ("verify", i, verify_prompt(p, c), Verification, p, exp) for i, (p, c, exp) in enumerate(VERIFY_CASES)
    ]
    for task, i, prompt, schema_cls, passage, expected in tasks:
        try:
            resp, dt = chat(client, model, prompt, schema_cls.model_json_schema(), num_ctx)
        except httpx.HTTPError as e:
            calls.append(CallResult(model, task, i, 0.0, None, None, False, False, error=repr(e)))
            continue
        raw = resp["message"]["content"]
        cr = CallResult(
            model,
            task,
            i,
            dt,
            resp.get("eval_count"),
            resp.get("eval_duration"),
            json_valid=False,
            schema_valid=False,
            raw=raw,
            expected=expected,
        )
        try:
            json.loads(raw)
            cr.json_valid = True
            obj = schema_cls.model_validate_json(raw)
            cr.schema_valid = True
            if isinstance(obj, Verification) and passage is not None:
                cr.verdict = obj.verdict
                cr.quote_verbatim = bool(obj.quote.strip()) and obj.quote in passage
        except (json.JSONDecodeError, ValidationError) as e:
            cr.error = f"{type(e).__name__}: {str(e)[:200]}"
        calls.append(cr)
        print(
            f"  {task}#{i}: {dt:5.1f}s json={cr.json_valid} schema={cr.schema_valid} "
            f"verbatim={cr.quote_verbatim} verdict={cr.verdict}/{expected}",
            flush=True,
        )

    # Unload so the next model starts from a clean GPU.
    client.post("/api/generate", json={"model": model, "keep_alive": 0})

    n = len(calls)
    verify_calls = [c for c in calls if c.task == "verify"]
    toks = [c.eval_count / (c.eval_duration_ns / 1e9) for c in calls if c.eval_count and c.eval_duration_ns]
    pct = lambda k, d: 100.0 * k / d if d else 0.0  # noqa: E731
    return ModelSummary(
        model=model,
        num_ctx=num_ctx,
        label=label,
        n=n,
        json_valid_pct=pct(sum(c.json_valid for c in calls), n),
        schema_valid_pct=pct(sum(c.schema_valid for c in calls), n),
        quote_verbatim_pct=pct(sum(bool(c.quote_verbatim) for c in verify_calls), len(verify_calls)),
        verdict_acc_pct=pct(sum(c.verdict == c.expected for c in verify_calls), len(verify_calls)),
        avg_latency_s=statistics.mean(c.latency_s for c in calls if c.latency_s) if calls else 0.0,
        tokens_per_s=statistics.mean(toks) if toks else 0.0,
        size_gb=size / 1e9,
        vram_gb=size_vram / 1e9,
        gpu_pct=pct(size_vram, size),
        calls=calls,
    )


def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["qwen3:8b", "qwen3:4b", "llama3.1:8b"])
    ap.add_argument("--num-ctx", type=int, nargs="+", default=[s.ollama_num_ctx])
    ap.add_argument("--verify-only", action="store_true", help="run only the 5 verification cases")
    ap.add_argument("--label", default="", help="free-text tag for this run, e.g. the server env")
    args = ap.parse_args()

    with httpx.Client(base_url=s.ollama_url, timeout=s.llm_timeout_s) as client:
        installed = {m["name"] for m in client.get("/api/tags").json()["models"]}
        summaries: list[ModelSummary] = []
        for model in args.models:
            if model not in installed:
                print(f"SKIP {model}: not installed")
                continue
            for num_ctx in args.num_ctx:
                print(f"== {model} (num_ctx={num_ctx}) {args.label}", flush=True)
                summaries.append(run_model(client, model, num_ctx, args.verify_only, args.label))

    out_dir = Path("data/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps([asdict(x) for x in summaries], indent=2), encoding="utf-8")

    tasks = "5 verify" if args.verify_only else "5 query + 5 verify"
    print(f"\ntemperature=0, {tasks} prompts per row, label={args.label!r}")
    print(
        "| model | num_ctx | JSON valid | schema valid | quote verbatim | verdict acc | avg latency | tok/s "
        "| loaded size | on GPU |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in summaries:
        print(
            f"| {m.model} | {m.num_ctx} | {m.json_valid_pct:.0f}% | {m.schema_valid_pct:.0f}% | "
            f"{m.quote_verbatim_pct:.0f}% | {m.verdict_acc_pct:.0f}% | "
            f"{m.avg_latency_s:.1f}s | {m.tokens_per_s:.1f} | {m.size_gb:.2f} GB | "
            f"{m.gpu_pct:.0f}% ({m.vram_gb:.2f} GB) |"
        )
    print(f"\nRaw results: {out}")


if __name__ == "__main__":
    main()
