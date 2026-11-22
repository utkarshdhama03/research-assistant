"""Manual smoke test of the LLM layer against the local Ollama. Not part of pytest.

1. warm_up(), 2. plan for 3 real questions (outputs, tokens, latencies, repairs),
3. calibration: real chars-per-token from Ollama's prompt_eval_count on our prompts.

    uv run python scripts/smoke_llm.py
"""

import statistics
import time
from pathlib import Path
from typing import Any

import httpx

from research_assistant.config import get_settings
from research_assistant.llm.client import LLMCallRecord, OllamaClient
from research_assistant.llm.errors import LLMError
from research_assistant.llm.prompt_loader import load_prompt, untrusted
from research_assistant.llm.schemas import ResearchPlan
from research_assistant.tools.chunk import chunk_text
from research_assistant.tools.extract import extract

QUESTIONS = [
    "Why do okapis have striped legs?",
    "How does the Robots Exclusion Protocol (RFC 9309) handle errors when fetching robots.txt?",
    "What are the trade-offs between HNSW and IVFFlat indexes in pgvector?",
]
PLAN_NUM_PREDICT = 700
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
WEB_FIXTURES = {
    "simplewiki_okapi.html": "text/html",
    "python_docs_secrets.html": "text/html",
    "rfc2606.txt": "text/plain; charset=utf-8",
    "rfc9309.pdf": "application/pdf",
}


def show(record: LLMCallRecord, wall: float) -> str:
    return (
        f"outcome={record.outcome} prompt_tokens={record.prompt_tokens} eval_tokens={record.eval_tokens} "
        f"attempts={record.attempts} repairs={record.repairs} load={record.load_duration_s:.2f}s "
        f"ollama_total={record.total_duration_s:.2f}s wall={wall:.2f}s"
    )


def count_prompt_tokens(http: httpx.Client, messages: list[dict[str, str]]) -> int:
    s = get_settings()
    body: dict[str, Any] = {
        "model": s.chat_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "truncate": False,
        "keep_alive": s.llm_keep_alive,
        "options": {"num_ctx": s.ollama_num_ctx, "temperature": 0, "num_predict": 1},
    }
    resp = http.post(s.ollama_url.rstrip("/") + "/api/chat", json=body, timeout=s.llm_timeout_s)
    resp.raise_for_status()
    return int(resp.json()["prompt_eval_count"])


def calibrate() -> None:
    s = get_settings()
    system = load_prompt("plan").render(question="x").system
    with httpx.Client() as http:
        overhead = count_prompt_tokens(
            http, [{"role": "system", "content": "a"}, {"role": "user", "content": "b"}]
        )
        overhead -= 2  # the two one-letter contents
        print(f"\n[calibrate] chat-template overhead for system+user: {overhead} tokens")

        samples: list[tuple[str, int, int]] = []  # (label, chars, tokens without overhead)
        for q in QUESTIONS:
            p = load_prompt("plan").render(question=q)
            n = count_prompt_tokens(
                http, [{"role": "system", "content": p.system}, {"role": "user", "content": p.user}]
            )
            samples.append((f"plan: {q[:40]}", len(p.system) + len(p.user), n - overhead))
        for name, content_type in WEB_FIXTURES.items():
            text = extract((FIXTURES / name).read_bytes(), content_type, min_chars=0).text
            chunks = chunk_text(
                text, target_tokens=s.chunk_target_tokens, overlap=0.0, chars_per_token=s.chars_per_token
            )
            user = "\n\n".join(untrusted(c.text, f"C{i}") for i, c in enumerate(chunks[:3], start=1))
            n = count_prompt_tokens(
                http, [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
            samples.append((f"3 web chunks: {name}", len(system) + len(user), n - overhead))

    ratios = []
    for label, chars, tokens in samples:
        ratio = chars / tokens
        ratios.append(ratio)
        print(f"   {label:52} chars={chars:6d} tokens={tokens:5d} chars/token={ratio:.2f}")
    print(
        f"[calibrate] chars/token min={min(ratios):.2f} median={statistics.median(ratios):.2f} "
        f"max={max(ratios):.2f}; configured llm_chars_per_token={s.llm_chars_per_token}"
    )


def main() -> int:
    s = get_settings()
    print(
        f"model={s.chat_model} num_ctx={s.ollama_num_ctx} keep_alive={s.llm_keep_alive} "
        f"max_repairs={s.llm_max_repairs}"
    )
    with OllamaClient(s) as client:
        t = time.perf_counter()
        record = client.warm_up()
        print(f"[warm_up] {show(record, time.perf_counter() - t)}")

        for q in QUESTIONS:
            print(f"\n== {q}")
            prompt = load_prompt("plan").render(question=q)
            t = time.perf_counter()
            try:
                result = client.generate_structured(prompt, ResearchPlan, num_predict=PLAN_NUM_PREDICT)
            except LLMError as e:
                print(f"[plan] FAILED {type(e).__name__}: {e}")
                if e.record is not None:
                    print(f"[plan] {show(e.record, time.perf_counter() - t)}")
                continue
            print(f"[plan] {show(result.record, time.perf_counter() - t)}")
            for i, sq in enumerate(result.value.sub_questions, start=1):
                print(f"   {i}. {sq.question}")
                for query in sq.search_queries:
                    print(f"        - {query}")
    calibrate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
