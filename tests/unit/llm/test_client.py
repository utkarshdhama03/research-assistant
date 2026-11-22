"""OllamaClient tests: scripted /api/chat responses over httpx.MockTransport, no live model."""

import json
import math
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from research_assistant.config import Settings
from research_assistant.llm.client import (
    ESTIMATE_MARGIN,
    REPAIR_INSTRUCTION,
    LLMCallRecord,
    OllamaClient,
)
from research_assistant.llm.errors import (
    LLMOutputInvalid,
    LLMOutputTruncated,
    LLMResponseError,
    LLMTimeout,
    LLMUnavailable,
    PromptTooLarge,
    PromptTruncated,
)
from research_assistant.llm.prompt_loader import RenderedPrompt, load_prompt
from research_assistant.llm.schemas import ResearchPlan

VALID_PLAN = {
    "sub_questions": [
        {"question": "What is an okapi?", "search_queries": ["okapi animal", "okapia johnstoni"]},
        {"question": "Where do okapis live?", "search_queries": ["okapi habitat"]},
        {
            "question": "Why are okapi legs striped?",
            "search_queries": ["okapi stripes function", "okapi camouflage"],
        },
    ]
}
TWO_SUB_QUESTIONS = {"sub_questions": VALID_PLAN["sub_questions"][:2]}

Reply = httpx.Response | Callable[[httpx.Request], httpx.Response]


def chat_response(
    content: str,
    *,
    done_reason: str = "stop",
    prompt_eval_count: int | None = 1000,
    eval_count: int = 50,
    load_s: float = 0.5,
    total_s: float = 2.0,
) -> httpx.Response:
    body: dict[str, Any] = {
        "model": "qwen3:8b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "eval_count": eval_count,
        "load_duration": int(load_s * 1e9),
        "total_duration": int(total_s * 1e9),
    }
    if prompt_eval_count is not None:
        body["prompt_eval_count"] = prompt_eval_count
    return httpx.Response(200, json=body)


def error_response(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": message})


class FakeOllama:
    def __init__(self, *replies: Reply) -> None:
        self.replies = list(replies)
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        self.bodies.append(json.loads(request.content))
        if not self.replies:
            raise AssertionError("FakeOllama: no scripted reply left")
        reply = self.replies.pop(0)
        return reply(request) if callable(reply) else reply


def raises(exc: type[httpx.TransportError]) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc("scripted", request=request)

    return handler


def make_client(
    ollama: FakeOllama,
    *,
    sleeps: list[float] | None = None,
    records: list[LLMCallRecord] | None = None,
    on_call: Callable[[LLMCallRecord], None] | None = None,
    **overrides: Any,
) -> OllamaClient:
    values: dict[str, Any] = {
        "chat_model": "qwen3:8b",
        "ollama_num_ctx": 6144,
        "llm_keep_alive": "30m",
        "llm_max_repairs": 2,
        "llm_retries": 3,
        "llm_backoff_s": 2.0,
        "llm_chars_per_token": 3.0,
    }
    values.update(overrides)
    if on_call is None and records is not None:
        on_call = records.append
    return OllamaClient(
        Settings(_env_file=None, **values),
        client=httpx.Client(transport=httpx.MockTransport(ollama)),
        on_call=on_call,
        sleep=(sleeps.append if sleeps is not None else lambda _s: None),
    )


def plan_prompt(question: str = "Why do okapis have striped legs?") -> RenderedPrompt:
    return load_prompt("plan").render(question=question)


# -- structured output -----------------------------------------------------------------------


def test_valid_output() -> None:
    ollama = FakeOllama(chat_response(json.dumps(VALID_PLAN), prompt_eval_count=850, eval_count=88))
    prompt = plan_prompt()
    result = make_client(ollama).generate_structured(prompt, ResearchPlan, num_predict=700)

    assert isinstance(result.value, ResearchPlan)
    assert len(result.value.sub_questions) == 3
    assert (result.model, result.prompt_name, result.prompt_sha256) == (
        "qwen3:8b",
        "plan",
        prompt.template_sha256,
    )
    assert (result.prompt_tokens, result.eval_tokens, result.repairs) == (850, 88, 0)
    assert (result.load_duration_s, result.total_duration_s) == (0.5, 2.0)
    assert result.record.attempts == 1 and result.record.outcome == "ok"


def test_every_request_sends_fixed_options() -> None:
    ollama = FakeOllama(
        chat_response("not json"),
        chat_response(json.dumps(TWO_SUB_QUESTIONS)),
        chat_response(json.dumps(VALID_PLAN)),
    )
    make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert len(ollama.bodies) == 3
    for body in ollama.bodies:
        assert body["model"] == "qwen3:8b"
        assert body["think"] is False
        assert body["truncate"] is False
        assert body["stream"] is False
        assert body["keep_alive"] == "30m"
        assert body["options"] == {"num_ctx": 6144, "temperature": 0, "num_predict": 700}
        assert body["format"] == ResearchPlan.model_json_schema()


def test_invalid_then_repaired() -> None:
    first = json.dumps(TWO_SUB_QUESTIONS)
    ollama = FakeOllama(
        chat_response(first, eval_count=40, total_s=1.0),
        chat_response(json.dumps(VALID_PLAN), prompt_eval_count=600, eval_count=60, total_s=1.5),
    )
    prompt = plan_prompt()
    result = make_client(ollama).generate_structured(prompt, ResearchPlan, num_predict=700)

    assert result.repairs == 1 and result.record.attempts == 2
    assert result.eval_tokens == 100 and result.total_duration_s == 2.5  # summed over attempts
    assert result.prompt_tokens == 600  # from the last response
    repair_messages = ollama.bodies[1]["messages"]
    assert [m["role"] for m in repair_messages] == ["system", "user", "assistant", "user"]
    assert repair_messages[0]["content"] == prompt.system and repair_messages[1]["content"] == prompt.user
    assert repair_messages[2]["content"] == first
    assert repair_messages[3]["content"].startswith(REPAIR_INSTRUCTION.split("{")[0])
    assert "sub_questions" in repair_messages[3]["content"] and "at least 3" in repair_messages[3]["content"]


def test_invalid_json_then_repaired() -> None:
    ollama = FakeOllama(chat_response('{"sub_questions": ['), chat_response(json.dumps(VALID_PLAN)))
    result = make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert result.repairs == 1
    assert "Invalid JSON" in ollama.bodies[1]["messages"][3]["content"]


def test_invalid_beyond_max_repairs() -> None:
    raws = ["nope", json.dumps(TWO_SUB_QUESTIONS), json.dumps({"sub_questions": [], "extra": 1})]
    ollama = FakeOllama(*(chat_response(r) for r in raws))
    records: list[LLMCallRecord] = []
    with pytest.raises(LLMOutputInvalid) as info:
        make_client(ollama, records=records).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert info.value.raw == raws[-1]
    assert "sub_questions" in info.value.errors
    assert len(ollama.bodies) == 3  # 1 + llm_max_repairs
    assert (records[0].outcome, records[0].repairs, records[0].attempts) == ("LLMOutputInvalid", 2, 3)


def test_zero_repairs_configured() -> None:
    ollama = FakeOllama(chat_response("nope"))
    with pytest.raises(LLMOutputInvalid):
        make_client(ollama, llm_max_repairs=0).generate_structured(
            plan_prompt(), ResearchPlan, num_predict=700
        )
    assert len(ollama.bodies) == 1


def test_truncated_output_is_an_error_not_repaired() -> None:
    partial = '{"sub_questions": [{"question": "What is'
    ollama = FakeOllama(chat_response(partial, done_reason="length"))
    with pytest.raises(LLMOutputTruncated) as info:
        make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=50)
    assert (info.value.raw, info.value.num_predict) == (partial, 50)
    assert len(ollama.bodies) == 1


def test_truncated_output_even_if_it_parses() -> None:
    ollama = FakeOllama(chat_response(json.dumps(VALID_PLAN), done_reason="length"))
    with pytest.raises(LLMOutputTruncated):
        make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)


@pytest.mark.parametrize("done_reason", ["load", "unload", None])
def test_unexpected_done_reason(done_reason: str | None) -> None:
    response = chat_response(json.dumps(VALID_PLAN))
    body = response.json()
    body["done_reason"] = done_reason
    with pytest.raises(LLMResponseError, match="done_reason"):
        make_client(FakeOllama(httpx.Response(200, json=body))).generate_structured(
            plan_prompt(), ResearchPlan, num_predict=700
        )


# -- context guard ---------------------------------------------------------------------------


def test_prompt_too_large_is_not_sent() -> None:
    ollama = FakeOllama()
    records: list[LLMCallRecord] = []
    client = make_client(ollama, records=records, ollama_num_ctx=600)
    with pytest.raises(PromptTooLarge) as info:
        client.generate_structured(plan_prompt("okapi " * 300), ResearchPlan, num_predict=300)
    assert ollama.bodies == []
    assert (info.value.source, info.value.repairs, info.value.num_ctx) == ("estimate", 0, 600)
    assert records[0].outcome == "PromptTooLarge" and records[0].attempts == 0


def test_num_predict_counts_toward_the_guard() -> None:
    client = make_client(FakeOllama(), ollama_num_ctx=6144)
    with pytest.raises(PromptTooLarge):
        client.generate_structured(plan_prompt(), ResearchPlan, num_predict=6144)


def test_repair_that_would_exceed_num_ctx_is_not_sent() -> None:
    prompt = plan_prompt()
    probe = make_client(FakeOllama())
    first = probe.estimate_tokens(
        [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]
    )
    num_predict = 300
    num_ctx = math.ceil(first * ESTIMATE_MARGIN) + num_predict + 50  # first attempt fits, a long repair won't
    long_invalid = json.dumps({"sub_questions": [], "notes": "x" * 2000})
    ollama = FakeOllama(chat_response(long_invalid, prompt_eval_count=first))
    records: list[LLMCallRecord] = []
    client = make_client(ollama, records=records, ollama_num_ctx=num_ctx)

    with pytest.raises(PromptTooLarge) as info:
        client.generate_structured(prompt, ResearchPlan, num_predict=num_predict)
    assert len(ollama.bodies) == 1  # the repair was never sent
    assert (info.value.source, info.value.repairs) == ("estimate", 1)
    assert "after 1 repairs" in str(info.value)
    assert (records[0].outcome, records[0].repairs, records[0].attempts) == ("PromptTooLarge", 1, 1)


def test_ollama_context_exceeded_400_is_prompt_too_large() -> None:
    msg = "request (24484 tokens) exceeds the available context size (6144 tokens), try increasing it"
    ollama = FakeOllama(error_response(400, msg))
    with pytest.raises(PromptTooLarge) as info:
        make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert (info.value.source, info.value.tokens) == ("ollama", 24484)
    assert len(ollama.bodies) == 1


def test_ollama_side_truncation_is_detected() -> None:
    ollama = FakeOllama(chat_response(json.dumps(VALID_PLAN), prompt_eval_count=12))
    with pytest.raises(PromptTruncated) as info:
        make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert info.value.prompt_eval_count == 12 and info.value.estimated_tokens > 24


def test_missing_prompt_eval_count_is_an_error() -> None:
    ollama = FakeOllama(chat_response(json.dumps(VALID_PLAN), prompt_eval_count=None))
    with pytest.raises(LLMResponseError, match="prompt_eval_count"):
        make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)


# -- transport -------------------------------------------------------------------------------


def test_retry_on_5xx_with_backoff() -> None:
    sleeps: list[float] = []
    ollama = FakeOllama(
        error_response(500, "boom"), error_response(503, "busy"), chat_response(json.dumps(VALID_PLAN))
    )
    result = make_client(ollama, sleeps=sleeps).generate_structured(
        plan_prompt(), ResearchPlan, num_predict=700
    )
    assert sleeps == [2.0, 4.0]
    assert result.record.attempts == 3 and result.repairs == 0


def test_retry_on_model_load_failure() -> None:
    crash = "llama runner process has terminated: CUDA error: shared object initialization failed"
    ollama = FakeOllama(error_response(500, crash), chat_response(json.dumps(VALID_PLAN)))
    assert make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700).value


def test_retry_on_connection_error() -> None:
    ollama = FakeOllama(raises(httpx.ConnectError), chat_response(json.dumps(VALID_PLAN)))
    assert make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700).value


def test_retries_exhausted_is_unavailable() -> None:
    sleeps: list[float] = []
    crash = "llama runner process has terminated"
    ollama = FakeOllama(*(error_response(500, crash) for _ in range(4)))
    with pytest.raises(LLMUnavailable, match="after 4 attempts.*llama runner"):
        make_client(ollama, sleeps=sleeps).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert sleeps == [2.0, 4.0, 8.0]


def test_timeout_is_not_retried() -> None:
    ollama = FakeOllama(raises(httpx.ReadTimeout))
    with pytest.raises(LLMTimeout):
        make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert len(ollama.bodies) == 1


def test_missing_model_is_unavailable_without_retry() -> None:
    ollama = FakeOllama(error_response(404, "model 'qwen3:8b' not found"))
    with pytest.raises(LLMUnavailable, match="not found"):
        make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert len(ollama.bodies) == 1


def test_other_4xx_is_response_error() -> None:
    ollama = FakeOllama(error_response(400, "invalid format"))
    with pytest.raises(LLMResponseError, match="invalid format"):
        make_client(ollama).generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert len(ollama.bodies) == 1


# -- text and warm-up ------------------------------------------------------------------------


def test_generate_text() -> None:
    ollama = FakeOllama(chat_response("# Report\n\nText."))
    result = make_client(ollama).generate_text(plan_prompt(), num_predict=400)
    assert result.value == "# Report\n\nText."
    assert "format" not in ollama.bodies[0]
    assert ollama.bodies[0]["think"] is False and ollama.bodies[0]["options"]["num_ctx"] == 6144


def test_generate_text_empty_is_invalid() -> None:
    with pytest.raises(LLMOutputInvalid):
        make_client(FakeOllama(chat_response("  \n"))).generate_text(plan_prompt(), num_predict=400)


def test_warm_up() -> None:
    ollama = FakeOllama(chat_response("OK", done_reason="length", prompt_eval_count=14, load_s=7.5))
    records: list[LLMCallRecord] = []
    record = make_client(ollama, records=records).warm_up()
    body = ollama.bodies[0]
    assert body["options"] == {"num_ctx": 6144, "temperature": 0, "num_predict": 1}
    assert body["keep_alive"] == "30m" and body["think"] is False
    assert (record.kind, record.outcome, record.load_duration_s) == ("warm_up", "ok", 7.5)
    assert records == [record]


# -- on_call ---------------------------------------------------------------------------------


def test_on_call_once_per_finished_call_on_success() -> None:
    ollama = FakeOllama(
        error_response(500, "boom"), chat_response("nope"), chat_response(json.dumps(VALID_PLAN))
    )
    records: list[LLMCallRecord] = []
    result = make_client(ollama, records=records).generate_structured(
        plan_prompt(), ResearchPlan, num_predict=700
    )
    assert records == [result.record]  # one record for 3 requests (1 retry, 1 repair)
    record = records[0]
    assert (record.outcome, record.kind, record.attempts, record.repairs, record.error) == (
        "ok",
        "structured",
        3,
        1,
        None,
    )
    assert record.prompt_name == "plan" and len(record.prompt_sha256) == 64


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (chat_response("x", done_reason="length"), LLMOutputTruncated),
        (error_response(404, "model 'qwen3:8b' not found"), LLMUnavailable),
        (chat_response(json.dumps(VALID_PLAN), prompt_eval_count=1), PromptTruncated),
    ],
)
def test_on_call_fires_on_failure_before_raising(reply: httpx.Response, error: type[Exception]) -> None:
    records: list[LLMCallRecord] = []
    with pytest.raises(error) as info:
        make_client(FakeOllama(reply), records=records).generate_structured(
            plan_prompt(), ResearchPlan, num_predict=700
        )
    assert len(records) == 1
    assert records[0].outcome == error.__name__
    assert records[0].error == str(info.value)
    assert info.value.record is records[0]  # type: ignore[attr-defined]


def test_raising_callback_does_not_mask_the_original_error() -> None:
    def broken(_record: LLMCallRecord) -> None:
        raise RuntimeError("audit log down")

    client = make_client(FakeOllama(chat_response("x", done_reason="length")), on_call=broken)
    with pytest.raises(LLMOutputTruncated) as info:
        client.generate_structured(plan_prompt(), ResearchPlan, num_predict=700)
    assert any("audit log down" in note for note in info.value.__notes__)


def test_raising_callback_on_success_propagates() -> None:
    def broken(_record: LLMCallRecord) -> None:
        raise RuntimeError("audit log down")

    client = make_client(FakeOllama(chat_response(json.dumps(VALID_PLAN))), on_call=broken)
    with pytest.raises(RuntimeError, match="audit log down"):
        client.generate_structured(plan_prompt(), ResearchPlan, num_predict=700)


def test_on_call_for_guard_failure_reports_no_request() -> None:
    records: list[LLMCallRecord] = []
    client = make_client(FakeOllama(), records=records, ollama_num_ctx=600)
    with pytest.raises(PromptTooLarge):
        client.generate_text(plan_prompt("okapi " * 300), num_predict=300)
    assert (records[0].kind, records[0].outcome, records[0].attempts, records[0].prompt_tokens) == (
        "text",
        "PromptTooLarge",
        0,
        None,
    )
