"""Ollama /api/chat client over plain httpx.

Every request: stream=false, think=false, truncate=false, temperature 0, the global num_ctx and
keep_alive from config, num_predict per call.

Context guard, three layers:
1. before each request (every repair too): estimated prompt tokens + num_predict must fit in num_ctx,
   else PromptTooLarge and nothing is sent;
2. truncate=false makes Ollama answer 400 instead of silently dropping the start of the prompt;
   that 400 becomes PromptTooLarge;
3. after each response: prompt_eval_count far below the estimate means the prompt was cut anyway
   (e.g. a future Ollama ignoring truncate) -> PromptTruncated.

on_call receives one LLMCallRecord per finished call, success or failure. On failure the error is
raised after the callback; a callback that raises is logged and attached as a note, never replacing
the original error. On success a raising callback propagates (the audit trail must not fail silently).
"""

import hashlib
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, Self

import httpx
from pydantic import BaseModel, ValidationError

from research_assistant.config import Settings, get_settings
from research_assistant.llm.errors import (
    LLMError,
    LLMOutputInvalid,
    LLMOutputTruncated,
    LLMResponseError,
    LLMTimeout,
    LLMUnavailable,
    PromptTooLarge,
    PromptTruncated,
)
from research_assistant.llm.prompt_loader import RenderedPrompt

log = logging.getLogger(__name__)

# Chat-template tokens per message (role markers etc.), on top of chars / llm_chars_per_token.
# Measured: a system+user pair costs 21 tokens beyond its text with qwen3:8b.
MESSAGE_OVERHEAD_TOKENS = 11
# The pre-call guard pads the estimate; the post-call check flags counts below this share of it.
ESTIMATE_MARGIN = 1.15
TRUNCATION_RATIO = 0.5
REPAIR_INSTRUCTION = (
    "Your previous reply did not match the required JSON schema.\n"
    "Validation errors:\n{errors}\n"
    "Reply again with only the corrected JSON object."
)
WARM_UP_MESSAGE = "Reply with OK."
_CONTEXT_EXCEEDED = re.compile(r"request \((\d+) tokens\) exceeds the available context size")
_MAX_ERROR_CHARS = 2000

Message = dict[str, str]


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    outcome: str  # "ok" or the error class name
    kind: str  # "structured", "text" or "warm_up"
    model: str
    prompt_name: str
    prompt_sha256: str  # of the prompt template file
    prompt_tokens: int | None  # prompt_eval_count of the last response
    eval_tokens: int  # summed over all requests of the call
    load_duration_s: float
    total_duration_s: float
    attempts: int  # HTTP requests sent (transport retries and repairs included)
    repairs: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class LLMResult[V]:
    value: V
    record: LLMCallRecord

    @property
    def model(self) -> str:
        return self.record.model

    @property
    def prompt_name(self) -> str:
        return self.record.prompt_name

    @property
    def prompt_sha256(self) -> str:
        return self.record.prompt_sha256

    @property
    def prompt_tokens(self) -> int | None:
        return self.record.prompt_tokens

    @property
    def eval_tokens(self) -> int:
        return self.record.eval_tokens

    @property
    def load_duration_s(self) -> float:
        return self.record.load_duration_s

    @property
    def total_duration_s(self) -> float:
        return self.record.total_duration_s

    @property
    def repairs(self) -> int:
        return self.record.repairs


OnCall = Callable[[LLMCallRecord], None]


class LLMClient(Protocol):
    def generate_structured[M: BaseModel](
        self, prompt: RenderedPrompt, schema: type[M], *, num_predict: int
    ) -> LLMResult[M]: ...

    def generate_text(self, prompt: RenderedPrompt, *, num_predict: int) -> LLMResult[str]: ...

    def warm_up(self) -> LLMCallRecord: ...


@dataclass
class _Stats:
    model: str
    attempts: int = 0
    repairs: int = 0
    prompt_tokens: int | None = None
    eval_tokens: int = 0
    load_s: float = 0.0
    total_s: float = 0.0

    def add(self, data: dict[str, Any]) -> None:
        self.model = str(data.get("model") or self.model)
        count = data.get("prompt_eval_count")
        self.prompt_tokens = count if isinstance(count, int) else None
        self.eval_tokens += int(data.get("eval_count") or 0)
        self.load_s += int(data.get("load_duration") or 0) / 1e9
        self.total_s += int(data.get("total_duration") or 0) / 1e9

    def record(self, kind: str, name: str, sha: str, outcome: str, error: str | None) -> LLMCallRecord:
        return LLMCallRecord(
            outcome=outcome,
            kind=kind,
            model=self.model,
            prompt_name=name,
            prompt_sha256=sha,
            prompt_tokens=self.prompt_tokens,
            eval_tokens=self.eval_tokens,
            load_duration_s=self.load_s,
            total_duration_s=self.total_s,
            attempts=self.attempts,
            repairs=self.repairs,
            error=error,
        )


def _describe(error: ValidationError) -> str:
    parts = [
        f"{'.'.join(str(p) for p in e['loc']) or '<root>'}: {e['msg']}"
        for e in error.errors(include_url=False)
    ]
    return "\n".join(parts)[:_MAX_ERROR_CHARS]


def _error_text(resp: httpx.Response) -> str:
    try:
        return str(resp.json()["error"])[:_MAX_ERROR_CHARS]
    except (ValueError, KeyError, TypeError):
        return resp.text[:_MAX_ERROR_CHARS]


class OllamaClient:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: httpx.Client | None = None,
        on_call: OnCall | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        s = settings or get_settings()
        self._url = s.ollama_url.rstrip("/") + "/api/chat"
        self._model = s.chat_model
        self._num_ctx = s.ollama_num_ctx
        self._keep_alive = s.llm_keep_alive
        self._timeout_s = s.llm_timeout_s
        self._max_repairs = s.llm_max_repairs
        self._retries = s.llm_retries
        self._backoff_s = s.llm_backoff_s
        self._chars_per_token = s.llm_chars_per_token
        self._owns_client = client is None
        self._client = client or httpx.Client()
        self._on_call = on_call
        self._sleep = sleep

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    # -- public API ----------------------------------------------------------------------------

    def estimate_tokens(self, messages: list[Message]) -> int:
        chars = sum(len(m["content"]) for m in messages)
        return math.ceil(chars / self._chars_per_token) + MESSAGE_OVERHEAD_TOKENS * len(messages)

    def generate_structured[M: BaseModel](
        self, prompt: RenderedPrompt, schema: type[M], *, num_predict: int
    ) -> LLMResult[M]:
        return self._run(
            "structured",
            prompt.name,
            prompt.template_sha256,
            lambda st: self._structured(prompt, schema, num_predict, st),
        )

    def generate_text(self, prompt: RenderedPrompt, *, num_predict: int) -> LLMResult[str]:
        return self._run(
            "text", prompt.name, prompt.template_sha256, lambda st: self._text(prompt, num_predict, st)
        )

    def warm_up(self) -> LLMCallRecord:
        """Load the model with the run's num_ctx and keep_alive, so the first real call doesn't reload it."""
        sha = hashlib.sha256(WARM_UP_MESSAGE.encode()).hexdigest()
        return self._run("warm_up", "warm_up", sha, self._warm_up).record

    # -- call wrapper --------------------------------------------------------------------------

    def _run[V](self, kind: str, name: str, sha: str, fn: Callable[[_Stats], V]) -> LLMResult[V]:
        stats = _Stats(model=self._model)
        try:
            value = fn(stats)
        except Exception as e:
            record = stats.record(kind, name, sha, type(e).__name__, str(e))
            if isinstance(e, LLMError):
                e.record = record
            self._notify_failure(record, e)
            raise
        record = stats.record(kind, name, sha, "ok", None)
        if self._on_call is not None:
            self._on_call(record)
        return LLMResult(value, record)

    def _notify_failure(self, record: LLMCallRecord, error: Exception) -> None:
        if self._on_call is None:
            return
        try:
            self._on_call(record)
        except Exception as callback_error:
            log.exception("on_call failed while reporting %s", record.outcome)
            error.add_note(f"on_call callback also failed: {callback_error!r}")

    # -- call kinds ----------------------------------------------------------------------------

    def _structured[M: BaseModel](
        self, prompt: RenderedPrompt, schema: type[M], num_predict: int, stats: _Stats
    ) -> M:
        fmt = schema.model_json_schema()
        messages = self._messages(prompt)
        for repair in range(self._max_repairs + 1):
            stats.repairs = repair
            raw = self._chat(messages, num_predict, fmt, stats)
            try:
                return schema.model_validate_json(raw)
            except ValidationError as e:
                errors = _describe(e)
            if repair == self._max_repairs:
                raise LLMOutputInvalid(
                    f"output still invalid after {repair} repairs: {errors}", raw=raw, errors=errors
                )
            messages = [
                *messages,
                {"role": "assistant", "content": raw},
                {"role": "user", "content": REPAIR_INSTRUCTION.format(errors=errors)},
            ]
        raise AssertionError("unreachable")

    def _text(self, prompt: RenderedPrompt, num_predict: int, stats: _Stats) -> str:
        raw = self._chat(self._messages(prompt), num_predict, None, stats)
        if not raw.strip():
            raise LLMOutputInvalid("empty output", raw=raw, errors="empty output")
        return raw

    def _warm_up(self, stats: _Stats) -> None:
        body = self._body([{"role": "user", "content": WARM_UP_MESSAGE}], num_predict=1, fmt=None)
        stats.add(self._post(body, stats))

    # -- one request ---------------------------------------------------------------------------

    @staticmethod
    def _messages(prompt: RenderedPrompt) -> list[Message]:
        return [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]

    def _body(self, messages: list[Message], num_predict: int, fmt: dict[str, Any] | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "think": False,
            "truncate": False,
            "keep_alive": self._keep_alive,
            "options": {"num_ctx": self._num_ctx, "temperature": 0, "num_predict": num_predict},
        }
        if fmt is not None:
            body["format"] = fmt
        return body

    def _chat(
        self, messages: list[Message], num_predict: int, fmt: dict[str, Any] | None, stats: _Stats
    ) -> str:
        estimate = self.estimate_tokens(messages)
        padded = math.ceil(estimate * ESTIMATE_MARGIN)
        if padded + num_predict > self._num_ctx:
            raise PromptTooLarge(
                f"estimated {padded} prompt tokens + num_predict {num_predict} > num_ctx {self._num_ctx} "
                f"(after {stats.repairs} repairs); not sent",
                tokens=padded,
                num_ctx=self._num_ctx,
                num_predict=num_predict,
                repairs=stats.repairs,
                source="estimate",
            )
        data = self._post(self._body(messages, num_predict, fmt), stats)
        stats.add(data)

        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMResponseError("Ollama response has no message content")
        if stats.prompt_tokens is None:
            raise LLMResponseError("Ollama response has no prompt_eval_count; cannot check for truncation")
        if stats.prompt_tokens < estimate * TRUNCATION_RATIO:
            raise PromptTruncated(
                f"Ollama evaluated {stats.prompt_tokens} prompt tokens but {estimate} were estimated: "
                "the prompt was cut",
                estimated_tokens=estimate,
                prompt_eval_count=stats.prompt_tokens,
            )
        done_reason = data.get("done_reason")
        if done_reason == "length":
            raise LLMOutputTruncated(
                f"output stopped at num_predict={num_predict}", raw=content, num_predict=num_predict
            )
        if done_reason != "stop":
            raise LLMResponseError(f"unexpected done_reason {done_reason!r}")
        return content

    def _post(self, body: dict[str, Any], stats: _Stats) -> dict[str, Any]:
        last_error = ""
        for attempt in range(self._retries + 1):
            if attempt:
                self._sleep(self._backoff_s * 2 ** (attempt - 1))
            stats.attempts += 1
            try:
                resp = self._client.post(self._url, json=body, timeout=self._timeout_s)
            except httpx.TimeoutException as e:
                raise LLMTimeout(f"no response from Ollama within {self._timeout_s}s") from e
            except httpx.TransportError as e:
                last_error = f"{type(e).__name__}: {e}"
                continue
            if resp.status_code >= 500:  # includes model-load failures (e.g. CUDA init crash)
                last_error = f"HTTP {resp.status_code}: {_error_text(resp)}"
                continue
            if resp.status_code != 200:
                raise self._client_error(resp, body, stats)
            try:
                data = resp.json()
            except ValueError as e:
                raise LLMResponseError("Ollama returned invalid JSON") from e
            if not isinstance(data, dict):
                raise LLMResponseError("Ollama returned a non-object JSON payload")
            return data
        raise LLMUnavailable(f"Ollama failed after {self._retries + 1} attempts: {last_error}")

    def _client_error(self, resp: httpx.Response, body: dict[str, Any], stats: _Stats) -> LLMError:
        text = _error_text(resp)
        if m := _CONTEXT_EXCEEDED.search(text):
            num_predict = int(body["options"]["num_predict"])
            return PromptTooLarge(
                f"Ollama: {text} (after {stats.repairs} repairs)",
                tokens=int(m.group(1)),
                num_ctx=self._num_ctx,
                num_predict=num_predict,
                repairs=stats.repairs,
                source="ollama",
            )
        if resp.status_code == 404:
            return LLMUnavailable(f"Ollama: {text}")
        return LLMResponseError(f"Ollama HTTP {resp.status_code}: {text}")
