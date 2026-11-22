"""FakeLLM: a scripted LLMClient for tests of code that calls the LLM (graph nodes, later phases)."""

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from research_assistant.llm.client import LLMCallRecord, LLMResult, OnCall
from research_assistant.llm.errors import LLMError, LLMOutputInvalid
from research_assistant.llm.prompt_loader import RenderedPrompt

Scripted = str | BaseModel | Exception


@dataclass(frozen=True, slots=True)
class FakeCall:
    kind: str
    prompt: RenderedPrompt
    schema: type[BaseModel] | None
    num_predict: int


class FakeLLM:
    """Each generate_* call consumes the next scripted reply, in order.

    - str: raw model output. For generate_structured it is validated against the schema like real
      output (invalid -> LLMOutputInvalid); for generate_text it is returned as is.
    - BaseModel: returned for generate_structured after re-validation against the requested schema.
    - Exception: raised.
    Running out of replies is an AssertionError. on_call gets a record per call, like OllamaClient.
    With `responder`, replies come from responder(call) instead of the queue (e.g. by prompt name).
    """

    def __init__(
        self,
        replies: Iterable[Scripted] = (),
        *,
        on_call: OnCall | None = None,
        responder: Callable[[FakeCall], Scripted] | None = None,
    ) -> None:
        self.replies: deque[Scripted] = deque(replies)
        self.responder = responder
        self.calls: list[FakeCall] = []
        self.warm_ups = 0
        self._on_call = on_call

    def generate_structured[M: BaseModel](
        self, prompt: RenderedPrompt, schema: type[M], *, num_predict: int
    ) -> LLMResult[M]:
        reply = self._next(FakeCall("structured", prompt, schema, num_predict))
        try:
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, BaseModel):
                if not isinstance(reply, schema):
                    raise TypeError(f"scripted {type(reply).__name__}, call asked for {schema.__name__}")
                value = schema.model_validate(reply.model_dump())
            else:
                try:
                    value = schema.model_validate_json(reply)
                except ValidationError as e:
                    raise LLMOutputInvalid(f"scripted output invalid: {e}", raw=reply, errors=str(e)) from e
        except Exception as e:
            self._report("structured", prompt, type(e).__name__, str(e), e)
            raise
        return LLMResult(value, self._report("structured", prompt, "ok", None, None))

    def generate_text(self, prompt: RenderedPrompt, *, num_predict: int) -> LLMResult[str]:
        reply = self._next(FakeCall("text", prompt, None, num_predict))
        if isinstance(reply, Exception):
            self._report("text", prompt, type(reply).__name__, str(reply), reply)
            raise reply
        if not isinstance(reply, str):
            raise TypeError("generate_text needs a scripted str")
        return LLMResult(reply, self._report("text", prompt, "ok", None, None))

    def warm_up(self) -> LLMCallRecord:
        self.warm_ups += 1
        record = _record("warm_up", "warm_up", "", "ok", None)
        if self._on_call is not None:  # like OllamaClient.warm_up
            self._on_call(record)
        return record

    def _next(self, call: FakeCall) -> Scripted:
        self.calls.append(call)
        if self.responder is not None:
            return self.responder(call)
        if not self.replies:
            raise AssertionError(
                f"FakeLLM has no scripted reply left for call {len(self.calls)} ({call.prompt.name})"
            )
        return self.replies.popleft()

    def _report(
        self, kind: str, prompt: RenderedPrompt, outcome: str, error: str | None, exc: Exception | None
    ) -> LLMCallRecord:
        record = _record(kind, prompt.name, prompt.template_sha256, outcome, error)
        if isinstance(exc, LLMError):
            exc.record = record
        if self._on_call is not None:
            self._on_call(record)
        return record


def _record(kind: str, name: str, sha: str, outcome: str, error: str | None) -> LLMCallRecord:
    return LLMCallRecord(
        outcome=outcome,
        kind=kind,
        model="fake",
        prompt_name=name,
        prompt_sha256=sha,
        prompt_tokens=0,
        eval_tokens=0,
        load_duration_s=0.0,
        total_duration_s=0.0,
        attempts=1,
        repairs=0,
        error=error,
    )
