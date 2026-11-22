"""LLM-layer errors. Every error raised by a client call carries the call's `record` (LLMCallRecord)."""

from typing import Any


class LLMError(Exception):
    record: Any = None  # LLMCallRecord, attached by the client before raising


class PromptVariableError(LLMError):
    """A prompt template was rendered with a missing or unexpected variable, or is malformed."""


class PromptTooLarge(LLMError):
    """The request would not fit in num_ctx. Raised before sending (estimate) or from Ollama's 400."""

    def __init__(
        self, message: str, *, tokens: int, num_ctx: int, num_predict: int, repairs: int, source: str
    ) -> None:
        super().__init__(message)
        self.tokens = tokens
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.repairs = repairs
        self.source = source  # "estimate" or "ollama"


class PromptTruncated(LLMError):
    """Ollama evaluated far fewer prompt tokens than estimated: the prompt was cut."""

    def __init__(self, message: str, *, estimated_tokens: int, prompt_eval_count: int) -> None:
        super().__init__(message)
        self.estimated_tokens = estimated_tokens
        self.prompt_eval_count = prompt_eval_count


class LLMOutputTruncated(LLMError):
    """Generation stopped at num_predict (done_reason == "length"). Never a partial success."""

    def __init__(self, message: str, *, raw: str, num_predict: int) -> None:
        super().__init__(message)
        self.raw = raw
        self.num_predict = num_predict


class LLMOutputInvalid(LLMError):
    """The output still failed validation after all repairs."""

    def __init__(self, message: str, *, raw: str, errors: str) -> None:
        super().__init__(message)
        self.raw = raw
        self.errors = errors


class LLMResponseError(LLMError):
    """Ollama rejected the request (non-retryable 4xx) or returned an unexpected payload."""


class LLMUnavailable(LLMError):
    """Ollama unreachable, model missing, or still failing (5xx / load failure) after retries."""


class LLMTimeout(LLMUnavailable):
    """The call exceeded llm_timeout_s. Not retried."""


class UnknownAliasError(LLMError, LookupError):
    """The model referred to an evidence alias that is not in this prompt's AliasMap."""
