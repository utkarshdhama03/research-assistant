"""One JSON error shape for every failure: {"error": {"code", "message", "details"?}}.

Validation details carry only the location and message, never the submitted input. Unexpected errors
return a generic 500; the traceback goes to the server log only.
"""

import logging
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from research_assistant.api.schemas import ErrorBody, ErrorDetail, ErrorInfo

log = logging.getLogger(__name__)

_CODES = {
    400: "bad_request",
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    503: "unavailable",
}


class ApiError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str | None = None,
        details: list[ErrorDetail] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code or _CODES.get(status, "error")
        self.message = message
        self.details = details
        self.headers = headers


def error_response(
    status: int, message: str, *, code: str | None = None, details: list[ErrorDetail] | None = None
) -> JSONResponse:
    body = ErrorBody(
        error=ErrorInfo(code=code or _CODES.get(status, "error"), message=message, details=details)
    )
    return JSONResponse(body.model_dump(mode="json", exclude_none=True), status_code=status)


def validation_details(errors: Sequence[Any]) -> list[ErrorDetail]:
    """loc + msg of pydantic/FastAPI validation errors; the offending input is left out."""
    return [
        ErrorDetail(
            loc=[p if isinstance(p, int) else str(p) for p in e.get("loc", ())], msg=str(e.get("msg", ""))
        )
        for e in errors
    ]


def install_error_handlers(app: FastAPI) -> None:
    async def api_error(_request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, ApiError)
        response = error_response(exc.status, exc.message, code=exc.code, details=exc.details)
        response.headers.update(exc.headers or {})
        return response

    async def http_error(_request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, StarletteHTTPException)
        message = exc.detail if isinstance(exc.detail, str) else "request failed"
        return error_response(exc.status_code, message)

    async def request_validation(_request: Request, exc: Exception) -> JSONResponse:
        assert isinstance(exc, RequestValidationError)
        return error_response(422, "invalid request", details=validation_details(exc.errors()))

    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return error_response(500, "internal server error")

    app.add_exception_handler(ApiError, api_error)
    app.add_exception_handler(StarletteHTTPException, http_error)
    app.add_exception_handler(RequestValidationError, request_validation)
    app.add_exception_handler(Exception, unexpected)
