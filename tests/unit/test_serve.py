"""`ra serve` start-up checks and the shared log format (no server is started)."""

import logging
import re

import pytest

from research_assistant import cli
from research_assistant.api.app import bind_error, is_loopback
from research_assistant.config import Settings
from research_assistant.logs import LOG_DATEFMT, LOG_FORMAT, UtcFormatter

KEY = "q7Vd0p-3xYz_RkT9mWb2LcN8sHf4JgA1uEi6oPl5"


@pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "localhost", "LOCALHOST", "::1", "[::1]"])
def test_loopback_hosts(host: str) -> None:
    assert is_loopback(host)
    assert bind_error(Settings(_env_file=None, api_host=host)) is None


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.20", "research.lan"])
def test_non_loopback_host_needs_auth(host: str) -> None:
    assert not is_loopback(host)
    error = bind_error(Settings(_env_file=None, api_host=host))
    assert error is not None and "AUTH_ENABLED" in error
    assert bind_error(Settings(_env_file=None, api_host=host, auth_enabled=True, access_key=KEY)) is None


def test_ra_serve_refuses_to_start_on_a_public_host_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[object] = []
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None, api_host="0.0.0.0"))
    monkeypatch.setattr("research_assistant.api.app.serve", lambda *a: started.append(a))
    assert cli.main(["serve"]) == 2
    assert started == []  # never reached uvicorn: nothing was bound


def test_ra_serve_starts_on_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[object] = []
    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(cli, "default_session_factory", lambda: None)
    monkeypatch.setattr("research_assistant.api.app.serve", lambda *a: started.append(a))
    assert cli.main(["serve"]) == 0 and len(started) == 1


def test_log_lines_are_utc_iso_level_logger_message() -> None:
    name = "research_assistant.worker"
    record = logging.LogRecord(name, logging.INFO, __file__, 1, "claimed %s", ("r1",), None)
    record.created = 0.0
    line = UtcFormatter(LOG_FORMAT, LOG_DATEFMT).format(record)
    assert line == "1970-01-01T00:00:00Z INFO research_assistant.worker: claimed r1"
    assert re.fullmatch(r"\S+Z [A-Z]+ [\w.]+: .*", line)
