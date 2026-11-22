import pytest
from pydantic import ValidationError

from research_assistant.config import PLACEHOLDER_CONTACT_URL, Settings, get_settings


def test_config_imports_and_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.embed_dim == 768
    assert s.database_url.startswith("postgresql+psycopg://")
    assert s.max_sub_questions >= 1


def test_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CHAT_MODEL", "some-model")
    assert Settings(_env_file=None).chat_model == "some-model"


def test_get_settings_returns_settings() -> None:
    assert isinstance(get_settings(), Settings)


@pytest.mark.parametrize("value", [PLACEHOLDER_CONTACT_URL, "", "   "])
def test_placeholder_contact_is_rejected(value: str) -> None:
    with pytest.raises(ValidationError, match="CONTACT_URL is unset or still the placeholder"):
        Settings(_env_file=None, contact_url=value)


def test_missing_contact_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONTACT_URL", raising=False)
    with pytest.raises(ValidationError, match="CONTACT_URL"):
        Settings(_env_file=None)


def test_user_agent_includes_contact() -> None:
    s = Settings(_env_file=None, contact_url=" mailto:ops@example.org ")
    assert s.contact_url == "mailto:ops@example.org"
    assert s.fetch_user_agent == "ResearchAssistant/0.1 (+mailto:ops@example.org)"


GOOD_KEY = "q7Vd0p-3xYz_RkT9mWb2LcN8sHf4JgA1uEi6oPl5"


@pytest.mark.parametrize(
    ("key", "message"),
    [
        (None, "ACCESS_KEY is not set"),
        ("   ", "ACCESS_KEY is not set"),
        ("short-key-1234", "at least 32 characters"),
        ("change-me-change-me-change-me-change-me", "placeholder"),
        ("REPLACE_WITH_A_LONG_RANDOM_KEY_OF_32_CHARS", "placeholder"),
    ],
)
def test_auth_needs_a_real_access_key(key: str | None, message: str) -> None:
    with pytest.raises(ValidationError, match=message) as info:
        Settings(_env_file=None, auth_enabled=True, access_key=key)
    assert "secrets.token_urlsafe" in str(info.value)  # tells how to make one
    if key and key.strip():
        assert key.strip() not in str(info.value)  # the rejected key is never echoed
    assert "postgresql" not in str(info.value)  # nor any other setting (DATABASE_URL has a password)


def test_auth_with_a_good_key_and_auth_off_without_one() -> None:
    s = Settings(_env_file=None, auth_enabled=True, access_key=GOOD_KEY)
    assert s.access_key is not None and s.access_key.get_secret_value() == GOOD_KEY
    assert GOOD_KEY not in repr(s)  # SecretStr keeps it out of reprs and logs
    assert Settings(_env_file=None).auth_enabled is False  # local development default


def test_auth_settings_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("ACCESS_KEY", GOOD_KEY)
    monkeypatch.setenv("API_ALLOWED_HOSTS", '["research.lan:8000"]')
    s = Settings(_env_file=None)
    assert s.auth_enabled and s.api_allowed_hosts == ["research.lan:8000"] and s.session_ttl_h == 12
