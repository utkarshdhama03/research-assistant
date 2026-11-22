"""Application settings. All configuration comes from environment variables or `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_CONTACT_URL = "mailto:you@example.com"
USER_AGENT_PRODUCT = "ResearchAssistant/0.1"
ACCESS_KEY_MIN_CHARS = 32
# An ACCESS_KEY containing one of these (case-insensitive) is a placeholder, not a key.
ACCESS_KEY_PLACEHOLDER_WORDS = ("change-me", "changeme", "change_me", "replace", "example", "placeholder")
GENERATE_KEY_HINT = 'python -c "import secrets; print(secrets.token_urlsafe(32))"'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # A failed check must not print the values it was given (ACCESS_KEY, DATABASE_URL's password).
        hide_input_in_errors=True,
    )

    # Where site operators can reach you. Sent in the fetcher's User-Agent and used by SearXNG
    # (docker compose reads CONTACT_URL from .env). The placeholder is rejected at startup.
    contact_url: str = Field(default=PLACEHOLDER_CONTACT_URL, validate_default=True)

    # Database
    database_url: str = "postgresql+psycopg://research:research@127.0.0.1:5434/research"
    test_database_url: str = "postgresql+psycopg://research:research@127.0.0.1:5434/research_test"

    # Ollama
    ollama_url: str = "http://localhost:11434"
    chat_model: str = "qwen3:8b"
    verify_model: str = "qwen3:8b"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    # One context window for every role: changing num_ctx between requests reloads the model.
    ollama_num_ctx: int = Field(default=6144, ge=512)
    # Measured 2026-11-17: longest call 26 s; worst case ~115 s (900 output tokens + cold reload).
    llm_timeout_s: float = Field(default=120.0, gt=0)
    # Ollama duration string; long enough that the model stays loaded through a run's fetch phase.
    llm_keep_alive: str = "30m"
    llm_max_repairs: int = Field(default=2, ge=0)
    llm_retries: int = Field(default=3, ge=0)
    llm_backoff_s: float = Field(default=2.0, ge=0)
    # Prompt-size estimate for the context guard. Calibrated 2026-11-16 against qwen3:8b
    # prompt_eval_count: 3.78-4.58 on our prompts and web chunks; 3.7 sits just below the minimum.
    llm_chars_per_token: float = Field(default=3.7, gt=0)

    # Search
    searxng_url: str = "http://localhost:8888"
    search_timeout_s: float = Field(default=15.0, gt=0)
    search_retries: int = Field(default=2, ge=0)
    search_backoff_s: float = Field(default=0.5, ge=0)
    # ddgs is used only when SearXNG errors or returns nothing, and only if this is true.
    search_fallback_enabled: bool = True
    # Minimum time between two consecutive searches of a run (engines throttle bursts: the quality2
    # eval met DuckDuckGo CAPTCHAs). 0 disables pacing (eval replay sets it to 0).
    search_min_interval_s: float = Field(default=1.5, ge=0)

    # Fetch
    fetch_timeout_s: float = Field(default=20.0, gt=0)
    fetch_host_min_delay_s: float = Field(default=1.0, ge=0)
    robots_max_bytes: int = Field(default=512_000, ge=1)

    # Extract, chunk, embed
    extract_min_chars: int = Field(default=300, ge=0)
    chunk_target_tokens: int = Field(default=400, ge=1)
    chunk_overlap: float = Field(default=0.15, ge=0, lt=0.5)
    chars_per_token: float = Field(default=4.0, gt=0)
    embed_batch_size: int = Field(default=32, ge=1)
    embed_retries: int = Field(default=2, ge=0)

    # Budgets
    # Must stay within the ResearchPlan schema bounds (3-5); a test checks they match.
    max_sub_questions: int = Field(default=5, ge=3, le=5)
    # Search results per query, picks per sub-question (select schema bound), fetches per run.
    max_sources_per_sub_question: int = Field(default=3, ge=1, le=10)
    max_sources_per_run: int = Field(default=15, ge=1)
    max_chunks_per_source: int = Field(default=40, ge=1)
    # Wall-clock budget for a run; research stops starting new work at research_budget_share of it.
    run_deadline_s: float = Field(default=1800.0, gt=0)
    research_budget_share: float = Field(default=0.7, gt=0, le=1)
    # Chunks retrieved (run-wide) per report section.
    write_top_k: int = Field(default=6, ge=1)
    plan_num_predict: int = Field(default=700, ge=16)
    select_num_predict: int = Field(default=150, ge=16)
    write_num_predict: int = Field(default=900, ge=16)
    # Phase 5: backfill, query dedupe, claims, verification, gap loop
    max_backfills_per_sub_question: int = Field(default=2, ge=0)
    query_jaccard_threshold: float = Field(default=0.6, gt=0, le=1)
    analyze_top_k: int = Field(default=6, ge=1)
    max_claims_per_sub_question: int = Field(default=6, ge=1)
    min_supported_claims: int = Field(default=2, ge=0)
    # 0.96 from the Phase 5 live runs (see decision log); recalibrate in Phase 6.
    claim_dedupe_threshold: float = Field(default=0.96, gt=0, le=1)
    # analyze, verify and gap stop starting new work at this share of RUN_DEADLINE_S
    verify_budget_share: float = Field(default=0.9, gt=0, le=1)
    analyze_num_predict: int = Field(default=900, ge=16)
    verify_num_predict: int = Field(default=300, ge=16)
    gap_num_predict: int = Field(default=150, ge=16)
    # A written sentence containing one of these (case-insensitive, whole words) that none of its cited
    # claims contains is replaced by the cited claims themselves. Env: JSON list.
    speculation_phrases: list[str] = Field(
        default_factory=lambda: [
            "may have",
            "might have",
            "could have",
            "likely",
            "possibly",
            "potentially",
            "perhaps",
            "suggests that",
            "contributing to",
            "which may",
            "which could",
        ]
    )
    # Sentence content check in write: every other word of a written sentence must occur in the claims
    # it cites. Negations (not, no, never, without) and quantity words (only, more, most, few) are
    # deliberately NOT stopwords: a writer must not flip or inflate a claim. Env: JSON lists.
    sentence_stopwords: list[str] = Field(
        default_factory=lambda: (
            "a about above after again against all am an and any are as at be because been before being "
            "below between but by can could did do does doing down during each for from further had has "
            "have having he her here hers herself him himself his how i if in into is it its itself just "
            "me my myself now of off on once or other our ours ourselves out over own same she should so "
            "some such than that the their theirs them themselves then there they those through to too "
            "under until up was we were what when where which who whom why will with would you your yours "
            "yourself yourselves may might must shall upon via per s"
        ).split()
    )
    sentence_connectives: list[str] = Field(
        default_factory=lambda: (
            "however additionally also contrast whereas furthermore meanwhile both while this these moreover "
            "overall notably similarly likewise instead specifically particular example namely finally"
        ).split()
    )
    max_sources_per_query: int = Field(default=5, ge=1)
    max_fetch_bytes: int = Field(default=5_000_000, ge=1)
    max_gap_loops: int = Field(default=1, ge=0)
    node_timeout_s: float = Field(default=600.0, gt=0)

    # Memory: offer sources from earlier runs to the select step (copied into the run when chosen).
    memory_enabled: bool = True
    memory_max_age_days: int = Field(default=30, ge=1)
    memory_max_candidates: int = Field(default=3, ge=1)
    # Cosine similarity (1 - distance) of the sub-question to a source's best chunk. Provisional: 2026-11-19
    # live lookups (2 questions) gave 0.73-0.86 for related pages and 0.65-0.71 for unrelated ones.
    memory_min_similarity: float = Field(default=0.75, ge=0, le=1)
    # Sources copied from earlier runs per run (selected candidates and URL matches). They do not count
    # toward MAX_SOURCES_PER_RUN, which bounds fetch attempts.
    max_reused_sources_per_run: int = Field(default=4, ge=0)
    # Chunks fetched from the HNSW index (iterative scan) before grouping them into sources.
    memory_overfetch_chunks: int = Field(default=200, ge=10, le=1000)

    # API (`ra serve`) and worker (`ra worker`)
    # `ra serve` refuses any non-loopback API_HOST unless AUTH_ENABLED (docker compose binds 0.0.0.0 inside
    # the container with auth on, and publishes the port on 127.0.0.1 only).
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    # Host header values accepted besides 127.0.0.1:<API_PORT> and localhost:<API_PORT> (exact, with the
    # port, e.g. ["research.lan:8000"]). Env: JSON list.
    api_allowed_hosts: list[str] = Field(default_factory=list)
    # Single-user login. AUTH_ENABLED is false for local development and true in docker compose.
    auth_enabled: bool = False
    access_key: SecretStr | None = None
    session_ttl_h: float = Field(default=12.0, gt=0)
    # Built UI (frontend/dist) served at / by the API; unset: the API serves no UI (dev uses Vite).
    ui_dist_dir: Path | None = None
    api_question_max_chars: int = Field(default=500, ge=5)
    worker_poll_s: float = Field(default=2.0, gt=0)
    worker_heartbeat_s: float = Field(default=10.0, gt=0)
    # /health reports the worker as down when its last heartbeat is older than this.
    worker_stale_s: float = Field(default=30.0, gt=0)
    sse_poll_s: float = Field(default=0.5, gt=0)
    sse_max_streams: int = Field(default=8, ge=1)

    @field_validator("contact_url")
    @classmethod
    def _real_contact(cls, value: str) -> str:
        value = value.strip()
        if not value or value == PLACEHOLDER_CONTACT_URL:
            raise ValueError(
                f"CONTACT_URL is unset or still the placeholder {PLACEHOLDER_CONTACT_URL!r}. Set it in .env "
                "to a URL or mailto: where site operators can reach you (sent in the User-Agent and used "
                "by SearXNG), then restart SearXNG with `docker compose up -d searxng`."
            )
        return value

    @model_validator(mode="after")
    def _auth_key(self) -> Self:
        if not self.auth_enabled:
            return self
        key = self.access_key.get_secret_value().strip() if self.access_key is not None else ""
        hint = f"Generate one with: {GENERATE_KEY_HINT}"
        if not key:
            raise ValueError(f"AUTH_ENABLED is true but ACCESS_KEY is not set in .env. {hint}")
        if len(key) < ACCESS_KEY_MIN_CHARS:
            raise ValueError(f"ACCESS_KEY must be at least {ACCESS_KEY_MIN_CHARS} characters. {hint}")
        if any(word in key.lower() for word in ACCESS_KEY_PLACEHOLDER_WORDS):
            raise ValueError(f"ACCESS_KEY is still a placeholder. {hint}")
        return self

    @property
    def fetch_user_agent(self) -> str:
        return f"{USER_AGENT_PRODUCT} (+{self.contact_url})"


@lru_cache
def get_settings() -> Settings:
    return Settings()
