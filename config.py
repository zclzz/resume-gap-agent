"""Typed configuration for the resume-gap-agent.

All tunables live here as a single Pydantic ``Settings`` object. Nothing in the
codebase should hard-code a model name, threshold, or count -- read it from
``Config`` instead. Values can be overridden via environment variables (prefix
``RGA_``) or a ``.env`` file.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Central configuration object. One home for every tunable."""

    model_config = SettingsConfigDict(
        env_prefix="RGA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Component selection (the "mock vs real" switches for offline demos) #
    # ------------------------------------------------------------------ #
    mode: Literal["deterministic", "agentic"] = Field(
        default="deterministic",
        description=(
            "deterministic = fixed graph, LLM only extracts (offline mock ok). "
            "agentic = LLM sourcing agent (MCP tools) + LLM critic decide control flow "
            "(requires a real tool-calling LLM)."
        ),
    )
    use_mock_llm: bool = Field(
        default=True,
        description="If true, use the deterministic MockLLMClient (no API key needed).",
    )
    use_mock_source: bool = Field(
        default=True,
        description="If true, source postings from bundled fixtures instead of a live API.",
    )
    agent_recursion_limit: int = Field(
        default=25, gt=0, description="Cap on ReAct agent steps (LangGraph recursion_limit)."
    )

    # ------------------------------------------------------------------ #
    # LLM settings                                                       #
    # ------------------------------------------------------------------ #
    llm_provider: Literal["mock", "anthropic", "openai"] = Field(
        default="anthropic",
        description="Real backend used when use_mock_llm is False.",
    )

    # API keys accept either the RGA_-prefixed name or the vendor's standard
    # env var (e.g. OPENAI_API_KEY), so an existing .env just works.
    anthropic_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RGA_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        description="Anthropic API key.",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("RGA_OPENAI_API_KEY", "OPENAI_API_KEY"),
        description="OpenAI API key.",
    )
    llm_model: str = Field(
        default="claude-opus-4-8",
        description="Model id for the Anthropic backend.",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Model id for the OpenAI backend.",
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, gt=0)
    llm_parse_retries: int = Field(
        default=1, ge=0, description="Retries on JSON parse/validation failure."
    )

    # ------------------------------------------------------------------ #
    # Sourcing settings                                                  #
    # ------------------------------------------------------------------ #
    target_postings: int = Field(default=20, gt=0, description="Ideal posting count.")
    min_postings: int = Field(
        default=10, gt=0, description="Minimum postings for a confident report."
    )
    entry_level_titles: tuple[str, ...] = Field(
        default=(
            "junior",
            "entry",
            "graduate",
            "associate",
            "trainee",
            "intern",
            "i",
            "l1",
        ),
        description="Title tokens that mark a posting as entry-level.",
    )

    # Adzuna (live source) --------------------------------------------- #
    adzuna_app_id: str | None = Field(default=None)
    adzuna_app_key: str | None = Field(default=None)
    adzuna_country: str = Field(default="sg", description="Adzuna country code.")

    # ------------------------------------------------------------------ #
    # Analysis thresholds                                                #
    # ------------------------------------------------------------------ #
    critical_frequency: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Skills required by >= this share of postings are critical gaps.",
    )
    important_frequency: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Skills required by >= this share are important gaps.",
    )
    coverage_min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Critic loops back if requirement coverage confidence is below this.",
    )
    max_critic_loops: int = Field(
        default=2, ge=0, description="Cap on critic-driven re-sourcing loops."
    )

    # ------------------------------------------------------------------ #
    # Infrastructure                                                     #
    # ------------------------------------------------------------------ #
    checkpoint_db: str = Field(
        default="checkpoints.sqlite",
        description="SQLite file backing the LangGraph checkpointer.",
    )
    log_level: str = Field(default="INFO")


@lru_cache
def get_config() -> Config:
    """Return a cached process-wide Config instance."""
    return Config()
