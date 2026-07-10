"""Component factories: turn a ``Config`` into concrete backends.

This is the single place the mock/real switch is resolved, so the graph never
hard-codes a backend.
"""

from __future__ import annotations

from config import Config
from llm.base import LLMClient
from llm.mock import MockLLMClient
from sources.base import JobSource
from sources.mock import MockJobSource


def build_llm_client(config: Config) -> LLMClient:
    # use_mock_llm short-circuits to the offline backend regardless of provider.
    provider = "mock" if config.use_mock_llm else config.llm_provider
    if provider == "mock":
        return MockLLMClient()
    if provider == "openai":
        from llm.openai import OpenAILLMClient  # lazy: keeps offline runs SDK-free

        return OpenAILLMClient(config)
    from llm.anthropic import AnthropicLLMClient  # lazy: keeps offline runs SDK-free

    return AnthropicLLMClient(config)


def build_job_source(config: Config) -> JobSource:
    if config.use_mock_source:
        return MockJobSource()
    from sources.adzuna import AdzunaJobSource  # lazy: keeps offline runs httpx-free

    return AdzunaJobSource(config)
