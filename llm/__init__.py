"""LLM client layer: an ABC plus concrete backends."""

from llm.base import LLMClient, LLMParseError, parse_strict_json
from llm.mock import MockLLMClient

__all__ = ["LLMClient", "LLMParseError", "MockLLMClient", "parse_strict_json"]

