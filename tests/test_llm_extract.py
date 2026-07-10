"""Deterministic tests for the shared LLM parse/retry logic (no network).

These exercise the transport-agnostic helpers that every real backend
(Anthropic, OpenAI) relies on, using a fake ``complete`` callable.
"""

from __future__ import annotations

import pytest
import structlog

from llm.base import LLMParseError, extract_with_retry, parse_strict_json
from models import RequirementExtraction, ResumeProfile

log = structlog.get_logger("test")


def test_parse_strict_json_plain():
    profile = parse_strict_json('{"skills": ["Python"], "raw_summary": "hi"}', ResumeProfile)
    assert profile.skills == ["Python"]


def test_parse_strict_json_strips_markdown_fence():
    text = "```json\n{\"requirements\": [{\"skill\": \"SQL\", \"category\": \"technical\"}]}\n```"
    ext = parse_strict_json(text, RequirementExtraction)
    assert ext.requirements[0].skill == "SQL"


def test_parse_strict_json_rejects_garbage():
    with pytest.raises(LLMParseError):
        parse_strict_json("not json at all", ResumeProfile)


def test_extract_with_retry_recovers_after_bad_response():
    calls = {"n": 0}

    def flaky_complete(system: str, user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "oops not json"  # first attempt fails -> triggers retry
        return '{"skills": ["Python", "SQL"], "raw_summary": "ok"}'

    profile = extract_with_retry(
        flaky_complete, "sys", "user", ResumeProfile, retries=1, kind="resume", log=log
    )
    assert calls["n"] == 2
    assert profile.skills == ["Python", "SQL"]


def test_extract_with_retry_gives_up_after_retries():
    def always_bad(system: str, user: str) -> str:
        return "still not json"

    with pytest.raises(LLMParseError):
        extract_with_retry(
            always_bad, "sys", "user", ResumeProfile, retries=1, kind="resume", log=log
        )


def test_config_reads_plain_openai_api_key(monkeypatch):
    from config import Config

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    cfg = Config(_env_file=None)  # type: ignore[call-arg]
    assert cfg.openai_api_key == "sk-test-123"
