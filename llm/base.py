"""LLM client abstraction.

The LLM is used only at the *edges* of the pipeline: turning raw resume text
into a ``ResumeProfile`` and turning a single posting into a
``RequirementExtraction``. Both are strict-JSON tasks. The deterministic diff
never goes near a prompt.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from models import RequirementExtraction, ResumeProfile
from sources.base import JobPosting

TModel = TypeVar("TModel", bound=BaseModel)

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class LLMParseError(ValueError):
    """Raised when an LLM response cannot be parsed into the target model."""


def parse_strict_json(text: str, model_cls: type[TModel]) -> TModel:
    """Parse an LLM response into ``model_cls``, tolerating markdown fences.

    Raises :class:`LLMParseError` on malformed JSON or schema violations so the
    caller can retry.
    """
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMParseError(f"invalid JSON: {exc}") from exc
    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise LLMParseError(f"schema validation failed: {exc}") from exc


def extract_with_retry(
    complete,
    system: str,
    user: str,
    model_cls: type[TModel],
    *,
    retries: int,
    kind: str,
    log,
) -> TModel:
    """Call ``complete(system, user)`` and parse into ``model_cls``, retrying on
    parse failure. Shared by every real backend so retry behavior is identical.

    ``complete`` is a backend-specific callable that returns the model's raw text.
    On a parse failure the user prompt is nudged toward strict JSON and retried,
    up to ``retries`` extra attempts.
    """
    attempts = retries + 1
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        raw = complete(system, user)
        try:
            return parse_strict_json(raw, model_cls)
        except LLMParseError as exc:
            last_error = exc
            log.warning("llm_parse_retry", kind=kind, attempt=attempt, error=str(exc))
            user = user + "\n\nYour previous response was not valid JSON. Return STRICT JSON only."
    raise LLMParseError(f"{kind}: failed after {attempts} attempts: {last_error}")


class LLMClient(ABC):
    """Abstract LLM client. Two purpose-specific extraction methods."""

    name: str = "abstract"

    @abstractmethod
    def extract_resume(self, resume_text: str) -> ResumeProfile:
        """Extract a normalized ``ResumeProfile`` from raw resume text."""
        raise NotImplementedError

    @abstractmethod
    def extract_requirements(self, posting: JobPosting) -> RequirementExtraction:
        """Extract the requirements from a single posting."""
        raise NotImplementedError
