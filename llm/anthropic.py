"""Real LLM client backed by the Anthropic API.

Lazy-imports the ``anthropic`` SDK so that offline (mock) notebook runs never
require it to be installed. Requirement/resume extraction requests strict JSON,
parses into Pydantic, and retries once (configurable) on parse failure.
"""

from __future__ import annotations

from config import Config
from llm.base import LLMClient, extract_with_retry
from logging_conf import get_logger
from models import RequirementExtraction, ResumeProfile
from prompts import templates
from sources.base import JobPosting

log = get_logger("llm.anthropic")


class AnthropicLLMClient(LLMClient):
    """Anthropic-backed extraction client."""

    name = "anthropic"

    def __init__(self, config: Config) -> None:
        if not config.anthropic_api_key:
            raise ValueError(
                "RGA_ANTHROPIC_API_KEY is required for AnthropicLLMClient. "
                "Use MockLLMClient for offline runs."
            )
        try:
            import anthropic  # noqa: PLC0415  (lazy, optional dependency)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "The 'anthropic' package is required. Install with: pip install anthropic"
            ) from exc

        self._config = config
        self._client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    # ------------------------------------------------------------------ #
    def _complete(self, system: str, user: str) -> str:
        resp = self._client.messages.create(
            model=self._config.llm_model,
            max_tokens=self._config.llm_max_tokens,
            temperature=self._config.llm_temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate text blocks from the response.
        return "".join(block.text for block in resp.content if block.type == "text")

    def _extract(self, system: str, user: str, model_cls, *, kind: str):
        return extract_with_retry(
            self._complete,
            system,
            user,
            model_cls,
            retries=self._config.llm_parse_retries,
            kind=kind,
            log=log,
        )

    # ------------------------------------------------------------------ #
    def extract_resume(self, resume_text: str) -> ResumeProfile:
        user = templates.RESUME_EXTRACTION_USER.format(resume_text=resume_text)
        return self._extract(
            templates.RESUME_EXTRACTION_SYSTEM, user, ResumeProfile, kind="resume"
        )

    def extract_requirements(self, posting: JobPosting) -> RequirementExtraction:
        user = templates.REQUIREMENT_EXTRACTION_USER.format(
            title=posting.title,
            company=posting.company or "N/A",
            description=posting.description,
        )
        return self._extract(
            templates.REQUIREMENT_EXTRACTION_SYSTEM,
            user,
            RequirementExtraction,
            kind="requirements",
        )
