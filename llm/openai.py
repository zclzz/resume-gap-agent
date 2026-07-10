"""Real LLM client backed by the OpenAI API.

Lazy-imports the ``openai`` SDK so offline (mock) runs never require it. Uses
OpenAI's JSON mode (``response_format={"type": "json_object"}``) so the model
returns strict JSON, which is then parsed into Pydantic with one retry on
failure. Shares the exact prompt templates and retry logic with the Anthropic
backend -- only the transport differs.
"""

from __future__ import annotations

from config import Config
from llm.base import LLMClient, extract_with_retry
from logging_conf import get_logger
from models import RequirementExtraction, ResumeProfile
from prompts import templates
from sources.base import JobPosting

log = get_logger("llm.openai")


class OpenAILLMClient(LLMClient):
    """OpenAI-backed extraction client."""

    name = "openai"

    def __init__(self, config: Config) -> None:
        if not config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY (or RGA_OPENAI_API_KEY) is required for OpenAILLMClient. "
                "Use MockLLMClient for offline runs."
            )
        try:
            from openai import OpenAI  # noqa: PLC0415 (lazy, optional dependency)
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "The 'openai' package is required. Install with: pip install openai"
            ) from exc

        self._config = config
        self._client = OpenAI(api_key=config.openai_api_key)

    def _complete(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._config.openai_model,
            temperature=self._config.llm_temperature,
            max_tokens=self._config.llm_max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""

    def extract_resume(self, resume_text: str) -> ResumeProfile:
        user = templates.RESUME_EXTRACTION_USER.format(resume_text=resume_text)
        return extract_with_retry(
            self._complete,
            templates.RESUME_EXTRACTION_SYSTEM,
            user,
            ResumeProfile,
            retries=self._config.llm_parse_retries,
            kind="resume",
            log=log,
        )

    def extract_requirements(self, posting: JobPosting) -> RequirementExtraction:
        user = templates.REQUIREMENT_EXTRACTION_USER.format(
            title=posting.title,
            company=posting.company or "N/A",
            description=posting.description,
        )
        return extract_with_retry(
            self._complete,
            templates.REQUIREMENT_EXTRACTION_SYSTEM,
            user,
            RequirementExtraction,
            retries=self._config.llm_parse_retries,
            kind="requirements",
            log=log,
        )
