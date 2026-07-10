"""Deterministic, offline LLM client.

Returns canned extractions so the notebook runs with no API key. Posting
requirements come from ``data/fixtures/extractions.json`` when the posting URL is
known; otherwise a deterministic keyword scan of the description is used as a
fallback (so newly-added fixtures still work). The resume extraction returns a
single canned ``ResumeProfile`` matching the bundled ``sample_resume.txt``.
"""

from __future__ import annotations

import json
from pathlib import Path

from llm.base import LLMClient
from models import ExtractedRequirement, RequirementExtraction, ResumeProfile
from sources.base import JobPosting

_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "extractions.json"

# phrase (lowercased, matched as substring) -> (canonical skill, category)
_KEYWORD_MAP: list[tuple[str, str, str]] = [
    ("natural language processing", "NLP", "domain"),
    ("large language models", "large language models", "domain"),
    ("computer vision", "computer vision", "domain"),
    ("deep learning", "deep learning", "domain"),
    ("machine learning", "machine learning", "domain"),
    ("data visualization", "data visualization", "technical"),
    ("a/b testing", "A/B testing", "domain"),
    ("scikit-learn", "scikit-learn", "technical"),
    ("tensorflow", "TensorFlow", "technical"),
    ("pytorch", "PyTorch", "technical"),
    ("kubernetes", "Kubernetes", "technical"),
    ("docker", "Docker", "technical"),
    ("pandas", "pandas", "technical"),
    ("numpy", "numpy", "technical"),
    ("statistics", "statistics", "domain"),
    ("ci/cd", "CI/CD", "technical"),
    ("rest api", "REST API", "technical"),
    ("spark", "Spark", "technical"),
    ("python", "Python", "technical"),
    ("sql", "SQL", "technical"),
    ("aws", "AWS", "technical"),
    ("gcp", "GCP", "technical"),
    ("git", "Git", "technical"),
    ("nlp", "NLP", "domain"),
    ("communication", "communication", "soft"),
]


def keyword_extract(description: str) -> RequirementExtraction:
    """Deterministically derive requirements from a posting description.

    Longer phrases are checked first and, once matched, their span is blanked so
    a shorter alias (e.g. "nlp" after "natural language processing") does not
    double-count the same skill.
    """
    haystack = f" {description.lower()} "
    seen: set[str] = set()
    reqs: list[ExtractedRequirement] = []
    for phrase, skill, category in _KEYWORD_MAP:
        if phrase in haystack and skill not in seen:
            seen.add(skill)
            reqs.append(
                ExtractedRequirement(skill=skill, category=category, entry_level_expected=True)  # type: ignore[arg-type]
            )
            haystack = haystack.replace(phrase, " ")
    return RequirementExtraction(requirements=reqs)


class MockLLMClient(LLMClient):
    """Canned, deterministic extractions for offline runs."""

    name = "mock"

    def __init__(self, fixture_path: Path | str | None = None) -> None:
        self.fixture_path = Path(fixture_path) if fixture_path else _FIXTURE
        self._data = self._load()

    def _load(self) -> dict:
        if self.fixture_path.exists():
            return json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return {}

    def extract_resume(self, resume_text: str) -> ResumeProfile:
        canned = self._data.get("resume")
        if canned:
            return ResumeProfile.model_validate(canned)
        # Fallback: minimal profile so the pipeline still runs.
        return ResumeProfile(raw_summary=resume_text[:200])

    def extract_requirements(self, posting: JobPosting) -> RequirementExtraction:
        postings = self._data.get("postings", {})
        canned = postings.get(posting.url) if posting.url else None
        if canned is not None:
            return RequirementExtraction.model_validate({"requirements": canned})
        return keyword_extract(posting.description)
