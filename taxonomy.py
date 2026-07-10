"""Skill-name normalization.

Deterministic, dependency-free canonicalization of skill strings so that
"sklearn", "Scikit Learn", and "scikit-learn" all collapse to one token before
the pure-Python gap diff runs. This is intentionally *not* an LLM step: keeping
it deterministic makes the diff reproducible and testable.
"""

from __future__ import annotations

import re

# Canonical skill -> set of aliases (all compared lowercased & punctuation-stripped).
_ALIAS_GROUPS: dict[str, list[str]] = {
    "python": ["py", "python3"],
    "scikit-learn": ["sklearn", "scikit learn", "sci-kit learn"],
    "machine learning": ["ml"],
    "deep learning": ["dl"],
    "natural language processing": ["nlp"],
    "computer vision": ["cv"],
    "pytorch": ["torch"],
    "tensorflow": ["tf", "tensor flow"],
    "amazon web services": ["aws"],
    "google cloud platform": ["gcp", "google cloud"],
    "microsoft azure": ["azure"],
    "sql": ["structured query language"],
    "postgresql": ["postgres", "psql"],
    "javascript": ["js"],
    "typescript": ["ts"],
    "continuous integration": ["ci", "ci/cd", "cicd"],
    "large language models": ["llm", "llms"],
    "data visualization": ["dataviz", "data viz"],
    "statistics": ["stats"],
    "a/b testing": ["ab testing", "a b testing"],
    "rest api": ["restful api", "rest apis", "restful apis"],
}

# Reverse index: alias -> canonical.
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for _canonical, _aliases in _ALIAS_GROUPS.items():
    _ALIAS_TO_CANONICAL[_canonical] = _canonical
    for _alias in _aliases:
        _ALIAS_TO_CANONICAL[_alias] = _canonical


_PUNCT_RE = re.compile(r"[^a-z0-9+#./ ]+")
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Lowercase, strip punctuation (keeping a few skill-significant chars), squeeze ws."""
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def normalize_skill(skill: str) -> str:
    """Return the canonical form of a single skill string."""
    cleaned = _clean(skill)
    if not cleaned:
        return ""
    # Alias table lookup, then fall back to the cleaned form.
    return _ALIAS_TO_CANONICAL.get(cleaned, cleaned)


def normalize_skills(skills: list[str]) -> list[str]:
    """Normalize a list of skills, dropping empties and preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for skill in skills:
        norm = normalize_skill(skill)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out
