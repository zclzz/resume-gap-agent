"""Pydantic data models shared across the pipeline.

Every payload that crosses a node boundary is one of these typed models -- no
raw dicts flow between nodes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

RequirementCategory = Literal["technical", "domain", "soft", "credential"]


class GapSeverity(str, Enum):
    critical = "critical"
    important = "important"
    nice = "nice_to_have"


class ResumeProfile(BaseModel):
    """Normalized view of an applicant's resume."""

    skills: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    education: list[str] = Field(default_factory=list)
    credentials: list[str] = Field(default_factory=list)
    raw_summary: str = ""


class ExtractedRequirement(BaseModel):
    """A requirement as pulled from a *single* posting by the LLM (pre-aggregation).

    No frequency yet -- that is computed deterministically during aggregation.
    """

    skill: str
    category: RequirementCategory
    entry_level_expected: bool = True


class RequirementExtraction(BaseModel):
    """Strict-JSON wrapper the LLM must return for one posting."""

    requirements: list[ExtractedRequirement] = Field(default_factory=list)


class Requirement(BaseModel):
    """A single skill/requirement extracted (and aggregated) from postings."""

    skill: str
    category: RequirementCategory
    frequency: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of analyzed postings that require this skill.",
    )
    entry_level_expected: bool = True


class TargetRole(BaseModel):
    """Normalized target role/level/location plus a canonical search query."""

    role: str
    level: str
    location: str | None = None
    canonical_query: str


class SourcingResult(BaseModel):
    """Structured output the LLM sourcing agent must return."""

    posting_ids: list[str] = Field(default_factory=list)
    query_used: str = ""
    reasoning: str = ""


class CriticDecision(BaseModel):
    """The LLM critic's judgement on whether coverage is sufficient."""

    needs_more: bool = Field(description="True if we should source MORE postings.")
    reason: str = Field(description="One-sentence justification.")
    refinement_hint: str | None = Field(
        default=None,
        description="If needs_more, a concrete suggestion for the next search (e.g. reword the query).",
    )


class Gap(BaseModel):
    """A requirement the resume does not (fully) satisfy."""

    requirement: Requirement
    resume_evidence: str | None = None
    severity: GapSeverity
    remediation: str


class GapReport(BaseModel):
    """The final deliverable produced by the pipeline."""

    target_role: str
    level: str
    postings_analyzed: int
    matched_skills: list[str] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
