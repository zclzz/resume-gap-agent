"""Typed LangGraph state.

The whole graph state is one Pydantic model -- no raw dicts. Nodes return a
partial dict of field updates which LangGraph merges into this model.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from models import (
    ExtractedRequirement,
    GapReport,
    Requirement,
    ResumeProfile,
    TargetRole,
)
from sources.base import JobPosting


class GraphState(BaseModel):
    # --- inputs ---------------------------------------------------------
    resume_text: str = ""
    target_role: str = ""
    level: str = "entry"
    location: str | None = None

    # --- parsed ---------------------------------------------------------
    profile: ResumeProfile | None = None
    target: TargetRole | None = None

    # --- sourcing / extraction -----------------------------------------
    postings: list[JobPosting] = Field(default_factory=list)
    extractions: list[ExtractedRequirement] = Field(default_factory=list)
    postings_analyzed: int = 0

    # --- aggregation ----------------------------------------------------
    requirements: list[Requirement] = Field(default_factory=list)

    # --- critic ---------------------------------------------------------
    coverage_confidence: float = 0.0
    critic_loops: int = 0
    needs_more: bool = False
    # Agentic mode: the critic's suggestion passed back to the sourcing agent.
    refinement_hint: str | None = None

    # --- output ---------------------------------------------------------
    report: GapReport | None = None
