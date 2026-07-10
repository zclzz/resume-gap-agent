"""The deterministic core: resume-vs-requirements diff in pure Python.

No LLM, no LangGraph, no I/O. Given a ``ResumeProfile`` and aggregated
``Requirement`` list, produce a ``GapReport`` using plain set logic. This module
is imported by the graph node *and* by the unit tests -- the tests exercise it
directly with hand-built inputs, no mocks required.
"""

from __future__ import annotations

from config import Config
from models import Gap, GapReport, GapSeverity, Requirement, ResumeProfile
from taxonomy import normalize_skill, normalize_skills

# ---------------------------------------------------------------------------- #
# Match classification                                                         #
# ---------------------------------------------------------------------------- #

MatchStatus = str  # "matched" | "weak" | "missing"


def classify_match(
    requirement: Requirement,
    resume_skill_set: set[str],
    resume_haystack: str,
) -> tuple[MatchStatus, str | None]:
    """Classify a single requirement against a resume.

    * ``matched`` -- the canonical skill is in the resume's explicit skill set.
    * ``weak``    -- the skill isn't listed, but its words appear in the resume
                     free-text (evidence without an explicit claim).
    * ``missing`` -- no evidence anywhere.
    """
    canonical = normalize_skill(requirement.skill)
    if canonical in resume_skill_set:
        return "matched", canonical
    if canonical and canonical in resume_haystack:
        return "weak", canonical
    return "missing", None


# ---------------------------------------------------------------------------- #
# Severity                                                                      #
# ---------------------------------------------------------------------------- #


def severity_for(frequency: float, config: Config) -> GapSeverity:
    """Map a requirement frequency to a gap severity using config thresholds."""
    if frequency >= config.critical_frequency:
        return GapSeverity.critical
    if frequency >= config.important_frequency:
        return GapSeverity.important
    return GapSeverity.nice


# ---------------------------------------------------------------------------- #
# Remediation (deterministic templates keyed by category)                      #
# ---------------------------------------------------------------------------- #

_REMEDIATION_TEMPLATES: dict[str, str] = {
    "technical": "Build a small portfolio project that uses {skill}; add it to your resume.",
    "domain": "Take an introductory course or read up on {skill} and note it in your summary.",
    "soft": "Highlight concrete examples that demonstrate {skill} in your experience bullets.",
    "credential": "Pursue the {skill} certification or an equivalent recognized credential.",
}


def remediation_for(requirement: Requirement) -> str:
    template = _REMEDIATION_TEMPLATES.get(
        requirement.category, _REMEDIATION_TEMPLATES["technical"]
    )
    return template.format(skill=requirement.skill)


# ---------------------------------------------------------------------------- #
# Top-level diff                                                                #
# ---------------------------------------------------------------------------- #


def compute_gap_report(
    resume: ResumeProfile,
    requirements: list[Requirement],
    *,
    target_role: str,
    level: str,
    postings_analyzed: int,
    config: Config,
) -> GapReport:
    """Produce a full ``GapReport`` deterministically.

    A requirement that is ``matched`` becomes an entry in ``matched_skills``;
    ``weak`` and ``missing`` requirements become ``Gap`` objects (weak gaps carry
    the resume evidence snippet). Gaps are sorted by severity then descending
    frequency so the most urgent items lead the report.
    """
    resume_skill_set = set(normalize_skills(resume.skills))
    # Free-text haystack: summary + education + credentials, normalized.
    haystack_parts = [resume.raw_summary, *resume.education, *resume.credentials]
    resume_haystack = normalize_skill(" ".join(haystack_parts))

    matched: list[str] = []
    gaps: list[Gap] = []

    for req in requirements:
        status, evidence = classify_match(req, resume_skill_set, resume_haystack)
        if status == "matched":
            matched.append(req.skill)
            continue

        gaps.append(
            Gap(
                requirement=req,
                resume_evidence=evidence if status == "weak" else None,
                severity=severity_for(req.frequency, config),
                remediation=remediation_for(req),
            )
        )

    _severity_rank = {
        GapSeverity.critical: 0,
        GapSeverity.important: 1,
        GapSeverity.nice: 2,
    }
    gaps.sort(key=lambda g: (_severity_rank[g.severity], -g.requirement.frequency))

    return GapReport(
        target_role=target_role,
        level=level,
        postings_analyzed=postings_analyzed,
        matched_skills=matched,
        gaps=gaps,
    )
