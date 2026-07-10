"""Deterministic tests for the pure-Python gap diff and taxonomy.

No LLM, no LangGraph, no network. These exercise the core diff logic directly.
"""

from __future__ import annotations

import pytest

from config import Config
from graph.gap_diff import (
    classify_match,
    compute_gap_report,
    remediation_for,
    severity_for,
)
from models import GapSeverity, Requirement, ResumeProfile
from taxonomy import normalize_skill, normalize_skills


@pytest.fixture
def config() -> Config:
    # Explicit thresholds so the test is independent of env/.env overrides.
    return Config(
        critical_frequency=0.6,
        important_frequency=0.3,
        _env_file=None,  # type: ignore[call-arg]
    )


# --------------------------------------------------------------------------- #
# taxonomy                                                                     #
# --------------------------------------------------------------------------- #


def test_normalize_skill_aliases():
    assert normalize_skill("sklearn") == "scikit-learn"
    assert normalize_skill("Scikit Learn") == "scikit-learn"
    assert normalize_skill("PY") == "python"
    assert normalize_skill("ML") == "machine learning"
    assert normalize_skill("AWS") == "amazon web services"


def test_normalize_skills_dedupes_and_orders():
    result = normalize_skills(["Python", "py", "SQL", "sql", "sklearn"])
    assert result == ["python", "sql", "scikit-learn"]


def test_normalize_skill_empty():
    assert normalize_skill("   ") == ""
    assert normalize_skills(["", "  ", "Python"]) == ["python"]


# --------------------------------------------------------------------------- #
# severity                                                                     #
# --------------------------------------------------------------------------- #


def test_severity_thresholds(config: Config):
    assert severity_for(0.9, config) is GapSeverity.critical
    assert severity_for(0.6, config) is GapSeverity.critical
    assert severity_for(0.45, config) is GapSeverity.important
    assert severity_for(0.3, config) is GapSeverity.important
    assert severity_for(0.1, config) is GapSeverity.nice


# --------------------------------------------------------------------------- #
# classify_match                                                               #
# --------------------------------------------------------------------------- #


def _req(skill: str, freq: float = 0.5, category: str = "technical") -> Requirement:
    return Requirement(skill=skill, category=category, frequency=freq)  # type: ignore[arg-type]


def test_classify_matched():
    skills = set(normalize_skills(["Python", "scikit-learn"]))
    status, evidence = classify_match(_req("sklearn"), skills, "")
    assert status == "matched"
    assert evidence == "scikit-learn"


def test_classify_weak_from_freetext():
    skills = set(normalize_skills(["Python"]))
    haystack = normalize_skill("Built dashboards using docker in a side project")
    status, evidence = classify_match(_req("Docker"), skills, haystack)
    assert status == "weak"
    assert evidence == "docker"


def test_classify_missing():
    skills = set(normalize_skills(["Python"]))
    status, evidence = classify_match(_req("Kubernetes"), skills, "no relevant text")
    assert status == "missing"
    assert evidence is None


# --------------------------------------------------------------------------- #
# remediation                                                                  #
# --------------------------------------------------------------------------- #


def test_remediation_uses_category_template():
    tech = remediation_for(_req("PyTorch", category="technical"))
    cred = remediation_for(Requirement(skill="AWS Cert", category="credential", frequency=0.4))
    assert "PyTorch" in tech and "portfolio" in tech.lower()
    assert "certification" in cred.lower()


# --------------------------------------------------------------------------- #
# compute_gap_report (end-to-end, deterministic)                              #
# --------------------------------------------------------------------------- #


def test_compute_gap_report_partitions_and_sorts(config: Config):
    resume = ResumeProfile(
        skills=["Python", "SQL", "scikit-learn", "pandas"],
        years_experience=0.5,
        education=["B.Sc. Computer Science"],
        credentials=["ML Specialization"],
        raw_summary="Built models and dashboards; some exposure to docker in coursework.",
    )
    requirements = [
        _req("Python", freq=1.0),           # matched
        _req("scikit-learn", freq=0.8),     # matched
        _req("PyTorch", freq=0.7),          # missing -> critical
        _req("Docker", freq=0.4),           # weak (in summary) -> important
        _req("Kubernetes", freq=0.2),       # missing -> nice
    ]

    report = compute_gap_report(
        resume,
        requirements,
        target_role="Machine Learning Engineer",
        level="entry",
        postings_analyzed=12,
        config=config,
    )

    assert report.postings_analyzed == 12
    assert set(report.matched_skills) == {"Python", "scikit-learn"}

    # Three gaps, sorted by severity then descending frequency.
    assert [g.requirement.skill for g in report.gaps] == ["PyTorch", "Docker", "Kubernetes"]
    assert [g.severity for g in report.gaps] == [
        GapSeverity.critical,
        GapSeverity.important,
        GapSeverity.nice,
    ]

    # Weak gap carries evidence; missing gaps do not.
    docker_gap = next(g for g in report.gaps if g.requirement.skill == "Docker")
    assert docker_gap.resume_evidence == "docker"
    pytorch_gap = next(g for g in report.gaps if g.requirement.skill == "PyTorch")
    assert pytorch_gap.resume_evidence is None


def test_compute_gap_report_empty_requirements(config: Config):
    resume = ResumeProfile(skills=["Python"], raw_summary="hello")
    report = compute_gap_report(
        resume, [], target_role="MLE", level="entry", postings_analyzed=0, config=config
    )
    assert report.gaps == []
    assert report.matched_skills == []
