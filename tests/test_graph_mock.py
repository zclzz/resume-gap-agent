"""Full-pipeline test against the mock source + mock LLM (no network, no API key)."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import Config
from graph.build import build_checkpointer, build_graph
from graph.state import GraphState
from models import GapReport

_RESUME = (Path(__file__).resolve().parent.parent / "data" / "sample_resume.txt").read_text(
    encoding="utf-8"
)


@pytest.fixture
def config() -> Config:
    return Config(use_mock_llm=True, use_mock_source=True, _env_file=None)  # type: ignore[call-arg]


def _run(config: Config) -> GapReport:
    graph = build_graph(config, checkpointer=build_checkpointer(":memory:"))
    result = graph.invoke(
        GraphState(
            resume_text=_RESUME,
            target_role="Machine Learning Engineer",
            level="entry",
            location="Singapore",
        ),
        config={"configurable": {"thread_id": "test-1"}},
    )
    # LangGraph returns the state as a dict-like; normalize to GraphState.
    state = GraphState.model_validate(result)
    assert state.report is not None
    return state.report


def test_graph_runs_end_to_end(config: Config):
    report = _run(config)
    assert report.target_role == "Machine Learning Engineer"
    assert report.postings_analyzed >= config.min_postings
    # Resume clearly has Python + scikit-learn, which appear in postings.
    assert "Python" in report.matched_skills
    assert any("scikit" in s.lower() for s in report.matched_skills)


def test_graph_identifies_critical_gap(config: Config):
    report = _run(config)
    gap_skills = {g.requirement.skill.lower() for g in report.gaps}
    # PyTorch is required by many postings but absent from the resume.
    assert "pytorch" in gap_skills
    # Every gap has a severity and remediation.
    assert all(g.remediation for g in report.gaps)


def test_graph_is_deterministic(config: Config):
    a = _run(config)
    b = _run(config)
    assert [g.requirement.skill for g in a.gaps] == [g.requirement.skill for g in b.gaps]
    assert a.matched_skills == b.matched_skills
