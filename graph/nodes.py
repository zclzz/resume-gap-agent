"""LangGraph nodes.

Each node logs entry/exit with key counts and latency. LLMs are used only in
``parse_resume`` and ``extract_requirements``; every other node is deterministic
Python. Nodes take the typed ``GraphState`` and return a partial dict of updates.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from config import Config
from graph.aio import run_coro as _run
from graph.gap_diff import compute_gap_report
from graph.state import GraphState
from llm.base import LLMClient
from logging_conf import get_logger
from models import ExtractedRequirement, Requirement, TargetRole
from sources.base import JobSource
from taxonomy import normalize_skill

log = get_logger("graph.nodes")


@dataclass
class NodeContext:
    """Dependencies injected into every node (bound via closures in build.py)."""

    config: Config
    llm: LLMClient
    source: JobSource
    # Agentic-mode extras (None in deterministic mode).
    chat_model: Any = None
    mcp_tools: list[Any] = field(default_factory=list)
    mcp_client: Any = None


# --------------------------------------------------------------------------- #
# Nodes                                                                        #
# --------------------------------------------------------------------------- #


def parse_resume(state: GraphState, ctx: NodeContext) -> dict:
    t0 = time.perf_counter()
    log.info("parse_resume.start", chars=len(state.resume_text), llm=ctx.llm.name)
    profile = ctx.llm.extract_resume(state.resume_text)
    log.info(
        "parse_resume.done",
        skills=len(profile.skills),
        years_experience=profile.years_experience,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return {"profile": profile}


def parse_target_role(state: GraphState, ctx: NodeContext) -> dict:
    t0 = time.perf_counter()
    log.info("parse_target_role.start", role=state.target_role, level=state.level)
    role = state.target_role.strip()
    level = state.level.strip().lower()
    canonical = " ".join(f"{level} {role}".lower().split())
    target = TargetRole(
        role=role, level=level, location=state.location, canonical_query=canonical
    )
    log.info(
        "parse_target_role.done",
        canonical_query=target.canonical_query,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return {"target": target}


def source_postings(state: GraphState, ctx: NodeContext) -> dict:
    t0 = time.perf_counter()
    query = state.target.canonical_query if state.target else state.target_role
    limit = ctx.config.target_postings
    log.info("source_postings.start", query=query, limit=limit, source=ctx.source.name)

    raw = _run(ctx.source.search(query, state.level, state.location, limit))
    filtered = [
        p for p in raw if ctx.source.matches_level(p, state.level, ctx.config.entry_level_titles)
    ]

    log.info(
        "source_postings.done",
        returned=len(raw),
        entry_level=len(filtered),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return {"postings": filtered}


def extract_requirements(state: GraphState, ctx: NodeContext) -> dict:
    t0 = time.perf_counter()
    log.info("extract_requirements.start", postings=len(state.postings), llm=ctx.llm.name)
    extractions: list[ExtractedRequirement] = []
    for posting in state.postings:
        result = ctx.llm.extract_requirements(posting)
        # Dedupe per posting so each skill counts at most once toward frequency.
        seen: set[str] = set()
        for req in result.requirements:
            key = normalize_skill(req.skill)
            if key and key not in seen:
                seen.add(key)
                extractions.append(req)
    log.info(
        "extract_requirements.done",
        postings=len(state.postings),
        extractions=len(extractions),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return {"extractions": extractions, "postings_analyzed": len(state.postings)}


def aggregate_requirements(state: GraphState, ctx: NodeContext) -> dict:
    t0 = time.perf_counter()
    total = max(state.postings_analyzed, 1)
    log.info("aggregate_requirements.start", extractions=len(state.extractions), postings=total)

    # Group by normalized skill; keep readable display name + majority category.
    display: dict[str, Counter] = {}
    categories: dict[str, Counter] = {}
    entry_votes: dict[str, list[bool]] = {}
    counts: Counter = Counter()

    for req in state.extractions:
        key = normalize_skill(req.skill)
        if not key:
            continue
        counts[key] += 1
        display.setdefault(key, Counter())[req.skill] += 1
        categories.setdefault(key, Counter())[req.category] += 1
        entry_votes.setdefault(key, []).append(req.entry_level_expected)

    requirements: list[Requirement] = []
    for key, count in counts.items():
        skill_name = display[key].most_common(1)[0][0]
        category = categories[key].most_common(1)[0][0]
        entry_expected = sum(entry_votes[key]) >= (len(entry_votes[key]) / 2)
        requirements.append(
            Requirement(
                skill=skill_name,
                category=category,
                frequency=round(count / total, 3),
                entry_level_expected=entry_expected,
            )
        )
    requirements.sort(key=lambda r: r.frequency, reverse=True)

    log.info(
        "aggregate_requirements.done",
        unique_skills=len(requirements),
        top=[r.skill for r in requirements[:5]],
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return {"requirements": requirements}


def gap_diff(state: GraphState, ctx: NodeContext) -> dict:
    t0 = time.perf_counter()
    log.info(
        "gap_diff.start",
        resume_skills=len(state.profile.skills) if state.profile else 0,
        requirements=len(state.requirements),
    )
    assert state.profile is not None, "profile must be parsed before gap_diff"
    report = compute_gap_report(
        state.profile,
        state.requirements,
        target_role=state.target_role,
        level=state.level,
        postings_analyzed=state.postings_analyzed,
        config=ctx.config,
    )
    log.info(
        "gap_diff.done",
        matched=len(report.matched_skills),
        gaps=len(report.gaps),
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return {"report": report}


def critic(state: GraphState, ctx: NodeContext) -> dict:
    t0 = time.perf_counter()
    analyzed = state.postings_analyzed
    confidence = min(1.0, analyzed / ctx.config.target_postings)
    under_min = analyzed < ctx.config.min_postings
    low_confidence = confidence < ctx.config.coverage_min_confidence
    can_loop = state.critic_loops < ctx.config.max_critic_loops
    needs_more = (under_min or low_confidence) and can_loop
    next_loops = state.critic_loops + 1 if needs_more else state.critic_loops

    log.info(
        "critic.done",
        postings_analyzed=analyzed,
        coverage_confidence=round(confidence, 3),
        needs_more=needs_more,
        critic_loops=next_loops,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return {
        "coverage_confidence": confidence,
        "needs_more": needs_more,
        "critic_loops": next_loops,
    }


def report(state: GraphState, ctx: NodeContext) -> dict:
    rep = state.report
    log.info(
        "report.done",
        target_role=rep.target_role if rep else None,
        postings_analyzed=rep.postings_analyzed if rep else 0,
        matched=len(rep.matched_skills) if rep else 0,
        gaps=len(rep.gaps) if rep else 0,
    )
    return {}


def route_after_critic(state: GraphState) -> str:
    """Conditional edge: loop back to sourcing if the critic wants more coverage."""
    return "source_postings" if state.needs_more else "report"
