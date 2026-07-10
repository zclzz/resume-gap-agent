"""Agentic-mode nodes: an LLM sourcing agent and an LLM critic.

These replace the deterministic ``source_postings`` and ``critic`` nodes when
``config.mode == "agentic"``. Here the LLM genuinely drives control flow:

* ``sourcing_agent`` is a ReAct agent that decides how to query, whether to
  refine the search, and which postings to keep -- by calling MCP tools.
* ``critic_agent`` decides (via structured output) whether coverage is good
  enough or the pipeline should loop back and source more.

Everything downstream (extraction, aggregation, gap diff) stays deterministic.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from graph.aio import run_coro
from graph.mcp_client import parse_tool_result
from graph.nodes import NodeContext
from graph.state import GraphState
from logging_conf import get_logger
from models import CriticDecision, SourcingResult
from prompts import agents as prompts
from sources.base import JobPosting

log = get_logger("graph.agents")


def _tool_by_name(ctx: NodeContext, name: str):
    for tool in ctx.mcp_tools:
        if tool.name == name:
            return tool
    raise KeyError(f"MCP tool not found: {name}")


def _reconstruct_postings(ctx: NodeContext, posting_ids: list[str]) -> list[JobPosting]:
    """Fetch full posting details for the agent's chosen ids via MCP."""
    if not posting_ids:
        return []
    details_tool = _tool_by_name(ctx, "get_posting_details")
    raw = run_coro(details_tool.ainvoke({"posting_ids": posting_ids}))
    postings: list[JobPosting] = []
    for item in parse_tool_result(raw):
        if "error" in item or "description" not in item:
            continue
        postings.append(JobPosting(**item))
    return postings


def _fallback_search(ctx: NodeContext, state: GraphState) -> list[str]:
    """If the agent's output is unusable, do a plain search so the graph survives."""
    search_tool = _tool_by_name(ctx, "search_jobs")
    raw = run_coro(
        search_tool.ainvoke(
            {
                "role": state.target_role,
                "level": state.level,
                "location": state.location or "",
                "limit": ctx.config.target_postings,
            }
        )
    )
    return [row["posting_id"] for row in parse_tool_result(raw) if "posting_id" in row]


def sourcing_agent(state: GraphState, ctx: NodeContext) -> dict:
    t0 = time.perf_counter()
    refinement = (
        f"A previous pass judged coverage insufficient. Refinement hint: {state.refinement_hint}"
        if state.refinement_hint
        else ""
    )
    system = prompts.SOURCING_AGENT_SYSTEM.format(
        level=state.level,
        target=ctx.config.target_postings,
        min=ctx.config.min_postings,
    )
    task = prompts.SOURCING_AGENT_TASK.format(
        role=state.target_role,
        level=state.level,
        location=state.location or "any",
        target=ctx.config.target_postings,
        min=ctx.config.min_postings,
        refinement=refinement,
    )
    log.info(
        "sourcing_agent.start",
        role=state.target_role,
        tools=[t.name for t in ctx.mcp_tools],
        looped=bool(state.refinement_hint),
    )

    # checkpointer=False: the sub-agent must NOT inherit the outer graph's sync
    # SqliteSaver (it has no thread_id and would break on the async path).
    agent = create_react_agent(
        ctx.chat_model, ctx.mcp_tools, prompt=system, checkpointer=False
    )
    try:
        result = run_coro(
            agent.ainvoke(
                {"messages": [("user", task)]},
                config={"recursion_limit": ctx.config.agent_recursion_limit},
            )
        )
        messages = result["messages"]
    except GraphRecursionError:
        # Agent explored past its step budget without answering; the plain-search
        # fallback below keeps the pipeline alive.
        log.warning(
            "sourcing_agent.recursion_limit", limit=ctx.config.agent_recursion_limit
        )
        messages = []
    tool_calls = sum(1 for m in messages if getattr(m, "type", "") == "tool")
    final_text = messages[-1].content if messages else ""

    # Parse the agent's structured decision; fall back to a plain search.
    parsed: SourcingResult | None = None
    try:
        cleaned = final_text.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        parsed = SourcingResult.model_validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("sourcing_agent.parse_failed", error=str(exc))

    if parsed and parsed.posting_ids:
        posting_ids = parsed.posting_ids
        query_used = parsed.query_used
        reasoning = parsed.reasoning
    else:
        posting_ids = _fallback_search(ctx, state)
        query_used = f"{state.level} {state.target_role}"
        reasoning = "fallback: used plain search after unparseable agent output"

    postings = _reconstruct_postings(ctx, posting_ids)
    entry_level = [
        p
        for p in postings
        if ctx.source.matches_level(p, state.level, ctx.config.entry_level_titles)
    ]

    log.info(
        "sourcing_agent.done",
        agent_tool_calls=tool_calls,
        query_used=query_used,
        chosen=len(posting_ids),
        reconstructed=len(postings),
        entry_level=len(entry_level),
        reasoning=reasoning,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    # Clear any refinement hint now that we've acted on it.
    return {"postings": entry_level, "refinement_hint": None}


def critic_agent(state: GraphState, ctx: NodeContext) -> dict:
    t0 = time.perf_counter()
    top = "\n".join(
        f"- {r.skill} @ {r.frequency:.0%}" for r in state.requirements[:8]
    ) or "(none)"
    task = prompts.CRITIC_TASK.format(
        role=state.target_role,
        level=state.level,
        analyzed=state.postings_analyzed,
        target=ctx.config.target_postings,
        min=ctx.config.min_postings,
        loops=state.critic_loops,
        max_loops=ctx.config.max_critic_loops,
        top_requirements=top,
        matched=len(state.report.matched_skills) if state.report else 0,
        gaps=len(state.report.gaps) if state.report else 0,
    )
    log.info("critic_agent.start", postings_analyzed=state.postings_analyzed, loops=state.critic_loops)

    structured = ctx.chat_model.with_structured_output(CriticDecision)
    decision: CriticDecision = structured.invoke(
        [("system", prompts.CRITIC_SYSTEM), ("user", task)]
    )

    # The LLM decides -- but a hard cap prevents infinite looping.
    can_loop = state.critic_loops < ctx.config.max_critic_loops
    needs_more = bool(decision.needs_more) and can_loop
    next_loops = state.critic_loops + 1 if needs_more else state.critic_loops

    log.info(
        "critic_agent.done",
        llm_needs_more=decision.needs_more,
        capped_needs_more=needs_more,
        reason=decision.reason,
        critic_loops=next_loops,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
    )
    return {
        "needs_more": needs_more,
        "critic_loops": next_loops,
        "refinement_hint": decision.refinement_hint if needs_more else None,
        "coverage_confidence": min(1.0, state.postings_analyzed / ctx.config.target_postings),
    }
