"""Graph assembly + SQLite checkpointer.

Wires the nodes into the pipeline topology and compiles with a
``SqliteSaver`` so a run can resume after a failure without re-sourcing or
re-calling the LLM for nodes that already completed.

Topology::

    parse_resume -> parse_target_role -> source_postings -> extract_requirements
      -> aggregate_requirements -> gap_diff -> critic
    critic --(needs_more)--> source_postings
    critic --(else)--------> report -> END
"""

from __future__ import annotations

import sqlite3
from functools import partial

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from config import Config
from factory import build_job_source, build_llm_client
from graph import nodes
from graph.nodes import NodeContext
from graph.state import GraphState
from llm.base import LLMClient
from sources.base import JobSource


def build_checkpointer(db_path: str = ":memory:") -> SqliteSaver:
    """Create a SqliteSaver backed by ``db_path`` (a file for cross-run resume)."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


def _wire_agentic(ctx: NodeContext, config: Config) -> None:
    """Attach a tool-calling chat model + MCP tools to the context for agentic mode."""
    if not config.openai_api_key:
        raise ValueError(
            "Agentic mode requires a tool-calling LLM. Set OPENAI_API_KEY "
            "(the mock LLM cannot plan or call tools)."
        )
    from langchain_openai import ChatOpenAI

    from graph.mcp_client import load_mcp

    ctx.chat_model = ChatOpenAI(
        model=config.openai_model,
        temperature=config.llm_temperature,
        api_key=config.openai_api_key,
    )
    ctx.mcp_client, ctx.mcp_tools = load_mcp(config)


def build_graph(
    config: Config,
    *,
    llm: LLMClient | None = None,
    source: JobSource | None = None,
    checkpointer: SqliteSaver | None = None,
):
    """Assemble and compile the resume-gap graph.

    Backends default to whatever ``config`` selects (mock by default); pass
    ``llm``/``source`` to override without touching the graph. The compiled
    graph is returned ready to ``invoke`` with a ``{"configurable":
    {"thread_id": ...}}`` config.
    """
    ctx = NodeContext(
        config=config,
        llm=llm or build_llm_client(config),
        source=source or build_job_source(config),
    )

    # In agentic mode, the sourcing + critic nodes are LLM-driven and need a
    # tool-calling chat model + MCP tools; wire those up and swap the nodes.
    if config.mode == "agentic":
        from graph import agents

        _wire_agentic(ctx, config)
        source_node = agents.sourcing_agent
        critic_node = agents.critic_agent
    else:
        source_node = nodes.source_postings
        critic_node = nodes.critic

    builder = StateGraph(GraphState)

    # Bind the shared context into each node via partial.
    builder.add_node("parse_resume", partial(nodes.parse_resume, ctx=ctx))
    builder.add_node("parse_target_role", partial(nodes.parse_target_role, ctx=ctx))
    builder.add_node("source_postings", partial(source_node, ctx=ctx))
    builder.add_node("extract_requirements", partial(nodes.extract_requirements, ctx=ctx))
    builder.add_node("aggregate_requirements", partial(nodes.aggregate_requirements, ctx=ctx))
    builder.add_node("gap_diff", partial(nodes.gap_diff, ctx=ctx))
    builder.add_node("critic", partial(critic_node, ctx=ctx))
    builder.add_node("report", partial(nodes.report, ctx=ctx))

    builder.add_edge(START, "parse_resume")
    builder.add_edge("parse_resume", "parse_target_role")
    builder.add_edge("parse_target_role", "source_postings")
    builder.add_edge("source_postings", "extract_requirements")
    builder.add_edge("extract_requirements", "aggregate_requirements")
    builder.add_edge("aggregate_requirements", "gap_diff")
    builder.add_edge("gap_diff", "critic")
    builder.add_conditional_edges(
        "critic",
        nodes.route_after_critic,
        {"source_postings": "source_postings", "report": "report"},
    )
    builder.add_edge("report", END)

    return builder.compile(checkpointer=checkpointer or build_checkpointer())
