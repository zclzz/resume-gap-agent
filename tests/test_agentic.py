"""Cheap, offline tests for agentic-mode wiring (no network, no OpenAI calls).

The full agentic pipeline is exercised live in the notebook; here we lock down
the pieces that can be tested deterministically: config gating, tool-result
parsing, and the MCP server's tools (via a real stdio subprocess, no LLM).
"""

from __future__ import annotations

import os
import sys

import pytest

from config import Config
from graph.mcp_client import parse_tool_result


def test_parse_tool_result_json_string():
    assert parse_tool_result('{"a": 1}') == [{"a": 1}]
    assert parse_tool_result('[{"a": 1}, {"b": 2}]') == [{"a": 1}, {"b": 2}]


def test_parse_tool_result_content_blocks():
    blocks = [
        {"type": "text", "text": '{"posting_id": "x", "title": "T"}'},
        {"type": "text", "text": '{"posting_id": "y", "title": "U"}'},
    ]
    parsed = parse_tool_result(blocks)
    assert [p["posting_id"] for p in parsed] == ["x", "y"]


def test_parse_tool_result_empty():
    assert parse_tool_result(None) == []


def test_agentic_mode_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RGA_OPENAI_API_KEY", raising=False)
    from graph.build import build_graph

    cfg = Config(mode="agentic", openai_api_key=None, _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="Agentic mode requires"):
        build_graph(cfg)


@pytest.mark.asyncio
async def test_mcp_server_tools_over_stdio():
    """Spin up the real MCP server and exercise its tools (no LLM involved)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    conns = {
        "jobs": {
            "command": sys.executable,
            "args": [os.path.join(root, "mcp_server.py")],
            "transport": "stdio",
            "cwd": root,
            "env": {**os.environ, "RGA_USE_MOCK_SOURCE": "true"},
        }
    }
    tools = {t.name: t for t in await MultiServerMCPClient(conns).get_tools()}
    assert {"search_jobs", "get_posting_details", "normalize_skill_name"} <= set(tools)

    rows = parse_tool_result(
        await tools["search_jobs"].ainvoke(
            {"role": "machine learning engineer", "level": "entry", "location": "SG", "limit": 3}
        )
    )
    assert rows and all("posting_id" in r for r in rows)

    details = parse_tool_result(
        await tools["get_posting_details"].ainvoke({"posting_ids": [rows[0]["posting_id"]]})
    )
    assert details[0]["description"]

    norm = await tools["normalize_skill_name"].ainvoke({"name": "sklearn"})
    assert "scikit-learn" in (norm if isinstance(norm, str) else str(norm))
