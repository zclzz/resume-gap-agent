"""MCP client wiring for agentic mode.

Loads the tools exposed by ``mcp_server.py`` over stdio and turns them into
LangChain tools the ReAct agent can call. Also provides a robust parser for the
content-block format that MCP tool results come back in.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from config import Config
from graph.aio import ensure_stderr_fileno, run_coro
from logging_conf import get_logger

log = get_logger("graph.mcp")

_SERVER_PATH = str((Path(__file__).resolve().parent.parent / "mcp_server.py"))
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def build_connections(config: Config) -> dict:
    """Stdio connection spec for the MCP server subprocess."""
    return {
        "jobs": {
            "command": sys.executable,
            "args": [_SERVER_PATH],
            "transport": "stdio",
            "cwd": _PROJECT_ROOT,
            # Propagate the source switch so the server matches this run.
            "env": {**os.environ, "RGA_USE_MOCK_SOURCE": str(config.use_mock_source).lower()},
        }
    }


def load_mcp(config: Config):
    """Return (client, tools). The client must stay referenced for tools to work."""
    # Must happen before importing the MCP client (Windows/Jupyter stderr fix).
    ensure_stderr_fileno()
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(build_connections(config))
    tools = run_coro(client.get_tools())
    log.info("mcp_tools_loaded", tools=[t.name for t in tools])
    return client, tools


def parse_tool_result(result: Any) -> list[dict]:
    """Normalize an MCP/LangChain tool result into a list of dicts.

    Results may arrive as a JSON string, or as a list of ``{"type": "text",
    "text": "<json>"}`` content blocks (one per returned list item).
    """
    if result is None:
        return []
    if isinstance(result, str):
        value = json.loads(result)
        return value if isinstance(value, list) else [value]
    if isinstance(result, dict):
        return [result]
    items: list[dict] = []
    for element in result:
        if isinstance(element, dict) and element.get("type") == "text":
            items.append(json.loads(element["text"]))
        elif isinstance(element, dict):
            items.append(element)
    return items
