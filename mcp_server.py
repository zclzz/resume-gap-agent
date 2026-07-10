"""MCP server exposing job-sourcing + skills tools.

Run as its own process over stdio (``python mcp_server.py``). The LangGraph
sourcing agent connects to it as an MCP client and calls these tools to decide
*how* to search and *which* postings to keep.

Design note: tools are written to be **stateless across calls** - everything a
tool needs is reconstructable from its arguments plus data loaded at startup.
So it does not matter whether the MCP client keeps one persistent subprocess or
spawns a fresh one per call. For the mock source that works because every
fixture posting is indexed at import; for live sources (Adzuna), ``search_jobs``
persists results to a small on-disk cache so ``get_posting_details`` can resolve
a ``posting_id`` from a different subprocess.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from config import get_config
from factory import build_job_source
from logging_conf import configure_logging
from taxonomy import _ALIAS_GROUPS, normalize_skill

mcp = FastMCP("resume-gap-tools")

_config = get_config()
# stdout is the JSON-RPC transport: any log line there corrupts the protocol.
configure_logging(_config.log_level, colors=False, stream=sys.stderr)
_source = build_job_source(_config)

# posting_id -> JobPosting, for stateless detail lookups (mock loads all fixtures).
_by_id: dict[str, object] = {}

# On-disk spillover of _by_id for live sources: the MCP client may spawn a fresh
# server subprocess per tool call, so search results must outlive this process.
_CACHE_PATH = Path(__file__).resolve().parent / "data" / ".posting_cache.json"


def _read_cache() -> dict[str, dict]:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _persist(entries: dict[str, object]) -> None:
    cache = _read_cache()
    cache.update({pid: p.model_dump() for pid, p in entries.items()})
    _CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")


def _register(posting) -> str:
    pid = posting.url or f"{posting.source}:{posting.title}"
    _by_id[pid] = posting
    return pid


# Pre-index everything the mock source knows about, so details resolve without
# a prior search call in the same process.
for _p in getattr(_source, "_postings", []):
    _register(_p)


@mcp.tool()
async def search_jobs(
    role: str, level: str = "entry", location: str = "Singapore", limit: int = 12
) -> list[dict]:
    """Search job postings. Returns compact results (id, title, company, level).

    Call this to explore. Refine ``role`` wording and re-call if results look off
    or too few. Use ``get_posting_details`` to read a posting's full text.
    """
    postings = await _source.search(role, level, location, limit)
    out = []
    found: dict[str, object] = {}
    for p in postings:
        pid = _register(p)
        found[pid] = p
        out.append(
            {"posting_id": pid, "title": p.title, "company": p.company, "level": p.level}
        )
    _persist(found)
    return out


@mcp.tool()
def get_posting_details(posting_ids: list[str]) -> list[dict]:
    """Fetch full details (incl. description) for one or more posting_ids."""
    cache: dict[str, dict] | None = None  # lazy: only read on an in-memory miss
    out = []
    for pid in posting_ids:
        p = _by_id.get(pid)
        if p is not None:
            out.append(p.model_dump())
            continue
        if cache is None:
            cache = _read_cache()
        if pid in cache:
            out.append(cache[pid])
        else:
            out.append({"posting_id": pid, "error": "unknown posting_id"})
    return out


@mcp.tool()
def normalize_skill_name(name: str) -> str:
    """Return the canonical form of a skill string (e.g. 'sklearn' -> 'scikit-learn')."""
    return normalize_skill(name)


@mcp.tool()
def list_known_skills() -> list[str]:
    """List the canonical skill names the taxonomy recognizes."""
    return sorted(_ALIAS_GROUPS.keys())


if __name__ == "__main__":
    mcp.run(transport="stdio")
