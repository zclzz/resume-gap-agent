"""Async bridge + Windows/Jupyter compatibility shims for MCP-over-stdio.

Two problems show up when driving async MCP subprocesses from a synchronous
LangGraph node inside a Jupyter kernel on Windows:

1. The kernel's event loop is a *selector* loop, which cannot spawn async
   subprocesses on Windows -> ``NotImplementedError``. Fix: run coroutines on a
   fresh ``ProactorEventLoop`` (in a worker thread if a loop is already running).
2. ``sys.stderr`` in the kernel is ipykernel's stream, which has no real
   ``fileno()``. The MCP stdio client binds ``errlog=sys.stderr`` *at import
   time* and passes it as the child's stderr -> ``io.UnsupportedOperation:
   fileno``. Fix: expose a real ``fileno()`` on ``sys.stderr`` **before** MCP is
   imported (see :func:`ensure_stderr_fileno`).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
from typing import Any, Coroutine


def _new_event_loop() -> asyncio.AbstractEventLoop:
    # ProactorEventLoop is required for subprocesses on Windows.
    if sys.platform == "win32":
        return asyncio.ProactorEventLoop()
    return asyncio.new_event_loop()


def run_coro(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine to completion on a subprocess-capable event loop.

    Uses a fresh Proactor loop. If a loop is already running in this thread
    (e.g. an awaited notebook cell), the work is offloaded to a worker thread
    with its own loop.
    """
    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    def _run_in_fresh_loop() -> Any:
        loop = _new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    if not loop_running:
        return _run_in_fresh_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_in_fresh_loop).result()


class _FilenoStderr:
    """Wraps a stream that lacks ``fileno()`` (ipykernel's) and lends it a real
    one (backed by os.devnull), while still forwarding writes so messages show."""

    def __init__(self, target: Any) -> None:
        self._target = target
        self._devnull = open(os.devnull, "w")

    def write(self, s: str) -> int:
        try:
            return self._target.write(s)
        except Exception:
            return len(s)

    def flush(self) -> None:
        try:
            self._target.flush()
        except Exception:
            pass

    def fileno(self) -> int:
        return self._devnull.fileno()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


def ensure_stderr_fileno() -> None:
    """Make MCP-over-stdio work on Windows/Jupyter, regardless of import order.

    Two things:
    1. Give ``sys.stderr`` a real ``fileno()`` so *future* MCP imports bind a
       usable ``errlog`` default.
    2. If MCP was *already* imported (e.g. a prior failed cell captured the bad
       stderr as its default), retroactively override that captured default so it
       is self-healing without a kernel restart.

    Idempotent; a no-op off Windows or when stderr already has a fileno.
    """
    if sys.platform != "win32":
        return
    # (1) fix stderr for future imports
    if not isinstance(sys.stderr, _FilenoStderr):
        try:
            sys.stderr.fileno()
        except Exception:
            sys.stderr = _FilenoStderr(sys.stderr)
    # (2) heal an already-imported MCP stdio client's captured errlog default
    _patch_mcp_stdio_errlog()


def _patch_mcp_stdio_errlog() -> None:
    stdio = sys.modules.get("mcp.client.stdio")
    if stdio is None:
        return
    fn = getattr(stdio, "stdio_client", None)
    target = getattr(fn, "__wrapped__", None) or fn
    defaults = getattr(target, "__defaults__", None)
    if not defaults:
        return
    try:
        defaults[-1].fileno()  # current errlog default already usable
    except Exception:
        target.__defaults__ = defaults[:-1] + (open(os.devnull, "w"),)
