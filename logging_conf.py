"""structlog configuration with a notebook-friendly console renderer.

Call :func:`configure_logging` once (e.g. in the notebook setup cell). Every
node then does ``log = get_logger(__name__)`` and emits readable, aligned key/
value lines instead of raw JSON.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", *, colors: bool = True, stream=None) -> None:
    """Configure structlog + stdlib logging for pretty in-cell output.

    ``stream`` defaults to stdout. Processes whose stdout is a protocol channel
    (e.g. the MCP stdio server) must pass ``sys.stderr`` instead.
    """
    stream = stream if stream is not None else sys.stdout

    logging.basicConfig(
        format="%(message)s",
        stream=stream,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    # HTTP client libraries log full request URLs at INFO. Adzuna credentials
    # (app_id/app_key) ride in the URL query string, so those lines would leak
    # secrets into notebook output - keep these loggers at WARNING always.
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(colors=colors),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=stream),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
