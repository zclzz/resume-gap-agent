"""Job sourcing abstraction.

``JobSource`` is the swap point for where postings come from. The analysis graph
only ever sees ``JobPosting`` models, so a new backend (Adzuna, JSearch,
Greenhouse, ...) can be dropped in without touching the graph.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from pydantic import BaseModel


class JobPosting(BaseModel):
    """A normalized job posting. Every backend maps its raw payload to this."""

    title: str
    company: str | None = None
    level: str | None = None
    description: str
    url: str | None = None
    source: str


class JobSource(ABC):
    """Abstract base for anything that can return normalized postings."""

    #: Human-readable backend name, surfaced in ``JobPosting.source``.
    name: str = "abstract"

    @abstractmethod
    async def search(
        self,
        role: str,
        level: str,
        location: str | None,
        limit: int,
    ) -> list[JobPosting]:
        """Return up to ``limit`` normalized postings for the given query."""
        raise NotImplementedError

    _SENIOR_MARKERS = (
        "senior",
        "sr.",
        "lead",
        "principal",
        "staff",
        "head",
        "manager",
        "director",
        "chief",
        "vice president",
    )

    def matches_level(
        self, posting: JobPosting, level: str, entry_tokens: tuple[str, ...]
    ) -> bool:
        """Heuristic seniority filter based on title/level tokens.

        Titles with no seniority marker at all are accepted for every level -
        most postings don't state seniority in the title, and rejecting them
        would empty the result set.
        """
        haystack = f"{posting.title} {posting.level or ''}".lower()
        is_senior = any(marker in haystack for marker in self._SENIOR_MARKERS)
        # Entry tokens include short ones like "i"/"l1", so match whole words
        # only - a substring check would hit the "i" in "senior".
        words = set(re.split(r"[^a-z0-9]+", haystack))
        is_entry = any(token in words for token in entry_tokens)

        # Senior markers dominate entry tokens: "Associate Director" is senior
        # even though "associate" alone would read as entry.
        if is_senior:
            return level.strip().lower() == "senior"
        if level.strip().lower() in ("entry", "junior", "graduate"):
            return True
        # senior / mid on an unmarked title: pass unless explicitly junior.
        return not is_entry

    def is_entry_level(self, posting: JobPosting, entry_tokens: tuple[str, ...]) -> bool:
        """Backward-compatible entry-level filter (see :meth:`matches_level`)."""
        return self.matches_level(posting, "entry", entry_tokens)
