"""Offline job source backed by bundled fixture postings.

This is the notebook default: it reads ``data/fixtures/postings.json`` so the
whole pipeline runs with zero credentials.
"""

from __future__ import annotations

import json
from pathlib import Path

from sources.base import JobPosting, JobSource

_DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "fixtures" / "postings.json"


class MockJobSource(JobSource):
    """Return postings from a JSON fixture, filtered by a naive role match."""

    name = "mock"

    def __init__(self, fixture_path: Path | str | None = None) -> None:
        self.fixture_path = Path(fixture_path) if fixture_path else _DEFAULT_FIXTURE
        self._postings = self._load()

    def _load(self) -> list[JobPosting]:
        raw = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return [JobPosting(**item) for item in raw]

    async def search(
        self,
        role: str,
        level: str,
        location: str | None,
        limit: int,
    ) -> list[JobPosting]:
        role_tokens = {t for t in role.lower().split() if len(t) > 2}
        scored: list[tuple[int, JobPosting]] = []
        for posting in self._postings:
            title = posting.title.lower()
            overlap = sum(1 for t in role_tokens if t in title)
            scored.append((overlap, posting))
        # Prefer title-matching postings, but fall back to all fixtures so the
        # mock always returns enough rows for the demo.
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [posting for _, posting in scored[:limit]]
