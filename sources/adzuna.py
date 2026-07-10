"""Live job source backed by the Adzuna REST API.

Free API keys: https://developer.adzuna.com/ . Singapore is supported via
country code ``sg``. Lazy-imports ``httpx`` so offline runs never require it.
The same graph consumes these postings -- only the source changes.
"""

from __future__ import annotations

from config import Config
from logging_conf import get_logger
from sources.base import JobPosting, JobSource

log = get_logger("sources.adzuna")

_BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"


class AdzunaJobSource(JobSource):
    """Query the Adzuna search API and normalize results to ``JobPosting``."""

    name = "adzuna"

    def __init__(self, config: Config) -> None:
        if not (config.adzuna_app_id and config.adzuna_app_key):
            raise ValueError(
                "RGA_ADZUNA_APP_ID and RGA_ADZUNA_APP_KEY are required for AdzunaJobSource."
            )
        self._config = config

    async def search(
        self,
        role: str,
        level: str,
        location: str | None,
        limit: int,
    ) -> list[JobPosting]:
        import httpx  # lazy, optional dependency

        url = _BASE_URL.format(country=self._config.adzuna_country)
        params = {
            "app_id": self._config.adzuna_app_id,
            "app_key": self._config.adzuna_app_key,
            "results_per_page": min(limit, 50),
            # Adzuna ANDs every word in ``what``; level words ("entry") appear in
            # almost no titles and zero out the results, so query the role only
            # and leave seniority to the downstream is_entry_level filter.
            "what": role.strip(),
            "content-type": "application/json",
        }
        if location:
            params["where"] = location

        log.info("adzuna.search", what=params["what"], where=location, limit=limit)
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        postings: list[JobPosting] = []
        for item in payload.get("results", []):
            postings.append(
                JobPosting(
                    title=item.get("title", "").strip(),
                    company=(item.get("company") or {}).get("display_name"),
                    level=None,
                    description=item.get("description", ""),
                    url=item.get("redirect_url"),
                    source=self.name,
                )
            )
        log.info("adzuna.search.done", returned=len(postings))
        return postings
