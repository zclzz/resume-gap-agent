"""Job sourcing layer: an ABC plus concrete backends."""

from sources.base import JobPosting, JobSource
from sources.mock import MockJobSource

__all__ = ["JobPosting", "JobSource", "MockJobSource"]
