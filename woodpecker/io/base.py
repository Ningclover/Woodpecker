"""Abstract base class for all data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod


class DataSource(ABC):
    """Load data from some source and return a domain-specific data object."""

    @abstractmethod
    def load(self, path: str, **kwargs):
        """Load data from *path* and return the appropriate data object."""
