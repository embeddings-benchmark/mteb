from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CacheBackendProtocol(Protocol):
    """Protocol for a vector cache map (used to store text/image embeddings).

    Implementations may back the cache with different storage backends.

    The cache maps an input item (text or image) to its vector embedding,
    identified by a deterministic hash.
    """

    def __init__(self, directory: Path | None = None, **kwargs: Any) -> None:
        """Initialize the cache backend.

        Args:
            directory: Directory path to store cache files.
            **kwargs: Additional backend-specific arguments.
        """

    def add(self, item: list[dict[str, Any]], vectors: np.ndarray) -> None:
        """Add a vector to the cache.

        Args:
            item: Input item containing 'text' or 'image'.
            vectors: Embedding vector of shape (dim,) or (1, dim).
        """

    def get_vector(self, item: dict[str, Any]) -> np.ndarray | None:
        """Retrieve the cached vector for the given item.

        Args:
            item: Input item.

        Returns:
            Cached vector as np.ndarray, or None if not found.
        """

    def save(self) -> None:
        """Persist cache data to disk (index + metadata)."""

    def load(self) -> None:
        """Load cache from disk (index + metadata)."""

    def close(self) -> None:
        """Release resources or flush data."""

    def __contains__(self, item: dict[str, Any]) -> bool:
        """Check whether the cache contains an item."""
