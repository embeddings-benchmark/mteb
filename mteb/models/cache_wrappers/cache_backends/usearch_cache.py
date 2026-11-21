from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mteb._content_hashes import hash_item
from mteb._requires_package import _is_package_available

if TYPE_CHECKING:
    from usearch.index import Index

    from mteb.types import Array

logger = logging.getLogger(__name__)


class USearchCache:
    """USearch-based vector cache that uses a content hash as the lookup key."""

    def __init__(self, directory: str | Path):
        if not _is_package_available("usearch"):
            raise ImportError(
                "USearch is required for USearchCache. Please install with `pip install mteb[usearch]`."
            )

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_file = self.directory / "vectors.usearch"
        self.map_file = self.directory / "index.json"

        self.hash_to_index: dict[str, int] = {}
        self.index: Index | None = None
        self.vector_dim: int | None = None

        logger.info(f"Initialized USearch VectorCacheMap in {self.directory}")
        self.load()

    def add(self, items: list[dict[str, Any]], vectors: Array) -> None:
        """Add vectors to the USearch index."""
        from usearch.index import Index

        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if self.vector_dim is None:
            self.vector_dim = vectors.shape[1]
        elif vectors.shape[1] != self.vector_dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} does not match the cache dimension {self.vector_dim}."
            )
        index = self.index
        if index is None:
            index = self.index = Index(ndim=self.vector_dim, metric="l2sq", dtype="f32")

        keys: list[int] = []
        vectors_to_add = []
        next_key = len(self.hash_to_index)
        for item, vector in zip(items, vectors, strict=True):
            item_hash = hash_item(item)
            if item_hash in self.hash_to_index:
                continue
            self.hash_to_index[item_hash] = next_key
            keys.append(next_key)
            vectors_to_add.append(vector)
            next_key += 1
        if len(vectors_to_add) > 0:
            index.add(
                np.asarray(keys, dtype=np.uint64),
                np.vstack(vectors_to_add).astype(np.float32),
            )

    def get_vector(self, item: dict[str, Any]) -> Array | None:
        """Retrieve vector from index by hash."""
        if self.index is None:
            return None
        item_hash = hash_item(item)
        if item_hash not in self.hash_to_index:
            return None
        idx = self.hash_to_index[item_hash]
        vector = self.index.get(idx)
        if vector is None:
            msg = f"Vector id {idx} missing for hash {item_hash}"
            logger.warning(msg)
            warnings.warn(msg, stacklevel=2)
            return None
        return np.asarray(vector)

    def save(self) -> None:
        """Persist USearch index and mapping to disk."""
        if self.index is not None:
            self.index.save(str(self.index_file))
        with self.map_file.open("w", encoding="utf-8") as f:
            json.dump(self.hash_to_index, f, indent=2)
        logger.info(f"Saved USearch cache to {self.directory}")

    def load(self) -> None:
        """Load USearch index and mapping from disk."""
        from usearch.index import Index

        if self.map_file.exists():
            with self.map_file.open(encoding="utf-8") as f:
                self.hash_to_index = json.load(f)
        if self.index_file.exists():
            try:
                restored = Index.restore(str(self.index_file))
                if restored is None:
                    raise ValueError(f"{self.index_file} is not a USearch index")
                self.index = restored
                self.vector_dim = restored.ndim
                logger.info(f"Loaded USearch index with {len(restored)} vectors")
            except Exception as e:
                logger.error(f"Failed to load USearch index: {e}")
                self.index = None
        else:
            self.index = None

    def close(self) -> None:
        """Close cache."""
        self.save()
        self.index = None

    def __contains__(self, item: dict[str, Any]) -> bool:
        return hash_item(item) in self.hash_to_index

    def __del__(self) -> None:
        self.close()
