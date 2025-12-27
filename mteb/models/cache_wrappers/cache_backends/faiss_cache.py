import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mteb._requires_package import requires_package
from mteb.types import BatchedInput

from ._hash_utils import _hash_item

logger = logging.getLogger(__name__)


class FaissCache:
    """FAISS-based vector cache that uses embeddings directly as lookup keys."""

    def __init__(self, directory: str | Path):
        requires_package(
            self,
            "faiss",
            "FAISS-based vector cache",
            install_instruction="pip install mteb[faiss-cpu]",
        )
        import faiss

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_file = self.directory / "vectors.faiss"
        self.map_file = self.directory / "index.json"

        self.hash_to_index: dict[str, int] = {}
        self.index: faiss.Index | None = None
        self.vector_dim: int | None = None

        logger.info(f"Initialized FAISS VectorCacheMap in {self.directory}")
        self.load()

    def add(self, items: list[dict[str, Any]], vectors: np.ndarray) -> None:
        """Add vector to FAISS index."""
        import faiss

        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if self.vector_dim is None:
            self.vector_dim = vectors.shape[1]
            self.index = faiss.IndexFlatL2(self.vector_dim)
        elif self.index is None:
            self.index = faiss.IndexFlatL2(self.vector_dim)

        start_id = len(self.hash_to_index)
        vectors_to_add = []
        for i, (item, vectors) in enumerate(zip(items, vectors)):
            item_hash = _hash_item(item)
            if item_hash in self.hash_to_index:
                continue
            self.hash_to_index[item_hash] = start_id + i
            vectors_to_add.append(vectors)
        if len(vectors_to_add) > 0:
            vectors_array = np.vstack(vectors_to_add).astype(np.float32)
            self.index.add(vectors_array)

    def get_vector(self, item: BatchedInput) -> np.ndarray | None:
        """Retrieve vector from index by hash."""
        if self.index is None:
            return None
        item_hash = _hash_item(item)
        if item_hash not in self.hash_to_index:
            return None
        idx = self.hash_to_index[item_hash]
        try:
            return self.index.reconstruct(idx)
        except Exception:
            msg = f"Vector id {idx} missing for hash {item_hash}"
            logger.warning(msg)
            warnings.warn(msg)
            return None

    def save(self) -> None:
        """Persist FAISS index and mapping to disk."""
        import faiss

        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
        with self.map_file.open("w") as f:
            json.dump(self.hash_to_index, f, indent=2)
        logger.info(f"Saved FAISS cache to {self.directory}")

    def load(self) -> None:
        """Load FAISS index and mapping from disk."""
        import faiss

        if self.map_file.exists():
            with self.map_file.open() as f:
                self.hash_to_index = json.load(f)
        if self.index_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.index = None
        else:
            self.index = None

    def close(self) -> None:
        """Close cache."""
        self.save()
        self.index = None

    def __contains__(self, item: BatchedInput) -> bool:
        return _hash_item(item) in self.hash_to_index

    def __del__(self):
        self.close()
