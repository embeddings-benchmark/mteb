import json
import logging
from pathlib import Path

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
            install_instruction="pip install mteb[faiss-cpu] or mteb[faiss-gpu]",
        )
        import faiss

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.index_file = self.directory / "index.faiss"
        self.map_file = self.directory / "index.json"

        self.hash_to_index: dict[str, int] = {}
        self.index: faiss.Index | None = None
        self.vector_dim: int | None = None

        logger.info(f"Initialized FAISS VectorCacheMap in {self.directory}")
        self.load()

    def add(self, item: BatchedInput, vector: np.ndarray) -> None:
        """Add vector to FAISS index."""
        import faiss

        if vector.ndim == 1:
            vector = vector[None, :]

        if self.vector_dim is None:
            self.vector_dim = vector.shape[1]
            self.index = faiss.IndexFlatL2(self.vector_dim)
        elif self.index is None:
            self.index = faiss.IndexFlatL2(self.vector_dim)

        item_hash = _hash_item(item)
        if item_hash in self.hash_to_index:
            idx = self.hash_to_index[item_hash]
            logger.debug(f"Overwriting vector at id {idx}")
            self.index.reconstruct(idx)  # just to check existence
            self.index.remove_ids(np.array([idx]))
        new_id = len(self.hash_to_index)
        self.index.add(vector.astype(np.float32))
        self.hash_to_index[item_hash] = new_id
        logger.debug(f"Added vector id={new_id}, total={len(self.hash_to_index)}")

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
            logger.warning(f"Vector id {idx} missing for hash {item_hash}")
            return None

    def save(self) -> None:
        """Persist FAISS index and mapping to disk."""
        import faiss

        if self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
        with self.map_file.open("w") as f:
            json.dump(self.hash_to_index, f, indent=2)
        logger.info(f"Saved FAISS cache to {self.directory}")

    def load(self, name: str | None = None) -> None:
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
