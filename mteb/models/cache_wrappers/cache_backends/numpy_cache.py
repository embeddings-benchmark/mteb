import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ._hash_utils import _hash_item

logger = logging.getLogger(__name__)


class NumpyCache:
    """Generic vector cache for both text and images."""

    def __init__(self, directory: str | Path, initial_vectors: int = 100_000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vectors_file = self.directory / "vectors.npy"
        self.index_file = self.directory / "index.json"
        self.dimension_file = self.directory / "dimension"
        self.hash_to_index: dict[str, int] = {}
        self.vectors: np.memmap | None = None
        self.vector_dim: int | None = None
        self.initial_vectors = initial_vectors
        logger.info(f"Initialized VectorCacheMap in directory: {self.directory}")
        self._initialize_vectors_file()

    def add(self, items: list[dict[str, Any]], vectors: np.ndarray) -> None:
        """Add a vector to the cache."""
        try:
            if self.vector_dim is None:
                self.vector_dim = (
                    vectors.shape[0] if vectors.ndim == 1 else vectors.shape[1]
                )
                self._initialize_vectors_file()
                self._save_dimension()
                logger.info(f"Initialized vector dimension to {self.vector_dim}")

            if self.vectors is None:
                raise RuntimeError(
                    "Vectors file not initialized. Call _initialize_vectors_file() first."
                )

            for item, vec in zip(items, vectors):
                item_hash = _hash_item(item)
                if item_hash in self.hash_to_index:
                    msg = f"Hash collision or duplicate item for hash {item_hash}. Overwriting existing vector."
                    logger.warning(msg)
                    warnings.warn(msg)
                    index = self.hash_to_index[item_hash]
                else:
                    index = len(self.hash_to_index)
                    if index >= len(self.vectors):
                        self._double_vectors_file()
                    self.hash_to_index[item_hash] = index

                self.vectors[index] = vec
                logger.debug(
                    f"Added new item-vector pair. Total pairs: {len(self.hash_to_index)}"
                )
        except Exception as e:
            logger.error(f"Error adding item-vector pair: {str(e)}")
            raise

    def _initialize_vectors_file(self) -> None:
        if self.vector_dim is None:
            logger.info("Vector dimension not set. Waiting for first add() call.")
            return
        if not self.vectors_file.exists():
            logger.info(
                f"Creating initial vectors file with {self.initial_vectors} vectors"
            )
            self.vectors = np.memmap(
                self.vectors_file,
                dtype="float32",
                mode="w+",
                shape=(self.initial_vectors, self.vector_dim),
            )
        else:
            self.vectors = np.memmap(
                self.vectors_file,
                dtype="float32",
                mode="r+",
                shape=(-1, self.vector_dim),
            )
        logger.info(f"Vectors file initialized with shape: {self.vectors.shape}")

    def _double_vectors_file(self) -> None:
        if self.vectors is None or self.vector_dim is None:
            raise RuntimeError(
                "Vectors file not initialized. Call _initialize_vectors_file() first."
            )
        current_size = len(self.vectors)
        new_size = current_size * 2
        logger.info(f"Doubling vectors file from {current_size} to {new_size} vectors")
        self.vectors.flush()
        new_vectors = np.memmap(
            str(self.vectors_file),
            dtype=np.float32,
            mode="r+",
            shape=(new_size, self.vector_dim),
        )
        new_vectors[:current_size] = self.vectors[:]
        self.vectors = new_vectors

    def _save_dimension(self) -> None:
        with self.dimension_file.open("w") as f:
            f.write(str(self.vector_dim))
        logger.info(
            f"Saved vector dimension {self.vector_dim} to {self.dimension_file}"
        )

    def _load_dimension(self) -> None:
        if self.dimension_file.exists():
            with self.dimension_file.open() as f:
                self.vector_dim = int(f.read().strip())
            logger.info(
                f"Loaded vector dimension {self.vector_dim} from {self.dimension_file}"
            )
        else:
            msg = "Dimension file not found. Vector dimension remains uninitialized."
            logger.warning(msg)
            warnings.warn(msg)

    def save(self) -> None:
        """Persist VectorCacheMap to disk."""
        try:
            if self.vectors is not None:
                self.vectors.flush()

            # Convert hash_to_index dict to a format suitable for JSON
            # JSON doesn't support integer keys, so we keep everything as strings
            serializable_index = {
                str(hash_): int(index)  # Ensure indices are serialized as integers
                for hash_, index in self.hash_to_index.items()
            }

            with self.index_file.open("w", encoding="utf-8") as f:
                json.dump(serializable_index, f, indent=2)
            self._save_dimension()
            logger.info(f"Saved VectorCacheMap to {self.directory}")
        except Exception as e:
            logger.error(f"Error saving VectorCacheMap: {str(e)}")
            raise

    def load(self) -> None:
        """Load VectorCacheMap from disk."""
        try:
            self._load_dimension()
            if self.index_file.exists() and self.vectors_file.exists():
                with self.index_file.open(encoding="utf-8") as f:
                    loaded_index = json.load(f)
                    self.hash_to_index = {
                        str(hash_): int(index)  # Ensure we maintain the correct types
                        for hash_, index in loaded_index.items()
                    }

                if self.vector_dim is not None:
                    self.vectors = np.memmap(
                        self.vectors_file,
                        dtype="float32",
                        mode="r+",
                        shape=(-1, self.vector_dim),
                    )
                    logger.info(f"Loaded vectors file with shape: {self.vectors.shape}")
                else:
                    msg = "Vector dimension not set. Unable to load vectors file."
                    logger.warning(msg)
                    warnings.warn(msg)
                logger.info(f"Loaded VectorCacheMap from {self.directory}")
            else:
                msg = "No existing files found. Initialized empty VectorCacheMap."
                logger.warning(msg)
                warnings.warn(msg)
        except Exception as e:
            logger.error(f"Error loading VectorCacheMap: {str(e)}")
            raise

    def get_vector(self, item: dict[str, Any]) -> np.ndarray | None:
        """Retrieve vector from index by hash."""
        if self.vectors is None:
            return None

        try:
            item_hash = _hash_item(item)
            if item_hash not in self.hash_to_index:
                logger.debug(f"Item hash not found in index: {item_hash}")
                return None
            index = self.hash_to_index[item_hash]
            return self.vectors[index]
        except Exception as e:
            logger.error(f"Error retrieving vector for item: {str(e)}")
            raise

    def __contains__(self, item: dict[str, Any]) -> bool:
        return _hash_item(item) in self.hash_to_index

    def __del__(self):
        self.close()

    def close(self) -> None:
        """Delete all ve"""
        if hasattr(self, "vectors") and self.vectors is not None:
            self.vectors.flush()
            del self.vectors
            self.vectors = None
