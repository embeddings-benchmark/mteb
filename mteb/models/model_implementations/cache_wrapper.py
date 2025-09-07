from __future__ import annotations

import hashlib
import io
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader

from mteb import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.models_protocols import Encoder
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class VectorCacheMap:
    """Generic vector cache for both text and images."""

    def __init__(self, directory: str | Path, initial_vectors: int = 100000):
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

    def _hash_item(self, item: str | Image.Image) -> str:
        if isinstance(item, str):
            return hashlib.sha256(item.encode()).hexdigest()
        elif isinstance(item, Image.Image):
            with io.BytesIO() as output:
                item.save(output, format="PNG")  # normalize to PNG
                img_bytes = output.getvalue()
            return hashlib.sha256(img_bytes).hexdigest()
        else:
            raise TypeError(f"Unsupported cache key type: {type(item)}")

    def add(self, item: str | Image.Image, vector: np.ndarray) -> None:
        try:
            if self.vector_dim is None:
                self.vector_dim = vector.shape[0]
                self._initialize_vectors_file()
                self._save_dimension()
                logger.info(f"Initialized vector dimension to {self.vector_dim}")

            item_hash = self._hash_item(item)
            if item_hash in self.hash_to_index:
                logger.warning(
                    "Hash collision or duplicate item. Overwriting existing vector."
                )
                index = self.hash_to_index[item_hash]
            else:
                index = len(self.hash_to_index)
                if index >= len(self.vectors):
                    self._double_vectors_file()
                self.hash_to_index[item_hash] = index

            self.vectors[index] = vector
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
            self.vectors = np.memmap(self.vectors_file, dtype="float32", mode="r+")
            self.vectors = self.vectors.reshape(-1, self.vector_dim)
        logger.info(f"Vectors file initialized with shape: {self.vectors.shape}")

    def _double_vectors_file(self) -> None:
        current_size = len(self.vectors)
        new_size = current_size * 2
        logger.info(f"Doubling vectors file from {current_size} to {new_size} vectors")
        self.vectors.flush()
        new_vectors = np.memmap(
            self.vectors_file,
            dtype="float32",
            mode="r+",
            shape=(new_size, self.vector_dim),
        )
        new_vectors[:current_size] = self.vectors[:]
        self.vectors = new_vectors

    def _save_dimension(self) -> None:
        with open(self.dimension_file, "w") as f:
            f.write(str(self.vector_dim))
        logger.info(
            f"Saved vector dimension {self.vector_dim} to {self.dimension_file}"
        )

    def _load_dimension(self) -> None:
        if self.dimension_file.exists():
            with open(self.dimension_file) as f:
                self.vector_dim = int(f.read().strip())
            logger.info(
                f"Loaded vector dimension {self.vector_dim} from {self.dimension_file}"
            )
        else:
            logger.warning(
                "Dimension file not found. Vector dimension remains uninitialized."
            )

    def save(self) -> None:
        try:
            if self.vectors is not None:
                self.vectors.flush()

            # Convert hash_to_index dict to a format suitable for JSON
            # JSON doesn't support integer keys, so we keep everything as strings
            serializable_index = {
                str(hash_): int(index)  # Ensure indices are serialized as integers
                for hash_, index in self.hash_to_index.items()
            }

            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(serializable_index, f, indent=2)
            self._save_dimension()
            logger.info(f"Saved VectorCacheMap to {self.directory}")
        except Exception as e:
            logger.error(f"Error saving VectorCacheMap: {str(e)}")
            raise

    def load(self, name: str | None = None) -> None:
        try:
            self._load_dimension()
            if self.index_file.exists() and self.vectors_file.exists():
                with open(self.index_file, encoding="utf-8") as f:
                    loaded_index = json.load(f)
                    self.hash_to_index = {
                        str(hash_): int(index)  # Ensure we maintain the correct types
                        for hash_, index in loaded_index.items()
                    }

                if self.vector_dim is not None:
                    self.vectors = np.memmap(
                        self.vectors_file, dtype="float32", mode="r+"
                    )
                    self.vectors = self.vectors.reshape(-1, self.vector_dim)
                    logger.info(f"Loaded vectors file with shape: {self.vectors.shape}")
                else:
                    logger.warning(
                        "Vector dimension not set. Unable to load vectors file."
                    )
                logger.info(
                    f"Loaded VectorCacheMap ({name or ''}) from {self.directory}"
                )
            else:
                logger.warning(
                    f"No existing files found. Initialized empty VectorCacheMap ({name or ''})."
                )
        except Exception as e:
            logger.error(f"Error loading VectorCacheMap ({name or ''}): {str(e)}")
            raise

    def get_vector(self, item: str | Image.Image) -> np.ndarray | None:
        try:
            item_hash = self._hash_item(item)
            if item_hash not in self.hash_to_index:
                logger.debug(f"Item hash not found in index: {item_hash}")
                return None
            index = self.hash_to_index[item_hash]
            return self.vectors[index]
        except Exception as e:
            logger.error(f"Error retrieving vector for item: {str(e)}")
            raise

    def __contains__(self, item: str | Image.Image) -> bool:
        return self._hash_item(item) in self.hash_to_index

    def __del__(self):
        self.close()

    def close(self) -> None:
        if hasattr(self, "vectors") and self.vectors is not None:
            self.vectors.flush()
            del self.vectors
            self.vectors = None
        logger.info(f"Closed VectorCacheMap in directory: {self.directory}")


class CachedEmbeddingWrapper(AbsEncoder):
    """Wraps an encoder and caches embeddings for text and images."""

    def __init__(self, model: Encoder, cache_path: str | Path):
        self._model = model
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        if not hasattr(model, "encode"):
            raise ValueError("Model must have an 'encode' method.")
        self.cache_dict: dict[str, VectorCacheMap] = {}
        logger.info("Initialized CachedEmbeddingWrapper")

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        **kwargs: Any,
    ) -> Array:
        task_name = task_metadata.name
        try:
            if task_name not in self.cache_dict:
                self.cache_dict[task_name] = VectorCacheMap(self.cache_path / task_name)
                self.cache_dict[task_name].load(name=task_name)

            all_items: list[str | Image.Image] = []
            for batch in inputs:
                if "text" in batch:
                    all_items.extend(batch["text"])
                if "image" in batch:
                    if isinstance(batch["image"][0], list):
                        flat_imgs = [img for sub in batch["image"] for img in sub]
                        all_items.extend(flat_imgs)
                    else:
                        all_items.extend(batch["image"])

            results: list[np.ndarray] = []
            uncached_items: list[str | Image.Image] = []
            uncached_indices: list[int] = []

            for i, item in enumerate(all_items):
                vector = self.cache_dict[task_name].get_vector(item)
                if vector is not None:
                    results.append(vector)
                else:
                    uncached_items.append(item)
                    uncached_indices.append(i)

            if uncached_items:
                logger.info(f"Encoding {len(uncached_items)} new items")
                # Build a simple DataLoader with only uncached items
                dummy_ds = []
                for u in uncached_items:
                    if isinstance(u, str):
                        dummy_ds.append({"text": [u]})
                    elif isinstance(u, Image.Image) or (
                        isinstance(u, list) and isinstance(u[0], Image.Image)
                    ):
                        dummy_ds.append({"image": [u]})

                dl = DataLoader(Dataset.from_list(dummy_ds), batch_size=batch_size)
                new_vectors = self._model.encode(
                    dl,
                    task_metadata=task_metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    prompt_type=prompt_type,
                    **kwargs,
                )
                if isinstance(new_vectors, torch.Tensor):
                    new_vectors = new_vectors.cpu().numpy()
                for item, vec in zip(uncached_items, new_vectors):
                    self.cache_dict[task_name].add(item, vec)
                self.cache_dict[task_name].save()
                results.extend(new_vectors)
            else:
                logger.info("All items found in cache")

            final_results = [None] * len(all_items)
            uncached_idx = 0
            for i in range(len(all_items)):
                if i in uncached_indices:
                    final_results[i] = results[
                        len(all_items) - len(uncached_items) + uncached_idx
                    ]
                    uncached_idx += 1
                else:
                    final_results[i] = results[i - uncached_idx]

            return np.array(final_results)
        except Exception as e:
            logger.error(f"Error in cached encoding: {str(e)}")
            raise

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__[name]
        except KeyError:
            return getattr(self._model, name)

    def __dir__(self) -> list[str]:
        return list(set(super().__dir__() + dir(self._model)))  # type: ignore

    def __del__(self):
        self.close()

    def close(self) -> None:
        for task in list(self.cache_dict.keys()):
            self.cache_dict[task].close()
        logger.info("Closed CachedEmbeddingWrapper")
