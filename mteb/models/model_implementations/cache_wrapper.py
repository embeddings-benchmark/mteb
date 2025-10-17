import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from torch.utils.data import DataLoader

from mteb._create_dataloaders import create_dataloader
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class _VectorCacheMap:
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
        item_hash = ""
        if "text" in item:
            item_hash = hashlib.sha256(item["text"].encode()).hexdigest()

        if "image" in item:
            image: Image.Image = item["image"]
            item_hash += hashlib.sha256(image.tobytes()).hexdigest()

        if item_hash == 0:
            raise TypeError(f"Unsupported cache key type: {type(item)}")

        return item_hash

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

            with self.index_file.open("w", encoding="utf-8") as f:
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
                with self.index_file.open(encoding="utf-8") as f:
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

    def get_vector(self, item: BatchedInput) -> np.ndarray | None:
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


class CachedEmbeddingWrapper(AbsEncoder):
    """Wraps an encoder and caches embeddings for text and images.

    Examples:
        >>> import mteb
        >>> from mteb.models.model_implementations.cache_wrapper import CachedEmbeddingWrapper
        >>> from pathlib import Path
        >>> model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> cache_path = Path.cwd() / "cache"
        >>> cached_model = CachedEmbeddingWrapper(model, cache_path)
        >>> task = mteb.get_task("NanoArguAnaRetrieval")
        >>> mteb.evaluate(cached_model, task)
    """

    def __init__(self, model: EncoderProtocol, cache_path: str | Path):
        """Args:
        model: Model to be wrapped.
        cache_path: Path to the directory where cached embeddings are stored.
        """
        self._model = model
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        if not hasattr(model, "encode"):
            raise ValueError("Model must have an 'encode' method.")
        self.cache_dict: dict[str, _VectorCacheMap] = {}
        logger.info("Initialized CachedEmbeddingWrapper")

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        return self._model.mteb_model_meta

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
        """Encodes the given sentences using the encoder.

        Args:
            inputs: Batch of inputs to encode.
            task_metadata: The metadata of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
                The order of priorities for prompt selection are:
                    1. Composed prompt of task name + prompt type (query or passage)
                    2. Specific task prompt
                    3. Composed prompt of task type + prompt type (query or passage)
                    4. Specific task type prompt
                    5. Specific prompt type (query or passage)
            hf_split: Split of current task
            hf_subset: Subset of current task
            prompt_type: The name type of prompt. (query or passage)
            batch_size: Batch size
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded input in a numpy array or torch tensor of the shape (Number of sentences) x (Embedding dimension).
        """
        task_name = task_metadata.name
        try:
            if task_name not in self.cache_dict:
                self.cache_dict[task_name] = _VectorCacheMap(
                    self.cache_path / task_name
                )
                self.cache_dict[task_name].load(name=task_name)

            results: list[np.ndarray] = []
            uncached_items: list[BatchedInput] = []
            uncached_indices: list[int] = []
            all_items = inputs.dataset

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
                dataset = Dataset.from_list(uncached_items)
                dl = create_dataloader(
                    dataset,
                    task_metadata=task_metadata,
                    prompt_type=prompt_type,
                    batch_size=batch_size,
                )
                new_vectors = self._model.encode(
                    dl,
                    task_metadata=task_metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    prompt_type=prompt_type,
                    batch_size=batch_size,
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

            final_results = []
            uncached_idx = 0
            for i in range(len(all_items)):
                if i in uncached_indices:
                    final_results.append(
                        results[len(all_items) - len(uncached_items) + uncached_idx]
                    )
                    uncached_idx += 1
                else:
                    final_results.append(results[i - uncached_idx])

            return np.array(final_results)
        except Exception as e:
            logger.error(f"Error in cached encoding: {str(e)}")
            raise

    def __del__(self):
        self.close()

    def close(self) -> None:
        """Unload cache from memory."""
        for task in list(self.cache_dict.keys()):
            self.cache_dict[task].close()
