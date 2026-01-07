import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb._create_dataloaders import create_dataloader
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.cache_wrappers.cache_backend_protocol import (
    CacheBackendProtocol,
)
from mteb.models.cache_wrappers.cache_backends.numpy_cache import NumpyCache
from mteb.models.model_meta import ModelMeta
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class CachedEmbeddingWrapper:
    """Wraps an encoder and caches embeddings for text and images.

    Examples:
        >>> import mteb
        >>> from mteb.models.cache_wrappers import CachedEmbeddingWrapper
        >>> from pathlib import Path
        >>> model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
        >>> cache_path = Path.cwd() / "cache"
        >>> cached_model = CachedEmbeddingWrapper(model, cache_path)
        >>> task = mteb.get_task("NanoArguAnaRetrieval")
        >>> mteb.evaluate(cached_model, task)
    """

    def __init__(
        self,
        model: EncoderProtocol,
        cache_path: str | Path,
        cache_backend: type[CacheBackendProtocol] = NumpyCache,
    ) -> None:
        """Init

        Args:
            model: Model to be wrapped.
            cache_path: Path to the directory where cached embeddings are stored.
            cache_backend: Cache backend class to use for storing embeddings.
        """
        self._model = model
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        if not hasattr(model, "encode"):
            raise ValueError("Model must have an 'encode' method.")
        self.cache_backend = cache_backend
        self.cache_dict: dict[str, CacheBackendProtocol] = {}
        logger.info("Initialized CachedEmbeddingWrapper")

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        """Return wrapped model meta data."""
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
            task_metadata: The metadata of the task.
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
            cache = self._get_or_create_cache(task_name)

            uncached_items: list[dict[str, Any]] = []
            uncached_indices: list[int] = []
            all_items: Dataset = inputs.dataset
            cached_vectors: dict[int, np.ndarray] = {}

            for i, item in enumerate(all_items):
                vector = cache.get_vector(item)
                if vector is not None:
                    cached_vectors[i] = vector
                else:
                    uncached_items.append(item)
                    uncached_indices.append(i)

            newly_encoded: dict[int, np.ndarray] = {}
            if uncached_items:
                logger.info(f"Encoding {len(uncached_items)} new items")
                # Build a simple DataLoader with only uncached items
                dataset = Dataset.from_list(uncached_items)
                dl = create_dataloader(
                    dataset,
                    task_metadata=task_metadata,
                    prompt_type=prompt_type,
                    **kwargs,
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
                cache.add(uncached_items, new_vectors)
                cache.save()
                for vector, original_idx in zip(new_vectors, uncached_indices):
                    newly_encoded[original_idx] = vector
            else:
                logger.info("All items found in cache")

            final_results = []
            for i in range(len(all_items)):
                if i in cached_vectors:
                    final_results.append(cached_vectors[i])
                else:
                    final_results.append(newly_encoded[i])

            return np.array(final_results)
        except Exception as e:
            logger.error(f"Error in cached encoding: {str(e)}")
            raise

    def _get_or_create_cache(self, task_name: str) -> CacheBackendProtocol:
        """Get or create cache for a specific task.

        Args:
            task_name: Name of the task

        Returns:
            Cache backend instance for the task
        """
        if task_name not in self.cache_dict:
            cache = self.cache_backend(self.cache_path / task_name)
            cache.load()
            self.cache_dict[task_name] = cache
        return self.cache_dict[task_name]

    def __del__(self):
        self.close()

    def close(self) -> None:
        """Unload cache from memory."""
        for task in list(self.cache_dict.keys()):
            self.cache_dict[task].close()

    def similarity(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Refer to [EncoderProtocol.similarity][mteb.models.EncoderProtocol.similarity] for more details."""
        return self._model.similarity(embeddings1, embeddings2)

    def similarity_pairwise(
        self,
        embeddings1: Array,
        embeddings2: Array,
    ) -> Array:
        """Refer to [EncoderProtocol.similarity][mteb.models.EncoderProtocol.similarity_pairwise] for more details."""
        return self._model.similarity_pairwise(embeddings1, embeddings2)
