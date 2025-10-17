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
from mteb.models.cache_wrappers.cache_backends.cache_map import VectorCacheMap
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
        cache_backend: type[CacheBackendProtocol] = VectorCacheMap,
    ) -> None:
        """Init

        Args:
            model: Model to be wrapped.
            cache_path: Path to the directory where cached embeddings are stored.
            backend: Cache backend class to use for storing embeddings.
        """
        self._model = model
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        if not hasattr(model, "encode"):
            raise ValueError("Model must have an 'encode' method.")
        self.backend = backend
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
            if task_name not in self.cache_dict:
                self.cache_dict[task_name] = self.backend(self.cache_path / task_name)
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
