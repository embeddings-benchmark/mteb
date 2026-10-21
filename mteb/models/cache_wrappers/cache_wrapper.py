from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from mteb.models.cache_wrappers.cache_backends.multi_vector_cache import (
    MultiVectorCache,
)
from mteb.models.cache_wrappers.cache_backends.numpy_cache import NumpyCache
from mteb.models.model_meta import (
    _merge_precision_into_experiment_kwargs,
    _serialize_experiment_kwargs_to_name,
)
from mteb.models.sentence_transformer_wrapper import MultiVectorSearchEncoderWrapper

_EXPERIMENTS_FOLDER_NAME = "experiments"

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset
    from torch.utils.data import DataLoader

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.models.cache_wrappers.cache_backend_protocol import (
        CacheBackendProtocol,
    )
    from mteb.models.model_meta import ModelMeta
    from mteb.models.models_protocols import EncoderProtocol
    from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class _MultiVectorModel(Protocol):
    """A search-only multi-vector model, such as [`MultiVectorWrapper`][mteb.models.MultiVectorWrapper]."""

    mteb_model_meta: ModelMeta
    corpus_chunk_size: int

    def _encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array: ...

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array: ...

    def similarity_pairwise(self, embeddings1: Array, embeddings2: Array) -> Array: ...


def _to_cpu(vectors: Array) -> Array:
    """Move torch tensors to the CPU, like the cached vectors: dense ones as numpy arrays, sparse ones as tensors.

    Ragged lists are kept as they are.
    """
    import torch

    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu()
        return vectors if vectors.is_sparse else vectors.numpy()
    return vectors


class _CachedEncoderBase:
    """Shared caching logic for the cached encoder wrappers."""

    _model: EncoderProtocol | _MultiVectorModel

    def __init__(
        self,
        model: EncoderProtocol | _MultiVectorModel,
        cache_path: str | Path,
        cache_backend: type[CacheBackendProtocol],
    ) -> None:
        self._model = model
        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_backend = cache_backend
        self.cache_dict: dict[
            tuple[str, PromptType | None, str | None], CacheBackendProtocol
        ] = {}
        logger.info(f"Initialized {type(self).__name__}")

    @property
    def mteb_model_meta(self) -> ModelMeta | None:
        """The wrapped model meta data."""
        return self._model.mteb_model_meta

    def _cached_encode(
        self,
        encode_fn: Callable[..., Array],
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Return cached embeddings, encoding only the inputs missing from the cache with `encode_fn`."""
        from torch.utils.data import DataLoader

        task_name = task_metadata.name
        try:
            cache = self._get_or_create_cache(task_name, prompt_type, kwargs)

            all_items: Dataset = inputs.dataset
            cached_vectors: list[Array] = []
            cached_indices: list[int] = []
            uncached_items: list[dict[str, Any]] = []
            uncached_indices: list[int] = []
            for i, item in enumerate(all_items):
                vector = cache.get_vector(item)
                if vector is not None:
                    cached_vectors.append(vector)
                    cached_indices.append(i)
                else:
                    uncached_items.append(item)
                    uncached_indices.append(i)

            new_vectors = None
            if uncached_items:
                logger.info(f"Encoding {len(uncached_items)} new items")
                # `inputs` is already prepared for the task (e.g. corpus titles joined into the
                # text), so it is re-batched as is rather than passed to `create_dataloader` again
                dl = DataLoader(
                    all_items.select(uncached_indices),
                    batch_size=inputs.batch_size,
                    collate_fn=inputs.collate_fn,
                    num_workers=inputs.num_workers,
                    pin_memory=inputs.pin_memory,
                    prefetch_factor=inputs.prefetch_factor,
                    persistent_workers=inputs.persistent_workers,
                    shuffle=False,
                )
                new_vectors = _to_cpu(
                    encode_fn(
                        dl,
                        task_metadata=task_metadata,
                        hf_split=hf_split,
                        hf_subset=hf_subset,
                        prompt_type=prompt_type,
                        **kwargs,
                    )
                )
                cache.add(uncached_items, new_vectors)
                cache.save()
            else:
                logger.info("All items found in cache")

            # Position of each input in `[*cached_vectors, *new_vectors]`
            order = np.argsort(cached_indices + uncached_indices, kind="stable")
            return self._combine(cached_vectors, new_vectors, order)
        except Exception as e:
            logger.error(f"Error in cached encoding: {str(e)}")
            raise

    @staticmethod
    def _combine(
        cached_vectors: list[Array], new_vectors: Array | None, order: np.ndarray
    ) -> Array:
        """Combine the cached vectors and the newly encoded ones, in input order."""
        import torch

        parts: list[Array] = []
        if cached_vectors:
            if isinstance(cached_vectors[0], torch.Tensor):
                # Stacking sparse tensors loses their coalesced flag, so coalesce to match the mixed case
                stacked = torch.stack(cast("list[torch.Tensor]", cached_vectors))
                parts.append(stacked.coalesce() if stacked.is_sparse else stacked)
            else:
                parts.append(np.stack(cached_vectors))
        if new_vectors is not None:
            parts.append(new_vectors)
        if not parts:
            return np.array([])
        if len(parts) == 1:
            # Either everything was cached or nothing was, so the order is unchanged
            return parts[0]
        if isinstance(parts[0], torch.Tensor):
            combined = torch.cat(cast("list[torch.Tensor]", parts))
            return combined.index_select(0, torch.from_numpy(order)).coalesce()
        return cast("Array", np.concatenate(parts)[order])

    def _experiment_name(self, encode_kwargs: dict[str, Any]) -> str | None:
        """Serialize the settings that change the produced embeddings into a directory name.

        This mirrors the experiment namespacing used by the result cache so that, e.g.,
        an evaluation run with ``encode_kwargs={"precision": "int8"}`` does not reuse (or
        overwrite) the cached float embeddings of the default run. Both ablation
        parameters carried on the wrapped model's ``experiment_kwargs`` and a
        ``precision`` encode kwarg are taken into account.
        """
        meta = self.mteb_model_meta
        experiment_kwargs = _merge_precision_into_experiment_kwargs(
            meta.experiment_kwargs if meta is not None else None, encode_kwargs
        )

        return _serialize_experiment_kwargs_to_name(experiment_kwargs or None)

    def _get_or_create_cache(
        self,
        task_name: str,
        prompt_type: PromptType | None,
        encode_kwargs: dict[str, Any] | None = None,
    ) -> CacheBackendProtocol:
        """Get or create cache for a specific task, prompt type and experiment.

        Args:
            task_name: Name of the task
            prompt_type: Prompt role used to encode the inputs
            encode_kwargs: Keyword arguments forwarded to the wrapped model's ``encode``.
                Settings that change the embeddings (e.g. ``precision``) namespace the cache.

        Returns:
            Cache backend instance for the task
        """
        experiment_name = self._experiment_name(encode_kwargs or {})
        cache_key = (task_name, prompt_type, experiment_name)
        if cache_key not in self.cache_dict:
            cache_path = self.cache_path / task_name
            if experiment_name is not None:
                cache_path = cache_path / _EXPERIMENTS_FOLDER_NAME / experiment_name
            if prompt_type is not None:
                cache_path /= prompt_type.value
            cache = self.cache_backend(cache_path)
            cache.load()
            self.cache_dict[cache_key] = cache
        return self.cache_dict[cache_key]

    def __del__(self) -> None:
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


class CachedEmbeddingWrapper(_CachedEncoderBase):
    """Wraps an encoder and caches embeddings for text, images, audio, and video.

    Dense embeddings are cached with [`NumpyCache`][mteb.models.cache_wrappers.cache_backends.NumpyCache]
    by default. For sparse encoders, pass
    [`SparseCache`][mteb.models.cache_wrappers.cache_backends.SparseCache] as `cache_backend`. Multi-vector
    (late-interaction) models are cached with [`CachedMultiVectorWrapper`][mteb.models.CachedMultiVectorWrapper].

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

    _model: EncoderProtocol

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
        if isinstance(model, MultiVectorSearchEncoderWrapper):
            raise ValueError(
                "Multi-vector models don't expose an 'encode' method. Use CachedMultiVectorWrapper instead."
            )
        if not hasattr(model, "encode"):
            raise ValueError("Model must have an 'encode' method.")
        super().__init__(model, cache_path, cache_backend)

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
        return self._cached_encode(
            self._model.encode,
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            batch_size=batch_size,
            **kwargs,
        )


class CachedMultiVectorWrapper(MultiVectorSearchEncoderWrapper, _CachedEncoderBase):
    """Wraps a multi-vector (late-interaction) model and caches its per-token embeddings.

    Like the wrapped model, it only supports search (retrieval and reranking) tasks.

    Examples:
        >>> import mteb
        >>> from mteb.models import CachedMultiVectorWrapper, MultiVectorWrapper
        >>> model = MultiVectorWrapper("lightonai/LateOn")
        >>> cached_model = CachedMultiVectorWrapper(model, "cache")
        >>> task = mteb.get_task("NanoArguAnaRetrieval")
        >>> mteb.evaluate(cached_model, task)
    """

    _model: _MultiVectorModel

    def __init__(
        self,
        model: _MultiVectorModel,
        cache_path: str | Path,
        cache_backend: type[CacheBackendProtocol] = MultiVectorCache,
    ) -> None:
        """Init

        Args:
            model: Multi-vector model to be wrapped, e.g. a [`MultiVectorWrapper`][mteb.models.MultiVectorWrapper].
            cache_path: Path to the directory where cached embeddings are stored.
            cache_backend: Cache backend class to use for storing embeddings.
        """
        if not hasattr(model, "_encode"):
            raise ValueError("Model must have an '_encode' method.")
        super().__init__(model, cache_path, cache_backend)
        self.corpus_chunk_size = model.corpus_chunk_size

    def _encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        return self._cached_encode(
            self._model._encode,
            inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            prompt_type=prompt_type,
            **kwargs,
        )

    @staticmethod
    def _combine(
        cached_vectors: list[Array], new_vectors: Array | None, order: np.ndarray
    ) -> Array:
        """Combine the cached and newly encoded per-token embeddings into one list, in input order."""
        combined = [*cached_vectors, *(new_vectors if new_vectors is not None else [])]
        # A list of `(num_tokens, dim)` arrays, typed as `Array` like the wrapped `_encode`
        return cast("Array", [combined[i] for i in order])
